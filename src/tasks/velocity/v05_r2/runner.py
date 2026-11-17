"""v0.5-r2 resident training loop, ten-stage curriculum and degeneration guard.

SDS v0.5-r2 revision 1.0 sections 2, 4, 5, 6, 7.  This module owns the
training-side orchestration the frozen v0.3/v0.4 core never had:

* the ten-stage ``F0..S5`` curriculum over the durable ``AttemptLedger`` state
  (SDS 5.1 budgets and gate intervals, "gate before budget exhaustion", the
  two-consecutive-pass streak, the S1 perturbation check on the passing
  checkpoint, pilot P1 fixed at 500 updates) with ``case_pass`` gate semantics;
* the complete-update boundary: durable schema-7 checkpoint -> 50-update
  degeneration diagnostic -> collapse terminal -> due full gate -> stage
  transition / budget check, in exactly that order;
* resident evaluation scheduling (SDS 7.4): evaluation never exits the worker and
  never resets the training environment.  It pauses the rollout, runs in an
  independent evaluation environment with a frozen model copy and its own case
  RNG, then restores every training-side RNG before the next rollout.  A
  training episode may therefore cross diagnostic and gate boundaries (an S5
  0.2 m/s episode needs >= 49 s while 50 updates are 48 s);
* schema-7 checkpoints through :mod:`v05_r2.recovery` (``build_checkpoint`` /
  ``preload_checkpoint``) and the SDS 7.3 resume order: load learning state ->
  bind the curriculum/slot class (no implicit reset) -> restore the
  Python/NumPy/Torch/CUDA and environment generators -> exactly one explicit
  new-episode reset -> refresh ``obs``;
* the evaluation transaction (``planned -> result_committed -> consumed``) is
  recorded by the ``AttemptLedger`` itself, so a resumed run replays
  committed-but-unconsumed tasks before more PPO and can never skip or double
  count a case.

r2 additions (SDS 1.2 C6/C7/C8):

* ``route_error`` is validated on every observation the runner adopts (reset,
  collect boundary, post-transition reset); a missing key is a hard error and
  is never zero-filled.  The rollout route is already part of
  ``ppo.storage['obs']`` and reaches the actor bypass / 211-input critic
  through ``update_math``'s minibatch slices.
* ``route/*`` aggregates are emitted read-only from the PPO diagnostics and the
  episode route evidence; a non-finite route is a hard error, never a warning.
* the pilot/formal update cap is the single ``spec.mode_budget(mode)`` source
  (pilot 2500, formal 30000); :meth:`Curriculum.update_cap` is what the worker
  loop and the supervisor must agree with.

Unit contract (SDS 2.5/7.6): one PPO update is 48*num_envs control samples and
20 Adam steps; ``physics_substeps`` is exactly ``4 * control_samples``.  The two
numbers are stored separately in ``learning`` and never share an axis.

The ledger is the single source of the durable curriculum, streaks and
evaluation tasks; :class:`Curriculum` is the read-only view that answers the
gate/budget questions of the runner.  A missing/incomplete sibling module is a
hard, named error at the point of use, never a silent fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from .recovery import (AttemptLedger, RecoveryBlocked, build_checkpoint, capture_rng,
                       preload_checkpoint, restore_rng, split_task_key, task_key,
                       updated_curriculum, validate_checkpoint, validate_learning)
from .spec import (CASE_PASS_REQUIRED, CASE_PASS_REQUIRED_FINAL, CONTROL_DT,
                   EPISODES_PER_SPEED_FINAL, FINAL_TEST_SEED, MODE_BUDGETS,
                   PHYSICS_DT, PPO_EPOCHS,
                   PPO_MINIBATCHES, PPO_ROLLOUT, SPEEDS, STAGE_BY_ID, STAGE_ORDER,
                   canonical_json, mode_budget, sha256)

ADAM_STEPS_PER_UPDATE = PPO_EPOCHS * PPO_MINIBATCHES
SUBSTEPS_PER_CONTROL_SAMPLE = int(round(CONTROL_DT / PHYSICS_DT))
RESUME_MODE = 'new_episode'
REWARD_STATE = 'new_episode_no_half_trajectory'

#: SDS 7.2 save cadence and SDS 6.3 diagnostic cadence.
SAVE_INTERVAL = 100
DIAGNOSTIC_INTERVAL = 50
DIAGNOSTIC_PER_SPEED = 5
DIAGNOSTIC_CASES = DIAGNOSTIC_PER_SPEED * len(SPEEDS)
COLLAPSE_START_STAGE_UPDATE = 100
COLLAPSE_STALL_FRACTION = .80
COLLAPSE_MIN_CASES = 12
COLLAPSE_MIN_FIRST_STEP_ABSENT = 12
COLLAPSE_CONSECUTIVE = 2
#: SDS 6.1 gate case counts.
GATE_CASES = 20
FINAL_TEST_CASES = EPISODES_PER_SPEED_FINAL
#: SDS 7.1 pilot shape.  The numeric caps come from ``spec.mode_budget`` (SDS
#: 7.3 single source); nothing here re-declares 2000/500.
PILOT_STAGES = ('F0', 'P1')
PILOT_BUDGETS = dict(mode_budget('pilot')['stages'])
PILOT_P1_UPDATES = int(mode_budget('pilot')['stages']['P1'])
PILOT_MAX_UPDATES = int(mode_budget('pilot')['total'])
#: ``smoke`` is the isolated W7 resource-probe mode: it keeps the frozen stage
#: table but is never a pilot/formal learning source.
CURRICULUM_MODES = ('formal', 'pilot', 'probe', 'smoke')
PILOT_FINAL_GATES = (400, 500)
PILOT_GATE_UPDATES = (100, 200, 300, 400, 500)

#: SDS 2.1: the five r2 observation keys; route_error is [B,3] float32.
OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', 'privileged', 'route_error')
ROUTE_ERROR_KEY = 'route_error'
ROUTE_ERROR_DIM = 3
#: SDS 4.8: the four episodes.jsonl route evidence fields (no namespace).
EPISODE_ROUTE_FIELDS = ('route_final', 'route_initial', 'route_max_abs_y',
                        'route_max_abs_heading_deg')
#: SDS 4.8: the read-only aggregate names emitted with every training update.
ROUTE_DIAGNOSTIC_FIELDS = ('abs_y_mean', 'abs_y_max', 'normalized_abs_y_max',
                           'saturation_rate', 'heading_deg_mean', 'heading_deg_max',
                           'nonfinite_count')

#: Import-time guard: a spec/mode-budget drift is a hard error, never a default.
if mode_budget('pilot')['total'] != 2500 or mode_budget('formal')['total'] != 30000:
    raise RuntimeError('v0.5-r2 pilot/formal update caps must be 2500/30000')
if set(MODE_BUDGETS) < {'pilot', 'formal'}:
    raise RuntimeError('v0.5-r2 mode_budget must define pilot and formal')
if PILOT_BUDGETS != {'F0': 2000, 'P1': 500}:
    raise RuntimeError('v0.5-r2 pilot stage budgets must be F0=2000, P1=500')


def validate_route_observation(obs, *, consumer, num_envs=None):
    """Strict ``route_error`` validation for every adopted runner observation.

    The runner never builds an observation: it validates the r2 env's five-key
    dict.  Missing key, wrong shape/dtype/device/batch or non-finite values are
    hard errors (SDS 4.7); there is no zero fill and no cache.
    """
    if not isinstance(obs, dict):
        raise ValueError(f'{consumer}: observation must be a dict')
    missing = [key for key in OBSERVATION_KEYS if key not in obs]
    if missing:
        raise ValueError(f'{consumer}: observation is missing {missing}')
    route = obs[ROUTE_ERROR_KEY]
    if not isinstance(route, torch.Tensor):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} must be a tensor')
    if route.dtype != torch.float32:
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} dtype must be float32')
    if route.ndim != 2 or route.shape[1] != ROUTE_ERROR_DIM:
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} shape must be [B,3], '
                         f'got {tuple(route.shape)}')
    if num_envs is not None and int(route.shape[0]) != int(num_envs):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} has {int(route.shape[0])} rows, '
                         f'expected num_envs={int(num_envs)}')
    for key in ('proprio', 'history', 'heightmap', 'privileged'):
        value = obs[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{consumer}: observation key {key} must be a tensor')
        if int(value.shape[0]) != int(route.shape[0]):
            raise ValueError(f'{consumer}: {key} batch differs from {ROUTE_ERROR_KEY}')
        if value.device != route.device:
            raise ValueError(f'{consumer}: {key} device differs from {ROUTE_ERROR_KEY}')
    if not bool(torch.isfinite(route).all()):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} contains NaN/Inf (SDS 4.7)')
    return route


def route_diagnostics(route_error):
    """SDS 4.8 route aggregates (same definition as ``v05_r2.ppo``)."""
    route = torch.as_tensor(route_error).detach()
    if route.ndim != 2 or route.shape[1] != ROUTE_ERROR_DIM:
        raise ValueError(f'route_error shape must be [N,3], got {tuple(route.shape)}')
    if not bool(torch.isfinite(route).all()):
        raise ValueError('route/nonfinite_count must be 0; route contains NaN/Inf')
    route = route.to(torch.float64)
    normalized = route[:, 0].abs()
    heading = torch.atan2(route[:, 1], route[:, 2]).abs()
    return dict(abs_y_mean=float(normalized.mean() * .20),
                abs_y_max=float(normalized.max() * .20),
                normalized_abs_y_max=float(normalized.max()),
                saturation_rate=float((normalized >= 4. - 1e-6).to(torch.float64).mean()),
                heading_deg_mean=float(torch.rad2deg(heading.mean())),
                heading_deg_max=float(torch.rad2deg(heading.max())),
                nonfinite_count=0, samples=int(route.shape[0]))


def route_episode_audit(records, *, required=True):
    """Check the four SDS 4.8 per-episode route evidence fields.

    The r2 environment owns those fields; the runner only audits them.  With
    ``required=True`` (pilot/formal) a missing field is a hard error because the
    episode route evidence is part of the pass/analysis contract; a smoke or
    unit-test env may set ``required=False`` while the identity is still checked.
    """
    rows = [row for row in records or [] if isinstance(row, dict)]
    missing = {field: 0 for field in EPISODE_ROUTE_FIELDS}
    for row in rows:
        for field in EPISODE_ROUTE_FIELDS:
            if row.get(field) is None:
                missing[field] += 1
    audit = dict(episodes=len(rows), required=bool(required),
                 missing=missing, complete=all(count == 0 for count in missing.values()))
    if required and not audit['complete']:
        raise ValueError(f'episode route evidence is incomplete: {missing}; '
                         'SDS 4.8 requires route_final/route_initial/'
                         'route_max_abs_y/route_max_abs_heading_deg')
    return audit


def integer(value, name, minimum=0):
    """Strict non-negative integer (``bool`` is not an integer here)."""
    if type(value) is not int or value < minimum:
        raise ValueError(f'Invalid {name}')
    return value


def _file_sha(path):
    return sha256(path)


def _canonical_sha(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


# ----------------------------------------------------------- gate record helpers
_SPEED_COUNT_KEYS = ('case_pass_count', 'speed_pass_count', 'climb_success_count',
                     'failure_count')


def _speed_rows(record, *, expected_total):
    """Normalise ``per_speed`` gate rows from an evaluation report.

    The evaluation module owns the raw evidence and the meaning of each counter
    (SDS 6.1); the runner only checks that three sane speed rows exist and applies
    the ``case_pass_count`` threshold.  ``c``/``total`` accept the ``vx``/
    ``episodes`` spellings so the adapter does not force a cosmetic rename.
    """
    if not isinstance(record, dict):
        raise ValueError('Gate record must be a dict')
    rows = record.get('per_speed', record.get('speeds'))
    if not isinstance(rows, list) or len(rows) != len(SPEEDS):
        raise ValueError('Exactly three per-speed gate rows are required')
    normalised = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid per-speed gate row')
        speed = row.get('c', row.get('vx'))
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
            raise ValueError('Missing per-speed command')
        total = integer(row.get('total', row.get('episodes')), 'gate case count', 1)
        if total != expected_total:
            raise ValueError(f'Gate case count {total} differs from the protocol {expected_total}')
        values = {}
        for key in _SPEED_COUNT_KEYS:
            value = row.get(key)
            values[key] = None if value is None else integer(value, key)
            if values[key] is not None and values[key] > total:
                raise ValueError(f'{key} exceeds the case count')
        if values['case_pass_count'] is None:
            raise ValueError('case_pass_count is mandatory (joint_successes is not a v0.5 gate)')
        normalised.append(dict(c=float(speed), total=total, **values))
    if sorted(row['c'] for row in normalised) != sorted(float(speed) for speed in SPEEDS):
        raise ValueError('Missing or duplicate gate speed')
    return sorted(normalised, key=lambda row: row['c'])


def _record_passed(rows, *, required):
    return all(row['case_pass_count'] >= required for row in rows)


class DiagnosticEvidenceError(ValueError):
    """The 50-update diagnostic evidence is insufficient; never a non-hit."""


def diagnostic_decision(summary, *, stairs, stage_update):
    """Apply the SDS 6.3 collapse rule to one 15-case diagnostic summary.

    ``summary['per_case']`` must expose one row per case with ``samples`` (the
    active and not-yet-arrived returned control samples), ``stall_samples``
    (those with ``stall_on=True``) and ``first_step_acquired`` (the original real
    footfall ``step == 1`` event).  A zero denominator is invalid evidence and
    raises instead of passing.
    """
    if stage_update < 1:
        raise ValueError('A diagnostic needs a completed stage update')
    if not isinstance(summary, dict):
        raise DiagnosticEvidenceError('Diagnostic summary must be a dict')
    rows = summary.get('per_case')
    if not isinstance(rows, list) or len(rows) != DIAGNOSTIC_CASES:
        raise DiagnosticEvidenceError(
            f'The diagnostic must return {DIAGNOSTIC_CASES} cases '
            f'(three speeds x {DIAGNOSTIC_PER_SPEED})')
    cases = []
    for row in rows:
        if not isinstance(row, dict):
            raise DiagnosticEvidenceError('Invalid diagnostic case row')
        speed = row.get('c', row.get('vx'))
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) \
                or float(speed) not in SPEEDS:
            raise DiagnosticEvidenceError('Diagnostic case has no protocol speed')
        samples = row.get('samples', row.get('active_samples'))
        stalls = row.get('stall_samples', row.get('stall_on_samples',
                                                  row.get('stall_active_samples')))
        try:
            # SDS 6.3: a zero denominator is invalid evidence, never a pass.
            samples = integer(samples, 'diagnostic denominator', 1)
            stalls = integer(stalls, 'diagnostic stall numerator')
        except ValueError as error:
            raise DiagnosticEvidenceError(
                'Diagnostic case has an invalid active-sample denominator/numerator') from error
        if stalls > samples:
            raise DiagnosticEvidenceError('Stall numerator exceeds the active denominator')
        first_step = row.get('first_step_acquired')
        if first_step is None:
            # SDS 6.3: the F0 diagnostic does not require the first-step condition.
            if stairs:
                raise DiagnosticEvidenceError('Diagnostic case lacks first_step_acquired')
        elif type(first_step) is not bool:
            raise DiagnosticEvidenceError('Diagnostic first_step_acquired must be boolean')
        cases.append(dict(c=float(speed), samples=samples, stall_samples=stalls,
                          stall_fraction=stalls / samples, first_step_acquired=first_step))
    for speed in SPEEDS:
        if sum(case['c'] == float(speed) for case in cases) != DIAGNOSTIC_PER_SPEED:
            raise DiagnosticEvidenceError('Diagnostic speed batch must have five cases')
    stalled = sum(case['stall_fraction'] >= COLLAPSE_STALL_FRACTION - 1e-12 for case in cases)
    first_step_absent = sum(case['first_step_acquired'] is False for case in cases)
    eligible = stage_update >= COLLAPSE_START_STAGE_UPDATE
    hit = (eligible and stalled >= COLLAPSE_MIN_CASES
           and (not stairs or first_step_absent >= COLLAPSE_MIN_FIRST_STEP_ABSENT))
    return dict(cases=cases, stalled_cases=stalled, first_step_absent_cases=first_step_absent,
                stall_fraction_threshold=COLLAPSE_STALL_FRACTION,
                min_stalled_cases=COLLAPSE_MIN_CASES,
                min_first_step_absent_cases=(0 if not stairs else COLLAPSE_MIN_FIRST_STEP_ABSENT),
                eligible=eligible, collapse_hit=bool(hit), stairs=bool(stairs))


class EvaluationError(RuntimeError):
    """An evaluation could not produce usable evidence (never a non-hit)."""


class PilotLedgerConflict(RuntimeError):
    """The durable ledger forbids the pilot P1 fixed-500 rule (C1 interface gap)."""


# ------------------------------------------------------------------ curriculum
class Curriculum:
    """Read-only view over the ``AttemptLedger``-derived curriculum.

    The ledger owns the durable state (``stage``, ``stage_iteration``,
    ``spent``, ``streak``, ``pending_transition``, tasks and results); this view
    answers the questions the training loop needs and never mutates the ledger.
    ``mode='pilot'`` restates the SDS 7.1 pilot shape (F0 2000, P1 fixed 500,
    report-only gates) on top of the same ledger state.  ``mode='smoke'`` keeps
    the frozen stage table for the isolated W7 resource probe (its separate
    ledger never counts against pilot/formal).
    """
    def __init__(self, ledger, *, mode=None, seed=42):
        required = ('mode', 'curriculum', 'completed_ppo_updates', 'tasks', 'results',
                    'qualified_update', 'gate_streak', 'collapse_streak')
        missing = [name for name in required if not hasattr(ledger, name)]
        if missing:
            raise ValueError(f'ledger view is missing {missing}')
        self.ledger = ledger
        self.mode = ledger.mode if mode is None else mode
        if self.mode not in CURRICULUM_MODES:
            raise ValueError(f'mode must be one of {CURRICULUM_MODES}')
        if self.mode != ledger.mode:
            raise ValueError('Curriculum/ledger mode mismatch')
        integer(seed, 'seed', 1)
        self.seed = int(seed)

    # ------------------------------------------------------------- identity
    @property
    def stages(self):
        return list(STAGE_ORDER)

    @property
    def budgets(self):
        """Effective per-stage caps from the single ``spec.mode_budget`` source.

        ``smoke`` has a fixed two-update cap and no stage table, so its
        structural per-stage cap falls back to the frozen stage table; the
        effective ``update_cap`` stays the smoke cap (SDS 7.3).
        """
        stages = mode_budget(self.mode)['stages']
        return dict(stages or mode_budget('formal')['stages'])

    @property
    def total_budget(self):
        return int(mode_budget(self.mode)['total'])

    @property
    def update_cap(self):
        """The effective update cap (pilot 2500 / formal 30000; SDS 7.3)."""
        return self.total_budget

    @property
    def effective_cap_source(self):
        return 'mode_budget'

    def budget_audit(self):
        """Four-place budget identity: worker loop, stage machine, supervisor, resume."""
        from .recovery import audit_budget
        audit = audit_budget(self.ledger.expected_learning(), self.state, mode=self.mode)
        audit.update(worker_max_updates=self.update_cap,
                     supervisor_max_updates=self.update_cap,
                     stage_machine_cap=self.update_cap,
                     resume_cap=self.update_cap,
                     same_source=True)
        return audit

    def result_budget_fields(self):
        """The ``result.json`` budget block (SDS 7.3): one source, no re-derivation.

        ``total_budget`` and ``update_cap`` are both the effective cap (pilot
        2500 / formal 30000); ``stage_budgets`` is the same mode table.  The r1
        ten-stage sum (29500) must never appear here.
        """
        return dict(mode=self.mode, total_budget=self.update_cap,
                    update_cap=self.update_cap, stage_budgets=self.budgets,
                    effective_cap_source=self.effective_cap_source)

    @property
    def state(self):
        return self.ledger.curriculum

    def state_dict(self):
        """The compact schema-7 curriculum block (recovery owns the schema)."""
        return self.ledger.curriculum

    @property
    def stage(self):
        return self.state['stage']

    @property
    def stage_iteration(self):
        return self.state['stage_iteration']

    @property
    def iteration(self):
        return self.ledger.completed_ppo_updates

    @property
    def consumed(self):
        return dict(self.state['spent'])

    @property
    def previous_stage(self):
        index = STAGE_ORDER.index(self.stage)
        return STAGE_ORDER[index - 1] if index else None

    @property
    def environment_stage_iteration(self):
        return self.stage_iteration

    @property
    def gate_interval(self):
        return STAGE_BY_ID[self.stage].gate

    @property
    def consecutive_passes(self):
        return self.state['streak']['gate']

    @property
    def collapse_streak(self):
        return self.state['streak']['collapse']

    @property
    def pending_transition(self):
        return self.state['pending_transition']

    @property
    def env_steps(self):
        return PPO_ROLLOUT * 512 * self.iteration

    # -------------------------------------------------------------- status
    @property
    def exhausted(self):
        return self.stage_iteration >= self.budgets[self.stage]

    @property
    def complete(self):
        return self.ledger.qualified_update is not None

    @property
    def qualified_update(self):
        return self.ledger.qualified_update

    @property
    def frozen_candidate_sha(self):
        """Checkpoint SHA of the second passing S5 gate (the only W10 model)."""
        update = self.qualified_update
        if update is None:
            return None
        for key, record in self.ledger.tasks.items():
            parts = split_task_key(key)
            if (parts['kind'] == 'validation' and parts['stage'] == 'S5'
                    and parts['global_update'] == update and record['state'] == 'consumed'):
                return parts['checkpoint_sha256']
        return None

    @property
    def final_result(self):
        for row in self.ledger.results:
            if row['kind'] == 'test':
                return bool(row['passed'])
        return None

    def _tasks(self, kind, *, stage=None, update=None, checkpoint_sha=None,
               state=None):
        found = []
        for key, record in self.ledger.tasks.items():
            parts = split_task_key(key)
            if parts['kind'] != kind:
                continue
            if stage is not None and parts['stage'] != stage:
                continue
            if update is not None and parts['global_update'] != update:
                continue
            if checkpoint_sha is not None and parts['checkpoint_sha256'] != checkpoint_sha:
                continue
            if state is not None and record['state'] != state:
                continue
            found.append((parts, record))
        return found

    def task(self, kind, *, stage=None, update=None, checkpoint_sha=None):
        found = self._tasks(kind, stage=stage, update=update,
                            checkpoint_sha=checkpoint_sha)
        return found[0][1] if found else None

    @property
    def validation_due(self):
        if self.complete or self.stage_iteration <= 0:
            return False
        if self.stage_iteration % self.gate_interval:
            return False
        # A planned/committed gate belongs to the pending-evaluation replay.
        return not self._tasks('validation', stage=self.stage, update=self.iteration)

    def _perturbed_done(self, pending):
        return bool(self._tasks('perturbed', stage=pending['from_stage'],
                                update=pending['at_update'],
                                checkpoint_sha=pending['at_checkpoint'],
                                state='consumed'))

    @property
    def needs_perturbed(self):
        pending = self.pending_transition
        return (pending is not None and pending.get('requires') == 'perturbed'
                and not self._perturbed_done(pending))

    @property
    def can_advance(self):
        pending = self.pending_transition
        if pending is None or pending.get('to_stage') is None:
            return False
        if pending.get('requires') == 'perturbed' and not self._perturbed_done(pending):
            return False
        if self.mode == 'pilot' and self.stage not in PILOT_STAGES[:-1]:
            return False
        return True

    @property
    def pilot_conflict(self):
        """True only for a real pending transition out of a pilot P1.

        With the adjudicated recovery pilot branch this is always false: a pilot P1
        validation updates the gate streak and never creates ``pending_transition``.
        A terminal qualification (``to_stage=None``) and ``pending_transition=None``
        are both *not* conflicts, so the guard cannot fire falsely.
        """
        if self.mode != 'pilot' or self.stage not in PILOT_STAGES[1:]:
            return False
        pending = self.pending_transition
        return pending is not None and pending.get('to_stage') is not None

    def gate_passed_at(self, global_update, *, stage=None):
        """Whether the full gate at ``global_update`` passed (SDS 6.1)."""
        stage = self.stage if stage is None else stage
        for row in self.ledger.results:
            if (row['kind'] == 'validation' and row['stage'] == stage
                    and row['global_update'] == global_update):
                return bool(row['passed'])
        return None

    def pilot_verdict(self):
        """SDS 7.1 trial verdict from the durable gate results."""
        if self.mode != 'pilot':
            raise ValueError('Pilot verdicts only exist in pilot mode')
        base = self.consumed['F0']
        gates = {iteration: self.gate_passed_at(base + iteration, stage='P1')
                 for iteration in PILOT_GATE_UPDATES}
        verdict = dict(mode='pilot', seed=self.seed, stage=self.stage,
                       stage_iteration=self.stage_iteration, gates=gates,
                       f0_consumed=self.consumed['F0'], p1_consumed=self.consumed['P1'],
                       total_updates=self.iteration, status='running', reasons=[])
        if self.stage == 'F0':
            if self.exhausted:
                verdict.update(status='failed', reasons=['f0_budget_exhausted_without_gate'])
            return verdict
        if self.stage != 'P1' or self.consumed['P1'] < self.budgets['P1']:
            return verdict
        reasons = [f'p1_gate_{iteration}_not_passed'
                   for iteration in PILOT_FINAL_GATES if gates[iteration] is not True]
        verdict.update(status='passed' if not reasons else 'failed', reasons=reasons)
        return verdict


# --------------------------------------------------------------------- runner
class V05Runner:
    """Resident v0.5-r2 training loop, boundary state machine and schema-7 owner.

    Construction takes the already-built PPO object (``v05/ppo.py`` owns the
    frozen update mathematics), the ``AttemptLedger`` (durable state machine) and
    the B-package evaluation services.  When ``evaluation``/``make_cases`` are
    omitted the runner imports ``v05/evaluation.py`` lazily on first use.
    """
    def __init__(self, *, model, env, ppo, curriculum, ledger, telemetry=None,
                 device='cpu', seed=42, mode=None, run_id=None, output_dir=None,
                 save_interval=SAVE_INTERVAL, diagnostic_interval=DIAGNOSTIC_INTERVAL,
                 eval_num_envs=8, evaluation=None, make_cases=None,
                 save_checkpoints=True, final_test=True, clock=time.time,
                 require_route_episode_evidence=None):
        if not isinstance(curriculum, Curriculum):
            raise ValueError('A v0.5-r2 Curriculum is required')
        if not isinstance(ledger, AttemptLedger):
            raise ValueError('A v05 AttemptLedger is required')
        if mode is not None and mode != curriculum.mode:
            raise ValueError('Runner/curriculum mode mismatch')
        integer(seed, 'seed', 1)
        if seed != curriculum.seed:
            raise ValueError('Runner/curriculum seed mismatch')
        if getattr(ppo, 'model', model) is not model:
            raise ValueError('The PPO must own the training model')
        if getattr(ppo, 'env', env) is not env:
            raise ValueError('The PPO must own the training environment')
        if (not hasattr(ppo, 'optimizer') or not hasattr(ppo, 'learning_state')
                or not hasattr(ppo, 'learning')):
            raise ValueError('The PPO must expose optimizer, learning_state and learning')
        integer(getattr(env, 'num_envs', None), 'env.num_envs', 1)
        if int(getattr(getattr(ppo, 'cfg', None), 'rollout', -1)) != PPO_ROLLOUT:
            raise ValueError('The v0.5-r2 PPO rollout must be 48 control samples')
        self.model = model
        self.env = env
        self.ppo = ppo
        self.curriculum = curriculum
        self.ledger = ledger
        self.telemetry = telemetry
        self.device = device
        self.mode = curriculum.mode
        self.seed = int(seed)
        self.run_id = str(run_id or f'v05_{self.mode}_seed{self.seed}')
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.save_interval = integer(save_interval, 'save_interval', 1)
        self.diagnostic_interval = integer(diagnostic_interval, 'diagnostic_interval', 1)
        self.eval_num_envs = integer(eval_num_envs, 'eval_num_envs', 1)
        self.evaluation = evaluation
        self.make_cases_fn = make_cases
        self.save_checkpoints = bool(save_checkpoints)
        self.final_test = bool(final_test)
        self.clock = clock
        # SDS 4.8: pilot/formal need the four episode route fields; smoke and
        # unit-test envs keep the same audit without the hard requirement.
        self.require_route_episode_evidence = (
            self.mode in ('pilot', 'formal', 'probe')
            if require_route_episode_evidence is None
            else bool(require_route_episode_evidence))
        self.terminal = None
        self.obs = None
        self.reset_info = None
        self.reset_route = None
        self.last_route_audit = None
        self.last_route_diagnostics = None
        self.last_case_seed = None
        self.lineage = []
        self.telemetry_hooks_missing = set()
        self._attach_ledger()
        self._check_learning_boundary()

    # ------------------------------------------------------------- identity
    @property
    def num_envs(self):
        return int(self.env.num_envs)

    @property
    def learning(self):
        learning = dict(self.ppo.learning)
        if set(learning) != {'update', 'optimizer_step', 'control_samples',
                             'physics_substeps'}:
            raise ValueError('PPO learning block must carry the schema-7 unit fields')
        return learning

    @property
    def stage(self):
        return self.curriculum.stage

    @property
    def stage_iteration(self):
        return self.curriculum.stage_iteration

    @property
    def iteration(self):
        return self.learning['update']

    def _attach_ledger(self):
        self.ppo.control_step_callback = self.ledger.control_step
        self.ppo.update_callback = self.ledger.update_event

    def _check_learning_boundary(self):
        # A resume-prep process constructs a fresh PPO (update 0) against a ledger
        # that may already be ahead; only a PPO ahead of the durable ledger is
        # impossible.  collect/update still require exact equality.
        if self.learning['update'] > self.ledger.completed_ppo_updates:
            raise ValueError('PPO is ahead of the durable ledger boundary')
        if self.learning['update'] > self.curriculum.update_cap:
            raise ValueError('Completed updates exceed the mode update cap')
        if self.ledger.completed_ppo_updates and self.ledger.curriculum['spent'][
                self.ledger.curriculum['stage']] != self.ledger.curriculum['stage_iteration']:
            raise ValueError('Ledger curriculum is inconsistent with its own spent table')

    def _require_attached(self):
        # Bound methods compare equal by (owner, function); identity is not stable
        # across attribute access, so equality is the correct attachment check.
        if (self.ppo.control_step_callback != self.ledger.control_step
                or self.ppo.update_callback != self.ledger.update_event):
            raise ValueError('Durable callbacks must stay attached to the ledger')

    def _require_running(self):
        if self.terminal is not None:
            raise RuntimeError(f'Training already terminated: {self.terminal}')
        if self.curriculum.pilot_conflict:
            raise PilotLedgerConflict(
                'The durable ledger created a P1 -> P2 transition while mode=pilot. '
                'SDS 7.1 keeps the pilot P1 fixed at 500 updates, so no pending_transition '
                'may exist for mode=pilot and stage=P1: the ledger pilot branch must only '
                'advance the gate streak, never create a transition.')

    # ------------------------------------------------------------ lifecycle
    def _environment_reset(self):
        """Call ``env.reset()`` and return ``(obs, info)``.

        ``v05_r2.env.V05R2Env`` inherits the v02 formal lifecycle and returns
        ``(obs, info)``; an injected test env returning a bare observation dict is
        also accepted, but a tuple of any other shape is an error.  The caller
        never treats the post-step auto-reset state as a pre-reset snapshot.
        """
        result = self.env.reset()
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError('env.reset() must return (obs, info)')
            obs, info = result
        else:
            obs, info = result, {}
        if not isinstance(obs, dict):
            raise TypeError('env.reset() observation must be a dict of tensors')
        return obs, info

    def _adopt_observation(self, obs, info):
        route = validate_route_observation(obs, consumer='runner.reset',
                                           num_envs=self.num_envs)
        self.obs = obs
        self.reset_info = info if isinstance(info, dict) else {}
        # The post-reset route is the new episode's initial route; it is kept for
        # the ``route_initial`` evidence and is never overwritten by a cached
        # terminal value (SDS 4.5).
        self.reset_route = route.detach().clone()
        if hasattr(self.ppo, 'obs'):
            self.ppo.obs = self.obs
        return obs

    def reset_episode(self):
        """One explicit new episode; used for a fresh run and after resume."""
        obs, info = self._environment_reset()
        return self._adopt_observation(obs, info)

    def collect(self):
        self._require_running()
        self._require_attached()
        if self.learning['update'] != self.ledger.completed_ppo_updates:
            raise ValueError('PPO/ledger completed-update counter mismatch')
        self.ledger.begin_attempt(self.ledger.curriculum, self.num_envs)
        records = self.ppo.collect()
        # SDS 4.4: the rollout buffer carries route_error [48*B,3] and the next
        # observation carries the post-step route for the next rollout.
        storage = getattr(self.ppo, 'storage', None)
        if not isinstance(storage, dict) or not isinstance(storage.get('obs'), dict):
            raise ValueError('PPO.collect did not produce a rollout with observations')
        route = storage['obs'].get(ROUTE_ERROR_KEY)
        if not isinstance(route, torch.Tensor) or route.ndim != 2                 or int(route.shape[1]) != ROUTE_ERROR_DIM:
            raise ValueError('PPO.collect rollout is missing the [48*B,3] route_error key')
        if int(route.shape[0]) != PPO_ROLLOUT * self.num_envs:
            raise ValueError('PPO.collect route rows must be rollout*num_envs')
        if not bool(torch.isfinite(route).all()):
            raise ValueError('route/nonfinite_count must be 0 in the rollout')
        validate_route_observation(self.obs, consumer='runner.collect.next_observation',
                                   num_envs=self.num_envs)
        self.last_route_diagnostics = dict(self.ppo.last_route_diagnostics or {})
        self.last_route_audit = route_episode_audit(
            records, required=self.require_route_episode_evidence)
        return records

    def update(self):
        """One PPO update (20 Adam steps) plus the exact unit accounting."""
        self._require_running()
        self._require_attached()
        if self.learning['update'] != self.ledger.completed_ppo_updates:
            raise ValueError('PPO/ledger completed-update counter mismatch')
        before = self.learning
        self.ppo.update()
        after = self.learning
        if after['update'] != before['update'] + 1:
            raise RuntimeError('A PPO update must advance exactly one completed update')
        if after['optimizer_step'] != before['optimizer_step'] + ADAM_STEPS_PER_UPDATE:
            raise RuntimeError('A PPO update must execute exactly 20 Adam steps')
        if after['control_samples'] != before['control_samples'] + PPO_ROLLOUT * self.num_envs:
            raise RuntimeError('Control-sample accounting drifted')
        if after['physics_substeps'] != SUBSTEPS_PER_CONTROL_SAMPLE * after['control_samples']:
            raise RuntimeError('Physics substeps must be exactly 4x control samples')
        expected = updated_curriculum(self.ledger.curriculum, num_envs=self.num_envs)
        self.ledger.complete_attempt(expected, after)
        if self.ledger.completed_ppo_updates != after['update']:
            raise RuntimeError('Ledger and PPO update counters disagree')
        self.sync_stage_iteration()
        route = self._route_record()
        self._emit('train', dict(update=after['update'], stage=self.stage,
                                 stage_iteration=self.stage_iteration,
                                 learning=after,
                                 route=route,
                                 ppo=self._ppo_record()))
        return after

    def _route_record(self):
        """Read-only ``route/*`` block for the train event and updates.jsonl.

        SDS 4.8: these fields are never fed back into observations, normalizers,
        rewards or advantages.  ``route/nonfinite_count`` must be 0 and is a hard
        error, not a pass metric.
        """
        diagnostics = getattr(self.ppo, 'last_route_diagnostics', None)
        if not isinstance(diagnostics, dict):
            raise ValueError('The r2 PPO update produced no route diagnostics')
        missing = [field for field in ROUTE_DIAGNOSTIC_FIELDS if field not in diagnostics]
        if missing:
            raise ValueError(f'route diagnostics are missing {missing}')
        if diagnostics['nonfinite_count'] != 0:
            raise FloatingPointError('route/nonfinite_count must be 0')
        record = {field: diagnostics[field] for field in ROUTE_DIAGNOSTIC_FIELDS}
        record['samples'] = diagnostics.get('samples')
        self.last_route_diagnostics = record
        return record

    @property
    def policy_state(self):
        """The complete policy state, including the r2 route bypass tensors.

        Evaluation/play/evidence code reads this instead of reaching into the
        model: it is the exact ``model.state_dict()`` and a schema-7 body must
        contain ``actor.route_fusion.weight [512,3]``,
        ``actor.route_fusion.bias [512]`` and ``critic.0.weight [512,211]``.
        """
        from .recovery import validate_route_model_state
        state = {name: value.detach().clone()
                 for name, value in self.model.state_dict().items()}
        validate_route_model_state(state)
        return state

    def _ppo_record(self):
        record = getattr(self.ppo, 'diagnostics_jsonl_record', None)
        if callable(record):
            try:
                return record()
            except Exception:  # noqa: BLE001 - diagnostics must never break training
                return None
        return None

    def sync_stage_iteration(self):
        """Keep the environment's S5 spawn ramp on the current stage update.

        ``V05Env.set_stage`` only resets when the stage actually changes, so
        restating the same stage advances the ramp without touching a running
        episode (SDS 5.1/7.4).
        """
        self.env.set_stage(self.stage, stage_iteration=self.stage_iteration)

    def train_step(self):
        self.collect()
        self.update()
        return self.boundary()

    # -------------------------------------------------------------- boundary
    def _diagnostic_due(self):
        if self.curriculum.complete or self.stage_iteration <= 0:
            return False
        if self.stage_iteration % self.diagnostic_interval:
            return False
        return not self.curriculum._tasks('diagnostic', stage=self.stage,
                                          update=self.iteration)

    def boundary(self):
        """Complete-update boundary with the SDS 6.3 event order.

        Durable checkpoint -> diagnostic -> collapse terminal -> due full gate ->
        promotion/budget check.  Evaluation errors become ``evaluation_error``
        and never clear a collapse streak.
        """
        self._require_running()
        events = []
        update = self.iteration
        diagnostic_due = self._diagnostic_due()
        gate_due = self.curriculum.validation_due
        final_due = (self.curriculum.complete and self.curriculum.final_result is None
                     and self.final_test)
        save_due = ((self.stage_iteration % self.save_interval == 0)
                    or diagnostic_due or gate_due or final_due)
        digest = None
        if save_due:
            digest = self.save_checkpoint('update')
        if diagnostic_due:
            try:
                outcome = self.run_diagnostic(digest)
            except (EvaluationError, DiagnosticEvidenceError) as error:
                return self._evaluation_terminal('diagnostic', error, events)
            events.append(dict(event='diagnostic', update=update, stage=self.stage,
                               stage_iteration=self.stage_iteration, **outcome))
            if outcome['collapse_streak'] >= COLLAPSE_CONSECUTIVE:
                self.terminal = 'behavior_collapse'
                self.save_checkpoint('behavior_collapse')
                self._emit('collapse', dict(update=update, stage=self.stage,
                                            stage_iteration=self.stage_iteration,
                                            streak=outcome['collapse_streak']))
                return dict(status='behavior_collapse', update=update, events=events,
                            stage=self.stage, stage_iteration=self.stage_iteration,
                            collapse_streak=outcome['collapse_streak'])
        transitioned = False
        if gate_due:
            try:
                gate = self.run_full_gate(digest)
                events.append(dict(event='validation', update=update, stage=self.stage,
                                   stage_iteration=self.stage_iteration,
                                   passed=gate['passed'],
                                   eval_seed=gate.get('eval_seed'),
                                   retention_eval_seed=gate.get('retention_eval_seed'),
                                   retention=gate['retention'],
                                   streak=gate['streak']))
                if self.curriculum.needs_perturbed:
                    perturbed = self.run_perturbed(digest)
                    events.append(dict(event='perturbed', update=update, stage='S1',
                                       stage_iteration=self.stage_iteration,
                                       passed=perturbed['passed'],
                                       eval_seed=perturbed.get('eval_seed'),
                                       streak=perturbed['streak']))
                if self.curriculum.can_advance:
                    events.append(self.transition())
                    transitioned = True
            except EvaluationError as error:
                return self._evaluation_terminal('validation', error, events)
        if (self.curriculum.complete and self.curriculum.final_result is None
                and self.final_test):
            try:
                passed = self.run_final_test()
            except EvaluationError as error:
                return self._evaluation_terminal('final_test', error, events)
            status = 'final_passed' if passed else 'final_test_failed'
            self.terminal = status
            self.save_checkpoint('trial_end')
            events.append(dict(event='final', update=update, passed=passed,
                               eval_seed=FINAL_TEST_SEED,
                               checkpoint_sha256=self.curriculum.frozen_candidate_sha))
            return dict(status=status, update=update, events=events,
                        checkpoint_sha256=self.curriculum.frozen_candidate_sha,
                        final_result=passed)
        if self.mode == 'pilot':
            verdict = self.curriculum.pilot_verdict()
            if verdict['status'] != 'running':
                status = 'pilot_complete' if verdict['status'] == 'passed' else 'pilot_failed'
                self.terminal = status
                self.save_checkpoint('trial_end')
                return dict(status=status, update=update, verdict=verdict, events=events,
                            collapse_streak=self.curriculum.collapse_streak)
        if self.curriculum.exhausted or self.iteration >= self.curriculum.update_cap:
            self.terminal = 'budget_exhausted'
            self.save_checkpoint('trial_end')
            return dict(status='budget_exhausted', update=update, events=events,
                        stage=self.stage, stage_iteration=self.stage_iteration,
                        total_updates=self.iteration,
                        update_cap=self.curriculum.update_cap,
                        effective_cap_source=self.curriculum.effective_cap_source)
        return dict(status='stage_transition' if transitioned else 'continue',
                    update=update, events=events)

    def _evaluation_terminal(self, kind, error, events):
        self.terminal = 'evaluation_error'
        payload = dict(status='evaluation_error', kind=kind, update=self.iteration,
                       error=repr(error), events=events,
                       collapse_streak=self.curriculum.collapse_streak)
        try:
            self.save_checkpoint('evaluation_error')
        except Exception as save_error:  # noqa: BLE001 - report both failures
            payload['checkpoint_error'] = repr(save_error)
        self._emit('evaluation_error', payload)
        return payload

    # ------------------------------------------------------------ evaluation
    def _tasks(self):
        return self.ledger.tasks

    def _case_seed(self, kind):
        """The **effective** case-table seed for one kind (evidence only).

        SDS 6.3 offsets are applied exactly once, inside
        ``evaluation.make_cases`` / ``evaluation._seed_for``.  This accessor
        returns the same effective value for reports/identity/logging and must
        never be passed back into ``make_cases`` (that would add the offset
        twice and silently use 400642 instead of 200342).
        """
        from .evaluation import _seed_for
        return int(_seed_for(kind, self.seed))

    def _services(self):
        if self.evaluation is not None and self.make_cases_fn is not None:
            return self.evaluation, self.make_cases_fn
        try:
            from . import evaluation as module
        except Exception as error:  # noqa: BLE001 - incomplete B package
            raise EvaluationError(
                'v05_r2/evaluation.py is required for v0.5-r2 gates and diagnostics') from error
        return (self.evaluation if self.evaluation is not None else module.evaluate_checkpoint,
                self.make_cases_fn if self.make_cases_fn is not None else module.make_cases)

    def _evaluation_dir(self, kind, stage, update):
        if self.output_dir is None:
            return None
        return self.output_dir / 'evaluations' / f'{kind}_{stage}_{update:06d}'

    def _run_task(self, kind, *, stage, checkpoint_sha):
        """Plan -> execute -> commit one evaluation and return its outcome.

        The durable transaction lives in the ledger; a repeated call with the
        same key is idempotent (planned -> run, committed -> consume, consumed ->
        reuse the recorded outcome).
        """
        evaluation, make_cases = self._services()
        update = self.iteration
        # SDS 6.3: pass the raw training seed; ``evaluation.make_cases`` owns the
        # single offset application (validation/flat 200300, perturbed 300300,
        # diagnostic 500300; test is the fixed 400342).
        cases = make_cases(stage, kind, self.seed)
        self.last_case_seed = int(getattr(cases, 'seed', self._case_seed(kind)))
        case_sha = _case_sha(cases)
        key = task_key(run_id=self.run_id, mode=self.mode, stage=stage,
                       global_update=update, kind=kind, checkpoint_sha256=checkpoint_sha,
                       case_sha256=case_sha)
        plan = self.ledger.plan_evaluation(
            run_id=self.run_id, mode=self.mode, stage=stage, global_update=update,
            kind=kind, checkpoint_sha256=checkpoint_sha, case_sha256=case_sha,
            planned_wall=self.clock())
        if plan['state'] == 'planned':
            result = self._evaluate(kind, stage=stage, cases=cases,
                                    checkpoint_sha=checkpoint_sha,
                                    output_dir=self._evaluation_dir(kind, stage, update))
            self._commit_result(key, result)
        record = self.ledger.tasks[key]
        if record['state'] == 'consumed':
            prior = record['consumption']
            return dict(passed=bool(prior['passed']),
                        collapse_hit=bool(prior['collapse_hit']),
                        outcome=prior['outcome'], key=key,
                        eval_seed=int(getattr(cases, 'seed', self._case_seed(kind))),
                        streak=self.ledger.curriculum['streak'])
        outcome = self._consume(key, record)
        outcome['eval_seed'] = int(getattr(cases, 'seed', self._case_seed(kind)))
        return outcome

    def _interpret(self, kind, report, *, stage, update):
        episodes = report.get('episodes')
        total = integer(episodes, 'report episodes', 1) if episodes is not None else (
            FINAL_TEST_CASES if kind == 'test' else GATE_CASES)
        if kind == 'diagnostic':
            decision = diagnostic_decision(
                self._diagnostic_payload(report), stairs=stage != 'F0',
                stage_update=self.curriculum.consumed[stage])
            return dict(passed=not decision['collapse_hit'],
                        collapse_hit=bool(decision['collapse_hit']),
                        outcome='collapse_hit' if decision['collapse_hit'] else 'clear',
                        decision=decision)
        rows = _speed_rows(report, expected_total=total)
        required = (CASE_PASS_REQUIRED_FINAL if kind == 'test'
                    else CASE_PASS_REQUIRED)
        passed = _record_passed(rows, required=required)
        return dict(passed=bool(passed), collapse_hit=False,
                    outcome='passed' if passed else 'failed',
                    per_speed=rows, required=required)

    def _diagnostic_payload(self, report):
        """Per-case diagnostic evidence; fail-closed when the evaluation omits it.

        The evaluation work package commits the same per-case block under
        ``per_case`` and/or ``diagnostic_cases`` (top-level or nested under
        ``diagnostic``); every spelling is accepted.  Field aliases are handled by
        :func:`diagnostic_decision`.
        """
        rows = report.get('per_case')
        if rows is None:
            rows = report.get('diagnostic_cases')
        if rows is None:
            diagnostic = report.get('diagnostic')
            if isinstance(diagnostic, dict):
                rows = diagnostic.get('per_case')
                if rows is None:
                    rows = diagnostic.get('cases')
        if rows is None:
            raise DiagnosticEvidenceError(
                'kind=diagnostic needs per-case evidence (c, samples/active_samples, '
                'stall_samples, first_step_acquired) under per_case or '
                'diagnostic_cases; the committed report has none. SDS 6.3 cannot be '
                'evaluated from speed-window counts.')
        return dict(per_case=rows)

    def _consume(self, key, record):
        report_path = record.get('result', {}).get('report_path')
        if not record.get('result') or not report_path or not Path(report_path).is_file():
            raise EvaluationError('Committed evaluation report is missing; re-run the same case')
        report = json.loads(Path(report_path).read_text())
        parts = split_task_key(key)
        interpretation = self._interpret(parts['kind'], report,
                                         stage=parts['stage'], update=parts['global_update'])
        result = self.ledger.consume_evaluation(
            key, outcome=interpretation['outcome'], passed=interpretation['passed'],
            collapse_hit=interpretation['collapse_hit'],
            consumed_wall=self.clock())
        return dict(interpretation, key=key, streak=result['streak'])

    def _commit_result(self, key, result):
        summary = _result_summary(result)
        manifest = _result_manifest(result, summary)
        out = Path(manifest).parent
        report = out / 'report.json'
        if not report.is_file():
            raise EvaluationError('Evaluation did not commit report.json')
        episodes = out / 'episodes.jsonl'
        npz = list(_result_paths(result, summary).get('npz', []))
        evidence = [_file_sha(path) for path in [report, manifest, episodes, *npz]
                    if Path(path).is_file()]
        if not evidence:
            raise EvaluationError('Evaluation committed no raw evidence')
        self.ledger.commit_evaluation_result(
            key, result_path=str(report), result_sha256=_file_sha(report),
            report_path=str(report), report_sha256=_file_sha(report),
            evidence_sha256=evidence, summary_sha256=_canonical_sha(summary),
            committed_wall=self.clock())

    def _evaluate(self, kind, *, stage, cases, checkpoint_sha, output_dir):
        evaluation, _ = self._services()
        if output_dir is None:
            raise EvaluationError('An evaluation requires the run output directory')
        with self._evaluation_transaction():
            frozen = self._frozen_model()
            try:
                result = evaluation(
                    frozen, cases, device=self.device, num_envs=self.eval_num_envs,
                    stage=stage, kind=kind, run_id=self.run_id,
                    global_update=self.iteration, checkpoint_sha=checkpoint_sha,
                    output_dir=str(output_dir), record_video=False,
                    mode=self.mode, seed=self.seed)
            except (EvaluationError, DiagnosticEvidenceError):
                raise
            except Exception as error:  # noqa: BLE001 - evaluation-service failures
                raise EvaluationError(f'{kind} evaluation failed: {error!r}') from error
        summary = _result_summary(result)
        bound_key = getattr(result, 'key', None)
        if bound_key is not None:
            expected = (self.run_id, self.mode, stage, self.iteration, kind,
                        checkpoint_sha, _case_sha(cases))
            if tuple(bound_key) != expected:
                raise EvaluationError('Evaluation result is bound to another task key')
        return result

    def _frozen_model(self):
        model = deepcopy(self.model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    @contextmanager
    def _evaluation_transaction(self):
        """Pause the rollout: freeze training state and restore every RNG stream."""
        rng = capture_rng(self.env)
        learning = self.learning
        counters = self._env_counters()
        error = None
        try:
            yield
        except BaseException as caught:  # noqa: BLE001 - restore then re-raise
            error = caught
            raise
        finally:
            restore_rng(rng, self.env)
            if error is None:
                if self.learning != learning:
                    raise RuntimeError('Evaluation advanced the training learning state')
                if not _same_counters(self._env_counters(), counters):
                    raise RuntimeError('Evaluation advanced the training environment')

    def _env_reset_count(self):
        value = getattr(self.env, 'reset_count', None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _env_counters(self):
        counters = {}
        for name in ('common_step_counter', 'episode_length_buf'):
            value = getattr(self.env, name, None)
            if isinstance(value, torch.Tensor):
                counters[name] = value.detach().clone()
            elif isinstance(value, int):
                counters[name] = value
        return counters

    # -------------------------------------------------------------- gates
    def run_diagnostic(self, checkpoint_sha):
        stage = self.stage
        outcome = self._run_task('diagnostic', stage=stage, checkpoint_sha=checkpoint_sha)
        compact = dict(outcome=outcome['outcome'],
                       collapse_hit=outcome['collapse_hit'],
                       eval_seed=outcome.get('eval_seed', self._case_seed('diagnostic')),
                       collapse_streak=self.curriculum.collapse_streak,
                       stalled_cases=outcome.get('decision', {}).get('stalled_cases'),
                       first_step_absent_cases=outcome.get('decision', {}).get(
                           'first_step_absent_cases'))
        self._emit('diagnostic', dict(update=self.iteration, stage=stage,
                                      stage_iteration=self.stage_iteration, **compact))
        return compact

    def run_full_gate(self, checkpoint_sha):
        """Full gate for the current stage plus the same-checkpoint F0 retention."""
        stage = self.stage
        retention = None
        if stage != 'F0':
            retention = self._run_task('flat_retention', stage='F0',
                                       checkpoint_sha=checkpoint_sha)
            if retention['outcome'] not in ('passed', 'failed'):
                raise EvaluationError('The F0 retention test produced no verdict')
        gate = self._run_task('validation', stage=stage, checkpoint_sha=checkpoint_sha)
        if gate['outcome'] not in ('passed', 'failed'):
            raise EvaluationError('The full gate produced no verdict')
        self._emit('validation', dict(update=self.iteration, stage=stage,
                                      stage_iteration=self.stage_iteration,
                                      eval_seed=gate.get('eval_seed'),
                                      retention_eval_seed=(
                                          None if retention is None
                                          else retention.get('eval_seed')),
                                      passed=gate['passed'],
                                      retention=None if retention is None else retention['passed'],
                                      streak=gate['streak']['gate']))
        return dict(passed=gate['passed'],
                    eval_seed=gate.get('eval_seed'),
                    retention_eval_seed=(None if retention is None
                                         else retention.get('eval_seed')),
                    retention=None if retention is None else retention['passed'],
                    streak=gate['streak']['gate'],
                    flat_retention_passed=None if retention is None else retention['passed'])

    def run_perturbed(self, checkpoint_sha):
        """S1 second-pass perturbation check on the very same checkpoint."""
        outcome = self._run_task('perturbed', stage='S1', checkpoint_sha=checkpoint_sha)
        return dict(passed=outcome['passed'], eval_seed=outcome.get('eval_seed'),
                    streak=outcome['streak']['gate'])

    def run_final_test(self):
        """W10 final test: only the frozen second-pass S5 checkpoint, once."""
        candidate = self.curriculum.frozen_candidate_sha
        if candidate is None:
            raise EvaluationError('No frozen S5 candidate to test')
        outcome = self._run_task('test', stage='S5', checkpoint_sha=candidate)
        return bool(outcome['passed'])

    # ------------------------------------------------------------ transition
    def transition(self):
        """Qualified stage transition: checkpoint, ledger event, env reset.

        All learning state (model/Adam/std) is preserved; only the environment
        episode and slot binding change.
        """
        pending = self.ledger.pending_transition
        if not self.curriculum.can_advance:
            raise RuntimeError('The stage has not qualified for a transition')
        pre = self.save_checkpoint('stage_transition_pre_reset')
        self.ledger.commit_transition(from_stage=pending['from_stage'],
                                      to_stage=pending['to_stage'],
                                      at_update=pending['at_update'],
                                      gate_streak=self.ledger.gate_streak)
        committed = self.save_checkpoint('stage_transition_committed')
        self.env.set_stage(pending['to_stage'], stage_iteration=0,
                           previous_stage=pending['from_stage'])
        obs, info = self._environment_reset()
        self._adopt_observation(obs, info)
        start = self.save_checkpoint('stage_start')
        self._emit('curriculum/stage', dict(from_stage=pending['from_stage'],
                                            to_stage=pending['to_stage'],
                                            update=self.iteration))
        return dict(event='stage_transition', update=self.iteration,
                    from_stage=pending['from_stage'], to_stage=pending['to_stage'],
                    checkpoint_pre_reset=pre, checkpoint_committed=committed,
                    checkpoint_stage_start=start, learning_preserved=True)

    def controlled_pause(self, reason='monitoring_blocked'):
        """Controlled pause at a complete update boundary (SDS 7.2)."""
        if reason not in ('monitoring_blocked', 'disk_space_pause', 'operator_pause'):
            raise ValueError(f'Unknown pause reason {reason!r}')
        digest = self.save_checkpoint(reason)
        self.terminal = reason
        self._emit('pause', dict(reason=reason, update=self.iteration, sha256=digest))
        return dict(status=reason, checkpoint=str(self._checkpoint_path(reason)),
                    sha256=digest)

    # ------------------------------------------------------------ checkpoint
    def _checkpoint_path(self, kind):
        if self.output_dir is None:
            raise ValueError('A checkpoint requires an output directory')
        return self.output_dir / 'checkpoints' / f'{kind}_{self.iteration:06d}.pt'

    def _normalizer_state(self):
        return {key: value for key, value in self.model.state_dict().items()
                if key.startswith('obs_normalizer.')}

    def _checkpoint_task_state(self):
        """Durable task/streak snapshot for the schema-7 ``ledger`` block."""
        return self.ledger.checkpoint_task_state()

    def schema7_state(self):
        """Build the exact schema-7 envelope through ``v05_r2.recovery``."""
        from .recovery import source_block
        return build_checkpoint(
            model_state=self.model.state_dict(),
            optimizer_state=self.ppo.optimizer.state_dict(),
            normalizer=self._normalizer_state(),
            learning=self.learning,
            rng=capture_rng(self.env),
            curriculum=self.ledger.curriculum,
            ledger=self._checkpoint_task_state(),
            evaluation=self.ledger.checkpoint_evaluation(),
            source=source_block(mode=self.mode, seed=self.seed, device=self.device,
                                resume=self.lineage),
            mode=self.mode, seed=self.seed, device=self.device,
            parameters=list(self.model.parameters()))

    def save_checkpoint(self, kind, *, path=None, metadata=None):
        """Write one complete schema-7 boundary checkpoint and commit its digest."""
        if not self.save_checkpoints:
            return None
        if self.ledger is None:
            raise ValueError('A formal schema-7 checkpoint requires the durable ledger')
        if metadata:
            self.lineage.append(dict(kind=kind, metadata=json.loads(canonical_json(metadata))))
        state = self.schema7_state()
        path = Path(path) if path is not None else self._checkpoint_path(kind)
        if path.exists():
            raise ValueError(f'Refusing to overwrite checkpoint {path}')
        from .recovery import atomic_torch_save
        atomic_torch_save(path, state)
        digest = _file_sha(path)
        self.ledger.record_checkpoint(path, digest, self.ledger.anchor(),
                                      state['contract_sha256'])
        self._emit('checkpoint', dict(kind=kind, path=str(path), sha256=digest,
                                      update=self.iteration, stage=self.stage))
        return digest

    def resume(self, checkpoint, expected_sha, *, replay_pending=True):
        """SDS 7.3 recovery order; one and only one explicit new-episode reset."""
        state = preload_checkpoint(checkpoint, expected_sha, mode=self.mode)
        validate_checkpoint(state, parameters=list(self.model.parameters()))
        # SDS 7.5: the model body is the exact r2 state_dict, checked against the
        # caller's model (strict: no strict=False, no missing route tensor).
        from .model import validate_model_state
        validate_model_state(state['model'], self.model)
        learning = state['learning']
        validate_learning(learning, num_envs=self.num_envs)
        # SDS 7.3: resume is the fourth consumer of the same mode budget source.
        from .recovery import audit_budget, validate_route_model_state
        validate_route_model_state(state['model'])
        audit_budget(learning, state['curriculum'], mode=self.mode)
        if self.ledger.completed_ppo_updates != learning['update']:
            raise RecoveryBlocked(
                'checkpoint update differs from the durable ledger boundary; select the '
                'latest complete ledger boundary before resuming')
        checkpoint_curriculum = state['curriculum']
        ledger_curriculum = self.ledger.curriculum
        if (checkpoint_curriculum['stage'] != ledger_curriculum['stage']
                or checkpoint_curriculum['stage_iteration']
                != ledger_curriculum['stage_iteration']):
            raise RecoveryBlocked(
                'checkpoint stage boundary differs from the durable ledger boundary; '
                'select the latest complete ledger boundary before resuming')
        # Same complete-update boundary: later evaluation consumption or a pending
        # transition on this update is uniquely reconciled by adopting the durable
        # ledger curriculum (SDS 7.2/7.3).
        curriculum = ledger_curriculum
        self.model.load_state_dict(state['model'], strict=True)
        self.ppo.optimizer.load_state_dict(state['optimizer'])
        self.ppo.learning_state = dict(
            optimizer_initialized=learning['update'] > 0,
            completed_ppo_updates=learning['update'],
            optimizer_steps=learning['optimizer_step'])
        self.lineage = deepcopy(state['source'].get('resume', []))
        self.lineage.append(dict(event='resume', checkpoint=str(checkpoint),
                                 update=learning['update'], learning=dict(learning)))
        # SDS 7.3 order: bind the curriculum/slots *before* the RNG restore.  The
        # real V05Env resets inside set_stage when the stage actually changes; that
        # transient binding reset happens before the restore and is discarded by it.
        # After the restore exactly one explicit new-episode reset is permitted.
        pending = self.ledger.pending_transition
        if pending is not None and pending['to_stage'] is None:
            raise RecoveryBlocked('the terminal pending qualified state is not resumable')
        if pending is not None:
            bind_stage, bind_iteration = pending['to_stage'], 0
            bind_previous = pending['from_stage']
        else:
            bind_stage, bind_iteration = curriculum['stage'], curriculum['stage_iteration']
            bind_previous = None
        self.env.set_stage(bind_stage, stage_iteration=bind_iteration,
                           previous_stage=bind_previous)
        resets_before_restore = self._env_reset_count()
        restore_rng(state['rng'], self.env)
        if (resets_before_restore is not None
                and self._env_reset_count() != resets_before_restore):
            raise RuntimeError('restore_rng must not reset the environment')
        if pending is not None:
            self.ledger.commit_transition(from_stage=pending['from_stage'],
                                          to_stage=pending['to_stage'],
                                          at_update=pending['at_update'],
                                          gate_streak=self.ledger.gate_streak)
        resets_before_new_episode = self._env_reset_count()
        obs, info = self._environment_reset()
        self._adopt_observation(obs, info)
        self.sync_stage_iteration()
        resets_after_new_episode = self._env_reset_count()
        if (resets_before_new_episode is not None
                and resets_after_new_episode != resets_before_new_episode + 1):
            raise RuntimeError(
                'resume must perform exactly one explicit new-episode reset after the RNG '
                'restore; a stage binding or slot restatement reset the environment again')
        self.terminal = None
        replayed = None
        if replay_pending:
            replayed = self.replay_pending_evaluations()
            if self.ledger.collapse_streak >= COLLAPSE_CONSECUTIVE:
                self.terminal = 'behavior_collapse'
        return dict(status='resumed', update=learning['update'], stage=self.stage,
                    stage_iteration=self.stage_iteration,
                    pending_evaluations=replayed, terminal=self.terminal)

    def replay_pending_evaluations(self):
        """Replay committed or planned tasks before more PPO (SDS 7.3)."""
        replayed = []
        for key, record in self.ledger.pending_evaluations().items():
            parts = split_task_key(key)
            if record['state'] == 'planned':
                evaluation, make_cases = self._services()
                cases = make_cases(parts['stage'], parts['kind'], self.seed)
                self.last_case_seed = int(getattr(cases, 'seed',
                                                  self._case_seed(parts['kind'])))
                if _case_sha(cases) != parts['case_sha256']:
                    raise RecoveryBlocked(
                        'pending evaluation case table differs from the durable key')
                result = self._evaluate(parts['kind'], stage=parts['stage'], cases=cases,
                                        checkpoint_sha=parts['checkpoint_sha256'],
                                        output_dir=self._evaluation_dir(
                                            parts['kind'], parts['stage'],
                                            parts['global_update']))
                self._commit_result(key, result)
                record = self.ledger.tasks[key]
            outcome = self._consume(key, record)
            replayed.append(dict(key=key, kind=parts['kind'], passed=outcome['passed'],
                                 collapse_hit=outcome['collapse_hit'],
                                 outcome=outcome['outcome']))
        return replayed

    # -------------------------------------------------------------- telemetry
    def _emit(self, kind, metrics):
        if self.telemetry is None:
            return
        emit = getattr(self.telemetry, 'emit', None)
        if callable(emit):
            emit(kind, metrics)
        else:
            self.telemetry_hooks_missing.add(kind)


# ------------------------------------------------------- evaluation adapters
def _result_summary(result):
    if isinstance(result, dict):
        return result.get('summary', result)
    summary = getattr(result, 'summary', None)
    if not isinstance(summary, dict):
        raise EvaluationError('Evaluation result must expose a summary dict')
    return summary


def _result_paths(result, summary):
    paths = getattr(result, 'raw_paths', None)
    if not isinstance(paths, dict):
        paths = summary.get('raw_paths')
    return paths if isinstance(paths, dict) else {}


def _result_manifest(result, summary):
    paths = _result_paths(result, summary)
    manifest = paths.get('manifest')
    if not isinstance(manifest, str) or not Path(manifest).is_file():
        raise EvaluationError('Evaluation result has no committed manifest.json')
    return manifest


def _case_sha(cases):
    digest = getattr(cases, 'sha256', None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise EvaluationError('Case table must expose a 64-hex sha256')
    return digest


def _same_counters(first, second):
    if set(first) != set(second):
        return False
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                    or not torch.equal(left, right)):
                return False
        elif left != right:
            return False
    return True


__all__ = [
    'ADAM_STEPS_PER_UPDATE', 'COLLAPSE_CONSECUTIVE', 'COLLAPSE_MIN_CASES',
    'COLLAPSE_STALL_FRACTION', 'COLLAPSE_START_STAGE_UPDATE', 'Curriculum',
    'DIAGNOSTIC_CASES', 'DIAGNOSTIC_INTERVAL', 'DiagnosticEvidenceError',
    'EPISODE_ROUTE_FIELDS', 'EvaluationError', 'GATE_CASES', 'OBSERVATION_KEYS',
    'PILOT_BUDGETS', 'PILOT_FINAL_GATES', 'PILOT_MAX_UPDATES', 'PILOT_P1_UPDATES',
    'ROUTE_DIAGNOSTIC_FIELDS', 'ROUTE_ERROR_KEY',
    'PILOT_GATE_UPDATES', 'PilotLedgerConflict', 'RESUME_MODE', 'REWARD_STATE',
    'SAVE_INTERVAL', 'V05Runner', 'diagnostic_decision', 'integer',
    'route_diagnostics', 'route_episode_audit', 'validate_route_observation']
