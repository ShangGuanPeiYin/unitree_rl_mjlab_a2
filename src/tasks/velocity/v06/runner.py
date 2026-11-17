"""v0.6 runner: one continuous F0->H->D->L->V->R training run.

The runner owns the training loop, the checkpoint discipline, the gate schedule
(100 updates), the diagnostic schedule (50), the retention sets (200) and the
stop-state classification.  The curriculum rules stay in agent B's
``v06/curriculum.py``: this module only calls
``CurriculumManager.schedule/observe/on_update/retention_due/state_dict/
load_state_dict`` and reacts to the phase/substage it reports.  No curriculum
threshold, EMA or budget rule is re-implemented here.

States come from the frozen spec single source (Q002-4): the eight stop states
plus the ``running``/``ready_for_final_test``/``complete`` lifecycle.  R passing
twice only reaches ``ready_for_final_test``; ``complete`` exists solely after the
frozen 300-case final test passed 95/100 per speed.  A user stop is never a
training or monitoring failure, and an engineering error keeps its typed
``error_type`` instead of being relabelled as a behavioural stop.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import evaluation as evaluation_module
from . import model as model_module
from . import ppo as ppo_module
from . import recovery as recovery_module
from . import spec
from . import speed

#: Q002-4: the frozen spec is the single source of every training state.
STOP_STATES = tuple(spec.STOP_STATES)
LIFECYCLE_STATES = tuple(spec.LIFECYCLE_STATES)
ALL_STATES = tuple(spec.ALL_TRAINING_STATES)
#: Distinct failure classes; a user stop is never a training or communication failure.
FAILURE_CLASS = {
    'complete': None,
    'running': None,
    'ready_for_final_test': None,
    'user_requested_stop': 'user',
    'behavior_regression': 'training_failure',
    'behavior_collapse': 'training_failure',
    'gate_failed': 'training_failure',
    'budget_exhausted': 'training_failure',
    'resource_insufficient': 'resource',
    'curriculum_stalled': 'training_failure',
    'monitoring_blocked': 'communication',
}
#: Q002-5 budget-scope reasons.
PHASE_BUDGET_REASON = 'phase_budget_exhausted'
SUBSTAGE_BUDGET_REASON = 'substage_budget_exhausted'
FINAL_TEST_REASON = 'final_test_failed'
#: Phase-transition freeze rules (SDS 5.5/5.6/6.1): consecutive full-gate passes
#: required before a phase may advance, and the minimum update count before a
#: gate may count toward the streak.
PHASE_GATE_RULES = {
    'F0': dict(required=2, min_updates=0),
    'H': dict(required=2, min_updates=0),
    'D': dict(required=2, min_updates=0),
    'L': dict(required=2, min_updates=0),
    'V': dict(required=2, min_updates=1000, minimum_updates=1200),
    'R': dict(required=2, min_updates=400, minimum_updates=600),
}


#: Sentinel for ``run_gate(round=...)``: the default derives the case-table round
#: from the update number, while a pending-boundary replay passes ``None`` to
#: reproduce the recorded (round-less) pending case-table identity exactly.
_ROUND_FROM_UPDATE = object()


class RunnerError(RuntimeError):
    """The runner cannot continue (configuration or integration error)."""


class RunStopped(RuntimeError):
    """The run ended; ``state`` names the SDS terminal state."""

    def __init__(self, state, *, evidence=None):
        super().__init__(f'run stopped: {state}')
        self.state = state
        self.evidence = evidence or {}


def classify_failure(state):
    return FAILURE_CLASS.get(state)


def classify_stop(*, user_requested=False, resource_available=True,
                  behavior_regression=False, gate_failed=False,
                  budget_exhausted=False, complete=False):
    """Terminal-state priority: user > resource > regression > total budget > gate.

    The 30,000-update budget is the run-level stop; ``gate_failed`` is the earlier
    stop for a *phase* budget that ended without a legal advance.
    """
    if complete:
        return 'complete'
    if user_requested:
        return 'user_requested_stop'
    if not resource_available:
        return 'resource_insufficient'
    if behavior_regression:
        return 'behavior_regression'
    if budget_exhausted:
        return 'budget_exhausted'
    if gate_failed:
        return 'gate_failed'
    return None


@dataclass
class GateRecord:
    phase: str
    substage: object
    kind: str
    update: int
    passed: bool
    case_table_sha256: str
    checkpoint_sha256: object
    failed_cases: int
    summary: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(phase=self.phase, substage=self.substage, kind=self.kind,
                    update=int(self.update), passed=bool(self.passed),
                    case_table_sha256=self.case_table_sha256,
                    checkpoint_sha256=self.checkpoint_sha256,
                    failed_cases=int(self.failed_cases), summary=deepcopy(self.summary))


def _canonical(value):
    return recovery_module.canonical(value)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _nested_equal(left, right):
    """Deep equality for tensor/dict/list checkpoint blocks (no torch.allclose)."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False
        return (tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(_nested_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, float) or isinstance(right, float):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return left == right
    return left == right


def _rng_equal(left, right):
    """Compare two ``recovery.capture_rng`` snapshots exactly."""
    try:
        if set(left) != set(right):
            return False
        return all(_nested_equal(left[key], right[key]) for key in left)
    except (TypeError, ValueError):
        return False


class V06Runner:
    """Single continuous v0.6 training run with schema-8 checkpoints."""

    def __init__(self, *, model, ppo, curriculum, directory, env=None, seed=42,
                 device='cpu', mode='train', telemetry=None, case_runner=None,
                 stop_flag=None, resource_guard=None, acceptance_file=None,
                 source=None, container=None, evaluate=True, autosave=True,
                 require_resume_evidence=False, clock=time.time):
        if mode not in ('train', 'resume', 'probe'):
            raise RunnerError('v0.6 runner modes are train, resume and probe only')
        self.model = model
        self.ppo = ppo
        self.curriculum = curriculum
        self.env = env
        self.directory = Path(directory)
        self.seed = int(seed)
        self.device = str(device)
        self.mode = mode
        self.container = container
        # The engineering probe bills 55+1 updates without gates/diagnostics/
        # retention evaluation or interval autosaves (it saves its own checkpoint).
        self.evaluate = bool(evaluate)
        self.autosave = bool(autosave)
        self.telemetry = telemetry
        self.case_runner = case_runner
        self.stop_flag = stop_flag
        self.resource_guard = resource_guard
        self.acceptance_file = acceptance_file
        self.clock = clock
        self.source = source or recovery_module.source_block(
            seed=self.seed, device=self.device, mode=mode)
        self.starts = self.clock()
        # ---------------------------------------------------------- run state
        self.update = int(self.ppo.learning['update'])
        self.phase = str(self._curriculum_state().get('phase', 'F0')).upper()
        self.substage = self._curriculum_state().get('substage')
        self.gate_streak = 0
        self.gates = []
        # Q001-7: flat and H00 retention sets keep independent streaks.
        self.retention = dict(streaks={'flat': 0, 'h00': 0}, last={}, history=[],
                              enabled={'flat': False, 'h00': False})
        # Q001-7: inherited stall-collapse counter (independent of every other stop).
        self.diagnostic = dict(streak=0, generation=None, last=None, history=[],
                               required=None, cases=0, stalled_cases=0,
                               first_step_absent_cases=0, hits=0)
        self.minimum_level_failure_streak = 0
        self.checkpoints = []
        self.diagnostics = []
        self.transitions = []
        self.pending_transition = None
        self.stop_state = None
        self.stop_reason = None
        self.error_type = None
        self.stop_evidence = {}
        self.lifecycle_state = 'running'
        self.curriculum_complete = False
        self.pending_evaluations = []
        self.pending_boundary = None
        self.incomplete_update = None
        self.final = None
        self.last_metrics = {}
        self.last_phase_save = None
        # Q006 section 4: a resumed run must persist and independently re-verify
        # ``resume_loaded`` (and, after a pending boundary, ``resume_ready``)
        # before any evaluation or learning update is allowed.
        self.require_resume_evidence = bool(require_resume_evidence)
        self.resume_origin = None
        self.resume_chain = []
        self.resume_loaded = None
        self.resume_ready = None
        self._resume_had_pending_boundary = False
        self.pending_task_migration = None
        self.evaluation_transaction = None
        self._evaluation_transaction_durable = False
        # Q007 section 2: a checkpoint may carry CUDA RNG state even when this CPU
        # process cannot restore it.  Keep that recorded blob as evidence and pass
        # it through unchanged instead of silently replacing it with [].
        self.cuda_rng_state = None
        # Q007 section 6: a resumed run persists each completed pending evaluation
        # as its own checkpoint so a later failure (e.g. the 600 dual retention)
        # never loses an already finished set or replays it.
        self.save_pending_results = bool(require_resume_evidence)

    # ------------------------------------------------------------ curriculum
    def _curriculum_state(self):
        state = self.curriculum.state_dict()
        if not isinstance(state, dict):
            raise RunnerError('CurriculumManager.state_dict() must return a mapping')
        return state

    def _sync_phase(self):
        state = self._curriculum_state()
        self.phase = str(state.get('phase', self.phase)).upper()
        self.substage = state.get('substage')
        return state

    def _feed_gate(self, passed, update):
        """Hand the gate outcome to the curriculum manager (no local rule copy)."""
        for name in ('record_gate', 'note_gate', 'observe_gate'):
            hook = getattr(self.curriculum, name, None)
            if callable(hook):
                try:
                    parameters = inspect.signature(hook).parameters
                except (TypeError, ValueError):
                    parameters = {}
                payload = dict(passed=bool(passed), update=int(update), kind='gate')
                if 'phase' in parameters:
                    payload['phase'] = self.phase
                if 'substage' in parameters:
                    payload['substage'] = self.substage
                return hook(**payload)
        return None

    def _feed_retention(self, kind, failed_cases, passed, *, update=None):
        for name in ('record_retention', 'note_retention', 'observe_retention'):
            hook = getattr(self.curriculum, name, None)
            if callable(hook):
                try:
                    parameters = inspect.signature(hook).parameters
                except (TypeError, ValueError):
                    parameters = {}
                payload = dict(kind=kind, failed_cases=int(failed_cases),
                               passed=bool(passed))
                if 'update' in parameters:
                    payload['update'] = update
                return hook(**payload)
        return None

    def _on_update_hook(self, update, gate=None):
        """``on_update`` carrying this boundary's gate into the curriculum.

        The gate is evaluated *before* the tick (so the just-trained update is
        measured) and handed to the curriculum in the same call, which ingests it
        with the post-tick accounting, commits any legal transition and only then
        judges the budget (Q002-1/Q002-5).  Retention results never travel here
        (the curriculum ignores retention kinds in the gate path).
        """
        gates = [dict(gate)] if gate else []
        try:
            parameters = inspect.signature(self.curriculum.on_update).parameters
        except (TypeError, ValueError):
            parameters = {}
        if 'gates' in parameters:
            return self.curriculum.on_update(update, gates=gates)
        return self.curriculum.on_update(update)

    # ---------------------------------------------------------------- helpers
    @property
    def learning(self):
        return self._learning_block()

    @property
    def phase_update(self):
        spent = dict(self._curriculum_state().get('spent_by_phase', {}))
        return int(spent.get(self.phase, 0))

    def _current_tile(self, override=None):
        """The terrain a gate/retention/diagnostic must measure (never H00 by default).

        F0 measures the flat tile, D/L their substage tile and V/R the final L40
        stair.  Phase H has no substage, so it must use the live training
        environment's current ``frontier_tile_id`` (or the curriculum state's
        frontier); a missing frontier is a hard error instead of an H00 fallback.
        """
        phase = self.phase
        if phase == 'F0':
            return int(spec.FLAT_TILE_ID)
        if phase in ('D', 'L'):
            if self.substage is None:
                raise RunnerError(f'{phase} evaluation requires a substage')
            tile = int(spec.tile_id_for(str(self.substage)))
        elif phase in ('V', 'R'):
            tile = int(spec.FINAL_TILE_ID)
        elif phase == 'H':
            tile = getattr(self.env, 'frontier_tile_id', None)
            if tile is None:
                tile = self._curriculum_state().get('frontier_tile_id')
            if tile is None:
                raise RunnerError(
                    'H evaluation requires the current frontier_tile_id; the training '
                    'environment must expose it (defaulting to H00 is forbidden)')
            tile = int(tile)
        else:
            raise RunnerError(f'unknown phase {phase!r}')
        if override is not None and int(override) != tile:
            raise RunnerError(
                f'gate tile override {int(override)} differs from the current {phase} '
                f'tile {tile}')
        return tile

    @property
    def budget(self):
        return dict(update=self.update, total=speed.total_budget(),
                    remaining=speed.total_budget() - self.update,
                    phase=self.phase,
                    phase_budget=speed.phase_budget(self.phase) if self.phase in
                    speed.phase_order() else None,
                    phase_spent=self.phase_update)

    def _ensure_directories(self):
        for name in ('checkpoints', 'gates', 'diagnostics', 'retention', 'final_test'):
            (self.directory / name).mkdir(parents=True, exist_ok=True)

    def _write_json(self, relative, value):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
        path.write_text(payload + '\n', encoding='utf-8')
        return path

    # ------------------------------------------------------------- checkpoint
    def _learning_block(self):
        learning = dict(self.ppo.learning)
        learning.update(num_envs=int(getattr(self.env, 'num_envs', ppo_module.NUM_ENVS)),
                        rollout=ppo_module.ROLLOUT, epochs=ppo_module.EPOCHS,
                        minibatches=ppo_module.MINIBATCHES,
                        learning_rate=float(self.ppo.optimizer.param_groups[0]['lr']))
        return learning

    def _normalizer_block(self):
        return {
            'actor': {key: value.detach().clone()
                      for key, value in self.model.actor.normalizer.state_dict().items()},
            'critic': {key: value.detach().clone()
                       for key, value in self.model.critic_normalizer.state_dict().items()},
        }

    def _evaluation_block(self):
        tables = {}
        for record in self.gates:
            key = f'{record.phase}:{record.substage}:{record.kind}:{record.update}'
            tables[key] = dict(kind=record.kind, phase=record.phase,
                               substage=record.substage, update=record.update,
                               case_table_sha256=record.case_table_sha256,
                               passed=record.passed)
        return dict(cases=tables, latest=(self.gates[-1].as_dict() if self.gates else None),
                    seed=int(self.seed), offset_source=evaluation_module.OFFSET_SOURCE)

    def _gate_block(self):
        return dict(identity=self.gate_identity(), streak=int(self.gate_streak),
                    records=[record.as_dict() for record in self.gates[-20:]],
                    phase=self.phase, substage=self.substage, update=self.update)

    def gate_identity(self):
        return dict(task=model_module.TASK_ID, recipe=model_module.RECIPE_VERSION,
                    phase=self.phase, substage=self.substage,
                    gate_interval=int(speed.GATE_INTERVAL),
                    cases_per_speed=int(speed.GATE_CASES),
                    pass_min=int(speed.GATE_PASS_MIN),
                    final_cases=int(speed.FINAL_CASES),
                    final_pass_min=int(speed.FINAL_PASS_MIN),
                    seed=int(self.seed),
                    offset_source=evaluation_module.OFFSET_SOURCE,
                    forward_distance_definition=speed.forward_distance_definition())

    def _retention_schedule(self):
        """Mirror the curriculum's per-set Q002-6 schedule (single source = B)."""
        state = self._curriculum_state()
        schedule = {}
        for kind in ('flat', 'h00'):
            schedule[kind] = dict(
                qualified_at=(state.get('retention_qualified_at') or {}).get(kind),
                next_due=(state.get('retention_next_due') or {}).get(kind),
                last_completed=(state.get('retention_last_completed') or {}).get(kind),
                streak=int((state.get('retention_streaks') or {}).get(kind, 0)),
                enabled=bool((state.get('retention_enabled') or {}).get(kind, False)))
        return schedule

    def _retention_block(self):
        """Per-kind retention evidence (Q001-7: independent, never merged)."""
        return dict(streaks={str(key): int(value)
                             for key, value in self.retention['streaks'].items()},
                    enabled=dict(self.retention['enabled']),
                    last=deepcopy(self.retention['last']),
                    history=[dict(item) for item in self.retention['history'][-20:]],
                    cases=int(speed.RETENTION_CASES),
                    fail_cases=int(speed.RETENTION_FAIL_CASES),
                    consecutive=int(speed.RETENTION_CONSECUTIVE),
                    interval=int(speed.RETENTION_INTERVAL),
                    definition='behavior_regression after two consecutive >=12/15 '
                               'failures, per independent flat/H00 set')

    def _diagnostic_block(self):
        """Inherited stall-collapse counter (Q001-7), independent of every other stop."""
        return dict(streak=int(self.diagnostic['streak']),
                    required=self.diagnostic['required'],
                    generation=self.diagnostic['generation'],
                    last=deepcopy(self.diagnostic['last']),
                    history=[dict(item) for item in self.diagnostic['history'][-20:]],
                    cases=int(self.diagnostic.get('cases', 0)),
                    stalled_cases=int(self.diagnostic.get('stalled_cases', 0)),
                    first_step_absent_cases=int(
                        self.diagnostic.get('first_step_absent_cases', 0)),
                    hits=int(self.diagnostic.get('hits', 0)),
                    threshold=float(evaluation_module.STALL_FRACTION),
                    min_cases=int(evaluation_module.COLLAPSE_MIN_CASES),
                    min_first_step_absent=int(
                        evaluation_module.COLLAPSE_MIN_FIRST_STEP_ABSENT),
                    start_stage_update=int(evaluation_module.COLLAPSE_START_STAGE_UPDATE),
                    definition='inherited r2 stagnation collapse; streak only from '
                               'spec.DIAGNOSTIC_COLLAPSE_STREAK')

    def _initialization_block(self):
        metadata = getattr(self.model, 'transfer_metadata', None)
        try:
            model_module.validate_transfer_metadata(metadata)
        except ValueError as error:
            raise RunnerError(
                f'the model has no admissible ETH transfer metadata; v0.6 cannot save '
                f'a checkpoint without the original initialization evidence: {error}'
            ) from None
        return dict(contract=self.gate_identity(),
                    model=_digest({key: value for key, value in
                                   self.model.state_dict().items()}),
                    learning_state=deepcopy(self._learning_block()),
                    optimizer='single_adam',
                    scope='initialization', transfer_metadata=deepcopy(metadata),
                    resume_origin=deepcopy(self.resume_origin),
                    resume_chain=[dict(item) for item in self.resume_chain],
                    audit_schema=8, source_checkpoint_sha256=metadata.get('sha256'),
                    source_model_sha256=metadata.get('sha256'),
                    source_sha256=metadata.get('sha256'),
                    migrated_model_sha256=_digest(self.model.state_dict()),
                    seed=int(self.seed), source_path=metadata.get('source_path'),
                    resume=False, physical_trajectory_continuous=False,
                    resume_mode=recovery_module.RESUME_MODE,
                    notes='ETH Actor MLP + frozen 45-dim normalizer only; bypasses zero')

    def _pending_update_marker(self):
        """One uncommitted billed update (exception-path evidence), else None."""
        spent = sum(int(value) for value in
                    dict(self._curriculum_state().get('spent_by_phase') or {}).values())
        return int(self.update) if int(self.update) == spent + 1 else None

    def _rng_block(self):
        """Captured RNG; a CPU resume carries the recorded CUDA blob as evidence."""
        rng = recovery_module.capture_rng(self.env)
        if not torch.cuda.is_available() and self.cuda_rng_state is not None:
            rng['cuda'] = deepcopy(self.cuda_rng_state)
        return rng

    def checkpoint_state(self, pending_update=None, pending_boundary=None,
                         incomplete_update=None):
        optimizer_state = deepcopy(self.ppo.optimizer.state_dict())
        learning = self._learning_block()
        return recovery_module.build_checkpoint(
            model_state={key: value.detach().clone()
                         for key, value in self.model.state_dict().items()},
            optimizer_state=optimizer_state,
            learning_rate=float(self.ppo.optimizer.param_groups[0]['lr']),
            normalizer=self._normalizer_block(), learning=learning,
            rng=self._rng_block(),
            curriculum=self._curriculum_state(),
            evaluation=self._evaluation_block(), initialization=self._initialization_block(),
            gate=self._gate_block(), retention=self._retention_block(),
            ledger=dict(checkpoints=[dict(item) for item in self.checkpoints[-40:]],
                        diagnostics=[dict(item) for item in self.diagnostics[-40:]],
                        transitions=[dict(item) for item in self.transitions[-40:]],
                        diagnostic_collapse=self._diagnostic_block(),
                        pending_boundary=deepcopy(
                            self.pending_boundary if pending_boundary is None
                            else pending_boundary),
                        incomplete_update=deepcopy(
                            self.incomplete_update if incomplete_update is None
                            else incomplete_update),
                        evaluation_transaction=deepcopy(self.evaluation_transaction),
                        stop_state=self.stop_state,
                        stop_reason=self.stop_reason,
                        error_type=self.error_type,
                        lifecycle_state=self.lifecycle_state,
                        pending_evaluations=[dict(item) for item in
                                             self.pending_evaluations],
                        terminal=self._terminal_record(),
                        retention_schedule=self._retention_schedule()),
            pending_transition=deepcopy(self.pending_transition), source=self.source,
            recipe=spec_recipe(), container=self.container,
            pending_update=pending_update)

    def save(self, reason, *, update=None, pending_update=None, pending_boundary=None,
             incomplete_update=None):
        """Write a schema-8 checkpoint (every 100 updates, gate, transition, stop).

        ``pending_update`` is only used by the exception path: the completed update
        is billed in ``learning`` but its curriculum tick could not commit, so the
        checkpoint explicitly carries that one-update difference instead of failing
        to save any evidence.
        """
        self._ensure_directories()
        update = int(self.update if update is None else update)
        make_checkpoint = self.checkpoint_state
        state = make_checkpoint(pending_update=pending_update,
                                pending_boundary=pending_boundary,
                                incomplete_update=incomplete_update)
        path = self.directory / 'checkpoints' / f'{reason}_{update:06d}.pt'
        sha = recovery_module.atomic_save(path, state)
        record = dict(path=str(path), sha256=sha, update=update, reason=str(reason),
                      phase=self.phase, substage=self.substage,
                      pending_update=(None if pending_update is None
                                      else int(pending_update)),
                      pending_boundary=bool(
                          state['ledger'].get('pending_boundary') is not None),
                      created=float(self.clock()))
        self.checkpoints.append(record)
        self._write_json(f'checkpoints/{reason}_{update:06d}.json',
                         dict(record=record, audit=recovery_module.audit_budget(
                             state['learning'], state['curriculum'],
                             pending_update=pending_update)))
        return record

    # ------------------------------------------------- Q006 resume evidence
    def save_verified(self, reason, *, allow_bridge=False, allow_probe=None):
        """Atomically save a checkpoint, then reload it into a bare model + Adam.

        Q006 section 4.3: the saved ``resume_loaded``/``resume_ready`` evidence is
        only accepted after the *formal* recovery reader loads it into an
        independent ``V06ActorCritic`` and a fresh single Adam and every tensor,
        Adam entry, normalizer, learning rate and counter agrees.  The active
        model and RNG are snapshotted and restored, so verification has no side
        effect on the running instance.
        """
        record = self.save(reason)
        verification = self.verify_saved_checkpoint(
            record['path'], record['sha256'], allow_bridge=allow_bridge,
            allow_probe=allow_probe)
        record['verification'] = verification
        self._write_json(
            f'checkpoints/{reason}_{int(record["update"]):06d}.verification.json',
            verification)
        self._write_json(
            f'recovery/{reason}_{int(record["update"]):06d}.json',
            dict(reason=str(reason), path=record['path'], sha256=record['sha256'],
                 update=int(record['update']), phase=self.phase,
                 substage=self.substage,
                 pending_boundary_cleared=bool(self.pending_boundary is None),
                 resume_origin=deepcopy(self.resume_origin),
                 verification=verification))
        if str(reason).startswith('resume_loaded'):
            self.resume_loaded = record
        elif str(reason).startswith('resume_ready'):
            self.resume_ready = record
        return record

    def verify_saved_checkpoint(self, path, expected_sha, *, allow_bridge=False,
                                allow_probe=None):
        """Formal-recovery round trip of one saved file (no active-state change)."""
        if allow_probe is None:
            allow_probe = self.mode == 'probe'
        mode = 'probe' if allow_probe else 'resume'
        allow_bridge = bool(allow_bridge and not allow_probe)
        active_model_sha = model_module.state_sha256(self.model.state_dict())
        rng_before = recovery_module.capture_rng(self.env)
        checks = None
        state = None
        try:
            state = recovery_module.preload_checkpoint(
                path, expected_sha, mode=mode, allow_bridge=allow_bridge)
            bare = model_module.V06ActorCritic()
            recovery_module.load_model_state(bare, state['model'])
            optimizer = ppo_module.build_optimizer(bare)
            optimizer.load_state_dict(state['optimizer'])
            normalizer = state['normalizer']
            bare.actor.normalizer.load_state_dict(normalizer['actor'])
            bare.critic_normalizer.load_state_dict(normalizer['critic'])
            learning = dict(state['learning'])
            checkpoint_metadata = (state['initialization'].get('transfer_metadata'))
            checks = dict(
                model_tensors_match=_nested_equal(bare.state_dict(),
                                                  state['model']),
                active_model_matches_checkpoint=_nested_equal(
                    self.model.state_dict(), state['model']),
                optimizer_state_match=_nested_equal(optimizer.state_dict(),
                                                    state['optimizer']),
                normalizer_match=_nested_equal(
                    dict(actor=bare.actor.normalizer.state_dict(),
                         critic=bare.critic_normalizer.state_dict()),
                    normalizer),
                learning_rate_match=bool(
                    abs(float(optimizer.param_groups[0]['lr'])
                        - float(learning['learning_rate'])) <= 1e-15),
                optimizer_lr_matches_learning=bool(
                    float(state['optimizer']['param_groups'][0]['lr'])
                    == float(learning['learning_rate'])),
                counters=dict(update=int(learning['update']),
                              optimizer_step=int(learning['optimizer_step']),
                              adam_steps=int(learning['adam_steps']),
                              control_samples=int(learning['control_samples']),
                              physics_substeps=int(learning['physics_substeps'])),
                counter_update_matches_runner=(
                    int(learning['update']) == int(self.update)),
                metadata_present=bool(checkpoint_metadata),
                metadata_deepcopied=bool(
                    self.model.transfer_metadata is not checkpoint_metadata),
                metadata_match=_nested_equal(self.model.transfer_metadata or {},
                                             checkpoint_metadata or {}))
        finally:
            # Verification is a read-only audit: the active RNG is restored even
            # if the independent reload raises.
            recovery_module.restore_rng(rng_before, self.env)
        if checks is None or state is None:  # pragma: no cover - the finally re-raises
            raise recovery_module.RecoveryBlocked(
                'the post-save recovery verification produced no evidence')
        checks['active_model_unchanged'] = bool(
            model_module.state_sha256(self.model.state_dict()) == active_model_sha)
        checks['active_rng_unchanged'] = bool(_rng_equal(
            rng_before, recovery_module.capture_rng(self.env)))
        aligned = all((
            checks['model_tensors_match'],
            checks['active_model_matches_checkpoint'],
            checks['optimizer_state_match'],
            checks['normalizer_match'],
            checks['learning_rate_match'],
            checks['optimizer_lr_matches_learning'],
            checks['counter_update_matches_runner'],
            checks['metadata_present'],
            checks['metadata_deepcopied'],
            checks['metadata_match'],
            checks['active_model_unchanged'],
            checks['active_rng_unchanged']))
        verification = dict(
            checkpoint=str(path), sha256=str(expected_sha), mode=mode,
            allow_bridge=bool(allow_bridge),
            transfer_metadata_sha256=(
                checkpoint_metadata.get('sha256')
                if isinstance(checkpoint_metadata, dict) else None),
            resume_origin=deepcopy(state['initialization'].get('resume_origin')),
            learning=dict(learning), checks=checks, aligned=bool(aligned))
        if not aligned:
            raise recovery_module.RecoveryBlocked(
                'the saved checkpoint did not pass an independent formal-recovery '
                'reload; the run must stop before any evaluation or learning update',
                audit=verification)
        return verification

    def _check_resume_evidence(self, *, stage):
        """Refuse evaluation/learning until the Q006 evidence exists and verifies."""
        if stage == 'learning' and self.evaluation_transaction is not None:
            if (self._evaluation_transaction_durable and self.pending_boundary is None
                    and self.evaluation_transaction.get('stage')=='applied'):
                self._validate_evaluation_transaction()
                self.evaluation_transaction = None
                self._evaluation_transaction_durable = False
            else:
                raise recovery_module.RecoveryBlocked('Q010 current evaluation must be persisted and committed before new learning')
        if not self.require_resume_evidence:
            return True
        if self.resume_loaded is None:
            raise RunnerError(
                'Q006 requires an atomically saved and independently reloaded '
                'resume_loaded checkpoint before any evaluation or learning update')
        if stage == 'learning' and (self._resume_had_pending_boundary
                                    or self.source.get('continuation')) \
                and self.resume_ready is None:
            if self.pending_boundary is None:
                raise RunnerError(
                    'Q006 requires a verified resume_ready checkpoint for the completed '
                    'update-100 boundary before the first new learning update')
        return True

    def load(self, path, expected_sha=None, *, allow_probe=False, allow_bridge=False):
        """``--resume-run``: adopt a validated checkpoint and start a new episode.

        ``allow_probe=True`` is only used by the probe worker's own recovery check;
        the formal training resume keeps it False so ``v06-probe-state-v1`` (and
        every other engineering container) stays refused.
        """
        allow_probe = bool(allow_probe or self.mode == 'probe')
        # Q005: the restricted bridge flag must reach the preload validation too,
        # otherwise the documented cross-source checkpoint is refused here even
        # though the caller authorised it (integration fix, no other change).
        state = recovery_module.preload_checkpoint(
            path, expected_sha, mode='probe' if allow_probe else 'resume',
            allow_bridge=bool(allow_bridge))
        contract = recovery_module.resume_contract(
            state, init_sha256=model_module.INIT_SHA256,
            recipe=model_module.RECIPE_VERSION, seed=self.seed,
            num_envs=int(getattr(self.env, 'num_envs', ppo_module.NUM_ENVS)),
            allow_probe=allow_probe, allow_bridge=allow_bridge)
        # Q006 section 4.1: the original ETH initialization evidence must exist and
        # agree with the approved ETH source/schema *before* anything is adopted.
        # It is restored as a deep copy and never resets the fusion layers, the
        # trained weights, Adam, the normalizer, the RNG or the curriculum.
        if state['source'].get('continuation'):
            # The C1 parent chain survives normal second/third recoveries.
            self.source = deepcopy(state['source'])
        initialization = state.get('initialization')
        metadata = (initialization or {}).get('transfer_metadata') \
            if isinstance(initialization, dict) else None
        try:
            model_module.validate_transfer_metadata(metadata)
        except ValueError as error:
            raise recovery_module.RecoveryBlocked(
                f'checkpoint ETH transfer metadata is not admissible before new '
                f'learning: {error}',
                audit=dict(path=str(path), checkpoint_sha256=state.get(
                    'checkpoint_sha256'))) from None
        self.resume_chain = [dict(item) for item in
                             (initialization.get('resume_chain') or [])]
        bridge = contract.get('bridge') or {}
        self.resume_origin = dict(
            source_checkpoint=state.get('checkpoint_sha256'),
            source_path=str(path), source_update=int(state['learning']['update']),
            resume_mode=contract['resume_mode'],
            segment=(dict(bridge.get('segment')) if bridge.get('segment') else None),
            bridge=(dict(bridge) if bridge else None),
            note='ETH transfer_metadata is the original initialization evidence; this '
                 'block records only where the running segment was resumed from')
        self.resume_origin['loaded_index'] = len(self.resume_chain)
        migration = contract.get('pending_task_migration')
        if migration is not None:
            # Q007 section 4: replace ONLY the one validated unfinished flat task,
            # record the append-only migration manifest before any evaluation, and
            # keep the completed validation/diagnostic records untouched.
            recovery_module.apply_pending_task_migration(state, migration)
            self.pending_task_migration = deepcopy(migration)
            self._ensure_directories()
            self._write_json(
                f'recovery/task_migration_{int(state["learning"]["update"]):06d}.json',
                migration)
            self.resume_origin['pending_task_migration'] = dict(
                schema=migration.get('schema'),
                plan_sha256=migration.get('plan_sha256'),
                old_task_sha256=(migration.get('old_task') or {}).get(
                    'case_table_sha256'),
                old_tile=(migration.get('old_task') or {}).get('tile'),
                new_case_table_sha256=migration.get('new_case_table_sha256'),
                q007_decision_sha256=migration.get('q007_decision_sha256'),
                inherited_evaluation_calls=0,
                manifest=f'recovery/task_migration_'
                         f'{int(state["learning"]["update"]):06d}.json')
        else:
            self.pending_task_migration = None
        self.resume_chain = (self.resume_chain + [dict(self.resume_origin)])[-8:]
        recovery_module.load_model_state(self.model, state['model'])
        self.model.transfer_metadata = deepcopy(metadata)
        self.ppo.optimizer.load_state_dict(state['optimizer'])
        for group in self.ppo.optimizer.param_groups:
            group['lr'] = float(state['learning']['learning_rate'])
        normalizer = state['normalizer']
        self.model.actor.normalizer.load_state_dict(normalizer['actor'])
        self.model.critic_normalizer.load_state_dict(normalizer['critic'])
        self.curriculum.load_state_dict(state['curriculum'])
        self.cuda_rng_state = (deepcopy(state['rng'].get('cuda'))
                               if state['rng'].get('cuda') else None)
        c1_initialization_audit = None
        if self.source.get('continuation'):
            # C1 ruling: restore RNG -> original schedule once -> original full
            # new-episode reset once. Schedule may initialize changed slots too.
            # These read-only digests do not restore, reseed or isolate any flow.
            from .continuation_c1 import tree_digest as c1_tree_digest
            def c1_rng_snapshot():
                rng = recovery_module.capture_rng(self.env)
                values = {key: c1_tree_digest(value) for key, value in rng.items()}
                values['curriculum_sampler'] = c1_tree_digest(
                    self._curriculum_state().get('sampler_rng_state'))
                return values
            c1_initialization_audit = dict(
                source_checkpoint=state.get('checkpoint_sha256'),
                before_restore=c1_rng_snapshot(),
                checkpoint_rng={key: c1_tree_digest(value)
                                for key, value in state['rng'].items()},
                sequence=['restore_checkpoint_rng', 'original_schedule_once',
                          'original_ppo_reset_once'],
                extra_changed_slot_initialization_accepted=True,
                full_evaluation_rng_isolation='not_established_Q008',
                physical_trajectory_continuous=False)
        recovery_module.restore_rng(state['rng'], self.env)
        if c1_initialization_audit is not None:
            c1_initialization_audit['after_restore'] = c1_rng_snapshot()
        self.ppo.learning_state.update(
            completed_ppo_updates=int(state['learning']['update']),
            optimizer_steps=int(state['learning']['optimizer_step']),
            optimizer_initialized=True)
        self.update = int(state['learning']['update'])
        self.gates = [_gate_record(item) for item in state['gate'].get('records', [])]
        self.gate_streak = int(state['gate'].get('streak', 0))
        retention = state.get('retention') or {}
        streaks = dict(retention.get('streaks') or {})
        if not streaks and 'streak' in retention:  # pre-Q001 single-counter block
            streaks = {'flat': int(retention.get('streak', 0))}
        self.retention = dict(streaks={key: int(value) for key, value in streaks.items()},
                              last=dict(retention.get('last') or {}),
                              enabled=dict(retention.get('enabled', {'flat': False,
                                                                     'h00': False})),
                              history=[dict(item) for item in
                                       retention.get('history', [])])
        self.checkpoints = [dict(item) for item in
                            (state['ledger'].get('checkpoints') or [])]
        self.diagnostics = [dict(item) for item in
                            (state['ledger'].get('diagnostics') or [])]
        self.transitions = [dict(item) for item in
                            (state['ledger'].get('transitions') or [])]
        diagnostic = state['ledger'].get('diagnostic_collapse') or {}
        if diagnostic:
            self.diagnostic.update(dict(
                streak=int(diagnostic.get('streak', 0)),
                required=diagnostic.get('required'),
                generation=diagnostic.get('generation'),
                last=deepcopy(diagnostic.get('last')),
                history=[dict(item) for item in diagnostic.get('history', [])],
                cases=int(diagnostic.get('cases', 0)),
                stalled_cases=int(diagnostic.get('stalled_cases', 0)),
                first_step_absent_cases=int(
                    diagnostic.get('first_step_absent_cases', 0)),
                hits=int(diagnostic.get('hits', 0))))
        self.pending_transition = state.get('pending_transition')
        ledger = state['ledger']
        saved_stop = ledger.get('stop_state')
        if saved_stop is not None:
            raise recovery_module.RecoveryBlocked(
                f'checkpoint belongs to a terminated run ({saved_stop}); a stopped run '
                'is never resumed or reclassified')
        self.stop_state = saved_stop
        self.stop_reason = ledger.get('stop_reason')
        self.error_type = ledger.get('error_type')
        saved_lifecycle = ledger.get('lifecycle_state') or 'running'
        if saved_lifecycle not in LIFECYCLE_STATES:
            raise recovery_module.RecoveryBlocked(
                f'checkpoint lifecycle state {saved_lifecycle!r} is not a v0.6 state')
        self.lifecycle_state = saved_lifecycle
        self.curriculum_complete = saved_lifecycle == 'ready_for_final_test'
        self.pending_evaluations = [dict(item) for item in
                                    (ledger.get('pending_evaluations') or [])]
        self.pending_boundary = (deepcopy(ledger.get('pending_boundary'))
                                 if ledger.get('pending_boundary') is not None else None)
        self.incomplete_update = (deepcopy(ledger.get('incomplete_update'))
                                  if ledger.get('incomplete_update') is not None else None)
        if self.incomplete_update is not None:
            raise recovery_module.RecoveryBlocked(
                'checkpoint contains a partial PPO update; the billed interval is not '
                'resumable')
        self._sync_phase()
        self.final = deepcopy((ledger.get('terminal') or {}).get('final_test'))
        self.evaluation_transaction = deepcopy(ledger.get('evaluation_transaction'))
        if self.evaluation_transaction is not None:
            self._validate_evaluation_transaction()
            self._evaluation_transaction_durable = self.evaluation_transaction.get('stage')=='applied' and self.pending_boundary is None
            if self.evaluation_transaction.get('stage') == 'applied' and self.pending_boundary is not None:
                task = next(task for task in self.pending_boundary['tasks']
                            if task.get('gate_kind') == self.evaluation_transaction['identity']['kind']
                            and task.get('case_table_sha256') == self.evaluation_transaction['identity']['case_table_sha256'])
                if self._result_id(self.update,task) in self.pending_boundary['results']:
                    # This complete checkpoint itself proves result+pending commit.
                    self.evaluation_transaction = None
        # A restored run starts a NEW episode and never re-runs a billed update.
        if c1_initialization_audit is not None:
            self._reset_environment_schedule()
            c1_initialization_audit['after_schedule'] = c1_rng_snapshot()
        if hasattr(self.env, 'reset'):
            self.ppo.reset()
        if c1_initialization_audit is not None:
            c1_initialization_audit['after_final_reset'] = c1_rng_snapshot()
            c1_initialization_audit['phase'] = self.phase
            c1_initialization_audit['substage'] = self.substage
            c1_initialization_audit['phase_update'] = self.phase_update
            c1_initialization_audit['generation'] = self._curriculum_state().get('generation')
            self.c1_initialization_audit = c1_initialization_audit
            self._write_json(f'recovery/c1_initialization_{int(self.update):06d}.json',
                             c1_initialization_audit)
        self._resume_had_pending_boundary = self.pending_boundary is not None
        self._write_json('resume.json', dict(contract=contract,
                                             checkpoint=state.get('checkpoint_sha256'),
                                             new_episode=True,
                                             billing_resumed_at=self.update,
                                             transfer_metadata_sha256=metadata.get(
                                                 'sha256'),
                                             transfer_metadata_deepcopied=bool(
                                                 self.model.transfer_metadata is not
                                                 metadata),
                                             resume_origin=deepcopy(self.resume_origin),
                                             pending_boundary_requires_completion=bool(
                                                 self._resume_had_pending_boundary),
                                             pending_task_migration=(
                                                 None if self.pending_task_migration is
                                                 None else dict(
                                                     schema=self.pending_task_migration.get(
                                                         'schema'),
                                                     plan_sha256=self.pending_task_migration.get(
                                                         'plan_sha256'),
                                                     new_case_table_sha256=self
                                                     .pending_task_migration.get(
                                                         'new_case_table_sha256'))),
                                             require_resume_evidence=bool(
                                                 self.require_resume_evidence)))
        return contract

    # -------------------------------------------- Q010 current-result transaction
    def _evaluation_identity(self, table, update):
        return dict(update=int(update),phase=self.phase,substage=self.substage,
            generation=self._curriculum_state().get('generation'),kind=table.kind,
            case_table_sha256=table.sha256,round=table.round,
            model_sha256=model_module.state_sha256(self.model.state_dict()),
            optimizer_sha256=_digest(evaluation_module._optimizer_digest(self.ppo.optimizer.state_dict())),
            source_closure_sha256=self.source.get('source_closure_sha256'),
            contract_sha256=recovery_module.contract_sha(spec.contract()))

    def _validate_evaluation_transaction(self):
        transaction = self.evaluation_transaction
        if transaction is None:
            return None
        from . import evaluation_records as records_io
        if not isinstance(transaction,dict) or transaction.get('schema') != 'a2-v06-q010-current-evaluation-v1':
            raise recovery_module.RecoveryBlocked('unknown or missing Q010 evaluation transaction schema')
        if transaction.get('stage') not in ('evaluated','applied'):
            raise recovery_module.RecoveryBlocked('Q010 evaluation application_incomplete requires adjudication')
        try:
            payload = transaction['result']
            table = evaluation_module.table_from_json(payload['table'])
            expected = self._evaluation_identity(table,self.update)
            if transaction.get('identity') != expected:
                raise ValueError('current model/source/update/phase/generation/task binding differs')
            result = records_io.restore_result(payload,expected)
            if self.pending_boundary is not None:
                tasks = self.pending_boundary.get('tasks') or []
                matches = [task for task in tasks if task.get('gate_kind') == table.kind
                           and task.get('case_table_sha256') == table.sha256]
                if int(self.pending_boundary.get('update',-1)) != self.update or len(matches) != 1:
                    raise ValueError('cached result does not belong to the current pending task')
            if transaction['stage'] == 'applied':
                records_io.verify_sidecar(transaction.get('case_records'),payload)
                if table.kind == 'test':
                    matches = ([self.final] if self.final and self.final.get('update')==self.update
                               and self.final.get('evidence')==result.evidence_sha256
                               and self.final.get('case_records')==transaction.get('case_records') else [])
                elif table.kind == 'diagnostic':
                    matches = [row for row in self.diagnostics if int(row.get('update',-1))==self.update
                               and row.get('evidence',{}).get('records_sha256')==result.evidence_sha256['records_sha256']]
                else:
                    matches = [row for row in self.gates if row.update==self.update and row.kind==table.kind
                               and row.case_table_sha256==table.sha256
                               and (row.summary.get('case_records') or {}).get('records_sha256')==result.evidence_sha256['records_sha256']]
                if len(matches) != 1:
                    raise ValueError('applied cache lacks exactly one matching committed result identity')
                if table.kind in ('flat_retention','stair_retention'):
                    matches = [row for row in self.retention['history'] if int(row.get('update',-1))==self.update
                               and row.get('kind')==table.kind and row.get('case_table_sha256')==table.sha256]
                    if len(matches) != 1:
                        raise ValueError('applied retention cache lacks its matching per-set result')
            return result
        except (ValueError,KeyError,TypeError,OSError,evaluation_module.EvaluationError) as error:
            raise recovery_module.RecoveryBlocked('Q010 cached evaluation is invalid: '+str(error)) from error

    def _evaluate_with_record_transaction(self, kind, update, *, round=None, tile_id):
        from . import evaluation_records as records_io
        table = evaluation_module.make_cases(self.phase,self.substage,kind,self.seed,
                                             round=round,tile_id=tile_id)
        identity = self._evaluation_identity(table,update)
        if self.evaluation_transaction is not None:
            result = self._validate_evaluation_transaction()
            if self.evaluation_transaction['identity'] != identity:
                raise recovery_module.RecoveryBlocked('Q010 cannot replace an unfinished evaluation with another task')
            return result
        result = evaluation_module.evaluate_checkpoint(self.model,self.phase,self.substage,kind,self.seed,
            self.case_runner,optimizer=self.ppo.optimizer,counters=dict(update=int(update)),
            round=round,device=self.device,tile_id=tile_id)
        self.evaluation_transaction = dict(schema='a2-v06-q010-current-evaluation-v1',
            identity=identity,stage='evaluated',result=records_io.pack_result(result,identity),
            evaluation_checkpoint_sha256=(self.checkpoints[-1]['sha256'] if self.checkpoints else None))
        return result

    def _persist_evaluation_records(self, result):
        from . import evaluation_records as records_io
        identity = self.evaluation_transaction['identity']
        path = self.directory/'evaluation_records'/(
            f'{identity["kind"]}_{int(identity["update"]):06d}_{result.table.sha256[:12]}.json')
        binding = records_io.write_sidecar(path,self.evaluation_transaction['result'])
        self.evaluation_transaction['case_records'] = deepcopy(binding)
        return binding

    def _save_evaluation_commit(self, reason):
        """A name collision proves durability only for the same complete state."""
        try:
            return self.save(reason)
        except FileExistsError:
            path = self.directory/'checkpoints'/f'{reason}_{int(self.update):06d}.pt'
            existing = recovery_module.preload_checkpoint(path,mode=(
                'probe' if self.container==recovery_module.PROBE_CONTAINER else 'train'))
            expected = self.checkpoint_state()
            for key in ('identity','contract_sha256','model','optimizer','normalizer',
                        'learning','curriculum','evaluation','initialization','gate',
                        'retention','source','recipe','pending_transition'):
                if not _nested_equal(existing.get(key),expected.get(key)):
                    raise recovery_module.RecoveryBlocked(
                        'existing Q010 checkpoint conflicts with current '+key)
            for key in ('diagnostics','diagnostic_collapse','pending_boundary',
                        'incomplete_update','evaluation_transaction','stop_state',
                        'stop_reason','error_type','lifecycle_state','terminal',
                        'retention_schedule'):
                if not _nested_equal(existing['ledger'].get(key),expected['ledger'].get(key)):
                    raise recovery_module.RecoveryBlocked(
                        'existing Q010 checkpoint conflicts with current ledger.'+key)
            return dict(path=str(path),sha256=existing['checkpoint_sha256'],
                        update=self.update,reason=reason)

    def _finish_direct_evaluation_transaction(self):
        if self.pending_boundary is None:
            # Clear only after a full checkpoint proves the applied result.
            identity = self.evaluation_transaction['identity']
            reason = f'evaluation_{identity["kind"]}_{len(self.gates)}_{len(self.diagnostics)}'
            self._save_evaluation_commit(reason)
            self.evaluation_transaction = None
            self._evaluation_transaction_durable = False

    # ------------------------------------------------------------------ gates
    def run_gate(self, kind='validation', *, round=_ROUND_FROM_UPDATE,
                 retention_kind=None, update=None, tile_id=None):
        self._check_resume_evidence(stage='evaluation')
        if self.case_runner is None:
            raise RunnerError('no evaluation case runner is wired; the gate needs the '
                              'V06Env-backed deterministic-mean runner')
        record_update = int(self.update if update is None else update)
        case_round = (record_update if round is _ROUND_FROM_UPDATE else round)
        if kind in ('flat_retention', 'stair_retention'):
            # Q007: a retention set keeps its frozen terrain in every training phase.
            tile = retention_target_tile(kind)
            if tile_id is not None and int(tile_id) != int(tile):
                raise RunnerError(
                    f'{kind} must measure the frozen retention tile {tile}, not '
                    f'{int(tile_id)}')
        elif kind in ('test',):
            tile = int(spec.FINAL_TILE_ID)
        else:
            tile = self._current_tile(tile_id)
        result = self._evaluate_with_record_transaction(kind,record_update,round=case_round,tile_id=tile)
        binding = self._persist_evaluation_records(result)
        if self.evaluation_transaction['stage'] == 'applied':
            record = next(item for item in self.gates if item.update==record_update and item.kind==kind
                          and item.case_table_sha256==result.table.sha256)
            record.summary['case_records']=deepcopy(binding)
            if kind in ('flat_retention','stair_retention'):
                for row in self.retention['history']:
                    if row.get('kind')==kind and row.get('update')==record_update and row.get('case_table_sha256')==result.table.sha256:
                        row['case_records']=deepcopy(binding)
                short_kind=retention_kind or _retention_short_kind(kind)
                if self.retention['last'].get(short_kind,{}).get('update')==record_update:
                    self.retention['last'][short_kind]['case_records']=deepcopy(binding)
                self._write_json(f'retention/{short_kind}_{record_update:06d}.json',self.retention['last'][short_kind])
        else:
            record = GateRecord(phase=self.phase,substage=self.substage,kind=kind,update=record_update,
                passed=bool(result.gate['gate_pass']),case_table_sha256=result.table.sha256,
                checkpoint_sha256=self.evaluation_transaction.get('evaluation_checkpoint_sha256'),
                failed_cases=int(result.gate['failed_cases']),summary=dict(result.gate,case_records=binding))
        self._write_json(f'gates/{kind}_{record_update:06d}.json',dict(gate=record.as_dict(),
            summary=result.summary,evidence=result.evidence_sha256,case_records=binding))
        if self.evaluation_transaction['stage'] != 'applied':
            self.evaluation_transaction['stage'] = 'applying'
            try:
                self.gates.append(record)
                if result.gate['gate_eligible']:
                    self._apply_gate_streak(record)
                if kind in ('stair_retention','flat_retention'):
                    self._apply_retention(record,result,retention_kind or _retention_short_kind(kind),update=record_update)
            except Exception as error:
                self.evaluation_transaction.update(stage='application_incomplete',application_error=repr(error))
                raise
            self.evaluation_transaction['stage'] = 'applied'
        if self.telemetry is not None and hasattr(self.telemetry,'gate'):
            self.telemetry.gate(record.as_dict())
        self._finish_direct_evaluation_transaction()
        return result

    def _apply_gate_streak(self, record):
        rule = PHASE_GATE_RULES.get(record.phase, PHASE_GATE_RULES['R'])
        phase_update = self.phase_update
        eligible = (record.passed and phase_update >= rule['min_updates'])
        self.gate_streak = (self.gate_streak + 1) if eligible else 0
        if self.gate_streak >= rule['required']:
            self.pending_transition = dict(kind='gate_passed', phase=self.phase,
                                           substage=self.substage,
                                           update=self.update,
                                           gate_streak=self.gate_streak,
                                           required=rule['required'],
                                           case_table_sha256=record.case_table_sha256)
        self._write_json('gate_streak.json', dict(
            update=self.update, phase=self.phase, substage=self.substage,
            streak=int(self.gate_streak), required=rule['required'],
            min_updates=rule['min_updates'],
            minimum_updates=rule.get('minimum_updates', rule['min_updates']),
            pending=deepcopy(self.pending_transition)))

    def _apply_retention(self, record, result, kind, *, update=None):
        """Record one retention set; each kind keeps its own streak (Q001-7)."""
        failed = int(result.gate['failed_cases'])
        decision = self._feed_retention(kind, failed, record.passed,
                                        update=int(record.update if update is None
                                                   else update))
        if decision is None:
            # Fallback only when the curriculum has no retention hook (never in
            # production): keep the same independent per-kind counter.
            previous = int(self.retention['streaks'].get(kind, 0))
            decision = speed.retention_decision(previous, failed)
        streak = int(decision.get('streak', 0))
        self.retention['streaks'][kind] = streak
        entry = dict(passed=bool(record.passed), failed_cases=failed,
                     cases=int(speed.RETENTION_CASES),
                     case_records=deepcopy(record.summary.get('case_records')),
                     case_table_sha256=record.case_table_sha256,
                     update=int(record.update), kind=record.kind,
                     retention_kind=str(kind),
                     qualified_at=decision.get('qualified_at'),
                     next_due=decision.get('next_due'),
                     last_completed=decision.get('last_completed', record.update),
                     duplicate=bool(decision.get('duplicate', False)),
                     streak=streak, required=int(speed.RETENTION_CONSECUTIVE),
                     stop=bool(decision.get('stop', False)))
        self.retention['last'][str(kind)] = entry
        self.retention['history'].append(entry)
        self._write_json(f'retention/{kind}_{int(record.update):06d}.json', entry)
        if self.telemetry is not None and hasattr(self.telemetry, 'retention'):
            self.telemetry.retention(entry)
        if entry['stop']:
            self._stop('behavior_regression', dict(
                kind=record.kind, retention_kind=str(kind), failed_cases=failed,
                cases=int(speed.RETENTION_CASES), streak=streak,
                update=self.update, case_table_sha256=record.case_table_sha256))

    # -------------------------------------------------------- update boundary
    def _diagnostic_due(self):
        return (self.update > 0
                and self.update % int(speed.DIAGNOSTIC_INTERVAL) == 0)

    def _retention_due(self):
        state = self._curriculum_state()
        hook = getattr(self.curriculum, 'retention_due', None)
        due = None
        if callable(hook):
            try:
                due = hook(self.update)
            except TypeError:
                due = hook(global_update=self.update)
        if due is not None:
            if isinstance(due, str):
                return [due]
            kinds = [str(item) for item in due]
            if not kinds:
                return None
            return kinds
        if self.update <= 0 or self.update % int(speed.RETENTION_INTERVAL):
            return None
        kinds = [kind for kind in ('flat', 'h00')
                 if self.retention['enabled'].get(kind)]
        return kinds or None

    def boundary(self):
        """Complete update boundary (Q002-1/5 ordering).

        1. save the completed update first when the save cadence is due;
        2. evaluate the due gate/diagnostic/retention sets for this update
           (the gate sees the phase that just trained; the diagnostic gets the
           post-tick generation accounting);
        3. call ``on_update(update, gates=[gate])`` so the curriculum ingests the
           gate with its post-tick accounting, commits a legal transition and
           only then judges the phase budget;
        4. apply the environment schedule, persist transition evidence;
        5. terminal/budget classification last.
        """
        actions = []
        before = self._curriculum_state()
        delta = max(int(self.update) - int(before.get('last_update', self.update)), 0)
        # (1) due evaluations first: the gate must be evaluated for the update
        # that just trained and handed to the same on_update tick (Q002-1).
        gate = None
        if self.evaluate and self.update > 0 and self.update % int(speed.GATE_INTERVAL) == 0:
            self.run_gate('validation')
            gate = self.gates[-1].as_dict()
            actions.append('gate')
        if self.evaluate and self._diagnostic_due():
            self._diagnostic(
                generation_updates=int(before.get('generation_updates', 0)) + delta,
                generation=before.get('generation'))
            actions.append('diagnostic')
        for kind in (self._retention_due() or []) if self.evaluate else []:
            gate_kind = _retention_gate_kind(self.curriculum, kind)
            entry = dict(kind='retention', retention_kind=str(kind),
                         gate_kind=gate_kind, update=int(self.update),
                         phase=self.phase, substage=self.substage)
            self._begin_evaluation(entry)
            self.run_gate(gate_kind, retention_kind=kind)
            self._finish_evaluation(entry)
            actions.append(gate_kind)
        # (3) tick + gate ingestion + legal transition + budget judgment.
        transitions = self._on_update_hook(self.update, gate)
        self._sync_phase()
        # A new generation clears the inherited stall-collapse streak (Q001-7).
        if (before.get('generation') is not None
                and int(before['generation']) != int(
                    self._curriculum_state().get('generation', before['generation']))):
            self.diagnostic['streak'] = 0
            self.diagnostic['generation'] = self._curriculum_state().get('generation')
        changed = (str(before.get('phase')).upper() != self.phase
                   or before.get('substage') != self.substage)
        if transitions or changed:
            for event in transitions or []:
                item = event.as_dict() if hasattr(event, 'as_dict') else dict(event)
                item['update'] = self.update
                self.transitions.append(item)
                self._consume_terminal_event(item)
            actions.append('curriculum_transition')
            self._write_json(f'transitions/{self.update:06d}.json',
                             dict(before=before, after=self._curriculum_state(),
                                  events=[dict(item) for item in self.transitions[-5:]],
                                  phase=self.phase, substage=self.substage))
        # (4) persist the completed update and its transition evidence.
        if self.autosave and self.update > 0 \
                and self.update % int(speed.SAVE_INTERVAL) == 0:
            self.save('interval')
            actions.append('save')
        if gate is not None:
            self.save('gate')
        if (transitions or changed) and self.stop_state is None:
            self.save('transition')
        self.pending_transition = None
        # (5) environment schedule for the next rollout.
        self._reset_environment_schedule()
        # (6) terminal/budget classification last.
        self._check_stop_conditions()
        return dict(update=self.update, phase=self.phase, substage=self.substage,
                    actions=actions, stop_state=self.stop_state,
                    lifecycle_state=self.lifecycle_state)

    # ---------------------------------------------- pending evaluation txn
    def _begin_evaluation(self, entry):
        self.pending_evaluations.append(dict(entry))
        self._write_json('pending_evaluations.json',
                         dict(pending=[dict(item) for item in self.pending_evaluations]))
        return entry

    def _finish_evaluation(self, entry):
        self.pending_evaluations = [item for item in self.pending_evaluations
                                    if item != entry]
        self._write_json('pending_evaluations.json',
                         dict(pending=[dict(item) for item in self.pending_evaluations]))

    def _complete_pending_evaluations(self):
        """Finish a retention check that a checkpoint interrupted (Q002-6).

        A resumed run completes the pending evaluation once; the curriculum
        ``record_retention`` is idempotent per completion update, so the same
        result can never be counted twice.
        """
        completed = []
        for entry in [dict(item) for item in self.pending_evaluations]:
            kind = entry.get('retention_kind')
            if entry.get('kind') != 'retention' or kind is None:
                self._finish_evaluation(entry)
                continue
            last = self.retention['last'].get(str(kind), {})
            if int(last.get('update', -1)) == int(entry['update']) \
                    and not last.get('duplicate'):
                self._finish_evaluation(entry)
                continue
            gate_kind = entry.get('gate_kind') or _retention_gate_kind(self.curriculum, kind)
            self.run_gate(gate_kind, retention_kind=kind, update=int(entry['update']))
            self._finish_evaluation(entry)
            completed.append(entry)
        return completed

    def _consume_terminal_event(self, event):
        """React to a curriculum ``kind='final'`` event (Q001 / W0 4.4).

        ``complete`` (R passed twice) stops training **immediately** and enters the
        frozen test instead of running to 30,000 updates; the budget and stall
        statuses are mapped to the SDS terminal states.
        """
        if str(event.get('kind')) != 'final':
            return None
        status = event.get('status')
        if status == 'complete':
            # Q002-3: passing R twice is only "ready for the final test".
            self.curriculum_complete = True
            self._mark_lifecycle('ready_for_final_test', reason='final_gates_passed',
                                 evidence=dict(event=event))
            return 'ready_for_final_test'
        if status == 'minimum_level_failure':
            return self._stop('curriculum_stalled', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                curriculum_status=status, event=event, budget=self._budget_layers(),
                reason='minimum_level_failure'))
        if status == 'budget_exhausted':
            layers = self._budget_layers()
            if layers['total_exhausted']:
                return self._stop('budget_exhausted', dict(
                    update=self.update, phase=self.phase, substage=self.substage,
                    curriculum_status=status, event=event, budget=layers,
                    reason='total_budget_exhausted'),
                    reason='total_budget_exhausted')
            reason = (SUBSTAGE_BUDGET_REASON if layers['substage_exhausted']
                      else PHASE_BUDGET_REASON)
            return self._stop('gate_failed', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                curriculum_status=status, event=event, budget=layers, reason=reason),
                reason=reason)
        return None

    def _reset_environment_schedule(self):
        if self.env is None:
            return False
        hook = getattr(self.env, 'set_schedule', None)
        if not callable(hook):
            return False
        schedule = self.curriculum.schedule()
        iteration = int(self.phase_update)
        try:
            hook(schedule, iteration)
        except TypeError:
            hook(schedule)
        return True

    def _diagnostic(self, *, generation_updates=None, generation=None):
        """15-case diagnostic with a durable same-result Q010 transaction."""
        self._check_resume_evidence(stage='evaluation')
        if self.case_runner is None:
            return None
        result = self._evaluate_with_record_transaction('diagnostic',self.update,tile_id=self._current_tile())
        binding = self._persist_evaluation_records(result)
        if self.evaluation_transaction['stage'] == 'applied':
            record = next(row for row in self.diagnostics if int(row.get('update',-1))==self.update
                          and row.get('evidence',{}).get('records_sha256')==result.evidence_sha256['records_sha256'])
            record['case_records']=deepcopy(binding)
            self._write_json(f'diagnostics/diagnostic_{self.update:06d}.json',record)
            self._finish_direct_evaluation_transaction()
            return record
        state = self._curriculum_state()
        generation = state.get('generation') if generation is None else generation
        generation_updates = (int(state.get('generation_updates',0) or 0)
                              if generation_updates is None else int(generation_updates))
        decision = evaluation_module.diagnostic_decision(result.records,self.phase,
            generation_updates=generation_updates,generation=generation)
        decision['update'] = self.update
        collapse = evaluation_module.diagnostic_collapse_state(self.diagnostic,decision)
        changed = deepcopy(self.diagnostic)
        changed.update(dict(streak=int(collapse['streak']),required=int(collapse['required_streak']),
            generation=generation,last=decision,cases=int(len(decision['cases'])),
            stalled_cases=int(decision['stalled_cases']),first_step_absent_cases=int(decision['first_step_absent_cases']),
            hits=int(self.diagnostic.get('hits',0))+int(bool(decision['collapse_hit']))))
        changed['history'].append(dict(update=self.update,phase=self.phase,generation=generation,
            eligible=decision['eligible'],collapse_hit=decision['collapse_hit'],streak=int(collapse['streak']),
            required=int(collapse['required_streak']),stalled_cases=int(decision['stalled_cases']),
            first_step_absent_cases=int(decision['first_step_absent_cases'])))
        changed['history'] = changed['history'][-40:]
        record = dict(update=self.update,phase=self.phase,substage=self.substage,
            gate=evaluation_module.gate_result(result.summary),diagnostic=decision,collapse=collapse,
            evidence=result.evidence_sha256,case_records=binding)
        # No counter changes precede successful raw evidence + main binding writes.
        self._write_json(f'diagnostics/diagnostic_{self.update:06d}.json',record)
        self.evaluation_transaction['stage'] = 'applying'
        try:
            self.diagnostic = changed
            self.diagnostics.append(record)
            if self.telemetry is not None and hasattr(self.telemetry,'retention'):
                self.telemetry.retention(dict(kind='diagnostic_collapse',update=self.update,
                    streak=int(collapse['streak']),required=int(collapse['required_streak']),
                    eligible=bool(decision['eligible']),collapse_hit=bool(decision['collapse_hit']),
                    stalled_cases=int(decision['stalled_cases']),first_step_absent_cases=int(decision['first_step_absent_cases'])))
            if collapse['collapse']:
                self._stop('behavior_collapse',dict(kind='diagnostic_collapse',update=self.update,phase=self.phase,
                    generation=generation,streak=int(collapse['streak']),required=int(collapse['required_streak']),
                    stalled_cases=int(decision['stalled_cases']),first_step_absent_cases=int(decision['first_step_absent_cases']),
                    evidence_sha256=result.table.sha256))
        except Exception as error:
            self.evaluation_transaction.update(stage='application_incomplete',application_error=repr(error))
            raise
        self.evaluation_transaction['stage'] = 'applied'
        self._finish_direct_evaluation_transaction()
        return record

    def _check_stop_conditions(self):
        """Terminal classification at a complete update boundary (Q002-5).

        Order: behavioural/curriculum stops first (never relabelled by budget),
        then the *total* budget, then a phase/substage budget that has no legal
        advance.  A single failed gate with budget left is never a stop.
        """
        if self.stop_state is not None or self.lifecycle_state == 'complete':
            return
        curriculum = self._curriculum_state()
        finished = bool(curriculum.get('finished', False))
        status = curriculum.get('stop_status')
        layers = self._budget_layers()
        if finished and status == 'complete':
            # R passed twice: qualification only, never training success.
            self.curriculum_complete = True
            self._mark_lifecycle('ready_for_final_test', reason='final_gates_passed')
            return
        if finished and status == 'minimum_level_failure':
            self._stop('curriculum_stalled', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                curriculum_status=status,
                minimum_level_failure_streak=curriculum.get(
                    'minimum_level_failure_streak'),
                budget=layers, reason='minimum_level_failure'),
                reason='minimum_level_failure')
            return
        if self._monitoring_blocked():
            self._stop('monitoring_blocked', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                reason='monitoring_delivery_unconfirmed', budget=layers),
                reason='monitoring_delivery_unconfirmed')
            return
        user_requested = bool(self.stop_flag() if callable(self.stop_flag) else self.stop_flag)
        resource_available = True
        if callable(self.resource_guard):
            snapshot = self.resource_guard()
            if isinstance(snapshot, dict) and snapshot.get('ok') is False:
                resource_available = False
                self.stop_evidence = dict(resource=snapshot)
        if user_requested:
            self._stop('user_requested_stop', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                budget=layers), reason='user_requested_stop')
            return
        if not resource_available:
            self._stop('resource_insufficient', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                budget=layers, resource=self.stop_evidence.get('resource')),
                reason='resource_insufficient')
            return
        # Budget judgment (last).  The curriculum only reports ``finished`` when
        # there is no legal advance, so its budget status is authoritative.
        if finished and status == 'budget_exhausted':
            if layers['total_exhausted']:
                self._stop('budget_exhausted', dict(
                    update=self.update, phase=self.phase, substage=self.substage,
                    budget=layers, curriculum_status=status,
                    reason='total_budget_exhausted',
                    phase_budget_exhausted=layers['phase_exhausted'],
                    substage_budget_exhausted=layers['substage_exhausted']),
                    reason='total_budget_exhausted')
                return
            reason = (SUBSTAGE_BUDGET_REASON if layers['substage_exhausted']
                      else PHASE_BUDGET_REASON)
            self._stop('gate_failed', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                budget=layers, curriculum_status=status, reason=reason), reason=reason)
            return
        if layers['total_exhausted']:
            self._stop('budget_exhausted', dict(
                update=self.update, phase=self.phase, substage=self.substage,
                budget=layers, reason='total_budget_exhausted',
                phase_budget_exhausted=layers['phase_exhausted'],
                substage_budget_exhausted=layers['substage_exhausted'],
                curriculum_status=status), reason='total_budget_exhausted')
            return
        if layers['substage_exhausted'] or layers['phase_exhausted']:
            # Only stop when the curriculum reports it AND there is no latched
            # legal advance (a legal final-update advance is committed first).
            if finished or status == 'budget_exhausted':
                reason = (SUBSTAGE_BUDGET_REASON if layers['substage_exhausted']
                          else PHASE_BUDGET_REASON)
                self._stop('gate_failed', dict(
                    update=self.update, phase=self.phase, substage=self.substage,
                    budget=layers, curriculum_status=status, reason=reason),
                    reason=reason)

    def _monitoring_blocked(self):
        telemetry = self.telemetry
        if telemetry is None:
            return False
        state = getattr(telemetry, 'state', None)
        if isinstance(state, dict) and state.get('monitoring_blocked'):
            return True
        return bool(getattr(telemetry, 'monitoring_blocked', False))

    def _budget_layers(self):
        state = self._curriculum_state()
        phase_spent = int((state.get('spent_by_phase') or {}).get(self.phase, 0))
        substage_spent = int((state.get('spent_by_substage') or {}).get(
            self.substage, 0)) if self.substage else 0
        substage_budget = None
        if self.substage is not None:
            try:
                entry = spec.phase_spec(self.phase)
                budgets = dict(zip(tuple(entry.substages),
                                   tuple(entry.substage_budgets)))
                substage_budget = budgets.get(self.substage)
            except (ValueError, KeyError):
                substage_budget = None
        return dict(
            update=int(self.update), total_budget=int(speed.total_budget()),
            total_spent=int(self.update),
            total_exhausted=bool(self.update >= speed.total_budget()),
            total_remaining=int(speed.total_budget() - self.update),
            phase=self.phase, phase_budget=(speed.phase_budget(self.phase)
                                            if self.phase in speed.phase_order() else None),
            phase_spent=phase_spent,
            phase_exhausted=bool(self.phase in speed.phase_order()
                                 and phase_spent >= speed.phase_budget(self.phase)),
            substage=self.substage, substage_budget=substage_budget,
            substage_spent=substage_spent,
            substage_exhausted=bool(substage_budget is not None
                                    and substage_spent >= substage_budget))

    def _terminal_record(self):
        """Terminal evidence block (Q002-4): same source as spec states."""
        state = self._curriculum_state()
        return dict(
            state=(self.lifecycle_state if self.lifecycle_state == 'complete'
                   else (self.stop_state if self.stop_state is not None
                         else (self.lifecycle_state if self.lifecycle_state != 'running'
                               else None))),
            lifecycle_state=self.lifecycle_state,
            stop_state=self.stop_state,
            reason=self.stop_reason,
            error_type=self.error_type,
            failure_class=classify_failure(self.stop_state),
            global_update=int(self.update),
            phase=self.phase, substage=self.substage,
            generation=state.get('generation'),
            budget=self._budget_layers(),
            evidence=deepcopy(self.stop_evidence),
            final_test=deepcopy(self.final),
        )

    def _mark_lifecycle(self, state, *, reason=None, evidence=None):
        if state not in LIFECYCLE_STATES:
            raise RunnerError(f'unknown lifecycle state {state!r}')
        self.lifecycle_state = state
        if reason is not None:
            self.stop_reason = reason
        self._write_json('terminal.json', self._terminal_record())
        return state

    def _stop(self, state, evidence, *, reason=None):
        if self.stop_state is not None:
            return self.stop_state
        if state not in STOP_STATES:
            raise RunnerError(f'unknown stop state {state!r}')
        self.stop_state = state
        if self.lifecycle_state != 'complete':
            self.lifecycle_state = 'running'
        self.stop_reason = reason or state
        self.stop_evidence = dict(evidence, stop_state=state, reason=self.stop_reason,
                                 failure_class=classify_failure(state),
                                 error_type=self.error_type,
                                 update=self.update, phase=self.phase,
                                 substage=self.substage,
                                 timestamp=float(self.clock()))
        self._write_json('stop.json', self.stop_evidence)
        self._write_json('terminal.json', self._terminal_record())
        try:
            self.save('stop')
        except FileNotFoundError:  # directory already contains this checkpoint
            pass
        return state

    # -------------------------------------------------------------- training
    # ------------------------------------------------ two-phase boundary
    def _rollback_one_tick(self, update):
        """Put the committed update back for the curriculum's own decision pass."""
        state = deepcopy(self._curriculum_state())
        phase = str(state['phase']).upper()
        if int(state.get('last_update', 0)) != int(update):
            raise RunnerError('rollback requires the committed last_update')
        state['spent_by_phase'][phase] = int(state['spent_by_phase'].get(phase, 0)) - 1
        if state.get('substage'):
            state['spent_by_substage'][str(state['substage'])] = int(
                state['spent_by_substage'].get(str(state['substage']), 0)) - 1
        state['generation_updates'] = int(state.get('generation_updates', 0)) - 1
        state['last_update'] = int(update) - 1
        self.curriculum.load_state_dict(state)
        return state

    def commit_learning_update(self, update):
        """Phase A: bill the completed PPO update to the OLD phase only.

        No window consumption, no advance/regress, no budget judgment.  Repeated
        commits for the same update are idempotent; going backwards is refused.
        """
        update = int(update)
        state = self._curriculum_state()
        last = int(state.get('last_update', 0))
        if last == update:
            return dict(update=update, committed=False, reason='already_committed')
        if last > update:
            raise RunnerError(f'cannot commit update {update} after {last}')
        delta = update - last
        if delta != 1:
            raise RunnerError(f'commit must advance exactly one update, got {delta}')
        phase = str(state['phase']).upper()
        new = deepcopy(state)
        new['spent_by_phase'][phase] = int(new['spent_by_phase'].get(phase, 0)) + delta
        if new.get('substage'):
            name = str(new['substage'])
            new['spent_by_substage'][name] = int(
                new['spent_by_substage'].get(name, 0)) + delta
        new['generation_updates'] = int(new.get('generation_updates', 0)) + delta
        new['last_update'] = update
        self.curriculum.load_state_dict(new)
        record = dict(update=update, committed=True, phase=phase,
                      substage=new.get('substage'),
                      generation=new.get('generation'),
                      spent={str(key): int(value)
                             for key, value in new['spent_by_phase'].items()})
        self._write_json(f'commit_{update:06d}.json', record)
        return record

    def _due_tasks(self, update):
        """Due evaluations of this boundary, with their frozen case identities."""
        tasks = []
        if update > 0 and update % int(speed.GATE_INTERVAL) == 0:
            tile = self._current_tile()
            table = evaluation_module.make_cases(self.phase, self.substage,
                                                 'validation', self.seed, tile_id=tile)
            tasks.append(dict(kind='validation', gate_kind='validation',
                              retention_kind=None, tile=tile,
                              phase=self.phase, substage=self.substage,
                              case_table_sha256=table.sha256))
        if self._diagnostic_due():
            tile = self._current_tile()
            table = evaluation_module.make_cases(self.phase, self.substage,
                                                 'diagnostic', self.seed, tile_id=tile)
            tasks.append(dict(kind='diagnostic', gate_kind='diagnostic',
                              retention_kind=None, tile=tile,
                              phase=self.phase, substage=self.substage,
                              case_table_sha256=table.sha256))
        for kind in self._retention_due() or []:
            gate_kind = _retention_gate_kind(self.curriculum, kind)
            # Q007: the retention target is frozen (flat=0, H00=1) and never the
            # current training-phase tile.
            tile = retention_target_tile(gate_kind)
            table = evaluation_module.make_cases(self.phase, self.substage,
                                                 gate_kind, self.seed, tile_id=tile)
            tasks.append(dict(kind='retention', gate_kind=gate_kind,
                              retention_kind=str(kind), tile=tile,
                              phase=self.phase, substage=self.substage,
                              case_table_sha256=table.sha256))
        return tasks

    def _result_id(self, update, task):
        return ':'.join((str(update), str(task['kind']),
                         str(task.get('retention_kind')), self.phase,
                         str(task['case_table_sha256'])))

    def _run_pending_tasks(self, pending):
        """Execute the outstanding evaluations once, deduplicated by result id."""
        self.pending_boundary = pending
        for task in pending['tasks']:
            result_id = self._result_id(pending['update'], task)
            if result_id in pending['results']:
                continue
            # Q006 section 5 / Q007: the replay must use the same frozen case table
            # the original boundary committed, never a splice of another round.
            recorded = task.get('case_table_sha256')
            target_kind = (task['gate_kind'] if task['kind'] == 'retention'
                           else task['kind'])
            tile = int(task['tile'])
            if task['kind'] == 'retention':
                # Q007 section 4: an old pending retention task may be replaced ONLY
                # by the validated Q007 migration applied in ``load`` (it already
                # carries the frozen tile/new table SHA).  Anything else stays a
                # hard refusal; there is no generic auto-correction.
                frozen = retention_target_tile(task['gate_kind'])
                if tile != frozen:
                    raise recovery_module.RecoveryBlocked(
                        'the pending retention task targets a non-frozen tile; only '
                        'the restricted Q007 exception_000400 task migration may '
                        'replace it')
                if recorded is not None:
                    table = evaluation_module.make_cases(
                        self.phase, self.substage, target_kind, self.seed,
                        tile_id=tile)
                    if table.sha256 != recorded:
                        raise recovery_module.RecoveryBlocked(
                            'the pending retention case table identity does not match '
                            'the frozen replay table; refusing to splice another round')
            if recorded is not None and task['kind'] != 'retention':
                table = evaluation_module.make_cases(
                    self.phase, self.substage, target_kind, self.seed, tile_id=tile)
                if table.sha256 != recorded:
                    raise recovery_module.RecoveryBlocked(
                        'the pending boundary case table identity does not match the '
                        'replayed evaluation table; refusing to splice another round')
            kind = task['gate_kind']
            if task['kind'] == 'diagnostic':
                self._diagnostic(
                    generation_updates=int(self._curriculum_state().get(
                        'generation_updates', 0)),
                    generation=self._curriculum_state().get('generation'))
            else:
                # Q006/Q007: replay the recorded table identity, whose pending
                # tasks were built with round=None (the pending task SHA is the
                # frozen Q005 value), not with the update-number round metadata.
                self.run_gate(kind, retention_kind=task.get('retention_kind'),
                              update=int(pending['update']), tile_id=int(task['tile']),
                              round=None)
            pending['results'].append(result_id)
            self._write_json('pending_boundary.json', pending)
            if self.save_pending_results or self.evaluation_transaction is not None:
                # Keep the single actual result until a complete checkpoint also
                # contains its applied counters and the pending result identity.
                reason = f'pending_{len(pending["results"])}'
                self._save_evaluation_commit(reason)
                self.evaluation_transaction = None
        return pending

    def _finalize_curriculum(self, pending, gate):
        """Phase B: one curriculum decision pass over the billed old phase."""
        if self.evaluation_transaction is not None:
            raise recovery_module.RecoveryBlocked(
                'Q010 evaluation must be durably committed before curriculum transition')
        if pending.get('curriculum_decided'):
            return []
        self._rollback_one_tick(int(pending['update']))
        transitions = self._on_update_hook(int(pending['update']),
                                           dict(gate) if gate else None)
        pending['curriculum_decided'] = True
        self._write_json('pending_boundary.json', pending)
        return transitions

    def finalize_boundary(self, update, records=None, metrics=None):
        """Phase B: evaluate the due tasks, then make exactly one curriculum pass."""
        update = int(update)
        state_before = self._curriculum_state()
        tasks = self._due_tasks(update)
        pending = dict(update=update,
                       phase=str(state_before['phase']).upper(),
                       substage=state_before.get('substage'),
                       generation=state_before.get('generation'),
                       tasks=tasks, results=[], curriculum_decided=False,
                       committed=True)
        actions = []
        gate = None
        if tasks:
            self.pending_boundary = pending
            # Persist the billed-but-undecided boundary BEFORE any evaluation.
            self.save('boundary')
            actions.append('boundary_save')
            self._run_pending_tasks(pending)
            for task in tasks:
                if task['kind'] == 'validation':
                    gate = next((record.as_dict() for record in self.gates
                                 if record.kind == 'validation'
                                 and record.update == update), None)
                actions.append(task['kind'] if task['kind'] != 'retention'
                               else task['gate_kind'])
        transitions = self._finalize_curriculum(pending, gate)
        self._sync_phase()
        self.pending_boundary = None
        changed = (str(state_before['phase']).upper() != self.phase
                   or state_before.get('substage') != self.substage)
        if transitions or changed:
            for event in transitions or []:
                item = event.as_dict() if hasattr(event, 'as_dict') else dict(event)
                item['update'] = update
                self.transitions.append(item)
                self._consume_terminal_event(item)
            actions.append('curriculum_transition')
            self._write_json(f'transitions/{update:06d}.json',
                             dict(before=state_before, after=self._curriculum_state(),
                                  events=[dict(item) for item in self.transitions[-5:]],
                                  phase=self.phase, substage=self.substage))
            if self.stop_state is None:
                self.save('transition')
        if self.update > 0 and self.update % int(speed.SAVE_INTERVAL) == 0:
            self.save('interval')
            actions.append('save')
        if gate is not None and self.stop_state is None:
            self.save('gate')
        self._reset_environment_schedule()
        self._check_stop_conditions()
        self.pending_boundary = None
        return dict(update=update, phase=self.phase, substage=self.substage,
                    actions=actions, stop_state=self.stop_state,
                    lifecycle_state=self.lifecycle_state)

    def _complete_pending_boundary(self):
        """Resume: finish the interrupted boundary without replaying PPO/billing."""
        pending = deepcopy(self.pending_boundary)
        if pending is None:
            return None
        if int(pending['update']) != int(self.update):
            raise recovery_module.RecoveryBlocked(
                'pending boundary update does not match the checkpoint learning update')
        # Q006 section 5: replay the committed transaction under its own identity
        # (update/phase/generation), never under the current, possibly moved state.
        if pending.get('phase') is not None \
                and str(pending['phase']).upper() != str(self.phase).upper():
            raise recovery_module.RecoveryBlocked(
                'pending boundary phase does not match the checkpoint curriculum phase')
        if pending.get('generation') is not None:
            current_generation = self._curriculum_state().get('generation')
            if int(pending['generation']) != int(current_generation):
                raise recovery_module.RecoveryBlocked(
                    'pending boundary generation does not match the checkpoint '
                    'curriculum generation')
        self._run_pending_tasks(pending)
        gate = next((record.as_dict() for record in self.gates
                     if record.kind == 'validation'
                     and record.update == int(pending['update'])), None)
        transitions = self._finalize_curriculum(pending, gate)
        for event in transitions or []:
            item = event.as_dict() if hasattr(event, 'as_dict') else dict(event)
            item['update'] = int(pending['update'])
            self.transitions.append(item)
            self._consume_terminal_event(item)
        self.pending_boundary = None
        self._write_json('pending_boundary.json', dict(completed=True,
                                                       update=int(pending['update'])))
        self._reset_environment_schedule()
        self._check_stop_conditions()
        return pending['update']

    def _curriculum_finished(self):
        state = self._curriculum_state()
        return bool(state.get('finished', False))

    def train_one_update(self):
        self._check_resume_evidence(stage='learning')
        if self.stop_state is not None:
            raise RunStopped(self.stop_state, evidence=self.stop_evidence)
        if self.lifecycle_state != 'running':
            raise RunStopped(self.lifecycle_state,
                             evidence=dict(lifecycle_state=self.lifecycle_state))
        if self.curriculum_complete or self._curriculum_finished():
            # A finished/qualified curriculum never authorizes another billed update.
            self._check_stop_conditions()
            if self.lifecycle_state != 'running' or self.stop_state is not None:
                raise RunStopped(self.lifecycle_state if self.stop_state is None
                                 else self.stop_state, evidence=self.stop_evidence)
            raise RunStopped('ready_for_final_test',
                             evidence=dict(curriculum_finished=True))
        if self.update >= speed.total_budget():
            self._stop('budget_exhausted', dict(reason='total budget reached before update'))
            raise RunStopped('budget_exhausted', evidence=self.stop_evidence)
        try:
            records = self.ppo.collect()
            outcomes = [_outcome(record) for record in records]
            if outcomes:
                self.curriculum.observe(outcomes)
            metrics = self.ppo.update()
        except Exception as error:  # noqa: BLE001 - PPO half-update is not a full update
            self.error_type = type(error).__name__
            incomplete = dict(update=int(self.update), error=repr(error),
                              error_type=self.error_type,
                              optimizer_steps=self.ppo.learning_state.get(
                                  'optimizer_steps'),
                              resumable=False)
            self.incomplete_update = incomplete
            self._write_json('ppo_incomplete.json', incomplete)
            try:
                self.save('incomplete', incomplete_update=incomplete)
            except Exception as save_error:  # noqa: BLE001 - never mask the original
                self._write_json('incomplete_save_error.json',
                                 dict(error=repr(save_error), update=int(self.update)))
            raise
        self.update = int(self.ppo.learning['update'])
        self.last_metrics = dict(metrics)
        self.commit_learning_update(self.update)
        boundary = self.finalize_boundary(self.update, records, metrics)
        if self.telemetry is not None and hasattr(self.telemetry, 'update'):
            self.telemetry.update(metrics, boundary)
        result = dict(update=self.update, phase=self.phase, substage=self.substage,
                      metrics=metrics, boundary=boundary,
                      learning=self.learning, budget=self.budget)
        if self.stop_state is not None:
            result['stop_state'] = self.stop_state
        return result

    def run(self, max_updates=None):
        limit = int(speed.total_budget() if max_updates is None else max_updates)
        try:
            # Q002-6: a checkpoint that interrupted a retention check completers
            # it here, once, before any new learning update.
            self._complete_pending_evaluations()
            completed = self._complete_pending_boundary()
            if (completed is not None and self.require_resume_evidence
                    and self.stop_state is None and self.resume_ready is None):
                # Q006 section 4.4: the completed update-100 gate/streak/curriculum
                # transaction gets its own verified checkpoint before update 101.
                self.save_verified('resume_ready')
            while (self.stop_state is None and self.lifecycle_state == 'running'
                   and not self.curriculum_complete and self.update < limit):
                self.train_one_update()
        except RunStopped:
            pass
        except Exception as error:  # save complete evidence, never auto-retry
            self.stop_evidence = dict(stop_state='exception', error=repr(error),
                                     update=self.update, phase=self.phase,
                                     substage=self.substage)
            try:
                self.save('exception', pending_update=self._pending_update_marker())
            except Exception as save_error:  # noqa: BLE001 - evidence must still exist
                self._write_json('exception_save_error.json',
                                 dict(error=repr(save_error),
                                      update=int(self.update)))
            finally:
                self._write_json('exception.json', self.stop_evidence)
            raise
        # Q002: R passing twice only reaches ready_for_final_test; the frozen
        # 300-case test (which may be zero-learning at the exact total budget)
        # decides complete vs gate_failed(final_test_failed).  Budget labels are
        # judged last and never overwrite a completion path.
        if self.stop_state is None and self.lifecycle_state == 'running':
            self._check_stop_conditions()
            if self.stop_state is None and self.curriculum_complete \
                    and self.lifecycle_state == 'running':
                self._mark_lifecycle('ready_for_final_test',
                                     reason='final_gates_passed')
        if self.stop_state is None and self.lifecycle_state == 'ready_for_final_test':
            try:
                self.final_test()
            except Exception as error:
                # The final zero-learning transaction is outside the learning loop,
                # but its actual cases must receive the same exception evidence.
                self.stop_evidence = dict(stop_state='exception',error=repr(error),
                    update=self.update,phase=self.phase,substage=self.substage)
                try:
                    self.save('exception',pending_update=self._pending_update_marker())
                except Exception as save_error:
                    try:
                        self._write_json('exception_save_error.json',dict(error=repr(save_error),update=self.update))
                    except Exception:
                        pass  # Preserve the original error if storage itself failed.
                try:
                    self._write_json('exception.json',self.stop_evidence)
                except Exception:
                    pass
                raise
        elif self.stop_state is None:
            self._stop('budget_exhausted', dict(
                update=self.update, limit=int(limit),
                budget=self._budget_layers()), reason='run_limit_reached')
        return self.result()

    # ------------------------------------------------------------- final test
    def final_test(self):
        """The frozen final test: three speeds x 100 cases, >=95 per speed (SDS 6.4).

        R passing twice only reaches ``ready_for_final_test``; ``complete`` exists
        solely when all three speeds reach the threshold.  A valid test that falls
        short is ``gate_failed`` with ``reason=final_test_failed`` and keeps every
        per-speed count; an evaluation exception or incomplete case set is an
        engineering error and is never scored as a final failure or a pass.
        """
        if self.case_runner is None:
            raise RunnerError('final test needs the deterministic-mean case runner')
        if self.lifecycle_state != 'ready_for_final_test' and not self.curriculum_complete:
            raise RunnerError(
                'final_test requires ready_for_final_test (R two consecutive gates)')
        if self.phase != 'R':
            raise RunnerError('final_test requires the actual R curriculum phase')
        if self.evaluation_transaction is None:
            frozen = self.save('final_frozen')
        else:
            self._validate_evaluation_transaction()
            frozen = self.evaluation_transaction.get('frozen_checkpoint')
            if self.evaluation_transaction['identity']['kind']!='test' or not frozen:
                raise recovery_module.RecoveryBlocked('final test cache lacks its frozen checkpoint binding')
        try:
            result = self._evaluate_with_record_transaction(
                'test',self.update,round=None,tile_id=int(spec.FINAL_TILE_ID))
            self.evaluation_transaction['frozen_checkpoint'] = deepcopy(frozen)
        except Exception as error:  # engineering error, never a scored outcome
            self.error_type = type(error).__name__
            self.final = dict(checkpoint=frozen, update=self.update, valid=False,
                              passed=None, error_type=self.error_type,
                              error=repr(error))
            self._write_json('final_test/error.json',
                             dict(error_type=self.error_type, error=repr(error),
                                  checkpoint=frozen, update=self.update))
            raise
        binding = self._persist_evaluation_records(result)
        if self.evaluation_transaction['stage']=='applied':
            self.final['case_records']=deepcopy(binding)
            self._write_json('final_test/summary.json',self.final)
            self._finish_direct_evaluation_transaction()
            return self.final
        gate = evaluation_module.gate_result(result.summary)
        complete_cases = all(int(entry['total']) == int(speed.FINAL_CASES)
                             for entry in result.summary['per_speed'])
        self.final = dict(
            checkpoint=frozen, phase_document='R', update=self.update,
            spawn_profile=result.table.spawn_profile,
            spawn_parameters=dict(result.table.spawn_parameters),
            target_lesson=result.table.target_lesson,
            target_tile_id=result.table.target_tile_id,
            gate=gate, summary=result.summary, evidence=result.evidence_sha256,
            case_records=binding,
            speeds=[float(value) for value in result.table.speeds],
            cases_per_speed=int(result.table.episodes),
            pass_min=int(speed.FINAL_PASS_MIN),
            per_speed=[dict(item) for item in result.summary['per_speed']],
            valid=bool(complete_cases),
            passed=bool(complete_cases and gate['gate_pass']),
            error_type=None,
        )
        self._write_json('final_test/summary.json', self.final)
        self._write_json('final_test/records.json', dict(records=result.records))
        if not complete_cases:
            self.error_type = 'FinalTestIncomplete'
            raise evaluation_module.EvaluationError(
                'the final test did not produce 100 valid cases per speed; this is not '
                'a scored final failure and cannot be complete')
        self.evaluation_transaction['stage']='applying'
        try:
            if not self.final['passed']:
                self._stop('gate_failed', dict(
                    stage='final_test', gate=gate, checkpoint=frozen,
                    per_speed=list(self.final['per_speed']),
                    cases_per_speed=int(speed.FINAL_CASES),
                    pass_min=int(speed.FINAL_PASS_MIN),
                    spawn_profile=result.table.spawn_profile,
                    budget=self._budget_layers(), reason=FINAL_TEST_REASON),
                    reason=FINAL_TEST_REASON)
            else:
                self._mark_lifecycle('complete', reason='final_test_passed',
                                     evidence=dict(gate=gate, checkpoint=frozen))
                self._write_json('final_test/complete.json',
                                 dict(state='complete', gate=gate, checkpoint=frozen,
                                      update=self.update))
        except Exception as error:
            self.evaluation_transaction.update(stage='application_incomplete',application_error=repr(error))
            raise
        self.evaluation_transaction['stage']='applied'
        self._finish_direct_evaluation_transaction()
        return self.final

    # ------------------------------------------------------------------ result
    def result(self):
        learning = self.learning
        curriculum = self._curriculum_state()
        state = (self.lifecycle_state if self.lifecycle_state == 'complete'
                 else (self.stop_state if self.stop_state is not None
                       else (self.lifecycle_state if self.lifecycle_state != 'running'
                             else None)))
        return dict(
            state=state,
            lifecycle_state=self.lifecycle_state,
            stop_state=self.stop_state,
            stop_reason=self.stop_reason,
            error_type=self.error_type,
            failure_class=classify_failure(self.stop_state),
            terminal=self._terminal_record(),
            stop_evidence=deepcopy(self.stop_evidence),
            update=int(self.update), phase=self.phase, substage=self.substage,
            generation=curriculum.get('generation'),
            learning=learning, curriculum=curriculum,
            budget=self._budget_layers(),
            budget_audit=recovery_module.audit_budget(learning, curriculum),
            gates=[record.as_dict() for record in self.gates],
            gate_streak=int(self.gate_streak),
            retention=deepcopy(self.retention),
            retention_schedule=self._retention_schedule(),
            diagnostic=deepcopy(self.diagnostic),
            pending_evaluations=[dict(item) for item in self.pending_evaluations],
            transitions=[dict(item) for item in self.transitions],
            diagnostics=[dict(item) for item in self.diagnostics],
            checkpoints=[dict(item) for item in self.checkpoints],
            final=deepcopy(self.final),
            acceptance_file=str(self.acceptance_file) if self.acceptance_file else None,
            sources=deepcopy(self.source),
            duration_s=float(self.clock() - self.starts),
        )


def _retention_short_kind(gate_kind):
    """Map the evaluation gate kind to the short retention kind (Q001-7)."""
    return {'flat_retention': 'flat', 'stair_retention': 'h00'}.get(
        str(gate_kind), str(gate_kind))


def _retention_gate_kind(curriculum, kind):
    """Map a short retention kind to its evaluation gate kind (B's mapping first)."""
    hook = getattr(curriculum, 'retention_gate_kind', None)
    if callable(hook):
        return str(hook(kind))
    return {'flat': 'flat_retention', 'h00': 'stair_retention'}.get(
        str(kind), str(kind))


def retention_target_tile(kind):
    """The frozen, phase-independent retention terrain (Q007): flat=0, H00=1."""
    gate_kind = str(kind)
    if gate_kind not in ('flat_retention', 'stair_retention'):
        gate_kind = {'flat': 'flat_retention', 'h00': 'stair_retention'}.get(
            gate_kind, gate_kind)
    if gate_kind == 'flat_retention':
        return int(spec.FLAT_TILE_ID)
    if gate_kind == 'stair_retention':
        return int(spec.tile_id_for('H00'))
    raise RunnerError(f'{kind!r} is not a retention set')


def _outcome(record):
    """Convert an environment episode record into the curriculum outcome shape."""
    if not isinstance(record, dict):
        raise RunnerError('environment episode records must be mappings')
    return record


def _gate_record(value):
    return GateRecord(phase=value['phase'], substage=value.get('substage'),
                      kind=value['kind'], update=int(value['update']),
                      passed=bool(value['passed']),
                      case_table_sha256=value['case_table_sha256'],
                      checkpoint_sha256=value.get('checkpoint_sha256'),
                      failed_cases=int(value.get('failed_cases', 0)),
                      summary=dict(value.get('summary') or {}))


def spec_recipe():
    config = spec_recipe_config()
    return config


def spec_recipe_config():
    from . import spec
    return spec.recipe_config()


def _same_case_scalar(left, right):
    """Exact-enough equality for the per-case scalar cross-check."""
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    try:
        first, second = float(left), float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    return abs(first - second) <= 1e-9


def _case_scalar(value, label, key):
    """One case's scalar: an env-0 value is never allowed to stand for the batch.

    The evaluation copies one case to every environment, so a per-environment
    tensor/list must be constant across the batch; a disagreement is an
    evaluation error.  Episode records and ``diagnostics_cases()`` rows are
    already per-case scalars.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.detach().reshape(-1)
        if flat.numel() == 0:
            raise evaluation_module.EvaluationError(f'{label}[{key!r}] is empty')
        if flat.dtype == torch.bool:
            values = [bool(item) for item in flat.tolist()]
        elif flat.dtype.is_floating_point:
            values = [float(item) for item in flat.tolist()]
        else:
            values = [int(item) for item in flat.tolist()]
        first = values[0]
        if not all(_same_case_scalar(first, item) for item in values[1:]):
            raise evaluation_module.EvaluationError(
                f'{label}[{key!r}] differs across the evaluation batch; a case must be '
                'copied to every environment with one identical value')
        return first
    if isinstance(value, (list, tuple)):
        if not value:
            raise evaluation_module.EvaluationError(f'{label}[{key!r}] is empty')
        first = value[0]
        if not all(_same_case_scalar(first, item) for item in value[1:]):
            raise evaluation_module.EvaluationError(
                f'{label}[{key!r}] differs across the evaluation batch')
        return _case_scalar(first, label, key)
    if isinstance(value, (bool, int, float, str)):
        return value
    raise evaluation_module.EvaluationError(
        f'{label}[{key!r}] has unsupported type {type(value).__name__}')


def _case_field(key, sources, *, aliases=()):
    """First non-null scalar for ``key``/aliases, cross-checked across all sources."""
    keys = (key,) + tuple(aliases)
    found = []
    for label, container in sources:
        if not isinstance(container, dict):
            continue
        for candidate in keys:
            if container.get(candidate) is not None:
                found.append((f'{label}[{candidate!r}]',
                              _case_scalar(container[candidate], label, candidate)))
                break
    if not found:
        return None
    first_label, first = found[0]
    for label, value in found[1:]:
        if not _same_case_scalar(first, value):
            raise evaluation_module.EvaluationError(
                f'case evidence {key!r} disagrees: {first_label}={first!r} vs '
                f'{label}={value!r}')
    return first


def _env_scalar(value, label, key, index=None):
    """A per-case value: diagnostics rows are scalar, info is per-env [B]."""
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.detach().reshape(-1)
        if index is not None and flat.numel() > int(index):
            value = flat[int(index)]
        elif flat.numel() == 1:
            value = flat[0]
        else:
            raise evaluation_module.EvaluationError(
                f'{label}[{key!r}] is a per-env tensor but the case index is unknown')
        return _env_scalar(value.item() if value.dim() == 0 else value, label, key)
    if isinstance(value, (list, tuple)):
        if index is not None and len(value) > int(index):
            return _env_scalar(value[int(index)], label, key)
        if len(value) == 1:
            return _env_scalar(value[0], label, key)
        raise evaluation_module.EvaluationError(
            f'{label}[{key!r}] is a batch list but the case index is unknown')
    if isinstance(value, (bool, int, float, str)):
        return value
    raise evaluation_module.EvaluationError(
        f'{label}[{key!r}] has unsupported type {type(value).__name__}')


def _first_case_scalar(record, names, *, label, required=True):
    """Read only one already-frozen episode; never consult another live source."""
    for name in names:
        if name in record and record[name] is not None:
            return _env_scalar(record[name], label, name)
    if required:
        raise evaluation_module.EvaluationError(f'{label} lacks first-episode {names[0]}')
    return None


def _first_case_integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise evaluation_module.EvaluationError(f'{label} must be an integer >= {minimum}')
    return int(value)


def _validate_first_case(snapshot, *, index, batch_step, evidence_kind, flat):
    label = f'first_episode[{index}]'
    step = _first_case_integer(_first_case_scalar(snapshot,
        ('control_samples', 'terminal_step', 'control_step'), label=label), label+'.steps', minimum=1)
    for name in ('control_samples', 'terminal_step', 'control_step'):
        if name in snapshot and _first_case_integer(snapshot[name], label+'.'+name, minimum=1) != step:
            raise evaluation_module.EvaluationError(label+' has inconsistent episode step counters')
    if step != int(batch_step):
        raise evaluation_module.EvaluationError(label+' was not frozen at its first episode boundary')
    for name in ('success', 'failure', 'timeout', 'first_step_valid'):
        value = _first_case_scalar(snapshot, (name,), label=label)
        if not isinstance(value, (bool, np.bool_)):
            raise evaluation_module.EvaluationError(label+'.'+name+' must be boolean')
    if evidence_kind == 'terminal' and not any(bool(snapshot[k]) for k in ('success','failure','timeout')):
        raise evaluation_module.EvaluationError(label+' is not a terminal episode')
    active = _first_case_integer(_first_case_scalar(snapshot,('active_samples',),label=label),label+'.active_samples')
    stalled = _first_case_integer(_first_case_scalar(snapshot,('stall_samples',),label=label),label+'.stall_samples')
    if not 0 <= stalled <= active <= step:
        raise evaluation_module.EvaluationError(label+' has inconsistent first-episode diagnostic counters')
    for names in [('mean_vx',),('mean_speed',),('max_abs_y','max_lateral'),('max_abs_yaw_deg',)]:
        value = _first_case_scalar(snapshot,names,label=label)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)):
            raise evaluation_module.EvaluationError(label+'.'+names[0]+' must be finite')
    reason = _first_case_scalar(snapshot,('reason',),label=label)
    if evidence_kind == 'terminal':
        if not isinstance(reason,str) or not reason:
            raise evaluation_module.EvaluationError(label+'.reason must be a nonempty terminal string')
    elif isinstance(reason,bool) or not isinstance(reason,(int,str)) or reason == '':
        raise evaluation_module.EvaluationError(label+'.reason must retain the same-frame integer code/string')
    if flat:
        for names in [('forward_distance','net_displacement','flat_net_displacement'),
                      ('forward_distance_start','start_local_x','local_x_start','x_start','x0','flat_start_x'),
                      ('forward_distance_end','end_local_x','local_x_end','x_end','x500','flat_end_x')]:
            value = _first_case_scalar(snapshot,names,label=label)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)):
                raise evaluation_module.EvaluationError(label+'.'+names[0]+' must be finite')
    return step


def environment_case_runner(model, env_factory, *, device='cpu', step_limit=None):
    """Q010: one immutable first episode (or exact F0 500-step window) per slot.

    Finished slots continue through the original simulator/action path while
    other slots finish. Their later episodes never replace or supplement the
    frozen case. The additional physical computation is reported separately.
    """
    def run(cases):
        cases = list(cases)
        if not cases:
            raise evaluation_module.EvaluationError('empty evaluation batch')
        count = len(cases)
        ids = [case.get('case_id') for case in cases]
        if any(not isinstance(value,str) or not value for value in ids) or len(set(ids)) != count:
            raise evaluation_module.EvaluationError('case IDs must be explicit and unique within the batch')
        env = env_factory(cases)
        if hasattr(env,'num_envs') and int(env.num_envs) != count:
            raise evaluation_module.EvaluationError('one explicit case is required per environment slot')
        out = env.reset()
        obs = out[0] if isinstance(out,tuple) else out
        limit = int(step_limit or _case_step_limit(cases[0]))
        frozen = {}; origins = {}; first_batch_steps = {}
        loops = 0

        def mask(value,label):
            values = torch.as_tensor(value).detach().reshape(-1).cpu()
            if values.numel() == 1 and count == 1:
                values = values.expand(count)
            if values.numel() != count or not bool(torch.isfinite(values).all()):
                raise evaluation_module.EvaluationError(label+' does not cover the evaluation slots')
            if not bool(((values==0)|(values==1)).all()):
                raise evaluation_module.EvaluationError(label+' is not a boolean mask')
            return [bool(x) for x in values.tolist()]

        for loops in range(1,limit+1):
            action = model.act(obs,deterministic=True)
            obs,reward,terminated,truncated,info = env.step(action)
            if not isinstance(info,dict):
                raise evaluation_module.EvaluationError('evaluation step lacks pre-reset info')
            term = mask(terminated,'terminated'); trunc = mask(truncated,'truncated')
            rows = info.get('episodes') or []
            if not isinstance(rows,(list,tuple)):
                raise evaluation_module.EvaluationError('pre-reset episodes must be a list')
            frame_seen = set()
            for record in rows:
                if not isinstance(record,dict) or 'env_id' not in record:
                    raise evaluation_module.EvaluationError('every terminal episode needs an explicit env_id')
                index = _first_case_integer(record['env_id'],'episode.env_id')
                if index >= count or index in frame_seen:
                    raise evaluation_module.EvaluationError('terminal env_id is out of bounds or duplicated in one frame')
                frame_seen.add(index)
                if not (term[index] or trunc[index]):
                    raise evaluation_module.EvaluationError('terminal episode has no same-frame done mask')
                if index in frozen:
                    continue
                snapshot = deepcopy(record)
                _validate_first_case(snapshot,index=index,batch_step=loops,evidence_kind='terminal',
                                     flat=str(cases[index].get('measurement_stage'))=='F0')
                if bool(snapshot['timeout']) != trunc[index] or bool(snapshot['success'] or snapshot['failure']) != term[index]:
                    raise evaluation_module.EvaluationError('first terminal flags disagree with the same-frame masks')
                frozen[index] = snapshot; origins[index] = 'terminal'; first_batch_steps[index] = loops
            for index in range(count):
                if (term[index] or trunc[index]) and index not in frame_seen:
                    raise evaluation_module.EvaluationError('a done slot has no same-frame first terminal evidence')
            if loops == int(speed.F0_GATE_STEPS):
                live = [i for i,c in enumerate(cases) if i not in frozen
                        and str(c.get('measurement_stage'))=='F0']
                if live:
                    hook = getattr(env,'diagnostics_cases',None)
                    case_rows = hook() if callable(hook) else None
                    if not isinstance(case_rows,(list,tuple)) or len(case_rows) != count:
                        raise evaluation_module.EvaluationError('the F0 500-step window lacks same-frame per-slot diagnostics')
                    for index in live:
                        row = case_rows[index]
                        if not isinstance(row,dict) or 'env_id' not in row \
                                or _first_case_integer(row['env_id'],'window.env_id') != index:
                            raise evaluation_module.EvaluationError('F0 window case/env indices disagree')
                        snapshot = dict(env_id=index,control_samples=row.get('step'),
                            terminal_step=row.get('step'))
                        for key in ('success','failure','timeout','reason'):
                            if key not in info:
                                raise evaluation_module.EvaluationError('F0 window lacks same-frame '+key)
                            value = info[key]
                            width = (value.numel() if torch.is_tensor(value) else
                                     len(value) if isinstance(value,(list,tuple)) else 1)
                            if width != count:
                                raise evaluation_module.EvaluationError('F0 window info must provide one '+key+' per slot')
                            value = _env_scalar(value,'window.info',key,index)
                            if key == 'reason':
                                if isinstance(value,bool) or not isinstance(value,(int,str)):
                                    raise evaluation_module.EvaluationError('F0 window reason is not a finite code/string')
                                snapshot['reason'] = deepcopy(value)
                                snapshot['pre_reset_reason'] = deepcopy(value)
                            else:
                                snapshot[key] = value
                        for key in ('active_samples','stall_samples','first_step_valid',
                                    'max_abs_y','max_abs_yaw_deg','forward_distance',
                                    'forward_distance_start','forward_distance_end','mean_vx','mean_speed'):
                            if key not in row:
                                raise evaluation_module.EvaluationError('F0 window lacks first-episode '+key)
                            snapshot[key] = deepcopy(row[key])
                        snapshot = deepcopy(snapshot)
                        _validate_first_case(snapshot,index=index,batch_step=loops,evidence_kind='window_end',flat=True)
                        if any(bool(snapshot[k]) for k in ('success','failure','timeout')):
                            raise evaluation_module.EvaluationError('a physical terminal at 500 must use its first episode record')
                        frozen[index] = snapshot; origins[index] = 'window_end'; first_batch_steps[index] = loops
            if len(frozen) == count:
                break
        if len(frozen) != count:
            raise evaluation_module.EvaluationError('incomplete first-episode evaluation; F0 requires its full 500-step window')
        records = []
        for index,case in enumerate(cases):
            episode = frozen[index]
            def get(*keys):
                return _first_case_scalar(episode,keys,label=f'first_episode[{index}]',required=False)
            success = bool(episode['success']); failure = bool(episode['failure'])
            case_steps = _first_case_integer(get('control_samples','terminal_step','control_step'),'case.steps',minimum=1)
            flat = str(case.get('measurement_stage'))=='F0'
            net_forward = get('forward_distance','net_displacement','flat_net_displacement')
            start_x = get('forward_distance_start','start_local_x','local_x_start','x_start','x0','flat_start_x')
            end_x = get('forward_distance_end','end_local_x','local_x_end','x_end','x500','flat_end_x')
            forward_distance = None; evidence = None; flat_pass = None
            max_abs_y = get('max_abs_y','max_lateral'); max_abs_yaw_deg = get('max_abs_yaw_deg')
            if flat:
                forward_distance = speed.net_displacement(start_x,end_x)
                if abs(float(net_forward)-forward_distance) > 1e-6:
                    raise evaluation_module.EvaluationError('env forward_distance disagrees with the frozen net displacement (Q003): '
                        f'D_env={net_forward!r} start={start_x!r} end={end_x!r} net={forward_distance!r} tolerance=1e-6')
                if not failure and case_steps != int(speed.F0_GATE_STEPS):
                    raise evaluation_module.EvaluationError('an F0 case ended before 500 steps without a failure')
                evidence = dict(case_id=case['case_id'],env_index=index,steps=case_steps,D_env=float(net_forward),
                    start=float(start_x),end=float(end_x),canonical_net=float(forward_distance),
                    abs_difference=abs(float(net_forward)-forward_distance),tolerance=1e-6,
                    evidence_kind=origins[index])
                flat_pass = speed.flat_case_pass(speed.flat_case_metrics(speed=float(case['vx']),steps=case_steps,
                    max_abs_y=float(max_abs_y),max_abs_yaw_deg=float(max_abs_yaw_deg),
                    forward=float(forward_distance),failure=failure))
            records.append(dict(case_id=case['case_id'],env_index=index,steps=case_steps,
                climb_success=success,failure=failure,timeout=bool(episode['timeout']),reason=episode['reason'],
                active_samples=int(episode['active_samples']),stall_samples=int(episode['stall_samples']),
                first_step_valid=bool(episode['first_step_valid']),mean_vx=float(episode['mean_vx']),
                mean_speed=float(episode['mean_speed']),forward_distance=forward_distance,
                forward_distance_evidence=evidence,forward_distance_definition=speed.forward_distance_definition(),
                max_abs_y=max_abs_y,max_abs_yaw_deg=max_abs_yaw_deg,forward=forward_distance,
                flat_pass=flat_pass,measurement='flat' if flat else 'stair',evidence_kind=origins[index],
                first_episode=deepcopy(episode),first_observed_at_batch_step=first_batch_steps[index],
                reason_label=('f0_window_end' if origins[index]=='window_end' else None),
                batch_loop_control_steps=loops,physical_control_samples=loops,
                physical_physics_substeps=loops*int(spec.DECIMATION),
                extra_control_samples_after_case=loops-case_steps,
                extra_physics_substeps_after_case=(loops-case_steps)*int(spec.DECIMATION)))
        run.last_cost = dict(case_control_samples=sum(r['steps'] for r in records),
            physical_control_samples=count*loops,physical_physics_substeps=count*loops*int(spec.DECIMATION),
            extra_control_samples=sum(r['extra_control_samples_after_case'] for r in records),
            learning_updates=0,counts_against_training=False)
        return records

    run.supports_batch = True
    return run


def _case_step_limit(case):
    """SDS 6.2 timeout: ``ceil((2 * (2.2 + N*sqrt(h^2+d^2))/c + 10)/0.02)``."""
    if str(case.get('measurement_stage')) == 'F0':
        return int(speed.F0_GATE_STEPS)
    spec = spec_tile(case)
    n = float(spec.N)
    h = float(spec.h)
    d = float(spec.step_depth)
    c = max(float(case.get('vx', 0.2)), 1e-6)
    return int(math.ceil((2 * (2.2 + n * math.sqrt(h * h + d * d)) / c + 10) / 0.02))


def spec_tile(case):
    from . import spec
    return spec.terrain_spec(int(case.get('tile_id', spec.tile_id_for('L40'))))


__all__ = [
    'FAILURE_CLASS', 'GateRecord', 'PHASE_GATE_RULES', 'RunStopped', 'RunnerError',
    'STOP_STATES', 'V06Runner', 'classify_failure', 'classify_stop',
    'environment_case_runner', 'spec_recipe',
]
