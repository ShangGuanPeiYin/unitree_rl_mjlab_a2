"""CPU state-machine tests for the v0.5 training loop and curriculum (SDS 9.2-14/15).

Everything here runs on CPU with a fake environment, a fake evaluation service and
an in-memory ledger that mirrors ``v05.recovery.AttemptLedger``; no MuJoCo, no
CUDA, no network and no real training run is started.  A few integration tests use
the real ``AttemptLedger``, the real ``v05.ppo.PPO`` and the real schema-6
envelope so the runner's calls are proven against the durable interfaces.
"""
import sys
sys.path.insert(0, 'scripts')
import a2_v05_bootstrap
a2_v05_bootstrap.activate('.')

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.tasks.velocity.v05 import recovery, runner as v05_runner
from src.tasks.velocity.v05.recovery import (AttemptLedger, preload_checkpoint, task_key,
                                             learning_state, updated_curriculum)
from src.tasks.velocity.v05.runner import (COLLAPSE_CONSECUTIVE, Curriculum,
                                           DiagnosticEvidenceError, EvaluationError,
                                           PilotLedgerConflict, V05Runner,
                                           diagnostic_decision)
from src.tasks.velocity.v05 import spec

SPEEDS = spec.SPEEDS
FROZEN_CONTRACT = 'a' * 64


def digest(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


# ------------------------------------------------------------------- fixtures
def gate_summary(*, passed=True, total=20, passed_count=None):
    count = (total if passed else 0) if passed_count is None else passed_count
    return {'per_speed': [dict(c=speed, total=total, case_pass_count=count,
                               speed_pass_count=count, climb_success_count=count,
                               failure_count=total - count) for speed in SPEEDS]}


def diagnostic_summary(*, stalled_cases=0, samples=100, stalled_samples=None,
                       first_step_absent=0):
    """15 diagnostic cases in speed order with the SDS 6.3 evidence fields."""
    rows = []
    for speed in SPEEDS:
        for _ in range(5):
            case = len(rows)
            stalled = case < stalled_cases
            rows.append(dict(
                c=speed, samples=samples,
                stall_samples=(stalled_samples if stalled_samples is not None else samples)
                if stalled else 0,
                first_step_acquired=case >= first_step_absent))
    return {'per_case': rows}


class FakeEnv:
    """Observable stand-in: counts resets, steps and stage bindings."""
    device = 'cpu'

    def __init__(self, num_envs=8, seed=11):
        self.num_envs = int(num_envs)
        self.rng = torch.Generator().manual_seed(seed)
        self.stage = 'F0'
        self.stage_iteration = 0
        self.resets = 0
        self.reset_count = 0
        self.stage_resets = 0
        self.steps = 0
        self.stage_calls = []
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0

    def set_stage(self, stage, stage_iteration=0, previous_stage=None):
        changed = stage != self.stage
        self.stage = stage
        self.stage_iteration = int(stage_iteration)
        self.stage_calls.append((stage, int(stage_iteration), previous_stage))
        if changed:
            # Mirrors V05Env: set_stage resets only when the stage really changes.
            self.stage_resets += 1
            self._reset()
        return None

    def _obs(self):
        return {'proprio': torch.zeros(self.num_envs, 45),
                'heightmap': torch.zeros(self.num_envs, 3, 1, 1)}

    def _reset(self):
        self.resets += 1
        self.reset_count += 1
        # All training streams are consumed so an RNG restore is observable.
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, generator=self.rng)
        self.episode_length_buf.zero_()
        return self._obs()

    def reset(self):
        return self._reset(), {}

    def step(self, action):
        self.steps += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs), zeros, zeros, {}


class TinyModel(torch.nn.Module):
    """One trainable tensor plus the PPO-facing hook of the real network."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def update_critic_statistics(self, obs):
        return None


class FakePPO:
    """PPO stand-in emitting the exact ledger callback sequence."""
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.cfg = type('Cfg', (), dict(rollout=48, epochs=5, minibatches=4))()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=spec.PPO_INITIAL_LR)
        self.learning_state = dict(optimizer_initialized=False, completed_ppo_updates=0,
                                   optimizer_steps=0)
        self.obs = None
        self.storage = None
        self.control_step_callback = None
        self.update_callback = None
        self.diagnostics_callback = None
        self.collections = 0

    @property
    def learning(self):
        update = self.learning_state['completed_ppo_updates']
        samples = 48 * self.env.num_envs * update
        return dict(update=update, optimizer_step=self.learning_state['optimizer_steps'],
                    control_samples=samples, physics_substeps=4 * samples)

    def collect(self):
        if self.obs is None:
            self.obs, _ = self.env.reset()
        for step in range(48):
            self.obs, _, _, _, _ = self.env.step(torch.zeros(self.env.num_envs, 12))
            if self.control_step_callback is not None:
                self.control_step_callback(dict(step_index=step + 1,
                                                num_envs=self.env.num_envs,
                                                physics_samples_increment=self.env.num_envs,
                                                info={}))
        self.collections += 1
        return {}

    def update(self):
        base = self.learning_state['completed_ppo_updates']
        if self.update_callback is not None:
            self.update_callback('optimizer_started',
                                 dict(learning_state=self.learning,
                                      expected_optimizer_steps=20))
        for _ in range(20):
            self.optimizer.zero_grad(set_to_none=True)
            self.model.weight.square().backward()
            self.optimizer.step()
            self.learning_state['optimizer_steps'] += 1
            if self.update_callback is not None:
                self.update_callback('optimizer_step_completed',
                                     dict(learning_state=self.learning))
        self.learning_state['optimizer_initialized'] = True
        self.learning_state['completed_ppo_updates'] = base + 1
        if self.update_callback is not None:
            self.update_callback('update_completed',
                                 dict(learning_state=self.learning,
                                      optimizer_steps_in_update=20))
        return {}


class FakeLedger(AttemptLedger):
    """In-memory mirror of the durable ledger's runner-facing contract.

    It is a subclass only so ``build_checkpoint`` accepts it; no file is created.
    Event validation, hash chaining and fsync are the real ledger's job and are
    covered by ``RealLedgerIntegrationTests``.
    """

    def __init__(self, *, mode='formal', run_id='v05unit'):
        self.header = dict(schema=1, run_id=run_id, mode=mode)
        self._curriculum = dict(stage='F0', stage_iteration=0,
                                spent={stage: 0 for stage in spec.STAGE_ORDER},
                                streak=dict(gate=0, collapse=0), pending_transition=None)
        self._completed = 0
        self._active = None
        self._tasks = {}
        self._results = []
        self._qualified = None
        self._num_envs = 8
        self._checkpoints = []
        self.events = []

    # ------------------------------------------------------------ properties
    @property
    def mode(self):
        return self.header['mode']

    @property
    def curriculum(self):
        return self._curriculum

    @curriculum.setter
    def curriculum(self, value):
        self._curriculum = value

    @property
    def completed_ppo_updates(self):
        return self._completed

    @completed_ppo_updates.setter
    def completed_ppo_updates(self, value):
        self._completed = value

    @property
    def tasks(self):
        return self._tasks

    @property
    def results(self):
        return self._results

    @property
    def checkpoints(self):
        return self._checkpoints

    @property
    def qualified_update(self):
        return self._qualified

    @qualified_update.setter
    def qualified_update(self, value):
        self._qualified = value

    @property
    def control_samples(self):
        return 48 * self._num_envs * self._completed

    @property
    def physics_substeps(self):
        return 4 * self.control_samples

    @property
    def gate_streak(self):
        return self.curriculum['streak']['gate']

    @property
    def collapse_streak(self):
        return self.curriculum['streak']['collapse']

    @property
    def pending_transition(self):
        return self.curriculum['pending_transition']

    def anchor(self):
        return dict(seq=len(self.events), digest=digest(f'events-{len(self.events)}'))

    def state_dict(self):
        return dict(header=deepcopy(self.header), events=[], seq=0, digest='0' * 64)

    # -------------------------------------------------------------- attempts
    def begin_attempt(self, curriculum, num_envs):
        if curriculum != self.curriculum:
            raise ValueError('Curriculum differs from the ledger state')
        if self.curriculum['pending_transition'] is not None:
            raise ValueError('Cannot attempt training across a pending stage transition')
        stage = self.curriculum['stage']
        if self.curriculum['spent'][stage] >= spec.STAGE_BY_ID[stage].budget:
            raise ValueError('Stage budget already exhausted')
        self._num_envs = num_envs
        self._active = dict(controls=0, num_envs=num_envs, phase='collect',
                            optimizer_steps=0)
        self.events.append('begin_attempt')

    def control_step(self, event):
        if self._active is None or self._active['phase'] != 'collect':
            raise ValueError('No live attempt')
        if event.get('num_envs') != self._active['num_envs']:
            raise ValueError('Control step environment mismatch')
        self._active['controls'] += 1

    def update_event(self, kind, payload):
        if self._active is None:
            raise ValueError('No live attempt')
        if kind == 'optimizer_started':
            self._active['phase'] = 'optimizer_started'
        elif kind == 'optimizer_step_completed':
            self._active['optimizer_steps'] += 1
        elif kind == 'update_completed':
            self._active['phase'] = 'complete_pending'
        else:
            raise ValueError(f'Unsupported core callback {kind!r}')

    def complete_attempt(self, curriculum, learning):
        if self._active is None or self._active['phase'] != 'complete_pending':
            raise ValueError('Complete attempt without the exact boundary')
        expected = updated_curriculum(self.curriculum, num_envs=self._active['num_envs'])
        if curriculum != expected:
            raise ValueError('Curriculum update missing, duplicated or inconsistent')
        self.curriculum = deepcopy(curriculum)
        self.completed_ppo_updates += 1
        self._active = None
        self.events.append('complete_attempt')

    # ---------------------------------------------------------- evaluations
    def _task_record(self, key):
        parts = dict(zip(('run_id', 'mode', 'stage', 'global_update', 'kind',
                          'checkpoint_sha256', 'case_sha256'), key.split('|')))
        return dict(state='planned', run_id=parts['run_id'], mode=parts['mode'],
                    stage=parts['stage'], global_update=int(parts['global_update']),
                    kind=parts['kind'], checkpoint_sha256=parts['checkpoint_sha256'],
                    case_sha256=parts['case_sha256'], planned_wall=0.)

    def plan_evaluation(self, *, run_id, mode, stage, global_update, kind,
                        checkpoint_sha256, case_sha256, planned_wall=None):
        key = task_key(run_id=run_id, mode=mode, stage=stage, global_update=global_update,
                       kind=kind, checkpoint_sha256=checkpoint_sha256,
                       case_sha256=case_sha256)
        if key in self.tasks:
            return dict(task_key=key, state=self.tasks[key]['state'],
                        record=deepcopy(self.tasks[key]), replayed=True)
        self.tasks[key] = self._task_record(key)
        return dict(task_key=key, state='planned', record=deepcopy(self.tasks[key]),
                    replayed=False)

    def commit_evaluation_result(self, key, *, result_path, result_sha256, report_path,
                                 report_sha256, evidence_sha256, summary_sha256,
                                 committed_wall=None):
        record = self.tasks.get(key)
        if record is None or record['state'] != 'planned':
            raise ValueError('Result commitment without a planned task')
        record['state'] = 'result_committed'
        record['result'] = dict(result_path=result_path, result_sha256=result_sha256,
                                report_path=report_path, report_sha256=report_sha256,
                                evidence_sha256=list(evidence_sha256),
                                summary_sha256=summary_sha256,
                                committed_wall=float(committed_wall or 0.))
        return dict(task_key=key, state='result_committed')

    def consume_evaluation(self, key, *, outcome, passed, collapse_hit, consumed_wall=None):
        record = self.tasks.get(key)
        if record is None:
            raise ValueError('Consumption without a planned task')
        if record['state'] == 'consumed':
            prior = record['consumption']
            if (prior['outcome'], prior['passed'], prior['collapse_hit']) != (
                    outcome, bool(passed), bool(collapse_hit)):
                raise ValueError('Replayed consumption differs')
            return dict(task_key=key, state='consumed', replayed=True,
                        streak=deepcopy(self.curriculum['streak']))
        if record['state'] != 'result_committed':
            raise ValueError('Consumption requires a committed result')
        kind = record['kind']
        # Single source: the durable ledger's own streak/transition derivation.
        streak, pending = recovery.next_streak_state(
            self.curriculum, record, passed=bool(passed),
            collapse_hit=bool(collapse_hit), mode=self.mode)
        self.curriculum['streak'] = streak
        self.curriculum['pending_transition'] = pending
        record['state'] = 'consumed'
        record['consumption'] = dict(outcome=outcome, passed=bool(passed),
                                     collapse_hit=bool(collapse_hit),
                                     result_sha256=record['result']['result_sha256'],
                                     consumed_wall=float(consumed_wall or 0.),
                                     streak=deepcopy(streak))
        self.results.append(dict(task_key=key, kind=kind, stage=record['stage'],
                                 global_update=record['global_update'], outcome=outcome,
                                 passed=bool(passed), gate_streak=streak['gate'],
                                 collapse_streak=streak['collapse']))
        if (kind == 'validation' and streak['gate'] >= 2 and pending is not None
                and pending['to_stage'] is None):
            self.qualified_update = record['global_update']
        return dict(task_key=key, state='consumed', replayed=False,
                    streak=deepcopy(streak))

    def commit_transition(self, *, from_stage, to_stage, at_update, gate_streak):
        pending = self.curriculum['pending_transition']
        if pending is not None:
            if (pending['from_stage'], pending['to_stage'], pending['at_update']) != (
                    from_stage, to_stage, at_update):
                raise ValueError('Transition differs from the pending transition')
            if pending['requires'] == 'perturbed':
                satisfied = any(row['kind'] == 'perturbed' and row['passed']
                                and row['stage'] == from_stage
                                and row['global_update'] == at_update
                                for row in self.results)
                if not satisfied:
                    raise ValueError('S1 perturbation check is missing')
            self.curriculum.update(stage=to_stage, stage_iteration=0,
                                   streak=dict(gate=0, collapse=0),
                                   pending_transition=None)
            return dict(committed=True, replayed=False, stage=to_stage)
        if self.curriculum['stage'] == to_stage:
            return dict(committed=False, replayed=True, stage=to_stage)
        raise ValueError('No pending transition matches this commit')

    def pending_evaluations(self):
        return {key: deepcopy(record) for key, record in self.tasks.items()
                if record['state'] in ('planned', 'result_committed')}

    def checkpoint_task_state(self):
        return dict(tasks=deepcopy(self.tasks), results=deepcopy(self.results),
                    collapse_streak=self.collapse_streak, gate_streak=self.gate_streak,
                    anchor=self.anchor(), qualified_update=self.qualified_update,
                    pending_transition=deepcopy(self.curriculum['pending_transition']))

    def checkpoint_evaluation(self):
        consumed = [key for key, record in self.tasks.items()
                    if record['state'] == 'consumed']
        diagnostics = [record['global_update'] for record in self.tasks.values()
                       if record['state'] == 'consumed' and record['kind'] == 'diagnostic']
        return dict(consumed=sorted(consumed),
                    latest_diagnostic_update=max(diagnostics) if diagnostics else None)

    def record_checkpoint(self, path, checkpoint_sha, anchor, contract_sha256_value):
        self._checkpoints.append(dict(path=str(path), sha256=checkpoint_sha,
                                      anchor=anchor, contract=contract_sha256_value))


class FakeCaseTable:
    def __init__(self, stage, kind, seed):
        self.stage = stage
        self.kind = kind
        self.seed = int(seed)
        self.sha256 = digest(f'{stage}|{kind}|{self.seed}')


class FakeEvaluation:
    """Stand-in for ``v05.evaluation.evaluate_checkpoint`` that writes real files."""
    def __init__(self):
        self.calls = []
        self.diagnostic_plan = {}
        self.gate_pass = lambda kind, stage, update: True
        self.raise_on = set()
        self.seen_models = []

    def make_cases(self, stage, kind, seed):
        return FakeCaseTable(stage, kind, seed)

    def __call__(self, model, cases, *, device, num_envs, stage, kind, run_id,
                 global_update, checkpoint_sha, output_dir, record_video=False,
                 mode='formal', seed=42):
        out = Path(output_dir)
        self.calls.append(dict(kind=kind, stage=stage, update=int(global_update),
                               checkpoint_sha=checkpoint_sha, cases_sha=cases.sha256,
                               output_dir=str(out), num_envs=int(num_envs),
                               mode=mode, seed=seed))
        self.seen_models.append(model)
        if kind in self.raise_on:
            raise RuntimeError('simulated evaluation failure')
        out.mkdir(parents=True, exist_ok=False)
        episodes = 100 if kind == 'test' else (5 if kind == 'diagnostic' else 20)
        if kind == 'diagnostic':
            summary = self.diagnostic_plan.pop(int(global_update), diagnostic_summary())
        else:
            passed = bool(self.gate_pass(kind, stage, int(global_update)))
            summary = gate_summary(passed=passed, total=episodes)
        report = dict(summary, stage=stage, kind=kind, episodes=episodes,
                      checkpoint_sha256=checkpoint_sha, case_table_sha256=cases.sha256)
        (out / 'manifest.json').write_text(json.dumps(dict(
            run_id=run_id, mode=mode, stage=stage, kind=kind,
            global_update=int(global_update), checkpoint_sha256=checkpoint_sha,
            case_table_sha256=cases.sha256)))
        (out / 'report.json').write_text(json.dumps(report))
        (out / 'episodes.jsonl').write_text(
            '\n'.join(json.dumps({'case': index}) for index in range(episodes)) + '\n')
        (out / 'window_batch_0000.npz').write_bytes(b'fake-npz-evidence')
        key = (run_id, mode, stage, int(global_update), kind, checkpoint_sha, cases.sha256)
        return _EvalResult(key=key, summary=report,
                           raw_paths={'manifest': str(out / 'manifest.json'),
                                      'episodes': str(out / 'episodes.jsonl'),
                                      'npz': [str(out / 'window_batch_0000.npz')]})


class _EvalResult:
    def __init__(self, key, summary, raw_paths):
        self.key = key
        self.summary = summary
        self.raw_paths = raw_paths
        self.evidence_sha256 = {}


def frozen_source_identity():
    """Freeze the source-closure hashes so a parallel edit cannot rehash a run."""
    return (patch.object(recovery, 'contract_sha_for', return_value=FROZEN_CONTRACT),
            patch.object(recovery, 'recipe_config', return_value={'recipe': '0.5-r1'}))


def build(root, *, mode='formal', num_envs=8, seed=42, evaluation=None,
          save=True, final_test=True):
    env = FakeEnv(num_envs)
    model = TinyModel()
    ppo = FakePPO(model, env)
    ledger = FakeLedger(mode=mode, run_id=f'unit-{mode}-{seed}')
    curriculum = Curriculum(ledger, mode=mode, seed=seed)
    fake = evaluation or FakeEvaluation()
    runner = V05Runner(model=model, env=env, ppo=ppo, curriculum=curriculum,
                       ledger=ledger, device='cpu', seed=seed, output_dir=root,
                       evaluation=fake, make_cases=fake.make_cases,
                       save_checkpoints=save, final_test=final_test, eval_num_envs=8)
    runner.reset_episode()
    return dict(runner=runner, env=env, model=model, ppo=ppo, curriculum=curriculum,
                ledger=ledger, evaluation=fake)


def run_updates(fixture, count):
    last = None
    for _ in range(count):
        last = fixture['runner'].train_step()
    return last


def drive_ledger(ledger, *, num_envs=8, updates=1):
    """Charge ``updates`` complete updates directly into a FakeLedger."""
    for _ in range(updates):
        ledger.begin_attempt(ledger.curriculum, num_envs)
        for step in range(1, 49):
            ledger.control_step(dict(step_index=step, num_envs=num_envs,
                                     physics_samples_increment=num_envs))
        base = ledger.completed_ppo_updates
        ledger.update_event('optimizer_started', dict(
            learning_state=learning_state(base, num_envs=num_envs),
            expected_optimizer_steps=20))
        for step in range(1, 21):
            state = learning_state(base, num_envs=num_envs)
            state['optimizer_step'] = 20 * base + step
            ledger.update_event('optimizer_step_completed', dict(learning_state=state))
        ledger.update_event('update_completed', dict(
            learning_state=learning_state(base + 1, num_envs=num_envs),
            optimizer_steps_in_update=20))
        ledger.complete_attempt(updated_curriculum(ledger.curriculum, num_envs=num_envs),
                                learning_state(base + 1, num_envs=num_envs))


def consume_gate(ledger, *, passed, kind='validation', stage=None, checkpoint_sha=None,
                 outcome=None, case_seed=0):
    """Charge one evaluation task directly into a FakeLedger."""
    stage = ledger.curriculum['stage'] if stage is None else stage
    update = ledger.completed_ppo_updates
    checkpoint_sha = checkpoint_sha or digest(f'{update}-{stage}-{kind}')
    case_sha = digest(f'{stage}|{kind}|{case_seed}')
    key = task_key(run_id=ledger.header['run_id'], mode=ledger.mode, stage=stage,
                   global_update=update, kind=kind, checkpoint_sha256=checkpoint_sha,
                   case_sha256=case_sha)
    ledger.plan_evaluation(run_id=ledger.header['run_id'], mode=ledger.mode, stage=stage,
                           global_update=update, kind=kind,
                           checkpoint_sha256=checkpoint_sha, case_sha256=case_sha)
    ledger.commit_evaluation_result(
        key, result_path='/tmp/report.json', result_sha256='b' * 64,
        report_path='/tmp/report.json', report_sha256='b' * 64,
        evidence_sha256=['c' * 64], summary_sha256='d' * 64)
    return ledger.consume_evaluation(
        key, outcome=outcome or ('passed' if passed else 'failed'),
        passed=passed, collapse_hit=kind == 'diagnostic' and outcome == 'collapse_hit')


# ------------------------------------------------------------------ curriculum
class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ten_stage_table_budgets_and_gate_intervals(self):
        ledger = FakeLedger()
        curriculum = Curriculum(ledger)
        self.assertEqual(curriculum.stages,
                         ['F0', 'P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4', 'S5'])
        self.assertEqual(curriculum.budgets['F0'], 2000)
        self.assertEqual(curriculum.budgets['P1'], 1000)
        self.assertEqual(curriculum.budgets['S5'], 7000)
        self.assertEqual(sum(curriculum.budgets.values()), 30000)
        for stage in spec.STAGES:
            ledger.curriculum.update(stage=stage.id, stage_iteration=stage.gate,
                                     spent={s: 0 for s in spec.STAGE_ORDER})
            ledger.curriculum['spent'][stage.id] = stage.gate
            self.assertTrue(curriculum.validation_due, stage.id)
            ledger.curriculum['stage_iteration'] = stage.gate - 1
            self.assertFalse(curriculum.validation_due, stage.id)
        with self.assertRaises(ValueError):
            Curriculum(ledger, mode='other')
        with self.assertRaises(ValueError):
            Curriculum(ledger, seed=0)

    def test_gate_at_the_last_budget_update_can_still_promote(self):
        ledger = FakeLedger()
        drive_ledger(ledger, updates=1899)
        for _ in range(101):
            drive_ledger(ledger)
            if ledger.curriculum['stage_iteration'] in (1900, 2000):
                consume_gate(ledger, passed=True,
                             checkpoint_sha=digest(
                                 f'gate-{ledger.curriculum["stage_iteration"]}'))
        curriculum = Curriculum(ledger)
        self.assertEqual(curriculum.stage_iteration, 2000)
        self.assertTrue(curriculum.exhausted)
        # The final-budget gate was consumed (not due again) and the ledger set a
        # pending transition, so the last update still promotes (SDS 5.1).
        self.assertFalse(curriculum.validation_due)
        self.assertTrue(curriculum.can_advance)
        self.assertEqual(curriculum.pending_transition['to_stage'], 'P1')
        ledger.commit_transition(from_stage='F0', to_stage='P1', at_update=2000,
                                 gate_streak=2)
        self.assertEqual(curriculum.stage, 'P1')
        self.assertEqual(curriculum.stage_iteration, 0)
        self.assertFalse(curriculum.exhausted)

        failed = FakeLedger()
        drive_ledger(failed, updates=2000)
        failed_curriculum = Curriculum(failed)
        self.assertTrue(failed_curriculum.exhausted)
        self.assertFalse(failed_curriculum.can_advance)

    def test_streak_needs_two_consecutive_passes_and_resets_on_failure(self):
        ledger = FakeLedger()
        drive_ledger(ledger, updates=100)
        consume_gate(ledger, passed=True)
        curriculum = Curriculum(ledger)
        self.assertEqual(curriculum.consecutive_passes, 1)
        self.assertFalse(curriculum.can_advance)
        drive_ledger(ledger, updates=100)
        consume_gate(ledger, passed=False)
        self.assertEqual(curriculum.consecutive_passes, 0)
        self.assertFalse(curriculum.can_advance)
        drive_ledger(ledger, updates=100)
        consume_gate(ledger, passed=True)
        self.assertEqual(curriculum.consecutive_passes, 1)
        drive_ledger(ledger, updates=100)
        consume_gate(ledger, passed=True)
        self.assertEqual(curriculum.consecutive_passes, 2)
        self.assertTrue(curriculum.can_advance)

    def _at_s1_second_pass(self):
        ledger = FakeLedger()
        curriculum = Curriculum(ledger)
        while curriculum.stage != 'S1':
            drive_ledger(ledger)
            if curriculum.validation_due:
                consume_gate(ledger, passed=True)
            if curriculum.can_advance:
                pending = curriculum.pending_transition
                ledger.commit_transition(from_stage=pending['from_stage'],
                                         to_stage=pending['to_stage'],
                                         at_update=pending['at_update'],
                                         gate_streak=2)
        while not curriculum.needs_perturbed:
            drive_ledger(ledger)
            if curriculum.validation_due:
                consume_gate(ledger, passed=True)
        return ledger, curriculum

    def test_s1_perturbation_uses_the_second_pass_checkpoint(self):
        ledger, curriculum = self._at_s1_second_pass()
        self.assertEqual(curriculum.consecutive_passes, 2)
        self.assertEqual(curriculum.pending_transition['requires'], 'perturbed')
        checkpoint = curriculum.pending_transition['at_checkpoint']
        consume_gate(ledger, passed=True, kind='perturbed', stage='S1',
                     checkpoint_sha=checkpoint)
        self.assertFalse(curriculum.needs_perturbed)
        self.assertTrue(curriculum.can_advance)
        pending = curriculum.pending_transition
        ledger.commit_transition(from_stage=pending['from_stage'],
                                 to_stage=pending['to_stage'],
                                 at_update=pending['at_update'], gate_streak=2)
        self.assertEqual(ledger.curriculum['stage'], 'S2')

    def test_s1_perturbation_failure_keeps_s1_and_clears_the_streak(self):
        ledger, curriculum = self._at_s1_second_pass()
        checkpoint = curriculum.pending_transition['at_checkpoint']
        consume_gate(ledger, passed=False, kind='perturbed', stage='S1',
                     checkpoint_sha=checkpoint)
        self.assertEqual(curriculum.stage, 'S1')
        self.assertEqual(curriculum.consecutive_passes, 0)
        self.assertFalse(curriculum.needs_perturbed)
        self.assertFalse(curriculum.can_advance)
        drive_ledger(ledger)
        self.assertEqual(curriculum.stage_iteration, 401)

    def test_pilot_conflict_guard_only_fires_for_a_real_p1_transition(self):
        ledger = FakeLedger(mode='pilot')
        curriculum = Curriculum(ledger, mode='pilot', seed=42)
        ledger.curriculum.update(
            stage='P1', stage_iteration=250,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER}, F0=200, P1=250),
            streak=dict(gate=2, collapse=0), pending_transition=None)
        self.assertFalse(curriculum.pilot_conflict)
        # A terminal qualification is not a P1 -> P2 transition.
        ledger.curriculum['pending_transition'] = dict(
            from_stage='P1', to_stage=None, at_update=450, requires=None,
            at_checkpoint=digest('terminal'))
        self.assertFalse(curriculum.pilot_conflict)
        # Only a real pending transition out of a pilot P1 is a conflict.
        ledger.curriculum['pending_transition'] = dict(
            from_stage='P1', to_stage='P2', at_update=450, requires=None,
            at_checkpoint=digest('p2'))
        self.assertTrue(curriculum.pilot_conflict)
        # Formal mode never reports the pilot conflict.
        formal = Curriculum(FakeLedger(), mode='formal', seed=42)
        formal.ledger.curriculum.update(
            stage='P1', stage_iteration=250,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER}, F0=200, P1=250),
            streak=dict(gate=2, collapse=0),
            pending_transition=dict(from_stage='P1', to_stage='P2', at_update=450,
                                    requires=None, at_checkpoint=digest('p2')))
        self.assertFalse(formal.pilot_conflict)

    def test_pilot_verdict_requires_both_400_and_500(self):
        ledger = FakeLedger(mode='pilot')
        curriculum = Curriculum(ledger, mode='pilot', seed=42)
        # The durable pilot result rows are set directly to exercise the verdict
        # logic independently of the fixed-500 training loop.
        ledger.curriculum.update(
            stage='P1', stage_iteration=500,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER}, F0=200, P1=500),
            streak=dict(gate=2, collapse=0),
            pending_transition=dict(from_stage='P1', to_stage='P2', at_update=700,
                                    requires=None, at_checkpoint=digest('s5')))
        ledger.completed_ppo_updates = 700
        for iteration in (100, 200, 300, 400, 500):
            ledger.results.append(dict(
                task_key=digest(f'gate-{iteration}'), kind='validation', stage='P1',
                global_update=200 + iteration, outcome='passed',
                passed=iteration != 300, gate_streak=1, collapse_streak=0))
        verdict = curriculum.pilot_verdict()
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(verdict['gates'], {100: True, 200: True, 300: False,
                                            400: True, 500: True})
        ledger.results[-1]['passed'] = False
        self.assertEqual(curriculum.pilot_verdict()['status'], 'failed')
        self.assertIn('p1_gate_500_not_passed', curriculum.pilot_verdict()['reasons'])

    def test_pilot_f0_budget_exhaustion_fails_the_trial(self):
        ledger = FakeLedger(mode='pilot')
        drive_ledger(ledger, updates=2000)
        curriculum = Curriculum(ledger, mode='pilot')
        verdict = curriculum.pilot_verdict()
        self.assertEqual(verdict['status'], 'failed')
        self.assertIn('f0_budget_exhausted_without_gate', verdict['reasons'])

    def test_frozen_candidate_is_the_second_passing_s5_gate(self):
        ledger = FakeLedger()
        ledger.curriculum.update(
            stage='S5', stage_iteration=0,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER},
                       F0=200, P1=200, P2=200, P3=200, P4=200,
                       S1=400, S2=200, S3=200, S4=400))
        ledger.completed_ppo_updates = 2400
        curriculum = Curriculum(ledger)
        self.assertFalse(curriculum.complete)
        for index in (1, 2):
            drive_ledger(ledger)
            consume_gate(ledger, passed=True, checkpoint_sha=digest(f's5-gate-{index}'))
        self.assertTrue(curriculum.complete)
        self.assertEqual(curriculum.frozen_candidate_sha, digest('s5-gate-2'))
        self.assertIsNone(curriculum.final_result)


class PilotFixedTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_p1_continues_to_500_after_two_early_gate_passes(self):
        ledger = FakeLedger(mode='pilot')
        curriculum = Curriculum(ledger, mode='pilot', seed=42)
        while ledger.curriculum['stage'] == 'F0':
            drive_ledger(ledger)
            if curriculum.validation_due:
                consume_gate(ledger, passed=True)
            if curriculum.can_advance:
                pending = curriculum.pending_transition
                ledger.commit_transition(from_stage=pending['from_stage'],
                                         to_stage=pending['to_stage'],
                                         at_update=pending['at_update'],
                                         gate_streak=2)
        self.assertEqual(curriculum.stage, 'P1')
        for _ in range(500):
            drive_ledger(ledger)
            if curriculum.validation_due:
                # Early gates pass twice; the 500-update trial must still run on.
                consume_gate(ledger,
                             passed=curriculum.stage_iteration != 300)
            self.assertFalse(curriculum.pilot_conflict)
        self.assertEqual(curriculum.stage, 'P1')
        self.assertEqual(curriculum.consumed['P1'], 500)
        self.assertEqual(curriculum.iteration, 700)
        self.assertFalse(curriculum.can_advance)
        self.assertIsNone(curriculum.pending_transition)
        verdict = curriculum.pilot_verdict()
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(verdict['gates'], {100: True, 200: True, 300: False,
                                            400: True, 500: True})
        self.assertEqual(verdict['p1_consumed'], 500)


# ----------------------------------------------------------------- diagnostics
class DiagnosticTests(unittest.TestCase):
    def test_twelve_of_fifteen_stalled_at_eighty_percent_is_a_hit(self):
        summary = diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=80,
                                     first_step_absent=12)
        decision = diagnostic_decision(summary, stairs=True, stage_update=100)
        self.assertTrue(decision['collapse_hit'])
        self.assertEqual(decision['stalled_cases'], 12)
        self.assertEqual(decision['first_step_absent_cases'], 12)
        below = diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=79,
                                   first_step_absent=12)
        self.assertFalse(diagnostic_decision(below, stairs=True,
                                             stage_update=100)['collapse_hit'])
        eleven = diagnostic_summary(stalled_cases=11, samples=100, stalled_samples=100,
                                    first_step_absent=15)
        self.assertFalse(diagnostic_decision(eleven, stairs=True,
                                             stage_update=100)['collapse_hit'])

    def test_stairs_require_twelve_cases_without_a_first_step_footfall(self):
        summary = diagnostic_summary(stalled_cases=15, samples=100, stalled_samples=100,
                                     first_step_absent=11)
        self.assertFalse(diagnostic_decision(summary, stairs=True,
                                             stage_update=100)['collapse_hit'])
        self.assertTrue(diagnostic_decision(summary, stairs=False,
                                            stage_update=100)['collapse_hit'])

    def test_collapse_rule_starts_at_stage_update_100(self):
        summary = diagnostic_summary(stalled_cases=15, samples=100, stalled_samples=100,
                                     first_step_absent=15)
        decision = diagnostic_decision(summary, stairs=True, stage_update=99)
        self.assertFalse(decision['eligible'])
        self.assertFalse(decision['collapse_hit'])
        self.assertTrue(diagnostic_decision(summary, stairs=True,
                                            stage_update=100)['collapse_hit'])

    def test_f0_diagnostic_does_not_require_first_step_evidence(self):
        summary = diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90,
                                     first_step_absent=0)
        for row in summary['per_case']:
            row.pop('first_step_acquired')
        decision = diagnostic_decision(summary, stairs=False, stage_update=100)
        self.assertTrue(decision['collapse_hit'])
        self.assertEqual(decision['first_step_absent_cases'], 0)
        below = diagnostic_summary(stalled_cases=11, samples=100, stalled_samples=90,
                                   first_step_absent=0)
        for row in below['per_case']:
            row.pop('first_step_acquired')
        self.assertFalse(diagnostic_decision(below, stairs=False,
                                             stage_update=100)['collapse_hit'])
        # The F0 exception is only for first_step: the denominator stays mandatory.
        missing_denominator = diagnostic_summary(stalled_cases=12, samples=100,
                                                 stalled_samples=90)
        for row in missing_denominator['per_case']:
            row.pop('first_step_acquired')
        missing_denominator['per_case'][0]['samples'] = None
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision(missing_denominator, stairs=False, stage_update=100)

    def test_stairs_diagnostic_still_requires_first_step_evidence(self):
        summary = diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90,
                                     first_step_absent=12)
        summary['per_case'][0]['first_step_acquired'] = None
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision(summary, stairs=True, stage_update=100)
        summary['per_case'][0].pop('first_step_acquired')
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision(summary, stairs=True, stage_update=100)

    def test_zero_denominator_or_wrong_case_count_is_invalid_evidence(self):
        summary = diagnostic_summary()
        summary['per_case'][0]['samples'] = 0
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision(summary, stairs=True, stage_update=100)
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision({'per_case': []}, stairs=True, stage_update=100)
        bad = diagnostic_summary()
        bad['per_case'][0]['stall_samples'] = 10 ** 6
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision(bad, stairs=True, stage_update=100)
        missing = diagnostic_summary()
        del missing['per_case'][3]['first_step_acquired']
        with self.assertRaises(DiagnosticEvidenceError):
            diagnostic_decision(missing, stairs=True, stage_update=100)


# --------------------------------------------------------------------- runner
class RunnerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in frozen_source_identity():
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fx = build(self.root)

    def test_diagnostic_runs_before_the_gate_in_the_same_update(self):
        run_updates(self.fx, 100)
        calls = self.fx['evaluation'].calls
        self.assertEqual([call['kind'] for call in calls if call['update'] == 50],
                         ['diagnostic'])
        self.assertEqual([call['kind'] for call in calls if call['update'] == 100],
                         ['diagnostic', 'validation'])
        self.assertEqual(self.fx['curriculum'].consecutive_passes, 1)
        for update in (50, 100):
            self.assertTrue((self.root / 'checkpoints' /
                             f'update_{update:06d}.pt').is_file())
        ledger = self.fx['ledger']
        self.assertEqual(ledger.completed_ppo_updates, 100)
        self.assertEqual(ledger.control_samples, 48 * 8 * 100)
        self.assertEqual(ledger.physics_substeps, 4 * 48 * 8 * 100)

    def test_diagnostics_never_reset_or_rebind_the_training_environment(self):
        run_updates(self.fx, 52)
        env = self.fx['env']
        self.assertEqual(env.resets, 1)  # only the explicit initial new episode
        self.assertEqual(env.steps, 52 * 48)
        self.assertEqual({call[0] for call in env.stage_calls}, {'F0'})
        # An S5 0.2 m/s episode from x=-1 to x=8.8 needs >= 49 s while 50 updates
        # are 48 s: the episode must survive the 50-update diagnostic boundary.
        self.assertGreaterEqual(52 * 48 * spec.CONTROL_DT, 49.0)
        self.assertLess(50 * 48 * spec.CONTROL_DT, 49.0)

    def test_evaluation_restores_every_training_rng_and_keeps_state(self):
        run_updates(self.fx, 49)
        runner = self.fx['runner']
        runner.collect()
        runner.update()
        before = recovery.capture_rng(self.fx['env'])
        learning = deepcopy(runner.learning)
        steps = self.fx['env'].steps
        status = runner.boundary()  # the 50-update diagnostic happens here
        self.assertEqual(status['status'], 'continue')
        self.assertTrue(any(call['kind'] == 'diagnostic'
                            for call in self.fx['evaluation'].calls))
        after = recovery.capture_rng(self.fx['env'])
        self.assertEqual(after['python'], before['python'])
        np.testing.assert_equal(after['numpy'], before['numpy'])
        self.assertTrue(torch.equal(after['torch'], before['torch']))
        self.assertTrue(torch.equal(after['env'], before['env']))
        self.assertEqual(runner.learning, learning)
        self.assertEqual(self.fx['env'].steps, steps)
        self.assertEqual(self.fx['env'].resets, 1)
        self.assertTrue(all(model is not self.fx['model']
                            for model in self.fx['evaluation'].seen_models))
        self.assertEqual(self.fx['evaluation'].calls[-1]['mode'], 'formal')
        self.assertEqual(self.fx['evaluation'].calls[-1]['seed'], 42)

    def test_behavior_collapse_at_100_and_150_stops_before_any_later_gate(self):
        self.fx['evaluation'].diagnostic_plan = {
            100: diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90,
                                    first_step_absent=12),
            150: diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90,
                                    first_step_absent=12),
        }
        status = run_updates(self.fx, 150)
        self.assertEqual(status['status'], 'behavior_collapse')
        self.assertEqual(status['collapse_streak'], COLLAPSE_CONSECUTIVE)
        self.assertEqual(self.fx['runner'].terminal, 'behavior_collapse')
        kinds = [(call['kind'], call['update']) for call in self.fx['evaluation'].calls]
        self.assertIn(('validation', 100), kinds)
        self.assertNotIn(('validation', 200), kinds)
        self.assertTrue((self.root / 'checkpoints' /
                         'behavior_collapse_000150.pt').is_file())
        with self.assertRaises(RuntimeError):
            self.fx['runner'].train_step()

    def test_diagnostic_payload_accepts_every_committed_spelling(self):
        runner = self.fx['runner']
        rows = diagnostic_summary(stalled_cases=12)['per_case']
        for report in ({'per_case': rows},
                       {'diagnostic_cases': rows},
                       {'diagnostic': {'per_case': rows}},
                       {'diagnostic': {'cases': rows}}):
            self.assertEqual(runner._diagnostic_payload(report)['per_case'], rows)
        for report in ({}, {'diagnostic_aggregate': {'stalls': 12}},
                       {'diagnostic': {}}):
            with self.assertRaises(DiagnosticEvidenceError):
                runner._diagnostic_payload(report)

    def test_f0_diagnostic_without_first_step_does_not_error(self):
        # The real F0 report path: per-case evidence under diagnostic_cases with no
        # first_step_acquired field (SDS 6.3's F0 exception).
        summary = diagnostic_summary(stalled_cases=0)
        for row in summary['per_case']:
            row.pop('first_step_acquired')
        self.fx['evaluation'].diagnostic_plan = {
            50: {'diagnostic_cases': summary['per_case'],
                 'diagnostic_aggregate': {'stalled_cases': 0}}}
        status = run_updates(self.fx, 50)
        self.assertEqual(status['status'], 'continue')
        self.assertEqual(self.fx['runner'].terminal, None)
        kinds = [call['kind'] for call in self.fx['evaluation'].calls
                 if call['update'] == 50]
        self.assertEqual(kinds, ['diagnostic'])
        ledger = self.fx['ledger']
        diagnostic_key = next(key for key, record in ledger.tasks.items()
                              if '|diagnostic|' in key)
        self.assertEqual(ledger.tasks[diagnostic_key]['state'], 'consumed')

    def test_stairs_diagnostic_without_first_step_is_an_evaluation_error(self):
        runner = self.fx['runner']
        runner.curriculum.ledger.curriculum.update(
            stage='S1', stage_iteration=100,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER}, F0=200, P1=200,
                       P2=200, P3=200, P4=200, S1=100),
            streak=dict(gate=0, collapse=0), pending_transition=None)
        rows = diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90)[
            'per_case']
        for row in rows:
            row.pop('first_step_acquired')
        with self.assertRaises(DiagnosticEvidenceError):
            runner._interpret('diagnostic', {'diagnostic_cases': rows}, stage='S1',
                              update=500)
        rows = diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90,
                                  first_step_absent=12)['per_case']
        interpretation = runner._interpret('diagnostic', {'diagnostic_cases': rows},
                                           stage='S1', update=500)
        self.assertTrue(interpretation['collapse_hit'])

    def test_a_single_miss_clears_the_collapse_streak(self):
        self.fx['evaluation'].diagnostic_plan = {
            100: diagnostic_summary(stalled_cases=12, samples=100, stalled_samples=90,
                                    first_step_absent=12),
            150: diagnostic_summary(stalled_cases=11, samples=100, stalled_samples=90,
                                    first_step_absent=11),
        }
        status = run_updates(self.fx, 150)
        self.assertEqual(status['status'], 'continue')
        self.assertEqual(self.fx['curriculum'].collapse_streak, 0)

    def test_invalid_diagnostic_evidence_is_an_evaluation_error_not_a_non_hit(self):
        summary = diagnostic_summary()
        summary['per_case'][0]['samples'] = 0
        self.fx['evaluation'].diagnostic_plan = {100: summary}
        status = run_updates(self.fx, 100)
        self.assertEqual(status['status'], 'evaluation_error')
        self.assertEqual(self.fx['curriculum'].collapse_streak, 0)
        kinds = [call['kind'] for call in self.fx['evaluation'].calls
                 if call['update'] == 100]
        self.assertEqual(kinds, ['diagnostic'])  # the gate was never reached
        self.assertEqual(self.fx['runner'].terminal, 'evaluation_error')

    def test_evaluation_service_failure_is_a_terminal_evaluation_error(self):
        self.fx['evaluation'].raise_on = {'diagnostic'}
        status = run_updates(self.fx, 50)
        self.assertEqual(status['status'], 'evaluation_error')
        self.assertIn('simulated evaluation failure', status['error'])

    def test_stage_transition_resets_all_envs_and_keeps_learning_state(self):
        run_updates(self.fx, 200)
        self.assertEqual(self.fx['curriculum'].stage, 'P1')
        self.assertEqual(self.fx['env'].stage, 'P1')
        # initial episode + the set_stage binding reset + the explicit transition reset
        self.assertEqual(self.fx['env'].resets, 3)
        self.assertEqual(self.fx['env'].stage_resets, 1)
        self.assertEqual(self.fx['env'].stage_calls[-1], ('P1', 0, 'F0'))
        self.assertEqual(self.fx['curriculum'].stage_iteration, 0)
        # The transition reset must also refresh the acting observation.
        self.assertIsInstance(self.fx['runner'].obs, dict)
        self.assertIs(self.fx['ppo'].obs, self.fx['runner'].obs)
        learning = self.fx['runner'].learning
        self.assertEqual(learning['update'], 200)
        self.assertEqual(learning['optimizer_step'], 4000)
        for name in ('stage_transition_pre_reset', 'stage_transition_committed',
                     'stage_start'):
            self.assertTrue((self.root / 'checkpoints' /
                             f'{name}_000200.pt').is_file())
        # The P1 gate must also report the same-checkpoint F0 retention test.
        run_updates(self.fx, 100)
        kinds = [call['kind'] for call in self.fx['evaluation'].calls
                 if call['update'] == 300]
        self.assertEqual(kinds, ['diagnostic', 'flat_retention', 'validation'])

    def test_control_and_update_reject_unknown_or_terminal_states(self):
        runner = self.fx['runner']
        runner.terminal = 'behavior_collapse'
        with self.assertRaises(RuntimeError):
            runner.collect()
        with self.assertRaises(RuntimeError):
            runner.update()
        with self.assertRaises(RuntimeError):
            runner.boundary()

    def test_pilot_conflict_is_reported_not_silently_ignored(self):
        fixture = build(self.root / 'pilot', mode='pilot')
        ledger = fixture['ledger']
        ledger.curriculum.update(
            stage='P1', stage_iteration=201,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER}, F0=200, P1=201),
            streak=dict(gate=2, collapse=0),
            pending_transition=dict(from_stage='P1', to_stage='P2', at_update=401,
                                    requires=None, at_checkpoint=digest('x')))
        ledger.completed_ppo_updates = 401
        with self.assertRaises(PilotLedgerConflict):
            fixture['runner'].collect()


class CheckpointResumeTests(unittest.TestCase):
    """Schema-6 save/resume on the in-memory ledger mirror."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in frozen_source_identity():
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_resume_restores_rng_after_binding_and_resets_exactly_once(self):
        fixture = build(self.root)
        run_updates(fixture, 3)
        checkpoint = fixture['runner'].save_checkpoint('manual')
        state = preload_checkpoint(self.root / 'checkpoints' / 'manual_000003.pt',
                                   checkpoint, mode='formal')
        self.assertEqual(state['resume_mode'], 'new_episode')
        self.assertEqual(state['learning']['control_samples'], 48 * 8 * 3)
        self.assertEqual(state['learning']['physics_substeps'], 4 * 48 * 8 * 3)
        self.assertEqual(state['curriculum']['stage_iteration'], 3)

        def resumed_process():
            env = FakeEnv(8)
            model = TinyModel()
            fake = FakeEvaluation()
            ledger = FakeLedger(run_id='unit-formal-42')
            ledger.curriculum = deepcopy(state['curriculum'])
            ledger.completed_ppo_updates = 3
            runner = V05Runner(model=model, env=env, ppo=FakePPO(model, env),
                               curriculum=Curriculum(ledger, seed=42), ledger=ledger,
                               device='cpu', seed=42, output_dir=self.root,
                               evaluation=fake, make_cases=fake.make_cases,
                               eval_num_envs=8)
            observed = {}
            original = v05_runner.restore_rng

            def spy(state_, env_):
                observed['resets_at_restore'] = env_.resets
                observed['stage_at_restore'] = (env_.stage, env_.stage_iteration)
                return original(state_, env_)

            with patch.object(v05_runner, 'restore_rng', side_effect=spy):
                result = runner.resume(self.root / 'checkpoints' / 'manual_000003.pt',
                                       checkpoint)
            return runner, env, result, observed

        runner, env, result, observed = resumed_process()
        self.assertEqual(result['status'], 'resumed')
        self.assertEqual(observed['resets_at_restore'], 0)
        self.assertEqual(observed['stage_at_restore'], ('F0', 3))
        self.assertEqual(env.resets, 1)
        self.assertEqual(runner.learning['update'], 3)
        first = recovery.capture_rng(env)
        # A second process resuming the same checkpoint must reach the same RNG
        # state after its single new-episode reset (SDS 7.3).
        runner2, env2, result2, observed2 = resumed_process()
        self.assertEqual(result2['status'], 'resumed')
        self.assertEqual(observed2['resets_at_restore'], 0)
        self.assertEqual(env2.resets, 1)
        second = recovery.capture_rng(env2)
        self.assertEqual(first['python'], second['python'])
        np.testing.assert_equal(first['numpy'], second['numpy'])
        self.assertTrue(torch.equal(first['torch'], second['torch']))
        self.assertTrue(torch.equal(first['env'], second['env']))
        self.assertTrue(torch.equal(runner2.model.weight, runner.model.weight))

    def test_resume_adopts_a_same_update_ledger_gate_consumed_after_the_checkpoint(self):
        fixture = build(self.root)
        run_updates(fixture, 1)
        checkpoint = fixture['runner'].save_checkpoint('manual')
        ledger = fixture['ledger']
        # A gate is consumed after the checkpoint was written (SDS 7.2: the state
        # may lag the ledger only within the same complete update boundary).
        consume_gate(ledger, passed=True)
        self.assertEqual(ledger.gate_streak, 1)
        env = FakeEnv(8)
        model = TinyModel()
        fake = FakeEvaluation()
        view = FakeLedger(run_id='unit-formal-42')
        view.curriculum = deepcopy(ledger.curriculum)
        view.completed_ppo_updates = ledger.completed_ppo_updates
        runner = V05Runner(model=model, env=env, ppo=FakePPO(model, env),
                           curriculum=Curriculum(view, seed=42), ledger=view,
                           device='cpu', seed=42, output_dir=self.root,
                           evaluation=fake, make_cases=fake.make_cases, eval_num_envs=8)
        result = runner.resume(self.root / 'checkpoints' / 'manual_000001.pt', checkpoint)
        self.assertEqual(result['status'], 'resumed')
        self.assertEqual(runner.curriculum.consecutive_passes, 1)
        self.assertEqual(env.resets, 1)

    def _pending_resume(self, env_factory=FakeEnv):
        """A valid schema-6 checkpoint whose ledger has an uncommitted P1 -> P2 transition."""
        source = build(self.root / 'source')
        run_updates(source, 3)
        pending = dict(from_stage='P1', to_stage='P2', at_update=3, requires=None,
                       at_checkpoint=digest('p1-gate'))
        curriculum_state = dict(
            stage='P1', stage_iteration=3,
            spent=dict({stage: 0 for stage in spec.STAGE_ORDER}, P1=3),
            streak=dict(gate=2, collapse=0), pending_transition=pending)
        task_state = dict(tasks={}, results=[], collapse_streak=0, gate_streak=2,
                          anchor=dict(seq=0, digest=digest('anchor')),
                          qualified_update=None, pending_transition=pending)
        state = recovery.build_checkpoint(
            model_state=source['model'].state_dict(),
            optimizer_state=source['ppo'].optimizer.state_dict(),
            normalizer={}, learning=source['runner'].learning,
            rng=recovery.capture_rng(source['env']), curriculum=curriculum_state,
            ledger=task_state,
            evaluation=dict(consumed=[], latest_diagnostic_update=None),
            source=recovery.source_block(mode='formal', seed=42),
            mode='formal', seed=42, parameters=list(source['model'].parameters()))
        path = self.root / 'pending_transition.pt'
        recovery.atomic_torch_save(path, state)
        digest_value = recovery.sha256(path)
        env = env_factory(8)
        model = TinyModel()
        fake = FakeEvaluation()
        ledger = FakeLedger(run_id='unit-formal-42')
        ledger.curriculum = deepcopy(curriculum_state)
        ledger.completed_ppo_updates = 3
        runner = V05Runner(model=model, env=env, ppo=FakePPO(model, env),
                           curriculum=Curriculum(ledger, mode='formal', seed=42),
                           ledger=ledger, device='cpu', seed=42, output_dir=self.root,
                           evaluation=fake, make_cases=fake.make_cases, eval_num_envs=8)
        return runner, env, path, digest_value

    def test_pending_transition_resume_resets_once_after_the_rng_restore(self):
        runner, env, path, digest_value = self._pending_resume()
        observed = {}
        original = v05_runner.restore_rng

        def spy(state_, env_):
            restored = original(state_, env_)
            observed['resets_after_restore'] = env_.resets
            observed['stage_after_restore'] = env_.stage
            return restored

        with patch.object(v05_runner, 'restore_rng', side_effect=spy):
            result = runner.resume(path, digest_value)
        self.assertEqual(result['status'], 'resumed')
        # The binding reset happens before the RNG restore; exactly one explicit
        # new-episode reset follows it (SDS 7.3).
        self.assertEqual(observed['stage_after_restore'], 'P2')
        self.assertEqual(observed['resets_after_restore'], 1)
        self.assertEqual(env.resets, observed['resets_after_restore'] + 1)
        self.assertEqual(env.resets, 2)
        self.assertEqual(env.stage_resets, 1)
        self.assertIn(('P2', 0, 'P1'), env.stage_calls)
        self.assertEqual(runner.curriculum.stage, 'P2')
        self.assertEqual(runner.curriculum.stage_iteration, 0)
        self.assertIsNone(runner.curriculum.pending_transition)

    def test_binding_reset_after_the_rng_restore_is_detected(self):
        class AlwaysResettingEnv(FakeEnv):
            def set_stage(self, stage, stage_iteration=0, previous_stage=None):
                super().set_stage(stage, stage_iteration, previous_stage)
                # A non-compliant bind that resets even when the stage is unchanged.
                self._reset()

        runner, env, path, digest_value = self._pending_resume(
            env_factory=AlwaysResettingEnv)
        with self.assertRaises(RuntimeError) as caught:
            runner.resume(path, digest_value)
        self.assertIn('exactly one explicit new-episode reset', str(caught.exception))

    def test_schema6_rejects_legacy_or_mutated_envelopes(self):
        fixture = build(self.root)
        run_updates(fixture, 1)
        fixture['runner'].save_checkpoint('manual')
        path = self.root / 'checkpoints' / 'manual_000001.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(set(state), {'schema', 'identity', 'contract_sha256', 'model',
                                      'optimizer', 'normalizer', 'learning', 'rng',
                                      'curriculum', 'ledger', 'evaluation', 'source',
                                      'recipe', 'resume_mode'})
        mutations = {
            'schema': lambda value: value.update(schema=5),
            'learning': lambda value: value['learning'].update(control_samples=1),
            'physics': lambda value: value['learning'].update(physics_substeps=3),
            'adam': lambda value: value['learning'].update(optimizer_step=19),
            'resume': lambda value: value.update(resume_mode='latest'),
            'identity': lambda value: value['identity'].update(protocol='v04'),
            'rng': lambda value: value['rng'].update(torch=torch.ones(3, dtype=torch.uint8)),
            'extra': lambda value: value.update(unexpected=1),
            'curriculum': lambda value: value['curriculum'].update(stage_iteration=9),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                bad = deepcopy(state)
                mutate(bad)
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    preload_checkpoint(path, spec.sha256(path), mode='formal')
        torch.save(state, path)
        with self.assertRaises(ValueError):
            preload_checkpoint(path, 'f' * 64, mode='formal')


class RealLedgerIntegrationTests(unittest.TestCase):
    """The runner against the real AttemptLedger, real PPO and real schema-6."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in frozen_source_identity():
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fsync = patch('os.fsync')
        self.fsync.start()
        self.addCleanup(self.fsync.stop)

    def _runner(self, *, output, run_id='real', mode='formal', seed=42):
        env = FakeEnv(8)
        model = TinyModel()
        ppo = FakePPO(model, env)
        ledger = AttemptLedger(output / 'ledger', run_id=run_id, mode=mode)
        fake = FakeEvaluation()
        runner = V05Runner(model=model, env=env, ppo=ppo,
                           curriculum=Curriculum(ledger, mode=mode, seed=seed),
                           ledger=ledger, device='cpu', seed=seed, output_dir=output,
                           evaluation=fake, make_cases=fake.make_cases, eval_num_envs=8)
        runner.reset_episode()
        return runner, ledger, fake, env

    def test_real_ledger_accepts_the_runner_transactions(self):
        runner, ledger, fake, env = self._runner(output=self.root)
        fixture = dict(runner=runner, env=env, ledger=ledger)
        run_updates(fixture, 1)
        self.assertEqual(ledger.completed_ppo_updates, 1)
        self.assertEqual(ledger.control_samples, 48 * 8)
        self.assertEqual(ledger.physics_substeps, 4 * 48 * 8)
        digest_value = runner.save_checkpoint('manual')
        self.assertEqual(ledger.checkpoints[-1]['sha256'], digest_value)
        outcome = runner.run_diagnostic(digest_value)
        self.assertEqual(outcome['outcome'], 'clear')
        key = next(iter(ledger.tasks))
        self.assertEqual(ledger.tasks[key]['state'], 'consumed')
        self.assertEqual(ledger.collapse_streak, 0)

    def test_real_ledger_replays_a_planned_task_after_resume(self):
        runner, ledger, fake, env = self._runner(output=self.root)
        fixture = dict(runner=runner, env=env, ledger=ledger)
        run_updates(fixture, 1)
        digest_value = runner.save_checkpoint('manual')
        cases = fake.make_cases('F0', 'diagnostic', spec.DIAGNOSTIC_SEED_OFFSET + 42)
        key = task_key(run_id='real', mode='formal', stage='F0', global_update=1,
                       kind='diagnostic', checkpoint_sha256=digest_value,
                       case_sha256=cases.sha256)
        ledger.plan_evaluation(run_id='real', mode='formal', stage='F0',
                               global_update=1, kind='diagnostic',
                               checkpoint_sha256=digest_value,
                               case_sha256=cases.sha256)
        self.assertEqual(ledger.tasks[key]['state'], 'planned')
        env2 = FakeEnv(8)
        model2 = TinyModel()
        fake2 = FakeEvaluation()
        runner2 = V05Runner(
            model=model2, env=env2, ppo=FakePPO(model2, env2),
            curriculum=Curriculum(ledger, mode='formal', seed=42), ledger=ledger,
            device='cpu', seed=42, output_dir=self.root, evaluation=fake2,
            make_cases=fake2.make_cases, eval_num_envs=8)
        result = runner2.resume(self.root / 'checkpoints' / 'manual_000001.pt',
                                digest_value)
        self.assertEqual(len(result['pending_evaluations']), 1)
        self.assertEqual(result['pending_evaluations'][0]['outcome'], 'clear')
        self.assertEqual(ledger.tasks[key]['state'], 'consumed')
        self.assertEqual([call['kind'] for call in fake2.calls], ['diagnostic'])

    def test_real_ppo_update_matches_the_48_20_4x_contract(self):
        from src.tasks.velocity.v05.ppo import PPO as RealPPO
        from src.tasks.velocity.v05.spec import (PPO_CONTRACT, PPO_EPOCHS,
                                                 PPO_MINIBATCHES, PPO_ROLLOUT)

        class RealModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.std = torch.nn.Parameter(torch.full((12,), spec.INITIAL_STD))
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def action_distribution(self, obs):
                mean = torch.zeros(len(obs['proprio']), 12) + self.value * 0
                return torch.distributions.Normal(mean, self.std)

            def evaluate(self, obs):
                return self.value.expand(len(obs['proprio']), 1)

            def update_critic_statistics(self, obs):
                return None

            @torch.no_grad()
            def constrain_std(self):
                self.std.clamp_(spec.STD_MIN, spec.STD_MAX)

        env = FakeEnv(8)
        model = RealModel()
        ppo = RealPPO(model, env)
        ledger = AttemptLedger(self.root / 'ledger', run_id='ppo', mode='formal')
        fake = FakeEvaluation()
        runner = V05Runner(model=model, env=env, ppo=ppo,
                           curriculum=Curriculum(ledger, mode='formal', seed=42),
                           ledger=ledger, device='cpu', seed=42, output_dir=self.root,
                           evaluation=fake, make_cases=fake.make_cases, eval_num_envs=8)
        runner.reset_episode()
        status = runner.train_step()
        self.assertEqual(status['status'], 'continue')
        self.assertEqual(runner.learning,
                         dict(update=1, optimizer_step=20,
                              control_samples=PPO_ROLLOUT * 8,
                              physics_substeps=4 * PPO_ROLLOUT * 8))
        self.assertEqual(ledger.control_samples, PPO_ROLLOUT * 8)
        self.assertEqual(ledger.physics_substeps, 4 * PPO_ROLLOUT * 8)
        self.assertEqual(PPO_CONTRACT['control_samples_per_update'], 48 * 512)
        self.assertEqual(PPO_EPOCHS * PPO_MINIBATCHES, 20)
        self.assertTrue((model.std >= spec.STD_MIN - 1e-6).all())
        self.assertTrue((model.std <= spec.STD_MAX + 1e-6).all())
        self.assertEqual(PPO_ROLLOUT * 512, 24576)
        self.assertEqual(4 * PPO_ROLLOUT * 512, 98304)


class RealEnvironmentRegressionTests(unittest.TestCase):
    """The real V05Env reset returns ``(obs, info)``; the runner must unpack it.

    No GPU marker: these tests construct the MuJoCo/MJWarp stair scene on CPU and
    guard the actual failure (``runner.obs`` becoming a tuple and the next PPO
    collect crashing on ``self.obs.items()``).
    """
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from src.tasks.velocity.v05.env import V05Env
        from src.tasks.velocity.v05.model import V05ActorCritic
        from src.tasks.velocity.v05.ppo import PPO, PPOConfig
        cls.V05Env = V05Env
        cls.V05ActorCritic = V05ActorCritic
        cls.PPO = PPO
        cls.PPOConfig = PPOConfig

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in (patch.object(recovery, 'contract_sha_for',
                                     return_value='a' * 64),
                        patch.object(recovery, 'recipe_config',
                                     return_value={'recipe': '0.5-r1'})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.env = self.V05Env(num_envs=8, device='cpu', seed=42, stage='F0')
        self.addCleanup(self.env.close)
        self.model = self.V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42,
                                                  device='cpu')

    def _runner(self, ledger, *, env, model, ppo):
        return V05Runner(model=model, env=env, ppo=ppo,
                         curriculum=Curriculum(ledger, mode='formal', seed=42),
                         ledger=ledger, device='cpu', seed=42, output_dir=self.root,
                         save_interval=10 ** 9, diagnostic_interval=10 ** 9)

    def test_real_env_reset_episode_then_collect(self):
        ledger = AttemptLedger(self.root / 'ledger', run_id='env-regression',
                               mode='formal')
        ppo = self.PPO(self.model, self.env, self.PPOConfig())
        runner = self._runner(ledger, env=self.env, model=self.model, ppo=ppo)
        obs = runner.reset_episode()
        self.assertIsInstance(runner.obs, dict)
        self.assertEqual(set(obs), {'proprio', 'history', 'heightmap', 'privileged'})
        self.assertEqual(tuple(obs['proprio'].shape), (8, 45))
        self.assertEqual(tuple(obs['history'].shape), (8, 5, 45))
        self.assertEqual(tuple(obs['heightmap'].shape), (8, 3, 48, 32))
        self.assertEqual(tuple(obs['privileged'].shape), (8, 35))
        self.assertIs(runner.ppo.obs, runner.obs)
        runner.collect()  # the exact call that crashed on the real env
        self.assertEqual(ledger.sampled_control_samples, 48 * 8)
        self.assertIsInstance(runner.ppo.obs, dict)
        self.assertEqual(set(runner.ppo.obs), set(obs))

    def test_real_env_resume_then_collect(self):
        ledger = AttemptLedger(self.root / 'ledger', run_id='env-resume', mode='formal')
        ppo = self.PPO(self.model, self.env, self.PPOConfig())
        runner = self._runner(ledger, env=self.env, model=self.model, ppo=ppo)
        runner.reset_episode()
        digest_value = runner.save_checkpoint('regression')  # zero-update boundary
        checkpoint = self.root / 'checkpoints' / 'regression_000000.pt'
        env2 = self.V05Env(num_envs=8, device='cpu', seed=43, stage='F0')
        self.addCleanup(env2.close)
        model2 = self.V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 43,
                                              device='cpu')
        ppo2 = self.PPO(model2, env2, self.PPOConfig())
        ledger2 = AttemptLedger(self.root / 'ledger', run_id='env-resume', mode='formal')
        runner2 = self._runner(ledger2, env=env2, model=model2, ppo=ppo2)
        result = runner2.resume(checkpoint, digest_value)
        self.assertEqual(result['status'], 'resumed')
        self.assertIsInstance(runner2.obs, dict)
        self.assertEqual(set(runner2.obs), {'proprio', 'history', 'heightmap',
                                            'privileged'})
        self.assertIs(runner2.ppo.obs, runner2.obs)
        runner2.collect()  # the resumed process must collect too
        self.assertEqual(ledger2.sampled_control_samples, 48 * 8)
        self.assertIsInstance(runner2.ppo.obs, dict)




if __name__ == '__main__':
    unittest.main()
