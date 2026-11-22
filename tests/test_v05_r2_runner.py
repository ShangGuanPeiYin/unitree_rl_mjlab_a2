"""CPU-only v0.5-r2 training-loop tests (resident worker, cadence, budgets, route).

The simulated environment is never constructed here: ``env.py`` is GPU-only by
design (SDS 9.2), so this file uses an observable route stand-in with the same
five-key observation / pre-reset terminal / per-episode route evidence contract.
The durable ledger is a real ``AttemptLedger`` where the test needs checkpoint
and resume semantics, and an in-memory mirror where a 150-update cadence run
would otherwise fsync hundreds of thousands of events.

r2 items under test:

* every adopted observation contains ``route_error`` and a missing key is a hard
  error (never zero filled);
* the rollout storage carries ``[48*B,3]`` and the per-episode route evidence is
  audited;
* save every 100 / diagnostic every 50, collapse, pilot P1 fixed 500 and the
  2500 pilot cap all come from the same schedule;
* resume performs exactly one explicit new-episode reset after the RNG restore.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from src.tasks.velocity.v05_r2 import recovery, runner as runner_module, spec
from src.tasks.velocity.v05_r2.recovery import (AttemptLedger, learning_state, task_key,
                                                updated_curriculum)
from src.tasks.velocity.v05_r2.runner import (COLLAPSE_CONSECUTIVE, Curriculum,
                                              V05Runner, route_diagnostics,
                                              route_episode_audit)

SPEEDS = spec.SPEEDS
ROUTE_FIELDS = ('route_final', 'route_initial', 'route_max_abs_y',
                'route_max_abs_heading_deg')
DIGEST = 'a' * 64


def digest(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


# ------------------------------------------------------------------- fixtures
def gate_summary(*, passed=True, total=20, passed_count=None):
    count = (total if passed else 0) if passed_count is None else passed_count
    return {'per_speed': [dict(c=speed, total=total, case_pass_count=count,
                               speed_pass_count=count, climb_success_count=count,
                               failure_count=total - count) for speed in SPEEDS],
            'route': dict(abs_y_max=.2, heading_deg_max=3., nonfinite_count=0)}


def diagnostic_summary(*, stalled_cases=0, samples=100, stalled_samples=None,
                       first_step_absent=0):
    rows = []
    for speed in SPEEDS:
        for _ in range(5):
            case = len(rows)
            stalled = case < stalled_cases
            rows.append(dict(
                c=speed, samples=samples,
                stall_samples=(stalled_samples if stalled_samples is not None
                               else samples) if stalled else 0,
                first_step_acquired=case >= first_step_absent))
    return {'per_case': rows,
            'route': dict(abs_y_max=.3, heading_deg_max=4., nonfinite_count=0)}


class RouteEnv:
    """Observable r2 stand-in: five keys, per-step route, route episode evidence."""
    device = 'cpu'

    def __init__(self, num_envs=8, seed=11, *, with_route=True,
                 with_episode_route=True):
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
        self.with_route = with_route
        self.with_episode_route = with_episode_route
        self.last_reset_rng_state = None
        self.reset_route = None

    def set_stage(self, stage, stage_iteration=0, previous_stage=None):
        changed = stage != self.stage
        self.stage = stage
        self.stage_iteration = int(stage_iteration)
        self.stage_calls.append((stage, int(stage_iteration), previous_stage))
        if changed:
            self.stage_resets += 1
            self._reset()
        return None

    def _route(self, value):
        route = torch.zeros(self.num_envs, 3)
        route[:, 0] = value + .01 * torch.arange(self.num_envs)
        route[:, 1] = 0.
        route[:, 2] = 1.
        return route

    def _obs(self, value=0.):
        obs = {'proprio': torch.zeros(self.num_envs, 45),
               'history': torch.zeros(self.num_envs, 5, 45),
               'heightmap': torch.zeros(self.num_envs, 3, 1, 1),
               'privileged': torch.zeros(self.num_envs, 35)}
        if self.with_route:
            obs['route_error'] = self._route(value)
        return obs

    def _episode(self):
        record = {'slot_class': 'flat', 'duration': .02,
                  'reward_sums': {'reward_total': 0.},
                  'route_max_abs_y': .1, 'route_max_abs_heading_deg': 2.,
                  'route_final': [.1, 0., 1.], 'route_initial': [0., 0., 1.]}
        if not self.with_episode_route:
            for field in ROUTE_FIELDS:
                record.pop(field)
        return record

    def _reset(self):
        self.resets += 1
        self.reset_count += 1
        self.last_reset_rng_state = self.rng.get_state().clone()
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, generator=self.rng)
        self.episode_length_buf.zero_()
        obs = self._obs(.05)
        self.reset_route = obs.get('route_error')
        return obs

    def reset(self):
        return self._reset(), {}

    def step(self, action):
        self.steps += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        obs = self._obs(.05 + .01 * (self.steps % 48))
        terminal = self._obs(3.)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, torch.zeros(self.num_envs), zeros, zeros, {
            'terminal_observation': terminal,
            'episodes': [self._episode()],
            'reward_terms': {'speed': .1},
            'diagnostics': {'stall_on': zeros}}


class RouteModel(torch.nn.Module):
    """Tiny model carrying the real schema-7 route/normalizer tensor identities.

    It is a lightweight stand-in for ``V05R2ActorCritic`` (no encoders/MLP), but
    its ``state_dict`` exposes exactly the tensors the r2 recovery validator
    checks: the two route bypass tensors, the 211-input critic, the 45-dim ETH
    normalizer, the 80-dim critic normalizer and ``std``.
    """
    def __init__(self):
        super().__init__()
        self.actor_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.actor = torch.nn.Module()
        self.actor.route_fusion = torch.nn.Module()
        self.actor.route_fusion.weight = torch.nn.Parameter(torch.zeros(512, 3))
        self.actor.route_fusion.bias = torch.nn.Parameter(torch.zeros(512))
        self.actor.normalizer = torch.nn.Module()
        self.actor.normalizer.register_buffer('_mean', torch.zeros(1, 45))
        self.actor.normalizer.register_buffer('_var', torch.ones(1, 45))
        self.actor.normalizer.register_buffer('_std', torch.ones(1, 45))
        self.actor.normalizer.register_buffer('count', torch.tensor(0.))
        self.critic = torch.nn.ModuleList([torch.nn.Linear(211, 512, bias=True)])
        self.critic_normalizer = torch.nn.Module()
        self.critic_normalizer.register_buffer('mean', torch.zeros(80))
        self.critic_normalizer.register_buffer('var', torch.ones(80))
        self.critic_normalizer.register_buffer('count', torch.tensor(0., dtype=torch.float64))
        mask = torch.ones(80, dtype=torch.bool)
        mask[48:52] = False
        self.critic_normalizer.register_buffer('continuous', mask)
        self.std = torch.nn.Parameter(torch.full((12,), spec.INITIAL_STD))

    def update_critic_statistics(self, obs):
        return None

    def constrain_std(self):
        self.std.data.clamp_(spec.STD_MIN, spec.STD_MAX)


class FakePPO:
    """PPO stand-in emitting the exact ledger callback sequence and route block."""
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.cfg = type('Cfg', (), dict(rollout=48, epochs=5, minibatches=4,
                                        learning_rate=spec.PPO_INITIAL_LR))()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=spec.PPO_INITIAL_LR)
        self.learning_state = dict(optimizer_initialized=False, completed_ppo_updates=0,
                                   optimizer_steps=0)
        self.obs = None
        self.storage = None
        self.control_step_callback = None
        self.update_callback = None
        self.diagnostics_callback = None
        self.last_route_diagnostics = None
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
        rows = []
        records = []
        for step in range(48):
            rows.append(self.obs['route_error'].clone())
            self.obs, _, _, _, info = self.env.step(torch.zeros(self.env.num_envs, 12))
            records.extend(info.get('episodes', []))
            if self.control_step_callback is not None:
                self.control_step_callback(dict(step_index=step + 1,
                                                num_envs=self.env.num_envs,
                                                physics_samples_increment=self.env.num_envs,
                                                info=info))
        self.storage = {'obs': {'route_error': torch.stack(rows).flatten(0, 1)}}
        self.last_route_diagnostics = route_diagnostics(
            self.storage['obs']['route_error'])
        self.collections += 1
        return list(records)

    def update(self):
        base = self.learning_state['completed_ppo_updates']
        if self.update_callback is not None:
            self.update_callback('optimizer_started',
                                 dict(learning_state=self.learning,
                                      expected_optimizer_steps=20))
        for _ in range(20):
            self.optimizer.zero_grad(set_to_none=True)
            loss = sum(parameter.square().sum()
                       for parameter in self.model.parameters())
            loss.backward()
            self.optimizer.step()
            self.model.constrain_std()
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
    """In-memory mirror of the durable ledger's runner-facing contract."""
    def __init__(self, *, mode='formal', run_id='r2unit', num_envs=8):
        self.header = dict(schema=1, run_id=run_id, mode=mode)
        self._curriculum = dict(stage='F0', stage_iteration=0,
                                spent={stage: 0 for stage in spec.STAGE_ORDER},
                                streak=dict(gate=0, collapse=0), pending_transition=None)
        self._completed = 0
        self._active = None
        self._tasks = {}
        self._results = []
        self._qualified = None
        self._num_envs = num_envs
        self._checkpoints = []
        self.events = []

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

    @property
    def gate_streak(self):
        return self.curriculum['streak']['gate']

    @property
    def collapse_streak(self):
        return self.curriculum['streak']['collapse']

    @property
    def pending_transition(self):
        return self.curriculum['pending_transition']

    @property
    def control_samples(self):
        return 48 * self._num_envs * self._completed

    @property
    def physics_substeps(self):
        return 4 * self.control_samples

    def expected_learning(self):
        return learning_state(self._completed, num_envs=self._num_envs)

    def anchor(self):
        return dict(seq=len(self.events), digest=digest(f'events-{len(self.events)}'))

    def state_dict(self):
        return dict(header=deepcopy(self.header), events=[], seq=0, digest='0' * 64)

    # -------------------------------------------------------------- attempts
    def begin_attempt(self, curriculum, num_envs):
        if curriculum != self.curriculum:
            raise ValueError('Curriculum differs from the ledger state')
        self._num_envs = num_envs
        self._active = dict(controls=0, num_envs=num_envs, phase='collect',
                            optimizer_steps=0)
        self.events.append('begin_attempt')

    def control_step(self, event):
        self._active['controls'] += 1

    def update_event(self, kind, payload):
        if kind == 'optimizer_started':
            self._active['phase'] = 'optimizer_started'
        elif kind == 'optimizer_step_completed':
            self._active['optimizer_steps'] += 1
        elif kind == 'update_completed':
            self._active['phase'] = 'complete_pending'

    def complete_attempt(self, curriculum, learning):
        expected = updated_curriculum(self.curriculum, num_envs=self._active['num_envs'],
                                      mode=self.mode)
        if curriculum != expected:
            raise ValueError('Curriculum update inconsistent')
        self._curriculum = deepcopy(curriculum)
        self._completed += 1
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
        self.events.append('plan')
        return dict(task_key=key, state='planned', record=deepcopy(self.tasks[key]),
                    replayed=False)

    def commit_evaluation_result(self, key, *, result_path, result_sha256, report_path,
                                 report_sha256, evidence_sha256, summary_sha256,
                                 committed_wall=None):
        record = self.tasks[key]
        record['state'] = 'result_committed'
        record['result'] = dict(result_path=result_path, result_sha256=result_sha256,
                                report_path=report_path, report_sha256=report_sha256,
                                evidence_sha256=list(evidence_sha256),
                                summary_sha256=summary_sha256,
                                committed_wall=float(committed_wall or 0.))
        self.events.append('commit')
        return dict(task_key=key, state='result_committed')

    def consume_evaluation(self, key, *, outcome, passed, collapse_hit, consumed_wall=None):
        record = self.tasks[key]
        if record['state'] == 'consumed':
            prior = record['consumption']
            if (prior['outcome'], prior['passed'], prior['collapse_hit']) != (
                    outcome, bool(passed), bool(collapse_hit)):
                raise ValueError('Replayed consumption differs')
            return dict(task_key=key, state='consumed', replayed=True,
                        streak=deepcopy(self.curriculum['streak']))
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
        self.results.append(dict(task_key=key, kind=record['kind'], stage=record['stage'],
                                 global_update=record['global_update'], outcome=outcome,
                                 passed=bool(passed), gate_streak=streak['gate'],
                                 collapse_streak=streak['collapse']))
        if (record['kind'] == 'validation' and streak['gate'] >= 2
                and pending is not None and pending['to_stage'] is None):
            self._qualified = record['global_update']
        self.events.append('consume')
        return dict(task_key=key, state='consumed', replayed=False,
                    streak=deepcopy(streak))

    def commit_transition(self, *, from_stage, to_stage, at_update, gate_streak):
        pending = self.curriculum['pending_transition']
        if pending is None:
            raise ValueError('No pending transition')
        self.curriculum = dict(self.curriculum, stage=to_stage, stage_iteration=0,
                               streak=dict(gate=0, collapse=0),
                               pending_transition=None)
        self.events.append('transition')
        return dict(committed=True, stage=to_stage)

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
                       if record['state'] == 'consumed'
                       and record['kind'] == 'diagnostic']
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
    def __init__(self):
        self.calls = []
        self.diagnostic_plan = {}
        self.seen_models = []
        self.gate_pass = lambda kind, stage, update: True

    def make_cases(self, stage, kind, seed):
        return FakeCaseTable(stage, kind, seed)

    def __call__(self, model, cases, *, device, num_envs, stage, kind, run_id,
                 global_update, checkpoint_sha, output_dir, record_video=False,
                 mode='formal', seed=42):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)
        episodes = 100 if kind == 'test' else (5 if kind == 'diagnostic' else 20)
        if kind == 'diagnostic':
            summary = self.diagnostic_plan.pop(int(global_update), diagnostic_summary())
        else:
            summary = gate_summary(
                passed=bool(self.gate_pass(kind, stage, int(global_update))),
                total=episodes)
        report = dict(summary, stage=stage, kind=kind, episodes=episodes,
                      eval_seed=int(getattr(cases, 'seed', -1)),
                      checkpoint_sha256=checkpoint_sha, case_table_sha256=cases.sha256)
        (out / 'manifest.json').write_text(json.dumps(dict(
            run_id=run_id, mode=mode, stage=stage, kind=kind,
            global_update=int(global_update), checkpoint_sha256=checkpoint_sha,
            case_table_sha256=cases.sha256)))
        (out / 'report.json').write_text(json.dumps(report))
        (out / 'episodes.jsonl').write_text(
            '\n'.join(json.dumps({'case': index}) for index in range(episodes)) + '\n')
        (out / 'window_batch_0000.npz').write_bytes(b'fake-npz-evidence')
        self.calls.append(dict(kind=kind, stage=stage, update=int(global_update),
                               checkpoint_sha=checkpoint_sha))
        key = (run_id, mode, stage, int(global_update), kind, checkpoint_sha, cases.sha256)
        return types_result(key, report, out)


def types_result(key, summary, out):
    class Result:
        def __init__(self):
            self.key = key
            self.summary = summary
            self.raw_paths = {'manifest': str(out / 'manifest.json'),
                              'episodes': str(out / 'episodes.jsonl'),
                              'npz': [str(out / 'window_batch_0000.npz')]}
    return Result()


def stamp_optimizer(ppo, update):
    """Give the fake optimizer a complete Adam body at ``update`` (schema-7 check)."""
    state = ppo.optimizer.state
    for parameter in ppo.model.parameters():
        state[parameter] = dict(step=torch.tensor(float(20 * update)),
                                exp_avg=torch.zeros_like(parameter),
                                exp_avg_sq=torch.zeros_like(parameter))
    ppo.learning_state['optimizer_steps'] = 20 * update
    ppo.learning_state['completed_ppo_updates'] = update
    ppo.learning_state['optimizer_initialized'] = update > 0


def make_fixture(root, *, mode='formal', num_envs=8, seed=42, env=None, ppo=None,
                 ledger=None, evaluation=None, require_route_episode_evidence=None):
    env = RouteEnv(num_envs) if env is None else env
    model = RouteModel()
    ppo = FakePPO(model, env) if ppo is None else ppo
    ledger = (FakeLedger(mode=mode, run_id=f'r2-{mode}-{seed}', num_envs=num_envs)
              if ledger is None else ledger)
    curriculum = Curriculum(ledger, mode=mode, seed=seed)
    fake = evaluation or FakeEvaluation()
    runner = V05Runner(model=model, env=env, ppo=ppo, curriculum=curriculum,
                       ledger=ledger, device='cpu', seed=seed, output_dir=root,
                       evaluation=fake, make_cases=fake.make_cases,
                       save_checkpoints=True, final_test=True, eval_num_envs=8,
                       require_route_episode_evidence=require_route_episode_evidence)
    return dict(runner=runner, env=env, model=model, ppo=ppo, curriculum=curriculum,
                ledger=ledger, evaluation=fake)


class RunnerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class RouteObservationTests(RunnerTestCase):
    def test_reset_adopts_the_route_and_records_route_initial(self):
        fixture = make_fixture(self.root)
        obs = fixture['runner'].reset_episode()
        self.assertEqual(set(obs), {'proprio', 'history', 'heightmap', 'privileged',
                                    'route_error'})
        self.assertEqual(tuple(obs['route_error'].shape), (8, 3))
        self.assertEqual(obs['route_error'].dtype, torch.float32)
        self.assertTrue(torch.equal(fixture['runner'].reset_route,
                                    obs['route_error']))
        self.assertIs(fixture['runner'].obs, obs)

    def test_missing_route_observation_is_rejected_at_reset(self):
        env = RouteEnv(with_route=False)
        fixture = make_fixture(self.root, env=env)
        with self.assertRaises(ValueError) as caught:
            fixture['runner'].reset_episode()
        self.assertIn('route_error', str(caught.exception))
        self.assertIn('runner.reset', str(caught.exception))

    def test_collect_packages_route_and_audits_episode_evidence(self):
        fixture = make_fixture(self.root)
        fixture['runner'].reset_episode()
        fixture['runner'].collect()
        storage = fixture['ppo'].storage['obs']['route_error']
        self.assertEqual(tuple(storage.shape), (48 * 8, 3))
        self.assertEqual(fixture['runner'].last_route_audit['episodes'], 48)
        self.assertTrue(fixture['runner'].last_route_audit['complete'])
        self.assertEqual(fixture['runner'].last_route_diagnostics['nonfinite_count'], 0)

    def test_missing_episode_route_evidence_is_a_hard_error_in_formal_mode(self):
        env = RouteEnv(with_episode_route=False)
        fixture = make_fixture(self.root, env=env)
        fixture['runner'].reset_episode()
        with self.assertRaises(ValueError) as caught:
            fixture['runner'].collect()
        self.assertIn('route_final', str(caught.exception))
        # A smoke/unit-test env keeps the same audit without the hard requirement.
        smoke = make_fixture(self.root / 'smoke', mode='smoke',
                             env=RouteEnv(with_episode_route=False),
                             require_route_episode_evidence=False)
        smoke['runner'].reset_episode()
        smoke['runner'].collect()
        self.assertFalse(smoke['runner'].last_route_audit['complete'])
        self.assertFalse(smoke['runner'].last_route_audit['required'])

    def test_route_episode_audit_reports_every_missing_field(self):
        audit = route_episode_audit([{'route_final': [0., 0., 1.]}], required=False)
        self.assertEqual(audit['episodes'], 1)
        self.assertFalse(audit['complete'])
        self.assertEqual(audit['missing']['route_initial'], 1)
        self.assertEqual(audit['missing']['route_max_abs_y'], 1)
        with self.assertRaises(ValueError):
            route_episode_audit([{'route_final': [0., 0., 1.]}], required=True)

    def test_train_event_carries_the_read_only_route_block(self):
        fixture = make_fixture(self.root)
        emitted = []
        fixture['runner'].telemetry = type(
            'Telemetry', (), {'emit': lambda self, kind, metrics: emitted.append(
                (kind, metrics))})()
        fixture['runner'].reset_episode()
        fixture['runner'].train_step()
        train = [metrics for kind, metrics in emitted if kind == 'train']
        self.assertEqual(len(train), 1)
        route = train[0]['route']
        for field in runner_module.ROUTE_DIAGNOSTIC_FIELDS:
            self.assertIn(field, route)
        self.assertEqual(route['nonfinite_count'], 0)
        self.assertIn('policy_state', dir(fixture['runner']))
        state = fixture['runner'].policy_state
        self.assertEqual(tuple(state['actor.route_fusion.weight'].shape), (512, 3))
        self.assertEqual(tuple(state['actor.route_fusion.bias'].shape), (512,))
        self.assertEqual(tuple(state['critic.0.weight'].shape), (512, 211))


class CadenceTests(RunnerTestCase):
    def test_fifty_update_diagnostic_and_hundred_update_save(self):
        fixture = make_fixture(self.root)
        fixture['runner'].reset_episode()
        for _ in range(50):
            fixture['runner'].train_step()
        self.assertEqual(fixture['runner'].stage_iteration, 50)
        kinds = fixture['evaluation'].calls
        self.assertEqual([call['kind'] for call in kinds], ['diagnostic'])
        self.assertEqual(kinds[0]['update'], 50)
        checkpoints = [record['path'] for record in fixture['ledger'].checkpoints]
        self.assertEqual(len(checkpoints), 1)  # diagnostic point save at 50
        fixture['runner'].train_step()  # 51: nothing due
        self.assertEqual(len(fixture['evaluation'].calls), 1)
        for _ in range(49):
            fixture['runner'].train_step()
        self.assertEqual(fixture['runner'].stage_iteration, 100)
        kinds = [call['kind'] for call in fixture['evaluation'].calls]
        # At update 100 the 50-update diagnostic is due again, then the F0 gate
        # (no flat-retention run at F0 itself).
        self.assertEqual(kinds, ['diagnostic', 'diagnostic', 'validation'])
        update_paths = [record['path'] for record in fixture['ledger'].checkpoints]
        self.assertEqual(len(update_paths), 2)  # save at 100 (one per boundary)
        self.assertTrue(all(str(self.root) in path for path in update_paths))

    def test_checkpoint_contains_the_schema7_route_body(self):
        fixture = make_fixture(self.root)
        fixture['runner'].reset_episode()
        for _ in range(100):
            fixture['runner'].train_step()
        record = fixture['ledger'].checkpoints[-1]
        state = recovery.preload_checkpoint(record['path'], record['sha256'])
        self.assertEqual(state['schema'], 7)
        self.assertEqual(set(state), set(recovery.CHECKPOINT_KEYS))
        self.assertEqual(tuple(state['model']['actor.route_fusion.weight'].shape), (512, 3))
        self.assertEqual(tuple(state['model']['critic.0.weight'].shape), (512, 211))
        self.assertEqual(state['resume_mode'], 'new_episode')
        self.assertEqual(state['learning']['update'], 100)
        self.assertEqual(state['learning']['physics_substeps'],
                         4 * state['learning']['control_samples'])

    def test_collapse_needs_two_consecutive_hits_and_stops(self):
        fixture = make_fixture(self.root)
        fixture['evaluation'].diagnostic_plan = {
            100: diagnostic_summary(stalled_cases=12, first_step_absent=12),
            150: diagnostic_summary(stalled_cases=12, first_step_absent=12),
        }
        fixture['runner'].reset_episode()
        terminal = None
        for update in range(1, 151):
            result = fixture['runner'].train_step()
            if result['status'] == 'behavior_collapse':
                terminal = update
                break
        self.assertEqual(terminal, 150)
        self.assertEqual(fixture['runner'].terminal, 'behavior_collapse')
        self.assertEqual(fixture['runner'].curriculum.collapse_streak,
                         COLLAPSE_CONSECUTIVE)
        self.assertEqual(fixture['runner'].iteration, 150)

    def test_a_single_hit_clears_the_collapse_streak(self):
        fixture = make_fixture(self.root)
        fixture['evaluation'].diagnostic_plan = {
            100: diagnostic_summary(stalled_cases=12, first_step_absent=12),
            150: diagnostic_summary(stalled_cases=0),
        }
        fixture['runner'].reset_episode()
        for _ in range(150):
            fixture['runner'].train_step()
        self.assertIsNone(fixture['runner'].terminal)
        self.assertEqual(fixture['runner'].curriculum.collapse_streak, 0)


class PilotTests(RunnerTestCase):
    def test_pilot_cap_and_stage_budgets_come_from_mode_budget(self):
        fixture = make_fixture(self.root, mode='pilot')
        curriculum = fixture['curriculum']
        self.assertEqual(curriculum.update_cap, 2500)
        self.assertEqual(curriculum.total_budget, 2500)
        self.assertEqual(curriculum.budgets['F0'], 2000)
        self.assertEqual(curriculum.budgets['P1'], 500)
        self.assertEqual(curriculum.effective_cap_source, 'mode_budget')
        audit = curriculum.budget_audit()
        self.assertEqual(audit['worker_max_updates'], 2500)
        self.assertEqual(audit['supervisor_max_updates'], 2500)
        self.assertEqual(audit['stage_machine_cap'], 2500)
        self.assertEqual(audit['resume_cap'], 2500)
        self.assertTrue(audit['same_source'])
        fields = curriculum.result_budget_fields()
        self.assertEqual(fields['total_budget'], 2500)
        self.assertEqual(fields['update_cap'], 2500)
        # A pilot result reports its own two-stage table, never the formal sum.
        self.assertEqual(fields['stage_budgets'], {'F0': 2000, 'P1': 500})
        self.assertEqual(fields['mode'], 'pilot')
        self.assertEqual(fields['effective_cap_source'], 'mode_budget')
        formal = make_fixture(self.root / 'formal', mode='formal')
        self.assertEqual(formal['curriculum'].update_cap, 30000)
        self.assertEqual(formal['curriculum'].budgets['P1'], 1000)
        self.assertEqual(formal['curriculum'].result_budget_fields()['total_budget'],
                         30000)

    def test_pilot_p1_is_fixed_at_500_and_gates_at_400_and_500_decide(self):
        fixture = make_fixture(self.root, mode='pilot')
        ledger, ppo, runner = fixture['ledger'], fixture['ppo'], fixture['runner']
        # Reach the pilot P1 boundary at 400 updates with the durable evidence.
        ledger._curriculum['stage'] = 'P1'
        ledger._curriculum['spent']['F0'] = 2000
        ledger._curriculum['spent']['P1'] = 400
        ledger._curriculum['stage_iteration'] = 400
        ledger._curriculum['streak']['gate'] = 1
        ledger._completed = 2400
        stamp_optimizer(ppo, 2400)
        runner.reset_episode()
        result = runner.boundary()
        self.assertEqual(result['status'], 'continue')
        # 400 is both a diagnostic and a full-gate update for pilot P1.
        self.assertEqual([call['kind'] for call in fixture['evaluation'].calls],
                         ['diagnostic', 'flat_retention', 'validation'])
        self.assertEqual(ledger.gate_streak, 2)
        # SDS 7.1: pilot P1 never advances early and the verdict is still running.
        self.assertFalse(runner.curriculum.can_advance)
        self.assertEqual(runner.curriculum.pilot_verdict()['status'], 'running')
        # 500 updates: both final gates passed -> pilot_complete at the fixed cap.
        ledger._curriculum['spent']['P1'] = 500
        ledger._curriculum['stage_iteration'] = 500
        ledger._completed = 2500
        stamp_optimizer(ppo, 2500)
        result = runner.boundary()
        self.assertEqual(result['status'], 'pilot_complete')
        verdict = result['verdict']
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(verdict['gates'], {100: None, 200: None, 300: None,
                                            400: True, 500: True})
        self.assertEqual(verdict['p1_consumed'], 500)
        self.assertEqual(runner.terminal, 'pilot_complete')
        # The pilot cap is 2500: a 2501st update cannot be charged.
        self.assertEqual(ledger.completed_ppo_updates, 2500)

    def test_pilot_failure_at_500_is_reported_not_retried(self):
        fixture = make_fixture(self.root, mode='pilot')
        ledger, ppo, runner = fixture['ledger'], fixture['ppo'], fixture['runner']
        ledger._curriculum['stage'] = 'P1'
        ledger._curriculum['spent']['F0'] = 2000
        ledger._curriculum['spent']['P1'] = 500
        ledger._curriculum['stage_iteration'] = 500
        ledger._completed = 2500
        stamp_optimizer(ppo, 2500)
        # Both final gates run on this boundary; the 500 gate fails.
        fixture['evaluation'].gate_pass = (
            lambda kind, stage, update: not (kind == 'validation' and update == 2500))
        runner.reset_episode()
        result = runner.boundary()
        self.assertEqual(result['status'], 'pilot_failed')
        self.assertIn('p1_gate_500_not_passed', result['verdict']['reasons'])
        self.assertEqual(runner.terminal, 'pilot_failed')

    def test_budget_exhaustion_reports_the_effective_cap(self):
        # Formal F0 exhausts its 2000 stage cap without a second passing gate:
        # the boundary stops at budget_exhausted and reports the mode cap.
        fixture = make_fixture(self.root, mode='formal')
        ledger, ppo, runner = fixture['ledger'], fixture['ppo'], fixture['runner']
        ledger._curriculum['spent']['F0'] = 2000
        ledger._curriculum['stage_iteration'] = 2000
        ledger._completed = 2000
        stamp_optimizer(ppo, 2000)
        runner.reset_episode()
        result = runner.boundary()
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['update_cap'], 30000)
        self.assertEqual(result['effective_cap_source'], 'mode_budget')
        self.assertEqual(runner.terminal, 'budget_exhausted')
        # A pilot run reports the 2500 effective cap, never the r1 29500 sum.
        pilot = make_fixture(self.root / 'pilot', mode='pilot')
        self.assertEqual(pilot['curriculum'].update_cap, 2500)
        self.assertEqual(pilot['runner'].curriculum.total_budget, 2500)

    def test_runner_refuses_to_construct_beyond_the_cap(self):
        fixture = make_fixture(self.root, mode='pilot')
        fixture['ppo'].learning_state['completed_ppo_updates'] = 2501
        fixture['ledger']._completed = 2501
        with self.assertRaises(ValueError):
            V05Runner(model=fixture['model'], env=fixture['env'], ppo=fixture['ppo'],
                      curriculum=fixture['curriculum'], ledger=fixture['ledger'],
                      device='cpu', seed=42, output_dir=self.root)


class ResumeTests(RunnerTestCase):
    def test_resume_restores_rng_then_performs_exactly_one_new_episode_reset(self):
        fixture = make_fixture(self.root)
        runner = fixture['runner']
        runner.reset_episode()
        record_sha = runner.save_checkpoint('unit')
        record = fixture['ledger'].checkpoints[-1]
        self.assertEqual(record['sha256'], record_sha)

        # A fresh process: new env/runner, the same durable ledger boundary.
        fresh_env = RouteEnv()
        fresh_model = RouteModel()
        fresh_ppo = FakePPO(fresh_model, fresh_env)
        fresh_ledger = FakeLedger(run_id='r2-formal-42', num_envs=8)
        curriculum = Curriculum(fresh_ledger, mode='formal', seed=42)
        fake = FakeEvaluation()
        fresh = V05Runner(model=fresh_model, env=fresh_env, ppo=fresh_ppo,
                          curriculum=curriculum, ledger=fresh_ledger, device='cpu',
                          seed=42, output_dir=self.root / 'resume',
                          evaluation=fake, make_cases=fake.make_cases)
        reset_before = fresh_env.reset_count
        outcome = fresh.resume(record['path'], record_sha)
        self.assertEqual(outcome['status'], 'resumed')
        self.assertEqual(fresh_env.reset_count, reset_before + 1)
        state = recovery.preload_checkpoint(record['path'], record_sha)
        self.assertTrue(torch.equal(fresh_env.last_reset_rng_state, state['rng']['env']))
        self.assertEqual(set(fresh.obs), {'proprio', 'history', 'heightmap',
                                          'privileged', 'route_error'})
        self.assertEqual(fresh.iteration, 0)

    def test_resume_rejects_a_208_critic_checkpoint(self):
        fixture = make_fixture(self.root)
        runner = fixture['runner']
        runner.reset_episode()
        runner.save_checkpoint('unit')
        record = fixture['ledger'].checkpoints[-1]
        state = recovery.preload_checkpoint(record['path'], record['sha256'])
        state['model'] = dict(state['model'])
        state['model']['critic.0.weight'] = torch.zeros(512, 208)
        del state['model']['actor.route_fusion.weight']
        del state['model']['actor.route_fusion.bias']
        path = self.root / 'forged208.pt'
        recovery.atomic_torch_save(path, state)
        sha = recovery.sha256(path)
        with self.assertRaises(ValueError):
            recovery.preload_checkpoint(path, sha)
        with self.assertRaises(ValueError):
            runner.resume(path, sha)

    def test_resume_rejects_a_checkpoint_above_the_mode_cap(self):
        fixture = make_fixture(self.root, mode='pilot')
        runner = fixture['runner']
        runner.reset_episode()
        record_sha = runner.save_checkpoint('unit')
        record = fixture['ledger'].checkpoints[-1]
        with self.assertRaises(ValueError):
            recovery.audit_budget(learning_state(2501, num_envs=8),
                                  fixture['ledger'].curriculum, mode='pilot')


class StageTests(RunnerTestCase):
    def test_sync_stage_iteration_restates_without_resetting(self):
        fixture = make_fixture(self.root)
        runner = fixture['runner']
        runner.reset_episode()
        resets = fixture['env'].reset_count
        runner.sync_stage_iteration()
        self.assertEqual(fixture['env'].stage_calls[-1][:2], ('F0', 0))
        self.assertEqual(fixture['env'].reset_count, resets)

    def test_transition_binds_the_next_stage_and_saves_three_boundaries(self):
        fixture = make_fixture(self.root)
        ledger, runner, env = fixture['ledger'], fixture['runner'], fixture['env']
        runner.reset_episode()
        ledger._curriculum['pending_transition'] = dict(
            from_stage='F0', to_stage='P1', at_update=0, requires=None,
            at_checkpoint='a' * 64)
        result = runner.transition()
        self.assertEqual(result['to_stage'], 'P1')
        self.assertEqual(env.stage, 'P1')
        self.assertEqual(env.stage_calls[-1][2], 'F0')
        kinds = [Path(record['path']).name.split('_000000')[0]
                 for record in ledger.checkpoints]
        self.assertEqual(len(kinds), 3)
        self.assertTrue(all(name.startswith('stage') for name in kinds))
        self.assertEqual(runner.stage, 'P1')
        self.assertEqual(runner.stage_iteration, 0)


class CaseSeedTests(RunnerTestCase):
    """SDS 6.3 offsets are applied exactly once, inside ``evaluation.make_cases``."""

    def test_make_cases_uses_the_single_declared_offset(self):
        from src.tasks.velocity.v05_r2.evaluation import make_cases
        cases = {
            ('F0', 'validation', 42): 200342,
            ('F0', 'flat_retention', 42): 200342,
            ('P1', 'validation', 42): 200342,
            ('S1', 'perturbed', 42): 300342,
            ('P1', 'diagnostic', 42): 500342,
            ('S5', 'test', 42): 400342,
            ('F0', 'validation', 43): 200343,
            ('F0', 'flat_retention', 43): 200343,
            ('P1', 'diagnostic', 43): 500343,
            ('S5', 'test', 43): 400342,  # the final test seed is fixed, not seed+offset
        }
        for (stage, kind, seed), expected in cases.items():
            with self.subTest(stage=stage, kind=kind, seed=seed):
                table = make_cases(stage, kind, seed)
                self.assertEqual(table.seed, expected)
                # The double-offset defect produced seed+2*offset (400642/1000642).
                self.assertNotEqual(table.seed, expected + 200300)
                self.assertNotEqual(table.seed, expected + 300300)
                self.assertNotEqual(table.seed, expected + 500300)

    def test_case_table_sha_is_stable_and_seed_separated(self):
        from src.tasks.velocity.v05_r2.evaluation import make_cases
        first = make_cases('P1', 'validation', 42)
        self.assertEqual(first.sha256, make_cases('P1', 'validation', 42).sha256)
        self.assertNotEqual(first.sha256, make_cases('P1', 'validation', 43).sha256)
        self.assertNotEqual(make_cases('F0', 'validation', 42).sha256,
                            make_cases('F0', 'validation', 43).sha256)

    def test_runner_passes_the_raw_seed_and_records_the_effective_seed(self):
        from src.tasks.velocity.v05_r2 import evaluation as evaluation_module

        class RealSeedEvaluation(FakeEvaluation):
            def make_cases(self, stage, kind, seed):
                table = evaluation_module.make_cases(stage, kind, seed)
                self.case_seeds.append((stage, kind, int(seed), int(table.seed)))
                return table

        for seed, expected in ((42, dict(validation=200342, flat_retention=200342,
                                         diagnostic=500342, perturbed=300342)),
                               (43, dict(validation=200343, flat_retention=200343,
                                         diagnostic=500343, perturbed=300343))):
            with self.subTest(seed=seed):
                fake = RealSeedEvaluation()
                fake.case_seeds = []
                root = self.root / f'seed{seed}'
                fixture = make_fixture(root, seed=seed, evaluation=fake)
                ledger, ppo, runner = (fixture['ledger'], fixture['ppo'],
                                       fixture['runner'])
                # F0: validation only (no flat-retention run at F0 itself).
                runner.reset_episode()
                gate = runner.run_full_gate(DIGEST)
                self.assertEqual(gate['eval_seed'], expected['validation'])
                # P1: validation + flat retention + diagnostic (stage change only
                # for these tasks).
                ledger._curriculum['stage'] = 'P1'
                ledger._curriculum['spent']['F0'] = 2000
                ledger._curriculum['spent']['P1'] = 100
                ledger._curriculum['stage_iteration'] = 100
                ledger._completed = 2100
                ppo.learning_state['completed_ppo_updates'] = 2100
                ppo.learning_state['optimizer_steps'] = 20 * 2100
                runner.reset_episode()
                runner.run_diagnostic(DIGEST)
                full = runner.run_full_gate(DIGEST)
                self.assertEqual(full['eval_seed'], expected['validation'])
                self.assertEqual(full['retention_eval_seed'],
                                 expected['flat_retention'])
                runner.run_perturbed(DIGEST)
                recorded = {(stage, kind): (raw, effective)
                            for stage, kind, raw, effective in fake.case_seeds}
                for kind, effective in expected.items():
                    stage = {'validation': 'P1', 'flat_retention': 'F0',
                             'diagnostic': 'P1', 'perturbed': 'S1'}[kind]
                    self.assertIn((stage, kind), recorded)
                    raw, table_seed = recorded[(stage, kind)]
                    self.assertEqual(raw, seed)  # raw training seed, no double offset
                    self.assertEqual(table_seed, effective)
                # Every generated report carries the same effective seed.
                self.assertIn(('P1', 'validation', seed, expected['validation']),
                              fake.case_seeds)
                self.assertEqual(runner._case_seed('validation'), expected['validation'])
                self.assertEqual(runner._case_seed('diagnostic'), expected['diagnostic'])

    def test_runner_task_keys_use_the_effective_case_table(self):
        from src.tasks.velocity.v05_r2 import evaluation as evaluation_module

        class RealSeedEvaluation(FakeEvaluation):
            def make_cases(self, stage, kind, seed):
                return evaluation_module.make_cases(stage, kind, seed)

        fake = RealSeedEvaluation()
        fixture = make_fixture(self.root, evaluation=fake)
        ledger, runner = fixture['ledger'], fixture['runner']
        runner.reset_episode()
        gate = runner.run_full_gate(DIGEST)
        self.assertEqual(gate['eval_seed'], 200342)
        task = next(record for key, record in ledger.tasks.items()
                    if record['kind'] == 'validation')
        expected_sha = evaluation_module.make_cases('F0', 'validation', 42).sha256
        self.assertEqual(task['case_sha256'], expected_sha)
        # The wrong (double-offset) table can never be the bound key.
        self.assertNotEqual(
            task['case_sha256'],
            evaluation_module.make_cases('F0', 'validation', 200342).sha256)


class BudgetTests(unittest.TestCase):
    def test_modes_share_one_budget_source(self):
        self.assertEqual(spec.mode_budget('pilot')['total'], 2500)
        self.assertEqual(spec.mode_budget('formal')['total'], 30000)
        self.assertEqual(runner_module.PILOT_BUDGETS, {'F0': 2000, 'P1': 500})
        self.assertEqual(runner_module.PILOT_P1_UPDATES, 500)
        self.assertEqual(runner_module.PILOT_MAX_UPDATES, 2500)


if __name__ == '__main__':
    unittest.main()
