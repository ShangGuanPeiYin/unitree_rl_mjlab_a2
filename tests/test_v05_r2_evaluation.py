"""CPU evidence for v0.5-r2 case tables, gate arithmetic and route raw evidence.

SDS 0.5-r2 sections 4.8, 6 and 9.2.  Two layers:

* pure table/gate arithmetic against the frozen SDS numbers (no simulator);
* a deterministic kinematic ``FakeEnv`` that drives the whole
  :func:`evaluate_checkpoint` path -- padding mask, float32 NPZ with
  ``route_error [steps,3]``, JSONL, CPU recomputation, evidence verification,
  idempotent rerun -- without a simulator.

The r2 execution path itself is additionally exercised against a real CPU
``V05R2Env`` in ``tests/test_v05_r2_environment.py`` (device ``cpu``, no GPU).
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.tasks.velocity.v05_r2 import evaluation, spec
from src.tasks.velocity.v05_r2.evaluation import (CASE_PASS_REQUIRED, CASE_PASS_REQUIRED_FINAL,
                                                  CaseTable, EvalResult, _record_for_case,
                                                  case_table_sha, evaluate_checkpoint,
                                                  evaluation_key, gate_pass_counts,
                                                  make_cases, summarize, table_from_json,
                                                  verify_evaluation_evidence)
from src.tasks.velocity.v05_r2.speed import cpu_speed_oracle

DT = spec.CONTROL_DT
SPEEDS = tuple(spec.SPEEDS)


def small_table(stage='P1', kind='validation', *, episodes=None, measurement_stage=None,
                measurement=None):
    """A protocol-shaped fixed table with the real per-kind episode count."""
    if episodes is None:
        episodes = evaluation._expected_episodes(kind)
    measurement_stage = measurement_stage or ('F0' if kind == 'flat_retention' else stage)
    measurement = measurement or ('flat' if measurement_stage == 'F0' else 'stair')
    arrays, cases = {}, []
    rng = np.random.default_rng(12345)
    for speed in SPEEDS:
        entry = {'x': [], 'y': [], 'yaw': []}
        for index in range(episodes):
            pose = [-float(rng.uniform(.9, 1.1)), float(rng.uniform(-.05, .05)),
                    float(rng.uniform(-math.pi / 36, math.pi / 36))]
            entry['x'].append(pose[0])
            entry['y'].append(pose[1])
            entry['yaw'].append(pose[2])
            cases.append(dict(case_id=f'{stage}:{kind}:{float(speed):.1f}:{index:03d}',
                              vx=float(speed), episode_index=index, initial_pose=list(pose),
                              measurement_stage=measurement_stage))
        arrays[f'{float(speed):.1f}'] = entry
    table = CaseTable(stage=stage, kind=kind, seed=200342, arrays=arrays, sha256='',
                      episodes=episodes, measurement_stage=measurement_stage,
                      measurement=measurement, cases=cases)
    table.sha256 = case_table_sha(table)
    return table


def route_of(y, yaw):
    """The SDS 4.2 route vector for one pose (float32, as the env produces it)."""
    lateral = float(np.clip(np.float64(y) / .20, -4., 4.))
    return [lateral, float(np.sin(np.float64(yaw))), float(np.cos(np.float64(yaw)))]


# --------------------------------------------------------------------- tables
class CaseTableTests(unittest.TestCase):
    def test_seed_offsets_and_sizes(self):
        self.assertEqual(make_cases('F0', 'validation', 42).seed, 200300 + 42)
        self.assertEqual(make_cases('P1', 'validation', 42).seed, 200342)
        self.assertEqual(make_cases('S1', 'perturbed', 42).seed, 300342)
        self.assertEqual(make_cases('S1', 'diagnostic', 42).seed, 500342)
        self.assertEqual(make_cases('S5', 'test', 42).seed, 400342)
        for stage, kind in (('F0', 'validation'), ('P1', 'validation'), ('S1', 'perturbed'),
                            ('S5', 'validation'), ('S5', 'test'), ('S1', 'diagnostic'),
                            ('S1', 'flat_retention')):
            table = make_cases(stage, kind, 42)
            expected = 100 if kind == 'test' else 5 if kind == 'diagnostic' else 20
            self.assertEqual(table.episodes, expected)
            self.assertEqual(sorted(table.speeds), list(SPEEDS))
            for speed in table.speeds:
                self.assertEqual(len(table.arrays[f'{speed:.1f}']['x']), expected)

    def test_initial_pose_ranges_are_the_sds_rules(self):
        for stage in ('F0', 'P1'):
            table = make_cases(stage, 'validation', 42)
            for speed in table.speeds:
                entry = table.arrays[f'{speed:.1f}']
                self.assertTrue(all(-1.1 <= x <= -.9 for x in entry['x']))
                self.assertTrue(all(-.05 <= y <= .05 for y in entry['y']))
                self.assertTrue(all(-math.radians(5) <= a <= math.radians(5)
                                    for a in entry['yaw']))
        for stage in ('S1', 'S2', 'S3', 'S4'):
            table = make_cases(stage, 'validation', 42)
            for speed in table.speeds:
                entry = table.arrays[f'{speed:.1f}']
                self.assertTrue(all(list(pose) == [-1., 0., 0.]
                                    for pose in zip(entry['x'], entry['y'], entry['yaw'])))
        s5 = make_cases('S5', 'validation', 42)
        entry = s5.arrays['0.4']
        self.assertTrue(all(-1.2 <= x <= -.8 for x in entry['x']))
        self.assertTrue(all(-.1 <= y <= .1 for y in entry['y']))
        self.assertTrue(all(-math.radians(10) <= a <= math.radians(10) for a in entry['yaw']))
        diagnostic = make_cases('S5', 'diagnostic', 42)
        entry = diagnostic.arrays['0.4']
        self.assertTrue(all(list(pose) == [-1., 0., 0.]
                            for pose in zip(entry['x'], entry['y'], entry['yaw'])))

    def test_perturbed_is_s1_only_and_test_is_s5_only(self):
        with self.assertRaises(ValueError):
            make_cases('P1', 'perturbed', 42)
        with self.assertRaises(ValueError):
            make_cases('S1', 'test', 42)
        entry = make_cases('S1', 'perturbed', 42).arrays['0.4']
        self.assertTrue(all(-1.1 <= x <= -.9 for x in entry['x']))
        self.assertTrue(all(-.05 <= y <= .05 for y in entry['y']))
        self.assertTrue(all(-math.radians(5) <= a <= math.radians(5) for a in entry['yaw']))

    def test_local_rng_consumes_x_then_y_then_yaw_per_case(self):
        table = make_cases('P2', 'validation', 42)
        rng = np.random.default_rng(200342)
        for speed in SPEEDS:
            entry = table.arrays[f'{speed:.1f}']
            for index in range(table.episodes):
                x = -float(rng.uniform(.9, 1.1))
                y = float(rng.uniform(-.05, .05))
                yaw = float(rng.uniform(-math.pi / 36, math.pi / 36))
                self.assertEqual(entry['x'][index], x)
                self.assertEqual(entry['y'][index], y)
                self.assertEqual(entry['yaw'][index], yaw)

    def test_flat_retention_measures_on_f0(self):
        table = make_cases('S1', 'flat_retention', 42)
        self.assertEqual(table.stage, 'S1')
        self.assertEqual(table.measurement_stage, 'F0')
        self.assertEqual(table.measurement, 'flat')
        self.assertEqual(table.protocol_id, spec.FLAT_PROTOCOL_ID)
        entry = table.arrays['0.4']
        self.assertTrue(all(-1.1 <= x <= -.9 for x in entry['x']))
        self.assertNotEqual(table.sha256, make_cases('S1', 'validation', 42).sha256)

    # ------------------------------------------------------- r2 namespace binding
    def test_case_table_json_binds_the_r2_identity(self):
        table = make_cases('S1', 'validation', 42)
        value = table.to_json()
        self.assertEqual(value['task'], 'Unitree-A2-IdealHeightMap-Stairs-V05R2')
        self.assertEqual(value['recipe_version'], '0.5-r2')
        self.assertEqual(value['checkpoint_schema'], 7)
        self.assertEqual(value['architecture'], 'v05-route-state-v1')
        self.assertEqual(value['stair_protocol_id'], 'v05-r2-stair-climb-first-v1')
        self.assertEqual(value['flat_protocol_id'], 'v05-r2-flat-world-route-v1')
        self.assertEqual(value['route_state']['lateral_scale_m'], .20)
        self.assertEqual(value['route_state']['lateral_clip'], 4.)
        self.assertEqual(value['route_state']['yaw_axis'], 0.)

    def test_case_values_match_the_frozen_sds_numbers_but_not_the_r1_identity(self):
        # SDS 6.5 / Q003-r2 3.6: same numbers, same seeds, same generation rule --
        # but an r2-only case SHA.  The r1 module is imported read-only.
        from src.tasks.velocity.v05 import evaluation as r1_evaluation
        for stage, kind in (('F0', 'validation'), ('P1', 'validation'), ('S1', 'validation'),
                            ('S1', 'perturbed'), ('S5', 'validation'), ('S5', 'test'),
                            ('F0', 'diagnostic')):
            r1 = r1_evaluation.make_cases(stage, kind, 42)
            r2 = make_cases(stage, kind, 42)
            self.assertEqual(r2.arrays, r1.arrays, (stage, kind))
            self.assertEqual(r2.cases, r1.cases, (stage, kind))
            self.assertEqual(r2.seed, r1.seed, (stage, kind))
            self.assertNotEqual(r2.sha256, r1.sha256, (stage, kind))
            self.assertEqual(len(r2.sha256), 64)

    def test_tables_are_deterministic_and_round_trip(self):
        first = make_cases('S1', 'validation', 42)
        second = make_cases('S1', 'validation', 42)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.arrays, second.arrays)
        self.assertNotEqual(first.sha256, make_cases('S1', 'validation', 43).sha256)
        self.assertNotEqual(first.sha256, make_cases('S2', 'validation', 42).sha256)
        restored = table_from_json(first.to_json(), expected_sha=first.sha256)
        self.assertEqual(restored.sha256, first.sha256)
        self.assertEqual(restored.cases, first.cases)


# ---------------------------------------------------------------------- gates
class GateTests(unittest.TestCase):
    def records(self, table, *, pass_count=20, speed_pass=None, failure=None):
        rows = []
        for speed in table.speeds:
            for index in range(table.episodes):
                case_id = f'{table.stage}:{table.kind}:{speed:.1f}:{index:03d}'
                record = dict(case_id=case_id, vx=float(speed), episode_index=index,
                              reason='success', mean_vx=float(speed),
                              mean_relative_error=0., smooth_mae_relative=0.,
                              longest_deviation=0, speed_valid=True, speed_pass=True,
                              climb_success=True, failure=False, case_pass=True,
                              joint_success=True)
                if index >= pass_count:
                    record.update(case_pass=False, climb_success=False, reason='timeout')
                if speed_pass is not None and not speed_pass:
                    record.update(speed_pass=False, speed_valid=False, joint_success=False)
                if failure is not None and index == failure:
                    record.update(failure=True, case_pass=False, climb_success=False,
                                  reason='orientation')
                rows.append(record)
        return rows

    def test_stair_gate_requires_18_of_20_per_speed(self):
        table = make_cases('S1', 'validation', 42)
        failed = summarize(self.records(table, pass_count=17), table)
        self.assertFalse(failed['gate_pass'])
        passed = summarize(self.records(table, pass_count=18), table)
        self.assertTrue(passed['gate_pass'])
        self.assertEqual(passed['required'], CASE_PASS_REQUIRED)
        self.assertEqual(passed['required'], 18)

    def test_speed_pass_never_gates_a_stair_climb(self):
        table = make_cases('S1', 'validation', 42)
        summary = summarize(self.records(table, speed_pass=False), table)
        self.assertTrue(summary['gate_pass'])
        self.assertTrue(all(entry['speed_pass_count'] == 0
                            for entry in summary['per_speed']))

    def test_failure_enters_the_denominator(self):
        table = make_cases('S1', 'validation', 42)
        summary = summarize(self.records(table, pass_count=18, failure=0), table)
        for entry in summary['per_speed']:
            self.assertEqual(entry['total'], 20)
            self.assertEqual(entry['case_pass_count'], 17)
            self.assertEqual(entry['failure_count'], 1)
        self.assertFalse(summary['gate_pass'])

    def test_final_test_requires_95_of_100(self):
        table = make_cases('S5', 'test', 42)
        self.assertEqual(table.episodes, 100)
        self.assertTrue(summarize(self.records(table, pass_count=95), table)['gate_pass'])
        self.assertFalse(summarize(self.records(table, pass_count=94), table)['gate_pass'])
        self.assertEqual(summarize(self.records(table, pass_count=95), table)['required'],
                         CASE_PASS_REQUIRED_FINAL)

    def test_f0_case_pass_definition_is_not_relaxed(self):
        raw = dict(body_vx=np.full(500, .4, dtype=np.float32),
                   executed_c=np.full(500, .4, dtype=np.float32),
                   local_x=(-1. + .4 * DT * np.arange(500)).astype(np.float32),
                   local_y=np.zeros(500, dtype=np.float32),
                   yaw=np.zeros(500, dtype=np.float32),
                   route_error=np.tile(np.array([0., 0., 1.], dtype=np.float32), (500, 1)),
                   step=np.arange(1, 501, dtype=np.int64))
        metrics = cpu_speed_oracle(raw, 'f0_time', .4, 0)
        identity = dict(measurement='flat', measurement_stage='F0')
        case = dict(case_id='F0:validation:0.4:000', vx=.4, episode_index=0,
                    initial_pose=[-1., 0., 0.])
        record = _record_for_case(case, raw, failure=False, climb_success=False,
                                  reason='observed_500_steps',
                                  terminal=dict(failure=False, climb_success=False,
                                                reason='observed_500_steps'),
                                  metrics=metrics, identity=identity)
        self.assertTrue(record['case_pass'])
        self.assertFalse(record['climb_success'])
        self.assertIsNone(record['joint_success'])
        # Lateral 0.30 m over the 500 steps breaks the F0 constraint.
        off = dict(raw, local_y=np.full(500, .30, dtype=np.float32))
        off['route_error'] = np.tile(np.array(route_of(.30, 0.), dtype=np.float32), (500, 1))
        metrics = cpu_speed_oracle(off, 'f0_time', .4, 0)
        record = _record_for_case(case, off, failure=False, climb_success=False,
                                  reason='observed_500_steps',
                                  terminal=dict(failure=False, climb_success=False),
                                  metrics=metrics, identity=identity)
        self.assertFalse(record['case_pass'])
        self.assertFalse(record['lateral_heading_pass'])
        # 14 degrees of heading also breaks it (threshold 10 deg in radians).
        turn = dict(raw, yaw=np.full(500, math.radians(14.), dtype=np.float32))
        turn['route_error'] = np.tile(np.array(route_of(0., math.radians(14.)),
                                               dtype=np.float32), (500, 1))
        metrics = cpu_speed_oracle(turn, 'f0_time', .4, 0)
        record = _record_for_case(case, turn, failure=False, climb_success=False,
                                  reason='observed_500_steps',
                                  terminal=dict(failure=False, climb_success=False),
                                  metrics=metrics, identity=identity)
        self.assertFalse(record['case_pass'])
        # 480 observed steps is not the full 500-step F0 window.
        short = {key: value[:480] for key, value in raw.items()}
        metrics = cpu_speed_oracle(short, 'f0_time', .4, 0)
        record = _record_for_case(case, short, failure=False, climb_success=False,
                                  reason='observed_500_steps',
                                  terminal=dict(failure=False, climb_success=False),
                                  metrics=metrics, identity=identity)
        self.assertFalse(record['case_pass'])

    def test_gate_pass_counts_projection(self):
        table = make_cases('S1', 'validation', 42)
        summary = summarize(self.records(table, pass_count=18), table)
        result = EvalResult(key=('run', 'formal', 'S1', 200, 'validation', 'a' * 64,
                                 table.sha256),
                            summary=dict(summary, evidence_verified=True),
                            raw_paths={}, evidence_sha256={})
        counts = gate_pass_counts(result)
        self.assertTrue(counts['gate_pass'])
        self.assertEqual([entry['case_pass_count'] for entry in counts['speeds']], [18, 18, 18])
        self.assertEqual([entry['total'] for entry in counts['speeds']], [20, 20, 20])

    def test_key_tuple(self):
        key = evaluation_key('run', 'formal', 'S1', 200, 'validation', 'b' * 64, 'c' * 64)
        self.assertEqual(key, ('run', 'formal', 'S1', 200, 'validation', 'b' * 64, 'c' * 64))
        with self.assertRaises(ValueError):
            evaluation_key('run', 'formal', 'S1', 200, 'validation', 'short', 'c' * 64)

    def test_step_limits_come_from_the_spec_single_source(self):
        for stage in spec.STAGES:
            for speed in SPEEDS:
                if stage.N == 0:
                    self.assertEqual(evaluation.evaluation_step_limit(stage.id, speed), 500)
                    continue
                expected = spec.stair_episode_steps(stage.N, stage.h, speed)
                self.assertEqual(evaluation.evaluation_step_limit(stage.id, speed), expected,
                                 (stage.id, speed))
        self.assertEqual(evaluation.training_episode_limit(0, 0., .4), spec.F0_EPISODE_STEPS)
        for n, h, c, expected in ((4, .08, .4, 1266), (40, .2, .2, 7257),
                                  (40, .2, .6, 2753)):
            self.assertEqual(evaluation.training_episode_limit(n, h, c), expected)


# ------------------------------------------------------------ fake execution
class FakeModel:
    def __init__(self):
        self.tensor = torch.zeros(3)

    def __call__(self, obs):
        return torch.zeros(obs['proprio'].shape[0], 12)

    def state_dict(self):
        return {'w': self.tensor.clone()}


class FakeEnv:
    """Deterministic straight-line stand-in for :class:`V05R2Env`.

    It produces the r2 five-key observation and a *consistent* ``route_error``
    so the retained-evidence verification path (which re-derives the route from
    ``local_y``/``yaw``) is exercised for real.
    """

    fail_rows = set()
    fail_after = None
    stuck_rows = set()
    zero_denominator_rows = set()
    stall_fractions = {}
    timeout_after = 10 ** 9
    nan_diagnostics = False

    def __init__(self, num_envs, device, seed, stage, stage_iteration=0, evaluation=True,
                 fixed_vx=None, initial_cases=None, render_mode=None,
                 parking_diagnostic=False):
        if not initial_cases or len(initial_cases) != num_envs:
            raise ValueError('fake env needs one initial pose per environment')
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.stage = stage
        self.stage_spec = spec.STAGE_BY_ID[stage]
        self.c = float(fixed_vx)
        self.initial = np.asarray(initial_cases, dtype=np.float64)
        self.x = self.initial[:, 0].copy()
        self.y = self.initial[:, 1].copy()
        self.yaw = self.initial[:, 2].copy()
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.done = np.zeros(num_envs, dtype=bool)
        self.terminal_records = []
        self.tile_N = [self.stage_spec.N] * num_envs
        self.checkpoint_identity = None

    def _obs(self):
        route = torch.tensor(np.stack([np.clip(self.y / .20, -4., 4.),
                                       np.sin(self.yaw), np.cos(self.yaw)], -1),
                             dtype=torch.float32)
        return {'proprio': torch.zeros(self.num_envs, 45),
                'history': torch.zeros(self.num_envs, 5, 45),
                'heightmap': torch.zeros(self.num_envs, 3, 48, 32),
                'privileged': torch.zeros(self.num_envs, 35),
                'route_error': route}

    def reset(self):
        self.terminal_records = []
        return self._obs(), {}

    def step(self, action):
        self.terminal_records = []
        for row in range(self.num_envs):
            if self.done[row]:
                continue
            self.steps[row] += 1
            speed = 0. if row in self.stuck_rows else self.c
            self.x[row] += speed * DT
            reached = self.stage_spec.N > 0 and self.x[row] >= self.stage_spec.N * .2 + .8
            failed = (self.fail_after is not None and row in self.fail_rows
                      and self.steps[row] >= self.fail_after)
            timed_out = (not reached and not failed
                         and self.steps[row] >= self.timeout_after)
            if reached or failed or timed_out:
                self.done[row] = True
                active = int(self.steps[row]) - (1 if reached else 0)
                if row in self.zero_denominator_rows:
                    active = 0
                stall = int(round(active * self.stall_fractions.get(row, 0.)))
                self.terminal_records.append(dict(
                    env_id=row, terminal_step=int(self.steps[row]),
                    duration=float(self.steps[row]) * DT,
                    reason=('orientation' if failed else
                            'timeout' if timed_out else 'success'),
                    climb_success=bool(reached), climb_event=bool(reached),
                    climb_step=int(self.steps[row]) if reached else None,
                    failure=bool(failed), timeout=bool(timed_out), truncated=bool(timed_out),
                    active_samples=active, stall_active_samples=stall,
                    first_step_acquired=bool(reached and row not in self.stuck_rows),
                    initial_pose=list(self.initial[row]), vx=self.c, stage=self.stage,
                    N=int(self.stage_spec.N), route_final=route_of(self.y[row],
                                                                   self.yaw[row])))
        active = [int(self.steps[row]) - (1 if (self.stage_spec.N > 0
                  and self.x[row] >= self.stage_spec.N * .2 + .8) else 0)
                  for row in range(self.num_envs)]
        active = [0 if row in self.zero_denominator_rows else value
                  for row, value in enumerate(active)]
        stall = [int(round(value * self.stall_fractions.get(row, 0.)))
                 for row, value in enumerate(active)]
        obs = self._obs()
        diagnostics = dict(
            body_vx=torch.full((self.num_envs,), self.c),
            active_samples=torch.tensor(active),
            stall_active_samples=torch.tensor(stall),
            first_step_acquired=torch.tensor(
                [self.stage_spec.N > 0 and self.x[row] >= self.stage_spec.N * .2 + .8
                 and row not in self.stuck_rows for row in range(self.num_envs)]),
            command_c=torch.full((self.num_envs,), self.c),
            local_position=torch.tensor(np.stack((self.x, self.y, np.zeros(self.num_envs)), -1),
                                        dtype=torch.float32),
            heading=torch.tensor(self.yaw, dtype=torch.float32),
            control_step=torch.tensor(self.steps),
            route_error=obs['route_error'])
        reward = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        info = dict(terminal_observation=obs, episodes=self.terminal_records,
                    diagnostics=diagnostics, route_error=obs['route_error'],
                    route_error_terminal=obs['route_error'])
        return obs, reward, terminated, truncated, info

    def diagnostics_snapshot(self, env_id=0):
        metrics = {'window_kind': 'p_position', 'samples': int(self.steps[env_id]),
                   'mean_vx': float(self.c), 'mean_relative_error': 0.}
        if self.nan_diagnostics:
            metrics.update(mean_vx=float('nan'), mean_relative_error=float('inf'),
                           smooth_mae_relative=float('nan'))
        return dict(env_id=env_id, stage=self.stage, step=int(self.steps[env_id]),
                    command_executed=[self.c, 0., 0.],
                    linear_velocity_body=[self.c, 0., 0.], failure=False,
                    success=False, reason='running', speed_metrics=metrics,
                    route_error=route_of(self.y[env_id], self.yaw[env_id]))

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        return None


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.previous = evaluation._runtime_components
        evaluation._runtime_components = lambda: dict(env_class=FakeEnv)
        FakeEnv.fail_rows, FakeEnv.fail_after, FakeEnv.stuck_rows = set(), None, set()
        FakeEnv.zero_denominator_rows = set()
        FakeEnv.stall_fractions = {}
        FakeEnv.nan_diagnostics = False

    def tearDown(self):
        evaluation._runtime_components = self.previous

    def run_case(self, folder, table, *, tag='run', num_envs=8, record_video=False,
                 mode='formal'):
        return evaluate_checkpoint(
            FakeModel(), table, device='cpu', num_envs=num_envs, stage=table.stage,
            kind=table.kind, run_id=tag, global_update=10,
            checkpoint_sha='d' * 64, output_dir=Path(folder) / f'eval_{tag}',
            record_video=record_video, mode=mode)

    def test_padding_slots_are_excluded_from_denominators(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            self.assertEqual(len(result.summary['per_speed']), 3)
            for entry in result.summary['per_speed']:
                self.assertEqual(entry['total'], 20)
                self.assertEqual(entry['case_pass_count'], 20)
            self.assertTrue(result.summary['evidence_verified'])
            manifest = json.loads(Path(result.raw_paths['manifest']).read_text())
            self.assertEqual(manifest['num_envs'], 8)
            self.assertEqual(manifest['raw_schema'], 'v05-r2-lossless-window-evidence-1')
            self.assertEqual(manifest['architecture'], 'v05-route-state-v1')
            self.assertEqual(manifest['route_keys'], ['route_error'])
            self.assertEqual(len(manifest['source_sha256']), 64)

    def test_raw_evidence_holds_route_error_per_case(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertEqual(len(rows), 60)
            for row in rows:
                with np.load(row['raw_file'], allow_pickle=False) as storage:
                    self.assertIn('route_error', storage.files)
                    self.assertEqual(storage['route_error'].dtype, np.float32)
                    self.assertEqual(storage['route_error'].ndim, 2)
                    self.assertEqual(storage['route_error'].shape[1], 3)
                    self.assertEqual(storage['local_y'].dtype, np.float32)
                    self.assertEqual(storage['yaw'].dtype, np.float32)
                    self.assertEqual(storage['step'].dtype, np.int64)
                self.assertEqual(len(row['route_final']), 3)
                self.assertEqual(row['route_yaw_axis'], 0.)
                self.assertIn('route_abs_y_max', row)
                self.assertIn('route_heading_deg_max', row)
                self.assertIn('route_saturation_rate', row)
            # The per-case route values are exactly the pose-derived SDS formula.
            first = rows[0]
            with np.load(first['raw_file'], allow_pickle=False) as storage:
                lo, hi = int(storage['offsets'][first['raw_row']]), int(
                    storage['offsets'][first['raw_row'] + 1])
                route = storage['route_error'][lo:hi]
                y = storage['local_y'][lo:hi]
                yaw = storage['yaw'][lo:hi]
            expected = np.stack([np.clip(y.astype(np.float64) / .20, -4., 4.),
                                 np.sin(yaw.astype(np.float64)),
                                 np.cos(yaw.astype(np.float64))], -1)
            self.assertLessEqual(float(np.max(np.abs(route - expected))), 1e-6)

    def test_report_aggregates_route_without_touching_the_gate(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            route = result.summary['route']
            self.assertEqual(route['nonfinite_count'], 0)
            self.assertGreaterEqual(route['abs_y_max'], 0.)
            self.assertLessEqual(route['normalized_abs_y_max'], 4.)
            self.assertLessEqual(route['saturation_rate'], 1.)
            self.assertIsNotNone(route['abs_y_mean'])
            self.assertEqual(route['yaw_axis'], 0.)
            self.assertEqual(route['lateral_scale'], .20)
            self.assertEqual(route['clip'], 4.)
            self.assertEqual(result.summary['route_keys'], ['route_error'])
            # The stair gate still only counts safe arrivals.
            self.assertTrue(result.summary['gate_pass'])
            self.assertEqual(result.summary['required'], 18)

    def test_failure_is_retained_and_counted(self):
        FakeEnv.fail_rows = {0, 1, 2}
        FakeEnv.fail_after = 5
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            failed = [row for row in rows if row['failure']]
            self.assertTrue(failed)
            self.assertTrue(all(row['case_pass'] is False for row in failed))
            self.assertFalse(result.summary['gate_pass'])
            self.assertTrue(all(entry['case_pass_count'] == 11
                                for entry in result.summary['per_speed']))

    def test_flat_case_pass_requires_500_steps_and_no_failure(self):
        table = small_table('F0', 'validation', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertEqual(len(rows), 60)
            for row in rows:
                self.assertEqual(row['samples'], 400)
                self.assertTrue(row['speed_pass'])
                self.assertTrue(row['case_pass'])
                self.assertFalse(row['climb_success'])
                self.assertIsNone(row['joint_success'])
            self.assertTrue(result.summary['gate_pass'])
            FakeEnv.fail_rows, FakeEnv.fail_after = {1, 2, 3}, 50
            result = self.run_case(folder, table, tag='failed')
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertTrue(any(row['failure'] and not row['case_pass'] for row in rows))

    def test_identical_key_is_idempotent_and_foreign_key_is_rejected(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            first = self.run_case(folder, table)
            report_path = Path(first.raw_paths['manifest']).parent / 'report.json'
            digest = evaluation.file_sha256(report_path)
            second = self.run_case(folder, table)
            self.assertEqual(second.key, first.key)
            self.assertEqual(evaluation.file_sha256(report_path), digest)
            with self.assertRaises(FileExistsError):
                evaluate_checkpoint(FakeModel(), table, device='cpu', num_envs=8,
                                    stage=table.stage, kind=table.kind, run_id='run',
                                    global_update=11, checkpoint_sha='d' * 64,
                                    output_dir=Path(folder) / 'eval_run')

    def test_missing_or_tampered_route_evidence_invalidates_the_result(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            npz = Path(result.raw_paths['npz'][0])
            with np.load(npz, allow_pickle=False) as storage:
                arrays = {key: storage[key] for key in storage.files}
            # (a) a tampered route component must fail the independent recomputation.
            broken = dict(arrays)
            broken['route_error'] = arrays['route_error'].copy()
            broken['route_error'][0, 0] += .5
            np.savez(npz, **broken)
            with self.assertRaises(ValueError):
                verify_evaluation_evidence(result, expected_key=result.key)
            # (b) a route array that is missing altogether is incomplete evidence.
            missing = {key: value for key, value in arrays.items() if key != 'route_error'}
            np.savez(npz, **missing)
            with self.assertRaises(ValueError):
                verify_evaluation_evidence(result, expected_key=result.key)
            # (c) restoring the exact bytes restores verification.
            np.savez(npz, **arrays)
            verify_evaluation_evidence(result, expected_key=result.key)

    def test_missing_case_record_field_invalidates_the_result(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            episodes = Path(result.raw_paths['episodes'])
            lines = episodes.read_text().splitlines()
            row = json.loads(lines[0])
            row['route_abs_y_max'] = 123.0
            lines[0] = json.dumps(row)
            episodes.write_text('\n'.join(lines) + '\n')
            with self.assertRaises(ValueError):
                verify_evaluation_evidence(result, expected_key=result.key)

    def test_diagnostic_result_has_fifteen_per_case_records(self):
        FakeEnv.stall_fractions = {0: .9, 1: .5}
        table = small_table('S1', 'diagnostic', episodes=5)
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table, tag='diag')
            cases = result.summary['diagnostic_cases']
            self.assertEqual(len(cases), 15)
            for case in cases:
                for name in ('c', 'active_samples', 'stall_samples', 'first_step_acquired'):
                    self.assertIn(name, case)
                self.assertTrue(case['denominator_valid'])
            aggregate = result.summary['diagnostic_aggregate']
            self.assertEqual(aggregate['case_count'], 15)
            self.assertTrue(aggregate['evidence_valid'])
            self.assertFalse(result.summary['gate_pass'])
            self.assertFalse(result.summary['gate_eligible'])
            self.assertIsNone(result.summary['required'])

    def test_zero_denominator_diagnostic_case_is_invalid_evidence(self):
        FakeEnv.zero_denominator_rows = {0}
        table = small_table('S1', 'diagnostic', episodes=5)
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table, tag='diagzero')
            cases = result.summary['diagnostic_cases']
            invalid = [case for case in cases if not case['denominator_valid']]
            self.assertEqual(len(invalid), 3)
            for case in invalid:
                self.assertIsNone(case['stall_fraction'])
            self.assertFalse(result.summary['diagnostic_aggregate']['evidence_valid'])

    def test_unavailable_window_metrics_are_null(self):
        FakeEnv.fail_rows, FakeEnv.fail_after = set(range(8)), 5
        table = small_table('F0', 'validation', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            failed = [row for row in rows if row['failure']]
            self.assertTrue(failed)
            for row in failed:
                self.assertEqual(row['samples'], 0)
                self.assertIsNone(row['mean_vx'])
                self.assertFalse(row['case_pass'])
                # Route evidence covers every returned control sample, so it stays
                # readable even when the speed window never started -- and it still
                # cannot change the case result.
                self.assertEqual(row['route_samples'], 5)
                self.assertIsNotNone(row['route_abs_y_mean'])
                self.assertEqual(row['route_nonfinite_count'], 0)
            json.loads((Path(result.raw_paths['manifest']).parent / 'report.json').read_text())

    def test_probe_mode_is_never_gate_eligible(self):
        table = small_table('F0', 'validation', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table, tag='probe', mode='probe')
            self.assertEqual(result.summary['mode'], 'probe')
            self.assertFalse(result.summary['gate_eligible'])
            self.assertFalse(result.summary['gate_pass'])
            self.assertEqual(result.key[1], 'probe')

    def test_evaluation_does_not_consume_training_rng(self):
        table = small_table('P1', 'validation')
        np.random.seed(7)
        torch.manual_seed(7)
        before = (np.random.get_state()[1].copy(), torch.get_rng_state().clone())
        with tempfile.TemporaryDirectory() as folder:
            self.run_case(folder, table)
        after = (np.random.get_state()[1].copy(), torch.get_rng_state().clone())
        self.assertTrue(np.array_equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_missing_route_error_is_rejected_at_the_evaluation_seam(self):
        from src.tasks.velocity.v05_r2.env import validate_observations

        class MissingRouteEnv(FakeEnv):
            def _obs(self):
                obs = super()._obs()
                del obs['route_error']
                return obs

        previous = evaluation._runtime_components
        evaluation._runtime_components = lambda: dict(
            env_class=MissingRouteEnv, validate_observations=validate_observations)
        try:
            table = small_table('P1', 'validation')
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError) as caught:
                    self.run_case(folder, table)
                self.assertIn('route_error', str(caught.exception))
        finally:
            evaluation._runtime_components = previous

    def test_foreign_key_verification_is_rejected(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            other = tuple(result.key[:5]) + ('e' * 64, result.key[6])
            with self.assertRaises(ValueError):
                verify_evaluation_evidence(result, expected_key=other)


class RealR2EnvironmentEvaluationTests(unittest.TestCase):
    """Real MuJoCo/MJWarp **CPU** end-to-end evaluation evidence.

    This is the seam test: the r2 environment supplies the five-key observation
    (``route_error`` included), :func:`evaluate_checkpoint` drives it, and the
    retained NPZ/JSONL/report must contain independently re-derivable route
    evidence.  ``device='cpu'``; no CUDA is initialised.
    """

    def test_real_f0_diagnostic_retains_route_evidence(self):
        table = make_cases('F0', 'diagnostic', 42)
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_checkpoint(
                ZeroModel(), table, device='cpu', num_envs=8, stage='F0',
                kind='diagnostic', run_id='r2-real-e2e', global_update=0,
                checkpoint_sha='f' * 64, output_dir=Path(folder) / 'eval',
                mode='formal', seed=42)
            summary = result.summary
            self.assertTrue(summary['evidence_verified'])
            self.assertEqual(summary['kind'], 'diagnostic')
            self.assertFalse(summary['gate_pass'])
            self.assertIsNone(summary['required'])
            self.assertEqual(summary['route_keys'], ['route_error'])
            self.assertEqual(summary['route']['nonfinite_count'], 0)
            self.assertGreater(summary['route']['samples'], 0)
            self.assertLessEqual(summary['route']['normalized_abs_y_max'], 4.)
            self.assertIsNotNone(summary['route']['abs_y_max'])
            cases = summary['per_case']
            self.assertEqual(len(cases), 15)
            for case in cases:
                self.assertIsInstance(case['active_samples'], int)
                self.assertGreaterEqual(case['stall_samples'], 0)
            manifest = json.loads(Path(result.raw_paths['manifest']).read_text())
            self.assertEqual(manifest['architecture'], 'v05-route-state-v1')
            self.assertEqual(manifest['checkpoint_schema'], 7)
            self.assertEqual(len(manifest['source_sha256']), 64)
            self.assertEqual(manifest['route_state']['yaw_axis'], 0.0)
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertEqual(len(rows), 15)
            for row in rows:
                with np.load(row['raw_file'], allow_pickle=False) as storage:
                    self.assertIn('route_error', storage.files)
                    route = storage['route_error']
                    lo, hi = int(storage['offsets'][row['raw_row']]), int(
                        storage['offsets'][row['raw_row'] + 1])
                    piece = route[lo:hi]
                    self.assertEqual(piece.shape[1], 3)
                    self.assertEqual(piece.dtype, np.float32)
                    self.assertTrue(np.all(np.isfinite(piece)))
                    self.assertLessEqual(float(np.abs(piece[:, 0]).max()), 4.)
                    self.assertTrue(np.allclose(piece[:, 1] ** 2 + piece[:, 2] ** 2,
                                                1., atol=1e-5))
                self.assertIsNotNone(row['route_abs_y_max'])
                self.assertEqual(row['route_nonfinite_count'], 0)


class ZeroModel:
    """Stateless stand-in: the evaluation path must not need model internals."""

    def __init__(self):
        self.tensor = torch.zeros(3)

    def __call__(self, obs):
        return torch.zeros(obs['proprio'].shape[0], 12)

    def state_dict(self):
        return {'w': self.tensor.clone()}


if __name__ == '__main__':
    unittest.main()
