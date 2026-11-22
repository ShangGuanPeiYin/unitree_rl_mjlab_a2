"""CPU evidence for v0.5 case tables, gate arithmetic and retained raw evidence.

The executor test substitutes a deterministic fake environment/model so the whole
``evaluate_checkpoint`` path (padding mask, float32 NPZ, JSONL, CPU recomputation,
evidence verification, idempotent rerun) runs without a simulator.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.tasks.velocity.v05 import spec
from src.tasks.velocity.v05 import evaluation
from src.tasks.velocity.v05.evaluation import (CASE_PASS_REQUIRED, CASE_PASS_REQUIRED_FINAL,
                                               CaseTable, EvalResult, case_table_sha,
                                               evaluate_checkpoint, evaluation_key,
                                               gate_pass_counts, make_cases,
                                               table_from_json, verify_evaluation_evidence,
                                               _record_for_case, summarize)
from src.tasks.velocity.v05.speed import cpu_speed_oracle

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

    def test_initial_pose_ranges(self):
        f0 = make_cases('F0', 'validation', 42)
        p1 = make_cases('P1', 'validation', 42)
        for table in (f0, p1):
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
        perturbed = make_cases('S1', 'perturbed', 42)
        entry = perturbed.arrays['0.4']
        self.assertTrue(all(-1.1 <= x <= -.9 for x in entry['x']))
        self.assertTrue(all(-.05 <= y <= .05 for y in entry['y']))
        self.assertTrue(all(-math.radians(5) <= a <= math.radians(5) for a in entry['yaw']))

    def test_tables_are_deterministic_and_sha_stable(self):
        first = make_cases('S1', 'validation', 42)
        second = make_cases('S1', 'validation', 42)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.arrays, second.arrays)
        self.assertEqual(len(first.sha256), 64)
        self.assertNotEqual(first.sha256, make_cases('S1', 'validation', 43).sha256)
        self.assertNotEqual(make_cases('S1', 'validation', 42).sha256,
                            make_cases('S2', 'validation', 42).sha256)
        restored = table_from_json(first.to_json())
        self.assertEqual(restored.sha256, first.sha256)
        self.assertEqual(restored.cases, first.cases)

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
                    record['case_pass'] = False
                    record['climb_success'] = False
                    record['reason'] = 'timeout'
                if speed_pass is not None and not speed_pass:
                    record['speed_pass'] = False
                    record['speed_valid'] = False
                    record['joint_success'] = False
                if failure is not None and index == failure:
                    record['failure'] = True
                    record['case_pass'] = False
                    record['climb_success'] = False
                    record['reason'] = 'orientation'
                rows.append(record)
        return rows

    def test_stair_gate_requires_18_of_20_per_speed(self):
        table = make_cases('S1', 'validation', 42)
        failed = summarize(self.records(table, pass_count=17), table)
        self.assertFalse(failed['gate_pass'])
        self.assertTrue(all(entry['case_pass_count'] == 17 for entry in failed['per_speed']))
        passed = summarize(self.records(table, pass_count=18), table)
        self.assertTrue(passed['gate_pass'])
        self.assertEqual(passed['required'], CASE_PASS_REQUIRED)

    def test_speed_pass_never_gates_a_stair_climb(self):
        table = make_cases('S1', 'validation', 42)
        summary = summarize(self.records(table, speed_pass=False), table)
        self.assertTrue(summary['gate_pass'])
        self.assertTrue(all(entry['speed_pass_count'] == 0 for entry in summary['per_speed']))
        self.assertTrue(all(record['joint_success'] is False
                            for entry in summary['per_speed'] for record in entry['records']))

    def test_failure_enters_the_denominator(self):
        table = make_cases('S1', 'validation', 42)
        # 18 nominal passes, but one of them is a real failure: 17 < 18 => fail.
        summary = summarize(self.records(table, pass_count=18, failure=0), table)
        for entry in summary['per_speed']:
            self.assertEqual(entry['total'], 20)
            self.assertEqual(entry['case_pass_count'], 17)
            self.assertEqual(entry['failure_count'], 1)
            self.assertEqual(entry['failure_reasons'].get('orientation'), 1)
        self.assertFalse(summary['gate_pass'])

    def test_final_test_requires_95_of_100(self):
        table = make_cases('S5', 'test', 42)
        self.assertEqual(table.episodes, 100)
        self.assertTrue(summarize(self.records(table, pass_count=95), table)['gate_pass'])
        self.assertFalse(summarize(self.records(table, pass_count=94), table)['gate_pass'])
        self.assertEqual(summarize(self.records(table, pass_count=95), table)['required'],
                         CASE_PASS_REQUIRED_FINAL)

    def test_flat_case_pass_definition(self):
        raw = dict(body_vx=np.full(500, .4, dtype=np.float32),
                   executed_c=np.full(500, .4, dtype=np.float32),
                   local_x=(-1. + .4 * DT * np.arange(500)).astype(np.float32),
                   local_y=np.zeros(500, dtype=np.float32),
                   yaw=np.zeros(500, dtype=np.float32),
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
        # Failure at step 200 is still a failed case even though the speed window
        # (steps 101..500) is incomplete rather than absent.
        short = {key: value[:200] for key, value in raw.items()}
        metrics = cpu_speed_oracle(short, 'f0_time', .4, 0)
        record = _record_for_case(case, short, failure=True, climb_success=False,
                                  reason='orientation',
                                  terminal=dict(failure=True, climb_success=False,
                                                reason='orientation'),
                                  metrics=metrics, identity=identity)
        self.assertFalse(record['case_pass'])
        self.assertFalse(record['speed_valid'])
        # A safe stair arrival without the speed gate still passes the case.
        identity = dict(measurement='stair', measurement_stage='S1')
        stair_raw = dict(body_vx=np.zeros(100, dtype=np.float32),
                         executed_c=np.full(100, .4, dtype=np.float32),
                         local_x=(-1. + .1 * DT * np.arange(100)).astype(np.float32),
                         local_y=np.zeros(100, dtype=np.float32),
                         yaw=np.zeros(100, dtype=np.float32),
                         step=np.arange(1, 101, dtype=np.int64))
        metrics = cpu_speed_oracle(stair_raw, 's_oracle', .4, 4)
        record = _record_for_case(case, stair_raw, failure=False, climb_success=True,
                                  reason='success',
                                  terminal=dict(failure=False, climb_success=True,
                                                reason='success'),
                                  metrics=metrics, identity=identity)
        self.assertTrue(record['case_pass'])
        self.assertFalse(record['speed_pass'])
        self.assertFalse(record['joint_success'])

    def test_gate_pass_counts_projection(self):
        table = make_cases('S1', 'validation', 42)
        summary = summarize(self.records(table, pass_count=18), table)
        result = EvalResult(key=('run', 'formal', 'S1', 200, 'validation', 'a' * 64, table.sha256),
                            summary=dict(summary, evidence_verified=True),
                            raw_paths={}, evidence_sha256={})
        counts = gate_pass_counts(result)
        self.assertTrue(counts['gate_pass'])
        self.assertEqual([entry['case_pass_count'] for entry in counts['speeds']], [18, 18, 18])
        self.assertEqual([entry['total'] for entry in counts['speeds']], [20, 20, 20])
        self.assertTrue(counts['evidence_verified'])

    def test_key_tuple(self):
        key = evaluation_key('run', 'formal', 'S1', 200, 'validation', 'b' * 64, 'c' * 64)
        self.assertEqual(key, ('run', 'formal', 'S1', 200, 'validation', 'b' * 64, 'c' * 64))
        with self.assertRaises(ValueError):
            evaluation_key('run', 'formal', 'S1', 200, 'validation', 'short', 'c' * 64)

    def test_step_limits_come_from_the_spec_single_source(self):
        # (1) Ten stages x three speeds: the stair bound is spec.stair_episode_steps
        # verbatim -- no +2 protocol margin anywhere.
        for stage in spec.STAGES:
            for speed in SPEEDS:
                if stage.N == 0:
                    self.assertEqual(evaluation.evaluation_step_limit(stage.id, speed), 500)
                    continue
                expected = spec.stair_episode_steps(stage.N, stage.h, speed)
                self.assertEqual(evaluation.evaluation_step_limit(stage.id, speed), expected,
                                 (stage.id, speed))
                self.assertEqual(evaluation.training_episode_limit(stage.N, stage.h, speed),
                                 expected, (stage.id, speed))
                # A deliberate off-by-one guard: the old code added a margin.
                self.assertNotEqual(evaluation.evaluation_step_limit(stage.id, speed),
                                    expected + 2, (stage.id, speed))
        # (2) The F0 observation window is 500 steps (SDS 6.2) while the flat
        # *episode* cap stays the spec 1500 steps: two distinct concepts.
        self.assertEqual(spec.F0_OBSERVE_STEPS, 500)
        self.assertEqual(evaluation.training_episode_limit(0, 0., .4), spec.F0_EPISODE_STEPS)
        self.assertEqual(evaluation.training_episode_limit(0, 0., .2), 1500)
        self.assertEqual(evaluation.evaluation_step_limit('F0', .2), 500)
        # (3) Reference values from spec are reproduced exactly.
        for n, h, c, expected in ((4, .08, .4, 1266), (40, .2, .2, 7257),
                                  (40, .2, .6, 2753), (40, .2, .4, 3879),
                                  (4, .08, .2, 2031), (4, .08, .6, 1011)):
            self.assertEqual(evaluation.training_episode_limit(n, h, c), expected, (n, h, c))

    def test_environment_timeout_tensor_path_matches_the_spec(self):
        # Cross-implementation guard: the frozen v0.2 tensor formula that env.py
        # actually evaluates must agree with spec.stair_episode_steps for every
        # stage/speed, and flat must be masked to the 1500-step flat cap.
        from src.tasks.velocity.v02_formal.env import timeout_steps
        for stage in spec.STAGES:
            for speed in SPEEDS:
                n = torch.tensor([stage.N])
                h = torch.tensor([stage.h])
                command = torch.tensor([float(speed)])
                stairs = torch.tensor([stage.N > 0])
                # Exactly the expression v05/env.py:_task_state evaluates.
                effective = int(torch.where(
                    stairs, timeout_steps(n, h, command),
                    torch.full_like(n, spec.F0_EPISODE_STEPS))[0])
                self.assertEqual(effective,
                                 spec.episode_step_limit(stage.N, stage.h, speed),
                                 (stage.id, speed))
                if stage.N == 0:
                    # The bare stair formula on N=0/h=0 is NOT the flat cap; env.py
                    # masks it, so the flat behaviour stays 1500.  Recorded so a
                    # future refactor cannot unmask it silently.
                    self.assertNotEqual(int(timeout_steps(n, h, command)[0]), 1500)
                else:
                    self.assertEqual(int(timeout_steps(n, h, command)[0]),
                                     spec.stair_episode_steps(stage.N, stage.h, speed),
                                     (stage.id, speed))


class FakeModel:
    def __init__(self):
        self.tensor = torch.zeros(3)

    def __call__(self, obs):
        return torch.zeros(obs['proprio'].shape[0], 12)

    def state_dict(self):
        return {'w': self.tensor.clone()}


class FakeEnv:
    """Deterministic straight-line stand-in for :class:`V05Env`."""

    fail_rows = set()
    fail_after = None
    stuck_rows = set()
    zero_denominator_rows = set()
    stall_fractions = {}
    timeout_after = 10 ** 9
    nan_diagnostics = False
    video = False

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

    def reset(self):
        obs = {'proprio': torch.zeros(self.num_envs, 45),
               'history': torch.zeros(self.num_envs, 5, 45),
               'heightmap': torch.zeros(self.num_envs, 3, 48, 32),
               'privileged': torch.zeros(self.num_envs, 35)}
        self.terminal_records = []
        return obs, {}

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
                    N=int(self.stage_spec.N)))
        active = [int(self.steps[row]) - (1 if (self.stage_spec.N > 0
                  and self.x[row] >= self.stage_spec.N * .2 + .8) else 0)
                  for row in range(self.num_envs)]
        active = [0 if row in self.zero_denominator_rows else value
                  for row, value in enumerate(active)]
        stall = [int(round(value * self.stall_fractions.get(row, 0.)))
                 for row, value in enumerate(active)]
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
            control_step=torch.tensor(self.steps))
        obs = {'proprio': torch.zeros(self.num_envs, 45),
               'history': torch.zeros(self.num_envs, 5, 45),
               'heightmap': torch.zeros(self.num_envs, 3, 48, 32),
               'privileged': torch.zeros(self.num_envs, 35)}
        reward = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        info = dict(terminal_observation=obs, episodes=self.terminal_records,
                    diagnostics=diagnostics)
        return obs, reward, terminated, truncated, info

    def diagnostics_snapshot(self, env_id=0):
        metrics = {'window_kind': 'p_position', 'samples': int(self.steps[env_id]),
                   'mean_vx': float(self.c), 'mean_relative_error': 0.}
        if self.nan_diagnostics:
            # Reproduces the pre-fix real-env state: an empty window produced NaN.
            metrics['mean_vx'] = float('nan')
            metrics['mean_relative_error'] = float('inf')
            metrics['smooth_mae_relative'] = float('nan')
        return dict(env_id=env_id, stage=self.stage, step=int(self.steps[env_id]),
                    command_executed=[self.c, 0., 0.],
                    linear_velocity_body=[self.c, 0., 0.], failure=False,
                    success=False, reason='running', speed_metrics=metrics)

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
            # 20 cases / 8 slots per batch: the final batch keeps 8 physical slots
            # and only 4 active cases.
            manifest = json.loads(Path(result.raw_paths['manifest']).read_text())
            self.assertEqual(manifest['num_envs'], 8)
            with np.load(result.raw_paths['npz'][-1], allow_pickle=False) as storage:
                self.assertEqual(len(storage['offsets']), 4 + 1)  # 4 active + offset 0

    def test_every_case_has_lossless_raw_evidence(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertEqual(len(rows), 60)
            for row in rows:
                with np.load(row['raw_file'], allow_pickle=False) as storage:
                    self.assertEqual(storage['body_vx'].dtype, np.float32)
                    self.assertEqual(storage['step'].dtype, np.int64)
            stored = {row['case_id']: row for row in rows}
            self.assertEqual(len(stored), 60)
            for case in table.cases:
                self.assertIn(case['case_id'], stored)

    def test_failure_is_retained_and_counted(self):
        # Three failures per speed leave 17/20 case passes: below the 18 threshold.
        FakeEnv.fail_rows = {0, 1, 2}
        FakeEnv.fail_after = 5
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in Path(result.raw_paths['episodes']).read_text().splitlines()]
            failed = [row for row in rows if row['failure']]
            self.assertTrue(failed)
            self.assertTrue(all(row['case_pass'] is False for row in failed))
            self.assertFalse(result.summary['gate_pass'])
            self.assertTrue(all(entry['total'] == 20 for entry in result.summary['per_speed']))
            # Failures hit rows 0..2 of every 8-slot batch: 3 batches x 3 = 9 per speed.
            self.assertTrue(all(entry['case_pass_count'] == 11
                                for entry in result.summary['per_speed']))

    def test_flat_case_pass_requires_500_steps_and_no_failure(self):
        table = small_table('F0', 'validation', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            rows = [json.loads(line) for line in Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertEqual(len(rows), 60)
            for row in rows:
                self.assertEqual(row['samples'], 400)
                self.assertTrue(row['speed_pass'])
                self.assertTrue(row['case_pass'])
                self.assertFalse(row['climb_success'])
                self.assertIsNone(row['joint_success'])
            # SDS 9.2-9: an ideal F0 walk passes the gate while climb_success is
            # false and joint_success is null.
            self.assertTrue(result.summary['gate_pass'])
            self.assertTrue(all(entry['passed'] for entry in result.summary['per_speed']))
            # Early failure on a flat case is still a failed case.
            FakeEnv.fail_rows, FakeEnv.fail_after = {1, 2, 3}, 50
            result = self.run_case(folder, table, tag='failed')
            rows = [json.loads(line) for line in Path(result.raw_paths['episodes']).read_text().splitlines()]
            self.assertTrue(any(row['failure'] and not row['case_pass'] for row in rows))
            self.assertLessEqual(max(row['samples'] for row in rows if row['failure']), 400)

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

    def test_missing_or_tampered_raw_evidence_invalidates_the_gate(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            self.assertTrue(result.summary['evidence_verified'])
            npz = Path(result.raw_paths['npz'][0])
            original = npz.read_bytes()
            npz.write_bytes(original[:-16])
            with self.assertRaises(Exception):
                verify_evaluation_evidence(result, expected_key=result.key)
            npz.write_bytes(original)
            verify_evaluation_evidence(result, expected_key=result.key)
            episodes = Path(result.raw_paths['episodes'])
            lines = episodes.read_text().splitlines()
            row = json.loads(lines[0])
            row['case_pass'] = not row['case_pass']
            lines[0] = json.dumps(row)
            episodes.write_text('\n'.join(lines) + '\n')
            with self.assertRaises(ValueError):
                verify_evaluation_evidence(result, expected_key=result.key)

    def test_full_twenty_case_gate_passes_on_the_protocol_counts(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            self.assertTrue(result.summary['gate_pass'])
            counts = gate_pass_counts(result)
            self.assertTrue(counts['gate_pass'])
            for entry in counts['speeds']:
                self.assertEqual(entry['total'], 20)
                self.assertEqual(entry['case_pass_count'], 20)

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
                self.assertIsInstance(case['active_samples'], int)
                self.assertIsInstance(case['stall_samples'], int)
                self.assertIsInstance(case['first_step_acquired'], bool)
                self.assertTrue(case['denominator_valid'])
            aggregate = result.summary['diagnostic_aggregate']
            self.assertEqual(aggregate['case_count'], 15)
            self.assertEqual(aggregate['valid_case_count'], 15)
            self.assertEqual(aggregate['zero_denominator_count'], 0)
            self.assertTrue(aggregate['evidence_valid'])
            self.assertEqual(aggregate['stall_fraction_ge_threshold_count'], 3)
            self.assertEqual(aggregate['first_step_acquired_count'], 15)
            self.assertTrue(Path(result.summary['diagnostic_cases_path']).is_file())
            # Runner compatibility: the same rows are exposed as `per_case` and
            # nested under `diagnostic` as well.
            self.assertEqual(result.summary['per_case'], result.summary['diagnostic_cases'])
            self.assertEqual(result.summary['diagnostic']['per_case'], cases)
            self.assertEqual(result.summary['diagnostic']['cases'], cases)
            self.assertEqual(result.summary['diagnostic']['aggregate'],
                             result.summary['diagnostic_aggregate'])
            # A diagnostic never authorizes a promotion.
            self.assertFalse(result.summary['gate_pass'])
            self.assertFalse(result.summary['gate_eligible'])
            self.assertIsNone(result.summary['required'])
            counts = gate_pass_counts(result)
            self.assertFalse(counts['gate_pass'])
            self.assertIsNone(counts['required'])
            collapse = evaluation.diagnostic_collapse_counts(result)
            self.assertEqual(collapse['case_count'], 15)
            self.assertEqual(len(collapse['per_case']), 15)

    def test_flat_diagnostic_uses_pre_reset_denominators(self):
        # F0 has no environment terminal at the 500-step observation cap, so the
        # synthesized terminal must carry the real pre-reset denominators.
        table = small_table('F0', 'diagnostic', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table, tag='diagflat')
            cases = result.summary['diagnostic_cases']
            self.assertEqual(len(cases), 15)
            for case in cases:
                self.assertEqual(case['active_samples'], 500)
                self.assertTrue(case['denominator_valid'])
                self.assertFalse(case['first_step_acquired'])
            aggregate = result.summary['diagnostic_aggregate']
            self.assertEqual(aggregate['zero_denominator_count'], 0)
            self.assertTrue(aggregate['evidence_valid'])
            self.assertFalse(result.summary['gate_pass'])

    def test_zero_denominator_diagnostic_case_is_invalid_evidence(self):
        FakeEnv.zero_denominator_rows = {0}
        table = small_table('S1', 'diagnostic', episodes=5)
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table, tag='diagzero')
            cases = result.summary['diagnostic_cases']
            self.assertEqual(len(cases), 15)
            invalid = [case for case in cases if not case['denominator_valid']]
            self.assertEqual(len(invalid), 3)  # one per speed batch
            for case in invalid:
                self.assertEqual(case['active_samples'], 0)
                self.assertIsNone(case['stall_fraction'])
            aggregate = result.summary['diagnostic_aggregate']
            self.assertEqual(aggregate['zero_denominator_count'], 3)
            self.assertFalse(aggregate['evidence_valid'])
            self.assertEqual(aggregate['valid_case_count'], 12)

    def test_validation_result_has_no_diagnostic_block(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            self.assertIsNone(result.summary['diagnostic_cases'])
            self.assertIsNone(result.summary['diagnostic_cases_path'])
            self.assertIsNone(result.summary['diagnostic_aggregate'])
            self.assertIsNone(result.summary['per_case'])
            self.assertEqual(result.summary['diagnostic'], {})
            with self.assertRaises(ValueError):
                evaluation.diagnostic_collapse_counts(result)

    def test_unavailable_window_metrics_are_null_and_trace_is_json_safe(self):
        # Failure at step 5: the F0 window (steps 101..500) never starts, so every
        # speed metric is unavailable and must be written as null.
        FakeEnv.fail_rows, FakeEnv.fail_after = set(range(8)), 5
        table = small_table('F0', 'validation', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            trace = Path(result.raw_paths['trace'])
            self.assertTrue(trace.is_file())
            lines = [line for line in trace.read_text().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
            for line in lines:
                json.loads(line)  # strict JSON: no NaN/inf
            rows = [json.loads(line) for line in
                    Path(result.raw_paths['episodes']).read_text().splitlines()]
            failed = [row for row in rows if row['failure']]
            self.assertTrue(failed)
            for row in failed:
                self.assertEqual(row['samples'], 0)
                self.assertIsNone(row['mean_vx'])
                self.assertIsNone(row['mean_relative_error'])
                self.assertIsNone(row['smooth_mae_relative'])
                self.assertFalse(row['speed_valid'])
                self.assertFalse(row['speed_pass'])
                self.assertFalse(row['case_pass'])
            counts = gate_pass_counts(result)
            for entry in counts['speeds']:
                self.assertEqual(entry['speed_pass_count'], 0)
                self.assertFalse(entry['passed'])
            for entry in result.summary['per_speed']:
                self.assertEqual(entry['speed_valid_count'], 0)
                self.assertEqual(entry['speed_pass_count'], 0)
                self.assertFalse(entry['passed'])
            json.loads((Path(result.raw_paths['manifest']).parent / 'report.json').read_text())
            self.assertIsInstance(result.summary['nonfinite_fields'], int)

    def test_residual_nonfinite_diagnostics_are_sanitized_and_counted(self):
        FakeEnv.nan_diagnostics = True
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            trace = Path(result.raw_paths['trace']).read_text().splitlines()
            self.assertGreaterEqual(len(trace), 1)
            for line in trace:
                json.loads(line)
            self.assertGreaterEqual(result.summary['nonfinite_fields'], 1)
            json.loads((Path(result.raw_paths['manifest']).parent / 'report.json').read_text())

    def test_jsonable_sanitizes_and_counts_nonfinite_floats(self):
        counter = {'nonfinite': 0}
        value = evaluation.jsonable({'a': float('nan'), 'b': [float('inf'), 1.5],
                                     'c': {'d': float('-inf')}}, counter=counter)
        self.assertEqual(value, {'a': None, 'b': [None, 1.5], 'c': {'d': None}})
        self.assertEqual(counter['nonfinite'], 3)
        with self.assertRaises(ValueError):
            evaluation._check_json_safe({'a': float('nan')})
        self.assertTrue(evaluation._check_json_safe({'a': None}))

    def test_probe_mode_is_never_gate_eligible(self):
        table = small_table('F0', 'validation', measurement_stage='F0', measurement='flat')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table, tag='probe', mode='probe')
            summary = result.summary
            self.assertEqual(summary['mode'], 'probe')
            self.assertTrue(summary['probe'])
            self.assertFalse(summary['counts_against_formal'])
            self.assertFalse(summary['gate_eligible'])
            self.assertFalse(summary['gate_pass'])
            self.assertIn('never gate-eligible', summary['gate_note'])
            self.assertTrue(summary['evidence_verified'])
            self.assertEqual(result.key[1], 'probe')
            counts = gate_pass_counts(result)
            self.assertEqual(counts['mode'], 'probe')
            self.assertTrue(counts['probe'])
            self.assertFalse(counts['gate_eligible'])
            self.assertFalse(counts['gate_pass'])
            report = json.loads((Path(result.raw_paths['manifest']).parent
                                 / 'report.json').read_text())
            self.assertEqual(report['mode'], 'probe')
            self.assertFalse(report['gate_eligible'])
            # The probe-only directory/key never collides with a pilot/formal one.
            with self.assertRaises(FileExistsError):
                evaluate_checkpoint(FakeModel(), table, device='cpu', num_envs=8,
                                    stage=table.stage, kind=table.kind,
                                    run_id='probe', global_update=11,
                                    checkpoint_sha='d' * 64,
                                    output_dir=Path(folder) / 'eval_probe',
                                    mode='probe')

    def test_probe_mode_measures_diagnostic_and_flat_retention(self):
        for kind in ('diagnostic', 'flat_retention'):
            with self.subTest(kind=kind):
                table = (small_table('S1', 'diagnostic') if kind == 'diagnostic'
                         else small_table('S1', 'flat_retention'))
                with tempfile.TemporaryDirectory() as folder:
                    result = self.run_case(folder, table, tag='probe_' + kind,
                                           mode='probe')
                    self.assertEqual(result.summary['mode'], 'probe')
                    self.assertFalse(result.summary['gate_eligible'])
                    self.assertTrue(result.summary['evidence_verified'])
                    if kind == 'diagnostic':
                        self.assertEqual(len(result.summary['per_case']), 15)
                    else:
                        self.assertEqual(result.summary['kind'], 'flat_retention')
                        self.assertEqual(result.summary['measurement_stage'], 'F0')

    def test_bogus_mode_and_gate_kinds_are_rejected(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                self.run_case(folder, table, tag='bogus', mode='bogus')
            with self.assertRaises(ValueError):
                evaluation.check_evaluation_mode('probe', 'test')
            with self.assertRaises(ValueError):
                evaluation.check_evaluation_mode('probe', 'perturbed')
            self.assertFalse(evaluation.check_evaluation_mode('probe', 'validation'))
            self.assertFalse(evaluation.check_evaluation_mode('probe', 'flat_retention'))
            self.assertFalse(evaluation.check_evaluation_mode('probe', 'diagnostic'))

    def test_video_overlay_renders_a_frame(self):
        diagnostics = dict(stage='S1', step=3, command_executed=[.4, 0., 0.],
                           linear_velocity_body=[.4, 0., 0.], failure=False,
                           success=False, reason='running',
                           speed_metrics=dict(window_kind='s_oracle', samples=12,
                                              speed_pass=False))
        frame = evaluation.overlay_frame(np.zeros((60, 120, 3), dtype=np.uint8), diagnostics)
        self.assertEqual(frame.shape, (60, 120, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_foreign_key_verification_is_rejected(self):
        table = small_table('P1', 'validation')
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_case(folder, table)
            other = tuple(result.key[:5]) + ('e' * 64, result.key[6])
            with self.assertRaises(ValueError):
                verify_evaluation_evidence(result, expected_key=other)

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


@unittest.skipUnless(torch.cuda.is_available(),
                     'GPU-only: a real V05Env diagnostic evaluation needs cuda:0')
class RealEnvironmentEndToEndTests(unittest.TestCase):
    """The real environment's info/episodes/diagnostics keys through evaluation.

    Historical B/C seam bugs were key-name mismatches that the kinematic FakeEnv
    could not see, so this test drives a real ``V05Env`` plus a real
    ``V05ActorCritic`` through the complete 15-case F0 diagnostic evaluation.
    """

    def test_real_f0_diagnostic_evidence_contract(self):
        from src.tasks.velocity.v05.model import V05ActorCritic
        table = make_cases('F0', 'diagnostic', 42)
        torch.manual_seed(0)
        model = V05ActorCritic().to('cuda:0').eval()
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_checkpoint(
                model, table, device='cuda:0', num_envs=8, stage='F0',
                kind='diagnostic', run_id='b-real-e2e', global_update=0,
                checkpoint_sha='f' * 64, output_dir=Path(folder) / 'eval',
                mode='formal', seed=42)
            summary = result.summary
            self.assertTrue(summary['evidence_verified'])
            self.assertEqual(summary['kind'], 'diagnostic')
            self.assertFalse(summary['gate_pass'])
            self.assertFalse(summary['gate_eligible'])
            self.assertIsNone(summary['required'])
            self.assertIsInstance(summary['nonfinite_fields'], int)
            cases = summary['per_case']
            self.assertEqual(len(cases), 15)
            self.assertEqual(cases, summary['diagnostic_cases'])
            self.assertEqual(cases, summary['diagnostic']['per_case'])
            for case in cases:
                self.assertIsNotNone(case['active_samples'], case['case_id'])
                self.assertIsNotNone(case['stall_samples'], case['case_id'])
                self.assertIsInstance(case['active_samples'], int)
                self.assertGreaterEqual(case['stall_samples'], 0)
            aggregate = summary['diagnostic_aggregate']
            self.assertGreater(aggregate['valid_case_count'], 0)
            self.assertEqual(aggregate['case_count'], 15)
            self.assertTrue(Path(summary['diagnostic_cases_path']).is_file())
            trace = Path(result.raw_paths['trace'])
            lines = [line for line in trace.read_text().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
            for line in lines:
                json.loads(line)  # strict JSON: no NaN/inf, every line complete
            report = json.loads((Path(result.raw_paths['manifest']).parent
                                 / 'report.json').read_text())
            self.assertTrue(report['evidence_verified'])
            self.assertEqual(len(report['per_case']), 15)


if __name__ == '__main__':
    unittest.main()
