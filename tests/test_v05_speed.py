"""CPU evidence for the v0.5 speed windows (SDS 6.1, 6.2, 7.5)."""
import math
import unittest

import numpy as np
import torch

from src.tasks.velocity.v05 import spec
from src.tasks.velocity.v05.speed import (MEAN_RELATIVE_ERROR_MAX, SMOOTH_MAE_MAX,
                                          MAX_LONGEST_DEVIATION, SPEED_WINDOW,
                                          SpeedWindowState, SlotSpeedRouter,
                                          cpu_speed_oracle, cpu_speed_result,
                                          recompute_npz, tile_window_kind,
                                          window_kind_for_stage)

C = .4
DT = spec.CONTROL_DT


def trace(steps, vx, *, c=None, x0=-1., speed=1., y=None, yaw=None):
    """One synthetic consecutive episode with a constant executed command."""
    steps = np.asarray(steps, dtype=np.int64)
    c = float(vx if c is None else c)
    size = len(steps)
    x = x0 + speed * DT * np.arange(size)
    raw = dict(body_vx=np.full(size, vx, dtype=np.float32),
               executed_c=np.full(size, c, dtype=np.float32),
               local_x=x.astype(np.float32), step=steps)
    if y is not None:
        raw['local_y'] = np.asarray(y, dtype=np.float32)
    if yaw is not None:
        raw['yaw'] = np.asarray(yaw, dtype=np.float32)
    return raw


class F0WindowTests(unittest.TestCase):
    def test_exactly_400_samples_from_steps_101_to_500(self):
        raw = trace(np.arange(1, 501), C, y=np.zeros(500), yaw=np.zeros(500))
        result = cpu_speed_oracle(raw, 'f0_time', C, 0)
        self.assertEqual(result['samples'], 400)
        self.assertEqual(result['start_step'], 101)
        self.assertEqual(result['end_step'], 500)
        self.assertTrue(result['coverage_complete'])
        self.assertTrue(result['valid'])
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['mean_relative_error'], 0., places=6)
        # 25-frame smoothing leaves exactly 400-24 complete frames.
        self.assertEqual(result['window_count'], 400 - (SPEED_WINDOW - 1))
        self.assertEqual(result['longest_deviation'], 0)

    def test_short_episode_cannot_be_valid(self):
        raw = trace(np.arange(1, 500), C, y=np.zeros(499), yaw=np.zeros(499))
        result = cpu_speed_oracle(raw, 'f0_time', C, 0)
        self.assertEqual(result['samples'], 399)
        self.assertFalse(result['coverage_complete'])
        self.assertFalse(result['valid'])
        self.assertFalse(result['passed'])
        self.assertIn('incomplete_coverage', result['invalid_reasons'])

    def test_steps_before_window_do_not_enter_statistics(self):
        raw = trace(np.arange(1, 501), C, y=np.zeros(500), yaw=np.zeros(500))
        raw['body_vx'][:100] = 5.
        result = cpu_speed_oracle(raw, 'f0_time', C, 0)
        self.assertAlmostEqual(result['mean_vx'], float(C), places=6)

    def test_lateral_and_heading_use_radians_over_all_steps(self):
        raw = trace(np.arange(1, 501), C, y=np.zeros(500), yaw=np.full(500, math.radians(20.)))
        result = cpu_speed_oracle(raw, 'f0_time', C, 0)
        # 20 deg > 10 deg limit, expressed in radians and tracked over all 500 steps.
        self.assertFalse(result['lateral_heading_pass'])
        self.assertFalse(result['passed'])
        self.assertIn('lateral_heading', result['invalid_reasons'])
        edge = trace(np.arange(1, 501), C, y=np.zeros(500), yaw=np.full(500, math.radians(10.)))
        self.assertTrue(cpu_speed_oracle(edge, 'f0_time', C, 0)['passed'])

    def test_thresholds(self):
        raw = trace(np.arange(1, 501), C, y=np.zeros(500), yaw=np.zeros(500))
        # Mean 20 % below the command is exactly at the allowed mean error.
        raw['body_vx'][:] = C * (1. - MEAN_RELATIVE_ERROR_MAX)
        result = cpu_speed_oracle(raw, 'f0_time', C, 0)
        self.assertAlmostEqual(result['mean_relative_error'], MEAN_RELATIVE_ERROR_MAX, places=5)
        self.assertTrue(result['passed'])
        raw['body_vx'][:] = C * (1. - MEAN_RELATIVE_ERROR_MAX - 1e-3)
        self.assertFalse(cpu_speed_oracle(raw, 'f0_time', C, 0)['passed'])

    def test_online_state_matches_offline_oracle(self):
        steps = np.arange(1, 501)
        raw = trace(steps, .45, c=C, y=np.zeros(500), yaw=np.zeros(500))
        state = SpeedWindowState(1, 'f0_time', 'cpu')
        for index in range(len(steps)):
            state.update(step=torch.tensor([steps[index]]),
                         local_x=torch.tensor([raw['local_x'][index]]),
                         body_vx=torch.tensor([raw['body_vx'][index]]),
                         executed_c=torch.tensor([C]),
                         num_steps=torch.tensor([0]),
                         local_y=torch.tensor([raw['local_y'][index]]),
                         yaw=torch.tensor([raw['yaw'][index]]))
        metrics = state.snapshot()
        offline = cpu_speed_oracle(raw, 'f0_time', C, 0)
        self.assertEqual(int(metrics['samples'][0]), offline['samples'])
        self.assertAlmostEqual(float(metrics['mean_relative_error'][0]),
                               offline['mean_relative_error'], places=12)
        self.assertAlmostEqual(float(metrics['smooth_mae_relative'][0]),
                               offline['smooth_mae_relative'], places=12)
        self.assertEqual(int(metrics['longest_deviation'][0]), offline['longest_deviation'])
        self.assertEqual(bool(metrics['passed'][0]), offline['passed'])


class PositionWindowTests(unittest.TestCase):
    def stair_trace(self, *, n=4, c=.4, vx=None, total=2000, roll_back=False):
        steps = np.arange(1, total + 1)
        vx = c if vx is None else vx
        raw = trace(steps, vx, c=c, x0=-1., speed=vx)
        if roll_back:
            # Cross the entry line, fall back below it, then climb again.
            raw['local_x'][10:40] = np.linspace(raw['local_x'][10], -.8, 30)
            raw['local_x'][40:] = np.linspace(-.8, raw['local_x'][-1], total - 40)
        return raw

    def test_p_window_enter_exit_and_latched_frames(self):
        raw = self.stair_trace(roll_back=True)
        result = cpu_speed_oracle(raw, 'p_position', C, 1)
        enter = int(np.flatnonzero(raw['local_x'] >= -.5)[0])
        exit_ = int(np.flatnonzero(raw['local_x'] >= 1 * .2 + .5)[0])
        self.assertEqual(result['start_step'], int(raw['step'][enter]))
        self.assertEqual(result['end_step'], int(raw['step'][exit_]))
        # Every consecutive frame between entry and exit is counted, including the
        # frames where the body moved back below the entry line.
        self.assertEqual(result['samples'], exit_ - enter + 1)
        self.assertGreater(result['samples'], 25)

    def test_p_window_unfinished_and_short_windows(self):
        raw = self.stair_trace(total=200)
        result = cpu_speed_oracle(raw, 'p_position', C, 1)
        self.assertFalse(result['coverage_complete'])
        self.assertFalse(result['valid'])
        self.assertIn('not_ended', result['invalid_reasons'])
        # A window with fewer than 25 samples has no smoothed metric at all.
        tiny = dict(body_vx=np.array([C, C, C], dtype=np.float32),
                    executed_c=np.array([C, C, C], dtype=np.float32),
                    local_x=np.array([-.6, -.4, -.2], dtype=np.float32),
                    step=np.array([1, 2, 3], dtype=np.int64))
        metrics = cpu_speed_oracle(tiny, 'p_position', C, 1)
        self.assertEqual(metrics['samples'], 2)
        self.assertIsNone(metrics['smooth_mae_relative'])
        self.assertFalse(metrics['valid'])
        self.assertIn('insufficient_samples', metrics['invalid_reasons'])

    def test_p1_high_speed_window_is_long_enough(self):
        # P1 at 0.6 m/s: the window spans 1.2 m = 100 control samples >= 25.
        raw = self.stair_trace(n=1, c=.6, total=300)
        result = cpu_speed_oracle(raw, 'p_position', .6, 1)
        self.assertGreaterEqual(result['samples'], 100)
        self.assertTrue(result['valid'])
        self.assertTrue(result['passed'])
        from src.tasks.velocity.v05.evaluation import evaluation_step_limit
        self.assertGreaterEqual(evaluation_step_limit('P1', .6), result['samples'])

    def test_s_oracle_matches_the_frozen_v03_oracle(self):
        from src.tasks.velocity.v03.speed import cpu_speed_oracle as legacy
        for vx in (.2, .4, .6):
            raw = self.stair_trace(n=4, c=vx, total=3000)
            new = cpu_speed_oracle(raw, 's_oracle', vx, 4)
            old = legacy(raw['body_vx'], raw['executed_c'], raw['local_x'], raw['step'],
                         num_steps=4, input_kind='trajectory')
            self.assertEqual(new['entered'], old['entered'], vx)
            self.assertEqual(new['ended'], old['ended'], vx)
            for key in ('sample_count', 'start_step', 'end_step', 'window_count',
                        'speed_valid', 'speed_pass'):
                self.assertEqual(new[key], old[key], (vx, key))
            for key in ('mean_vx', 'mean_relative_error', 'smooth_mae_relative',
                        'longest_deviation_seconds'):
                self.assertLessEqual(abs(float(new[key]) - float(old[key])), 1e-6, (vx, key))

    def test_s_route_matches_online_state(self):
        raw = self.stair_trace(n=4, c=C, total=3000)
        state = SpeedWindowState(1, 's_oracle', 'cpu')
        for index in range(len(raw['step'])):
            state.update(step=torch.tensor([raw['step'][index]]),
                         local_x=torch.tensor([raw['local_x'][index]]),
                         body_vx=torch.tensor([raw['body_vx'][index]]),
                         executed_c=torch.tensor([C]),
                         num_steps=torch.tensor([4]))
        offline = cpu_speed_oracle(raw, 's_oracle', C, 4)
        metrics = state.snapshot()
        self.assertEqual(int(metrics['samples'][0]), offline['samples'])
        self.assertAlmostEqual(float(metrics['smooth_mae_relative'][0]),
                               offline['smooth_mae_relative'], places=12)
        self.assertEqual(bool(metrics['passed'][0]), offline['passed'])

    def test_flat_n0_never_reaches_a_position_window(self):
        self.assertEqual(tile_window_kind(0), 'f0_time')
        raw = trace(np.arange(1, 60), C)
        with self.assertRaises(ValueError):
            cpu_speed_oracle(raw, 's_oracle', C, 0)
        with self.assertRaises(ValueError):
            cpu_speed_oracle(raw, 'p_position', C, 0)
        state = SpeedWindowState(1, 'p_position', 'cpu')
        with self.assertRaises(ValueError):
            state.update(step=torch.tensor([1]), local_x=torch.tensor([0.]),
                         body_vx=torch.tensor([C]), executed_c=torch.tensor([C]),
                         num_steps=torch.tensor([0]))
        # A flat environment always routes to the F0 time window.
        router = SlotSpeedRouter(1, np.array([0], dtype=np.int64), 'cpu')
        self.assertEqual(router.kind_names, ['f0_time'])

    def test_router_uses_each_real_tile_including_previous(self):
        router = SlotSpeedRouter(3, np.array([0, 4, 5], dtype=np.int64), 'cpu')
        self.assertEqual(router.kind_names, ['f0_time', 'p_position', 's_oracle'])
        # tile 4 is P4 (N=1) even while the supervisor stage is S1 (N=4).
        self.assertEqual(router.N.tolist(), [0, 1, 4])
        self.assertEqual(window_kind_for_stage('S1'), 's_oracle')
        self.assertEqual(window_kind_for_stage('P1'), 'p_position')
        self.assertEqual(window_kind_for_stage('F0'), 'f0_time')
        for step in range(1, 200):
            x = -1. + .4 * DT * (step - 1)
            x = np.array([x, x, x], dtype=np.float32)
            router.update(step=torch.tensor([step] * 3), local_x=torch.from_numpy(x),
                          body_vx=torch.full((3,), .4), executed_c=torch.full((3,), .4),
                          local_y=torch.zeros(3), yaw=torch.zeros(3))
        snapshot = router.snapshot()
        # Steps 101..199 are inside the F0 window, so 99 samples are retained.
        self.assertEqual(int(snapshot['f0_time']['samples'][0]), 99)
        self.assertGreater(int(snapshot['p_position']['samples'][1]), 25)
        self.assertGreater(int(snapshot['s_oracle']['samples'][2]), 25)
        self.assertEqual(int(snapshot['p_position']['samples'][0]), 0)
        metrics = router.episode_metrics(1)
        self.assertEqual(metrics['window_kind'], 'p_position')
        self.assertEqual(metrics['num_steps'], 1)

    def test_partial_reset_leaves_other_environments(self):
        state = SpeedWindowState(2, 's_oracle', 'cpu')
        for step in range(1, 40):
            x = -1. + .4 * DT * (step - 1)
            state.update(step=torch.tensor([step, step]),
                         local_x=torch.tensor([x, x]), body_vx=torch.tensor([C, C]),
                         executed_c=torch.tensor([C, C]), num_steps=torch.tensor([4, 4]))
        before = state.snapshot()
        state.reset(torch.tensor([0], dtype=torch.int64))
        after = state.snapshot()
        self.assertEqual(int(after['samples'][0]), 0)
        self.assertEqual(int(after['samples'][1]), int(before['samples'][1]))
        self.assertEqual(int(after['control_samples'][1]), int(before['control_samples'][1]))


class NoneMetricTests(unittest.TestCase):
    """SDS 6.1: an unavailable metric is null, never NaN (JSON allow_nan=False)."""

    def test_empty_window_metrics_are_none(self):
        import json as _json
        state = SpeedWindowState(1, 'f0_time', 'cpu')
        state.update(step=torch.tensor([1]), local_x=torch.tensor([-1.]),
                     body_vx=torch.tensor([C]), executed_c=torch.tensor([C]),
                     num_steps=torch.tensor([0]), local_y=torch.tensor([0.]),
                     yaw=torch.tensor([0.]))
        result = state.result(0)
        self.assertEqual(result.samples, 0)
        self.assertIsNone(result.mean_vx)
        self.assertIsNone(result.mean_relative_error)
        self.assertIsNone(result.smooth_mae_relative)
        self.assertIsNone(result.longest_deviation)
        self.assertFalse(result.valid)
        _json.dumps(result.as_dict(), allow_nan=False)
        self.assertIsNone(result.as_dict()['mean_vx'])
        router = SlotSpeedRouter(1, np.array([0], dtype=np.int64), 'cpu')
        router.update(step=torch.tensor([1]), local_x=torch.tensor([-1.]),
                      body_vx=torch.tensor([C]), executed_c=torch.tensor([C]),
                      local_y=torch.tensor([0.]), yaw=torch.tensor([0.]))
        self.assertIsNone(router.results()[0].mean_vx)
        _json.dumps(router.episode_metrics(0), allow_nan=False)

    def test_short_window_result_and_cpu_result_are_none(self):
        raw = trace(np.arange(1, 10), C, y=np.zeros(9), yaw=np.zeros(9))
        result = cpu_speed_result(raw, 'f0_time', C, 0)
        self.assertEqual(result.samples, 0)
        self.assertIsNone(result.mean_vx)
        self.assertIsNone(result.mean_relative_error)
        oracle = cpu_speed_oracle(raw, 'f0_time', C, 0)
        self.assertIsNone(oracle['mean_vx'])
        self.assertIsNone(oracle['mean_relative_error'])
        self.assertIsNone(oracle['smooth_mae_relative'])
        import json as _json
        _json.dumps(oracle, allow_nan=False)


class OracleContractTests(unittest.TestCase):
    def test_speed_result_dataclass_and_recompute_npz(self):
        import tempfile
        from pathlib import Path
        raw = trace(np.arange(1, 501), C, y=np.zeros(500), yaw=np.zeros(500))
        result = cpu_speed_result(raw, 'f0_time', C, 0)
        self.assertEqual(result.window_kind, 'f0_time')
        self.assertEqual(result.samples, 400)
        self.assertTrue(result.valid and result.passed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.npz'
            np.savez(path, offsets=np.array([0, 500], dtype=np.int64),
                     **{key: np.asarray(value) for key, value in raw.items()})
            recomputed = recompute_npz(path, kind='f0_time', c=C, N=0, case_id=0)
            self.assertEqual(recomputed['samples'], 400)
            self.assertAlmostEqual(recomputed['mean_vx'], float(C), places=6)
            self.assertLessEqual(abs(recomputed['mean_relative_error'] - result.mean_relative_error), 1e-9)

    def test_invalid_inputs_are_rejected(self):
        raw = trace(np.arange(1, 60), C)
        nonfinite = dict(raw, body_vx=np.full(59, np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            cpu_speed_oracle(nonfinite, 's_oracle', C, 4)
        changed = dict(raw, executed_c=np.full(59, .3, dtype=np.float32))
        with self.assertRaises(ValueError):
            cpu_speed_oracle(changed, 's_oracle', C, 4)
        gaps = dict(raw, step=np.arange(1, 60) * 2)
        with self.assertRaises(ValueError):
            cpu_speed_oracle(gaps, 's_oracle', C, 4)
        with self.assertRaises(ValueError):
            cpu_speed_oracle(raw, 's_oracle', C, -1)


if __name__ == '__main__':
    unittest.main()
