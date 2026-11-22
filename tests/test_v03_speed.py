"""CPU behavioral evidence for inclusive stair segments and speed thresholds."""
import importlib.util
import io
import json
import unittest

import numpy as np
import torch


class SpeedTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('src.tasks.velocity.v03.speed'),
                             'v03 stair speed state is not implemented')
        from src.tasks.velocity.v03.speed import SegmentSpeedState, cpu_speed_oracle
        return SegmentSpeedState, cpu_speed_oracle

    def arrays(self, vx, *, c=.4, x=None, step=None):
        vx = np.asarray(vx, dtype=np.float32)
        size = len(vx)
        return dict(body_vx=vx, c=np.full(size, c, dtype=np.float32),
                    local_x=np.asarray(np.linspace(0., .8, size) if x is None else x, dtype=np.float32),
                    step=np.asarray(np.arange(1, size + 1) if step is None else step, dtype=np.int64))

    def run_trace(self, arrays, *, n=4, enabled=None):
        State, oracle = self.api()
        state = State(1, device='cpu')
        samples = []
        for j in range(len(arrays['body_vx'])):
            samples.append(state.update(
                step=torch.tensor([int(arrays['step'][j])], dtype=torch.int64),
                local_x=torch.tensor(arrays['local_x'][j:j+1], dtype=torch.float32),
                body_vx=torch.tensor(arrays['body_vx'][j:j+1], dtype=torch.float32),
                executed_c=torch.tensor(arrays['c'][j:j+1], dtype=torch.float32),
                num_steps=torch.tensor([n], dtype=torch.int64),
                enabled=None if enabled is None else torch.tensor([enabled[j]], dtype=torch.bool)))
        return state, samples

    def assert_preview_oracle(self, arrays, *, n=4):
        state, _ = self.run_trace(arrays, n=n)
        _, oracle = self.api()
        actual = state.episode_metrics(0, climb_success=True)
        expected = oracle(**arrays, num_steps=n, input_kind='trajectory', climb_success=True)
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            if isinstance(actual[name], float):
                self.assertAlmostEqual(actual[name], expected[name], delta=1e-8, msg=name)
            else:
                self.assertEqual(actual[name], expected[name], name)
        json.dumps(actual, allow_nan=False)
        return actual

    def test_exact_25_samples_pass_but_24_are_insufficient(self):
        for size, valid in [(24, False), (25, True)]:
            result = self.assert_preview_oracle(self.arrays([.4] * size))
            self.assertEqual(result['sample_count'], size)
            self.assertEqual(result['speed_valid'], valid)
            self.assertEqual(result['speed_pass'], valid)
            self.assertEqual(result['joint_success'], valid)
            self.assertEqual(result['mean_relative_error'], 0.)
            self.assertEqual(result['smooth_mae_relative'], 0. if valid else None)
            self.assertEqual(result['longest_deviation_samples'], 0 if valid else None)
            self.assertEqual(result['invalid_reasons'], [] if valid else ['insufficient_samples'])

    def test_missing_entry_or_exit_never_invents_a_valid_speed(self):
        for x, count, reasons in [([-1.] * 30, 0, ['not_entered', 'not_ended', 'insufficient_samples']),
                                  ([.1] * 30, 30, ['not_ended'])]:
            result = self.assert_preview_oracle(self.arrays([.4] * 30, x=x))
            self.assertEqual(result['sample_count'], count)
            self.assertEqual(result['invalid_reasons'], reasons)
            self.assertFalse(result['speed_pass'])
            self.assertFalse(result['joint_success'])
            self.assertEqual(result['mean_vx'], None if count == 0 else float(np.float32(.4)))

    def test_first_last_inclusive_single_crossing_and_post_end_are_exact(self):
        arrays = self.arrays([9., .4, .4, 8.], x=[-.1, 0., .8, 1.])
        state, samples = self.run_trace(arrays)
        self.assertEqual([int(s['sample_phase'][0]) for s in samples], [0, 1, 1, 2])
        result = state.episode_metrics(0)
        self.assertEqual((result['sample_count'], result['start_step'], result['end_step']), (2, 2, 3))
        self.assertEqual(result['phase_sample_counts'], [1, 2, 1])
        self.assertEqual(result['phase_mean_vx'], [9., float(np.float32(.4)), 8.])
        result = self.assert_preview_oracle(self.arrays([.4], x=[1.]))
        self.assertTrue(result['entered'] and result['ended'])
        self.assertEqual(result['sample_count'], 1)

    def test_reversal_and_standstill_remain_inside_first_segment(self):
        arrays = self.arrays([.4, 0., -.4, .4, .4], x=[0., .1, -.2, .2, .8])
        result = self.assert_preview_oracle(arrays)
        self.assertEqual(result['sample_count'], 5)
        self.assertEqual(result['phase_sample_counts'], [0, 5, 0])
        self.assertAlmostEqual(result['mean_vx'], .16, delta=1e-8)

    def test_mean_eps_and_symmetric_float32_command_boundaries(self):
        for c in [.2, .4, .6]:
            for sign in [-1., 1.]:
                for error, passed in [(.2, True), (.20000003, True), (.20001, False)]:
                    vx = np.float32(np.float64(np.float32(c)) * (1. + sign * error))
                    result = self.assert_preview_oracle(self.arrays([vx] * 25, c=c))
                    self.assertEqual(result['speed_pass'], passed, (c, sign, error, result))

    def test_smooth_mae_can_fail_despite_exact_mean(self):
        result = self.assert_preview_oracle(self.arrays([.64] * 100 + [.16] * 100))
        self.assertLess(result['mean_relative_error'], 1e-6)
        self.assertGreater(result['smooth_mae_relative'], .3)
        self.assertFalse(result['joint_success'])

    def test_mae_30_percent_boundary_is_inclusive_with_eps(self):
        # 26 samples: c*(1+25e), 24*c, c*(1-25e). Mean is c;
        # the two complete windows are c*(1+e), c*(1-e).
        for c in [.2, .4, .6]:
            for error, passed in [(.3, True), (.30000003, True), (.30001, False)]:
                command = np.float64(np.float32(c))
                vx = [command * (1 + 25 * error)] + [command] * 24 + [command * (1 - 25 * error)]
                result = self.assert_preview_oracle(self.arrays(vx, c=c))
                self.assertLess(result['mean_relative_error'], 1e-6)
                self.assertEqual(result['longest_deviation_samples'], 0)
                self.assertEqual(result['speed_pass'], passed, (c, error))

    def test_longest_deviation_uses_integer_50_51_and_eps(self):
        for size, k in [(74, 50), (75, 51)]:
            result = self.assert_preview_oracle(self.arrays([0.] * size))
            self.assertEqual(result['longest_deviation_samples'], k)
            self.assertEqual(result['longest_deviation_seconds'], k * .02)
        # L initial zeros yield L-24 zero windows and 12 further windows
        # below c/2 during recovery, so K=L-12. The 500-sample mean and
        # overall MAE both pass: this isolates the integer K gate itself.
        for zeros, k, passed in [(62, 50, True), (63, 51, False)]:
            result = self.assert_preview_oracle(self.arrays([0.] * zeros + [.4] * (500 - zeros)))
            self.assertLess(result['mean_relative_error'], .2)
            self.assertLess(result['smooth_mae_relative'], .3)
            self.assertEqual(result['longest_deviation_samples'], k)
            self.assertEqual(result['speed_pass'], passed)
            self.assertEqual(result['joint_success'], passed)
        for c in [.2, .4, .6]:
            for sign in [-1., 1.]:
                for error, k in [(.5, 0), (.50000004, 0), (.50001, 51)]:
                    vx = np.float32(np.float64(np.float32(c)) * (1. + sign * error))
                    result = self.assert_preview_oracle(self.arrays([vx] * 75, c=c))
                    self.assertEqual(result['longest_deviation_samples'], k, (c, sign, error))

    def test_long_deviation_fails_joint_even_with_good_mean(self):
        result = self.assert_preview_oracle(self.arrays([0.] * 100 + [.8] * 100))
        self.assertLess(result['mean_relative_error'], 1e-6)
        self.assertGreater(result['longest_deviation_samples'], 50)
        self.assertFalse(result['joint_success'])

    def test_speed_pass_is_independent_of_climb_and_failure(self):
        state, _ = self.run_trace(self.arrays([.4] * 25))
        _, oracle = self.api()
        for climb, failure, joint in [(False, False, False), (True, True, False), (True, False, True)]:
            for result in [state.episode_metrics(0, climb_success=climb, failure=failure),
                           oracle(**self.arrays([.4] * 25), num_steps=4,
                                  climb_success=climb, failure=failure)]:
                self.assertTrue(result['speed_pass'])
                self.assertEqual(result['sample_count'], 25)
                self.assertEqual(result['joint_success'], joint)

    def test_async_reset_clears_only_selected_env_and_allows_new_N_and_command(self):
        State, _ = self.api()
        state = State(2, device='cpu')
        for j in range(25):
            state.update(step=torch.tensor([j+1, j+1]), local_x=torch.tensor([j/40, j/40]),
                         body_vx=torch.tensor([.4, .4]), executed_c=torch.tensor([.4, .4]),
                         num_steps=torch.tensor([4, 4]))
        state.reset(torch.tensor([True, False]))
        info = state.update(step=torch.tensor([1, 26]), local_x=torch.tensor([0., .8]),
                            body_vx=torch.tensor([.2, .4]), executed_c=torch.tensor([.2, .4]),
                            num_steps=torch.tensor([8, 4]))
        self.assertEqual(info['body_mask'].tolist(), [True, True])
        self.assertEqual(state.episode_metrics(0)['sample_count'], 1)
        self.assertEqual(state.episode_metrics(1)['sample_count'], 26)
        self.assertTrue(state.episode_metrics(1)['speed_pass'])
        self.assertFalse(state.episode_metrics(0)['ended'])
        state.reset(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(state.episode_metrics(0)['sample_count'], 0)
        self.assertEqual(state.episode_metrics(1)['sample_count'], 26)
        state.reset()
        snap = state.snapshot()
        self.assertTrue(all(torch.isfinite(x).all().item() for x in snap.values()))
        self.assertEqual(snap['sample_count'].tolist(), [0, 0])
        self.assertEqual(snap['mean_valid'].tolist(), [False, False])
        self.assertEqual(snap['smooth_valid'].tolist(), [False, False])

    def test_parking_zero_command_skip_does_not_pollute_or_bind(self):
        arrays = self.arrays([100.] + [.4] * 25 + [100.], x=[-1.] + list(np.linspace(0., .8, 25)) + [1.])
        arrays['c'][[0, -1]] = 0.
        state, samples = self.run_trace(arrays, enabled=[False] + [True] * 25 + [False])
        self.assertEqual([int(samples[j]['sample_phase'][0]) for j in [0, -1]], [-1, -1])
        self.assertTrue(state.episode_metrics(0)['speed_pass'])
        self.assertEqual(state.episode_metrics(0)['phase_sample_counts'], [0, 25, 0])

    def test_bad_types_nonfinite_changed_command_N_and_nonmonotonic_step_raise(self):
        State, oracle = self.api()
        baseline = dict(step=torch.tensor([1]), local_x=torch.tensor([0.]),
                        body_vx=torch.tensor([.4]), executed_c=torch.tensor([.4]),
                        num_steps=torch.tensor([4]))
        for key, value in [('step', torch.tensor([1.])), ('step', torch.tensor([True])),
                           ('local_x', torch.tensor([0.], dtype=torch.float64)),
                           ('body_vx', torch.tensor([float('nan')])),
                           ('body_vx', torch.tensor([float('inf')])),
                           ('local_x', torch.tensor([float('-inf')])),
                           ('executed_c', torch.tensor([0.])),
                           ('executed_c', torch.tensor([float('nan')])),
                           ('num_steps', torch.tensor([0]))]:
            with self.assertRaises((TypeError, ValueError), msg=key):
                State(1).update(**(baseline | {key: value}))
        for key, value in [('step', torch.tensor([1])), ('step', torch.tensor([0])),
                           ('executed_c', torch.tensor([.40000004])), ('num_steps', torch.tensor([8]))]:
            state = State(1)
            state.update(**baseline)
            with self.assertRaises(ValueError, msg=key):
                state.update(**(baseline | {'step': torch.tensor([2])} | {key: value}))
            self.assertEqual(state.episode_metrics(0)['sample_count'], 1, 'rejected update must be atomic')
        for key, value in [('body_vx', np.array([np.nan], dtype=np.float32)),
                           ('c', np.array([0.], dtype=np.float32)),
                           ('step', np.array([1.], dtype=np.float32))]:
            with self.assertRaises((TypeError, ValueError)):
                oracle(**(self.arrays([.4]) | {key: value}), num_steps=4)

    def test_body_oracle_checks_raw_segment_metadata_and_command_consistency(self):
        _, oracle = self.api()
        empty = self.arrays([], x=[])
        result = oracle(**empty, num_steps=4, entered=False, ended=False)
        self.assertIsNone(result['mean_vx'])
        self.assertFalse(result['speed_valid'])
        a = self.arrays([.4] * 25)
        for kwargs in [dict(entered=False), dict(ended=False)]:
            with self.assertRaises(ValueError): oracle(**a, num_steps=4, **kwargs)
        for change in [dict(local_x=np.array([-.1] + [.8] * 24, dtype=np.float32)),
                       dict(local_x=np.array([0.] + [.8] * 24, dtype=np.float32)),
                       dict(c=np.array([.4] * 24 + [.40000004], dtype=np.float32)),
                       dict(step=np.ones(25, dtype=np.int64))]:
            with self.assertRaises(ValueError): oracle(**(a | change), num_steps=4)

    def test_missing_body_control_sample_or_skipping_active_body_is_rejected(self):
        State, oracle = self.api()
        a = self.arrays([.4] * 25)
        a['step'][12:] += 1
        for kind in ['body', 'trajectory']:
            with self.assertRaises(ValueError):
                oracle(**a, num_steps=4, input_kind=kind)
        baseline = dict(step=torch.tensor([1]), local_x=torch.tensor([0.]),
                        body_vx=torch.tensor([.4]), executed_c=torch.tensor([.4]),
                        num_steps=torch.tensor([4]))
        state = State(1)
        state.update(**baseline)
        with self.assertRaises(ValueError):
            state.update(**(baseline | {'step': torch.tensor([3])}))
        with self.assertRaises(ValueError):
            state.update(**(baseline | {'step': torch.tensor([2]), 'enabled': torch.tensor([False])}))
        self.assertEqual(state.episode_metrics(0)['sample_count'], 1)

    def test_original_float32_values_roundtrip_npz_and_used_command(self):
        _, oracle = self.api()
        arrays = self.arrays([.4] * 25, c=.4)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as loaded:
            for name in arrays:
                self.assertEqual(loaded[name].dtype, arrays[name].dtype)
                self.assertEqual(loaded[name].tobytes(), arrays[name].tobytes())
            result = oracle(**{k: loaded[k] for k in loaded.files}, num_steps=4)
        self.assertEqual(result['executed_c'], float(np.float32(.4)))
        self.assertNotEqual(result['executed_c'], .4)
        self.assertEqual(result['mean_relative_error'], 0.)

    def test_buffers_are_bounded_and_snapshot_is_detached_copy(self):
        State, _ = self.api()
        state = State(3)
        before = {k: (tuple(v.shape), v.numel()) for k, v in vars(state).items() if torch.is_tensor(v)}
        for j in range(300):
            state.update(step=torch.full((3,), j, dtype=torch.int64),
                         local_x=torch.full((3,), .1), body_vx=torch.full((3,), .4),
                         executed_c=torch.full((3,), .4), num_steps=torch.full((3,), 40))
        after = {k: (tuple(v.shape), v.numel()) for k, v in vars(state).items() if torch.is_tensor(v)}
        self.assertEqual(before, after)
        self.assertTrue(any(shape == (3, 25) for shape, _ in after.values()))
        self.assertFalse(any(isinstance(v, list) for v in vars(state).values()))
        snapshot = state.snapshot()
        snapshot['sample_count'].fill_(0)
        self.assertEqual(state.episode_metrics(0)['sample_count'], 300)

    def test_varied_float32_trace_matches_independent_oracle_within_1e8(self):
        # Fixed local generator; neither online state nor oracle constructs
        # the expected trajectory. This checks accumulated rounding only.
        rng = np.random.default_rng(2718)
        for c in [.2, .4, .6]:
            vx = (np.float32(c) + rng.normal(0., .25, 301)).astype(np.float32)
            x = np.concatenate(([-.2], np.linspace(0., .8, 299), [1.])).astype(np.float32)
            self.assert_preview_oracle(self.arrays(vx, c=c, x=x))


if __name__ == '__main__':
    unittest.main()
