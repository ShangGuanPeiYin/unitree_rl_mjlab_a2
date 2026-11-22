"""CPU evidence for the v0.5-r2 environment (SDS 0.5-r2 4, 5, 9.2, 9.3).

**Real MuJoCo/MJWarp simulation on the CPU.**  These tests deliberately construct
``V05R2Env(device='cpu', ...)`` and step the actual physics: the route contract,
the flat spawn distribution and the reward wiring only exist end-to-end, so a
kinematic stand-in could not see key-name or ordering defects.  No CUDA is
initialised here (``CUDA_VISIBLE_DEVICES`` is empty when the suite is run), which
is why the tests are not GPU-gated: ``device='cpu'`` is the supported benchmark
CPU path used by W0-W5.

Covered:
* route formula/shape/device/finiteness and the hard rejections (missing tile
  reference, wrong shape, non-finite pose);
* ``reset`` / normal ``step`` / ``info['terminal_observation']`` semantics
  (terminal = pre-reset, returned observation = post-reset, non-reset rows equal);
* the r2 flat training spawn (``x``/``y``/``yaw`` distribution, generator order,
  ``initial_cases`` precedence, subset reset) and the untouched ``N>0``/S5-ramp
  behaviour;
* reward/gate/state advancement unchanged (SDS formulas verified term by term,
  route independence, and a same-state r1-vs-r2 comparison of the shared
  observation/physics/reward path);
* the read-only ``route/*`` diagnostics and episode records.
"""
import math
import unittest

import numpy as np
import torch

from src.tasks.velocity.v05_r2 import spec
from src.tasks.velocity.v05_r2.env import (INITIAL_SAMPLE_ORDER, OBSERVATION_KEYS,
                                           ROUTE_LATERAL_SATURATION_M, SOURCE_FIXED,
                                           SOURCE_FLAT, SOURCE_INITIAL_CASES,
                                           SOURCE_S5_RAMP, V05R2Env, route_components,
                                           validate_observations, wrap_to_pi)
from src.tasks.velocity.v05_r2.speed import ROUTE_CLIP, ROUTE_LATERAL_SCALE

DT = spec.CONTROL_DT
DEVICE = 'cpu'


def make(stage='F0', num_envs=2, seed=7, **kwargs):
    return V05R2Env(num_envs=num_envs, device=DEVICE, seed=seed, stage=stage, **kwargs)


def zero_action(env):
    return torch.zeros(env.num_envs, 12, device=env.device)


def expected_route(local_y, yaw):
    """The SDS 4.2 vector as plain Python floats."""
    lateral = max(-ROUTE_CLIP, min(ROUTE_CLIP, float(local_y) / ROUTE_LATERAL_SCALE))
    return [lateral, math.sin(float(yaw)), math.cos(float(yaw))]


def pre_reset_route(env, row):
    """Route of the pre-reset sample, recomputed from the retained diagnostics."""
    diagnostics = env.diagnostics
    local = diagnostics['local_position'][row]
    yaw = diagnostics['heading'][row]
    return expected_route(float(local[1]), float(yaw))


class RouteObservationTests(unittest.TestCase):
    """SDS 4.3/4.7/9.3-21..23: the five-key observation and route validity."""

    def test_observation_keys_and_shapes(self):
        env = make('F0', num_envs=4)
        try:
            obs, info = env.reset()
            self.assertEqual(set(obs), {'proprio', 'history', 'heightmap',
                                        'privileged', 'route_error'})
            self.assertEqual(tuple(obs['proprio'].shape), (4, 45))
            self.assertEqual(tuple(obs['history'].shape), (4, 5, 45))
            self.assertEqual(tuple(obs['heightmap'].shape), (4, 3, 48, 32))
            self.assertEqual(tuple(obs['privileged'].shape), (4, 35))
            self.assertEqual(tuple(obs['route_error'].shape), (4, 3))
            self.assertEqual(obs['route_error'].dtype, torch.float32)
            self.assertEqual(obs['route_error'].device, obs['proprio'].device)
            self.assertTrue(torch.isfinite(obs['route_error']).all())
            self.assertIsInstance(info, dict)
            for key, value in obs.items():
                self.assertTrue(torch.isfinite(value).all(), key)
        finally:
            env.close()

    def test_route_matches_the_analytic_formula_from_the_same_frame(self):
        env = make('F0', num_envs=4, seed=11)
        try:
            env.reset()
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(zero_action(env))
                data = env.robot.data
                local = data.root_link_pos_w - env.origins[env.tile_ids]
                from src.tasks.velocity.v02_formal.env import euler_heading
                yaw = euler_heading(data.root_link_quat_w)[2]
                expected = torch.tensor(
                    [expected_route(float(local[i, 1]), float(yaw[i]))
                     for i in range(env.num_envs)], dtype=torch.float32)
                # 1e-6 covers the float32 sin/cos of the analytic path; the
                # implementation-internal identity is checked bitwise below.
                self.assertLessEqual(float((obs['route_error'] - expected).abs().max()), 1e-6)
                self.assertLessEqual(float((env.route_error() - obs['route_error']).abs().max()), 0.)
        finally:
            env.close()

    def test_route_is_bit_identical_to_the_task_frame(self):
        # SDS 9.3-23: the route uses the *same* frame as the reward/gate, so the
        # state vector stored by _task_state must be bit-equal to the observation.
        env = make('S1', num_envs=4, seed=5)
        try:
            env.reset()
            for _ in range(10):
                obs, _, _, _, _ = env.step(zero_action(env))
                self.assertTrue(torch.equal(env.sample_route, env.route_error()))
                self.assertTrue(torch.equal(env.sample_e_world,
                                            wrap_to_pi(env.diagnostics['heading'])))
        finally:
            env.close()

    def test_flat_tile_uses_the_same_three_components(self):
        env = make('F0', num_envs=4, seed=3)
        try:
            obs, _ = env.reset()
            self.assertTrue((env.tile_ids == 0).all())
            route = obs['route_error']
            # Flat is not a special case: the second/third components are the unit
            # heading vector of the real yaw, not zeros or ones by construction.
            from src.tasks.velocity.v02_formal.env import euler_heading
            yaw = euler_heading(env.robot.data.root_link_quat_w)[2]
            self.assertTrue(torch.allclose(route[:, 1], torch.sin(yaw), atol=1e-6))
            self.assertTrue(torch.allclose(route[:, 2], torch.cos(yaw), atol=1e-6))
            self.assertLessEqual(float(route[:, 1].abs().max()), 1.)
        finally:
            env.close()

    def test_invalid_tile_reference_is_rejected(self):
        env = make('S1', num_envs=4, seed=2)
        try:
            env.reset()
            env.tile_ids[1] = 999
            with self.assertRaises(RuntimeError) as caught:
                env.route_error()
            message = str(caught.exception)
            self.assertIn('S1', message)
            self.assertIn('env_id=1', message)
            self.assertIn('tile_id=999', message)
            self.assertIn('slot_class=', message)
            env.tile_ids[1] = 0
            env.tile_ids[2] = -1
            with self.assertRaises(RuntimeError):
                env.route_error()
        finally:
            env.close()

    def test_missing_frame_origin_reference_is_rejected(self):
        env = make('F0', num_envs=2, seed=2)
        try:
            env.reset()
            saved = env.origins
            env.origins = torch.zeros(0, 3, device=env.device)
            with self.assertRaises(RuntimeError):
                env.route_error()
            env.origins = saved
            self.assertEqual(tuple(env.route_error().shape), (2, 3))
        finally:
            env.close()

    def test_bad_batch_and_dtype_are_rejected(self):
        env = make('F0', num_envs=2, seed=2)
        try:
            env.reset()
            data = env.robot.data
            with self.assertRaises(TypeError):
                env._route_error(env.tile_ids.to(torch.int32), data.root_link_pos_w,
                                 data.root_link_quat_w)
            with self.assertRaises(ValueError):
                env._route_error(env.tile_ids[:1], data.root_link_pos_w[:1],
                                 data.root_link_quat_w[:1])
            with self.assertRaises(ValueError):
                env._route_error(env.tile_ids, data.root_link_pos_w[:, :2],
                                 data.root_link_quat_w)
        finally:
            env.close()

    def test_nonfinite_pose_raises_numerical_failure(self):
        from src.tasks.velocity.v02_formal.env import NumericalFailure
        env = make('F0', num_envs=2, seed=2)
        try:
            env.reset()
            data = env.robot.data
            broken = data.root_link_pos_w.clone()
            broken[0, 1] = float('nan')
            with self.assertRaises(NumericalFailure) as caught:
                env._route_error(env.tile_ids, broken, data.root_link_quat_w)
            self.assertIn('root_pos_w', caught.exception.diagnostics)
        finally:
            env.close()

    def test_terminal_observation_is_pre_reset_and_returned_route_is_post_reset(self):
        env = make('F0', num_envs=4, seed=13)
        try:
            env.reset()
            state = env.robot.data.default_root_state.clone()
            origins = env.origins[env.tile_ids]
            # Base every environment on its own tile origin (the default state is at
            # the world origin), then move only row 0 far outside the flat corridor
            # -> bounds failure, with a yaw the flat sampling can never reproduce in
            # one draw.
            state[:, 0] = origins[:, 0]
            state[:, 1] = origins[:, 1]
            state[0, 0] = origins[0, 0] + 0.
            state[0, 1] = origins[0, 1] + 5.
            state[0, 2] = env.robot.data.default_root_state[0, 2]
            state[0, 3] = math.cos(.15)
            state[0, 6] = math.sin(.15)
            env.robot.write_root_state_to_sim(state)
            obs, reward, terminated, truncated, info = env.step(zero_action(env))
            self.assertTrue(bool(terminated[0]) or bool(truncated[0]))
            terminal = info['terminal_observation']
            self.assertIn('route_error', terminal)
            self.assertEqual(tuple(terminal['route_error'].shape), (4, 3))
            self.assertGreaterEqual(float(terminal['route_error'][0, 0]), 4. - 1e-6)
            pre = pre_reset_route(env, 0)
            self.assertLessEqual(float((terminal['route_error'][0]
                                        - torch.tensor(pre, dtype=torch.float32)).abs().max()),
                                 1e-6)
            # The alignment components are the pre-reset yaw, not a post-reset one.
            self.assertAlmostEqual(float(torch.atan2(terminal['route_error'][0, 1],
                                                     terminal['route_error'][0, 2])),
                                   .3, places=4)
            # The returned new-episode observation is the *post-reset* state.
            from src.tasks.velocity.v02_formal.env import euler_heading
            local = env.robot.data.root_link_pos_w[0] - env.origins[env.tile_ids[0]]
            yaw = euler_heading(env.robot.data.root_link_quat_w)[2][0]
            post = torch.tensor(expected_route(float(local[1]), float(yaw)), dtype=torch.float32)
            self.assertLessEqual(float((obs['route_error'][0] - post).abs().max()), 1e-6)
            self.assertGreater(float((obs['route_error'][0] - terminal['route_error'][0]).abs().max()),
                               1e-3)
            # Non-reset rows keep terminal == returned observation.
            for row in (1, 2, 3):
                self.assertTrue(torch.equal(terminal['route_error'][row],
                                            obs['route_error'][row]))
            # The r1-info aliases point at the same formal values.
            self.assertTrue(torch.equal(info['route_error'], terminal['route_error']))
            self.assertTrue(torch.equal(info['route_error_terminal'],
                                        terminal['route_error']))
            self.assertTrue(torch.equal(info['route_error_next'], obs['route_error']))
            # The per-episode "last route" is the terminal value for the rows that
            # did not reset and the new-episode value for the row that did; the
            # emitted episode record already captured the pre-reset value.
            self.assertTrue(torch.equal(env.episode_route_final[1:], terminal['route_error'][1:]))
            self.assertTrue(torch.equal(env.episode_route_final[0], obs['route_error'][0]))
            self.assertLessEqual(abs(info['episodes'][0]['route_final'][0]
                                     - float(terminal['route_error'][0, 0])), 1e-6)
        finally:
            env.close()

    def test_failure_frame_route_is_not_zeroed(self):
        env = make('S1', num_envs=2, seed=19, fixed_vx=.4)
        try:
            env.reset()
            state = env.robot.data.default_root_state.clone()
            origins = env.origins[env.tile_ids]
            state[:, 0] = origins[:, 0]
            state[:, 1] = origins[:, 1]
            state[0, 0] = origins[0, 0] + 2.
            state[0, 1] = origins[0, 1] + 2.5   # outside the stair corridor -> failure
            state[0, 2] = env.robot.data.default_root_state[0, 2]
            state[0, 3] = math.cos(-.1)
            state[0, 6] = math.sin(-.1)
            env.robot.write_root_state_to_sim(state)
            _, _, _, _, info = env.step(zero_action(env))
            self.assertTrue(bool(info['failure'][0]))
            route = info['terminal_observation']['route_error'][0]
            self.assertTrue(bool(torch.isfinite(route).all()))
            self.assertAlmostEqual(float(route[0]), 4.0, places=5)  # |local_y| > 0.8
            self.assertAlmostEqual(float(torch.atan2(route[1], route[2])), -.2, places=5)
        finally:
            env.close()


class RouteSingleSourceTests(unittest.TestCase):
    """The three route-state sites in env.py read one decomposition, not three.

    Site A: :func:`route_components` / ``route_error_vector`` (the formula).
    Site B: :meth:`V05R2Env.route_metrics` (the read-only diagnostics tensors).
    Site C: ``info['diagnostics']['route']`` and ``diagnostics_snapshot()['route']``
            (the per-step and per-environment evidence blocks).

    All of them must agree bit-for-bit on |local_y|, the unclipped ratio, the
    clipped first component and the ``>= 0.80 m`` saturation flag.
    """

    def test_three_sites_agree_at_the_saturation_boundary(self):
        from src.tasks.velocity.v02_formal.env import euler_heading
        env = make('F0', num_envs=4, seed=151)
        try:
            env.reset()
            state = env.robot.data.default_root_state.clone()
            origins = env.origins[env.tile_ids]
            local_y = [-.8001, -.7999, .80, .9001]     # boundary / just inside / just past
            yaw = [.0, .2, -.3, .05]
            state[:, 0] = origins[:, 0]
            state[:, 1] = origins[:, 1] + torch.tensor(local_y, dtype=torch.float32)
            for row, angle in enumerate(yaw):
                state[row, 3] = math.cos(angle / 2)
                state[row, 6] = math.sin(angle / 2)
            env.robot.write_root_state_to_sim(state)
            obs, _, _, _, info = env.step(zero_action(env))
            # No row may reset here: 0.9 m is far inside the flat corridor (bounds 4 m).
            self.assertFalse(bool(info['failure'].any()))
            diagnostics = info['diagnostics']['route']
            metrics = env.route_metrics()
            data = env.robot.data
            local = data.root_link_pos_w - env.origins[env.tile_ids]
            site_a = route_components(local[:, 1], euler_heading(data.root_link_quat_w)[2])
            snapshot = env.diagnostics_snapshot(0)['route']
            # Site A vs B (tensors).
            self.assertTrue(torch.equal(site_a.abs_y, metrics['abs_y']))
            self.assertTrue(torch.equal(site_a.normalized_abs_y_raw,
                                        metrics['normalized_abs_y_raw']))
            self.assertTrue(torch.equal(site_a.normalized_abs_y, metrics['normalized_abs_y']))
            self.assertTrue(torch.equal(site_a.saturated, metrics['saturation']))
            self.assertTrue(torch.equal(site_a.route_error, obs['route_error']))
            # Site B vs C (per-step diagnostics tensors).
            for name, key in (('abs_y', 'abs_y'), ('normalized_abs_y_raw', 'normalized_abs_y_raw'),
                              ('normalized_abs_y', 'normalized_abs_y'),
                              ('saturated', 'saturation')):
                self.assertTrue(torch.equal(getattr(site_a, name), diagnostics[key]), name)
            self.assertTrue(torch.equal(site_a.route_error, info['route_error']))
            # Site B vs C' (per-environment snapshot, float values of row 0).
            self.assertAlmostEqual(snapshot['abs_y'], float(metrics['abs_y'][0]), places=6)
            self.assertAlmostEqual(snapshot['normalized_abs_y_raw'],
                                   float(metrics['normalized_abs_y_raw'][0]), places=6)
            self.assertAlmostEqual(snapshot['normalized_abs_y'],
                                   float(metrics['normalized_abs_y'][0]), places=6)
            self.assertEqual(snapshot['saturation'], bool(metrics['saturation'][0]))
            self.assertEqual(snapshot['saturation_m'], ROUTE_LATERAL_SATURATION_M)
            # Boundary semantics: the flag is exactly ``measured |local_y| >= 0.80``
            # (``>=``) on the real frame, and the clipped component never grows past 4.
            # The nominal 0.80 m row is a float32 frame-origin boundary (36 m frame
            # subtraction can land at 0.79999995), so it is checked against its own
            # measured value instead of a literal.
            measured = metrics['abs_y']
            self.assertTrue(torch.equal(
                metrics['saturation'],
                measured >= torch.tensor(ROUTE_LATERAL_SATURATION_M, dtype=torch.float32)))
            self.assertEqual(metrics['saturation'][0].item(), True)    # |−0.8001| m
            self.assertEqual(metrics['saturation'][1].item(), False)   # 0.7999 m
            self.assertEqual(metrics['saturation'][3].item(), True)    # 0.9001 m
            self.assertEqual(metrics['saturation'][2].item(),
                             bool(float(measured[2]) >= ROUTE_LATERAL_SATURATION_M))
            self.assertTrue(torch.allclose(
                metrics['normalized_abs_y'],
                torch.tensor([4., np.float32(.7999) / np.float32(.2), 4., 4.]), atol=1e-6))
            self.assertGreater(float(metrics['normalized_abs_y_raw'][3]), 4.)
            self.assertAlmostEqual(float(metrics['normalized_abs_y_raw'][3]), .9001 / .2,
                                   places=4)
        finally:
            env.close()
        # The episode accumulator reads the same |local_y| (site A), so a mixed
        # boundary batch cannot report a different maximum.  A fresh environment is
        # used because forcing the F0 episode cap on an already-stepped window would
        # break the speed window's consecutive-sample contract.
        other = make('F0', num_envs=2, seed=152)
        try:
            other.reset()
            state = other.robot.data.default_root_state.clone()
            origins = other.origins[other.tile_ids]
            state[:, 0] = origins[:, 0]
            state[:, 1] = origins[:, 1] + torch.tensor([.9001, .7999], dtype=torch.float32)
            other.robot.write_root_state_to_sim(state)
            other.episode_length_buf[:] = spec.F0_EPISODE_STEPS - 1
            _, _, _, _, info = other.step(zero_action(other))
            records = {int(record['env_id']): record for record in info['episodes']}
            self.assertAlmostEqual(records[0]['route_max_abs_y'], .9001, places=4)
            self.assertLess(records[1]['route_max_abs_y'], ROUTE_LATERAL_SATURATION_M + 1e-3)
            # (The rows have already auto-reset by now, so the pre-reset value is
            # read from the emitted record -- which is exactly the accumulator the
            # route evidence reports.)
        finally:
            other.close()


class ObservationValidatorTests(unittest.TestCase):
    """SDS 2.1/4.7: a contract violation is rejected, never zero-filled."""

    def observations(self, batch=3):
        return {'proprio': torch.zeros(batch, 45), 'history': torch.zeros(batch, 5, 45),
                'heightmap': torch.zeros(batch, 3, 48, 32),
                'privileged': torch.zeros(batch, 35),
                'route_error': torch.zeros(batch, 3)}

    def test_valid_observation_passes_and_returns_the_batch(self):
        self.assertEqual(validate_observations(self.observations(4)), 4)
        self.assertEqual(tuple(OBSERVATION_KEYS), ('proprio', 'history', 'heightmap',
                                                   'privileged', 'route_error'))

    def test_missing_key_is_rejected_with_the_key_and_consumer(self):
        for key in OBSERVATION_KEYS:
            obs = self.observations()
            del obs[key]
            with self.assertRaises(ValueError) as caught:
                validate_observations(obs, consumer='evaluation')
            self.assertIn(key, str(caught.exception))
            self.assertIn('evaluation', str(caught.exception))

    def test_bad_shape_dtype_batch_device_and_finiteness_are_rejected(self):
        bad_shapes = {'route_error': [torch.zeros(3, 2), torch.zeros(3, 4),
                                      torch.zeros(3, 3, 1)],
                      'proprio': [torch.zeros(3, 44)],
                      'history': [torch.zeros(3, 5, 44)],
                      'heightmap': [torch.zeros(3, 3, 48, 31)],
                      'privileged': [torch.zeros(3, 34)]}
        for key, values in bad_shapes.items():
            for value in values:
                obs = self.observations()
                obs[key] = value
                with self.assertRaises(ValueError):
                    validate_observations(obs)
        obs = self.observations()
        obs['route_error'] = obs['route_error'].to(torch.float64)
        with self.assertRaises(ValueError):
            validate_observations(obs)
        obs = self.observations()
        obs['route_error'] = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            validate_observations(obs)
        obs = self.observations()
        obs['route_error'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            validate_observations(obs)
        obs = self.observations()
        obs['route_error'] = torch.zeros(3, 3, device='meta')
        with self.assertRaises(ValueError):
            validate_observations(obs)
        obs = self.observations()
        obs['route'] = torch.zeros(3, 3)
        with self.assertRaises(ValueError):
            validate_observations(obs)
        with self.assertRaises(TypeError):
            validate_observations({'proprio': torch.zeros(3, 45)}['proprio'])

    def test_real_environment_observation_satisfies_the_contract(self):
        env = make('S1', num_envs=4, seed=131)
        try:
            obs, _ = env.reset()
            self.assertEqual(validate_observations(obs, consumer='env.reset'), 4)
            obs, _, _, _, _ = env.step(zero_action(env))
            self.assertEqual(validate_observations(obs, consumer='env.step'), 4)
        finally:
            env.close()


class FlatInitialStateTests(unittest.TestCase):
    """SDS 5.3/5.4: B1 (x correction) and B2 (new y/yaw randomization)."""

    def test_every_flat_slot_uses_the_r2_distribution(self):
        env = make('F0', num_envs=64, seed=21)
        try:
            poses = env.episode_initial_pose.detach().cpu().numpy()
            self.assertTrue(((poses[:, 0] >= -1.1 - 1e-6) & (poses[:, 0] <= -.9 + 1e-6)).all())
            self.assertTrue(((poses[:, 1] >= -.05 - 1e-6) & (poses[:, 1] <= .05 + 1e-6)).all())
            limit = math.pi / 36
            self.assertTrue((np.abs(poses[:, 2]) <= limit + 1e-6).all())
            # Both B1 and B2 are real randomizations, not a copied constant.
            self.assertGreater(float(poses[:, 0].std()), .02)
            self.assertGreater(float(poses[:, 1].std()), .005)
            self.assertGreater(float(poses[:, 2].std()), .01)
            self.assertTrue(all(source == SOURCE_FLAT for source in env.episode_initial_source))
            self.assertTrue((env.command[:, 0] >= .2 - 1e-6).all())
            self.assertTrue((env.command[:, 0] <= .6 + 1e-6).all())
            self.assertTrue(bool((env.command[:, 1:] == 0).all()))
            # The actual robot state equals the recorded initial pose.
            data = env.robot.data
            local = data.root_link_pos_w - env.origins[env.tile_ids]
            self.assertLessEqual(float((local[:, :2]
                                        - env.episode_initial_pose[:, :2]).abs().max()), 1e-5)
        finally:
            env.close()

    def test_initial_pose_is_sampled_x_then_y_then_yaw_then_command(self):
        env = make('F0', num_envs=8, seed=31)
        try:
            clone = torch.Generator(device=DEVICE)
            clone.set_state(env.rng.get_state())
            env.reset()
            draw = torch.rand((8, 3), generator=clone, device=DEVICE)
            command = .2 + .4 * torch.rand((8,), generator=clone, device=DEVICE)
            expected = torch.stack(((-1.1 + .2 * draw[:, 0]),
                                    (-.05 + .1 * draw[:, 1]),
                                    (-math.pi / 36 + (math.pi / 18) * draw[:, 2])), -1)
            self.assertLessEqual(float((env.episode_initial_pose - expected).abs().max()), 1e-7)
            self.assertLessEqual(float((env.command[:, 0] - command).abs().max()), 1e-7)
        finally:
            env.close()

    def test_same_seed_same_reset_sequence_is_reproducible(self):
        first = make('S1', num_envs=8, seed=42)
        second = make('S1', num_envs=8, seed=42)
        other = make('S1', num_envs=8, seed=43)
        try:
            for _ in range(3):
                first.reset()
                second.reset()
                other.reset()
                self.assertTrue(torch.equal(first.episode_initial_pose,
                                            second.episode_initial_pose))
                self.assertTrue(torch.equal(first.command, second.command))
                self.assertTrue(torch.equal(first.episode_route_initial,
                                            second.episode_route_initial))
                self.assertFalse(torch.equal(first.episode_initial_pose,
                                             other.episode_initial_pose))
            self.assertEqual(first.episode_rng_sha, second.episode_rng_sha)
        finally:
            for env in (first, second, other):
                env.close()

    def test_initial_cases_win_and_consume_no_reset_rng(self):
        cases = [[-1.0 + .01 * index, .02, .03] for index in range(4)]
        env = make('F0', num_envs=4, seed=5, evaluation=True, fixed_vx=.4,
                   initial_cases=cases)
        try:
            before = env._rng_state_sha256()
            obs, _ = env.reset()
            after = env._rng_state_sha256()
            self.assertEqual(before, after)
            expected = torch.tensor(cases, dtype=torch.float32)
            self.assertLessEqual(float((env.episode_initial_pose - expected).abs().max()), 1e-6)
            self.assertTrue(all(source == SOURCE_INITIAL_CASES
                                for source in env.episode_initial_source))
            self.assertTrue(torch.allclose(env.command[:, 0], torch.full((4,), .4)))
            # The route follows the explicit case, not a sampled pose.  The frame
            # origin is 36 m away, so re-deriving local_y from float32 world
            # coordinates carries a ~1e-6 lateral quantization; 2e-5 is the honest
            # bound for this bit-exact world round trip.
            self.assertLessEqual(float((obs['route_error'][:, 0]
                                        - torch.tensor([.1, .1, .1, .1])).abs().max()),
                                 2e-5)
        finally:
            env.close()

    def test_subset_reset_only_touches_those_slots(self):
        env = make('F0', num_envs=8, seed=61)
        try:
            env.reset()
            before = env.episode_initial_pose.clone()
            before_command = env.command.clone()
            clone = torch.Generator(device=DEVICE)
            clone.set_state(env.rng.get_state())
            env._reset_ids(torch.tensor([3], dtype=torch.int64))
            draw = torch.rand((1, 3), generator=clone, device=DEVICE)
            command = .2 + .4 * torch.rand((1,), generator=clone, device=DEVICE)
            expected = torch.tensor([[-1.1 + .2 * float(draw[0, 0]),
                                      -.05 + .1 * float(draw[0, 1]),
                                      -math.pi / 36 + (math.pi / 18) * float(draw[0, 2])]],
                                    dtype=torch.float32)
            self.assertLessEqual(float((env.episode_initial_pose[3] - expected[0]).abs().max()),
                                 1e-6)
            self.assertLessEqual(float((env.command[3, 0] - command[0]).abs()), 1e-7)
            others = [row for row in range(env.num_envs) if row != 3]
            self.assertTrue(torch.equal(env.episode_initial_pose[others], before[others]))
            self.assertTrue(torch.equal(env.command[others], before_command[others]))
            self.assertFalse(torch.equal(env.episode_initial_pose[3], before[3]))
        finally:
            env.close()

    def test_mixed_p1_flat_slots_are_randomized_and_stairs_are_fixed(self):
        env = make('P1', num_envs=8, seed=71)
        try:
            self.assertEqual(env.tile_ids.tolist(), [0, 0, 1, 1, 1, 1, 1, 1])
            self.assertEqual(env.stage_slots[:2], ['flat', 'flat'])
            poses = env.episode_initial_pose.detach().cpu().numpy()
            for row in (0, 1):
                self.assertTrue(-1.1 - 1e-6 <= poses[row, 0] <= -.9 + 1e-6)
                self.assertTrue(-.05 - 1e-6 <= poses[row, 1] <= .05 + 1e-6)
                self.assertEqual(env.episode_initial_source[row], SOURCE_FLAT)
            for row in range(2, 8):
                self.assertTrue(np.allclose(poses[row], [-1., 0., 0.], atol=1e-6))
                self.assertEqual(env.episode_initial_source[row], SOURCE_FIXED)
            # Every flat environment samples its own draw (no shared tile noise).
            self.assertNotEqual(float(poses[0, 0]), float(poses[1, 0]))
        finally:
            env.close()

    def test_mixed_s5_keeps_the_r1_ramp_and_the_fixed_previous_slot(self):
        env = make('S5', num_envs=8, seed=81, stage_iteration=0)
        try:
            tiles = env.tile_ids.tolist()
            slots = env.stage_slots
            current = [row for row, name in enumerate(slots) if name == 'current']
            previous = [row for row, name in enumerate(slots) if name == 'previous']
            flat = [row for row, name in enumerate(slots) if name == 'flat']
            self.assertEqual(tiles[current[0]], 9)
            if previous:
                self.assertEqual(tiles[previous[0]], 8)
            poses = env.episode_initial_pose.detach().cpu().numpy()
            for row in current:
                # stage_iteration == 0 -> zero ramp -> the fixed r1 spawn.
                self.assertTrue(np.allclose(poses[row], [-1., 0., 0.], atol=1e-6))
                self.assertEqual(env.episode_initial_source[row], SOURCE_S5_RAMP)
            for row in previous:
                self.assertTrue(np.allclose(poses[row], [-1., 0., 0.], atol=1e-6))
                self.assertEqual(env.episode_initial_source[row], SOURCE_FIXED)
            for row in flat:
                self.assertTrue(-1.1 - 1e-6 <= poses[row, 0] <= -.9 + 1e-6)
                self.assertEqual(env.episode_initial_source[row], SOURCE_FLAT)
            env.set_stage('S5', stage_iteration=200)
            poses = env.episode_initial_pose.detach().cpu().numpy()
            env.set_stage('S5', stage_iteration=400)
            poses400 = env.episode_initial_pose.detach().cpu().numpy()
            for row in current:
                # stage_iteration=200 -> half of the 400-update ramp.
                self.assertTrue(np.abs(poses[row, 0] + 1.) <= .1 + 1e-6)
                self.assertTrue(np.abs(poses[row, 1]) <= .05 + 1e-6)
                self.assertTrue(np.abs(poses[row, 2]) <= math.radians(5.) + 1e-6)
                self.assertTrue(np.abs(poses400[row, 0] + 1.) <= .2 + 1e-6)
                self.assertTrue(np.abs(poses400[row, 1]) <= .1 + 1e-6)
                self.assertTrue(np.abs(poses400[row, 2]) <= math.radians(10.) + 1e-6)
        finally:
            env.close()

    def test_evaluation_slots_use_explicit_cases_not_the_training_distribution(self):
        cases = [[-1.15, .06, math.radians(6.)]] * 2
        env = make('F0', num_envs=2, seed=91, evaluation=True, fixed_vx=.2,
                   initial_cases=cases)
        try:
            poses = env.episode_initial_pose.detach().cpu().numpy()
            self.assertTrue(np.allclose(poses, np.asarray(cases, dtype=np.float32), atol=1e-6))
            self.assertTrue(all(source == SOURCE_INITIAL_CASES
                                for source in env.episode_initial_source))
        finally:
            env.close()


class RewardAndGateTests(unittest.TestCase):
    """SDS 3/9.2: the reward composition, gates and advancement are unchanged."""

    def test_weighted_terms_sum_to_the_total(self):
        env = make('S1', num_envs=4, seed=101)
        try:
            env.reset()
            for _ in range(20):
                _, reward, _, _, info = env.step(zero_action(env))
                tensors = info['step_reward_tensors']
                weighted = [value for key, value in tensors.items()
                            if key.startswith('reward_weighted/')]
                self.assertLessEqual(float((sum(weighted) - tensors['reward_total']).abs().max()),
                                     1e-5)
                self.assertLessEqual(float((tensors['reward_total'] - reward).abs().max()), 1e-5)
        finally:
            env.close()

    def test_speed_amount_is_the_sds_formula_with_one_dt(self):
        env = make('S1', num_envs=4, seed=102)
        try:
            env.reset()
            for _ in range(10):
                _, _, _, _, info = env.step(zero_action(env))
            diagnostics = info['diagnostics']
            tensors = info['step_reward_tensors']
            vx = diagnostics['body_vx'].to(torch.float64)
            vy = env.robot.data.root_link_lin_vel_b[:, 1].to(torch.float64)
            c = diagnostics['command_c'].to(torch.float64)
            gate = diagnostics['gate_open']
            u = vx / c
            ey = vy / .2
            a = 1.5 * torch.exp(-4. * (u - 1.).square() - ey.square()) - .5
            over = .5 * (u - 1.2).clamp_min(0.).square().clamp_max(4.)
            back = .5 * (-u).clamp_min(0.).square().clamp_max(4.)
            side = .25 * ey.square().clamp_max(4.)
            rate = (torch.where(gate, a.clamp_min(0.), torch.zeros_like(a))
                    + a.clamp_max(0.) - over - back - side)
            expected = DT * rate
            actual = tensors['reward_weighted/speed'].to(torch.float64)
            self.assertLessEqual(float((expected - actual).abs().max()), 1e-6)
            self.assertLessEqual(float(tensors['reward_weighted/stall'].abs().max()), 1e-6)
        finally:
            env.close()

    def test_center_and_heading_are_the_sds_amounts(self):
        env = make('S1', num_envs=4, seed=103)
        try:
            env.reset()
            for _ in range(12):
                _, _, _, _, info = env.step(zero_action(env))
            diagnostics = info['diagnostics']
            tensors = info['step_reward_tensors']
            active = (diagnostics['command_c'] > spec.COMMAND_MIN).to(torch.float64)
            local_y = diagnostics['local_position'][:, 1].to(torch.float64)
            e_world = diagnostics['route']['e_world'].to(torch.float64)
            center = -.5 * (local_y / .2).square().clamp_max(4.) * active * DT
            heading = -.5 * (1. - torch.cos(e_world)) * active * DT
            self.assertLessEqual(float((tensors['reward_weighted/center'] - center).abs().max()),
                                 1e-9)
            self.assertLessEqual(float((tensors['reward_weighted/heading'] - heading).abs().max()),
                                 1e-9)
            # Route is never a reward term (SDS 3.9/9.3-20).
            self.assertFalse([key for key in tensors if 'route' in key])
            self.assertFalse([key for key in info['reward_terms'] if 'route' in key])
        finally:
            env.close()

    def test_route_cannot_change_reward_gate_or_termination(self):
        # Two identical runs; the second has route_error() replaced by zeros.
        baseline = make('S1', num_envs=4, seed=104)
        neutral = make('S1', num_envs=4, seed=104)
        try:
            baseline.reset()
            neutral.reset()
            neutral.route_error = lambda: torch.zeros(neutral.num_envs, 3,
                                                      device=neutral.device)
            for step in range(25):
                _, reward_a, terminated_a, truncated_a, info_a = baseline.step(zero_action(baseline))
                _, reward_b, terminated_b, truncated_b, info_b = neutral.step(zero_action(neutral))
                self.assertTrue(torch.equal(reward_a, reward_b), step)
                self.assertTrue(torch.equal(terminated_a, terminated_b), step)
                self.assertTrue(torch.equal(truncated_a, truncated_b), step)
                self.assertTrue(torch.equal(info_a['diagnostics']['gate_open'],
                                            info_b['diagnostics']['gate_open']), step)
                self.assertTrue(torch.equal(info_a['diagnostics']['footfall_count'],
                                            info_b['diagnostics']['footfall_count']), step)
                self.assertTrue(torch.equal(info_a['diagnostics']['stall_on'],
                                            info_b['diagnostics']['stall_on']), step)
        finally:
            baseline.close()
            neutral.close()

    def test_shared_physics_reward_and_flags_match_r1_on_identical_cases(self):
        # SDS 9.2-12 (Q002 scope): with explicit identical initial cases and the
        # same actions, the shared observation/physics/reward path must agree.
        from src.tasks.velocity.v05.env import V05Env as R1Env
        cases = [[-1.0, .02, .01], [-1.0, -.02, -.01]]
        r1 = R1Env(num_envs=2, device=DEVICE, seed=5, stage='P1', fixed_vx=.4,
                   initial_cases=cases)
        r2 = make('P1', num_envs=2, seed=5, fixed_vx=.4, initial_cases=cases)
        try:
            obs1, _ = r1.reset()
            obs2, _ = r2.reset()
            for key in ('proprio', 'history', 'heightmap', 'privileged'):
                self.assertEqual(tuple(obs1[key].shape), tuple(obs2[key].shape))
                self.assertLessEqual(float((obs1[key] - obs2[key]).abs().max()), 1e-6, key)
            max_observation_difference = 0.
            max_reward_difference = 0.
            for step in range(80):
                action = torch.zeros(2, 12, device=DEVICE)
                obs1, reward1, term1, trunc1, info1 = r1.step(action)
                obs2, reward2, term2, trunc2, info2 = r2.step(action)
                for key in ('proprio', 'history', 'heightmap', 'privileged'):
                    difference = float((obs1[key] - obs2[key]).abs().max())
                    max_observation_difference = max(max_observation_difference, difference)
                max_reward_difference = max(max_reward_difference,
                                            float((reward1 - reward2).abs().max()))
                self.assertTrue(torch.equal(term1, term2), step)
                self.assertTrue(torch.equal(trunc1, trunc2), step)
                self.assertTrue(torch.equal(info1['failure'], info2['failure']), step)
                for key, value in obs2.items():
                    self.assertTrue(torch.isfinite(value).all(), (step, key))
            self.assertLessEqual(max_observation_difference, 1e-6)
            self.assertLessEqual(max_reward_difference, 1e-6)
        finally:
            r1.close()
            r2.close()

    def test_reset_episode_limit_and_timeout_are_per_slot(self):
        env = make('S5', num_envs=4, seed=105, fixed_vx=.2)
        try:
            env.reset()
            limits = []
            for row in range(env.num_envs):
                n = int(env.tile_n[row])
                h = float(env.atlas_H[int(env.tile_ids[row])])
                limits.append(spec.episode_step_limit(n, h, .2))
                env.episode_length_buf[row] = limits[row] - 1
            _, _, terminated, truncated, info = env.step(zero_action(env))
            self.assertTrue(torch.isfinite(info['step_reward_tensors']['reward_total']).all())
            self.assertTrue(bool(truncated.all()), (truncated.tolist(), limits))
            reasons = {int(record['env_id']): record['reason'] for record in info['episodes']}
            for row in range(env.num_envs):
                self.assertEqual(reasons[row], 'timeout')
        finally:
            env.close()


class DiagnosticAndEvidenceTests(unittest.TestCase):
    """SDS 4.8: the read-only route diagnostics and episode records."""

    def test_route_diagnostics_are_present_and_consistent(self):
        env = make('S1', num_envs=4, seed=111)
        try:
            env.reset()
            _, _, _, _, info = env.step(zero_action(env))
            diagnostics = info['diagnostics']
            self.assertIn('route_error', diagnostics)
            self.assertEqual(tuple(diagnostics['route_error'].shape), (4, 3))
            route = diagnostics['route']
            for key in ('local_y', 'abs_y', 'normalized_abs_y', 'saturation', 'e_world',
                        'heading_deg', 'yaw_world', 'tile_ids', 'yaw_axis',
                        'nonfinite_count', 'lateral_scale', 'clip', 'components'):
                self.assertIn(key, route)
            self.assertEqual(route['yaw_axis'], 0.)
            self.assertEqual(route['lateral_scale'], .20)
            self.assertEqual(route['clip'], 4.)
            self.assertEqual(route['nonfinite_count'], 0)
            self.assertTrue(torch.allclose(route['abs_y'],
                                           diagnostics['local_position'][:, 1].abs()))
            self.assertTrue(torch.allclose(route['heading_deg'],
                                           route['e_world'].abs() * (180. / math.pi)))
            self.assertTrue(torch.allclose(
                route['normalized_abs_y'],
                (route['local_y'] / ROUTE_LATERAL_SCALE).abs().clamp_max(ROUTE_CLIP)))
            # The incremental aggregate helper exposes the SDS 4.8 tensor names.
            metrics = env.route_metrics()
            for key in ('local_y', 'abs_y', 'normalized_abs_y', 'saturation', 'e_world',
                        'heading_deg', 'nonfinite_count'):
                self.assertIn(key, metrics)
            self.assertEqual(metrics['nonfinite_count'], 0)
            snapshot = env.diagnostics_snapshot(0)
            self.assertIn('route_error', snapshot)
            self.assertEqual(len(snapshot['route_error']), 3)
            self.assertEqual(snapshot['route']['yaw_axis'], 0.)
            self.assertEqual(snapshot['initial_sample_order'], INITIAL_SAMPLE_ORDER)
            self.assertEqual(len(snapshot['rng_state_sha256_before_sample']), 64)
        finally:
            env.close()

    def test_episode_records_carry_route_and_initial_pose_provenance(self):
        env = make('F0', num_envs=2, seed=112)
        try:
            obs, _ = env.reset()
            initial_route = env.episode_route_initial.clone()
            for row in range(env.num_envs):
                env.episode_length_buf[row] = spec.F0_EPISODE_STEPS - 1
            _, _, _, truncated, info = env.step(zero_action(env))
            self.assertTrue(bool(truncated.all()))
            records = info['episodes']
            self.assertEqual(len(records), 2)
            for record, row in zip(records, range(2)):
                self.assertIn('route_final', record)
                self.assertIn('route_initial', record)
                self.assertIn('route_max_abs_y', record)
                self.assertIn('route_max_abs_heading_deg', record)
                self.assertEqual(len(record['route_final']), 3)
                self.assertEqual(len(record['route_initial']), 3)
                self.assertEqual(record['route_yaw_axis'], 0.)
                self.assertEqual(record['initial_sample_order'], 'x,y,yaw,command')
                self.assertEqual(len(record['rng_state_sha256_before_sample']), 64)
                self.assertEqual(record['initial_pose_source'], SOURCE_FLAT)
                self.assertLessEqual(abs(record['route_initial'][0]
                                         - float(initial_route[row, 0])), 1e-6)
                self.assertLessEqual(abs(record['route_final'][0]
                                         - float(info['terminal_observation']['route_error'][row, 0])),
                                     1e-6)
                self.assertGreaterEqual(record['route_max_abs_y'],
                                        abs(record['route_final'][0]) * ROUTE_LATERAL_SCALE - 1e-6)
                self.assertGreaterEqual(
                    record['route_max_abs_heading_deg'],
                    math.degrees(abs(float(torch.atan2(
                        torch.tensor(record['route_final'][1]),
                        torch.tensor(record['route_final'][2]))))) - 1e-3)
                self.assertEqual(record['protocol_id'], spec.FLAT_PROTOCOL_ID)
                self.assertEqual(record['code_version'], '0.5.1')
                self.assertEqual(record['recipe'], '0.5-r2')
        finally:
            env.close()

    def test_reward_terms_and_normalizers_never_see_route(self):
        env = make('F0', num_envs=2, seed=113)
        try:
            env.reset()
            _, _, _, _, info = env.step(zero_action(env))
            self.assertFalse([key for key in info['reward_terms'] if 'route' in key])
            self.assertFalse([key for key in info['step_reward_tensors'] if 'route' in key])
            # The environment owns no normalizer at all; route cannot enter one.
            self.assertFalse(hasattr(env, 'critic_normalizer'))
            self.assertFalse(hasattr(env, 'obs_normalizer'))
        finally:
            env.close()

    def test_flat_n0_still_opens_the_gate_and_pays_no_step_events(self):
        env = make('F0', num_envs=4, seed=114)
        try:
            env.reset()
            gate_open = False
            for _ in range(120):
                _, _, _, _, info = env.step(zero_action(env))
                gate_open |= bool(info['diagnostics']['gate_open'].any())
                self.assertFalse(bool(info['first_step_acquired'].any()))
                self.assertTrue(bool((info['diagnostics']['footfall_count'] == 0).all()))
                self.assertTrue(bool((info['diagnostics']['swing_paid_amount'] == 0).all()))
                self.assertTrue(torch.isfinite(info['terminal_observation']['route_error']).all())
            self.assertTrue(gate_open)
            self.assertTrue(torch.equal(env.tile_n, torch.zeros_like(env.tile_n)))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
