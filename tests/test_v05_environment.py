"""GPU evidence for the v0.5 environment (SDS 9.2 items 9-14).

The simulator tests are GPU-only by design: the CPU acceptance path never
constructs a simulated environment.  Every test below is skipped when CUDA is
unavailable, and the skip message states that limitation explicitly.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.v05 import spec  # noqa: E402
from src.tasks.velocity.v05.env import V05Env  # noqa: E402
from src.tasks.velocity.v05.speed import tile_window_kind  # noqa: E402

GPU = torch.cuda.is_available()
DEVICE = 'cuda:0'
GPU_REASON = 'GPU-only: constructing the MuJoCo/MJWarp stair scene is not allowed on CPU'
DT = spec.CONTROL_DT


def make(stage, num_envs=8, seed=0, device=DEVICE, **kwargs):
    return V05Env(num_envs=num_envs, device=device, seed=seed, stage=stage, **kwargs)


def zero_action(env):
    return torch.zeros(env.num_envs, 12, device=env.device)


@unittest.skipUnless(GPU, GPU_REASON)
class GeometryAndSlotTests(unittest.TestCase):
    def test_real_geometry_matches_all_ten_stages(self):
        from src.tasks.velocity.v02.terrain import atlas_metadata
        metadata = atlas_metadata()
        env = make('F0', num_envs=2)
        try:
            self.assertEqual([stage.id for stage in spec.STAGES],
                             ['F0', 'P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4', 'S5'])
            for stage in spec.STAGES:
                with self.subTest(stage=stage.id):
                    env.set_stage(stage.id)
                    self.assertEqual(env.stage, stage.id)
                    expected_tiles = spec.slot_tiles(stage.id, env.num_envs)
                    self.assertEqual(env.tile_ids.tolist(), expected_tiles)
                    self.assertEqual(env.stage_slots, spec.slot_layout(stage.id, env.num_envs))
                    n, h = env._parameters()
                    for slot, tile in enumerate(expected_tiles):
                        self.assertEqual(int(n[slot]), metadata[tile].N)
                        self.assertAlmostEqual(float(h[slot]), metadata[tile].h, places=7)
                    self.assertEqual(env.tile_n.tolist(), [metadata[t].N for t in expected_tiles])
                    if stage.id != 'F0':
                        previous = spec.previous_stage_id(stage.id)
                        if previous is not None and 'previous' in env.stage_slots:
                            index = env.stage_slots.index('previous')
                            self.assertEqual(int(env.tile_ids[index]),
                                             spec.STAGE_BY_ID[previous].tile_id)
                            # The previous slot keeps its own N/h, never the current one.
                            self.assertEqual(int(n[index]),
                                             spec.STAGE_BY_ID[previous].N)
        finally:
            env.close()

    def test_mixed_s1_uses_each_slot_own_configuration(self):
        env = make('S1', num_envs=8, fixed_vx=.4)
        try:
            self.assertEqual(env.tile_ids.tolist(), [0, 4, 5, 5, 5, 5, 5, 5])
            self.assertEqual(env.tile_n.tolist(), [0, 1, 4, 4, 4, 4, 4, 4])
            snapshot = env.speed_router.snapshot()
            self.assertEqual(env.speed_router.kind_names,
                             ['f0_time', 'p_position', 's_oracle', 's_oracle',
                              's_oracle', 's_oracle', 's_oracle', 's_oracle'])
            self.assertEqual([tile_window_kind(int(t)) for t in env.tile_ids],
                             env.speed_router.kind_names)
            self.assertTrue(snapshot['f0_time'] is not None)
            # Previous slot (P4, N=1) has a shorter protocol timeout than current S1.
            from src.tasks.velocity.v02_formal.env import timeout_steps
            n, h = env._parameters()
            limits = timeout_steps(n, h, torch.full_like(n, .4, dtype=torch.float32))
            self.assertLess(int(limits[1]), int(limits[2]))
        finally:
            env.close()

    def test_command_sampling_is_uniform_and_positive_per_slot(self):
        env = make('S1', num_envs=8, seed=3)
        try:
            seen = []
            for _ in range(6):
                env.reset()
                command = env.command.detach().cpu()
                self.assertTrue(torch.all(command[:, 0] >= .2))
                self.assertTrue(torch.all(command[:, 0] <= .6))
                self.assertTrue(torch.all(command[:, 1:] == 0))
                seen.extend(command[:, 0].tolist())
            self.assertGreater(max(seen) - min(seen), 1e-3)
            # No slot is a zero-command stand slot.
            self.assertTrue(all(value > .2 for value in seen))
        finally:
            env.close()


@unittest.skipUnless(GPU, GPU_REASON)
class LifecycleTests(unittest.TestCase):
    def test_f0_p1_and_mixed_s1_run_100_control_steps(self):
        for stage in ('F0', 'P1', 'S1'):
            with self.subTest(stage=stage):
                env = make(stage, num_envs=8)
                try:
                    obs, _ = env.reset()
                    self.assertEqual(obs['proprio'].shape, (8, 45))
                    self.assertEqual(obs['history'].shape, (8, 5, 45))
                    self.assertEqual(obs['heightmap'].shape, (8, 3, 48, 32))
                    self.assertEqual(obs['privileged'].shape, (8, 35))
                    for step in range(100):
                        action = torch.zeros(8, 12, device=env.device)
                        obs, reward, terminated, truncated, info = env.step(action)
                        self.assertEqual(tuple(obs['proprio'].shape), (8, 45))
                        self.assertTrue(torch.isfinite(reward).all(), (stage, step))
                        for key, value in obs.items():
                            self.assertTrue(torch.isfinite(value).all(), (stage, step, key))
                        for key in ('failure', 'climb', 'timeout', 'first_step_acquired',
                                    'stall_on', 'reset_mask'):
                            self.assertIn(key, info)
                        diagnostics = info['diagnostics']
                        self.assertEqual(tuple(diagnostics['swing_paid_fraction'].shape), (8, 4))
                        self.assertEqual(tuple(diagnostics['swing_paid_amount'].shape), (8, 4))
                        self.assertEqual(tuple(diagnostics['footfall_count'].shape), (8,))
                        self.assertEqual(tuple(diagnostics['tile_N'].shape), (8,))
                        self.assertEqual(tuple(diagnostics['command_c'].shape), (8,))
                        self.assertEqual(set(diagnostics['reward_terms']),
                                         set(env.last_reward_terms))
                finally:
                    env.close()

    def test_mixed_slots_with_random_actions_reach_the_target_query(self):
        # The swing target query must never see a flat slot as a step-1 target
        # (A's `_target_geom` regression) and must stay finite for a mixed batch.
        for stage in ('P1', 'S1'):
            with self.subTest(stage=stage):
                env = make(stage, num_envs=8)
                try:
                    env.reset()
                    torch.manual_seed(0)
                    for step in range(120):
                        action = .3 * torch.randn(8, 12, device=env.device)
                        obs, reward, terminated, truncated, info = env.step(action)
                        self.assertTrue(torch.isfinite(reward).all(), (stage, step))
                        diagnostics = info['diagnostics']
                        # The flat slot is never a stair target; the previous slot of
                        # S1 is its own P4 tile with N=1.
                        self.assertEqual(int(diagnostics['tile_N'][0]), 0)
                        self.assertEqual(int(diagnostics['tile_N'][1]), 0 if stage == 'P1' else 1)
                finally:
                    env.close()

    def test_environment_episode_limits_match_the_spec_per_slot(self):
        # Real env path: set each slot one step before its own spec limit and check
        # the environment truncates there (no +2 margin, flat masked to 1500).
        for stage_id in ('F0', 'P1', 'S1', 'S4', 'S5'):
            with self.subTest(stage=stage_id):
                env = make(stage_id, num_envs=8, fixed_vx=.2)
                try:
                    env.reset()
                    limits = []
                    for row in range(env.num_envs):
                        n = int(env.tile_n[row])
                        tile = int(env.tile_ids[row])
                        h = float(env.atlas_H[tile])
                        limits.append(spec.episode_step_limit(n, h, .2))
                        env.episode_length_buf[row] = limits[row] - 1
                    obs, reward, terminated, truncated, info = env.step(zero_action(env))
                    self.assertTrue(torch.isfinite(reward).all())
                    self.assertTrue(bool(truncated.all()),
                                    (stage_id, truncated.tolist(), limits))
                    reasons = {int(record['env_id']): record['reason']
                               for record in info['episodes']}
                    for row in range(env.num_envs):
                        self.assertEqual(reasons[row], 'timeout', (stage_id, row))
                        self.assertGreaterEqual(int(info['episodes'][0]['terminal_step']), 0)
                finally:
                    env.close()

    def test_flat_n0_end_to_end(self):
        env = make('F0', num_envs=8)
        try:
            env.reset()
            gate_open = False
            for _ in range(120):
                obs, reward, terminated, truncated, info = env.step(zero_action(env))
                gate_open |= bool(info['diagnostics']['gate_open'].any())
                self.assertFalse(bool(info['first_step_acquired'].any()))
                self.assertTrue(torch.all(info['diagnostics']['footfall_count'] == 0))
                self.assertTrue(torch.all(info['diagnostics']['swing_paid_amount'] == 0))
                self.assertEqual(set(env.speed_router.kind_names), {'f0_time'})
            self.assertTrue(gate_open, 'flat safe support must be able to open g')
            self.assertTrue(torch.all(env.tile_n == 0))
        finally:
            env.close()

    def test_reset_and_action_shapes_match_v04_contract(self):
        env = make('F0', num_envs=2)
        try:
            result = env.reset()
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            action = torch.full((2, 12), .4, device=env.device)
            output = env.step(action)
            self.assertEqual(len(output), 5)
            obs, reward, terminated, truncated, info = output
            self.assertEqual(obs['proprio'].shape, (2, 45))
            self.assertEqual(reward.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
            # ``action_manager.action`` keeps the raw 12-dim action; the joint term
            # applies the single 0.25 scale exactly once to the default pose.
            raw = env.action_manager.action.detach().cpu()
            self.assertEqual(raw.shape, (2, 12))
            self.assertTrue(torch.allclose(raw, action.detach().cpu(), atol=1e-6))
            term = env.action_manager.get_term('joint_pos')
            self.assertAlmostEqual(float(term._scale), 0.25, places=7)
            processed = term._processed_actions.detach().cpu()
            expected = term._offset.detach().cpu() + action.detach().cpu() * 0.25
            self.assertLessEqual(float((processed - expected).abs().max()), 1e-6)
            self.assertEqual(processed.shape, (2, 12))
            self.assertIn('terminal_observation', info)
            self.assertIn('episodes', info)
        finally:
            env.close()


@unittest.skipUnless(GPU, GPU_REASON)
class RewardWiringTests(unittest.TestCase):
    def test_weighted_terms_sum_to_the_total(self):
        env = make('S1', num_envs=8)
        try:
            env.reset()
            for _ in range(20):
                _, reward, _, _, info = env.step(zero_action(env))
                tensors = info['step_reward_tensors']
                weighted = [value for key, value in tensors.items()
                            if key.startswith('reward_weighted/')]
                total = sum(weighted)
                self.assertLessEqual(float((total - tensors['reward_total']).abs().max()), 1e-5)
                self.assertLessEqual(float((tensors['reward_total'] - reward).abs().max()), 1e-5)
                terms = info['reward_terms']
                summed = sum(value for key, value in terms.items()
                             if key.startswith('reward_weighted/'))
                self.assertLess(abs(summed - float(terms['reward_total'])), 1e-5)
        finally:
            env.close()

    def test_speed_amount_is_the_sds_formula_with_one_dt(self):
        env = make('S1', num_envs=8)
        try:
            env.reset()
            for _ in range(10):
                _, _, _, _, info = env.step(zero_action(env))
            diagnostics = info['diagnostics']
            tensors = info['step_reward_tensors']
            vx = diagnostics['body_vx'].to(torch.float64)
            c = diagnostics['command_c'].to(torch.float64)
            gate = diagnostics['gate_open']
            u = vx / c
            a = 1.5 * torch.exp(-4. * (u - 1.).square()) - .5
            over = .5 * (u - 1.2).clamp_min(0.).square().clamp_max(4.)
            back = .5 * (-u).clamp_min(0.).square().clamp_max(4.)
            rate = torch.where(gate, a.clamp_min(0.), torch.zeros_like(a)) + a.clamp_max(0.) - over - back
            expected = DT * rate
            actual = tensors['reward_weighted/speed'].to(torch.float64)
            self.assertLessEqual(float((expected - actual).abs().max()), 1e-6)
            self.assertLessEqual(float((tensors['reward_weighted/stall']).abs().max()), 1e-6)
        finally:
            env.close()

    def test_failure_frame_is_pre_reset_evidence(self):
        env = make('F0', num_envs=2)
        try:
            env.reset()
            state = env.robot.data.default_root_state.clone()
            state[:, 3] = math.cos(math.pi / 4)
            state[:, 4] = math.sin(math.pi / 4)
            state[:, 5] = 0.
            state[:, 6] = 0.
            env.robot.write_root_state_to_sim(state)
            obs, reward, terminated, truncated, info = env.step(zero_action(env))
            self.assertTrue(bool(terminated[0]))
            self.assertTrue(bool(info['failure'][0]))
            self.assertEqual(int(info['diagnostics']['control_step'][0]), 1)
            # The environment has already auto-reset, but the record is the failure.
            self.assertEqual(int(env.episode_length_buf[0]), 0)
            record = info['episodes'][0]
            self.assertTrue(record['failure'])
            self.assertEqual(record['terminal_step'], 1)
            self.assertEqual(record['reason'], 'orientation')
            # A real failure pays exactly -10 once and never a success event.
            self.assertEqual(float(info['reward_terms']['reward_weighted/failure']), -10.)
            self.assertEqual(float(info['reward_terms']['reward_weighted/success']), 0.)
            self.assertIn('reward_sums', record)
            self.assertIsInstance(record['speed_metrics'], dict)
            self.assertTrue(torch.isfinite(reward).all())
        finally:
            env.close()

    def test_timeout_pays_no_failure_event(self):
        env = make('F0', num_envs=2, fixed_vx=.4)
        try:
            env.reset()
            env.episode_length_buf[:] = spec.F0_EPISODE_STEPS
            _, reward, terminated, truncated, info = env.step(zero_action(env))
            self.assertFalse(bool(terminated.any()))
            self.assertTrue(bool(truncated[0]))
            self.assertEqual(float(info['reward_terms']['reward_weighted/failure']), 0.)
            self.assertEqual(float(info['reward_terms']['reward_weighted/success']), 0.)
            self.assertEqual(float(info['reward_terms']['reward_weighted/speed']) != 0., True)
            self.assertEqual(info['episodes'][0]['reason'], 'timeout')
            self.assertTrue(info['episodes'][0]['timeout'])
            self.assertFalse(info['episodes'][0]['failure'])
        finally:
            env.close()

    def test_event_amounts_stay_inside_the_protocol_caps(self):
        env = make('S1', num_envs=8)
        try:
            env.reset()
            for _ in range(100):
                _, _, _, _, info = env.step(zero_action(env))
                for record in info['episodes']:
                    n = int(record['N'])
                    self.assertLessEqual(record['footfall_events'], .5 * 4 * n + 1e-6)
                    self.assertLessEqual(abs(record['swing_paid']), .15 * 4 * max(n, 0) + 1e-6)
        finally:
            env.close()


class AdvanceOrderTests(unittest.TestCase):
    """SDS 4: footfall, swing and stall are each advanced exactly once, in order."""

    def test_task_state_calls_the_three_state_machines_in_sds_order(self):
        import re
        from pathlib import Path
        source = Path(ROOT, 'src/tasks/velocity/v05/env.py').read_text()
        body = source[source.index('def _task_state'):]
        body = body[:body.index('def _build_swing_sample')]
        calls = re.findall(r'self\._advance_(footfall|swing|stall)\(state\)', body)
        self.assertEqual(calls, ['footfall', 'swing', 'stall'])

    @unittest.skipUnless(GPU, GPU_REASON)
    def test_state_machines_are_order_independent_for_identical_inputs(self):
        # Two independent instances would be contaminated by engine round-off
        # (measured: same-order replays differ by ~1e-9 after a few steps), so this
        # freezes the exact inputs of one real control sample and feeds them to
        # fresh state machines in both orders.  Outputs must be bit-identical.
        from src.tasks.velocity.v05.env import TaskState
        from src.tasks.velocity.v05.rewards import (FootfallState, StallState,
                                                    SwingClearanceState)
        for stage in ('F0', 'P1', 'S1'):
            with self.subTest(stage=stage):
                env = make(stage, num_envs=8, fixed_vx=.4)
                try:
                    env.reset()
                    captured = {}
                    original_footfall = env._advance_footfall
                    original_swing = env._advance_swing

                    def capture_footfall(state):
                        captured['footfall'] = (
                            env.tile_ids.detach().clone(),
                            env.robot.data.site_pos_w[:, env.foot_ids, :2].detach().clone(),
                            env.contacts.top_forces.detach().clone(),
                            env.geom_tile, env.geom_step, env.geom_step_bounds,
                            state.failure.detach().clone())
                        local = (env.robot.data.root_link_pos_w
                                 - env.origins[env.tile_ids])
                        captured['stall'] = (
                            local[:, 0].detach().clone(),
                            env.sample_command[:, 0].detach().clone(),
                            env.climb.climbed.detach().clone(),
                            (~env._motion_parking_mask()).detach().clone())
                        return original_footfall(state)

                    def capture_swing(state):
                        captured['sample'] = env._build_swing_sample(state)
                        return original_swing(state)

                    env._advance_footfall = capture_footfall
                    env._advance_swing = capture_swing
                    for step in range(6):
                        action = .05 * torch.ones(8, 12, device=env.device)
                        obs, reward, terminated, truncated, info = env.step(action)
                    self.assertIn('footfall', captured)
                    self.assertIn('sample', captured)
                    self.assertIn('stall', captured)
                    footfall_args = captured['footfall']
                    sampled = captured['sample']
                    stall_args = captured['stall']

                    def build():
                        return (FootfallState(8, max_steps=spec.MAX_SWING_STEPS,
                                              device=env.device),
                                SwingClearanceState(8, max_steps=spec.MAX_SWING_STEPS,
                                                    device=env.device),
                                StallState(8, device=env.device))

                    def run_order(order):
                        foot, swing, stall = build()
                        stall.reset(torch.arange(8, device=env.device),
                                    stall_args[0].detach().clone())
                        outputs = {}
                        for name in order:
                            if name == 'footfall':
                                outputs['footfall'] = foot.update(*footfall_args)
                            elif name == 'swing':
                                outputs['swing'] = swing.update(sampled)
                            else:
                                outputs['stall'] = stall.update(*stall_args)
                        state = {}
                        for label, machine in (('footfall', foot), ('swing', swing),
                                               ('stall', stall)):
                            for key, value in vars(machine).items():
                                if torch.is_tensor(value):
                                    state[label + '.' + key] = value.clone()
                        return outputs, state

                    first_out, first_state = run_order(('footfall', 'swing', 'stall'))
                    second_out, second_state = run_order(('swing', 'footfall', 'stall'))
                    self.assertEqual(set(first_state), set(second_state))
                    for key in first_state:
                        self.assertTrue(torch.equal(first_state[key], second_state[key]),
                                        f'{stage}: state {key} depends on advance order')
                    for name in first_out:
                        for field in ('reward', 'penalty_on', 'events', 'positive_gate',
                                      'safe_support', 'selected_step'):
                            if hasattr(first_out[name], field):
                                self.assertTrue(torch.equal(
                                    getattr(first_out[name], field),
                                    getattr(second_out[name], field)),
                                    f'{stage}: {name}.{field} depends on advance order')
                finally:
                    env.close()


# ---------------------------------------------------------------- Q001 amendment
#: Q001 decision (SHA256 ``951d6b5f...f6c099``) section 2.1/2.2: the comparison
#: baseline uses ONLY the two old instances, and the set of asserted frames is
#: selected ONLY by the old-old trajectory.  The 10x / 1e-6 / 1e-5 / 1e-3 numbers
#: are a bounded engineering regression criterion, not a theoretical MuJoCo error
#: bound, and never an assertion about the whole 80-step trajectory.
Q001_SEEDS = (0, 42, 43)
Q001_STEPS = 80
Q001_PREFIX_LIMIT = 1e-3
Q001_BOUND_FACTOR = 10.0
Q001_BOUND_FLOOR = 1e-6
Q001_BOUND_SLACK = 1e-5
Q001_MIN_PREFIX = 20
Q001_OBS_GROUPS = ('proprio', 'history', 'heightmap', 'privileged')
Q001_EVIDENCE_DIR = ROOT / 'logs/a2_ideal_stairs_v05/implementation/acceptance/q001_physics_equivalence'


def q001_bound(old_spread):
    """``10 * max(old_spread, 1e-6) + 1e-5``; depends on the old pair only."""
    return Q001_BOUND_FACTOR * max(float(old_spread), Q001_BOUND_FLOOR) + Q001_BOUND_SLACK


def q001_stable_prefix(old_spreads):
    """Frames before the first old-spread > 1e-3 (old-old trajectory only)."""
    for index, value in enumerate(old_spreads):
        if value > Q001_PREFIX_LIMIT:
            return index
    return len(old_spreads)


def q001_evaluate(old_spreads, new_errors):
    """The single comparison decision used by the GPU trial and the CPU examples.

    The prefix and every bound come from ``old_spreads`` alone; ``new_errors`` can
    only cause a failure, never a skipped frame.  Insufficient prefix (fewer than
    ``Q001_MIN_PREFIX`` frames) fails as insufficient evidence.
    """
    if len(old_spreads) != len(new_errors):
        raise ValueError('old_spread and new_error series must have equal length')
    prefix = q001_stable_prefix(old_spreads)
    violations = []
    for step in range(prefix):
        bound = q001_bound(old_spreads[step])
        if float(new_errors[step]) > bound:
            violations.append(dict(step=step, old_spread=float(old_spreads[step]),
                                    new_error=float(new_errors[step]), bound=bound))
    enough = prefix >= Q001_MIN_PREFIX
    return dict(prefix=prefix, checked_frames=prefix, enough_evidence=enough,
                violations=violations, passed=bool(enough and not violations),
                first_violation=(violations[0] if violations else None))


def _q001_diff(left, right):
    """Maximum absolute difference over the four shared observation groups."""
    return max(float((left[key] - right[key]).abs().max()) for key in Q001_OBS_GROUPS)


def _q001_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


Q002_EVIDENCE_DIR = Q001_EVIDENCE_DIR
Q002_ADMISSION = 'accepted_with_known_limitation_Q002'
Q002_LONG_HORIZON = 'not_established'
Q002_HISTORY = 'failed_q001'


def _q002_write(name, payload):
    """Atomic JSON evidence write (never touches the canonical Q001 files)."""
    import json
    import os
    import tempfile
    Q002_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = Q002_EVIDENCE_DIR / name
    fd, temporary = tempfile.mkstemp(prefix='.' + name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, indent=1, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def _q002_source_sha(include_test=True):
    files = {
        'src/tasks/velocity/v05/env.py': ROOT / 'src/tasks/velocity/v05/env.py',
        'src/tasks/velocity/v04/env.py': ROOT / 'src/tasks/velocity/v04/env.py',
        'src/tasks/velocity/v02_formal/env.py': ROOT / 'src/tasks/velocity/v02_formal/env.py',
        'src/tasks/velocity/v05/spec.py': ROOT / 'src/tasks/velocity/v05/spec.py',
    }
    if include_test:
        files['tests/test_v05_environment.py'] = ROOT / 'tests/test_v05_environment.py'
    return {name: _q001_sha(path) for name, path in files.items()}


@unittest.skipUnless(GPU, GPU_REASON)
class FailureParityTests(unittest.TestCase):
    """Q002 4.2(iv): real failure parity and pre-reset evidence labelling.

    The forced posture failure exercises the same frozen failure algorithm in both
    implementations; the second test pins the label rule: after ``env.step()`` the
    auto-reset slots' ``sim.data.qpos/qvel`` are POST-reset state and are never
    recorded as pre-reset evidence.  Pre-reset evidence comes from
    ``info['terminal_observation']`` / ``info['episodes']`` snapshots.
    """

    def test_failure_decision_matches_the_old_environment(self):
        from src.tasks.velocity.v04.env import V04Env as Old
        old = Old(num_envs=1, device=DEVICE, seed=0, stage='S1', fixed_vx=.4,
                  initial_cases=[[-1., 0., 0.]])
        new = V05Env(num_envs=1, device=DEVICE, seed=0, stage='S1', fixed_vx=.4,
                     initial_cases=[[-1., 0., 0.]])
        try:
            old.reset()
            new.reset()
            for env in (old, new):
                state = env.robot.data.default_root_state.clone()
                state[:, 3] = math.cos(math.pi / 4)
                state[:, 4] = math.sin(math.pi / 4)
                env.robot.write_root_state_to_sim(state)
            _, _, old_terminated, _, old_info = old.step(zero_action(old))
            _, _, new_terminated, _, new_info = new.step(zero_action(new))
            self.assertEqual(bool(old_terminated[0]), bool(new_terminated[0]))
            self.assertTrue(bool(new_info['failure'][0]))
            record = new_info['episodes'][0]
            self.assertEqual(record['reason'], 'orientation')
            # Failure precedence: no success, no timeout, exactly -10 once.
            self.assertFalse(record['success'])
            self.assertFalse(record['timeout'])
            self.assertEqual(float(new_info['reward_terms']['reward_weighted/failure']), -10.)
            self.assertEqual(float(new_info['reward_terms']['reward_weighted/success']), 0.)
            self.assertEqual(old_info['episodes'][0]['reason'], 'orientation')
            _q002_write('q002_failure_parity.json', dict(
                evidence='Q002 4.2(iv) failure parity and precedence',
                new=new_info['episodes'][0], old=old_info['episodes'][0],
                source_sha256=_q002_source_sha()))
        finally:
            old.close()
            new.close()

    def test_pre_reset_evidence_is_the_terminal_snapshot_not_post_reset_state(self):
        # Random command sampling makes a reset's new command distinguishable.
        env = V05Env(num_envs=1, device=DEVICE, seed=7, stage='F0')
        try:
            obs, _ = env.reset()
            before_command = env.command[0].detach().clone()
            before_qpos = env.sim.data.qpos.detach().clone()
            state = env.robot.data.default_root_state.clone()
            state[:, 3] = math.cos(math.pi / 4)
            state[:, 4] = math.sin(math.pi / 4)
            env.robot.write_root_state_to_sim(state)
            obs, reward, terminated, truncated, info = env.step(zero_action(env))
            self.assertTrue(bool(terminated[0]))
            self.assertTrue(bool(info['failure'][0]))
            # Pre-reset snapshot: the diagnostic/terminal sample of the failed step.
            self.assertEqual(int(info['diagnostics']['control_step'][0]), 1)
            self.assertEqual(int(info['episodes'][0]['terminal_step']), 1)
            # proprio layout: ang_vel(3), gravity(3), command(3) -> command index 6.
            terminal_command = info['terminal_observation']['proprio'][0, 6]
            self.assertLessEqual(abs(float(terminal_command) - float(before_command[0])), 1e-6)
            # Returned observation and sim state are already POST auto-reset.
            self.assertEqual(int(env.episode_length_buf[0]), 0)
            after_command = env.command[0].detach().clone()
            self.assertGreater(abs(float(after_command[0]) - float(before_command[0])), 0.)
            self.assertLessEqual(abs(float(obs['proprio'][0, 6]) - float(after_command[0])), 1e-6)
            post_qpos = env.sim.data.qpos.detach().clone()
            # The returned observation is the post-reset snapshot and differs from the
            # pre-reset terminal_observation; qpos below is recorded as POST-reset.
            self.assertGreater(float((obs['proprio'] - info['terminal_observation']['proprio'])
                                     .abs().max()), 0.)
            # The evidence files never label this post-reset qpos/qvel as pre-reset.
            _q002_write('q002_pre_reset_evidence.json', dict(
                evidence='Q002 4.2(iv) pre-reset snapshot versus post-auto-reset state',
                pre_reset=dict(source='info.terminal_observation + info.episodes',
                               terminal_step=1,
                               terminal_command=float(terminal_command),
                               before_command=float(before_command[0])),
                post_auto_reset=dict(source='returned obs + env.sim.data after step',
                                     command=float(after_command[0]),
                                     qpos=post_qpos.detach().cpu().tolist(),
                                     qvel=env.sim.data.qvel.detach().cpu().tolist(),
                                     note='post-reset state; never labelled pre-reset'),
                label_rule='env.sim.data.qpos/qvel after step() are POST-reset',
                source_sha256=_q002_source_sha()))
        finally:
            env.close()


class Q001ComparatorTests(unittest.TestCase):
    """CPU counterexamples exercise the real comparator, not a copied formula."""

    def test_identical_old_pair_and_small_new_error_fails(self):
        result = q001_evaluate([0.] * 30, [0.0009] * 30)
        self.assertFalse(result['passed'])
        self.assertEqual(result['checked_frames'], 30)
        self.assertEqual(len(result['violations']), 30)
        self.assertAlmostEqual(result['first_violation']['bound'], 2e-5, places=12)
        self.assertEqual(result['first_violation']['step'], 0)

    def test_stable_old_pair_and_large_new_error_fails_without_filtering(self):
        result = q001_evaluate([0.] * 80, [0.01] * 80)
        self.assertFalse(result['passed'])
        # The frame set is fixed by the old pair: all 80 frames were asserted.
        self.assertEqual(result['checked_frames'], 80)
        self.assertEqual(len(result['violations']), 80)

    def test_bound_is_inclusive_at_the_boundary_and_next_float_fails(self):
        bound = q001_bound(0.)
        # 10*1e-6 + 1e-5 is 1.9999999999999998e-05 in binary floating point.
        self.assertAlmostEqual(bound, 2e-5, places=18)
        at_boundary = q001_evaluate([0.] * Q001_MIN_PREFIX, [bound] * Q001_MIN_PREFIX)
        self.assertTrue(at_boundary['passed'])
        import math
        over = q001_evaluate([0.] * Q001_MIN_PREFIX,
                             [bound] * (Q001_MIN_PREFIX - 1) + [math.nextafter(bound, math.inf)])
        self.assertFalse(over['passed'])
        self.assertEqual(over['first_violation']['step'], Q001_MIN_PREFIX - 1)

    def test_insufficient_prefix_is_a_failure_and_does_not_recheck_new(self):
        old_spreads = [0.] * 10 + [2e-3] * 70
        result = q001_evaluate(old_spreads, [0.] * 80)
        self.assertFalse(result['passed'])
        self.assertFalse(result['enough_evidence'])
        self.assertEqual(result['prefix'], 10)
        self.assertEqual(result['checked_frames'], 10)
        self.assertEqual(result['violations'], [])

    def test_frame_selection_and_bound_never_use_the_new_trajectory(self):
        # Huge new errors cannot extend, shrink or hide the old-defined prefix.
        result = q001_evaluate([0.] * 80, [5.] * 80)
        self.assertEqual(result['prefix'], 80)
        self.assertEqual(result['checked_frames'], 80)
        self.assertEqual(len(result['violations']), 80)
        # An old-spread jump bounds the prefix regardless of a perfect new series.
        result = q001_evaluate([0.] * 25 + [1e-2] * 55, [0.] * 80)
        self.assertEqual(result['prefix'], 25)
        self.assertTrue(result['passed'])


@unittest.skipUnless(GPU, GPU_REASON)
class Q001ReferenceDiagnostics(unittest.TestCase):
    """Supplementary (NOT a substitute for Q001): which reference does v0.5 track?

    Q001 compares v0.5 against two stock ``V04Env`` instances.  This diagnostic
    additionally compares v0.5 against the *unwrapped* references (``v02_formal``
    and a ``V04Env`` whose evidence ring is disabled) on the same early steps, so
    the reviewer can see whether a Q001 failure is a v0.5 port defect or the
    wrapped v0.4 reference's own per-instance behaviour.
    """

    def test_v05_matches_the_unwrapped_reference_on_the_early_steps(self):
        import json
        from src.tasks.velocity.v04.env import V04Env
        from src.tasks.velocity.v02_formal.env import FormalEnv
        seed = Q001_SEEDS[0]
        new = V05Env(num_envs=1, device=DEVICE, seed=seed, stage='S1', fixed_vx=.4,
                     initial_cases=[[-1., 0., 0.]])
        old_a = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1', fixed_vx=.4,
                       initial_cases=[[-1., 0., 0.]])
        old_b = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1', fixed_vx=.4,
                       initial_cases=[[-1., 0., 0.]])
        old_raw = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1', fixed_vx=.4,
                         initial_cases=[[-1., 0., 0.]])
        old_raw._contact_ring = None  # disable the evidence wrapper on this instance
        frozen = FormalEnv(num_envs=1, device=DEVICE, seed=seed, stage='S1', fixed_vx=.4,
                           initial_cases=[[-1., 0., 0.]])
        envs = dict(new=new, old_a=old_a, old_b=old_b, old_raw=old_raw, frozen=frozen)
        rows = []
        try:
            for env in envs.values():
                env.reset()
            for step in range(12):
                action = torch.full((1, 12), .02 * step, device=DEVICE)
                obs = {name: {k: v.detach().clone() for k, v in env.step(action)[0].items()}
                       for name, env in envs.items()}
                qpos = {name: env.sim.data.qpos.detach().clone()
                        for name, env in envs.items()}
                rows.append(dict(
                    step=step,
                    obs_new_frozen=_q001_diff(obs['new'], obs['frozen']),
                    obs_new_old_raw=_q001_diff(obs['new'], obs['old_raw']),
                    obs_new_old_a=_q001_diff(obs['new'], obs['old_a']),
                    obs_old_raw_old_a=_q001_diff(obs['old_raw'], obs['old_a']),
                    obs_old_a_old_b=_q001_diff(obs['old_a'], obs['old_b']),
                    qpos_new_frozen=float((qpos['new'] - qpos['frozen']).abs().max()),
                    qpos_new_old_raw=float((qpos['new'] - qpos['old_raw']).abs().max()),
                    qpos_new_old_a=float((qpos['new'] - qpos['old_a']).abs().max()),
                    qpos_old_a_old_b=float((qpos['old_a'] - qpos['old_b']).abs().max())))
        finally:
            for env in envs.values():
                env.close()
        payload = dict(
            supplementary=True, replaces_q001=False,
            note=('informational only: Q001 still compares against two stock V04Env '
                  'instances; this file shows what v0.5 tracks'),
            seed=seed, rows=rows,
            source_sha256={name: _q001_sha(path) for name, path in (
                ('src/tasks/velocity/v05/env.py', ROOT / 'src/tasks/velocity/v05/env.py'),
                ('src/tasks/velocity/v04/env.py', ROOT / 'src/tasks/velocity/v04/env.py'),
                ('src/tasks/velocity/v02_formal/env.py',
                 ROOT / 'src/tasks/velocity/v02_formal/env.py'),
                ('tests/test_v05_environment.py',
                 ROOT / 'tests/test_v05_environment.py'))})
        Q001_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        path = Q001_EVIDENCE_DIR / 'reference_diagnostic.json'
        path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + '\n')
        # The early-step identity is the claim this diagnostic actually makes.
        first = rows[0]
        self.assertEqual(first['obs_new_frozen'], 0.)
        self.assertEqual(first['obs_new_old_raw'], 0.)




class Q002CanonicalEvidenceTests(unittest.TestCase):
    """Q002 4.1/4.2: the canonical failed_q001 record is preserved, not rewritten.

    The long-horizon "prefix >= 20 and new_error <= 10x old-old bound" comparison is
    now a historical diagnostic: this test reads the canonical per-frame files and
    records the status triple, but never asserts that the old criterion passed.
    """

    def _load(self):
        import json
        summary_path = Q001_EVIDENCE_DIR / 'summary.json'
        if not summary_path.is_file():
            self.skipTest('canonical Q001 evidence directory is not present')
        return json.loads(summary_path.read_text())

    def test_canonical_failed_q001_is_preserved_with_the_status_triple(self):
        import json
        summary = self._load()
        self.assertEqual(summary['status'], Q002_HISTORY)
        self.assertEqual(summary['long_horizon_numeric_equivalence'], Q002_LONG_HORIZON)
        self.assertIn('unresolved', summary)
        # Production source SHAs of the canonical evidence still match the tree.
        for name in ('src/tasks/velocity/v05/env.py', 'src/tasks/velocity/v04/env.py',
                     'src/tasks/velocity/v02_formal/env.py',
                     'src/tasks/velocity/v05/spec.py'):
            self.assertEqual(_q001_sha(ROOT / name), summary['source_sha256'][name], name)
        canonical_files = {path.name: _q001_sha(path) for path in sorted(
            Q001_EVIDENCE_DIR.glob('*'))
            if path.name == 'summary.json' or path.name.startswith('seed_')}
        rows = []
        exact_prefixes, stable_prefixes = [], []
        for seed in Q001_SEEDS:
            data = json.loads((Q001_EVIDENCE_DIR / f'seed_{seed}.json').read_text())
            frames = data['frames']
            self.assertEqual(len(frames), Q001_STEPS)
            self.assertEqual([frame['step'] for frame in frames], list(range(Q001_STEPS)), seed)
            exact = next((index for index, frame in enumerate(frames)
                          if frame['new_error'] != 0.), len(frames))
            stable = q001_stable_prefix([frame['old_spread'] for frame in frames])
            exact_prefixes.append(exact)
            stable_prefixes.append(stable)
            self.assertTrue(all(frame['flags_consistent'] for frame in frames), seed)
            final = frames[-1]['terminal_flags']
            self.assertEqual(len(final), 3)
            # The termination evidence is a real, non-empty failure on every instance.
            for label, flag in final.items():
                self.assertTrue(flag['failure'] and flag['terminated'], (seed, label))
                self.assertFalse(flag['success'], (seed, label))
                self.assertFalse(flag['truncated'], (seed, label))
                self.assertIsInstance(flag['reason'], int)
            rows.append(dict(seed=seed, exact_old_new_zero_prefix=exact,
                             old_old_stable_prefix=stable,
                             decision=data['decision'],
                             replay_control_max_diff=data.get('replay_control_max_diff'),
                             final_frame_flags=final))
        # Q002 section 2 canonical facts, recomputed from the retained frames.
        self.assertEqual(exact_prefixes, [14, 8, 3])
        self.assertEqual(stable_prefixes, [25, 20, 7])
        payload = dict(
            schema='q002-summary-1',
            admission=Q002_ADMISSION,
            long_horizon_numeric_equivalence=Q002_LONG_HORIZON,
            q001_history=Q002_HISTORY,
            q001_canonical_status=summary['status'],
            q001_canonical_summary_sha256=_q001_sha(Q001_EVIDENCE_DIR / 'summary.json'),
            q001_canonical_files=canonical_files,
            exact_old_new_zero_prefix={str(seed): value
                                       for seed, value in zip(Q001_SEEDS, exact_prefixes)},
            old_old_stable_prefix={str(seed): value
                                   for seed, value in zip(Q001_SEEDS, stable_prefixes)},
            canonical_facts=rows,
            corrections=[
                'RETRACTED: same-instance reset replay bit-exactness. The canonical '
                'seed-0 replay probe records a nonzero maximum difference.',
                'No unified "0..15 steps bit-exact" claim: the exact old-new zero '
                'prefixes are 14/8/3 per seed and describe only these trials.',
                'Those prefixes are NOT called pre-contact: the initial foot-ground '
                'contact is contact.',
                '0.1145/0.1627 are maximum component differences of flattened mixed '
                'observations (from the Q002 decision engine probe); they are not '
                'metric position errors and do not establish physical equivalence.',
            ],
            note=('Q001 section 2.1 long-horizon ratio bound is retained as a historical '
                  'diagnostic only; admission uses Q002 4.2 items (i)-(iv)'),
            source_sha256=_q002_source_sha(),
            test_source_note=('canonical evidence was produced by an earlier revision of '
                              'tests/test_v05_environment.py; the Q002 split never rewrites '
                              'those files'),
        )
        _q002_write('summary_q002.json', payload)


@unittest.skipUnless(GPU, GPU_REASON)
class Q002InitFirstStepTests(unittest.TestCase):
    """Q002 4.2(i): common reset observations identical and first step <= 1e-6."""

    def test_reset_observations_identical_and_first_step_within_1em6(self):
        import json
        from src.tasks.velocity.v04.env import V04Env
        rows = []
        for seed in Q001_SEEDS:
            old_a = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1',
                           fixed_vx=.4, initial_cases=[[-1., 0., 0.]])
            old_b = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1',
                           fixed_vx=.4, initial_cases=[[-1., 0., 0.]])
            new = V05Env(num_envs=1, device=DEVICE, seed=seed, stage='S1',
                         fixed_vx=.4, initial_cases=[[-1., 0., 0.]])
            try:
                obs_a, _ = old_a.reset()
                obs_b, _ = old_b.reset()
                obs_n, _ = new.reset()
                reset_diffs = {}
                for key in Q001_OBS_GROUPS:
                    reset_diffs[key] = dict(
                        old_old=float((obs_a[key] - obs_b[key]).abs().max()),
                        old_a_new=float((obs_a[key] - obs_n[key]).abs().max()),
                        old_b_new=float((obs_b[key] - obs_n[key]).abs().max()))
                    self.assertEqual(reset_diffs[key]['old_old'], 0., (seed, key))
                    self.assertEqual(reset_diffs[key]['old_a_new'], 0., (seed, key))
                    self.assertEqual(reset_diffs[key]['old_b_new'], 0., (seed, key))
                action = torch.zeros(1, 12, device=DEVICE)  # canonical first step: 0.02*0
                oa = {k: v.detach().clone() for k, v in old_a.step(action)[0].items()}
                ob = {k: v.detach().clone() for k, v in old_b.step(action)[0].items()}
                on = {k: v.detach().clone() for k, v in new.step(action)[0].items()}
                first = dict(
                    old_old=_q001_diff(oa, ob),
                    old_a_new=_q001_diff(oa, on),
                    old_b_new=_q001_diff(ob, on))
                self.assertLessEqual(max(first['old_a_new'], first['old_b_new']), 1e-6, seed)
                rows.append(dict(seed=seed, reset_diffs=reset_diffs, first_step=first,
                                 first_step_bound=1e-6))
            finally:
                old_a.close()
                old_b.close()
                new.close()
        _q002_write('q002_init_first_step.json', dict(
            evidence='Q002 4.2(i) initialization and first control step',
            scenario=dict(stage='S1', fixed_vx=.4, initial_pose=[-1., 0., 0.],
                          num_envs=1, device=DEVICE, first_action='zeros (0.02*0)'),
            rows=rows, source_sha256=_q002_source_sha()))


@unittest.skipUnless(GPU, GPU_REASON)
class Q002PhysicsInvariantTests(unittest.TestCase):
    """Q002 4.2(ii): frozen physics configuration and ported structural invariants.

    Authorized changes (reward recipe, stage table/slots, command curriculum, new
    diagnostics) are deliberately not part of this comparison.
    """

    def test_frozen_physics_configuration_and_ported_structure(self):
        import json
        import re
        import numpy as np
        from src.tasks.velocity.v04.env import V04Env
        from src.tasks.velocity.v02_formal.env import FormalEnv
        old = V04Env(num_envs=1, device=DEVICE, seed=0, stage='S1', fixed_vx=.4,
                     initial_cases=[[-1., 0., 0.]])
        new = V05Env(num_envs=1, device=DEVICE, seed=0, stage='S1', fixed_vx=.4,
                     initial_cases=[[-1., 0., 0.]])
        checks = {}
        try:
            checks['physics_dt'] = dict(value=float(new.physics_dt), expected=.005,
                                        ok=float(new.physics_dt) == .005)
            checks['control_dt'] = dict(value=float(new.step_dt), expected=.02,
                                        ok=float(new.step_dt) == .02)
            checks['decimation'] = dict(value=int(new.cfg.decimation),
                                        old=int(old.cfg.decimation), expected=4,
                                        ok=int(new.cfg.decimation) == 4 == int(old.cfg.decimation))
            term = new.action_manager.get_term('joint_pos')
            old_term = old.action_manager.get_term('joint_pos')
            checks['action_scale'] = dict(value=float(term._scale),
                                          old=float(old_term._scale), expected=.25,
                                          ok=float(term._scale) == .25 == float(old_term._scale))
            raw = torch.full((1, 12), .3, device=DEVICE)
            processed = term._offset.detach() + raw * .25
            term.process_actions(raw)
            checks['action_single_scale'] = dict(
                ok=float((term._processed_actions.detach() - processed).abs().max()) <= 1e-6)
            names = [new.robot.joint_names[i] for i in new.joint_ids]
            old_names = [old.robot.joint_names[i] for i in old.joint_ids]
            expected_joints = [f'{name}_joint' for name in spec.JOINT_ORDER]
            checks['joint_order'] = dict(value=names, expected=expected_joints,
                                         ok=names == expected_joints == old_names)
            checks['action_targets'] = dict(
                value=[int(x) for x in term._target_ids],
                old=[int(x) for x in old_term._target_ids],
                ok=list(term._target_ids) == list(old_term._target_ids)
                == list(new.joint_ids))
            for label, getter in (
                    ('default_joint_pos', lambda env: env.robot.data.default_joint_pos),
                    ('soft_joint_pos_limits', lambda env: env.robot.data.soft_joint_pos_limits),
                    ('joint_pos_limits', lambda env: env.robot.data.joint_pos_limits)):
                checks[label] = dict(ok=bool(np.array_equal(
                    getter(new).detach().cpu().numpy(), getter(old).detach().cpu().numpy())))
            model_new, model_old = new.sim.mj_model, old.sim.mj_model
            for label, field in (('actuator_gainprm', 'actuator_gainprm'),
                                 ('actuator_biasprm', 'actuator_biasprm'),
                                 ('actuator_ctrlrange', 'actuator_ctrlrange'),
                                 ('actuator_gear', 'actuator_gear'),
                                 ('actuator_dyntype', 'actuator_dyntype'),
                                 ('actuator_gaintype', 'actuator_gaintype')):
                checks[label] = dict(ok=bool(np.array_equal(
                    np.asarray(getattr(model_new, field)),
                    np.asarray(getattr(model_old, field)))))
            geom_names_new = [str(model_new.geom(i).name or '') for i in range(model_new.ngeom)]
            geom_names_old = [str(model_old.geom(i).name or '') for i in range(model_old.ngeom)]
            checks['model_geom_names'] = dict(
                count=len(geom_names_new), ok=geom_names_new == geom_names_old)
            checks['model_geom_sizes'] = dict(ok=bool(np.array_equal(
                np.asarray(model_new.geom_size), np.asarray(model_old.geom_size))))
            checks['model_geom_contype'] = dict(ok=bool(np.array_equal(
                np.asarray(model_new.geom_contype), np.asarray(model_old.geom_contype))))
            checks['model_geom_conaffinity'] = dict(ok=bool(np.array_equal(
                np.asarray(model_new.geom_conaffinity),
                np.asarray(model_old.geom_conaffinity))))
            checks['contacts_lookup'] = dict(ok=(
                torch.equal(new.contacts.foot_lookup, old.contacts.foot_lookup)
                and torch.equal(new.contacts.nonfoot_lookup, old.contacts.nonfoot_lookup)
                and list(new.contacts.link_names) == list(old.contacts.link_names)
                and torch.equal(new.contacts.body_link_mask, old.contacts.body_link_mask)))
            for label, getter in (
                    ('terrain_geom_tile', lambda env: env.geom_tile),
                    ('terrain_geom_step', lambda env: env.geom_step),
                    ('terrain_geom_top', lambda env: env.geom_top),
                    ('terrain_step_bounds', lambda env: env.geom_step_bounds),
                    ('terrain_bounds', lambda env: env.bounds),
                    ('terrain_origins', lambda env: env.origins),
                    ('terrain_spawn_x', lambda env: env.spawn_x)):
                checks[label] = dict(ok=torch.equal(getter(new).detach(), getter(old).detach()))
            checks['obs_assembly_inherited'] = dict(
                observations=(V05Env._observations is FormalEnv._observations),
                proprio=(V05Env._proprio is FormalEnv._proprio),
                foot_heights=(V05Env._foot_heights is FormalEnv._foot_heights),
                ok=(V05Env._observations is FormalEnv._observations
                    and V05Env._proprio is FormalEnv._proprio
                    and V05Env._foot_heights is FormalEnv._foot_heights))
            # Sensing and four-substep advancement: the physics loop of V05Env.step
            # must contain the same ordered operations as the frozen control step.
            def loop_tokens(source):
                body = source[source.index('def step('):]
                body = body[:body.index('return obs')] if 'return obs' in body else body
                tokens = re.findall(
                    r'(process_action|apply_action|write_data_to_sim|sim\.step\(|'
                    r'sim\.forward|scene\.update|contacts\.sample|sim\.sense|'
                    r'episode_length_buf\s*\+=\s*1|common_step_counter\s*\+=\s*1|'
                    r'range\(self\.cfg\.decimation\))', body)
                return [''.join(token.split()) for token in tokens]
            new_tokens = loop_tokens(Path(ROOT, 'src/tasks/velocity/v05/env.py').read_text())
            old_tokens = loop_tokens(Path(ROOT, 'src/tasks/velocity/v02_formal/env.py').read_text())
            required = [''.join(token.split()) for token in (
                'process_action', 'range(self.cfg.decimation)', 'apply_action',
                'write_data_to_sim', 'sim.step(', 'sim.forward', 'scene.update',
                'contacts.sample', 'contacts.sample', 'sim.sense',
                'episode_length_buf += 1', 'common_step_counter += 1')]

            def is_subsequence(tokens):
                index = 0
                for required_token in required:
                    while index < len(tokens) and tokens[index] != required_token:
                        index += 1
                    if index == len(tokens):
                        return False
                    index += 1
                return True

            checks['physics_loop_tokens'] = dict(
                new=new_tokens, old=old_tokens, required=required,
                ok=is_subsequence(new_tokens) and is_subsequence(old_tokens))
            payload = dict(
                evidence='Q002 4.2(ii) frozen physics configuration and ported structure',
                authorized_changes=['reward recipe 0.5-r1', 'ten-stage table and slots',
                                    'per-slot command sampling', 'new v0.5 diagnostics',
                                    'event order footfall/swing/stall'],
                unauthorized_differences=[name for name, item in checks.items()
                                          if not item['ok']],
                checks=checks, source_sha256=_q002_source_sha())
            _q002_write('q002_physics_invariants.json', payload)
            failed = payload['unauthorized_differences']
            self.assertEqual(failed, [], json.dumps(failed))
        finally:
            old.close()
            new.close()


@unittest.skipUnless(GPU, GPU_REASON)
class Q002FlagsFinitenessTests(unittest.TestCase):
    """Q002 4.2(iii): 80-step full-tensor finiteness and flag consistency.

    This is a live re-check (the canonical run predates the Q002 split): every
    observation tensor of every instance is checked with ``torch.isfinite(...).all()``
    on the complete tensor, not from a saved maximum.  The Q001 long-horizon ratio
    decision is recorded for information only and never asserted.
    """

    def test_full_tensor_finiteness_and_flags_over_80_steps(self):
        import json
        from src.tasks.velocity.v04.env import V04Env
        summaries = []
        for seed in Q001_SEEDS:
            old_a = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1',
                           fixed_vx=.4, initial_cases=[[-1., 0., 0.]])
            old_b = V04Env(num_envs=1, device=DEVICE, seed=seed, stage='S1',
                           fixed_vx=.4, initial_cases=[[-1., 0., 0.]])
            new = V05Env(num_envs=1, device=DEVICE, seed=seed, stage='S1',
                         fixed_vx=.4, initial_cases=[[-1., 0., 0.]])
            frames = []
            try:
                old_a.reset()
                old_b.reset()
                new.reset()
                for step in range(Q001_STEPS):
                    action = torch.full((1, 12), .02 * step, device=DEVICE)
                    obs = {}
                    raw = {}
                    for label, env in (('old_a', old_a), ('old_b', old_b), ('new', new)):
                        observation, reward, terminated, truncated, info = env.step(action)
                        obs[label] = {k: v.detach().clone() for k, v in observation.items()}
                        terminal = getattr(env, 'last_termination', None)
                        raw[label] = dict(
                            terminated=bool(terminated[0]),
                            truncated=bool(truncated[0]),
                            failure=bool(terminal[1][0]) if terminal is not None else None,
                            success=bool(terminal[0][0]) if terminal is not None else None,
                            reason=int(terminal[3][0]) if terminal is not None else None,
                            reward=float(reward[0]))
                    finite = {label: {key: bool(torch.isfinite(value).all())
                                      for key, value in obs[label].items()}
                              for label in obs}
                    for label in finite:
                        self.assertEqual(set(finite[label]), set(Q001_OBS_GROUPS), label)
                        self.assertTrue(all(finite[label].values()), (seed, step, label))
                    flags = {label: {key: raw[label][key] for key in
                                     ('terminated', 'truncated', 'failure', 'success', 'reason')}
                             for label in raw}
                    consistent = flags['old_a'] == flags['old_b'] == flags['new']
                    group_diffs = {key: dict(
                        old_old=float((obs['old_a'][key] - obs['old_b'][key]).abs().max()),
                        old_a_new=float((obs['old_a'][key] - obs['new'][key]).abs().max()),
                        old_b_new=float((obs['old_b'][key] - obs['new'][key]).abs().max()))
                        for key in Q001_OBS_GROUPS}
                    frames.append(dict(
                        seed=seed, step=step, finite=finite, flags=flags,
                        flags_consistent=consistent,
                        old_spread=max(item['old_old'] for item in group_diffs.values()),
                        new_error=max(max(item['old_a_new'], item['old_b_new'])
                                      for item in group_diffs.values()),
                        group_diffs=group_diffs, reward={k: v['reward'] for k, v in raw.items()}))
                old_spreads = [frame['old_spread'] for frame in frames]
                new_errors = [frame['new_error'] for frame in frames]
                history = q001_evaluate(old_spreads, new_errors)
                for index, frame in enumerate(frames):
                    frame['in_stable_prefix'] = index < history['prefix']
                    frame['bound'] = q001_bound(frame['old_spread'])
                terminated_frames = [frame['step'] for frame in frames
                                     if any(flag['terminated'] or flag['truncated']
                                            for flag in frame['flags'].values())]
                summary = dict(
                    seed=seed,
                    evidence='Q002 4.2(iii) live 80-step finiteness and flags',
                    steps=Q001_STEPS,
                    full_tensor_finite_all_steps=all(
                        all(all(item.values()) for item in frame['finite'].values())
                        for frame in frames),
                    flags_consistent_all_steps=all(frame['flags_consistent'] for frame in frames),
                    termination_frames=terminated_frames,
                    flag_field_comparisons=Q001_STEPS * 3 * 5,
                    q001_history_this_run=history,
                    q001_history_status=('passed_under_Q001_amendment'
                                         if history['passed'] else Q002_HISTORY),
                    admission_note=('never used for admission; the canonical status is '
                                    + Q002_HISTORY))
                path = _q002_write(f'q002_recheck_seed_{seed}.json',
                                   dict(summary, frames=frames,
                                        source_sha256=_q002_source_sha()))
                summaries.append(dict(summary, evidence=path))
                self.assertTrue(summary['full_tensor_finite_all_steps'], seed)
                self.assertTrue(summary['flags_consistent_all_steps'], seed)
            finally:
                old_a.close()
                old_b.close()
                new.close()
        _q002_write('q002_recheck_summary.json', dict(
            schema='q002-recheck-1',
            admission=Q002_ADMISSION,
            long_horizon_numeric_equivalence=Q002_LONG_HORIZON,
            q001_history=Q002_HISTORY,
            note=('live re-check for finiteness/flags only; the Q001 long-horizon '
                  'ratio decision is informational and the canonical failed_q001 '
                  'record is unchanged'),
            records=summaries, source_sha256=_q002_source_sha()))
