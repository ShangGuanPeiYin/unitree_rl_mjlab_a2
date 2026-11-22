"""v0.5-r2 rewards are a line-for-line derivative of the frozen r1 module.

SDS 3 keeps every reward formula, weight, event amount and state machine
unchanged in r2, and SDS 3.9 adds that ``route_error`` never enters the reward.
This module therefore proves:

* the exported symbol set is identical to ``src/tasks/velocity/v05/rewards.py``;
* every pure function returns **bit-identical** tensors for identical fixed
  inputs (speed truth table, gates, all continuous terms, events);
* the three SDS 9.3 scalar anchors reproduce (``3.9733841257``,
  ``-2.4827493695``, ``-6.8393842614``, tolerance ``1e-6``) with
  ``t=1..1000`` and ``gamma^(t-1)`` on ``dt*rate``;
* ``StallState``, ``FootfallState`` and ``SwingClearanceState`` produce identical
  results and identical internal state for identical frame sequences;
* ``dt`` still applies exactly once to continuous terms and never to events.

No simulator, policy, training or GPU is involved.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_r2_bootstrap  # noqa: E402

a2_v05_r2_bootstrap.activate(ROOT)

import dataclasses  # noqa: E402
import math  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.v05 import rewards as r1  # noqa: E402  (read-only reference)
from src.tasks.velocity.v05_r2 import rewards as r2  # noqa: E402
from src.tasks.velocity.v05_r2 import spec  # noqa: E402

DT = .02
GAMMA = .995
STEPS = 1000
TRUTH_TABLE = ((0., -0.472527), (.25, -0.341901), (.5, 0.051819), (1., 1.),
               (1.2, 0.778216), (1.5, 0.006819), (2., -0.792527), (3., -2.12))
REQUIRED_EXPORTS = ('speed_rate', 'speed_reward', 'StallState', 'StallResult',
                    'SwingClearanceState', 'SwingSample', 'SwingResult', 'FootfallState',
                    'FootfallResult', 'footfall_reward', 'nonfoot_reward',
                    'orientation_reward')


def vec(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def flag(*values):
    return torch.tensor(values, dtype=torch.bool)


def rate(amount, dt=DT):
    return float(amount) / dt


def assert_same(test, left, right, path='value'):
    """Recursively compare tensors/floats/dataclasses for exact equality."""
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor, path)
        test.assertEqual(left.dtype, right.dtype, path)
        test.assertEqual(tuple(left.shape), tuple(right.shape), path)
        test.assertTrue(torch.equal(left, right), path)
        return
    if dataclasses.is_dataclass(left):
        # The two modules declare structurally identical records in different
        # namespaces, so identity is compared by name/fields, never by class object.
        test.assertEqual(type(left).__name__, type(right).__name__, path)
        test.assertEqual([field.name for field in dataclasses.fields(left)],
                         [field.name for field in dataclasses.fields(right)], path)
        for field in dataclasses.fields(left):
            assert_same(test, getattr(left, field.name), getattr(right, field.name),
                        f'{path}.{field.name}')
        return
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right), path)
        for key in left:
            assert_same(test, left[key], right[key], f'{path}[{key!r}]')
        return
    if isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(test, a, b, f'{path}[{index}]')
        return
    if isinstance(left, float):
        test.assertEqual(left, right, path)
        return
    test.assertEqual(left, right, path)


class ExportSymbolTests(unittest.TestCase):
    def test_export_set_matches_the_r1_module_exactly(self):
        r1_public = {name for name in dir(r1) if not name.startswith('_')}
        r2_public = {name for name in dir(r2) if not name.startswith('_')}
        self.assertEqual(r1_public, r2_public)
        self.assertLessEqual(set(REQUIRED_EXPORTS), r2_public)
        for name in REQUIRED_EXPORTS:
            self.assertTrue(hasattr(r2, name), name)

    def test_module_source_never_mentions_route_or_a_version_identity(self):
        source = (ROOT / 'src/tasks/velocity/v05_r2/rewards.py').read_text()
        self.assertNotIn('route_error', source)
        self.assertNotIn('route', source)
        # Only the derivation note and the two function labels name the version.
        self.assertEqual(source.count('0.5-r1'), 0)
        self.assertEqual(source.count('0.5-r2'), 5)
        self.assertEqual(source.count('from .spec import'), 1)
        # The r1 module is untouched and still imports its own spec.
        r1_source = (ROOT / 'src/tasks/velocity/v05/rewards.py').read_text()
        self.assertIn('from .spec import', r1_source)
        self.assertNotIn('0.5-r2', r1_source)


class PureFunctionParityTests(unittest.TestCase):
    """Identical inputs must give bit-identical r1/r2 outputs."""

    def test_speed_reward_and_rate(self):
        cases = []
        for u in (0., .25, .5, 1., 1.2, 1.5, 2., 3.):
            for vy in (0., .08, -.16):
                for gate in (True, False):
                    cases.append((vec(u * .4), vec(vy), vec(.4), flag(gate), flag(True)))
        cases.append((vec(0.), vec(0.), vec(0.), flag(True), flag(False)))
        cases.append((vec(.3, -.2, 1.1), vec(0., .2, -.3), vec(.4, .4, .6),
                      flag(True, False, True), flag(True, True, True)))
        for case in cases:
            amount = r2.speed_reward(*case[:4], enabled=case[4])
            assert_same(self, r1.speed_reward(*case[:4], enabled=case[4]), amount)
            assert_same(self, r1.speed_rate(*case[:4], enabled=case[4]),
                        r2.speed_rate(*case[:4], enabled=case[4]))

    def test_dt_override_is_identical(self):
        for dt in (.01, .02, .04):
            left = r1.speed_reward(vec(.6), vec(0.), vec(.4), flag(True),
                                   enabled=flag(True), dt=dt)
            right = r2.speed_reward(vec(.6), vec(0.), vec(.4), flag(True),
                                    enabled=flag(True), dt=dt)
            assert_same(self, left, right)
            assert_same(self, r1.orientation_reward(vec(.1), vec(.2), dt=dt),
                        r2.orientation_reward(vec(.1), vec(.2), dt=dt))

    def test_positive_gate(self):
        support = torch.tensor([[True, True, False, False],
                                [True, False, False, False],
                                [True, True, True, True]], dtype=torch.bool)
        roll = vec(.1, .6, -.1)
        pitch = vec(.1, .1, 1.2)
        nonfoot = flag(False, False, True)
        failure = flag(False, True, False)
        caller = flag(True, True, False)
        assert_same(self, r1.positive_gate(support, roll, pitch, nonfoot, failure, caller),
                    r2.positive_gate(support, roll, pitch, nonfoot, failure, caller))
        self.assertEqual(
            float(r2.positive_gate(support, roll, pitch, nonfoot, failure, caller).sum()), 1.)

    def test_event_and_continuous_terms(self):
        shapes = dict(joint_pos=torch.randn(3, 12, generator=torch.Generator().manual_seed(1)),
                      default=torch.zeros(3, 12),
                      speed=vec(.4, .0, .7),
                      action=torch.randn(3, 12, generator=torch.Generator().manual_seed(2)),
                      previous=torch.randn(3, 12, generator=torch.Generator().manual_seed(3)),
                      ang_vel=torch.randn(3, 3, generator=torch.Generator().manual_seed(4)),
                      command=vec(0., 0., .4, 0., 0., 0., 0., 0., .6).reshape(3, 3),
                      link_forces=torch.randn(3, 5, 3, generator=torch.Generator().manual_seed(5)),
                      roll=vec(.1, -.2, .3),
                      pitch=vec(.2, 1.1, -.2),
                      local_y=vec(0., .21, -.9),
                      local_yaw=vec(0., .1, -3.),
                      active=flag(True, True, False),
                      foot_xy=torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(6)),
                      support=torch.tensor([[True, True, False, False],
                                            [False, False, False, False],
                                            [True, True, True, True]], dtype=torch.bool),
                      union=torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(7)),
                      foot_forces=torch.randn(3, 4, 3,
                                              generator=torch.Generator().manual_seed(8)).abs(),
                      contact=torch.tensor([[True, False, True, False],
                                            [False, False, False, False],
                                            [True, True, True, True]], dtype=torch.bool),
                      duration=torch.tensor([[.02, .02, .02, .02],
                                             [.04, .02, .04, .02],
                                             [.01, .04, .01, .04]]),
                      foot_vel=torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(9)),
                      soft_limits=torch.cat((torch.full((3, 12, 1), -.5),
                                             torch.full((3, 12, 1), .5)), -1),
                      joint_acc=torch.randn(3, 12, generator=torch.Generator().manual_seed(10)),
                      angmom=torch.randn(3, 3, generator=torch.Generator().manual_seed(11)),
                      events=torch.tensor([[True, False, True, True],
                                           [False, False, False, False],
                                           [True, True, True, True]], dtype=torch.bool),
                      success=flag(False, False, True),
                      failure=flag(False, True, False))
        calls = {
            'footfall_reward': ((shapes['events'],), {}),
            'event_reward': ((shapes['success'], shapes['failure']), {}),
            'pose_reward': ((shapes['joint_pos'], shapes['default'], shapes['speed']), {}),
            'action_rate_reward': ((shapes['action'], shapes['previous']), {}),
            'track_angular_velocity_reward': ((shapes['ang_vel'], shapes['command']), {}),
            'stand_still_reward': ((shapes['joint_pos'], shapes['default'],
                                    shapes['command']), {}),
            'nonfoot_reward': ((shapes['link_forces'],), {}),
            'orientation_reward': ((shapes['roll'], shapes['pitch']), {}),
            'center_reward': ((shapes['local_y'], shapes['active']), {}),
            'heading_reward': ((shapes['local_yaw'], shapes['active']), {}),
            'slip_reward': ((shapes['foot_vel'], shapes['contact'], shapes['active']), {}),
            'soft_landing_reward': ((shapes['foot_forces'], shapes['contact'],
                                     shapes['duration'], shapes['active']), {}),
            'edge_reward': ((shapes['foot_xy'], shapes['support'], shapes['union']), {}),
            'joint_limits_reward': ((shapes['joint_pos'], shapes['soft_limits']), {}),
            'joint_acc_reward': ((shapes['joint_acc'],), {}),
            'body_ang_vel_reward': ((shapes['ang_vel'],), {}),
            'angular_momentum_reward': ((shapes['angmom'],), {}),
        }
        for name, (args, kwargs) in calls.items():
            with self.subTest(term=name):
                left = getattr(r1, name)(*args, **kwargs)
                right = getattr(r2, name)(*args, **kwargs)
                assert_same(self, left, right)
                self.assertTrue(torch.isfinite(right).all(), name)
        # Explicit magnitude checks keep the numbers frozen, not just equal.
        self.assertTrue(torch.allclose(
            r1.footfall_reward(shapes['events']), r2.footfall_reward(shapes['events'])))
        self.assertEqual(float(r2.footfall_reward(shapes['events'])[0]), 1.5)
        self.assertEqual(float(r2.event_reward(shapes['success'], shapes['failure'])[1]),
                         -10.)
        # center: -0.5 * min((0.2/0.2)^2, 4) * 0.02 = -0.01 (dt applied once).
        self.assertAlmostEqual(float(r2.center_reward(vec(.2), flag(True))), -.01,
                               delta=1e-9)
        self.assertAlmostEqual(float(r2.center_reward(vec(.4), flag(True))), -.04,
                               delta=1e-7)
        # heading: -0.5 * (1 - cos(pi/2)) * 0.02 = -0.01.
        self.assertAlmostEqual(float(r2.heading_reward(vec(math.pi / 2), flag(True))),
                               -.01, delta=1e-9)


class SpeedTruthTableAndAnchorTests(unittest.TestCase):
    def test_sds_3_3_truth_table(self):
        c = .4
        for u, expected in TRUTH_TABLE:
            with self.subTest(u=u):
                amount = r2.speed_reward(vec(u * c), vec(0.), vec(c), flag(True),
                                         enabled=flag(True))
                self.assertAlmostEqual(rate(amount), expected, delta=1e-6)
                self.assertAlmostEqual(float(amount), DT * expected, delta=2e-8)
                assert_same(self, r1.speed_reward(vec(u * c), vec(0.), vec(c), flag(True),
                                                  enabled=flag(True)), amount)
        for u, expected in TRUTH_TABLE:
            value = r2.speed_rate(vec(u * c), vec(0.), vec(c), flag(True), enabled=flag(True))
            self.assertAlmostEqual(float(value), expected, delta=1e-6)
            amount = r2.speed_reward(vec(u * c), vec(0.), vec(c), flag(True),
                                     enabled=flag(True))
            # Composed amount is dt*rate up to float32 rounding.
            self.assertAlmostEqual(float(amount), DT * float(value), delta=1e-7)

    def test_gate_only_disables_the_positive_part(self):
        c = .4
        open_gate = r2.speed_reward(vec(c), vec(0.), vec(c), flag(True), enabled=flag(True))
        closed = r2.speed_reward(vec(c), vec(0.), vec(c), flag(False), enabled=flag(True))
        self.assertAlmostEqual(float(open_gate), DT, delta=1e-7)
        # Closing g removes exactly the positive exponential part (a=1 here).
        self.assertAlmostEqual(float(closed), 0., delta=1e-7)
        assert_same(self, r1.speed_reward(vec(c), vec(0.), vec(c), flag(False),
                                          enabled=flag(True)), closed)
        # Every negative term survives a closed gate.
        negative = r2.speed_reward(vec(3. * c), vec(.4), vec(c), flag(False),
                                   enabled=flag(True))
        assert_same(self, r1.speed_reward(vec(3. * c), vec(.4), vec(c), flag(False),
                                          enabled=flag(True)), negative)
        self.assertLess(float(negative), -.05)

    def test_sds_9_3_scalar_anchors(self):
        c = .4
        still = r2.speed_rate(vec(0.), vec(0.), vec(c), flag(True), enabled=flag(True))
        matched = r2.speed_rate(vec(c), vec(0.), vec(c), flag(True), enabled=flag(True))
        self.assertAlmostEqual(float(still), spec.STILL_SPEED_RATE, delta=1e-6)
        self.assertAlmostEqual(float(still), -0.4725265416668987, delta=1e-7)
        self.assertAlmostEqual(float(matched), 1., delta=1e-7)
        assert_same(self, r1.speed_rate(vec(0.), vec(0.), vec(c), flag(True),
                                        enabled=flag(True)), still)
        assert_same(self, r1.speed_rate(vec(c), vec(0.), vec(c), flag(True),
                                        enabled=flag(True)), matched)

        def discounted(count):
            return (1. - GAMMA ** count) / (1. - GAMMA)

        stall_step = r2.STALL_REWARD
        self.assertEqual(stall_step, spec.CONTROL_DT * spec.STALL_RATE)
        # The scalar anchors use the exact double recipe constants; the torch
        # implementations are float32 and were compared above (SDS 9.3).
        exact_still = spec.STILL_SPEED_RATE
        exact_matched = spec.MATCH_SPEED_RATE
        self.assertEqual(exact_still, 1.5 * math.exp(-4.) - .5)
        speed_part = DT * exact_still * discounted(STEPS)
        stall_part = stall_step * GAMMA ** 98 * discounted(902)
        stationary = speed_part + stall_part
        matched_total = DT * exact_matched * discounted(STEPS)
        early = (DT * exact_still * discounted(100)
                 + stall_step * (GAMMA ** 98 + GAMMA ** 99)
                 + (-10.) * GAMMA ** 99)
        self.assertAlmostEqual(stationary, -2.4827493695, delta=1e-6)
        self.assertAlmostEqual(matched_total, 3.9733841257, delta=1e-6)
        self.assertAlmostEqual(early, -6.8393842614, delta=1e-6)
        # Exact double references from the r1 module stay reproducible.
        self.assertAlmostEqual(stationary, -2.482749369473146, delta=1e-9)
        self.assertAlmostEqual(matched_total, 3.9733841256846687, delta=1e-9)
        self.assertAlmostEqual(early, -6.839384261367943, delta=1e-9)
        checks = spec.REWARD_CONFIG['speed']['discount_checks']
        self.assertEqual((checks['gamma'], checks['steps'], checks['per_step']),
                         (.995, 1000, 'dt*rate'))
        # The rejected double-dt reading cannot produce the anchors.
        double = DT * DT * exact_still * discounted(STEPS) + stall_part
        self.assertGreater(abs(double - (-2.482749)), .1)


class StateMachineParityTests(unittest.TestCase):
    def _snapshot(self, state):
        return {name: value.clone() for name, value in vars(state).items()
                if isinstance(value, torch.Tensor)}

    def test_stall_state(self):
        left, right = r1.StallState(2), r2.StallState(2)
        left.reset(None, vec(0., .5))
        right.reset(None, vec(0., .5))
        first_penalty = {}
        for step in range(1, 121):
            x = vec(0., .5 + step * .01)
            climbed = flag(False, step > 100)
            result_left = left.update(x, vec(.4, .4), climbed, flag(True, True))
            result_right = right.update(x, vec(.4, .4), climbed, flag(True, True))
            assert_same(self, result_left, result_right, f'step {step}')
            assert_same(self, self._snapshot(left), self._snapshot(right), f'state {step}')
            if float(result_right.reward[0]) != 0. and 0 not in first_penalty:
                first_penalty[0] = step
        self.assertEqual(first_penalty.get(0), 99)
        self.assertAlmostEqual(float(right.reward[0]), spec.STALL_REWARD, delta=1e-8)
        # Climbed envs stop penalising.
        self.assertEqual(float(right.reward[1]), 0.)

    def test_footfall_state(self):
        tile_ids = torch.tensor([10], dtype=torch.long)
        geom_tile = torch.tensor([10, 10, 10], dtype=torch.long)
        geom_step = torch.tensor([0, 1, 2], dtype=torch.long)
        bounds = torch.tensor([[-.5, .5, -.4, .4], [0., .2, -.4, .4], [.2, .4, -.4, .4]])
        foot_xy = torch.stack((torch.full((4,), .1), torch.full((4,), .1)), -1)[None]
        forces = torch.zeros(1, 4, 3)
        forces[:, :, 1] = 6.
        left, right = r1.FootfallState(1), r2.FootfallState(1)
        for frame in range(3):
            result_left = left.update(tile_ids, foot_xy, forces, geom_tile, geom_step,
                                      bounds, flag(False))
            result_right = right.update(tile_ids, foot_xy, forces, geom_tile, geom_step,
                                        bounds, flag(False))
            assert_same(self, result_left, result_right, f'frame {frame}')
            assert_same(self, self._snapshot(left), self._snapshot(right))
            if frame == 0:
                self.assertFalse(result_right.events.any())
            if frame == 1:
                self.assertTrue(result_right.events.all())
                self.assertEqual(float(result_right.reward[0]), 2.)
        self.assertEqual(float(result_right.reward[0]), 0.)

    def test_swing_state(self):
        specs = [(0, 0, 0., (-.5, .5, -.4, .4)), (0, 1, .02, (0., .2, -.4, .4))]
        geom_tile = torch.tensor([item[0] for item in specs], dtype=torch.long)
        geom_step = torch.tensor([item[1] for item in specs], dtype=torch.long)
        geom_top = torch.tensor([item[2] for item in specs])
        bounds = torch.tensor([list(item[3]) for item in specs])

        def sample(module, *, found, forces, foot_w, foot_b, roll=.0):
            return module.SwingSample(
                tile_ids=torch.tensor([0], dtype=torch.long),
                num_steps=torch.tensor([1], dtype=torch.long),
                foot_pos_w=foot_w.clone(), foot_pos_b=foot_b.clone(),
                contact_found=found.clone(), top_forces=forces.clone(),
                geom_tile=geom_tile, geom_step=geom_step, geom_top=geom_top,
                geom_union_bounds=bounds,
                executed_c=vec(.4), caller_enabled=flag(True), failure=flag(False),
                roll=vec(roll), pitch=vec(.0), nonfoot_hit=flag(False))

        left, right = r1.SwingClearanceState(1), r2.SwingClearanceState(1)
        foot_w = torch.zeros(1, 4, 3)
        foot_b = torch.zeros(1, 4, 3)
        foot_w[..., 2] = spec.FOOT_RADIUS
        found = torch.ones(1, 4, dtype=torch.bool)
        forces = torch.zeros(1, 4, 2)
        forces[:, :, 0] = 6.

        def drive(label, module, state, *, found_now, forces_now, foot_w_now, foot_b_now,
                  roll=.0):
            result = state.update(sample(module, found=found_now, forces=forces_now,
                                         foot_w=foot_w_now, foot_b=foot_b_now, roll=roll))
            snapshot = {name: value.clone() for name, value in vars(state).items()
                        if isinstance(value, torch.Tensor)}
            return result, snapshot

        frames = 9
        paid = 0.
        for frame in range(frames):
            if frame < 2:  # two supported frames establish the anchor
                pass
            elif frame == 2:  # first contact-free frame
                found = torch.zeros(1, 4, dtype=torch.bool)
                found[0, 1:] = True
                forces = torch.zeros(1, 4, 2)
                forces[:, 1:, 0] = 6.
                foot_w[0, 0, 0] = -.05
                foot_w[0, 0, 1] = .1
            elif frame in (3, 4, 5):
                foot_w[0, 0, 2] = spec.FOOT_RADIUS + .01 * (frame - 2)
                foot_b[0, 0, 2] = .01 * (frame - 2)
                if frame == 3:
                    foot_w[0, 0, 0] = -.05
                    foot_b[0, 0, 0] = -.05
            elif frame in (6, 7):
                found[0, 0] = True
                forces[0, 0, 1] = 6.
                foot_w[0, 0, 0] = .1
                foot_w[0, 0, 2] = specs[1][2] + spec.FOOT_RADIUS
                foot_b[0, 0, 2] = specs[1][2]
            else:
                found[0, 0] = False
                forces[0, 0, 1] = 0.
            result_left, state_left = drive('left', r1, left, found_now=found,
                                            forces_now=forces, foot_w_now=foot_w,
                                            foot_b_now=foot_b)
            result_right, state_right = drive('right', r2, right, found_now=found,
                                              forces_now=forces, foot_w_now=foot_w,
                                              foot_b_now=foot_b)
            assert_same(self, result_left, result_right, f'frame {frame}')
            assert_same(self, state_left, state_right, f'frame {frame} state')
            paid += float(result_right.reward[0])
        self.assertGreater(paid, 0.)
        self.assertLessEqual(paid, spec.SWING_AMOUNT_PER_TARGET + 1e-6)
        self.assertGreater(float(right.actual_paid_amount.sum()), 0.)


class DtSemanticsTests(unittest.TestCase):
    def test_dt_halves_continuous_amounts_and_leaves_events_alone(self):
        full = r2.speed_reward(vec(.6), vec(0.), vec(.4), flag(True), enabled=flag(True))
        half = r2.speed_reward(vec(.6), vec(0.), vec(.4), flag(True), enabled=flag(True),
                               dt=DT / 2.)
        self.assertAlmostEqual(float(half), float(full) / 2., delta=1e-9)
        events = torch.tensor([[True, False, False, True]], dtype=torch.bool)
        self.assertEqual(float(r2.footfall_reward(events)), 1.)
        self.assertEqual(float(r2.event_reward(flag(True), flag(False))), 10.)
        self.assertEqual(float(r2.event_reward(flag(True), flag(True))), -10.)
        self.assertEqual(float(r2.event_reward(flag(False), flag(False))), 0.)

    def test_disabled_or_nonfinite_inputs_still_fail_closed(self):
        for module in (r1, r2):
            with self.assertRaises(ValueError):
                module.speed_reward(vec(0.), vec(0.), vec(0.), flag(True),
                                    enabled=flag(True))
            with self.assertRaises(ValueError):
                module.speed_reward(vec(float('nan')), vec(0.), vec(.4), flag(True),
                                    enabled=flag(True))
            with self.assertRaises((ValueError, TypeError)):
                module.center_reward(vec(0.), torch.tensor([0]))
        self.assertEqual(float(r2.speed_reward(vec(0.), vec(0.), vec(0.), flag(True),
                                               enabled=flag(False))), 0.)


if __name__ == '__main__':
    unittest.main()
