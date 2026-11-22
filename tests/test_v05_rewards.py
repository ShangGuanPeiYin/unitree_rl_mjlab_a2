"""Pure 0.5-r1 reward functions: speed truth table, gates and every continuous term.

All inputs are synthetic tensors: no simulator, policy, learning, W&B or GPU.  The
speed table in SDS 3.3 is a **per-second rate**, so the tests divide the returned
per-step amount by ``0.02 * dt`` before comparing.  Every continuous term applies
``dt`` exactly once; events (footfall/swing/success/failure) never multiply ``dt``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)

import math  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.v05 import rewards as r  # noqa: E402
from src.tasks.velocity.v05 import spec  # noqa: E402

DT = .02


def vec(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def flag(*values):
    return torch.tensor(values, dtype=torch.bool)


def rate(amount, dt=DT):
    """Convert a per-step speed amount back to the SDS 3.3 per-second rate.

    SDS 3.3's printed 0.02 is the control dt applied once, so the per-step amount
    is ``dt * rate`` (SDS 3.1/9.1-1/9.3).
    """
    return float(amount) / dt


TRUTH_TABLE = ((0., -0.472527), (.25, -0.341901), (.5, 0.051819), (1., 1.),
               (1.2, 0.778216), (1.5, 0.006819), (2., -0.792527), (3., -2.12))


class SpeedRewardTests(unittest.TestCase):
    def test_sds_3_3_truth_table_per_second_rate(self):
        c = .4
        for u, expected in TRUTH_TABLE:
            with self.subTest(u=u):
                amount = r.speed_reward(vec(u * c), vec(0.), vec(c), flag(True),
                                        enabled=flag(True))
                self.assertAlmostEqual(rate(amount), expected, delta=1e-6)
                # The amount is the per-second rate scaled by dt exactly once.
                self.assertAlmostEqual(float(amount), DT * expected, delta=2e-8)

    def test_unweighted_speed_rate_helper_matches_the_table(self):
        c = .4
        for u, expected in TRUTH_TABLE:
            with self.subTest(u=u):
                value = r.speed_rate(vec(u * c), vec(0.), vec(c), flag(True),
                                     enabled=flag(True))
                self.assertAlmostEqual(float(value), expected, delta=1e-6)
        disabled = r.speed_rate(vec(.5), vec(0.), vec(0.), flag(True), enabled=flag(False))
        self.assertEqual(float(disabled), 0.)
        for u in (0., .5, 1., 2.):
            rate_value = r.speed_rate(vec(u * c), vec(0.), vec(c), flag(True), enabled=flag(True))
            amount = r.speed_reward(vec(u * c), vec(0.), vec(c), flag(True), enabled=flag(True))
            # The composed amount is exactly dt * rate (one dt, no extra weight).
            self.assertAlmostEqual(float(amount), DT * float(rate_value), delta=1e-9)
        with self.assertRaises(ValueError):
            r.speed_rate(vec(0.), vec(0.), vec(0.), flag(True), enabled=flag(True))

    def test_dt_halves_the_continuous_amount_only(self):
        c = .4
        full = r.speed_reward(vec(1.5 * c), vec(0.), vec(c), flag(True), enabled=flag(True))
        half = r.speed_reward(vec(1.5 * c), vec(0.), vec(c), flag(True), enabled=flag(True),
                              dt=.01)
        self.assertAlmostEqual(float(half), .5 * float(full), delta=1e-9)
        self.assertAlmostEqual(rate(half, dt=.01), rate(full), delta=1e-6)
        # One-time event amounts are dt independent by construction.
        events = flag(True, False)
        self.assertEqual(float(r.footfall_reward(events)), .50)
        self.assertEqual(float(r.event_reward(flag(True), flag(False))), 10.)
        self.assertEqual(float(r.event_reward(flag(False), flag(True))), -10.)

    def test_exact_and_stationary_speeds(self):
        for c in (.2, .4, .6):
            exact = r.speed_reward(vec(c), vec(0.), vec(c), flag(True), enabled=flag(True))
            self.assertAlmostEqual(rate(exact), 1., delta=1e-6)
            still = r.speed_reward(vec(0.), vec(0.), vec(c), flag(True), enabled=flag(True))
            self.assertAlmostEqual(rate(still), 1.5 * math.exp(-4.) - .5, delta=1e-6)
        self.assertAlmostEqual(spec.STILL_SPEED_RATE, 1.5 * math.exp(-4.) - .5, places=15)
        self.assertAlmostEqual(spec.STILL_SPEED_RATE, -0.4725265416668987, places=15)
        # The exported per-step amounts are dt*rate (the 0.02 in SDS 3.3 is dt).
        self.assertAlmostEqual(spec.STILL_R_SPEED, DT * spec.STILL_SPEED_RATE, places=17)
        self.assertAlmostEqual(spec.STILL_R_SPEED, -0.009450530833337975, places=17)
        self.assertAlmostEqual(spec.MATCH_R_SPEED, DT, places=17)
        self.assertEqual(spec.SPEED_COEFFICIENT, spec.CONTROL_DT)
        self.assertEqual(spec.STALL_COEFFICIENT, spec.CONTROL_DT)

    def test_gate_removes_only_the_positive_exponential_part(self):
        vx = torch.tensor([0., .3, 1.35, -.2, .4, 1.6])
        vy = torch.tensor([0., .4, 0., -.3, .2, -.5])
        c = torch.tensor([.2, .4, .2, .4, .6, .4])
        closed, opened = torch.zeros(6, dtype=torch.bool), torch.ones(6, dtype=torch.bool)
        gated = r.speed_reward(vx, vy, c, closed, enabled=torch.ones(6, dtype=torch.bool))
        active = r.speed_reward(vx, vy, c, opened, enabled=torch.ones(6, dtype=torch.bool))
        u, ey = vx / c, vy / spec.SPEED_SIDE_SCALE
        a = 1.5 * torch.exp(-4. * (u - 1.).square() - ey.square()) - .5
        over = .5 * (u - 1.2).clamp_min(0.).square().clamp_max(4.)
        back = .5 * (-u).clamp_min(0.).square().clamp_max(4.)
        side = .25 * ey.square().clamp_max(4.)
        expected_closed = DT * (a.clamp_max(0.) - over - back - side)
        expected_open = DT * (a.clamp_min(0.) + a.clamp_max(0.) - over - back - side)
        torch.testing.assert_close(gated, expected_closed, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(active, expected_open, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(active - gated, DT * a.clamp_min(0.),
                                   rtol=1e-6, atol=1e-9)
        # Negative samples survive a closed gate.
        for index in (0, 3, 5):
            if float(a[index]) < 0:
                self.assertLess(float(gated[index]), 0.)

    def test_overspeed_deadband_and_caps(self):
        c = .4
        closed = flag(False)
        at_deadband = r.speed_reward(vec(1.2 * c), vec(0.), vec(c), closed, enabled=flag(True))
        just_over = r.speed_reward(vec(1.2 * c + 1e-4), vec(0.), vec(c), closed,
                                   enabled=flag(True))
        # a > 0 and over == 0 at the deadband, so a closed gate makes it exactly zero.
        self.assertEqual(float(at_deadband), 0.)
        self.assertLess(float(just_over), 0.)
        opened = r.speed_reward(vec(1.2 * c), vec(0.), vec(c), flag(True), enabled=flag(True))
        self.assertAlmostEqual(float(opened), DT * (1.5 * math.exp(-.16) - .5),
                               delta=1e-8)
        # u = 3 saturates the over-speed cap at 0.5*4 = 2 and the exponential at -0.5.
        self.assertAlmostEqual(rate(r.speed_reward(vec(3 * c), vec(0.), vec(c), closed,
                                                   enabled=flag(True))), -2.12, delta=1e-6)
        # Lateral term saturates at 0.25*4 = 1.
        lateral = rate(r.speed_reward(vec(c), vec(10.), vec(c), closed, enabled=flag(True)))
        self.assertAlmostEqual(lateral, (1.5 * math.exp(-2500.) - .5) - 1., delta=1e-6)

    def test_disabled_zero_command_is_exact_zero(self):
        for vx, vy in ((0., 0.), (.5, .2), (-1., 3.)):
            parking = r.speed_reward(vec(vx), vec(vy), vec(0.), flag(True), enabled=flag(False))
            self.assertEqual(float(parking), 0.)
        self.assertIs(parking.dtype, torch.float32)
        self.assertEqual(parking.shape, (1,))

    def test_invalid_speed_inputs_are_rejected(self):
        for c in (0., -.2):
            with self.assertRaises(ValueError):
                r.speed_reward(vec(0.), vec(0.), vec(c), flag(True), enabled=flag(True))
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                r.speed_reward(vec(bad), vec(0.), vec(.4), flag(True), enabled=flag(True))
            with self.assertRaises(ValueError):
                r.speed_reward(vec(0.), vec(bad), vec(.4), flag(True), enabled=flag(True))
            with self.assertRaises(ValueError):
                r.speed_reward(vec(0.), vec(0.), vec(bad), flag(True), enabled=flag(True))
        with self.assertRaises(TypeError):  # enabled is a required keyword
            r.speed_reward(vec(0.), vec(0.), vec(.4), flag(True))
        with self.assertRaises(TypeError):
            r.speed_reward(torch.zeros(1, 2), vec(0.), vec(.4), flag(True), enabled=flag(True))
        with self.assertRaises(ValueError):
            r.speed_reward(vec(0., 0.), vec(0.), vec(.4), flag(True), enabled=flag(True))
        with self.assertRaises(TypeError):
            r.speed_reward(vec(0.), vec(0.), vec(.4), torch.tensor([1]), enabled=flag(True))
        with self.assertRaises(ValueError):
            r.speed_reward(vec(0.), vec(0.), vec(.4), flag(True, False), enabled=flag(True))
        with self.assertRaises(TypeError):
            r.speed_reward(vec(0.), vec(0.), vec(0.), flag(True), enabled=torch.tensor([0]))
        for dt in (0., -DT, float('nan'), float('inf'), True, '0.02', None):
            with self.assertRaises(ValueError, msg=repr(dt)):
                r.speed_reward(vec(0.), vec(0.), vec(.4), flag(True), enabled=flag(True), dt=dt)


class GateTests(unittest.TestCase):
    def test_sds_3_2_gate_thresholds(self):
        b = 1
        support = flag(True, True, False, False).reshape(b, 4)
        open_kwargs = dict(roll=torch.zeros(b), pitch=torch.zeros(b),
                           nonfoot_hit=flag(False), failure=flag(False),
                           caller_enabled=flag(True))
        self.assertTrue(bool(r.positive_gate(support, **open_kwargs)[0]))
        one_support = flag(True, False, False, False).reshape(b, 4)
        self.assertFalse(bool(r.positive_gate(one_support, **open_kwargs)[0]))
        for roll in (30., 31., -30., -31.):
            kwargs = dict(open_kwargs, roll=torch.full((b,), math.radians(roll)))
            self.assertEqual(bool(r.positive_gate(support, **kwargs)[0]), abs(roll) <= 30,
                             msg=roll)
        for pitch, expected in ((-10., True), (-10.01, False), (55., True), (55.01, False)):
            kwargs = dict(open_kwargs, pitch=torch.full((b,), math.radians(pitch)))
            self.assertEqual(bool(r.positive_gate(support, **kwargs)[0]), expected, msg=pitch)
        for name in ('nonfoot_hit', 'failure'):
            kwargs = dict(open_kwargs, **{name: flag(True)})
            self.assertFalse(bool(r.positive_gate(support, **kwargs)[0]), name)
        self.assertFalse(bool(r.positive_gate(
            support, **dict(open_kwargs, caller_enabled=flag(False)))[0]))
        with self.assertRaises(ValueError):
            r.positive_gate(flag(True, True, True), **open_kwargs)
        with self.assertRaises(ValueError):
            r.positive_gate(support, **dict(open_kwargs, roll=torch.tensor([float('nan')])))
        with self.assertRaises(ValueError):
            r.positive_gate(support, **dict(open_kwargs, minimum_supports=0))


class ContinuousTermTests(unittest.TestCase):
    def test_orientation_penalty_boundaries(self):
        deg = math.radians

        def amount(pitch, roll=0.):
            return float(r.orientation_reward(vec(roll), vec(pitch)))

        for pitch in (-5., -10., -15., -48.27, 55., 56.):
            lower = max(deg(-5.) - deg(pitch), 0.)
            upper = max(deg(pitch) - deg(55.), 0.)
            self.assertAlmostEqual(amount(deg(pitch)),
                                   -DT * (upper ** 2 + 4. * lower ** 2), delta=2e-8)
        for roll in (0., 30., 31.):
            self.assertAlmostEqual(amount(0., deg(roll)), -DT * deg(roll) ** 2, delta=2e-9)
        self.assertEqual(spec.REWARD_CONFIG['orientation']['weight'], -1.)
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                r.orientation_reward(vec(bad), vec(0.))
        with self.assertRaises(ValueError):
            r.orientation_reward(vec(0.), vec(bad))

    def test_nonfoot_is_any_link_without_dilution(self):
        base = torch.zeros(1, 9, 3)
        exact = base.clone()
        exact[0, 0] = torch.tensor([6., 8., 0.])  # norm exactly 10 N
        self.assertEqual(float(r.nonfoot_reward(exact)), 0.)
        above = base.clone()
        above[0, 4] = torch.tensor([6., 8.001, 0.])
        self.assertAlmostEqual(float(r.nonfoot_reward(above)), -DT, delta=1e-9)
        multiple = base.clone()
        multiple[0, 0, 0] = 11.
        multiple[0, 8, 1] = 12.
        self.assertAlmostEqual(float(r.nonfoot_reward(multiple)), -DT, delta=1e-9)
        self.assertAlmostEqual(float(r.nonfoot_reward(multiple)) / DT, -1., delta=1e-7)
        for bad_shape in (torch.zeros(1, 9), torch.zeros(1, 9, 2), torch.zeros(9, 3)):
            with self.assertRaises(ValueError):
                r.nonfoot_reward(bad_shape)
        nonfinite = base.clone()
        nonfinite[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            r.nonfoot_reward(nonfinite)

    def test_pose_uses_the_new_0_05_weight_and_active_baseline(self):
        default = torch.zeros(1, 12)
        joint_pos = default.clone()
        joint_pos[0, 0] = .15
        sigma = torch.tensor([.15, .35, .5] * 4)
        raw = torch.exp(-(((joint_pos - default) / sigma).square()).mean(-1))
        expected_active = DT * .05 * (raw - 1.)
        self.assertAlmostEqual(float(r.pose_reward(joint_pos, default, vec(.4))),
                               float(expected_active), delta=1e-9)
        # A zero command uses the tighter sigma triple [0.05, 0.1, 0.15].
        sigma_still = torch.tensor([.05, .1, .15] * 4)
        raw_still = torch.exp(-(((joint_pos - default) / sigma_still).square()).mean(-1))
        expected_still = DT * .05 * raw_still
        self.assertAlmostEqual(float(r.pose_reward(joint_pos, default, vec(0.))),
                               float(expected_still), delta=1e-9)
        # Exactly at the default pose the active baseline cancels the reward.
        self.assertAlmostEqual(float(r.pose_reward(default, default, vec(.4))), 0., delta=1e-9)
        self.assertEqual(spec.REWARD_CONFIG['pose']['weight'], .05)

    def test_action_rate_uses_the_new_0_02_weight(self):
        action = torch.zeros(1, 12)
        previous = action.clone()
        previous[0, 3] = .1
        expected = DT * (-.02) * .01
        self.assertAlmostEqual(float(r.action_rate_reward(action, previous)), expected,
                               delta=1e-12)
        self.assertEqual(spec.REWARD_CONFIG['continuous_weights']['action_rate'], -.02)
        with self.assertRaises(ValueError):
            r.action_rate_reward(torch.zeros(12), torch.zeros(12))

    def test_stand_still_is_inactive_for_a_moving_command(self):
        default = torch.zeros(1, 12)
        joint_pos = default.clone()
        joint_pos[0, 0] = .1
        command = vec(.4, 0., 0.).reshape(1, 3)
        self.assertEqual(float(r.stand_still_reward(joint_pos, default, command)), 0.)
        still = r.stand_still_reward(joint_pos, default, torch.zeros(1, 3))
        self.assertAlmostEqual(float(still), -DT * .01, delta=1e-9)
        # A 1-D command norm is accepted with the same result.
        self.assertAlmostEqual(float(r.stand_still_reward(joint_pos, default, vec(0.))),
                               float(still), delta=1e-9)
        self.assertEqual(spec.REWARD_CONFIG['stand_still']['is_stall_term'], False)

    def test_track_angular_velocity_baseline(self):
        command = vec(0., 0., .5).reshape(1, 3)
        still_w = torch.zeros(1, 3)
        raw = math.exp(-.25 / .5)
        self.assertAlmostEqual(float(r.track_angular_velocity_reward(still_w, command)),
                               DT * (raw - 1.), delta=1e-9)
        # With an exactly zero command the exponential raw value is 1 and the
        # active baseline is not subtracted, so the weighted rate is +1.
        idle = torch.zeros(1, 3)
        self.assertAlmostEqual(float(r.track_angular_velocity_reward(still_w, idle)), DT,
                               delta=1e-9)
        matching = vec(0., 0., .5).reshape(1, 3)
        self.assertAlmostEqual(float(r.track_angular_velocity_reward(matching, command)),
                               0., delta=1e-12)

    def test_center_and_heading_apply_to_every_active_episode(self):
        self.assertAlmostEqual(float(r.center_reward(vec(.2), flag(True))), -DT * .5,
                               delta=1e-9)
        self.assertEqual(float(r.center_reward(vec(.2), flag(False))), 0.)
        # Clamped at 4.
        self.assertAlmostEqual(float(r.center_reward(vec(5.), flag(True))),
                               -DT * .5 * 4., delta=1e-9)
        yaw = .1
        self.assertAlmostEqual(float(r.heading_reward(vec(yaw), flag(True))),
                               -DT * .5 * (1. - math.cos(yaw)), delta=1e-9)
        self.assertEqual(float(r.heading_reward(vec(yaw), flag(False))), 0.)

    def test_slip_and_soft_landing_are_sums_not_means(self):
        velocity = torch.zeros(1, 4, 2)
        velocity[0, 0] = torch.tensor([2., 1.])
        velocity[0, 3] = torch.tensor([1., 0.])
        contact = flag(True, False, False, True).reshape(1, 4)
        self.assertAlmostEqual(float(r.slip_reward(velocity, contact, flag(True))),
                               -DT * .25 * 6., delta=1e-9)
        self.assertEqual(float(r.slip_reward(velocity, contact, flag(False))), 0.)
        forces = torch.zeros(1, 4, 3)
        forces[0, 0] = torch.tensor([6., 8., 0.])
        duration = torch.zeros(1, 4)
        duration[0, 0] = DT
        landing = flag(True, False, False, False).reshape(1, 4)
        self.assertAlmostEqual(
            float(r.soft_landing_reward(forces, landing, duration, flag(True))),
            -DT * .001 * 10., delta=1e-9)
        old = torch.zeros(1, 4)
        old[0, 0] = DT + 1e-6
        self.assertEqual(float(r.soft_landing_reward(forces, landing, old, flag(True))), 0.)

    def test_edge_penalty_inside_the_42mm_inset(self):
        bounds = torch.tensor([[[0., .2, -.4, .4]] * 4], dtype=torch.float32)
        feet = torch.zeros(1, 4, 2)
        feet[0, :, 0] = .1
        support = flag(True, False, False, False).reshape(1, 4)
        self.assertEqual(float(r.edge_reward(feet, support, bounds)), 0.)
        feet[0, 0, 0] = .0  # exactly on the boundary -> full inset penalty
        self.assertAlmostEqual(float(r.edge_reward(feet, support, bounds)),
                               -DT * 1. / 4., delta=1e-9)
        feet[0, 0, 0] = .021  # half of the inset
        self.assertAlmostEqual(float(r.edge_reward(feet, support, bounds)),
                               -DT * .25 / 4., delta=1e-9)
        self.assertEqual(float(r.edge_reward(feet, flag(False, False, False, False).reshape(1, 4),
                                             bounds)), 0.)

    def test_joint_limit_joint_acc_body_rate_and_angular_momentum(self):
        joint_pos = torch.zeros(1, 12)
        limits = torch.stack((torch.full((12,), -1.), torch.full((12,), 1.))).T[None]
        limits[0, 0] = torch.tensor([.1, .2])
        self.assertAlmostEqual(float(r.joint_limits_reward(joint_pos, limits)),
                               -DT * 10. * .1, delta=1e-8)
        acc = torch.zeros(1, 12)
        acc[0, 0] = 3.
        self.assertAlmostEqual(float(r.joint_acc_reward(acc)), -DT * 2.5e-7 * 9.,
                               delta=1e-12)
        body = torch.tensor([[.5, .5, 9.]])
        self.assertAlmostEqual(float(r.body_ang_vel_reward(body)), -DT * .05 * .5, delta=1e-8)
        momentum = torch.tensor([[1., 2., 3.]])
        self.assertAlmostEqual(float(r.angular_momentum_reward(momentum)),
                               -DT * .025 * 14., delta=1e-8)

    def test_event_amounts_and_failure_precedence(self):
        success = flag(False, False, True, True)
        failure = flag(False, True, False, True)
        reward = r.event_reward(success, failure)
        torch.testing.assert_close(reward, torch.tensor([0., -10., 10., -10.]))
        self.assertEqual(float(reward[3]), -10.)
        none = r.event_reward(flag(False), flag(False))
        self.assertEqual(float(none), 0.)
        with self.assertRaises(TypeError):
            r.event_reward(torch.tensor([1, 0]), failure)
        with self.assertRaises(ValueError):
            r.event_reward(flag(True), failure)


class DiscountScalarTests(unittest.TestCase):
    """SDS 9.3's three published discount sums must be reproducible (gamma=0.995)."""

    GAMMA = .995
    STEPS = 1000
    C = .4

    def discounted_sum(self, count):
        return (1. - self.GAMMA ** count) / (1. - self.GAMMA)

    def test_sds_9_3_stationary_matched_and_early_failure(self):
        c = self.C
        # The scalar anchors use the exact double recipe constants; the torch
        # implementations are float32 and are checked against them separately.
        still = spec.STILL_SPEED_RATE
        matched = spec.MATCH_SPEED_RATE
        self.assertAlmostEqual(
            float(r.speed_rate(vec(0.), vec(0.), vec(c), flag(True), enabled=flag(True))),
            still, delta=1e-6)
        self.assertAlmostEqual(
            float(r.speed_rate(vec(c), vec(0.), vec(c), flag(True), enabled=flag(True))),
            matched, delta=1e-6)
        self.assertEqual(spec.STILL_SPEED_RATE, 1.5 * math.exp(-4.) - .5)
        stall_step = spec.STALL_REWARD
        self.assertEqual(stall_step, spec.CONTROL_DT * spec.STALL_RATE)
        self.assertAlmostEqual(stall_step, -.005, delta=1e-15)
        # Stationary: speed every step plus the sustained-stall penalty from t=99.
        speed_part = DT * still * self.discounted_sum(self.STEPS)
        stall_part = stall_step * self.GAMMA ** 98 * self.discounted_sum(902)
        stationary = speed_part + stall_part
        self.assertAlmostEqual(speed_part, -1.8775294596, delta=1e-9)
        self.assertAlmostEqual(stall_part, -0.6052199098, delta=1e-9)
        self.assertAlmostEqual(stationary, -2.482749, delta=1e-6)
        self.assertAlmostEqual(stationary, -2.482749369473146, delta=1e-9)
        # Matched command: dt*rate = 0.02 per step for 1000 steps.
        matched_total = DT * matched * self.discounted_sum(self.STEPS)
        self.assertAlmostEqual(matched_total, 3.973384, delta=1e-6)
        self.assertAlmostEqual(matched_total, 3.9733841256846687, delta=1e-9)
        # Early failure at t=100: identical 1..100 speed/stall sequence + (-10).
        early = (DT * still * self.discounted_sum(100)
                 + stall_step * (self.GAMMA ** 98 + self.GAMMA ** 99)
                 + (-10.) * self.GAMMA ** 99)
        self.assertAlmostEqual(early, -6.839384, delta=1e-6)
        self.assertAlmostEqual(early, -6.839384261367943, delta=1e-9)
        # The rejected double-dt reading cannot produce these anchors.
        double = DT * DT * still * self.discounted_sum(self.STEPS) \
            + stall_step * self.GAMMA ** 98 * self.discounted_sum(902)
        self.assertGreater(abs(double - (-2.482749)), .1)

    def test_published_anchors_are_frozen_in_the_recipe(self):
        checks = spec.REWARD_CONFIG['speed']['discount_checks']
        self.assertEqual(checks['gamma'], .995)
        self.assertEqual(checks['steps'], 1000)
        self.assertEqual(checks['per_step'], 'dt*rate')
        self.assertEqual((checks['stationary'], checks['matched'], checks['early_failure']),
                         (-2.482749, 3.973384, -6.839384))
        self.assertEqual(spec.REWARD_CONFIG['speed']['per_step_formula'], 'dt * speed_rate')
        self.assertEqual(spec.REWARD_CONFIG['stall']['per_step_formula'], 'dt * stall_rate')


class CompositionTests(unittest.TestCase):
    def test_continuous_terms_and_events_add_up_to_the_total(self):
        b = 2
        c = torch.full((b,), .4)
        vx = torch.tensor([.4, .3])
        vy = torch.tensor([0., .05])
        gate = flag(True, True)
        executed = c
        speed = r.speed_reward(vx, vy, executed, gate, enabled=flag(True, True))
        stall = torch.zeros(b)
        stall_state = r.StallState(b)
        stall_state.reset(torch.arange(b), torch.zeros(b))
        for _ in range(51):
            stall = stall_state.update(torch.zeros(b), c, flag(False, False), flag(True, True)).reward
        footfall_events = flag(True, True, False, False).reshape(1, 4).expand(b, 4).clone()
        footfall = r.footfall_reward(footfall_events)
        swing = torch.tensor([.15, .0])
        success = flag(False, True)
        failure = flag(False, False)
        orientation = r.orientation_reward(torch.zeros(b), torch.zeros(b))
        nonfoot = r.nonfoot_reward(torch.zeros(b, 9, 3))
        pose = r.pose_reward(torch.zeros(b, 12), torch.zeros(b, 12), c)
        action_rate = r.action_rate_reward(torch.zeros(b, 12), torch.zeros(b, 12))
        track = r.track_angular_velocity_reward(torch.zeros(b, 3), c[:, None].expand(b, 3))
        stand = r.stand_still_reward(torch.zeros(b, 12), torch.zeros(b, 12), c[:, None].expand(b, 3))
        center = r.center_reward(torch.zeros(b), flag(True, True))
        heading = r.heading_reward(torch.zeros(b), flag(True, True))
        slip = r.slip_reward(torch.zeros(b, 4, 2), torch.zeros(b, 4, dtype=torch.bool),
                             flag(True, True))
        landing = r.soft_landing_reward(torch.zeros(b, 4, 3), torch.zeros(b, 4, dtype=torch.bool),
                                        torch.zeros(b, 4), flag(True, True))
        edge = r.edge_reward(torch.zeros(b, 4, 2), torch.zeros(b, 4, dtype=torch.bool),
                             torch.zeros(b, 4, 4))
        terms = dict(speed=speed, stall=stall, footfall=footfall, swing=swing,
                     success=r.event_reward(success, failure),
                     failure=failure.to(torch.float32) * spec.FAILURE_AMOUNT,
                     orientation=orientation, nonfoot=nonfoot, pose=pose,
                     action_rate=action_rate, track_angular_velocity=track,
                     stand_still=stand, center=center, heading=heading, slip=slip,
                     soft_landing=landing, edge=edge)
        total = torch.stack(list(terms.values())).sum(0)
        manual = (speed + stall + footfall + swing
                  + terms['success'] + terms['failure']
                  + orientation + nonfoot + pose + action_rate + track + stand
                  + center + heading + slip + landing + edge)
        torch.testing.assert_close(total, manual, rtol=0, atol=1e-5)
        # The one-time events are paid in full, never scaled by dt.
        self.assertAlmostEqual(float(terms['footfall'][0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(terms['swing'][0]), .15, delta=1e-6)

    def test_stationary_total_matches_the_scalar_sds_expectation(self):
        # SDS 3.3/9.3: a stationary 0.4 m/s command has the negative speed rate,
        # and the stall/window plus positive events cannot out-earn it here.
        b = 1
        c = torch.full((b,), .4)
        speed = r.speed_reward(torch.zeros(b), torch.zeros(b), c, flag(True), enabled=flag(True))
        self.assertAlmostEqual(rate(speed), 1.5 * math.exp(-4.) - .5, delta=1e-6)
        stall = r.StallState(1)
        stall.reset(torch.tensor([0]), torch.zeros(1))
        reward = torch.zeros(1)
        for _ in range(120):
            reward = stall.update(torch.zeros(1), c, flag(False), flag(True)).reward
        self.assertAlmostEqual(float(reward), -0.005, delta=1e-9)
        self.assertAlmostEqual(float(speed) + float(reward),
                               DT * (1.5 * math.exp(-4.) - .5) - .005, delta=1e-9)


if __name__ == '__main__':
    unittest.main()
