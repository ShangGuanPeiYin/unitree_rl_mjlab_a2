"""Pure v0.4 speed/orientation/nonfoot/event rewards and posture-gate boundaries.

All inputs are synthetic tensors: no simulator, policy, learning, W&B or GPU.
The gate boundary check constructs a minimal real ``SwingSample``/state so the
30 deg / -10 deg / 55 deg / nonfoot / failure thresholds are exercised where
they are implemented, not re-derived in the test.
"""
import importlib
import math
import unittest

import torch

DT = .02
POSITIVE_RATE = .5 * (1. - math.exp(-4.))  # 0.4908421805556329
POSITIVE_REWARD = DT * POSITIVE_RATE       # 0.009816843611112658


def api():
    try:
        return importlib.import_module('src.tasks.velocity.v04.rewards')
    except ModuleNotFoundError:
        raise AssertionError('V04 reward implementation is missing') from None


def vec(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def flag(*values):
    return torch.tensor(values, dtype=torch.bool)


def gate_sample(rewards, roll=0., pitch=0., nonfoot=False, failure=False, enabled=True):
    """One env, two feet on ground step 0; only the gate inputs vary."""
    b, g = 1, 1
    foot_w = torch.zeros(b, 4, 3)
    foot_w[..., 2] = .032
    found = torch.zeros(b, 4, dtype=torch.bool)
    found[:, :2] = True
    forces = torch.zeros(b, 4, g)
    forces[:, :2, 0] = 6.
    return rewards.SwingSample(
        tile_ids=torch.tensor([5], dtype=torch.long),
        num_steps=torch.tensor([4], dtype=torch.long),
        foot_pos_w=foot_w, foot_pos_b=torch.zeros(b, 4, 3),
        contact_found=found, top_forces=forces,
        geom_tile=torch.tensor([5], dtype=torch.long),
        geom_step=torch.tensor([0], dtype=torch.long),
        geom_top=torch.tensor([0.]),
        geom_union_bounds=torch.tensor([[-.5, .5, -.4, .4]]),
        executed_c=torch.full((b,), .4),
        caller_enabled=torch.full((b,), enabled),
        failure=torch.full((b,), failure),
        roll=torch.full((b,), roll), pitch=torch.full((b,), pitch),
        nonfoot_hit=torch.full((b,), nonfoot))


class SpeedRewardTests(unittest.TestCase):
    def test_stationary_and_exact_command_speed(self):
        r = api()
        still = r.speed_reward(torch.zeros(3), torch.zeros(3), vec(.2, .4, .6), flag(True, True, True))
        torch.testing.assert_close(still, torch.zeros(3), rtol=0, atol=0)
        for c in (.2, .4, .6):
            exact = r.speed_reward(vec(c), vec(0.), vec(c), flag(True))
            self.assertAlmostEqual(float(exact), POSITIVE_REWARD, delta=1e-6)
            self.assertAlmostEqual(float(exact) / DT, POSITIVE_RATE, delta=1e-7)
            # Gate only removes the positive part, so an exact-speed sample is
            # zero when the gate is closed.
            self.assertEqual(float(r.speed_reward(vec(c), vec(0.), vec(c), flag(False))), 0.)

    def test_overspeed_deadband_then_absolute_linear_penalty(self):
        r = api()
        c = .4
        closed = flag(False)

        def rate(vx):
            return float(r.speed_reward(vec(vx), vec(0.), vec(c), closed))

        self.assertEqual(rate(1.05 * c), 0.)
        self.assertAlmostEqual(rate(1.05 * c + 1e-4), -DT * 10. * 1e-4, delta=1e-8)
        self.assertAlmostEqual(rate(1.2 * c), -DT * 10. * (.15 * c), delta=1e-9)
        # Constant slope in absolute vx difference (not a relative square).
        self.assertAlmostEqual((rate(.56) - rate(.52)) / .04, -10. * DT, delta=2e-6)
        # Above the deadband the positive exponent part is still alive when the
        # gate is open; only the over-speed penalty is zero at exactly 1.05 c.
        open_at_deadband = float(r.speed_reward(vec(1.05 * c), vec(0.), vec(c), flag(True)))
        expected = DT * .5 * (math.exp(-4. * .05 ** 2) - math.exp(-4.))
        self.assertAlmostEqual(open_at_deadband, expected, delta=1e-9)

    def test_gate_removes_only_positive_exponential_part(self):
        r = api()
        vx = torch.tensor([0., .3, 1.35, -.2, .4], dtype=torch.float64)
        vy = torch.tensor([0., .4, 0., -.3, .2], dtype=torch.float64)
        c = torch.tensor([.2, .4, .2, .4, .6], dtype=torch.float64)
        closed, opened = torch.zeros(5, dtype=torch.bool), torch.ones(5, dtype=torch.bool)
        gated = r.speed_reward(vx, vy, c, closed)
        open_ = r.speed_reward(vx, vy, c, opened)
        ex, ey = (vx - c) / c, vy / .2
        a = .5 * (torch.exp(-4. * ex.square() - ey.square()) - math.exp(-4.))
        over = 10. * (vx - 1.05 * c).clamp_min(0.)
        back = 2. * (-vx).clamp_min(0.)
        side = .25 * torch.tanh(ey.square())
        expected_closed = DT * (a.clamp_max(0.) - over - back - side)
        expected_open = DT * (a.clamp_min(0.) + a.clamp_max(0.) - over - back - side)
        torch.testing.assert_close(gated, expected_closed, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(open_, expected_open, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(open_ - gated, DT * a.clamp_min(0.), rtol=1e-12, atol=1e-15)
        # Backward and lateral samples keep their penalties when the gate (or a
        # failure/nonfoot mask folded into it) is off.
        self.assertLess(float(gated[3]), 0.)
        self.assertLess(float(gated[1]), 0.)
        self.assertLess(float(gated[2]), 0.)
        self.assertAlmostEqual(float(gated[2]) / DT, -11.4091578, delta=2e-6)

    def test_parking_and_invalid_speed_inputs(self):
        r = api()
        parking = r.speed_reward(vec(.5), vec(.2), vec(0.), flag(True), enabled=flag(False))
        self.assertEqual(float(parking), 0.)
        for c in (0., -.2):
            with self.assertRaises(ValueError):
                r.speed_reward(vec(0.), vec(0.), vec(c), flag(True))
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                r.speed_reward(vec(bad), vec(0.), vec(.4), flag(True))
            with self.assertRaises(ValueError):
                r.speed_reward(vec(0.), vec(bad), vec(.4), flag(True))
            with self.assertRaises(ValueError):
                r.speed_reward(vec(0.), vec(0.), vec(bad), flag(True))
        with self.assertRaises(TypeError):  # 2-D input
            r.speed_reward(torch.zeros(1, 2), vec(0.), vec(.4), flag(True))
        with self.assertRaises(ValueError):  # length mismatch
            r.speed_reward(vec(0., 0.), vec(0.), vec(.4), flag(True))
        with self.assertRaises(TypeError):  # gate dtype
            r.speed_reward(vec(0.), vec(0.), vec(.4), torch.tensor([1]))
        with self.assertRaises(ValueError):  # gate shape
            r.speed_reward(vec(0.), vec(0.), vec(.4), flag(True, False))
        with self.assertRaises(TypeError):  # enabled dtype
            r.speed_reward(vec(0.), vec(0.), vec(0.), flag(True), enabled=torch.tensor([0]))
        for dt in (0., -DT, float('nan'), float('inf'), True, '0.02', None):
            with self.assertRaises(ValueError, msg=repr(dt)):
                r.speed_reward(vec(0.), vec(0.), vec(.4), flag(True), dt=dt)


class OrientationRewardTests(unittest.TestCase):
    def test_orientation_penalty_boundaries(self):
        r = api()
        deg = math.radians

        def rate(pitch_rad, roll_rad=0.):
            return float(r.orientation_reward(vec(roll_rad), vec(pitch_rad))) / DT

        for pitch in (-5., -10., -15., -48.27, 55., 56.):
            lower = max(-deg(5.) - deg(pitch), 0.)
            upper = max(deg(pitch) - deg(55.), 0.)
            self.assertAlmostEqual(rate(deg(pitch)), -(upper ** 2 + 4. * lower ** 2),
                                   delta=2e-6, msg=f'pitch {pitch}')
        for roll in (0., 30., 31.):
            self.assertAlmostEqual(rate(0., deg(roll)), -deg(roll) ** 2, delta=2e-6,
                                   msg=f'roll {roll}')
        combined = rate(deg(-48.27), deg(-31.))
        self.assertAlmostEqual(combined, -((deg(31.) ** 2) + 4. * (deg(43.27) ** 2)),
                               delta=2e-6)
        # These angles are reward penalties only, never termination inputs here.
        self.assertLessEqual(rate(deg(56.)), 0.)
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                r.orientation_reward(vec(bad), vec(0.))
            with self.assertRaises(ValueError):
                r.orientation_reward(vec(0.), vec(bad))
        with self.assertRaises(TypeError):
            r.orientation_reward(torch.zeros(1, 2), vec(0.))
        with self.assertRaises(ValueError):
            r.orientation_reward(vec(0., 0.), vec(0.))

    def test_posture_gate_thresholds(self):
        r = api()
        deg = math.radians
        cases = [
            ('level', 0., 0., True),
            ('roll +30', deg(30.), 0., True),
            ('roll -30', deg(-30.), 0., True),
            ('roll +31', deg(31.), 0., False),
            ('roll -31', deg(-31.), 0., False),
            ('pitch -10', 0., deg(-10.), True),
            ('pitch -10.01', 0., deg(-10.01), False),
            ('pitch 55', 0., deg(55.), True),
            ('pitch 55.01', 0., deg(55.01), False),
        ]
        for name, roll, pitch, expected in cases:
            state = r.SwingClearanceState(1)
            state.update(gate_sample(r))
            result = state.update(gate_sample(r, roll=roll, pitch=pitch))
            self.assertEqual(bool(result.positive_gate[0]), expected, msg=name)
        for name, kwargs in (
                ('nonfoot', dict(nonfoot=True)),
                ('failure', dict(failure=True)),
                ('caller disabled', dict(enabled=False))):
            state = r.SwingClearanceState(1)
            state.update(gate_sample(r))
            result = state.update(gate_sample(r, **kwargs))
            self.assertFalse(bool(result.positive_gate[0]), msg=name)


class NonfootRewardTests(unittest.TestCase):
    def test_exact_threshold_any_link_and_no_dilution(self):
        r = api()
        base = torch.zeros(1, 9, 3)
        exact = base.clone()
        exact[0, 0] = torch.tensor([6., 8., 0.])  # norm exactly 10 N
        self.assertEqual(float(r.nonfoot_reward(exact)), 0.)
        above = base.clone()
        above[0, 4] = torch.tensor([6., 8.001, 0.])
        self.assertAlmostEqual(float(r.nonfoot_reward(above)), -DT, delta=1e-9)
        one = base.clone()
        one[0, 7, 0] = 10.5
        self.assertAlmostEqual(float(r.nonfoot_reward(one)), -DT, delta=1e-9)
        multiple = base.clone()
        multiple[0, 0, 0] = 11.
        multiple[0, 8, 1] = 12.
        self.assertAlmostEqual(float(r.nonfoot_reward(multiple)), -DT, delta=1e-9)
        # A 9-link violation is one -dt, not -dt/9 and not -9*dt.
        self.assertAlmostEqual(float(r.nonfoot_reward(multiple)) / DT, -1., delta=1e-6)
        for bad_shape in (torch.zeros(1, 9), torch.zeros(1, 9, 2), torch.zeros(9, 3)):
            with self.assertRaises(ValueError):
                r.nonfoot_reward(bad_shape)
        with self.assertRaises(ValueError):
            r.nonfoot_reward(torch.zeros(1, 0, 3))
        nonfinite = base.clone()
        nonfinite[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            r.nonfoot_reward(nonfinite)


class EventRewardTests(unittest.TestCase):
    def test_success_failure_priority_and_one_time_events(self):
        r = api()
        success = flag(False, False, True, True)
        failure = flag(False, True, False, True)
        reward = r.event_reward(success, failure)
        torch.testing.assert_close(reward, torch.tensor([0., -4., 10., -4.]), rtol=0, atol=0)
        self.assertEqual(reward.dtype, torch.float32)
        self.assertEqual(reward.shape, (4,))
        # Failure wins in the same frame: only -4, never +10-4=+6.
        self.assertEqual(float(reward[3]), -4.)
        # Events only appear on the frame that reports them; a later no-event
        # frame cannot repay an earlier success or failure.
        none = r.event_reward(flag(False, False, False, False), flag(False, False, False, False))
        torch.testing.assert_close(none, torch.zeros(4), rtol=0, atol=0)
        self.assertEqual(int((reward > 0).sum()), 1)
        self.assertEqual(int((reward < 0).sum()), 2)
        self.assertEqual(torch.nonzero(reward < 0).flatten().tolist(), [1, 3])
        with self.assertRaises(TypeError):
            r.event_reward(torch.tensor([1, 0]), failure)
        with self.assertRaises(TypeError):
            r.event_reward(success, torch.tensor([0, 0, 0, 0]))
        with self.assertRaises(ValueError):
            r.event_reward(flag(True), failure)


if __name__ == '__main__':
    unittest.main()
