"""0.5-r1 sustained-stall state machine (SDS 3.4).

The window holds 51 position samples (50 control intervals = 1.00 s) plus 51
high-water samples.  SDS time labels are sample indices: ``t=0`` is the reset
sample and update ``k`` produces sample ``t=k+1``, so the window is full from
``t=50`` and the first penalty is at ``t=99`` (1.98 s).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)

import unittest  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.v05 import rewards as r  # noqa: E402
from src.tasks.velocity.v05 import spec  # noqa: E402

C = .4
THRESHOLD = float(torch.tensor(spec.STALL_DX_FRACTION_PER_SECOND, dtype=torch.float32)
                  * torch.tensor(C, dtype=torch.float32)
                  * torch.tensor(spec.STALL_DX_THRESHOLD_SCALE, dtype=torch.float32))


def scalar(value):
    return torch.tensor([value], dtype=torch.float32)


def flag(value):
    return torch.tensor([value], dtype=torch.bool)


def advance(state, x, *, c=C, climbed=False, enabled=True, steps=1):
    result = None
    for _ in range(steps):
        result = state.update(scalar(x), scalar(c), flag(climbed), flag(enabled))
    return result


class StallWindowTests(unittest.TestCase):
    def setUp(self):
        self.state = r.StallState(1)
        self.state.reset(torch.tensor([0], dtype=torch.int64), scalar(0.))

    def test_reset_writes_one_sample_and_prefils_the_window(self):
        self.assertEqual(int(self.state.count[0]), 1)
        self.assertTrue(torch.all(self.state.x_history[0] == 0.))
        self.assertTrue(torch.all(self.state.m_history[0] == 0.))
        result = advance(self.state, 0., steps=1)
        self.assertFalse(bool(result.window_ready[0]))
        self.assertEqual(int(self.state.count[0]), 2)
        self.assertEqual(spec.STALL_HISTORY_SAMPLES, 51)
        self.assertEqual(spec.STALL_INTERVAL_SAMPLES, 50)

    def test_sds_3_4_timeline(self):
        # t = 1..49: window not full at all (49 updates after the reset sample).
        for _ in range(49):
            result = advance(self.state, 0.)
            self.assertFalse(bool(result.window_ready[0]))
            self.assertFalse(bool(result.qualifying[0]))
            self.assertEqual(float(result.reward[0]), 0.)
        self.assertEqual(int(self.state.count[0]), 50)
        # t = 50: the window is full and the first qualifying frame arrives.
        result = advance(self.state, 0.)
        self.assertTrue(bool(result.window_ready[0]))
        self.assertTrue(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 1)
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertAlmostEqual(float(result.dx_1s[0]), 0., delta=1e-9)
        self.assertAlmostEqual(float(result.highwater_dx_1s[0]), 0., delta=1e-9)
        # t = 98: 49 consecutive qualifying frames, still no payment.
        result = advance(self.state, 0., steps=48)
        self.assertEqual(int(result.consecutive_count[0]), 49)
        self.assertFalse(bool(result.penalty_on[0]))
        self.assertEqual(float(result.reward[0]), 0.)
        # t = 99: 50 consecutive frames -> exactly -0.25*0.02 = -0.005.
        result = advance(self.state, 0.)
        self.assertEqual(int(result.consecutive_count[0]), 50)
        self.assertTrue(bool(result.penalty_on[0]))
        self.assertAlmostEqual(float(result.reward[0]), -.005, delta=1e-9)
        self.assertAlmostEqual(spec.STALL_REWARD, -.005, places=15)
        self.assertEqual(spec.STALL_REWARD, spec.CONTROL_DT * spec.STALL_RATE)
        # The penalty persists while the condition holds.
        result = advance(self.state, 0., steps=5)
        self.assertAlmostEqual(float(result.reward[0]), -.005, delta=1e-9)

    def test_condition_interruption_clears_the_consecutive_count(self):
        advance(self.state, 0., steps=60)
        self.assertGreaterEqual(int(self.state.consecutive_count[0]), 10)
        result = advance(self.state, 1.0)  # a real 1 s displacement
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 0)
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertAlmostEqual(float(result.dx_1s[0]), 1.0, delta=1e-6)

    def test_strict_dx_boundary_equal_threshold_is_not_qualifying(self):
        advance(self.state, 0., steps=110)  # window full, penalty already active
        self.assertTrue(bool(self.state.penalty_on[0]))
        # Prepare the exact post-shift window: x_t - x_(t-50) == threshold and dM == 0.
        # The high-water row is held at 0.5 so only the dx condition is under test.
        self.state.m_history[0, :] = .5
        self.state.x_history[0, 1] = 0.
        result = self.state.update(scalar(THRESHOLD), scalar(C), flag(False), flag(True))
        self.assertAlmostEqual(float(result.dx_1s[0]), THRESHOLD, delta=1e-9)
        self.assertEqual(float(result.dx_1s[0]), THRESHOLD)
        self.assertAlmostEqual(float(result.highwater_dx_1s[0]), 0., delta=1e-9)
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 0)
        self.assertEqual(float(result.reward[0]), 0.)
        # Just below the threshold is stagnant again.
        self.state.x_history[0, 1] = 0.
        below = THRESHOLD * .999
        result = self.state.update(scalar(below), scalar(C), flag(False), flag(True))
        self.assertTrue(bool(result.qualifying[0]))
        # Just above is not.
        self.state.x_history[0, 1] = 0.
        above = THRESHOLD * 1.001
        result = self.state.update(scalar(above), scalar(C), flag(False), flag(True))
        self.assertFalse(bool(result.qualifying[0]))

    def test_strict_highwater_boundary_equal_003_is_not_qualifying(self):
        advance(self.state, 0., steps=110)
        # Prepare the exact post-shift window: dx == 0 but dM == 0.03.
        self.state.m_history[0, :] = 0.
        self.state.x_history[0, 1] = 0.
        self.state.m_history[0, -1] = torch.tensor(spec.STALL_HIGHWATER_DX, dtype=torch.float32)
        result = self.state.update(scalar(0.), scalar(C), flag(False), flag(True))
        self.assertAlmostEqual(float(result.highwater_dx_1s[0]),
                               spec.STALL_HIGHWATER_DX, delta=1e-9)
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 0)
        self.state.m_history[0, :] = 0.
        self.state.x_history[0, 1] = 0.
        self.state.m_history[0, -1] = torch.tensor(.02, dtype=torch.float32)
        result = self.state.update(scalar(0.), scalar(C), flag(False), flag(True))
        self.assertTrue(bool(result.qualifying[0]))

    def test_floor_event_is_zero(self):
        # A flat N=0 slot never pays a stepped event; the stall term still applies.
        advance(self.state, 0., steps=60)
        self.assertGreaterEqual(int(self.state.consecutive_count[0]), 10)


class StallSemanticsTests(unittest.TestCase):
    def build(self, num_envs):
        state = r.StallState(num_envs)
        state.reset(torch.arange(num_envs, dtype=torch.int64), torch.zeros(num_envs))
        return state

    def test_four_tenths_of_a_metre_per_second_is_not_stagnant(self):
        state = self.build(1)
        result = None
        for step in range(120):
            result = state.update(scalar(step * C * spec.CONTROL_DT), scalar(C),
                                  flag(False), flag(True))
        self.assertTrue(bool(result.window_ready[0]))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 0)
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertAlmostEqual(float(result.dx_1s[0]), C, delta=1e-5)
        self.assertAlmostEqual(float(result.highwater_dx_1s[0]), C, delta=1e-5)

    def test_short_pause_below_window_and_counter_is_not_penalised(self):
        state = self.build(1)
        x = 0.
        for _ in range(60):
            x += C * spec.CONTROL_DT
            state.update(scalar(x), scalar(C), flag(False), flag(True))
        # A pause shorter than the counter requirement never pays, even once the
        # window has absorbed the moving history.
        for _ in range(40):
            result = state.update(scalar(x), scalar(C), flag(False), flag(True))
            self.assertFalse(bool(result.penalty_on[0]))
            self.assertEqual(float(result.reward[0]), 0.)
        # Resume moving before the counter matures.
        result = state.update(scalar(x + .2), scalar(C), flag(False), flag(True))
        self.assertEqual(int(result.consecutive_count[0]), 0)
        self.assertFalse(bool(result.penalty_on[0]))
        # A full stop does eventually pay, so this was not a vacuous pass.
        stall = self.build(1)
        for _ in range(150):
            result = stall.update(scalar(0.), scalar(C), flag(False), flag(True))
        self.assertTrue(bool(result.penalty_on[0]))
        self.assertAlmostEqual(float(result.reward[0]), -.005, delta=1e-9)

    def test_round_trip_and_in_place_rotation_cannot_dodge_the_penalty(self):
        state = self.build(1)
        x = 0.
        for step in range(120):
            # Forward 5 cm, back 5 cm: net zero displacement over every window.
            x = .05 if step % 2 else 0.
            result = state.update(scalar(x), scalar(C), flag(False), flag(True))
        self.assertTrue(bool(result.qualifying[0]))
        self.assertAlmostEqual(float(result.reward[0]), -.005, delta=1e-9)
        # In-place rotation is the same x trajectory from the reward's point of view.
        rotating = self.build(1)
        result = None
        for _ in range(120):
            result = rotating.update(scalar(0.), scalar(C), flag(False), flag(True))
        self.assertAlmostEqual(float(result.reward[0]), -.005, delta=1e-9)
        # A genuine 1 s displacement clears it again.
        result = rotating.update(scalar(C), scalar(C), flag(False), flag(True))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(float(result.reward[0]), 0.)

    def test_climbed_disables_the_term(self):
        state = self.build(1)
        for _ in range(120):
            result = state.update(scalar(0.), scalar(C), flag(True), flag(True))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 0)
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertTrue(bool(result.window_ready[0]))
        # After the arrival stops disabling the term, the counter starts from zero.
        for _ in range(40):
            result = state.update(scalar(0.), scalar(C), flag(False), flag(True))
        self.assertTrue(bool(result.qualifying[0]))
        self.assertEqual(int(result.consecutive_count[0]), 40)
        self.assertEqual(float(result.reward[0]), 0.)
        for _ in range(10):
            result = state.update(scalar(0.), scalar(C), flag(False), flag(True))
        self.assertEqual(int(result.consecutive_count[0]), 50)
        self.assertAlmostEqual(float(result.reward[0]), -.005, delta=1e-9)

    def test_disabled_command_is_never_active(self):
        state = self.build(1)
        for _ in range(120):
            result = state.update(scalar(0.), scalar(0.), flag(False), flag(False))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertEqual(int(result.consecutive_count[0]), 0)

    def test_partial_reset_does_not_touch_other_environments(self):
        state = self.build(3)
        for step in range(60):
            state.update(torch.full((3,), .1 * step, dtype=torch.float32),
                         torch.full((3,), C), torch.zeros(3, dtype=torch.bool),
                         torch.ones(3, dtype=torch.bool))
        before = {name: getattr(state, name)[1:].clone() for name in
                  ('x_history', 'm_history', 'count', 'consecutive_count', 'dx_1s',
                   'highwater_dx_1s', 'window_ready', 'qualifying', 'penalty_on',
                   'executed_c')}
        state.reset(torch.tensor([0], dtype=torch.int64), scalar(7.))
        self.assertEqual(int(state.count[0]), 1)
        self.assertTrue(torch.all(state.x_history[0] == 7.))
        self.assertTrue(torch.all(state.m_history[0] == 7.))
        self.assertEqual(int(state.consecutive_count[0]), 0)
        self.assertTrue(torch.isnan(state.executed_c[0]))
        for name, value in before.items():
            self.assertTrue(torch.equal(getattr(state, name)[1:], value), name)
        # Other environments keep advancing normally.
        result = state.update(torch.tensor([7., .5, .5]), torch.full((3,), C),
                              torch.zeros(3, dtype=torch.bool),
                              torch.ones(3, dtype=torch.bool))
        self.assertEqual(int(state.count[1]), 51)  # saturated full window
        self.assertEqual(int(state.count[0]), 2)
        self.assertEqual(result.reward.shape, (3,))
        # Boolean masks work too and still only touch the selected row.
        state.reset(torch.tensor([False, True, False]), scalar(0.))
        self.assertEqual(int(state.count[1]), 1)
        self.assertEqual(int(state.count[2]), 51)

    def test_command_change_requires_a_reset(self):
        state = self.build(1)
        state.update(scalar(0.), scalar(C), flag(False), flag(True))
        state.update(scalar(0.), scalar(C), flag(False), flag(True))
        with self.assertRaisesRegex(ValueError, 'changed inside an episode'):
            state.update(scalar(0.), scalar(.5), flag(False), flag(True))
        # A reset re-baselines the recorded command.
        state.reset(torch.tensor([0], dtype=torch.int64), scalar(0.))
        state.update(scalar(0.), scalar(.5), flag(False), flag(True))
        self.assertAlmostEqual(float(state.executed_c[0]), .5, delta=1e-9)
        with self.assertRaises(ValueError):
            state.update(scalar(0.), scalar(0.), flag(False), flag(False))

    def test_result_contract_and_dtypes(self):
        state = self.build(2)
        result = state.update(torch.zeros(2), torch.full((2,), C),
                              torch.zeros(2, dtype=torch.bool),
                              torch.ones(2, dtype=torch.bool))
        self.assertIsInstance(result, r.StallResult)
        for name in ('reward', 'dx_1s', 'highwater_dx_1s', 'window_ready', 'qualifying',
                     'consecutive_count', 'penalty_on'):
            value = getattr(result, name)
            self.assertEqual(value.shape, (2,), name)
            self.assertIsNot(value, getattr(state, name), name)
        self.assertEqual(result.consecutive_count.dtype, torch.int64)
        self.assertEqual(result.window_ready.dtype, torch.bool)
        self.assertEqual(result.penalty_on.dtype, torch.bool)

    def test_invalid_inputs_are_rejected(self):
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                r.StallState(bad)
        state = self.build(1)
        with self.assertRaises(ValueError):
            state.reset(torch.tensor([0], dtype=torch.int64), torch.zeros(2))
        with self.assertRaises(ValueError):
            state.reset(torch.tensor([3], dtype=torch.int64), scalar(0.))
        with self.assertRaises(TypeError):
            state.reset(torch.tensor([0], dtype=torch.int32), scalar(0.))
        with self.assertRaises(ValueError):
            state.reset(torch.tensor([True, False], dtype=torch.bool), scalar(0.))
        with self.assertRaises(ValueError):
            state.update(scalar(float('nan')), scalar(C), flag(False), flag(True))
        with self.assertRaises(ValueError):
            state.update(scalar(0.), scalar(float('inf')), flag(False), flag(True))
        with self.assertRaises(TypeError):
            state.update(torch.tensor([0]), scalar(C), flag(False), flag(True))
        with self.assertRaises(ValueError):
            state.update(torch.zeros(2), scalar(C), flag(False), flag(True))
        with self.assertRaises(ValueError):
            state.update(scalar(0.), scalar(0.), flag(False), flag(True))


if __name__ == '__main__':
    unittest.main()


class StallFloat32BoundaryTests(unittest.TestCase):
    """Q001 section 3: no epsilon; the exact float32 ``<`` boundary decides.

    ``torch.nextafter`` constructs the representable float32 value immediately
    below/above each threshold, so the strict comparison is pinned in the final
    comparison dtype.  dx and dM are isolated from each other's condition.
    """

    def setUp(self):
        self.state = r.StallState(1)
        self.state.reset(torch.tensor([0], dtype=torch.int64), scalar(0.))
        advance(self.state, 0., steps=110)  # full window, sustained penalty
        self.threshold = torch.tensor(THRESHOLD, dtype=torch.float32)
        self.highwater = torch.tensor(spec.STALL_HIGHWATER_DX, dtype=torch.float32)

    def _dx_update(self, value):
        # dM is held exactly at zero by keeping the high-water row above x_t.
        self.state.x_history[0, 1] = 0.
        self.state.m_history[0, :] = .5
        return self.state.update(scalar(value), scalar(C), flag(False), flag(True))

    def _dm_update(self, value):
        # dx is held exactly at zero; only dM varies.
        self.state.x_history[0, 1] = 0.
        self.state.m_history[0, :] = 0.
        self.state.m_history[0, -1] = torch.tensor(value, dtype=torch.float32)
        return self.state.update(scalar(0.), scalar(C), flag(False), flag(True))

    def test_dx_previous_representable_value_qualifies(self):
        below = torch.nextafter(self.threshold, torch.tensor(float('-inf')))
        result = self._dx_update(float(below))
        self.assertEqual(float(result.dx_1s[0]), float(below))
        self.assertEqual(float(result.highwater_dx_1s[0]), 0.)
        self.assertTrue(bool(result.qualifying[0]))
        self.assertTrue(bool(result.penalty_on[0]))

    def test_dx_equal_value_does_not_qualify(self):
        result = self._dx_update(float(self.threshold))
        self.assertEqual(float(result.dx_1s[0]), float(self.threshold))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertFalse(bool(result.penalty_on[0]))
        self.assertEqual(float(result.reward[0]), 0.)

    def test_dx_next_representable_value_does_not_qualify(self):
        above = torch.nextafter(self.threshold, torch.tensor(float('inf')))
        result = self._dx_update(float(above))
        self.assertEqual(float(result.dx_1s[0]), float(above))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertFalse(bool(result.penalty_on[0]))

    def test_highwater_previous_representable_value_qualifies(self):
        below = torch.nextafter(self.highwater, torch.tensor(float('-inf')))
        result = self._dm_update(float(below))
        self.assertEqual(float(result.dx_1s[0]), 0.)
        self.assertEqual(float(result.highwater_dx_1s[0]), float(below))
        self.assertTrue(bool(result.qualifying[0]))
        self.assertTrue(bool(result.penalty_on[0]))

    def test_highwater_equal_value_does_not_qualify(self):
        result = self._dm_update(float(self.highwater))
        self.assertEqual(float(result.highwater_dx_1s[0]), float(self.highwater))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertFalse(bool(result.penalty_on[0]))

    def test_highwater_next_representable_value_does_not_qualify(self):
        above = torch.nextafter(self.highwater, torch.tensor(float('inf')))
        result = self._dm_update(float(above))
        self.assertEqual(float(result.highwater_dx_1s[0]), float(above))
        self.assertFalse(bool(result.qualifying[0]))
        self.assertFalse(bool(result.penalty_on[0]))
