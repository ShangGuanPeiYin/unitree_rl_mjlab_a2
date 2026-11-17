"""Pure v0.6 reward terms, air-time state and reset-boundary behaviour.

Every input is a synthetic tensor; no simulator, policy, learning, W&B or GPU is
involved.  The numeric anchors are the SDS 3.4 air-time values
(``+0.0008`` / ``-0.0002`` / ``0``), the SDS 3.5 ``+0.4 per metre`` forward
anchor, the SDS 3.2/3.3 deadband and clearance targets and SDS 9.1 items 2-10.
"""
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
try:  # SDS 7.3: entry points bootstrap first; tolerate an already-imported package.
    import a2_v06_bootstrap

    a2_v06_bootstrap.activate(ROOT)
except (ImportError, RuntimeError):
    pass

from src.tasks.velocity.v06 import rewards, spec  # noqa: E402

DT = spec.CONTROL_DT


def vec(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def flag(*values):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])
    return torch.tensor(values, dtype=torch.bool)


def default_q(batch=1):
    q = torch.tensor(spec.A2_DEFAULT_JOINT_POS, dtype=torch.float32)
    return q.repeat(batch, 1)


def make_frame(batch=1, **overrides):
    zeros = torch.zeros
    q0 = default_q(batch)
    values = dict(
        executed_command=torch.full((batch,), .4),
        local_x=zeros(batch),
        local_x_prev=zeros(batch),
        body_lin_vel_b=zeros(batch, 3),
        body_ang_vel_b=zeros(batch, 3),
        gravity_b=torch.tensor([[0., 0., -1.]]).repeat(batch, 1),
        base_height_above_surface=torch.full((batch,), spec.BASE_HEIGHT_TARGET),
        q=q0,
        qdot=zeros(batch, 12),
        q_target_executed=q0.clone(),
        action=zeros(batch, 12),
        previous_action=zeros(batch, 12),
        foot_position_world=zeros(batch, 4, 3),
        foot_velocity_world=zeros(batch, 4, 3),
        foot_force_world=zeros(batch, 4, 3),
        contact_raw=torch.zeros(batch, 4, dtype=torch.bool),
        contact_filtered=torch.zeros(batch, 4, dtype=torch.bool),
        tile_ids=torch.zeros(batch, dtype=torch.long),
        local_y=zeros(batch),
        yaw_world=zeros(batch),
        resets=torch.zeros(batch, dtype=torch.bool),
        success=torch.zeros(batch, dtype=torch.bool),
        failure=torch.zeros(batch, dtype=torch.bool),
        foot_surface_height=zeros(batch, 4),
    )
    values.update(overrides)
    return rewards.RewardFrame(**values)


def foot_frame(tile_id, surface, clearance, velocity=0., contact=False, batch=1):
    """Feet at a declared clearance over one surface (all four feet identical)."""
    surface = torch.full((batch, 4), float(surface))
    z = surface + spec.FOOT_RADIUS + float(clearance)
    position = torch.zeros(batch, 4, 3)
    position[..., 2] = z
    velocity_tensor = torch.zeros(batch, 4, 3)
    velocity_tensor[..., 0] = float(velocity)
    contacts = torch.full((batch, 4), bool(contact), dtype=torch.bool)
    return make_frame(batch=batch, tile_ids=torch.full((batch,), tile_id, dtype=torch.long),
                      foot_position_world=position, foot_velocity_world=velocity_tensor,
                      foot_surface_height=surface, contact_raw=contacts,
                      contact_filtered=contacts.clone())


class ContinuousRateTests(unittest.TestCase):
    def test_tracking_lin_vel(self):
        frame = make_frame(executed_command=vec(.3))
        self.assertAlmostEqual(float(rewards.tracking_lin_vel_rate(
            frame.body_lin_vel_b, frame.executed_command)[0]),
            math.exp(-.09 / .25), places=6)
        matched = vec(.4, 0., 0.).reshape(1, 3)
        self.assertAlmostEqual(float(rewards.tracking_lin_vel_rate(
            matched, vec(.4))[0]), 1., places=6)
        side = vec(.4, .2, 0.).reshape(1, 3)
        self.assertAlmostEqual(float(rewards.tracking_lin_vel_rate(
            side, vec(.4))[0]), math.exp(-.04 / .25), places=6)
        with self.assertRaises(ValueError):
            rewards.tracking_lin_vel_rate(torch.zeros(3), vec(.4))

    def test_tracking_ang_vel(self):
        self.assertEqual(float(rewards.tracking_ang_vel_rate(torch.zeros(1, 3))[0]), 1.)
        self.assertAlmostEqual(float(rewards.tracking_ang_vel_rate(
            vec(0., 0., .5).reshape(1, 3))[0]), math.exp(-.25 / .25), places=6)

    def test_lin_vel_z_deadband(self):
        rates = rewards.lin_vel_z_rate(vec(0., 0., .15).reshape(1, 3))
        self.assertEqual(float(rates[0]), 0.)
        high = rewards.lin_vel_z_rate(vec(0., 0., .25).reshape(1, 3))
        self.assertAlmostEqual(float(high[0]), .01, places=7)
        # weighted rate = weight * deadband rate = -0.01 per second at v_z=0.25
        weighted_rate = spec.REWARD_WEIGHTS['lin_vel_z'] * float(high[0])
        self.assertAlmostEqual(weighted_rate, -.01, places=7)
        low = rewards.lin_vel_z_rate(vec(0., 0., -.25).reshape(1, 3))
        self.assertAlmostEqual(float(low[0]), .01, places=7)

    def test_orientation_roll_only(self):
        pitch = rewards.orientation_roll_rate(
            vec(math.sin(.4), 0., -math.cos(.4)).reshape(1, 3))
        self.assertEqual(float(pitch[0]), 0.)
        roll = rewards.orientation_roll_rate(
            vec(0., math.sin(.3), -math.cos(.3)).reshape(1, 3))
        self.assertAlmostEqual(float(roll[0]), math.sin(.3) ** 2, places=7)

    def test_base_height(self):
        self.assertEqual(float(rewards.base_height_rate(vec(.40))[0]), 0.)
        self.assertAlmostEqual(float(rewards.base_height_rate(vec(.50))[0]), .01, places=7)

    def test_action_rate(self):
        action = vec(1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0.).reshape(1, 12)
        previous = vec(0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.).reshape(1, 12)
        self.assertAlmostEqual(float(rewards.action_rate_rate(action, previous)[0]),
                               1. + 1. + 4., places=6)

    def test_similar_to_default(self):
        q = default_q()
        self.assertAlmostEqual(float(rewards.similar_to_default_rate(q)[0]), 0., places=5)
        q[0, 0] += .1
        self.assertAlmostEqual(float(rewards.similar_to_default_rate(q)[0]), .1, places=6)
        q[0, 0] -= .5
        self.assertAlmostEqual(float(rewards.similar_to_default_rate(q)[0]), .4, places=6)

    def test_dof_acc_and_dof_vel(self):
        qdot = vec(.3).repeat(1, 12)
        previous = torch.zeros(1, 12)
        self.assertAlmostEqual(float(rewards.dof_acc_rate(qdot, previous)[0]),
                               12. * (.3 / DT) ** 2, places=2)
        self.assertAlmostEqual(float(rewards.dof_vel_rate(qdot)[0]),
                               12. * .09, places=5)
        with self.assertRaises(ValueError):
            rewards.dof_acc_rate(qdot, previous, dt=0.)

    def test_ang_vel_xy(self):
        self.assertAlmostEqual(float(rewards.ang_vel_xy_rate(
            vec(.1, .2, 9.).reshape(1, 3))[0]), .05, places=6)

    def test_foot_slip(self):
        velocity = torch.zeros(1, 4, 3)
        velocity[0, 0, 0] = .3
        velocity[0, 1, 1] = .4
        velocity[0, 2, 2] = 99.
        contact = flag([True, True, False, False]).reshape(1, 4)
        self.assertAlmostEqual(float(rewards.foot_slip_rate(velocity, contact)[0]),
                               .09 + .16, places=6)

    def test_joint_tracking_uses_executed_target(self):
        q = default_q()
        target = q.clone()
        target[0, 0] += .1
        action = q.clone()
        action[0, 1] += 9.
        self.assertAlmostEqual(float(rewards.joint_tracking_rate(target, q)[0]),
                               .01, places=6)
        self.assertNotEqual(float(rewards.joint_tracking_rate(target, q)[0]),
                            float(rewards.joint_tracking_rate(action, q)[0]))

    def test_center_and_heading(self):
        self.assertAlmostEqual(float(rewards.center_rate(vec(.2))[0]), 1.)
        self.assertEqual(float(rewards.center_rate(vec(.4))[0]), 4.)
        self.assertAlmostEqual(float(rewards.center_rate(vec(.1))[0]), .25, places=6)
        self.assertAlmostEqual(float(rewards.heading_rate(vec(math.pi / 2))[0]), 1.,
                               places=6)
        self.assertAlmostEqual(float(rewards.heading_rate(vec(math.pi))[0]), 2.,
                               places=6)

    def test_joint_limits(self):
        q = default_q()
        self.assertEqual(float(rewards.joint_limits_rate(q)[0]), 0.)
        q[0, 0] = 1.5
        lower = spec.A2_SOFT_JOINT_LIMITS[0][0]
        upper = spec.A2_SOFT_JOINT_LIMITS[0][1]
        self.assertAlmostEqual(float(rewards.joint_limits_rate(q)[0]), 1.5 - upper,
                               places=6)
        q[0, 0] = -2.
        self.assertAlmostEqual(float(rewards.joint_limits_rate(q)[0]), lower + 2.,
                               places=6)


class ClearanceTests(unittest.TestCase):
    def test_target_levels(self):
        self.assertAlmostEqual(float(
            rewards.foot_clearance_rate(
                *self._foot(tile_id=21, surface=.2, clearance=.22, velocity=.5,
                            contact=False))[0]), 0., places=6)
        # Same 0.22 m clearance over the flat tile is far above the 0.075 target.
        flat = rewards.foot_clearance_rate(
            *self._foot(tile_id=0, surface=0., clearance=.22, velocity=.5,
                        contact=False))
        self.assertAlmostEqual(float(flat[0]), 4. * (.22 - .075) ** 2 * .5, places=5)

    def test_swing_foot_is_ignored_and_contact_is_zero(self):
        value = rewards.foot_clearance_rate(
            *self._foot(tile_id=21, surface=.2, clearance=.22, velocity=.5,
                        contact=True))
        self.assertEqual(float(value[0]), 0.)

    def test_edge_jump_uses_current_surface(self):
        """Crossing a 0.20 m step edge changes H under the same foot instantly."""
        surface_before = torch.full((1, 4), .2)
        surface_after = torch.full((1, 4), .4)
        position = torch.zeros(1, 4, 3)
        position[..., 2] = .2 + spec.FOOT_RADIUS + .22
        velocity = torch.zeros(1, 4, 3)
        velocity[..., 0] = .4
        contact = torch.zeros(1, 4, dtype=torch.bool)
        low = rewards.foot_clearance_rate(position, velocity, surface_before,
                                          contact, torch.tensor([21]))
        high = rewards.foot_clearance_rate(position, velocity, surface_after,
                                           contact, torch.tensor([21]))
        self.assertAlmostEqual(float(low[0]), 0., places=6)
        self.assertAlmostEqual(float(high[0]),
                               4. * (.02 - .22) ** 2 * .4, places=5)
        # The jump is not smoothed: half-way interpolation would give 0.
        self.assertGreater(abs(float(high[0]) - float(low[0])), 1e-3)

    def test_flat_clearance_target(self):
        value = rewards.foot_clearance_rate(
            *self._foot(tile_id=0, surface=0., clearance=.075, velocity=.3,
                        contact=False))
        self.assertAlmostEqual(float(value[0]), 0., places=6)

    @staticmethod
    def _foot(tile_id, surface, clearance, velocity, contact):
        position = torch.zeros(1, 4, 3)
        position[..., 2] = surface + spec.FOOT_RADIUS + clearance
        velocity_tensor = torch.zeros(1, 4, 3)
        velocity_tensor[..., 0] = velocity
        return (position, velocity_tensor, torch.full((1, 4), float(surface)),
                torch.full((1, 4), contact, dtype=torch.bool),
                torch.tensor([tile_id], dtype=torch.long))


def contact_row(*feet):
    return torch.tensor([feet], dtype=torch.bool)


def air_weighted(bonus):
    return float(spec.REWARD_WEIGHTS['feet_air_time'] * DT * bonus)


class AirTimeTests(unittest.TestCase):
    """Q001-1 state-machine acceptance: every value comes from ``AirTimeState``."""

    def test_q001_true_15_false_true_pays_anchor_and_never_repeats(self):
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        self.assertEqual(float(state.update(stand, DT)[0]), 0.)
        for _ in range(15):
            self.assertEqual(float(state.update(air, DT)[0]), 0.)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .30, places=6)
        landing = float(state.update(stand, DT)[0])
        self.assertAlmostEqual(air_weighted(landing), .0008, places=7)
        self.assertEqual(float(state.air_time.max()), 0.)
        for _ in range(10):
            self.assertEqual(float(state.update(stand, DT)[0]), 0.)
            self.assertEqual(float(state.air_time.max()), 0.)

    def test_q001_true_false_true_is_zero_and_clears(self):
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        self.assertEqual(float(state.update(stand, DT)[0]), 0.)
        self.assertEqual(float(state.update(air, DT)[0]), 0.)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .02, places=7)
        recovered = float(state.update(stand, DT)[0])
        self.assertEqual(recovered, 0.)
        self.assertEqual(float(state.air_time.max()), 0.)
        self.assertEqual(float(state.previous_raw_contact.float().sum()), 4.)

    def test_q001_repeated_single_frame_dropouts_do_not_accumulate(self):
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        state.update(stand, DT)
        for _ in range(5):
            self.assertEqual(float(state.update(air, DT)[0]), 0.)
            self.assertEqual(float(state.update(stand, DT)[0]), 0.)
            self.assertEqual(float(state.air_time.max()), 0.)
        for _ in range(15):
            state.update(air, DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .30, places=6)
        self.assertAlmostEqual(air_weighted(float(state.update(stand, DT)[0])),
                               .0008, places=7)

    def test_q001_birth_air_is_zero_until_first_contact_then_pays(self):
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        for _ in range(15):
            self.assertEqual(float(state.update(air, DT)[0]), 0.)
            self.assertEqual(float(state.air_time.max()), 0.)
        self.assertFalse(bool(state.has_contacted_once[0, 0]))
        # The first ground contact only initialises; it never pays.
        self.assertEqual(float(state.update(stand, DT)[0]), 0.)
        self.assertTrue(bool(state.has_contacted_once[0, 0]))
        self.assertEqual(float(state.air_time.max()), 0.)
        for _ in range(15):
            state.update(air, DT)
        self.assertAlmostEqual(air_weighted(float(state.update(stand, DT)[0])),
                               .0008, places=7)

    def test_q001_three_false_frames_exact_negative(self):
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        state.update(stand, DT)
        for _ in range(3):
            state.update(air, DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .06, places=7)
        landing = float(state.update(stand, DT)[0])
        self.assertAlmostEqual(landing, -.04, places=7)
        self.assertAlmostEqual(air_weighted(landing), -.00016, places=8)
        self.assertEqual(float(state.air_time.max()), 0.)

    def test_q001_pure_0p05_anchor_is_not_a_2p5_frame_claim(self):
        # Pure-function anchor only: dt=.02 has no half control frame, so this
        # is never presented as a 2.5-frame state-machine sequence.
        first = contact_row(True, False, False, False)
        has = contact_row(True, False, False, False)
        air = torch.tensor([[.05, 0., 0., 0.]])
        bonus = float(rewards.feet_air_time_bonus(first, air, has)[0])
        self.assertAlmostEqual(bonus, -.05, places=7)
        self.assertAlmostEqual(air_weighted(bonus), -.0002, places=7)

    def test_q001_four_feet_asynchronous_and_first_contact(self):
        state = rewards.AirTimeState(1)
        air = contact_row(False, True, True, True)
        state.update(contact_row(True, True, True, False), DT)  # foot3 never touched
        for _ in range(3):
            state.update(contact_row(False, True, False, False), DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .06, places=7)
        self.assertAlmostEqual(float(state.air_time[0, 2]), .06, places=7)
        self.assertEqual(float(state.air_time[0, 3]), 0.)
        # foot2 lands at 0.06 s -> -0.00016; foot0 stays airborne.
        bonus = float(state.update(contact_row(False, True, True, False), DT)[0])
        self.assertAlmostEqual(air_weighted(bonus), -.00016, places=8)
        for _ in range(11):
            state.update(contact_row(False, True, True, False), DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .30, places=6)
        # foot0 lands (0.30 s -> +0.0008); foot2 supported; foot3 still untouched.
        bonus = float(state.update(contact_row(True, True, True, False), DT)[0])
        self.assertAlmostEqual(air_weighted(bonus), .0008, places=7)
        self.assertEqual(float(state.air_time[0, 2]), 0.)
        # First-ever contact of foot3 initialises only.
        self.assertEqual(float(state.update(contact_row(True, True, True, True), DT)[0]), 0.)
        self.assertEqual(float(state.air_time.max()), 0.)

    def test_q001_partial_reset_keeps_other_env(self):
        state = rewards.AirTimeState(2)
        stand = contact_row(True, True, True, True).repeat(2, 1)
        air = contact_row(False, True, True, True).repeat(2, 1)
        state.update(stand, DT)
        for _ in range(5):
            state.update(air, DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .10, places=7)
        self.assertAlmostEqual(float(state.air_time[1, 0]), .10, places=7)
        state.reset(torch.tensor([1]))
        self.assertAlmostEqual(float(state.air_time[0, 0]), .10, places=7)
        self.assertEqual(float(state.air_time[1].max()), 0.)
        self.assertFalse(bool(state.has_contacted_once[1].any()))
        for _ in range(10):
            state.update(air, DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .30, places=6)
        self.assertEqual(float(state.air_time[1].max()), 0.)
        bonus = state.update(stand, DT)
        self.assertAlmostEqual(air_weighted(float(bonus[0])), .0008, places=7)
        self.assertEqual(float(bonus[1]), 0.)
        self.assertEqual(float(state.air_time[1].max()), 0.)

    def test_q001_two_frame_dropout_pays_the_short_air(self):
        # Per Q001-1 a two-frame dropout creates a real filtered falling edge;
        # the next raw recovery is a landing and pays the short negative air.
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        state.update(stand, DT)
        state.update(air, DT)
        state.update(air, DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .04, places=7)
        bonus = float(state.update(stand, DT)[0])
        self.assertAlmostEqual(bonus, -.06, places=7)
        self.assertAlmostEqual(air_weighted(bonus), -.00024, places=8)
        self.assertEqual(float(state.air_time.max()), 0.)

    def test_q001_force_threshold_is_strictly_greater_than_5n(self):
        state = rewards.AirTimeState(1)
        found = torch.tensor([[True, False, False, False]])
        exactly = torch.full((1, 4), float(spec.SUPPORT_FORCE_GT))
        above = torch.full((1, 4), float(spec.SUPPORT_FORCE_GT) + 1e-4)
        zero = torch.zeros(1, 4)

        def raw(force):
            return found & (force > spec.SUPPORT_FORCE_GT)

        for _ in range(4):
            self.assertEqual(float(state.update(raw(exactly), DT)[0]), 0.)
        self.assertFalse(bool(state.has_contacted_once[0, 0]))
        # Strictly greater than 5 N is the first ground contact.
        self.assertEqual(float(state.update(raw(above), DT)[0]), 0.)
        self.assertTrue(bool(state.has_contacted_once[0, 0]))
        for _ in range(15):
            state.update(raw(zero), DT)
        self.assertAlmostEqual(float(state.air_time[0, 0]), .30, places=6)
        self.assertAlmostEqual(air_weighted(float(state.update(raw(above), DT)[0])),
                               .0008, places=7)

    def test_q001_state_dict_mid_sequence_restore_is_frame_identical(self):
        state = rewards.AirTimeState(1)
        stand = contact_row(True, True, True, True)
        air = contact_row(False, True, True, True)
        mixed = contact_row(True, True, True, True)
        for frame in (stand, mixed, air, air, mixed, stand, air):
            state.update(frame, DT)
        saved = state.state_dict()
        tail = [air, air, stand, mixed, air, air, air, stand, stand, air, mixed, stand]
        original = []
        for frame in tail:
            bonus = state.update(frame, DT)
            original.append((float(bonus[0]), state.air_time.clone(),
                             state.previous_raw_contact.clone(),
                             state.previous_filtered_contact.clone(),
                             state.has_contacted_once.clone()))
        restored = rewards.AirTimeState(1)
        restored.load_state_dict(saved)
        for index, frame in enumerate(tail):
            bonus = restored.update(frame, DT)
            expected = original[index]
            self.assertAlmostEqual(float(bonus[0]), expected[0], places=8)
            for name, value in (('air_time', expected[1]),
                                ('previous_raw_contact', expected[2]),
                                ('previous_filtered_contact', expected[3]),
                                ('has_contacted_once', expected[4])):
                self.assertTrue(torch.equal(getattr(restored, name), value),
                                f'frame {index} {name}')

    def test_state_dict_rejects_missing_or_bad_shape_and_dtype(self):
        state = rewards.AirTimeState(2)
        state.update(contact_row(True, True, True, True).repeat(2, 1), DT)
        saved = state.state_dict()
        for name in ('previous_raw_contact', 'previous_filtered_contact',
                     'has_contacted_once', 'air_time'):
            broken = dict(saved)
            broken.pop(name)
            with self.assertRaises(ValueError):
                rewards.AirTimeState(2).load_state_dict(broken)
        broken = dict(saved)
        broken['air_time'] = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            rewards.AirTimeState(2).load_state_dict(broken)
        broken = dict(saved)
        broken['previous_raw_contact'] = torch.zeros(2, 4, dtype=torch.float32)
        with self.assertRaises(TypeError):
            rewards.AirTimeState(2).load_state_dict(broken)
        broken = dict(saved)
        broken['previous_filtered_contact'] = torch.zeros(2, 5, dtype=torch.bool)
        with self.assertRaises(ValueError):
            rewards.AirTimeState(2).load_state_dict(broken)


class ForwardAndEventTests(unittest.TestCase):
    def test_forward_per_metre(self):
        reward = rewards.forward_progress_reward(vec(.01), vec(0.), resets=flag([False]))
        self.assertAlmostEqual(float(reward[0]), .004, places=7)
        reward = rewards.forward_progress_reward(vec(1.), vec(0.), resets=flag([False]))
        self.assertAlmostEqual(float(reward[0]), .4, places=7)
        backward = rewards.forward_progress_reward(vec(0.), vec(1.), resets=flag([False]))
        self.assertAlmostEqual(float(backward[0]), -.4, places=7)

    def test_forward_reset_is_zero(self):
        reward = rewards.forward_progress_reward(vec(50.), vec(-1.), resets=flag([True]))
        self.assertEqual(float(reward[0]), 0.)

    def test_event_failure_precedence(self):
        paid = rewards.event_rewards(flag([True, True, False, False]),
                                     flag([True, False, True, False]))
        self.assertEqual(paid.tolist(), [-10., 10., -10., 0.])


class RewardComputerTests(unittest.TestCase):
    def test_continuous_terms_multiply_dt_once(self):
        computer = rewards.RewardComputer()
        frame = make_frame(executed_command=vec(.4))
        frame.body_ang_vel_b = vec(0., 0., .5).reshape(1, 3)
        frame.body_lin_vel_b = vec(.4, 0., .25).reshape(1, 3)
        result = computer(frame)
        for name in spec.REWARD_TERMS:
            weighted = result.weighted_terms[name]
            expected = DT * spec.REWARD_WEIGHTS[name] * result.raw_terms[name]
            self.assertTrue(torch.allclose(weighted, expected, atol=1e-7), name)
        # And the raw term agrees with the independently exposed function.
        self.assertAlmostEqual(
            float(result.raw_terms['lin_vel_z'][0]),
            float(rewards.lin_vel_z_rate(frame.body_lin_vel_b)[0]), places=7)
        self.assertAlmostEqual(
            float(result.raw_terms['tracking_ang_vel'][0]),
            float(rewards.tracking_ang_vel_rate(frame.body_ang_vel_b)[0]), places=7)

    def test_total_formula_and_sum(self):
        computer = rewards.RewardComputer()
        frame = make_frame(executed_command=vec(.5), base_height_above_surface=vec(.45))
        frame.local_x = vec(.010)
        frame.local_x_prev = vec(0.)
        frame.body_lin_vel_b = vec(.4, .1, .1).reshape(1, 3)
        frame.body_ang_vel_b = vec(.1, .2, .3).reshape(1, 3)
        frame.gravity_b = vec(0., .2, -math.sqrt(1. - .04)).reshape(1, 3)
        frame.local_y = vec(.1)
        frame.yaw_world = vec(.2)
        frame.q[0, 0] += .02
        frame.qdot[0, :] = .1
        frame.q_target_executed[0, 1] += .05
        frame.action[0, 2] = .3
        frame.previous_action[0, 2] = .1
        frame.contact_filtered[0, :2] = True
        frame.foot_velocity_world[0, 0, 0] = .2
        frame.foot_position_world[0, 0, 2] = .05
        frame.foot_surface_height[0, 0] = 0.
        frame.success[0] = True
        result = computer(frame)
        expected = (DT * sum(spec.REWARD_WEIGHTS[name] * float(result.raw_terms[name][0])
                             for name in spec.REWARD_TERMS)
                    + .4 * float(result.raw_terms['forward_progress'][0])
                    + spec.SUCCESS_REWARD + 0.)
        self.assertAlmostEqual(float(result.total[0]), expected, places=5)
        summed = sum(float(value[0]) for value in result.weighted_terms.values())
        self.assertAlmostEqual(float(result.total[0]), summed, places=5)
        self.assertLess(abs(float(result.total[0]) - summed), 1e-5)

    def test_reset_never_differences_across_episode(self):
        computer = rewards.RewardComputer()
        first = make_frame(resets=flag([True]), local_x=vec(-1.), local_x_prev=vec(3.))
        first.action[0, 0] = 1.
        first.qdot[0, 0] = 2.
        result = computer(first)
        self.assertEqual(float(result.raw_terms['action_rate'][0]), 0.)
        self.assertEqual(float(result.raw_terms['dof_acc'][0]), 0.)
        self.assertEqual(float(result.raw_terms['forward_progress'][0]), 0.)
        second = make_frame(resets=flag([False]), local_x=vec(-.99),
                            local_x_prev=vec(-1.))
        second.action[0, 0] = 1.2
        second.qdot[0, 0] = 2.5
        result = computer(second)
        self.assertAlmostEqual(float(result.raw_terms['forward_progress'][0]), .01,
                               places=6)
        self.assertAlmostEqual(float(result.raw_terms['action_rate'][0]), .04, places=6)
        self.assertGreater(float(result.raw_terms['dof_acc'][0]), 0.)
        # A reset teleport must not be paid as progress or as discontinuity.
        third = make_frame(resets=flag([True]), local_x=vec(10.),
                           local_x_prev=vec(-.99))
        third.action[0, 0] = -3.
        third.qdot[0, 0] = -9.
        result = computer(third)
        self.assertEqual(float(result.raw_terms['forward_progress'][0]), 0.)
        self.assertEqual(float(result.raw_terms['action_rate'][0]), 0.)
        self.assertEqual(float(result.raw_terms['dof_acc'][0]), 0.)

    def test_partial_reset_only_resets_ids(self):
        computer = rewards.RewardComputer()
        computer(make_frame(batch=2, resets=flag([True, True]),
                            local_x=vec(0., 0.), local_x_prev=vec(1., 1.)))
        step = make_frame(batch=2, resets=flag([False, True]),
                          local_x=vec(.01, 10.), local_x_prev=vec(0., 0.))
        step.action[:, 0] = 1.
        result = computer(step)
        self.assertAlmostEqual(float(result.weighted_terms['forward_progress'][0]), .004,
                               places=6)
        self.assertEqual(float(result.raw_terms['forward_progress'][1]), 0.)
        self.assertGreater(float(result.raw_terms['action_rate'][0]), 0.)
        self.assertEqual(float(result.raw_terms['action_rate'][1]), 0.)

    def test_first_call_uses_frame_previous_snapshot(self):
        computer = rewards.RewardComputer()
        frame = make_frame(local_x=vec(.01), local_x_prev=vec(0.),
                           resets=flag([False]))
        result = computer(frame)
        self.assertAlmostEqual(float(result.weighted_terms['forward_progress'][0]), .004,
                               places=6)

    def test_reset_all_and_pending_reset(self):
        computer = rewards.RewardComputer()
        computer.reset()
        frame = make_frame(batch=2, resets=flag([False, False]),
                           local_x=vec(5., 5.), local_x_prev=vec(0., 0.))
        result = computer(frame)
        self.assertEqual(float(result.total.sum()), float(result.total.sum()))
        self.assertEqual(computer.state.num_envs, 2)
        self.assertTrue(bool(computer.state.initialized[0]))
        computer.reset(torch.tensor([1]))
        self.assertFalse(bool(computer.state.initialized[1]))
        self.assertTrue(bool(computer.state.initialized[0]))

    def test_state_roundtrip(self):
        state = rewards.RewardState(2)
        state.previous_action[:] = .5
        state.previous_qdot[:] = .25
        state.previous_local_x[:] = -1.
        state.initialized[:] = True
        stand = flag([True, True, True, True]).repeat(2, 1)
        state.air_time.update(stand, DT)
        saved = state.state_dict()
        restored = rewards.RewardState(2)
        restored.load_state_dict(saved)
        for name in ('previous_action', 'previous_qdot', 'previous_local_x'):
            self.assertTrue(torch.equal(getattr(state, name), getattr(restored, name)))
        self.assertTrue(torch.equal(state.initialized, restored.initialized))
        self.assertTrue(torch.equal(state.air_time.air_time, restored.air_time.air_time))

    def test_frame_validation(self):
        with self.assertRaises(TypeError):
            rewards.RewardComputer()('not a frame')
        frame = make_frame()
        frame.tile_ids = torch.tensor([99], dtype=torch.long)
        with self.assertRaises(ValueError):
            rewards.RewardComputer()(frame)
        frame = make_frame()
        frame.contact_raw = torch.zeros(1, 4)
        with self.assertRaises(TypeError):
            rewards.RewardComputer()(frame)

    def test_unknown_weight_rejected(self):
        with self.assertRaises(ValueError):
            rewards.RewardComputer(weights={'nope': 1.})
        computer = rewards.RewardComputer(weights={'lin_vel_z': -2.})
        frame = make_frame()
        result = computer(frame)
        self.assertEqual(float(result.weighted_terms['lin_vel_z'][0]), 0.)
        self.assertNotEqual(spec.REWARD_WEIGHTS['lin_vel_z'], computer.weights['lin_vel_z'])


if __name__ == '__main__':
    unittest.main()
