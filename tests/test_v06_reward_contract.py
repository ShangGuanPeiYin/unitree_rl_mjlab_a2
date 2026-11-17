"""Frozen reward-composition contract (SDS 3.1/3.2/9.1 items 1-3, 17).

The mandatory invariants are:

* for every frame ``sum(weighted_terms.values()) == total`` within ``1e-5``;
* continuous terms carry ``dt`` exactly once and never twice;
* the forward term is ``0.4*(x_t-x_(t-1))`` with no extra ``dt``;
* terminal ``+10/-10`` are not scaled by ``dt`` and failure wins in one frame;
* the unweighted terms are the SDS 3.2 table and nothing else.
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


def default_q(batch=1, dtype=torch.float32):
    return torch.tensor(spec.A2_DEFAULT_JOINT_POS, dtype=dtype).repeat(batch, 1)


def make_frame(batch=1, dtype=torch.float32, **overrides):
    q0 = default_q(batch, dtype)
    values = dict(
        executed_command=torch.full((batch,), .4, dtype=dtype),
        local_x=torch.zeros(batch, dtype=dtype),
        local_x_prev=torch.zeros(batch, dtype=dtype),
        body_lin_vel_b=torch.zeros(batch, 3, dtype=dtype),
        body_ang_vel_b=torch.zeros(batch, 3, dtype=dtype),
        gravity_b=torch.tensor([[0., 0., -1.]], dtype=dtype).repeat(batch, 1),
        base_height_above_surface=torch.full((batch,), spec.BASE_HEIGHT_TARGET,
                                             dtype=dtype),
        q=q0,
        qdot=torch.zeros(batch, 12, dtype=dtype),
        q_target_executed=q0.clone(),
        action=torch.zeros(batch, 12, dtype=dtype),
        previous_action=torch.zeros(batch, 12, dtype=dtype),
        foot_position_world=torch.zeros(batch, 4, 3, dtype=dtype),
        foot_velocity_world=torch.zeros(batch, 4, 3, dtype=dtype),
        foot_force_world=torch.zeros(batch, 4, 3, dtype=dtype),
        contact_raw=torch.zeros(batch, 4, dtype=torch.bool),
        contact_filtered=torch.zeros(batch, 4, dtype=torch.bool),
        tile_ids=torch.zeros(batch, dtype=torch.long),
        local_y=torch.zeros(batch, dtype=dtype),
        yaw_world=torch.zeros(batch, dtype=dtype),
        resets=torch.zeros(batch, dtype=torch.bool),
        success=torch.zeros(batch, dtype=torch.bool),
        failure=torch.zeros(batch, dtype=torch.bool),
        foot_surface_height=torch.zeros(batch, 4, dtype=dtype),
    )
    values.update(overrides)
    return rewards.RewardFrame(**values)


def random_frame(batch=8, dtype=torch.float32, seed=0):
    """A bounded random frame with all physical values finite and in range."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.):
        return (torch.rand(*shape, generator=generator, dtype=dtype) * 2 - 1) * scale

    q0 = default_q(batch, dtype)
    surface = torch.randint(0, 4, (batch, 4), generator=generator).to(dtype) * .2
    resets = torch.rand(batch, generator=generator) < .2
    success = torch.rand(batch, generator=generator) < .1
    failure = torch.rand(batch, generator=generator) < .1
    return make_frame(
        batch=batch, dtype=dtype,
        executed_command=torch.full((batch,), .4, dtype=dtype) + rand(batch, scale=.3),
        local_x=rand(batch, scale=2.),
        local_x_prev=rand(batch, scale=2.),
        body_lin_vel_b=rand(batch, 3, scale=.8),
        body_ang_vel_b=rand(batch, 3, scale=.8),
        gravity_b=torch.stack((rand(batch, scale=.2), rand(batch, scale=.2),
                               -torch.ones(batch, dtype=dtype).clamp_min(.8)), dim=-1),
        base_height_above_surface=torch.full((batch,), .4, dtype=dtype)
        + rand(batch, scale=.1),
        q=q0 + rand(batch, 12, scale=.05),
        qdot=rand(batch, 12, scale=.5),
        q_target_executed=q0 + rand(batch, 12, scale=.05),
        action=rand(batch, 12, scale=.5),
        previous_action=rand(batch, 12, scale=.5),
        foot_position_world=torch.stack(
            (rand(batch, 4, scale=.2), rand(batch, 4, scale=.2),
             surface + spec.FOOT_RADIUS + rand(batch, 4, scale=.2).abs()), dim=-1),
        foot_velocity_world=rand(batch, 4, 3, scale=.6),
        contact_raw=torch.rand(batch, 4, generator=generator) < .6,
        contact_filtered=torch.rand(batch, 4, generator=generator) < .6,
        tile_ids=torch.randint(0, len(spec.TERRAIN_TABLE), (batch,),
                               generator=generator),
        local_y=rand(batch, scale=.3),
        yaw_world=rand(batch, scale=.5),
        resets=resets, success=success, failure=failure,
        foot_surface_height=surface,
    )


class TermSetContractTests(unittest.TestCase):
    def test_exactly_seventeen_continuous_terms_plus_three_amounts(self):
        computer = rewards.RewardComputer()
        result = computer(make_frame(resets=torch.tensor([True])))
        self.assertEqual(tuple(spec.REWARD_TERMS),
                         tuple(name for name in result.raw_terms
                               if name in spec.REWARD_TERMS))
        self.assertEqual(set(result.raw_terms),
                         set(spec.REWARD_TERMS) | {'forward_progress', 'success',
                                                   'failure'})
        self.assertEqual(set(result.weighted_terms), set(result.raw_terms))
        self.assertEqual(len(spec.REWARD_TERMS), 17)

    def test_weights_are_pre_dt(self):
        self.assertEqual(spec.REWARD_WEIGHTS,
                         {'tracking_lin_vel': 1.5, 'tracking_ang_vel': .8,
                          'lin_vel_z': -1.0, 'orientation_roll': -5.0,
                          'base_height': -.1, 'action_rate': -.01,
                          'similar_to_default': -.05, 'dof_acc': -2.5e-7,
                          'dof_vel': -5e-4, 'ang_vel_xy': -.05,
                          'feet_air_time': .2, 'foot_slip': -.15,
                          'foot_clearance': -.5, 'joint_tracking': -.1,
                          'center': -.5, 'heading': -.5, 'joint_limits': -10.0})


class SumEqualsTotalTests(unittest.TestCase):
    def test_random_batch_float32(self):
        frame = random_frame(batch=16, dtype=torch.float32, seed=1)
        result = rewards.RewardComputer()(frame)
        summed = torch.stack(list(result.weighted_terms.values())).sum(0)
        difference = (summed - result.total).abs().max()
        self.assertLessEqual(float(difference), 1e-5)
        self.assertEqual(result.total.shape, (16,))

    def test_random_batch_float64_is_tighter(self):
        frame = random_frame(batch=16, dtype=torch.float64, seed=2)
        result = rewards.RewardComputer()(frame)
        summed = torch.stack(list(result.weighted_terms.values())).sum(0)
        difference = (summed - result.total).abs().max()
        self.assertLessEqual(float(difference), 1e-8)

    def test_unclamped_total(self):
        frame = make_frame()
        frame.q[0, 0] = 2.5          # joint limit and default-pose penalties
        frame.qdot[0, :] = 5.
        frame.local_y = torch.tensor([3.])
        result = rewards.RewardComputer()(frame)
        self.assertLess(float(result.total[0]), -.3)
        # No positive clipping: total equals the raw algebraic sum.
        self.assertAlmostEqual(float(result.total[0]),
                               sum(float(value[0]) for value in
                                   result.weighted_terms.values()), places=5)


class DtContractTests(unittest.TestCase):
    def test_dt_applied_exactly_once(self):
        frame = make_frame(resets=torch.tensor([True]),
                           body_lin_vel_b=torch.tensor([[.5, .1, .25]]))
        frame.qdot[0, :] = .3
        frame.action[0, :] = .2
        frame.previous_action[0, :] = .1
        half = rewards.RewardComputer(dt=.01)(frame)
        full = rewards.RewardComputer(dt=.02)(frame)
        for name in spec.REWARD_TERMS:
            if name in ('action_rate', 'dof_acc', 'forward_progress'):
                continue
            self.assertAlmostEqual(float(half.weighted_terms[name][0]) * 2.,
                                   float(full.weighted_terms[name][0]), places=6,
                                   msg=name)
        # Forward and terminal amounts are independent of dt.
        self.assertAlmostEqual(float(half.weighted_terms['forward_progress'][0]),
                               float(full.weighted_terms['forward_progress'][0]))
        half_event = rewards.RewardComputer(dt=.01)(make_frame(
            resets=torch.tensor([True]), success=torch.tensor([True])))
        full_event = rewards.RewardComputer(dt=.02)(make_frame(
            resets=torch.tensor([True]), success=torch.tensor([True])))
        self.assertEqual(float(half_event.weighted_terms['success'][0]), 10.)
        self.assertEqual(float(full_event.weighted_terms['success'][0]), 10.)

    def test_forward_dt_is_not_reapplied(self):
        frame = make_frame(local_x=torch.tensor([.01]), local_x_prev=torch.tensor([0.]),
                           resets=torch.tensor([True]))
        # A reset row must be zero, so re-use the computer's initialization.
        frame = make_frame(local_x=torch.tensor([.01]), local_x_prev=torch.tensor([0.]))
        computer = rewards.RewardComputer()
        result = computer(frame)
        self.assertAlmostEqual(float(result.weighted_terms['forward_progress'][0]),
                               .004, places=7)
        self.assertAlmostEqual(float(result.raw_terms['forward_progress'][0]), .01,
                               places=7)

    def test_terminal_amounts_are_one_shot(self):
        plus = rewards.RewardComputer()(make_frame(resets=torch.tensor([True]),
                                                   success=torch.tensor([True])))
        minus = rewards.RewardComputer()(make_frame(resets=torch.tensor([True]),
                                                    failure=torch.tensor([True])))
        both = rewards.RewardComputer()(make_frame(resets=torch.tensor([True]),
                                                   success=torch.tensor([True]),
                                                   failure=torch.tensor([True])))
        self.assertEqual(float(plus.weighted_terms['success'][0]), 10.)
        self.assertEqual(float(plus.weighted_terms['failure'][0]), 0.)
        self.assertEqual(float(minus.weighted_terms['success'][0]), 0.)
        self.assertEqual(float(minus.weighted_terms['failure'][0]), -10.)
        self.assertEqual(float(both.weighted_terms['success'][0]), 0.)
        self.assertEqual(float(both.weighted_terms['failure'][0]), -10.)

    def test_timeout_adds_zero(self):
        frame = make_frame(resets=torch.tensor([True]))
        result = rewards.RewardComputer()(frame)
        self.assertEqual(float(result.weighted_terms['success'][0]), 0.)
        self.assertEqual(float(result.weighted_terms['failure'][0]), 0.)
        self.assertEqual(spec.TIMEOUT_REWARD, 0.)
        self.assertNotIn('timeout', result.weighted_terms)


class AirTimeContractTests(unittest.TestCase):
    def test_end_to_end_air_time_anchor(self):
        computer = rewards.RewardComputer()
        stand = torch.tensor([[True, True, True, True]])
        air_one = torch.tensor([[False, True, True, True]])
        frame = make_frame(batch=1)
        frame.contact_raw = stand.clone()
        computer(frame)
        frame.contact_raw = air_one.clone()
        for _ in range(15):
            frame = make_frame(batch=1, contact_raw=air_one.clone())
            result = computer(frame)
        self.assertAlmostEqual(float(computer.state.air_time.air_time[0, 0]), .30, places=6)
        landing = make_frame(batch=1, contact_raw=stand.clone())
        result = computer(landing)
        self.assertAlmostEqual(float(result.weighted_terms['feet_air_time'][0]),
                               .0008, places=7)
        self.assertAlmostEqual(float(result.raw_terms['feet_air_time'][0]), .2, places=6)

    def test_standing_never_pays(self):
        computer = rewards.RewardComputer()
        stand = torch.tensor([[True, True, True, True]])
        for _ in range(20):
            result = computer(make_frame(contact_raw=stand.clone()))
            self.assertEqual(float(result.weighted_terms['feet_air_time'][0]), 0.)


class ClearanceContractTests(unittest.TestCase):
    def test_clearance_error_is_negative_weighted(self):
        surface = torch.full((1, 4), .2)
        position = torch.zeros(1, 4, 3)
        position[..., 2] = surface + spec.FOOT_RADIUS + .22
        velocity = torch.zeros(1, 4, 3)
        velocity[..., 0] = .4
        frame = make_frame(tile_ids=torch.tensor([21], dtype=torch.long),
                           foot_position_world=position,
                           foot_velocity_world=velocity,
                           foot_surface_height=surface,
                           contact_filtered=torch.zeros(1, 4, dtype=torch.bool))
        result = rewards.RewardComputer()(frame)
        self.assertAlmostEqual(float(result.weighted_terms['foot_clearance'][0]), 0.,
                               places=7)
        # Lower the same foot to 0.12 m clearance: 4*(0.10)^2*0.4, weighted -0.5*dt.
        position[..., 2] = surface + spec.FOOT_RADIUS + .12
        frame.foot_position_world = position
        result = rewards.RewardComputer()(frame)
        raw = 4. * (.12 - .22) ** 2 * .4
        self.assertAlmostEqual(float(result.raw_terms['foot_clearance'][0]), raw,
                               places=5)
        self.assertAlmostEqual(float(result.weighted_terms['foot_clearance'][0]),
                               spec.REWARD_WEIGHTS['foot_clearance'] * DT * raw,
                               places=7)


if __name__ == '__main__':
    unittest.main()
