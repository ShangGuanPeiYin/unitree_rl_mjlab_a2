"""Numeric parity against the frozen C1 calculation paths, without simulation.

These tests protect unchanged behavior: reward weighting/state/reset/payment,
GAE timeout bootstrap, clipping/KL/Adam/std and gradients in a complete update.
No v07 production code is introduced or patched by this test-only work.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import ppo as native_ppo, rewards as native_rewards, model as native_model
from src.tasks.velocity.v06 import ppo as frozen_ppo, rewards as frozen_rewards, model as frozen_model


def _same(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def test_reward_all_terms_state_and_failure_priority_match_frozen_c1():
    generator = torch.Generator().manual_seed(20261117)
    old, new = frozen_rewards.RewardComputer(), native_rewards.RewardComputer()
    observed_nonzero = set()
    batch = 4
    for step in range(18):
        zeros = torch.zeros
        q0 = torch.zeros(batch, 12)
        frame = frozen_rewards.RewardFrame(
            executed_command=zeros(batch), local_x=zeros(batch), local_x_prev=zeros(batch),
            body_lin_vel_b=zeros(batch, 3), body_ang_vel_b=zeros(batch, 3),
            gravity_b=torch.tensor([[0., 0., -1.]]).repeat(batch, 1),
            base_height_above_surface=torch.full((batch,), .35), q=q0,
            qdot=zeros(batch, 12), q_target_executed=q0.clone(), action=zeros(batch, 12),
            previous_action=zeros(batch, 12), foot_position_world=zeros(batch, 4, 3),
            foot_velocity_world=zeros(batch, 4, 3), foot_force_world=zeros(batch, 4, 3),
            contact_raw=torch.zeros(batch, 4, dtype=torch.bool),
            contact_filtered=torch.zeros(batch, 4, dtype=torch.bool),
            tile_ids=torch.zeros(batch, dtype=torch.long), local_y=zeros(batch),
            yaw_world=zeros(batch), resets=torch.zeros(batch, dtype=torch.bool),
            success=torch.zeros(batch, dtype=torch.bool), failure=torch.zeros(batch, dtype=torch.bool),
            foot_surface_height=zeros(batch, 4))
        frame.executed_command = torch.tensor([.3, .4, .5, .6])
        frame.local_x = torch.arange(batch) * .3 + step * .008
        frame.local_x_prev = frame.local_x - .008
        frame.body_lin_vel_b = torch.randn(batch, 3, generator=generator) * .25
        frame.body_ang_vel_b = torch.randn(batch, 3, generator=generator) * .35
        frame.gravity_b = torch.tensor([[.12, .16, -.98]]).repeat(batch, 1)
        frame.base_height_above_surface = torch.tensor([.30, .35, .40, .45])
        frame.q = frame.q + torch.randn(batch, 12, generator=generator) * .06
        frame.q[3, 0] = 3.5
        frame.qdot = torch.randn(batch, 12, generator=generator) * .2
        frame.q_target_executed = frame.q_target_executed + .015
        frame.action = torch.randn(batch, 12, generator=generator) * .1
        frame.previous_action = torch.randn(batch, 12, generator=generator) * .1
        frame.foot_position_world = torch.randn(batch, 4, 3, generator=generator) * .04
        frame.foot_surface_height = torch.tensor([0., .2, .4, .6])[:, None].expand(-1, 4).clone()
        frame.foot_position_world[..., 2] = frame.foot_surface_height + .08
        frame.foot_velocity_world = torch.randn(batch, 4, 3, generator=generator) * .2
        frame.contact_raw[:] = step in (0, 12, 17)
        frame.contact_filtered = frame.contact_raw.clone()
        frame.tile_ids = torch.tensor([0, 17, 18, 21])
        frame.local_y = torch.tensor([.01, -.02, .03, -.04])
        frame.yaw_world = torch.tensor([.05, -.10, .15, -.20])
        if step == 8:
            frame.resets[1] = True
            frame.local_x[1] = 100.
        if step == 17:
            frame.success[:] = torch.tensor([True, True, False, False])
            frame.failure[:] = torch.tensor([False, True, True, False])
        reference = old(deepcopy(frame))
        actual = new(native_rewards.RewardFrame(**deepcopy(vars(frame))))
        _same(reference.raw_terms, actual.raw_terms)
        _same(reference.weighted_terms, actual.weighted_terms)
        _same(reference.total, actual.total)
        _same(old.state.state_dict(), new.state.state_dict())
        assert torch.isfinite(actual.total).all()
        observed_nonzero.update(k for k, v in actual.raw_terms.items() if torch.any(v != 0))
        if step == 8:
            assert actual.weighted_terms['forward_progress'][1] == 0
        if step == 17:
            assert actual.weighted_terms['success'][0] == 10
            assert actual.weighted_terms['success'][1] == 0
            assert actual.weighted_terms['failure'][1] == -10
    assert set(frozen_rewards.REWARD_TERMS) <= observed_nonzero


def test_gae_timeout_bootstrap_and_frozen_ppo_configuration_match_c1():
    assert asdict(native_ppo.frozen_config()) == asdict(frozen_ppo.frozen_config())
    args = dict(rewards=torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
                values=torch.tensor([[.2, .3], [.4, .5], [.6, .7]]),
                next_values=torch.tensor([[.4, .5], [.6, .7], [.8, .9]]),
                terminated=torch.tensor([[False, True], [False, False], [True, False]]),
                done=torch.tensor([[False, True], [True, False], [True, False]]),
                gamma=.99, gae_lambda=.95)
    expected = frozen_ppo.compute_gae(**deepcopy(args))
    actual = native_ppo.compute_gae(**deepcopy(args))
    _same(expected, actual)
    advantage, _returns = actual
    torch.testing.assert_close(advantage[1, 0], torch.tensor(3.194), rtol=0, atol=1e-6)
    torch.testing.assert_close(advantage[2, 0], torch.tensor(4.4), rtol=0, atol=1e-6)


def test_complete_ppo_update_parameters_adam_and_gradient_outputs_match_c1():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(381)
        old_model = frozen_model.V06ActorCritic()
        new_model = native_model.V07ActorCritic()
        new_model.load_state_dict(deepcopy(old_model.state_dict()))
        old_optimizer = frozen_ppo.build_optimizer(old_model)
        new_optimizer = native_ppo.build_optimizer(new_model)
        for optimizer in (old_optimizer, new_optimizer):
            optimizer.param_groups[0]['lr'] = 7.593750000000002e-05
        obs = frozen_model.synthetic_observations(batch=8, seed=42, device='cpu')
        with torch.no_grad():
            distribution = old_model.action_distribution(obs)
            values = old_model.evaluate(obs).squeeze(-1)
            actions = distribution.sample()
            old_means = distribution.mean + .15
            old_stds = distribution.stddev * .8
            old_logp = torch.distributions.Normal(old_means, old_stds).log_prob(actions).sum(-1)
        storage = dict(obs=obs, actions=actions, means=old_means, stds=old_stds, logp=old_logp,
                       advantage=torch.tensor([-2., -1., -.5, .5, 1., 2., .75, -.75]),
                       returns=values + torch.tensor([1., -1., .5, -.5, .3, -.3, .8, -.8]),
                       value=values + .5)
        new_model.load_state_dict(deepcopy(old_model.state_dict()))
        rng = torch.get_rng_state().clone()
        old_learning = dict(optimizer_steps=0, completed_ppo_updates=0)
        new_learning = deepcopy(old_learning)
        expected = frozen_ppo.update_math(model=old_model, optimizer=old_optimizer,
                                         storage=deepcopy(storage), device='cpu',
                                         learning_state=old_learning)
        torch.set_rng_state(rng)
        actual = native_ppo.update_math(model=new_model, optimizer=new_optimizer,
                                       storage=deepcopy(storage), device='cpu',
                                       learning_state=new_learning)
        _same(old_model.state_dict(), new_model.state_dict())
        _same(old_optimizer.state_dict(), new_optimizer.state_dict())
        _same(expected, actual)
        assert old_learning == new_learning
        assert new_learning['optimizer_steps'] == 20
        assert new_learning['completed_ppo_updates'] == 1
        assert new_learning['optimizer_initialized'] is True
        assert len(new_optimizer.state_dict()['state']) == 45
        assert all(int(state['step']) == 20 for state in new_optimizer.state_dict()['state'].values())
