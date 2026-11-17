"""CPU-only v0.6 PPO contract (SDS 4): one Adam, frozen numbers, KL schedule,
std projection, pre-clip group gradients, advantage normalization and timeout
bootstrap."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

from dataclasses import replace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.distributions import Normal  # noqa: E402

from src.tasks.velocity.v06 import ppo as target  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402


class TinyModel(nn.Module):
    """A two-bypass toy network exposing the v0.6 grouped-parameter surface."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Module()
        self.actor.fusion = nn.Linear(2, 4)
        self.actor.route_fusion = nn.Linear(2, 4)
        self.actor.mlp = nn.Linear(4, 12)
        self.critic = nn.Linear(7, 1)
        self.std = nn.Parameter(torch.full((12,), 0.30))

    def parameter_groups(self):
        groups = {name: [] for name in target.GRADIENT_GROUPS}
        for name, _parameter in self.named_parameters():
            if name == 'std':
                groups['std'].append(name)
            elif name.startswith(('actor.fusion.', 'actor.route_fusion.')):
                groups['fusion'].append(name)
            elif name.startswith('actor.'):
                groups['actor'].append(name)
            else:
                groups['critic'].append(name)
        return groups

    def optimizer_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _hidden(self, obs):
        return torch.tanh(self.actor.fusion(obs['x'][:, :2])
                          + self.actor.route_fusion(obs['x'][:, 1:]))

    def forward(self, obs):
        return self.actor.mlp(self._hidden(obs))

    def evaluate(self, obs):
        return self.critic(torch.cat((obs['x'], self._hidden(obs)), -1))

    def action_distribution(self, obs):
        return Normal(self(obs), self.std.clamp(0.05, 1.00))

    def constrain_std(self):
        with torch.no_grad():
            self.std.clamp_(0.05, 1.00)

    def update_critic_statistics(self, obs):
        return None


def storage(batch=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    obs = {'x': torch.randn(batch, 3, generator=generator)}
    means = torch.zeros(batch, 12)
    stds = torch.full((batch, 12), 0.30)
    actions = means + 0.30 * torch.randn(batch, 12, generator=generator)
    logp = Normal(means, stds).log_prob(actions).sum(-1)
    return dict(obs=obs, actions=actions, means=means, stds=stds, logp=logp,
                advantage=torch.randn(batch, generator=generator),
                returns=torch.randn(batch, generator=generator),
                value=torch.randn(batch, generator=generator))


def test_config_matches_frozen_spec_exactly():
    cfg = target.frozen_config()
    assert cfg.learning_rate == 3e-4
    assert cfg.learning_rate >= 1e-5 and cfg.learning_rate <= 3e-4
    assert (cfg.min_learning_rate, cfg.max_learning_rate) == (1e-5, 3e-4)
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip == 0.2
    assert cfg.entropy == 0.003
    assert cfg.value_coef == 1.0
    assert cfg.clipped_value_loss is True
    assert cfg.max_grad_norm == 1.0
    assert (cfg.initial_std, cfg.std_min, cfg.std_max) == (0.30, 0.05, 1.00)
    assert tuple(cfg.adam_betas) == (0.9, 0.999)
    assert cfg.adam_eps == 1e-8 and cfg.weight_decay == 0.0
    assert cfg.kl_high == 0.02 and cfg.kl_low == 0.005
    assert cfg == target.validate_config(cfg)
    assert cfg.num_envs == 512 and cfg.eval_num_envs == 8
    assert (cfg.rollout, cfg.epochs, cfg.minibatches) == (48, 5, 4)


def test_config_refuses_any_deviation():
    for change in (dict(learning_rate=2e-4), dict(gamma=0.995), dict(entropy=0.005),
                   dict(initial_std=0.15), dict(std_max=0.30), dict(clip=0.1)):
        with pytest.raises(ValueError):
            target.validate_config(replace(target.frozen_config(), **change))


def test_kl_adaptive_schedule_and_clamp():
    cfg = target.frozen_config()
    assert target.adapt_learning_rate(3e-4, 0.03, cfg) == pytest.approx(2e-4)
    assert target.adapt_learning_rate(3e-4, 0.001, cfg) == pytest.approx(3e-4)  # clamped
    assert target.adapt_learning_rate(1e-5, 0.03, cfg) == pytest.approx(1e-5)
    assert target.adapt_learning_rate(1e-5, 0.001, cfg) == pytest.approx(1.5e-5)
    assert target.adapt_learning_rate(3e-4, 0.01, cfg) == pytest.approx(3e-4)
    assert target.adapt_learning_rate(3e-4, 0.02, cfg) == pytest.approx(3e-4)


def test_control_sample_and_adam_arithmetic():
    assert target.control_samples(512) == 24576
    assert target.physics_substeps(24576) == 98304
    assert target.ADAM_STEPS_PER_UPDATE == 20
    assert target.control_samples(8) == 8 * 48


def test_group_gradient_norms_do_not_change_gradients():
    model = TinyModel()
    optimizer = target.build_optimizer(model)
    batch = storage()
    with torch.no_grad():
        obs = batch['obs']
    distribution = model.action_distribution(obs)
    loss = (distribution.log_prob(batch['actions']).sum(-1)
            + model.evaluate(obs).squeeze(-1)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    before = {name: parameter.grad.detach().clone()
              for name, parameter in model.named_parameters()}
    norms, counts = target.group_gradient_norms(model)
    after = {name: parameter.grad.detach() for name, parameter in model.named_parameters()}
    for name in before:
        assert torch.equal(before[name], after[name]), name
    assert set(norms) == set(target.GRADIENT_GROUPS)
    expected = {'actor': 2, 'critic': 2, 'fusion': 4, 'std': 1}
    assert counts == expected
    manual = sum(float((parameter.grad.detach() ** 2).sum())
                 for parameter in model.parameters() if parameter.grad is not None) ** 0.5
    assert sum(norms.values()) >= manual - 1e-9


def test_gae_terminal_masks_and_timeout_bootstrap():
    gamma, lam = 0.99, 0.95
    rewards = torch.ones(3, 1)
    values = torch.zeros(3, 1)
    # The episode ends at step 1; only that step's next_value carries a bootstrap.
    next_values = torch.tensor([[0.0], [100.0], [0.0]])
    done = torch.tensor([[False], [True], [False]])
    terminal = done.clone()  # success/failure: never bootstrapped
    advantage, _ = target.compute_gae(rewards, values, next_values, terminal, done,
                                      gamma=gamma, gae_lambda=lam)
    assert advantage[1, 0].item() == pytest.approx(1.0)
    assert advantage[0, 0].item() == pytest.approx(1.0 + gamma * lam)
    # A truncation (timeout) bootstraps from the reset-prior terminal observation
    # value, but the advantage still does not leak into the next episode.
    advantage_timeout, _ = target.compute_gae(
        rewards, values, next_values, torch.zeros(3, 1, dtype=torch.bool), done,
        gamma=gamma, gae_lambda=lam)
    assert advantage_timeout[1, 0].item() == pytest.approx(1.0 + gamma * 100.0)
    assert advantage_timeout[0, 0].item() == pytest.approx(
        1.0 + gamma * lam * (1.0 + gamma * 100.0))


def test_advantage_standardization():
    advantage = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = target.normalize_advantage(advantage)
    assert abs(float(normalized.mean())) < 1e-6
    assert float(normalized.std(unbiased=False)) == pytest.approx(1.0, abs=1e-6)


def test_update_math_single_adam_and_group_diagnostics():
    model = TinyModel()
    optimizer = target.build_optimizer(model)
    membership = target.optimizer_membership(optimizer, model)
    assert membership['exactly_once'] is True and membership['groups'] == 1
    batch = storage()
    metrics, diagnostics = target.update_math(
        model=model, optimizer=optimizer, storage=batch, device='cpu')
    assert diagnostics['adam_steps'] == 20
    assert metrics['optimizer_steps'] == 20
    assert 1e-5 - 1e-12 <= metrics['lr'] <= 3e-4 + 1e-12
    assert 0.05 - 1e-9 <= metrics['std_min'] and metrics['std_max'] <= 1.00 + 1e-9
    for group in target.GRADIENT_GROUPS:
        block = diagnostics[f'grad_norm_{group}']
        assert min(block['min'], block['mean'], block['max']) >= 0.0
    assert len(diagnostics['minibatches']) == 20
    row = diagnostics['minibatches'][0]
    for key in ('grad_norm_actor', 'grad_norm_critic', 'grad_norm_fusion',
                'grad_norm_std', 'grad_norm_preclip', 'kl', 'lr'):
        assert key in row
    assert diagnostics['grad_norm_fusion']['mean'] > 0.0
    assert diagnostics['grad_norm_actor']['mean'] > 0.0
    assert diagnostics['std_range'] == [0.05, 1.00]


def test_update_math_projects_std_every_step():
    model = TinyModel()
    with torch.no_grad():
        model.std.fill_(0.01)
    optimizer = target.build_optimizer(model)
    target.update_math(model=model, optimizer=optimizer, storage=storage(),
                       device='cpu')
    assert float(model.std.max()) <= 1.00 + 1e-12


def test_update_math_never_early_exits_or_duplicates_optimizer():
    model = TinyModel()
    optimizer = target.build_optimizer(model)
    seen, calls = [], []

    real_step = optimizer.step

    def step(*args, **kwargs):
        calls.append(1)
        return real_step(*args, **kwargs)

    optimizer.step = step
    target.update_math(model=model, optimizer=optimizer, storage=storage(),
                       device='cpu')
    assert len(calls) == 20
    assert len(optimizer.param_groups) == 1
    assert sorted(id(p) for p in optimizer.param_groups[0]['params']) == sorted(
        id(p) for p in model.optimizer_parameters())


def test_trainer_learning_ledger_and_collect_rollout():
    class Env:
        num_envs = 2
        device = 'cpu'

        def __init__(self):
            self.step_count = 0

        def reset(self):
            return ({'x': torch.zeros(2, 3)}, {})

        def step(self, action):
            self.step_count += 1
            obs = {'x': torch.ones(2, 3) * self.step_count}
            reward = torch.ones(2)
            terminated = torch.zeros(2, dtype=torch.bool)
            truncated = torch.zeros(2, dtype=torch.bool)
            if self.step_count % 3 == 0:
                truncated = torch.ones(2, dtype=torch.bool)
            info = {'terminal_observation': obs, 'episodes': [],
                    'reward_terms': {'tracking_lin_vel': 2.0}}
            return obs, reward, terminated, truncated, info

    model = TinyModel()
    trainer = target.PPOTrainer(model, Env())
    assert trainer.learning['update'] == 0
    records = trainer.collect()
    assert records == []
    assert trainer.storage['advantage'].shape[0] == 2 * 48
    advantage = trainer.storage['advantage']
    assert abs(float(advantage.mean())) < 1e-5
    train_metrics = trainer.update()
    assert train_metrics['lr'] <= 3e-4
    assert trainer.learning['update'] == 1
    assert trainer.learning['control_samples'] == 96
    assert trainer.learning['physics_substeps'] == 384
    assert trainer.learning['adam_steps'] == 20
    assert trainer.rollout_metrics['tracking_lin_vel'] == pytest.approx(2.0)


def test_update_math_rejects_non_divisible_batch():
    model = TinyModel()
    optimizer = target.build_optimizer(model)
    with pytest.raises(ValueError):
        target.update_math(model=model, optimizer=optimizer,
                           storage=storage(batch=10), device='cpu')
