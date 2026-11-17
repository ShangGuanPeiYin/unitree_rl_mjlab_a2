"""v0.6 PPO: one Adam, clipped value loss, KL-adaptive learning rate, per-group
pre-clip gradient norms and std projection after every optimizer step.

All numbers come from :data:`src.tasks.velocity.v07.model.PPO_SETTINGS` (which
reads the frozen ``spec.PPO_CONFIG``); this module owns no tunable value.

SDS section 4 requirements implemented here:

* exactly **one** Adam over every trainable parameter exactly once with
  ``betas=(0.9, 0.999)``, ``eps=1e-8``, ``weight_decay=0``;
* ``gamma=0.99``, ``GAE lambda=0.95``, ``clip=0.2``, ``entropy=0.003``,
  ``value_coef=1.0``, clipped value loss, whole-model gradient clip ``1.0``;
* learning rate ``3e-4`` clamped to ``[1e-5, 3e-4]``; minibatch ``KL > 0.02``
  divides by 1.5, ``0 < KL < 0.005`` multiplies by 1.5, then the clamp;
* ``std`` is projected to ``[0.05, 1.00]`` after each Adam step;
* actor / critic / fusion / std **pre-clip** gradient norms are recorded
  separately; the measurement never changes a gradient;
* advantage standardization and timeout bootstrap stay wired exactly as in the
  v0.3/v0.5 lineage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import torch

from .model import (ADAM_BETAS, ADAM_EPS, EPOCHS, MINIBATCHES, NUM_ENVS,
                    PPO_SETTINGS, PRIVILEGED_DIM, PROPRIO_DIM, ROLLOUT,
                    ROUTE_ERROR_KEY, STD_MAX, STD_MIN, WEIGHT_DECAY,
                    GRADIENT_GROUPS)
from . import spec

PPO_FIELDS = ('num_envs', 'eval_num_envs', 'rollout', 'epochs', 'minibatches',
              'learning_rate', 'min_learning_rate', 'max_learning_rate', 'desired_kl',
              'kl_high', 'kl_low', 'gamma', 'gae_lambda', 'clip', 'entropy',
              'value_coef', 'max_grad_norm', 'clipped_value_loss', 'initial_std',
              'std_min', 'std_max', 'adam_betas', 'adam_eps', 'weight_decay')
ACTION_DIM = int(spec.N_JOINTS)
ADAM_STEPS_PER_UPDATE = EPOCHS * MINIBATCHES
CONTROL_SAMPLES_PER_UPDATE = NUM_ENVS * ROLLOUT
PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE = int(spec.DECIMATION)
STD_TOUCH_TOLERANCE = 1e-6
MAX_KL_WARNING = 0.10
MEAN_CLIP_FRACTION_WARNING = 0.50


@dataclass(frozen=True)
class PPOConfig:
    """The frozen SDS section 4 configuration (CLI values are assertions only)."""

    num_envs: int = NUM_ENVS
    eval_num_envs: int = PPO_SETTINGS['eval_num_envs']
    rollout: int = ROLLOUT
    epochs: int = EPOCHS
    minibatches: int = MINIBATCHES
    learning_rate: float = float(PPO_SETTINGS['learning_rate'])
    min_learning_rate: float = float(PPO_SETTINGS['min_learning_rate'])
    max_learning_rate: float = float(PPO_SETTINGS['max_learning_rate'])
    desired_kl: float = float(PPO_SETTINGS['desired_kl'])
    kl_high: float = float(PPO_SETTINGS['kl_high'])
    kl_low: float = float(PPO_SETTINGS['kl_low'])
    gamma: float = float(PPO_SETTINGS['gamma'])
    gae_lambda: float = float(PPO_SETTINGS['gae_lambda'])
    clip: float = float(PPO_SETTINGS['clip'])
    entropy: float = float(PPO_SETTINGS['entropy'])
    value_coef: float = float(PPO_SETTINGS['value_coef'])
    max_grad_norm: float = float(PPO_SETTINGS['max_grad_norm'])
    clipped_value_loss: bool = bool(PPO_SETTINGS['clipped_value_loss'])
    initial_std: float = float(PPO_SETTINGS['initial_std'])
    std_min: float = float(PPO_SETTINGS['std_min'])
    std_max: float = float(PPO_SETTINGS['std_max'])
    adam_betas: tuple = tuple(float(value) for value in PPO_SETTINGS['adam_betas'])
    adam_eps: float = float(PPO_SETTINGS['adam_eps'])
    weight_decay: float = float(PPO_SETTINGS['weight_decay'])

    def as_dict(self):
        return asdict(self)


def frozen_config():
    return PPOConfig()


def validate_config(cfg=None):
    """Return ``cfg`` unchanged after an exact equality check against the spec."""
    cfg = PPOConfig() if cfg is None else cfg
    if not isinstance(cfg, PPOConfig) or asdict(cfg) != asdict(PPOConfig()):
        raise ValueError('v0.6 PPO configuration must equal the frozen spec.PPO_CONFIG values')
    return cfg


def control_samples(num_envs, cfg=None):
    cfg = validate_config(cfg)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    return cfg.rollout * num_envs


def physics_substeps(control_sample_count, *, substeps=PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE):
    if isinstance(control_sample_count, bool) or not isinstance(control_sample_count, int) \
            or control_sample_count < 0:
        raise ValueError('control samples must be a non-negative integer')
    if substeps != PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE:
        raise ValueError('the v0.6 decimation is fixed at 4 physics substeps')
    return control_sample_count * PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE


def build_optimizer(model, cfg=None):
    """The single v0.6 Adam: every trainable parameter exactly once."""
    cfg = validate_config(cfg)
    parameters = (model.optimizer_parameters() if hasattr(model, 'optimizer_parameters')
                  else [p for p in model.parameters() if p.requires_grad])
    if not parameters:
        raise ValueError('model exposes no trainable parameter')
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError('a trainable parameter appears more than once')
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(names) != len(parameters):
        raise ValueError('model.optimizer_parameters() does not cover every trainable parameter')
    return torch.optim.Adam(parameters, lr=cfg.learning_rate, betas=cfg.adam_betas,
                            eps=cfg.adam_eps, weight_decay=cfg.weight_decay)


def optimizer_membership(optimizer, model):
    """Audit that Adam holds exactly the model's trainable parameters once each."""
    seen, ids = [], {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            identifier = id(parameter)
            ids[identifier] = ids.get(identifier, 0) + 1
            seen.append(identifier)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    duplicates = sorted(key for key, count in ids.items() if count > 1)
    return dict(
        optimizer_parameters=len(seen), unique_parameters=len(ids),
        trainable_parameters=len(trainable), groups=len(optimizer.param_groups),
        duplicates=duplicates,
        exactly_once=bool(len(seen) == len(ids) == len(trainable)
                          and set(seen) == trainable and not duplicates),
        optimizer_class=type(optimizer).__name__,
        default_lr=float(optimizer.param_groups[0]['lr']),
        betas=tuple(float(value) for value in optimizer.param_groups[0]['betas']),
        eps=float(optimizer.param_groups[0]['eps']),
        weight_decay=float(optimizer.param_groups[0]['weight_decay']),
    )


# ------------------------------------------------------------------- rollout
def compute_gae(rewards, values, next_values, terminated, done, *, gamma, gae_lambda):
    """GAE with terminal masking and reset-prior timeout bootstrap.

    ``terminated`` marks a true terminal step (success/failure: never bootstrapped);
    ``done`` additionally contains truncations, whose ``next_values`` are the
    reset-prior terminal observation values, so the timeout bootstraps without
    propagating the advantage across the episode boundary.
    """
    for name, tensor in (('rewards', rewards), ('values', values),
                         ('next_values', next_values), ('terminated', terminated),
                         ('done', done)):
        if not torch.is_tensor(tensor):
            raise TypeError(f'{name} must be a tensor')
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError('reward/value shapes differ')
    if rewards.ndim != 2:
        raise ValueError('rollout tensors must be [T, N]')
    horizon = rewards.shape[0]
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for step in reversed(range(horizon)):
        delta = (rewards[step]
                 + gamma * (1.0 - terminated[step].to(rewards.dtype)) * next_values[step]
                 - values[step])
        gae = delta + gamma * gae_lambda * (1.0 - done[step].to(rewards.dtype)) * gae
        advantage[step] = gae
    returns = advantage + values
    return advantage, returns


def normalize_advantage(advantage):
    if not torch.is_tensor(advantage):
        raise TypeError('advantage must be a tensor')
    return (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)


def adapt_learning_rate(learning_rate, kl, cfg=None):
    """SDS section 4 KL adaptive schedule, clamped to the frozen lr range."""
    cfg = validate_config(cfg)
    if not math.isfinite(float(kl)):
        raise ValueError('KL must be finite')
    lr = float(learning_rate)
    if kl > cfg.kl_high:
        lr = lr / 1.5
    elif 0.0 < kl < cfg.kl_low:
        lr = lr * 1.5
    return min(max(lr, cfg.min_learning_rate), cfg.max_learning_rate)


# ------------------------------------------------------- gradient diagnostics
def group_gradient_norms(model):
    """Pre-clip L2 norm of each SDS gradient group; never modifies a gradient."""
    groups = (model.parameter_groups() if hasattr(model, 'parameter_groups') else {})
    parameters = {name: parameter for name, parameter in model.named_parameters()}
    norms, counts = {}, {}
    for group in GRADIENT_GROUPS:
        total = torch.zeros((), dtype=torch.float64)
        names = groups.get(group, [])
        count = 0
        for name in names:
            parameter = parameters[name]
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().to(torch.float64)
            total = total + grad.pow(2).sum()
            count += 1
        norms[group] = float(total.sqrt())
        counts[group] = count
    return norms, counts


def _stat(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return dict(min=float(tensor.min()), mean=float(tensor.mean()),
                max=float(tensor.max()), count=int(tensor.numel()))


# ---------------------------------------------------------------- update math
def update_math(*, model, optimizer, storage, device, cfg=None,
                learning_state=None, notify=None, validate=True):
    """One complete PPO update: ``epochs * minibatches`` Adam steps.

    Returns ``(metrics, diagnostics)``.  ``diagnostics`` carries the per-minibatch
    rows including the four pre-clip gradient groups.
    """
    cfg = frozen_config() if cfg is None else cfg
    if validate:
        cfg = validate_config(cfg)
    else:
        cfg = PPOConfig(**{**asdict(PPOConfig()),
                           **{name: getattr(cfg, name) for name in PPO_FIELDS}})
    state = learning_state if learning_state is not None else dict(
        optimizer_steps=0, completed_ppo_updates=0)
    if storage is None:
        raise RuntimeError('missing rollout')
    batch = int(len(storage['advantage']))
    if batch % cfg.minibatches:
        raise ValueError('rollout rows are not divisible into minibatches')

    def send(event, **payload):
        if notify is not None:
            notify(event, **payload)

    rows, result = [], []
    clipped_samples = total_samples = 0
    send('optimizer_started', expected_optimizer_steps=cfg.epochs * cfg.minibatches)
    for epoch in range(cfg.epochs):
        permutation = torch.randperm(batch, device=device)
        for minibatch, index in enumerate(permutation.chunk(cfg.minibatches)):
            obs = {key: value[index] for key, value in storage['obs'].items()}
            distribution = model.action_distribution(obs)
            value = model.evaluate(obs).squeeze(-1)
            with torch.no_grad():
                kl = (torch.log(distribution.stddev / storage['stds'][index])
                      + (storage['stds'][index] ** 2
                         + (storage['means'][index] - distribution.mean) ** 2)
                      / (2 * distribution.stddev ** 2) - 0.5).sum(-1).mean()
                if not math.isfinite(float(kl)):
                    raise ValueError('non-finite minibatch KL')
                learning_rate = adapt_learning_rate(
                    optimizer.param_groups[0]['lr'], float(kl), cfg)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
            ratio = torch.exp(distribution.log_prob(storage['actions'][index]).sum(-1)
                              - storage['logp'][index])
            policy_loss = torch.maximum(
                -storage['advantage'][index] * ratio,
                -storage['advantage'][index]
                * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
            if cfg.clipped_value_loss:
                clipped = storage['value'][index] + (value - storage['value'][index]).clamp(
                    -cfg.clip, cfg.clip)
                value_loss = torch.maximum(
                    (value - storage['returns'][index]) ** 2,
                    (clipped - storage['returns'][index]) ** 2).mean()
            else:
                value_loss = ((value - storage['returns'][index]) ** 2).mean()
            entropy = distribution.entropy().sum(-1).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy * entropy
            if not math.isfinite(float(loss.detach())):
                raise ValueError('non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError(f'non-finite gradient: {name}')
            if any(parameter.grad is None for parameter in model.parameters()
                   if parameter.requires_grad):
                raise ValueError('a trainable parameter received no gradient')
            # Read-only measurement BEFORE clipping: this never changes the gradient.
            group_norms, group_counts = group_gradient_norms(model)
            clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                          cfg.max_grad_norm)
            if not math.isfinite(float(clipped_norm)):
                raise ValueError('non-finite gradient norm')
            optimizer.step()
            state['optimizer_steps'] = state.get('optimizer_steps', 0) + 1
            state['optimizer_initialized'] = True
            model.constrain_std()
            send('optimizer_step_completed', epoch_index=epoch + 1,
                 minibatch_index=minibatch + 1)
            with torch.no_grad():
                std = model.std.detach().reshape(-1)
                deviation = (ratio.detach() - 1).abs()
                rows.append(dict(
                    epoch=epoch + 1, minibatch=minibatch + 1, kl=float(kl),
                    lr=float(learning_rate),
                    grad_norm_preclip=float(clipped_norm),
                    grad_norm_actor=group_norms['actor'],
                    grad_norm_critic=group_norms['critic'],
                    grad_norm_fusion=group_norms['fusion'],
                    grad_norm_std=group_norms['std'],
                    grad_parameter_counts=group_counts,
                    clip_fraction=float((deviation > cfg.clip).float().mean()),
                    loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                    value_loss=float(value_loss.detach()), entropy=float(entropy.detach()),
                    std_min=float(std.min()), std_max=float(std.max()),
                    std_touch_min=int((std <= cfg.std_min + STD_TOUCH_TOLERANCE).sum()),
                    std_touch_max=int((std >= cfg.std_max - STD_TOUCH_TOLERANCE).sum()),
                    std_dimensions=int(std.numel()),
                    approx_kl_above_high=bool(float(kl) > cfg.kl_high),
                    approx_kl_below_low=bool(0.0 < float(kl) < cfg.kl_low),
                ))
            clipped_samples += int((deviation > cfg.clip).sum())
            total_samples += int(index.numel())
            result.append([float(loss.detach()), float(policy_loss.detach()),
                           float(value_loss.detach()), float(entropy.detach()),
                           float(kl), float(learning_rate), float(clipped_norm)])
    state['completed_ppo_updates'] = state.get('completed_ppo_updates', 0) + 1
    send('update_completed', optimizer_steps_in_update=cfg.epochs * cfg.minibatches)
    import numpy as np
    mean = np.mean(result, axis=0)
    std = model.std.detach().reshape(-1).to(torch.float64)
    metrics = dict(zip(('loss', 'policy_loss', 'value_loss', 'entropy', 'kl', 'lr',
                        'gradient_norm'), mean.tolist()))
    metrics.update(
        clip_fraction=clipped_samples / max(total_samples, 1),
        lr=float(optimizer.param_groups[0]['lr']),
        std_mean=float(std.mean()), std_min=float(std.min()), std_max=float(std.max()),
        optimizer_steps=cfg.epochs * cfg.minibatches,
        completed_update=state['completed_ppo_updates'],
        **{f'std_{index}': float(value) for index, value in enumerate(std)})
    diagnostics = measure(minibatch_rows=rows, storage=storage, std=std, cfg=cfg)
    diagnostics.update(lr=float(optimizer.param_groups[0]['lr']),
                       completed_update=state['completed_ppo_updates'])
    return metrics, diagnostics


def measure(*, minibatch_rows, storage, std, cfg=None, explain=None):
    cfg = frozen_config() if cfg is None else cfg
    rows = list(minibatch_rows)
    if not rows:
        raise ValueError('no minibatch measured')
    kl = _stat([row['kl'] for row in rows])
    lr = _stat([row['lr'] for row in rows])
    grad = _stat([row['grad_norm_preclip'] for row in rows])
    clip = _stat([row['clip_fraction'] for row in rows])
    groups = {name: _stat([row[f'grad_norm_{name}'] for row in rows])
              for name in GRADIENT_GROUPS}
    std_tensor = torch.as_tensor(std, dtype=torch.float64).reshape(-1)
    values = torch.as_tensor(storage['value']).detach().to(torch.float64).reshape(-1)
    returns = torch.as_tensor(storage['returns']).detach().to(torch.float64).reshape(-1)
    variance, reason = explained_variance(values, returns)
    codes = []
    if kl['max'] > MAX_KL_WARNING:
        codes.append('max_kl_above_0.10')
    if clip['mean'] > MEAN_CLIP_FRACTION_WARNING:
        codes.append('mean_clip_fraction_above_0.50')
    return dict(
        minibatch_count=len(rows), adam_steps=len(rows),
        kl_min=kl['min'], kl_mean=kl['mean'], kl_max=kl['max'],
        lr_min=lr['min'], lr_mean=lr['mean'], lr_max=lr['max'],
        grad_norm_preclip_min=grad['min'], grad_norm_preclip_mean=grad['mean'],
        grad_norm_preclip_max=grad['max'],
        gradient_clip_ratio=sum(1 for row in rows
                                if row['grad_norm_preclip'] > cfg.max_grad_norm) / len(rows),
        grad_norm_actor=groups['actor'], grad_norm_critic=groups['critic'],
        grad_norm_fusion=groups['fusion'], grad_norm_std=groups['std'],
        clip_fraction_min=clip['min'], clip_fraction_mean=clip['mean'],
        clip_fraction_max=clip['max'],
        value_mean=float(values.mean()), value_std=float(values.std(unbiased=False)),
        return_mean=float(returns.mean()), return_std=float(returns.std(unbiased=False)),
        explained_variance=variance, explained_variance_reason=reason,
        std={f'{index:02d}': float(value) for index, value in enumerate(std_tensor)},
        std_dimensions=int(std_tensor.numel()),
        std_range=[cfg.std_min, cfg.std_max],
        std_touch_min_count=int((std_tensor <= cfg.std_min + STD_TOUCH_TOLERANCE).sum()),
        std_touch_max_count=int((std_tensor >= cfg.std_max - STD_TOUCH_TOLERANCE).sum()),
        warning_codes=codes, minibatches=rows,
        control_samples=cfg.num_envs * cfg.rollout,
        physics_substeps=physics_substeps(cfg.num_envs * cfg.rollout),
    )


def explained_variance(values, returns):
    values = torch.as_tensor(values).detach().to(torch.float64).reshape(-1)
    returns = torch.as_tensor(returns).detach().to(torch.float64).reshape(-1)
    if values.numel() != returns.numel() or values.numel() == 0:
        return None, 'empty_or_mismatched_samples'
    if not (torch.isfinite(values).all() and torch.isfinite(returns).all()):
        return None, 'nonfinite_value_or_return'
    variance_return = float(torch.var(returns, unbiased=False))
    if variance_return == 0.0:
        return None, 'zero_return_variance'
    variance_error = float(torch.var(returns - values, unbiased=False))
    return 1.0 - variance_error / variance_return, None


# -------------------------------------------------------------------- trainer
class PPOTrainer:
    """Rollout collection + the frozen update loop over a single Adam."""

    def __init__(self, model, env, cfg=None):
        self.model = model
        self.env = env
        self.cfg = validate_config(cfg)
        self.optimizer = build_optimizer(model, self.cfg)
        self.obs = None
        self.storage = None
        self.learning_state = dict(optimizer_initialized=False,
                                   completed_ppo_updates=0, optimizer_steps=0)
        self.rollout_metrics = {}
        self.last_update_diagnostics = None
        self.update_history = []
        self.control_step_callback = None
        self.update_callback = None

    # ------------------------------------------------------------- accounting
    @property
    def num_envs(self):
        return int(getattr(self.env, 'num_envs', self.cfg.num_envs))

    @property
    def control_samples_per_update(self):
        return self.cfg.rollout * self.num_envs

    @property
    def physics_substeps_per_update(self):
        return physics_substeps(self.control_samples_per_update)

    @property
    def adam_steps_per_update(self):
        return self.cfg.epochs * self.cfg.minibatches

    @property
    def learning(self):
        update = self.learning_state['completed_ppo_updates']
        samples = self.control_samples_per_update * update
        return dict(update=update, optimizer_step=self.learning_state['optimizer_steps'],
                    adam_steps=self.learning_state['optimizer_steps'],
                    control_samples=samples, physics_substeps=physics_substeps(samples))

    def membership(self):
        return optimizer_membership(self.optimizer, self.model)

    # -------------------------------------------------------------- lifecycle
    def reset(self):
        out = self.env.reset()
        self.obs = out[0] if isinstance(out, tuple) else out
        return self.obs

    def collect(self):
        if self.obs is None:
            self.reset()
        cfg, env = self.cfg, self.env
        n = self.num_envs
        device = getattr(env, 'device', 'cpu')
        obs_buffer = {key: torch.empty((cfg.rollout, *value.shape), dtype=value.dtype,
                                       device=device)
                      for key, value in self.obs.items()}
        fields = {key: torch.empty((cfg.rollout, n), device=device)
                  for key in ('logp', 'value', 'reward', 'terminated', 'done',
                              'next_value')}
        actions = torch.empty((cfg.rollout, n, ACTION_DIM), device=device)
        means = torch.empty_like(actions)
        stds = torch.empty_like(actions)
        records, reward_sums = [], {}
        self.model.train()
        self.model.update_critic_statistics(self.obs)
        for step in range(cfg.rollout):
            with torch.no_grad():
                for key, value in self.obs.items():
                    obs_buffer[key][step].copy_(value)
                distribution = self.model.action_distribution(self.obs)
                action = distribution.sample()
                if not torch.isfinite(action).all():
                    raise ValueError('non-finite sampled action')
                value = self.model.evaluate(self.obs).squeeze(-1)
                next_obs, reward, terminated, truncated, info = env.step(action)
                if self.control_step_callback is not None:
                    self.control_step_callback(dict(step_index=step + 1, num_envs=n,
                                                    info=info))
                terminal_obs = info.get('terminal_observation', next_obs)
                next_value = self.model.evaluate(terminal_obs).squeeze(-1)
                logp = distribution.log_prob(action).sum(-1)
                for key, tensor in (('logp', logp), ('value', value), ('reward', reward),
                                    ('terminated', terminated), ('done', terminated | truncated),
                                    ('next_value', next_value)):
                    fields[key][step].copy_(tensor)
                actions[step].copy_(action)
                means[step].copy_(distribution.mean)
                stds[step].copy_(distribution.stddev)
                records.extend(info.get('episodes', []))
                self.obs = next_obs
                for key, total in info.get('reward_terms', {}).items():
                    reward_sums[key] = reward_sums.get(key, 0.0) + float(total)
        advantage, returns = compute_gae(
            fields['reward'], fields['value'], fields['next_value'],
            fields['terminated'], fields['done'],
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        advantage = normalize_advantage(advantage)
        self.storage = dict(
            obs={key: value.flatten(0, 1) for key, value in obs_buffer.items()},
            actions=actions.flatten(0, 1), means=means.flatten(0, 1),
            stds=stds.flatten(0, 1), advantage=advantage.flatten(),
            returns=returns.flatten(),
            **{key: value.flatten() for key, value in fields.items()})
        self.rollout_metrics = {key: value / cfg.rollout
                                for key, value in reward_sums.items()}
        return records

    def update(self):
        if self.storage is None:
            raise RuntimeError('missing rollout')
        metrics, diagnostics = update_math(
            model=self.model, optimizer=self.optimizer, storage=self.storage,
            device=getattr(self.env, 'device', 'cpu'), cfg=self.cfg,
            learning_state=self.learning_state, notify=self._notify)
        diagnostics['reward_terms'] = dict(self.rollout_metrics)
        diagnostics['update'] = self.learning_state['completed_ppo_updates']
        diagnostics['control_samples'] = self.control_samples_per_update
        diagnostics['physics_substeps'] = self.physics_substeps_per_update
        self.last_update_diagnostics = diagnostics
        self.update_history.append({key: value for key, value in diagnostics.items()
                                    if key != 'minibatches'})
        for code in diagnostics['warning_codes']:
            warnings.warn(f'v0.6 PPO update {diagnostics["update"]}: {code}',
                          RuntimeWarning, stacklevel=2)
        result = dict(metrics)
        result.update(self.rollout_metrics)
        result.update({key: value for key, value in diagnostics.items()
                       if key != 'minibatches'})
        return result

    def _notify(self, event, **payload):
        if self.update_callback is not None:
            self.update_callback(event, dict(learning=self.learning, **payload))

    def train_step(self):
        records = self.collect()
        metrics = self.update()
        return records, metrics

    def diagnostics_jsonl_record(self):
        if self.last_update_diagnostics is None:
            raise ValueError('no completed update to record')
        return {key: value for key, value in self.last_update_diagnostics.items()
                if key != 'minibatches'}


#: Alias kept for consumers written against the W0 section 4.5 name.
PPO = PPOTrainer

__all__ = [
    'ACTION_DIM', 'ADAM_STEPS_PER_UPDATE', 'CONTROL_SAMPLES_PER_UPDATE',
    'MAX_KL_WARNING', 'MEAN_CLIP_FRACTION_WARNING', 'PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE',
    'PPO', 'PPOConfig', 'PPOTrainer', 'STD_TOUCH_TOLERANCE', 'adapt_learning_rate',
    'build_optimizer', 'compute_gae', 'control_samples', 'explained_variance',
    'frozen_config', 'group_gradient_norms', 'measure', 'normalize_advantage',
    'optimizer_membership', 'physics_substeps', 'update_math', 'validate_config',
]
