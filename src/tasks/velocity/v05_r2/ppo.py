"""v0.5-r2 PPO: the frozen v0.5 mathematics plus the ``route_error`` rollout key.

SDS v0.5-r2 2.5/4.4/7.6 keep every PPO number and every update operation of the
r1/v0.3 core unchanged.  The only r2 difference is the fifth observation key:

* ``collect`` allocates its rollout buffers from ``self.obs``' key set, so the
  five-key observation (``proprio/history/heightmap/privileged/route_error``)
  automatically produces ``obs['route_error'] == [48, B, 3]`` and a flattened
  ``storage['obs']['route_error'] == [48*B, 3]``;
* ``update_math`` slices ``obs = {k: v[idx] ...}`` for each of the 20
  minibatches, so the actor's ``route_fusion`` and the 211-input critic receive
  ``route_error[idx] == [B/4, 3]`` at every Adam step;
* ``route_error`` never enters reward, advantage, return, terminal bootstrap or
  any normalizer.  It is one more ``obs`` key and nothing else.

Missing, mis-shaped, wrong-dtype, wrong-device, wrong-batch or non-finite
``route_error`` is a hard ``ValueError`` naming the key and the consumer; there
is no zero fill, no clamp, no default (SDS 4.7).

Guarantees fixed by tests:

* ``update_math`` and ``v03.ppo.PPO.update`` produce bit-identical parameters,
  Adam moments, learning state and shared metrics from fixed storage + seed
  (``tests/test_v05_r2_ppo.py::EquivalentMathTests``).
* No early stop, no reward/termination write, no changed random order: only
  scalar readback is added after the existing operations.
* Non-finite loss/gradient/KL stay hard errors.

Unit contract (SDS 2.5/7.6): one update is ``rollout * num_envs`` control
samples (48*512 = 24576) and ``epochs * minibatches`` Adam steps (20).  A control
sample is ``DECIMATION`` physics substeps, so ``physics_substeps`` is strictly
``4 * control_samples``; the two are never mixed on one axis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
import torch

from ..rl.ideal_stairs_runner import finite
from ..v03.ppo import PPO as V03PPO
from . import spec as spec_module
from .spec import (DECIMATION, INITIAL_STD, PPO_CLIP, PPO_CLIPPED_VALUE_LOSS,
                   PPO_DESIRED_KL, PPO_ENTROPY, PPO_EPOCHS, PPO_GAMMA, PPO_INITIAL_LR,
                   PPO_LAMBDA, PPO_MAX_GRAD_NORM, PPO_MAX_LR, PPO_MIN_LR, PPO_MINIBATCHES,
                   PPO_NUM_ENVS, PPO_ROLLOUT, PPO_VALUE_COEF, STD_MAX, STD_MIN)

#: SDS 2.5 literal numbers.  Declared here and asserted against ``spec`` so a
#: drift between the two is an import-time hard error, never a silent default.
SDS_PPO = dict(
    num_envs=512, rollout=48, epochs=5, minibatches=4,
    learning_rate=5e-5, min_lr=1e-5, max_lr=1e-4, desired_kl=.01,
    gamma=.995, gae_lambda=.95, clip=.2, value_coef=1., max_grad_norm=1.,
    clipped_value_loss=True, entropy=.005, init_std=.15, std_min=.10, std_max=.30,
)

#: 48*512 control samples per update and 20 Adam steps per update (SDS 2.5).
CONTROL_SAMPLES_PER_UPDATE = SDS_PPO['rollout'] * SDS_PPO['num_envs']
ADAM_STEPS_PER_UPDATE = SDS_PPO['epochs'] * SDS_PPO['minibatches']
PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE = DECIMATION

#: SDS 7.6 warning thresholds.  They warn; they never stop training or tune.
MAX_KL_WARNING = .10
MEAN_CLIP_FRACTION_WARNING = .50
#: SDS 7.6: the first 20 updates keep all 20 minibatch rows.
FULL_MINIBATCH_UPDATES = 20
#: Tolerance for the std bound touch counts (SDS 7.6).
STD_TOUCH_TOLERANCE = 1e-6

#: SDS 2.1/4.4: the fifth observation key.  These are the observation-side
#: literals of the frozen route definition; ``spec`` owns the same contract.
ROUTE_ERROR_KEY = spec_module.ROUTE_ERROR_KEY
ROUTE_ERROR_DIM = spec_module.ROUTE_ERROR_DIM
ROUTE_SCALE_M = spec_module.ROUTE_LATERAL_SCALE
ROUTE_CLIP = spec_module.ROUTE_LATERAL_CLIP
ROUTE_OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', 'privileged',
                          ROUTE_ERROR_KEY)
#: SDS 4.8 route saturation point: |local_y| >= 0.80 m <=> |route_error[:,0]| >= 4.
ROUTE_SATURATION_M = ROUTE_SCALE_M * ROUTE_CLIP


def _spec_ppo():
    return dict(
        num_envs=PPO_NUM_ENVS, rollout=PPO_ROLLOUT, epochs=PPO_EPOCHS,
        minibatches=PPO_MINIBATCHES, learning_rate=PPO_INITIAL_LR, min_lr=PPO_MIN_LR,
        max_lr=PPO_MAX_LR, desired_kl=PPO_DESIRED_KL, gamma=PPO_GAMMA,
        gae_lambda=PPO_LAMBDA, clip=PPO_CLIP, value_coef=PPO_VALUE_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM, clipped_value_loss=PPO_CLIPPED_VALUE_LOSS,
        entropy=PPO_ENTROPY, init_std=INITIAL_STD, std_min=STD_MIN, std_max=STD_MAX,
    )


if _spec_ppo() != SDS_PPO:  # pragma: no cover - import-time contract guard
    raise RuntimeError(f'v0.5-r2 spec PPO contract differs from SDS 2.5: {_spec_ppo()!r}')
if CONTROL_SAMPLES_PER_UPDATE != 48 * 512 or ADAM_STEPS_PER_UPDATE != 20:
    raise RuntimeError('v0.5-r2 PPO budget identity drifted')
if ROUTE_ERROR_KEY not in spec_module.OBSERVATION:
    raise RuntimeError('v0.5-r2 spec observation must contain the route_error key')
if list(spec_module.OBSERVATION[ROUTE_ERROR_KEY]) != [ROUTE_ERROR_DIM]:
    raise RuntimeError('v0.5-r2 route_error must be [B,3] in the spec observation')


def route_error_contract():
    """The frozen r2 route/observation contract (SDS 2.1/4.2), import-checked."""
    return dict(key=ROUTE_ERROR_KEY, shape=[ROUTE_ERROR_DIM], scale_m=ROUTE_SCALE_M,
                clip=ROUTE_CLIP, saturation_m=ROUTE_SATURATION_M,
                observation_keys=list(ROUTE_OBSERVATION_KEYS))


def _batch_of(obs):
    for key in ('proprio', 'history', 'heightmap', 'privileged', ROUTE_ERROR_KEY):
        value = obs.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    return None


def validate_route_tensor(route, *, consumer, row_count=None):
    """Validate one bare ``route_error`` tensor (shape/dtype/range/finiteness)."""
    if not isinstance(route, torch.Tensor):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} must be a tensor, '
                         f'got {type(route).__name__}')
    if route.dtype != torch.float32:
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} dtype must be float32, '
                         f'got {route.dtype}')
    if route.ndim != 2 or route.shape[1] != ROUTE_ERROR_DIM:
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} shape must be [B,{ROUTE_ERROR_DIM}], '
                         f'got {tuple(route.shape)}')
    if row_count is not None and int(route.shape[0]) != int(row_count):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} has {int(route.shape[0])} rows, '
                         f'expected {int(row_count)}')
    if not bool(torch.isfinite(route).all()):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} contains NaN/Inf; the r2 route '
                         'observation must be finite (SDS 4.7)')
    numeric = route.to(torch.float64)
    if bool((numeric.abs() > ROUTE_CLIP + 1e-6).any()) or bool(
            (numeric[:, 1:].abs() > 1. + 1e-6).any()):
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} is outside the frozen '
                         f'[clip(local_y/0.20,-4,4), sin(e), cos(e)] range')
    return route


def validate_route_observation(obs, *, consumer, row_count=None):
    """Strictly validate ``obs['route_error']`` before any tensor operation.

    Checks presence, shape, dtype, device, batch agreement and finiteness
    (SDS 2.1/4.7).  ``row_count`` additionally pins the flattened rollout rows
    (``rollout * num_envs``) or one minibatch slice.  Returns the tensor.
    """
    if not isinstance(obs, dict):
        raise ValueError(f'{consumer}: observation must be a dict')
    if ROUTE_ERROR_KEY not in obs:
        raise ValueError(f'{consumer}: observation is missing {ROUTE_ERROR_KEY!r}; '
                         'the r2 route observation is required and is never zero-filled')
    route = validate_route_tensor(obs[ROUTE_ERROR_KEY], consumer=consumer,
                                  row_count=row_count)
    batch = _batch_of(obs)
    if batch is not None and int(route.shape[0]) != batch:
        raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} batch {int(route.shape[0])} '
                         f'differs from the other observation keys ({batch})')
    for key in ('proprio', 'history', 'heightmap', 'privileged'):
        value = obs.get(key)
        if isinstance(value, torch.Tensor) and value.device != route.device:
            raise ValueError(f'{consumer}: {ROUTE_ERROR_KEY} device {route.device} '
                             f'differs from {key} device {value.device}')
    return route


def route_aggregate(*, abs_y=None, normalized_abs_y=None, saturation=None,
                    heading_deg=None, nonfinite=0, exact):
    """Build the SDS 4.8 per-update ``route/*`` block from accumulated samples.

    ``abs_y`` (metres) is only exact when the environment exposes its real
    ``local_y`` (``route_metrics``/step diagnostics).  From ``route_error`` alone
    the first component is clipped at 4, so the reconstructed magnitude
    saturates; that case is reported explicitly with
    ``abs_y_source='route_component_clipped'`` instead of pretending to be the
    exact ``mean(|local_y|)``.
    """
    abs_y = torch.as_tensor(abs_y).detach().to(torch.float64).reshape(-1)
    normalized = torch.as_tensor(normalized_abs_y).detach().to(torch.float64).reshape(-1)
    saturation = torch.as_tensor(saturation).detach().to(torch.float64).reshape(-1)
    heading = torch.as_tensor(heading_deg).detach().to(torch.float64).reshape(-1)
    count = abs_y.numel()
    if count == 0:
        raise ValueError('route aggregate needs at least one sample')
    for name, value in (('abs_y', abs_y), ('normalized_abs_y', normalized),
                        ('saturation', saturation), ('heading_deg', heading)):
        if value.numel() != count:
            raise ValueError(f'route aggregate {name} sample count mismatch')
        if not torch.isfinite(value).all():
            raise ValueError('route/nonfinite_count must be 0; aggregate is non-finite')
    if nonfinite:
        raise ValueError('route/nonfinite_count must be 0')
    return dict(abs_y_mean=float(abs_y.mean()),
                abs_y_max=float(abs_y.max()),
                normalized_abs_y_max=float(normalized.max()),
                saturation_rate=float(saturation.mean()),
                heading_deg_mean=float(heading.mean()),
                heading_deg_max=float(heading.max()),
                nonfinite_count=0, samples=int(count),
                abs_y_source='env_local_y' if exact else 'route_component_clipped',
                component_scale_m=ROUTE_SCALE_M, clip=ROUTE_CLIP)


def route_diagnostics(route_error):
    """SDS 4.8 read-only ``route/*`` aggregates from the rollout route tensor.

    The first route component is ``clip(local_y/0.20, -4, 4)``, so the recovered
    lateral offset is ``component * 0.20`` inside the clip; the heading is
    ``|atan2(sin(e), cos(e))|``.  ``nonfinite_count`` is a hard check: it must
    stay zero and is never a metric to "pass" with.
    """
    route = torch.as_tensor(route_error).detach()
    if route.ndim != 2 or route.shape[1] != ROUTE_ERROR_DIM:
        raise ValueError(f'route_error shape must be [N,3], got {tuple(route.shape)}')
    finite_mask = torch.isfinite(route)
    nonfinite = int((~finite_mask).sum())
    if nonfinite:
        raise ValueError(f'route/nonfinite_count must be 0, got {nonfinite} samples')
    route = route.to(torch.float64)
    normalized_y = route[:, 0].abs()
    heading = torch.atan2(route[:, 1], route[:, 2]).abs()
    saturation = (normalized_y >= ROUTE_CLIP - 1e-6).to(torch.float64).mean()
    return dict(
        abs_y_mean=float(normalized_y.mean() * ROUTE_SCALE_M),
        abs_y_max=float(normalized_y.max() * ROUTE_SCALE_M),
        normalized_abs_y_max=float(normalized_y.max()),
        saturation_rate=float(saturation),
        heading_deg_mean=float(torch.rad2deg(heading.mean())),
        heading_deg_max=float(torch.rad2deg(heading.max())),
        nonfinite_count=0,
        samples=int(route.shape[0]),
        abs_y_source='route_component_clipped',
        component_scale_m=ROUTE_SCALE_M,
        clip=ROUTE_CLIP,
    )


@dataclass(frozen=True)
class PPOConfig:
    """Explicit v0.5-r2 PPO configuration (SDS 2.5).

    Fields used by the frozen update loop keep the v0.3 names, so the copied
    mathematics can read ``league`` attributes unchanged.  The extra fields are
    r2 contract numbers: they are checked by this module and by the schema-7
    checkpoint, and are never used to change the loop.
    """
    num_envs: int = PPO_NUM_ENVS
    rollout: int = PPO_ROLLOUT
    epochs: int = PPO_EPOCHS
    minibatches: int = PPO_MINIBATCHES
    learning_rate: float = PPO_INITIAL_LR
    min_lr: float = PPO_MIN_LR
    max_lr: float = PPO_MAX_LR
    desired_kl: float = PPO_DESIRED_KL
    gamma: float = PPO_GAMMA
    gae_lambda: float = PPO_LAMBDA
    clip: float = PPO_CLIP
    value_coef: float = PPO_VALUE_COEF
    max_grad_norm: float = PPO_MAX_GRAD_NORM
    clipped_value_loss: bool = PPO_CLIPPED_VALUE_LOSS
    entropy: float = PPO_ENTROPY
    init_std: float = INITIAL_STD
    std_min: float = STD_MIN
    std_max: float = STD_MAX


if asdict(PPOConfig()) != SDS_PPO:  # pragma: no cover - import-time contract guard
    raise RuntimeError('v0.5-r2 PPOConfig defaults differ from SDS 2.5')


def validate_config(cfg=None):
    """Return ``cfg`` unchanged after an exact SDS 2.5 equality check.

    CLI/checkpoint values are assertions only (SDS 10): there is deliberately no
    path that overrides one frozen number.
    """
    cfg = PPOConfig() if cfg is None else cfg
    if not isinstance(cfg, PPOConfig) or asdict(cfg) != SDS_PPO:
        raise ValueError('v0.5-r2 PPO configuration must equal the frozen SDS 2.5 values')
    return cfg


def control_samples(num_envs, cfg=None):
    """Control samples produced by one complete update with ``num_envs``."""
    cfg = validate_config(cfg)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    return cfg.rollout * num_envs


def physics_substeps(control_samples_, *, substeps=PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE):
    """Physics substeps for a control-sample count: strictly ``4 * control``."""
    if isinstance(control_samples_, bool) or not isinstance(control_samples_, int) or control_samples_ < 0:
        raise ValueError('control_samples must be a non-negative integer')
    if substeps != PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE:
        raise ValueError('the v0.5-r2 decimation is fixed at 4')
    return control_samples_ * PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE


def explained_variance(values, returns):
    """``1 - var(return - value) / var(return)`` in float64.

    Returns ``(value, None)`` or ``(None, reason)``.  A zero or non-finite
    denominator is reported as null with its reason instead of a fabricated 0
    (SDS 7.6).
    """
    values = torch.as_tensor(values).detach().to(torch.float64).reshape(-1)
    returns = torch.as_tensor(returns).detach().to(torch.float64).reshape(-1)
    if values.numel() != returns.numel() or values.numel() == 0:
        return None, 'empty_or_mismatched_samples'
    if not (torch.isfinite(values).all() and torch.isfinite(returns).all()):
        return None, 'nonfinite_value_or_return'
    variance_return = float(torch.var(returns, unbiased=False))
    if variance_return == 0.:
        return None, 'zero_return_variance'
    if not math.isfinite(variance_return):
        return None, 'nonfinite_return_variance'
    variance_error = float(torch.var(returns - values, unbiased=False))
    return 1. - variance_error / variance_return, None


def _summary(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return dict(min=float(tensor.min()), mean=float(tensor.mean()),
                max=float(tensor.max()), count=int(tensor.numel()))


def update_math(*, model, optimizer, cfg, storage, device, learning_state=None,
                notify=None):
    """Verbatim v0.3 ``update`` loop plus read-only per-minibatch measurement.

    Every statement of ``v03/ppo.py::PPO.update`` is preserved in the same order
    with the same tensors and the same RNG use; measurement happens after the
    existing operations and only reads detach()ed scalars.  The r2 rollout's
    ``route_error`` key is sliced with the other observation keys and reaches
    ``action_distribution(obs)``/``evaluate(obs)`` unchanged (SDS 4.4).

    Returns ``(metrics, diagnostics)``.  ``metrics`` matches the v0.3 return keys
    exactly (without its rollout metrics); ``diagnostics`` holds the SDS 7.6
    min/mean/max fields, the full per-minibatch rows and the SDS 4.8 route
    aggregates.
    """
    cfg = cfg if cfg is not None else PPOConfig()
    state = learning_state if learning_state is not None else dict(
        optimizer_initialized=False, completed_ppo_updates=0, optimizer_steps=0)

    def send(event, **payload):
        if notify is not None:
            notify(event, **payload)

    c = cfg
    s = storage
    if s is None:
        raise RuntimeError('Missing rollout')
    batch = len(s['advantage'])
    if batch % c.minibatches:
        raise ValueError('Rollout not divisible into minibatches')
    observation = s.get('obs')
    if not isinstance(observation, dict):
        raise ValueError('update_math: rollout storage is missing its obs dict')
    route = validate_route_observation(observation, consumer='PPO.update',
                                       row_count=batch)
    minibatch_rows_expected = batch // c.minibatches
    result = []
    rows = []
    clipped_samples = 0
    total_samples = 0
    # The durable owner acknowledges the uncertain-optimizer boundary before even
    # adaptive LR changes Adam param_groups; exceptions abort here.
    send('optimizer_started', expected_optimizer_steps=c.epochs * c.minibatches)
    for epoch in range(c.epochs):
        for minibatch, idx in enumerate(torch.randperm(batch, device=device).chunk(c.minibatches)):
            obs = {k: v[idx] for k, v in s['obs'].items()}
            validate_route_observation(obs, consumer='PPO.update.minibatch',
                                       row_count=minibatch_rows_expected)
            dist = model.action_distribution(obs)
            value = model.evaluate(obs).squeeze(-1)
            with torch.no_grad():
                kl = (torch.log(dist.stddev / s['stds'][idx])
                      + (s['stds'][idx] ** 2 + (s['means'][idx] - dist.mean) ** 2)
                      / (2 * dist.stddev ** 2) - .5).sum(-1).mean()
                finite('KL', kl)
                lr = optimizer.param_groups[0]['lr']
                if kl > 2 * c.desired_kl:
                    lr = max(c.min_lr, lr / 1.5)
                elif 0 < kl < c.desired_kl / 2:
                    lr = min(c.max_lr, lr * 1.5)
                for group in optimizer.param_groups:
                    group['lr'] = lr
            ratio = torch.exp(dist.log_prob(s['actions'][idx]).sum(-1) - s['logp'][idx])
            policy = torch.maximum(-s['advantage'][idx] * ratio,
                                   -s['advantage'][idx] * ratio.clamp(1 - c.clip, 1 + c.clip)).mean()
            clipped = s['value'][idx] + (value - s['value'][idx]).clamp(-c.clip, c.clip)
            value_loss = torch.maximum((value - s['returns'][idx]) ** 2,
                                       (clipped - s['returns'][idx]) ** 2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy + c.value_coef * value_loss - c.entropy * entropy
            finite('loss', loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    finite(f'gradient:{name}', parameter.grad)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
            finite('gradient norm', grad)
            optimizer.step()
            state['optimizer_steps'] += 1
            state['optimizer_initialized'] = True
            model.constrain_std()
            send('optimizer_step_completed', epoch_index=epoch + 1,
                 minibatch_index=minibatch + 1)
            # ---- measurement only (no tensor feeding back into the math) ----
            with torch.no_grad():
                clipped_in_minibatch = int(((ratio.detach() - 1).abs() > c.clip).sum())
                samples_in_minibatch = idx.numel()
                std = model.std.detach().reshape(-1)
                rows.append(dict(
                    epoch=epoch + 1, minibatch=minibatch + 1,
                    kl=float(kl), lr=float(lr),
                    grad_norm_preclip=float(grad),
                    clip_fraction=clipped_in_minibatch / samples_in_minibatch,
                    loss=float(loss.detach()), policy_loss=float(policy.detach()),
                    value_loss=float(value_loss.detach()), entropy=float(entropy.detach()),
                    std_min=float(std.min()), std_max=float(std.max()),
                    std_touch_min=int((std <= STD_MIN + STD_TOUCH_TOLERANCE).sum()),
                    std_touch_max=int((std >= STD_MAX - STD_TOUCH_TOLERANCE).sum()),
                    std_dimensions=int(std.numel()),
                    route_samples=int(obs[ROUTE_ERROR_KEY].shape[0]),
                ))
            clipped_samples += int(((ratio.detach() - 1).abs() > c.clip).sum())
            total_samples += idx.numel()
            result.append([float(loss.detach()), float(policy.detach()), float(value_loss.detach()),
                           float(entropy.detach()), float(kl), float(grad)])
    state['completed_ppo_updates'] += 1
    send('update_completed', optimizer_steps_in_update=c.epochs * c.minibatches)
    values = np.mean(result, axis=0)
    metrics = dict(zip(('loss', 'policy_loss', 'value_loss', 'entropy', 'kl', 'gradient_norm'),
                       values.tolist()))
    std = model.std.detach()
    metrics.update(clip_fraction=clipped_samples / total_samples,
                   lr=optimizer.param_groups[0]['lr'], std_mean=float(std.mean()),
                   std_min=float(std.min()), std_max=float(std.max()),
                   **{f'std_{i}': float(v) for i, v in enumerate(std)})
    diagnostics = measure(minibatch_rows=rows, storage=s, std=std, cfg=c,
                          num_envs=batch // c.rollout,
                          completed_update=state['completed_ppo_updates'],
                          route_error=route)
    return metrics, diagnostics


def measure(*, minibatch_rows, storage, std, cfg=None, num_envs=None,
            completed_update=None, route_error=None):
    """Build the SDS 7.6 diagnostic block from measured rows and the rollout."""
    cfg = cfg if cfg is not None else PPOConfig()
    if not minibatch_rows:
        raise ValueError('No minibatch measured')
    rows = list(minibatch_rows)
    kl = _summary([row['kl'] for row in rows])
    lr = _summary([row['lr'] for row in rows])
    grad = _summary([row['grad_norm_preclip'] for row in rows])
    clip = _summary([row['clip_fraction'] for row in rows])
    # SDS 7.6: the gradient-clipping ratio is the fraction of minibatches whose
    # pre-clip norm exceeded max_grad_norm (1.0), not a per-sample fraction.
    gradient_clip_ratio = sum(1 for row in rows
                              if row['grad_norm_preclip'] > cfg.max_grad_norm) / len(rows)
    values = torch.as_tensor(storage['value']).detach().to(torch.float64).reshape(-1)
    returns = torch.as_tensor(storage['returns']).detach().to(torch.float64).reshape(-1)
    variance, variance_reason = explained_variance(values, returns)
    std_tensor = torch.as_tensor(std).detach().to(torch.float64).reshape(-1)
    std_fields = {f'{i:02d}': float(v) for i, v in enumerate(std_tensor)}
    # The std bounds are SDS/spec constants (0.10/0.30), identical to the r1
    # configuration; reading them from the constants keeps this measurement
    # adaptor usable with the frozen v0.3 config in the equivalence test.
    touched_min = int((std_tensor <= STD_MIN + STD_TOUCH_TOLERANCE).sum())
    touched_max = int((std_tensor >= STD_MAX - STD_TOUCH_TOLERANCE).sum())
    if route_error is None:
        observation = storage.get('obs') if isinstance(storage, dict) else None
        if not isinstance(observation, dict):
            raise ValueError('measure: an r2 diagnostic requires the rollout route_error')
        route_error = validate_route_observation(observation, consumer='PPO.measure')
    route = route_diagnostics(route_error)
    diagnostics = dict(
        minibatch_count=len(rows),
        kl_min=kl['min'], kl_mean=kl['mean'], kl_max=kl['max'],
        lr_min=lr['min'], lr_mean=lr['mean'], lr_max=lr['max'],
        grad_norm_preclip_min=grad['min'], grad_norm_preclip_mean=grad['mean'],
        grad_norm_preclip_max=grad['max'],
        clip_fraction_min=clip['min'], clip_fraction_mean=clip['mean'],
        clip_fraction_max=clip['max'],
        gradient_clip_ratio=gradient_clip_ratio,
        value_mean=float(values.mean()), value_std=float(values.std(unbiased=False)),
        value_min=float(values.min()), value_max=float(values.max()),
        return_mean=float(returns.mean()), return_std=float(returns.std(unbiased=False)),
        return_min=float(returns.min()), return_max=float(returns.max()),
        explained_variance=variance,
        explained_variance_reason=variance_reason,
        std=std_fields, std_dimensions=int(std_tensor.numel()),
        std_touch_min_count=touched_min, std_touch_min_ratio=touched_min / std_tensor.numel(),
        std_touch_max_count=touched_max, std_touch_max_ratio=touched_max / std_tensor.numel(),
        std_touch_tolerance=STD_TOUCH_TOLERANCE,
        adam_steps=len(rows),
        control_samples=cfg.rollout * num_envs if num_envs is not None else None,
        physics_substeps=(physics_substeps(cfg.rollout * num_envs)
                          if num_envs is not None else None),
        completed_update=completed_update,
        route=route,
        route_error_contract=route_error_contract(),
    )
    codes = []
    if kl['max'] > MAX_KL_WARNING:
        codes.append('max_kl_above_0.10')
    if clip['mean'] > MEAN_CLIP_FRACTION_WARNING:
        codes.append('mean_clip_fraction_above_0.50')
    diagnostics['warning_codes'] = codes
    diagnostics['kl_warning'] = 'max_kl_above_0.10' in codes
    diagnostics['clip_fraction_warning'] = 'mean_clip_fraction_above_0.50' in codes
    diagnostics['minibatches'] = rows
    return diagnostics


class PPO(V03PPO):
    """v0.3 collect/measurement lineage with the explicit r2 PPO numbers.

    ``collect`` is a verbatim copy of ``v03/ppo.py``'s loop with strict
    ``route_error`` validation inserted *around* (never inside) the frozen
    operations.  ``update`` runs :func:`update_math`, which is a proven
    tensor-exact copy of the v0.3 loop.  Checkpoint save/load is owned by
    ``v05_r2/runner.py`` + ``v05_r2/recovery.py``.
    """
    def __init__(self, model, env, cfg=PPOConfig()):
        cfg = validate_config(cfg)
        # V03PPO.__init__ asserts the frozen v0.3 defaults, so the identical body
        # is installed here with the r2 numbers instead of calling it.
        self.model = model
        self.env = env
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.obs = None
        self.storage = None
        self.control_step_callback = None
        self.user_diagnostics_callback = None
        self.diagnostics_callback = self._capture_diagnostics
        self.update_callback = None
        self.learning_state = dict(optimizer_initialized=False,
                                   completed_ppo_updates=0, optimizer_steps=0)
        self.rollout_metrics = {}
        self.last_update_diagnostics = None
        self.update_history = []
        self.update_warnings = []
        self.minibatch_history = []
        self.last_route_diagnostics = None
        self._gate_open = 0
        self._gate_samples = 0

    # ------------------------------------------------------------ identity
    @property
    def control_samples_per_update(self):
        return control_samples(self.env.num_envs, self.cfg)

    @property
    def physics_substeps_per_update(self):
        return physics_substeps(self.control_samples_per_update)

    @property
    def learning(self):
        """schema-7 ``learning`` block (update/optimizer_step/control_samples/physics_substeps)."""
        update = self.learning_state['completed_ppo_updates']
        samples = control_samples(self.env.num_envs, self.cfg) * update
        return dict(update=update, optimizer_step=self.learning_state['optimizer_steps'],
                    control_samples=samples, physics_substeps=physics_substeps(samples))

    def _notify_update(self, event, **payload):
        """Emit the r2 learning identity to the durable ledger owner.

        The payload key set stays exactly the v0.3 one (``learning_state`` plus
        the event's own counts) because ``v05_r2/recovery.py`` validates complete
        payload sets; route diagnostics travel in this module's own containers
        and ``update()`` return value, never through the callback.
        """
        if self.update_callback is not None:
            self.update_callback(event, dict(learning_state=self.learning, **payload))

    # ----------------------------------------------------------- lifecycle
    def reset(self):
        super().reset()
        self._gate_open = 0
        self._gate_samples = 0
        self.last_route_diagnostics = None
        validate_route_observation(self.obs, consumer='PPO.reset')

    def collect(self):
        """Verbatim v0.3 rollout plus the r2 five-key route packaging (SDS 4.4)."""
        if self.obs is None:
            self.reset()
        validate_route_observation(self.obs, consumer='PPO.collect')
        c = self.cfg
        n = self.env.num_envs
        device = self.env.device
        obs = {k: torch.empty((c.rollout, *v.shape), dtype=v.dtype, device=device)
               for k, v in self.obs.items()}
        fields = {k: torch.empty((c.rollout, n), device=device)
                  for k in ('logp', 'value', 'reward', 'terminated', 'done', 'next_value')}
        actions = torch.empty((c.rollout, n, 12), device=device)
        means = torch.empty_like(actions)
        stds = torch.empty_like(actions)
        records = []
        reward_sums = {}
        self._route_samples = dict(abs_y=[], normalized_abs_y=[], saturation=[],
                                   heading_deg=[], exact=True)
        self.model.train()
        # Exactly once before rollout; frozen throughout all 20 Adam minibatches.
        self.model.update_critic_statistics(self.obs)
        for t in range(c.rollout):
            with torch.no_grad():
                for key, value in self.obs.items():
                    obs[key][t].copy_(value)
                dist = self.model.action_distribution(self.obs)
                action = dist.sample()
                finite('actions', action)
                value = self.model.evaluate(self.obs).squeeze(-1)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                validate_route_observation(next_obs, consumer='PPO.collect.step')
                self._accumulate_route(info)
                if self.control_step_callback is not None:
                    self.control_step_callback(dict(step_index=t + 1, num_envs=n,
                        physics_samples_increment=n, info=info))
                if self.diagnostics_callback is not None:
                    self.diagnostics_callback(info)
                terminal_obs = info.get('terminal_observation', next_obs)
                validate_route_observation(terminal_obs,
                                           consumer='PPO.collect.terminal_observation')
                next_value = self.model.evaluate(terminal_obs).squeeze(-1)
                for key, val in [('logp', dist.log_prob(action).sum(-1)), ('value', value),
                        ('reward', reward), ('terminated', terminated),
                        ('done', terminated | truncated), ('next_value', next_value)]:
                    fields[key][t].copy_(val)
                actions[t].copy_(action)
                means[t].copy_(dist.mean)
                stds[t].copy_(dist.stddev)
                records.extend(info.get('episodes', []))
                self.obs = next_obs
                for key, val in info.get('reward_terms', {}).items():
                    reward_sums[key] = reward_sums.get(key, 0.) + val
        advantage = torch.zeros_like(fields['reward'])
        last = torch.zeros(n, device=device)
        for t in reversed(range(c.rollout)):
            delta = (fields['reward'][t] + c.gamma * (1 - fields['terminated'][t])
                     * fields['next_value'][t] - fields['value'][t])
            last = delta + c.gamma * c.gae_lambda * (1 - fields['done'][t]) * last
            advantage[t] = last
        returns = advantage + fields['value']
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        self.storage = dict(obs={k: v.flatten(0, 1) for k, v in obs.items()},
            actions=actions.flatten(0, 1), means=means.flatten(0, 1), stds=stds.flatten(0, 1),
            advantage=advantage.flatten(), returns=returns.flatten(),
            **{k: v.flatten() for k, v in fields.items()})
        # SDS 4.4: the rollout buffer must be exactly [rollout*num_envs, 3].
        route = validate_route_observation(self.storage['obs'], consumer='PPO.collect.storage',
                                           row_count=c.rollout * n)
        self.last_route_diagnostics = self._route_aggregate_from_samples(route)
        self.rollout_metrics = {key: val / c.rollout for key, val in reward_sums.items()}
        self.rollout_metrics.update(
            map_valid_fraction=float(self.obs['heightmap'][:, 1].mean()),
            map_age_mean=float(self.obs['heightmap'][:, 2].mean()))
        return records

    def _accumulate_route(self, info):
        """Accumulate exact per-step route evidence (SDS 4.8) from one step.

        The environment step's ``route_error_terminal``/``route_error`` is the
        pre-reset frame for that control step; the optional ``diagnostics`` block
        carries the exact ``local_position`` and world ``heading``, which makes
        ``mean(|local_y|)`` exact instead of the clipped reconstruction.
        """
        if not isinstance(info, dict):
            return
        route = info.get('route_error_terminal', info.get('route_error'))
        diagnostics = info.get('diagnostics')
        if route is None and isinstance(diagnostics, dict):
            route = diagnostics.get('route_error')
        if route is None:
            # SDS 4.5: the pre-reset terminal observation is the same frame's
            # route and is the fallback when the env does not duplicate it.
            terminal = info.get('terminal_observation')
            if isinstance(terminal, dict):
                route = terminal.get(ROUTE_ERROR_KEY)
        if route is None:
            return
        route = validate_route_tensor(route, consumer='PPO.collect.route_diagnostics')
        route = route.detach().to(torch.float64)
        normalized = route[:, 0].abs()
        heading = torch.rad2deg(torch.atan2(route[:, 1], route[:, 2]).abs())
        if isinstance(diagnostics, dict) and torch.is_tensor(diagnostics.get('local_position')):
            abs_y = diagnostics['local_position'][:, 1].detach().to(torch.float64).abs()
            saturation = (abs_y >= ROUTE_SATURATION_M).to(torch.float64)
        else:
            abs_y = normalized * ROUTE_SCALE_M
            saturation = (normalized >= ROUTE_CLIP - 1e-6).to(torch.float64)
            self._route_samples['exact'] = False
        self._route_samples['abs_y'].append(abs_y)
        self._route_samples['normalized_abs_y'].append(normalized)
        self._route_samples['saturation'].append(saturation)
        self._route_samples['heading_deg'].append(heading)

    def _route_aggregate_from_samples(self, fallback_route):
        samples = getattr(self, '_route_samples', None)
        if not samples or not samples['abs_y']:
            return route_diagnostics(fallback_route)
        return route_aggregate(
            abs_y=torch.cat(samples['abs_y']),
            normalized_abs_y=torch.cat(samples['normalized_abs_y']),
            saturation=torch.cat(samples['saturation']),
            heading_deg=torch.cat(samples['heading_deg']),
            exact=bool(samples['exact']))

    def _capture_diagnostics(self, info):
        gate = info.get('gate_open') if isinstance(info, dict) else None
        if gate is not None:
            tensor = torch.as_tensor(gate).detach()
            self._gate_open += int(tensor.sum().item())
            self._gate_samples += int(tensor.numel())
        if self.user_diagnostics_callback is not None:
            self.user_diagnostics_callback(info)

    def update(self):
        if self.storage is None:
            raise RuntimeError('Missing rollout')
        metrics, diagnostics = update_math(
            model=self.model, optimizer=self.optimizer, cfg=self.cfg,
            storage=self.storage, device=self.env.device,
            learning_state=self.learning_state, notify=self._notify_update)
        gate_rate = (self._gate_open / self._gate_samples) if self._gate_samples else None
        # SDS 4.8: replace the measurement fallback with the accumulated exact
        # block (the learning math above is untouched).
        diagnostics['route'] = self._route_aggregate_from_samples(
            self.storage['obs'][ROUTE_ERROR_KEY])
        map_fields = ('map_valid_fraction', 'map_age_mean')
        diagnostics.update(
            gate_open_rate=gate_rate, gate_samples=self._gate_samples,
            control_samples=self.control_samples_per_update,
            physics_substeps=self.physics_substeps_per_update,
            reward_terms={key: value for key, value in self.rollout_metrics.items()
                          if key not in map_fields},
            map_metrics={key: value for key, value in self.rollout_metrics.items()
                         if key in map_fields},
            update=self.learning_state['completed_ppo_updates'],
        )
        self.last_route_diagnostics = dict(diagnostics['route'])
        self._gate_open = 0
        self._gate_samples = 0
        self.last_update_diagnostics = diagnostics
        self.update_warnings.extend(dict(update=diagnostics['update'], code=code)
                                    for code in diagnostics['warning_codes'])
        for code in diagnostics['warning_codes']:
            warnings.warn(f'v0.5-r2 PPO update {diagnostics["update"]}: {code}',
                          RuntimeWarning, stacklevel=2)
        # SDS 7.6: keep the first 20 updates complete, later updates aggregate.
        keep_rows = (diagnostics['update'] <= FULL_MINIBATCH_UPDATES
                     or len(self.update_history) < FULL_MINIBATCH_UPDATES)
        if keep_rows:
            self.minibatch_history.append(diagnostics['minibatches'])
        record = {key: value for key, value in diagnostics.items() if key != 'minibatches'}
        if keep_rows:
            record['minibatches'] = diagnostics['minibatches']
        self.update_history.append(record)
        result = dict(metrics)
        result.update(self.rollout_metrics)
        result.update({key: value for key, value in diagnostics.items() if key != 'minibatches'})
        return result

    def diagnostics_jsonl_record(self):
        """One JSON-safe aggregate row for ``updates.jsonl`` (no tensors)."""
        if self.last_update_diagnostics is None:
            raise ValueError('No completed update to record')
        row = {key: value for key, value in self.last_update_diagnostics.items()
               if key not in ('std', 'minibatches', 'reward_terms', 'map_metrics',
                              'route', 'route_error_contract')}
        row['std'] = dict(self.last_update_diagnostics['std'])
        row['ppo'] = {key: value for key, value in row.items()
                      if key.startswith(('kl_', 'lr_', 'grad_norm_', 'clip_fraction_',
                                         'gradient_clip', 'value_', 'return_',
                                         'explained_', 'std_'))}
        row['reward_terms'] = dict(self.last_update_diagnostics['reward_terms'])
        # SDS 4.8: route/* is a read-only diagnostic namespace (no namespace in
        # the JSONL field name; the telemetry layer prefixes ``route/``).
        row['route'] = dict(self.last_update_diagnostics['route'])
        return row


__all__ = [
    'ADAM_STEPS_PER_UPDATE', 'CONTROL_SAMPLES_PER_UPDATE', 'FULL_MINIBATCH_UPDATES',
    'MAX_KL_WARNING', 'MEAN_CLIP_FRACTION_WARNING', 'PPO', 'PPOConfig',
    'PHYSICS_SUBSTEPS_PER_CONTROL_SAMPLE', 'ROUTE_CLIP', 'ROUTE_ERROR_DIM',
    'ROUTE_ERROR_KEY', 'ROUTE_OBSERVATION_KEYS', 'ROUTE_SATURATION_M', 'ROUTE_SCALE_M',
    'SDS_PPO', 'STD_TOUCH_TOLERANCE', 'control_samples', 'explained_variance', 'measure',
    'physics_substeps', 'route_aggregate', 'route_diagnostics', 'route_error_contract',
    'update_math', 'validate_config', 'validate_route_observation',
    'validate_route_tensor',
]
