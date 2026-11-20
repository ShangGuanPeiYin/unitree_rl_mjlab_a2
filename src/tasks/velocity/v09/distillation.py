"""S1 action/value distillation with logical-batch gradient accumulation."""
from __future__ import annotations

import math
import torch

from . import spec


def distill_minibatch(model, optimizer, observations, teacher_mean, teacher_value,
                      *, value_scale, microbatch=spec.MICROBATCH_MAX):
    """Apply one Adam step, normalizing every microbatch by total sample count.

    Teacher labels were sampled on the executed trajectory. Their scales and
    normalizers must already be frozen before calling this training operation.
    """
    if not math.isfinite(value_scale) or value_scale < 1:
        raise ValueError('frozen calibration value_scale must be finite and >= 1')
    if not isinstance(microbatch, int) or not 1 <= microbatch <= spec.MICROBATCH_MAX:
        raise ValueError('microbatch must be an integer in [1,512]')
    count = teacher_mean.shape[0]
    if count < 1 or teacher_mean.shape != (count, 12) or teacher_value.shape != (count, 1):
        raise ValueError('teacher labels must have shapes [B,12] and [B,1]')
    if any(value.shape[0] != count for value in observations.values()):
        raise ValueError('observation and teacher label sample counts differ')
    if teacher_mean.requires_grad or teacher_value.requires_grad:
        raise ValueError('teacher labels must be detached')
    if not torch.isfinite(teacher_mean).all() or not torch.isfinite(teacher_value).all():
        raise ValueError('teacher labels must be finite')
    optimizer.zero_grad(set_to_none=True)
    action_sum = value_sum = 0.0
    for start in range(0, count, microbatch):
        stop = min(start + microbatch, count)
        obs = {key: value[start:stop] for key, value in observations.items()}
        action_error = (model(obs) - teacher_mean[start:stop]).square().sum() / (count * 12)
        value_error = ((model.evaluate(obs) - teacher_value[start:stop]) / value_scale).square().sum() / count
        loss = action_error + 0.1 * value_error
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('nonfinite S1 distillation loss')
        loss.backward()
        action_sum += float(action_error.detach())
        value_sum += float(value_error.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.MAX_GRAD_NORM,
                                         error_if_nonfinite=True)
    optimizer.step()
    return {'action_loss': action_sum, 'value_loss': value_sum,
            'loss': action_sum + 0.1 * value_sum, 'gradient_norm': float(norm),
            'samples': count, 'adam_steps': 1}


def train_iteration(model, optimizer, observations, teacher_mean, teacher_value,
                    *, value_scale, generator, microbatch=spec.MICROBATCH_MAX):
    """Train exactly five epochs of four logical batches from 48 physical steps."""
    if teacher_mean.ndim != 3 or teacher_mean.shape[0] != spec.ROLLOUT_STEPS:
        raise ValueError('a complete 48-control-step rollout is required')
    time_steps, num_envs, action_dim = teacher_mean.shape
    if action_dim != 12 or teacher_value.shape != (time_steps, num_envs, 1):
        raise ValueError('rollout teacher label shape mismatch')
    count = time_steps * num_envs
    flattened = {}
    for key, value in observations.items():
        if value.shape[:2] != (time_steps, num_envs):
            raise ValueError('rollout observation time/environment axes differ')
        flattened[key] = value.flatten(0, 1)
    means = teacher_mean.flatten(0, 1)
    values = teacher_value.flatten(0, 1)
    totals = dict(action_loss=0., value_loss=0., loss=0., gradient_norm=0.)
    steps = 0
    for _ in range(spec.EPOCHS):
        permutation = torch.randperm(count, generator=generator, device=means.device)
        for ids in permutation.chunk(spec.MINIBATCHES):
            batch = {key: value[ids] for key, value in flattened.items()}
            result = distill_minibatch(model, optimizer, batch, means[ids], values[ids],
                                      value_scale=value_scale, microbatch=microbatch)
            for key in totals:
                totals[key] += result[key]
            steps += 1
    return {**{key: value / steps for key, value in totals.items()},
            'samples': count, 'adam_steps': steps}
