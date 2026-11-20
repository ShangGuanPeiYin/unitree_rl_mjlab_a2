"""S1 physical rollout collection and immutable first-terminal batch evaluation.

The driver must obtain an external budget reservation before calling either
operation, and settle all advanced slots, including completed-slot waiting.
"""
from __future__ import annotations

from copy import deepcopy
import torch

from . import spec


def collect_rollout(env, teacher, student):
    """Cache native student inputs and 12+1 teacher labels at the same states."""
    teacher_obs, student_obs = env.observation_pair()
    storage = {key: torch.empty((spec.ROLLOUT_STEPS, *value.shape),
                dtype=value.dtype, device=value.device) for key, value in student_obs.items()}
    device = next(iter(student_obs.values())).device
    means = torch.empty(spec.ROLLOUT_STEPS, env.num_envs, 12, device=device)
    values = torch.empty(spec.ROLLOUT_STEPS, env.num_envs, 1, device=device)
    episode_records = []
    group_metadata = dict(
        tile_ids=torch.empty(spec.ROLLOUT_STEPS, env.num_envs, dtype=torch.long, device=device),
        flat_stop=torch.empty(spec.ROLLOUT_STEPS, env.num_envs, dtype=torch.bool, device=device),
        cruise_speed=torch.empty(spec.ROLLOUT_STEPS, env.num_envs, device=device))
    with torch.no_grad():
        for step in range(spec.ROLLOUT_STEPS):
            teacher_obs, student_obs = env.observation_pair()
            group_metadata['tile_ids'][step].copy_(env.tile_ids)
            group_metadata['flat_stop'][step].copy_(env.flat_stop_mask)
            group_metadata['cruise_speed'][step].copy_(env.start_vx)
            label_mean, label_value = teacher.labels(teacher_obs)
            student_mean = student(student_obs)
            for key, value in student_obs.items():
                storage[key][step].copy_(value)
            means[step].copy_(label_mean)
            values[step].copy_(label_value)
            actions = torch.where(env.teacher_control[:, None], label_mean, student_mean)
            _, _, _, _, info = env.step(actions)
            for record in info['episodes']:
                record['operation_course'] = env.course
            episode_records.extend(info['episodes'])
    return dict(observations=storage, teacher_mean=means, teacher_value=values,
                episodes=episode_records, group_metadata=group_metadata,
                control_steps=spec.ROLLOUT_STEPS,
                physical_control_samples=env.num_envs * spec.ROLLOUT_STEPS)


def evaluate_batch(env, policy, *, policy_map, max_steps, audit_teacher=None, audit_every=50):
    """Run one physical batch to the last first-terminal, retaining padding cost."""
    if policy_map not in ('teacher', 'student'):
        raise ValueError('explicit teacher/student input grid selection is required')
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('positive hard timeout required')
    records = [None] * env.num_envs
    evidence = [None] * env.num_envs
    audit_samples = []
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            teacher_obs, student_obs = env.observation_pair()
            if audit_teacher is not None and (step - 1) % audit_every == 0:
                active = torch.tensor([record is None for record in records],
                    device=student_obs['proprio'].device, dtype=torch.bool)
                mean, value = audit_teacher.labels(teacher_obs)
                audit_samples.append(dict(actor_index=step - 1,
                    slots=active.nonzero().flatten().cpu(),
                    student_observation={key: data[active].detach().cpu() for key, data in student_obs.items()},
                    teacher_observation={key: data[active].detach().cpu() for key, data in teacher_obs.items()},
                    teacher_mean=mean[active].detach().cpu(), teacher_value=value[active].detach().cpu()))
            actions = policy(teacher_obs if policy_map == 'teacher' else student_obs)
            _, _, _, _, info = env.step(actions)
            for record in info['episodes']:
                index = record['env_id']
                if records[index] is None:
                    records[index] = deepcopy(record)
                    evidence[index] = deepcopy(info.get('v09_evidence', {}).get(index))
            if all(record is not None for record in records):
                return dict(records=records, evidence=evidence, audit_samples=audit_samples, control_steps=step,
                            physical_control_samples=env.num_envs * step)
    raise RuntimeError('batch did not produce every first-terminal by its hard timeout')


class ValueMoments:
    """FP64 mergeable population moments from the independent calibration set."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        values = values.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError('calibration values must be finite')
        size = values.numel()
        if not size:
            return
        mean = float(values.mean())
        m2 = float((values - mean).square().sum())
        total = self.count + size
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * size / total
        self.mean += delta * size / total
        self.count = total

    @property
    def value_scale(self):
        if not self.count:
            raise ValueError('cannot calibrate from an empty value stream')
        return max((self.m2 / self.count) ** .5, 1.)


def calibrate_batch(env, teacher, *, max_steps, moments):
    """Sample values on first physical episodes only, billing all waiting slots."""
    records = [None] * env.num_envs
    value_chunks, slot_chunks = [], []
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            teacher_obs, _ = env.observation_pair()
            means, values = teacher.labels(teacher_obs)
            active = torch.tensor([record is None for record in records],
                                  device=values.device, dtype=torch.bool)
            active_values = values[active].detach().cpu()
            moments.update(active_values)
            value_chunks.append(active_values)
            slot_chunks.append(active.nonzero().flatten().cpu())
            _, _, _, _, info = env.step(means)
            for record in info['episodes']:
                index = record['env_id']
                if records[index] is None:
                    records[index] = deepcopy(record)
            if all(record is not None for record in records):
                return dict(records=records, control_steps=step,
                            physical_control_samples=env.num_envs * step,
                            calibration_values=torch.cat(value_chunks),
                            value_case_slots=torch.cat(slot_chunks))
    raise RuntimeError('calibration batch missing terminal evidence at hard timeout')
