"""S2.1 training environment seam.

The real 1024-env mjlab assembly is intentionally injected later; this module
provides the CPU fake vector environment used by the training-stack contract
tests and the S2.1 role/curriculum plan.
"""
from __future__ import annotations

import math
import numpy as np
import torch


ROLE_ORDER = ('flat_walk', 'flat_stop', 'stairs_walk', 'platform_walk',
              'l40_milestone')
ROLE_WEIGHTS = (0.20, 0.20, 0.40, 0.10, 0.10)
SPEEDS = (0.3, 0.5, 0.6)
PLATFORM_WIDTHS = (0.8, 4.0)
STAIR_COUNTS = (10, 20)
PLATFORM_HEIGHTS = (2.0, 4.0)
POST_DISTANCE = 5.0
COURSES = ('S2.1', 'S2.2', 'S2.3')
COURSE_CAPS = dict(S2_1=800_000_000, S2_2=1_200_000_000,
                   S2_3=1_200_000_000)


def largest_remainder(total, weights=ROLE_WEIGHTS, order=ROLE_ORDER):
    """Integer allocation with fixed-order tie-breaking (Q: role rule)."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError('total must be a nonnegative integer')
    if len(weights) != len(order) or not weights:
        raise ValueError('weights/order length mismatch')
    if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError('weights must be nonnegative and sum to 1')
    raw = [float(total) * float(w) for w in weights]
    base = [int(math.floor(v)) for v in raw]
    remaining = int(total) - sum(base)
    remainders = sorted(((raw[i] - base[i], -i, i) for i in range(len(raw))),
                        reverse=True)
    for _frac, _neg, index in remainders[:remaining]:
        base[index] += 1
    return {name: base[i] for i, name in enumerate(order)}


def _balanced(count, variants):
    """Deterministic round-robin balance; counts differ by at most one."""
    if count < 0 or not variants:
        raise ValueError('invalid balanced allocation')
    return [dict(variants[i % len(variants)]) for i in range(count)]


def _flat_variants(role):
    if role == 'flat_walk':
        return [dict(task='flat_walk', speed=s, stop_mode='none')
                for s in SPEEDS]
    return [dict(task='flat_stop', speed=s, stop_mode=mode)
            for s in SPEEDS for mode in ('instant', 'ramp_2s')]


def _stairs_variants():
    return [dict(task='stairs_walk', stair_count=levels,
                 platform_width=width, speed=speed, post_distance=POST_DISTANCE)
            for levels in STAIR_COUNTS for width in PLATFORM_WIDTHS
            for speed in SPEEDS]


def _platform_variants():
    return [dict(task='platform_walk', platform_height=height,
                 platform_width=width, speed=speed,
                 post_distance=POST_DISTANCE)
            for height in PLATFORM_HEIGHTS for width in PLATFORM_WIDTHS
            for speed in SPEEDS]


def _l40_variants():
    return [dict(task='legacy_l40', stair_count=40, speed=speed,
                 post_distance=POST_DISTANCE) for speed in SPEEDS]


def build_role_plan(num_envs=1024, course='S2.1'):
    if course not in COURSES:
        raise ValueError(f'unknown S2 course {course!r}')
    course_key = course.replace('.', '_')
    counts = largest_remainder(num_envs)
    plan = []
    for role in ROLE_ORDER:
        count = counts[role]
        if role in ('flat_walk', 'flat_stop'):
            variants = _balanced(count, _flat_variants(role))
        elif role == 'stairs_walk':
            variants = []
            if course_key == 'S2_1':
                for offset, levels in enumerate(STAIR_COUNTS):
                    level_count = count // 2 + (1 if offset < count % 2 else 0)
                    level_variants = [dict(task='stairs_walk', stair_count=levels,
                                           platform_width=width, speed=speed,
                                           post_distance=POST_DISTANCE)
                                      for width in PLATFORM_WIDTHS for speed in SPEEDS]
                    variants.extend(_balanced(level_count, level_variants))
            elif course_key == 'S2_2':
                main = [dict(task='stairs_walk', stair_count=40,
                             platform_width=width, speed=speed,
                             post_distance=distance)
                        for width in PLATFORM_WIDTHS for speed in SPEEDS
                        for distance in (5.0, 10.0)]
                variants = _balanced(count, main)
            else:
                main = [dict(task='stairs_stop', stair_count=40,
                             platform_width=width, speed=speed,
                             post_distance=distance, stop_mode=mode,
                             ramp_steps=(100 if mode == 'ramp_2s' else 0),
                             hold_steps=1000)
                        for width in PLATFORM_WIDTHS for speed in SPEEDS
                        for mode in ('instant', 'ramp_2s')
                        for distance in (3.0, 10.0)]
                variants = _balanced(count, main)
        elif role == 'platform_walk':
            variants = []
            if course_key == 'S2_1':
                for offset, height in enumerate(PLATFORM_HEIGHTS):
                    height_count = count // 2 + (1 if offset < count % 2 else 0)
                    height_variants = [dict(task='platform_walk', platform_height=height,
                                            platform_width=width, speed=speed,
                                            post_distance=POST_DISTANCE)
                                       for width in PLATFORM_WIDTHS for speed in SPEEDS]
                    variants.extend(_balanced(height_count, height_variants))
            elif course_key == 'S2_2':
                variants = _balanced(count, [
                    dict(task='platform_walk', platform_height=8.0,
                         platform_width=width, speed=speed, post_distance=10.0)
                    for width in PLATFORM_WIDTHS for speed in SPEEDS])
            else:
                variants = _balanced(count, [
                    dict(task='platform_stop', platform_height=8.0,
                         platform_width=width, speed=speed,
                         post_distance=distance, stop_mode=mode,
                         ramp_steps=(100 if mode == 'ramp_2s' else 0),
                         hold_steps=1000)
                    for width in PLATFORM_WIDTHS for speed in SPEEDS
                    for mode in ('instant', 'ramp_2s')
                    for distance in (3.0, 10.0)])
        else:
            if course_key == 'S2_1':
                variants = _balanced(count, _l40_variants())
            else:
                variants = _balanced(count, [
                    dict(task='stairs_walk', stair_count=levels,
                         platform_width=width, speed=speed,
                         post_distance=POST_DISTANCE)
                    for levels in STAIR_COUNTS for width in PLATFORM_WIDTHS
                    for speed in SPEEDS])
        if course_key == 'S2_2' and role == 'stairs_walk':
            rng = np.random.default_rng(42)
            for item in variants:
                item['post_distance'] = float(rng.uniform(5.0, 10.0))
        elif course_key == 'S2_3' and role in ('stairs_walk', 'platform_walk'):
            rng = np.random.default_rng(42 + (1 if role == 'stairs_walk' else 2))
            for item in variants:
                item['post_distance'] = float(rng.uniform(3.0, 10.0))
        for item in variants:
            plan.append(dict(role=role, **item))
    return dict(num_envs=num_envs, course=course, counts=counts,
                order=list(ROLE_ORDER), envs=plan)


class S2FakeTrainEnv:
    """Deterministic CPU vector env with the S2 observation shapes."""

    def __init__(self, *, num_envs=8, seed=42, role_plan=None):
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.role_plan = role_plan or build_role_plan(num_envs)
        self.step_count = 0
        self._rng = np.random.default_rng(int(seed))
        self._obs = None

    def _observation(self):
        n = self.num_envs
        proprio = torch.zeros(n, 45)
        proprio[:, 6] = 0.3
        proprio[:, 5] = -1.0
        history = proprio[:, None, :].expand(n, 5, 45).clone()
        heightmap = torch.zeros(n, 3, 40, 30)
        heightmap[:, 1] = 1.0
        route_error = torch.zeros(n, 3)
        route_error[:, 2] = 1.0
        privileged = torch.zeros(n, 35)
        return dict(proprio=proprio, history=history, heightmap=heightmap,
                    route_error=route_error, privileged=privileged)

    def reset(self):
        self.step_count = 0
        self._obs = self._observation()
        return {key: value.clone() for key, value in self._obs.items()}

    def step(self, actions):
        action = torch.as_tensor(actions, dtype=torch.float32)
        if action.ndim != 2 or action.shape != (self.num_envs, 12):
            raise ValueError(f'actions must be [{self.num_envs},12]')
        self.step_count += 1
        reward = -torch.sum(action ** 2, dim=1) * 0.01
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        # deterministic small termination fraction for accumulator tests
        done[self.step_count % max(1, self.num_envs // 4) :: max(1, self.num_envs // 4)] = True
        return self._obs, reward, done, dict(
            timeout=False, failure=False, s2_learning_samples=self.num_envs)


def validate_role_plan(plan):
    if plan.get('num_envs') not in (512, 1024):
        raise ValueError('S2 role plan requires 512/1024 envs')
    if plan.get('course', 'S2.1') not in COURSES:
        raise ValueError('unknown S2 course')
    counts = plan['counts']
    if sum(counts.values()) != plan['num_envs']:
        raise ValueError('role counts do not sum to num_envs')
    for role in ROLE_ORDER:
        if role not in counts:
            raise ValueError(f'missing role {role}')
    if counts['stairs_walk'] < 1 or counts['flat_walk'] < 1:
        raise ValueError('S2.1 requires flat and stairs roles')
    return True
