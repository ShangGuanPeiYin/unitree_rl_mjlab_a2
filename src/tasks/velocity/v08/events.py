"""v0.8 climb success, first-step and failure-mask semantics (SDS 6.2).

All tile-specific quantities are read from the v0.8 ``TerrainSpec`` table via
:mod:`src.tasks.velocity.v08.terrain`; every place the frozen R2 code used
``N * 0.20`` now uses ``N * current step depth`` (``N * d``).

Success (真实安全登顶) is exactly the frozen platform-contact definition:

* all four feet had a real safe top-platform support within the last 50
  control samples;
* a single foot needs 2 consecutive support samples;
* at least 2 feet support right now;
* body at least ``0.8 m`` past the top-platform front edge;
* ``|local_y| <= 0.20 m``;
* all four feet inside the platform's ``0.042 m`` safe inset;
* 5 consecutive control samples;
* no failure mask.

Failures are reported as individual masks (side / bounds / body / nonfoot
0.1 s / posture 70 deg / fall / contact overflow / nonfinite) so the runner can
separate "lateral departure" from "physical fall", and success + failure in the
same frame is resolved by failure priority in the caller.

Fall (Q002-2) is the frozen v0.2 *expected tread* criterion, not a recent support
surface and not an averaged foot plane::

    region = (N > 0) & (local_x >= 0)
    z_expected = min(max(floor(local_x / d) + 1, 0), N) * h
    fall = region & (local_z < z_expected - spec.FALL_DROP)   # strict "<"

with ``local_z`` relative to the current tile frame; the platform region uses
``z_expected = N*h`` and the flat tile / approach (``local_x < 0``) never
triggers it.  Equality never triggers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import spec as _spec
from .terrain import SAFE_INSET, build_terrain_specs

POSTURE_LIMIT = float(_spec.ATTITUDE_LIMIT_RADIANS)
BODY_HALF_WIDTH = float(_spec.STAIR_WIDTH) / 2.0
NONFOOT_LIMIT_S = float(_spec.NONFOOT_PERSIST_TIME)
SUPPORT_FORCE_GT = float(_spec.SUPPORT_FORCE_GT)
SINGLE_FOOT_FRAMES = int(_spec.SUCCESS_SINGLE_FOOT_FRAMES)
MIN_SUPPORT_FEET = int(_spec.SUCCESS_MIN_SUPPORT_FEET)
SUCCESS_FRAMES = int(_spec.SUCCESS_CONSECUTIVE_FRAMES)
SUPPORT_WINDOW = int(_spec.SUCCESS_RECENT_SUPPORT_STEPS)
PLATFORM_BODY_OFFSET = float(_spec.SUCCESS_BODY_PLATFORM_MARGIN)
LATERAL_LIMIT = float(_spec.SUCCESS_LATERAL_MAX)
#: Q002-2: the fall threshold is a required spec constant (0.50 m).  It is read
#: directly with no silent default so a missing value fails the import instead of
#: inventing a number.
FALL_DROP = float(_spec.FALL_DROP)
FLAT_EPISODE_STEPS = int(_spec.F0_EPISODE_STEPS)
CONTROL_DT = float(_spec.CONTROL_DT)
STAIR_TIMEOUT_BASE_M = float(_spec.STAIR_TIMEOUT_BASE_M)
STAIR_TIMEOUT_SECONDS = float(_spec.STAIR_TIMEOUT_SECONDS)
SAFE_INSET = float(SAFE_INSET)

#: Failure reason masks in the order their reason code is assigned (later wins,
#: exactly like the frozen lifecycle; ``timeout``/``success`` use codes 1/2).
FAILURE_NAMES = ("side", "bounds", "body", "nonfoot", "posture", "fall",
                 "contact_overflow", "nonfinite")
FAILURE_CODES = {name: 3 + index for index, name in enumerate(FAILURE_NAMES)}
REASON_NAMES = ("running", "timeout", "success") + FAILURE_NAMES
TIMEOUT_CODE = 1
SUCCESS_CODE = 2

def euler_heading(quat: torch.Tensor):
    """Roll/pitch/yaw of a ``wxyz`` quaternion, identical to the frozen contract.

    ``pitch`` is negated so that a positive rotation about ``+y`` (nose down)
    is reported as nose up, exactly like the frozen v0.2/v0.5 lifecycle.
    """
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = -torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to ``(-pi, pi]`` via ``atan2(sin, cos)``."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


_TABLE_CACHE: dict = {}


def _table(device):
    key = str(device)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        table = build_terrain_specs()
        n = torch.tensor([int(s.N) for s in table], device=device, dtype=torch.float32)
        h = torch.tensor([float(s.h) for s in table], device=device, dtype=torch.float32)
        d = torch.tensor([float(s.step_depth) for s in table], device=device,
                         dtype=torch.float32)
        bounds = torch.tensor([tuple(float(v) for v in s.valid_bounds_local)
                               for s in table], device=device, dtype=torch.float32)
        cached = (n, h, d, bounds)
        _TABLE_CACHE[key] = cached
    return cached


def tile_parameters(tile_ids: torch.Tensor):
    """Per-environment ``(N, h, d, bounds)`` of the assigned v0.8 tile."""
    n, h, d, bounds = _table(tile_ids.device)
    return n[tile_ids], h[tile_ids], d[tile_ids], bounds[tile_ids]


def staircase_expected_height(tile_ids: torch.Tensor, local_x: torch.Tensor):
    """Highest tread height the body should be above at ``local_x`` (SDS 6.2)."""
    n, h, d, _ = tile_parameters(tile_ids)
    steps = torch.floor(local_x / d.clamp_min(1e-6)) + 1.0
    return torch.minimum(steps.clamp_min(0.0), n) * h


def timeout_steps(tile_ids: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """Per-tile SDS 6.2 timeout, ``N * current step depth`` based.

    ``n_max = ceil((2*(2.2 + N*sqrt(h^2+d^2))/c + 10) / dt)``; flat tiles keep
    the 1500 control-step limit.
    """
    n, h, d, _ = tile_parameters(tile_ids)
    c = command.to(torch.float32).abs().clamp_min(0.01)
    diagonal = torch.sqrt(h.square() + d.square())
    seconds = 2.0 * (STAIR_TIMEOUT_BASE_M + n * diagonal) / c \
        + STAIR_TIMEOUT_SECONDS
    stair_steps = torch.ceil(seconds / CONTROL_DT).long()
    flat = n == 0
    return torch.where(flat, torch.full_like(stair_steps, FLAT_EPISODE_STEPS),
                       stair_steps)


def first_step_contact(top_forces, geom_tile, geom_step, tile_ids):
    """Per-foot ``> 5 N`` top-face contact with tread 1 of the assigned tile."""
    matched = ((geom_tile[None, :] == tile_ids[:, None]) & (geom_step == 1)[None, :])
    force = torch.where(matched[:, None, :], top_forces, -torch.inf)
    return (force.max(-1).values > SUPPORT_FORCE_GT)


def failure_masks(*, local, region, stairs, tile_ids, roll, pitch,
                  control_body_hit, body_link_force, body_link_mask,
                  nonfoot_duration, control_long_hit, contact_overflow,
                  nonfinite):
    """All v0.8 failure masks as separate boolean ``[B]`` tensors.

    ``local`` is the body position in the tile-local frame and ``region`` is
    ``stairs & local_x >= 0``; every mask is a boolean ``[B]`` tensor.
    """
    n, h, d, bounds = tile_parameters(tile_ids)
    # Q002-2 expected tread height in the current tile frame (not a support
    # plane); the strict "<" comparison means equality never triggers a fall.
    expected = torch.minimum((torch.floor(local[:, 0] / d) + 1.0).clamp_min(0.0), n) * h
    body_contact = control_body_hit | (
        body_link_force[:, body_link_mask].norm(dim=-1) > 0.0).any(-1)
    nonfoot = control_long_hit | (
        nonfoot_duration >= NONFOOT_LIMIT_S - 1e-6).any(-1)
    posture = (roll.abs() > POSTURE_LIMIT) | (pitch.abs() > POSTURE_LIMIT)
    side = region & (local[:, 1].abs() > BODY_HALF_WIDTH)
    fall = region & (local[:, 2] < expected - FALL_DROP)  # strict '<' (Q002-2)
    x0, x1, y0, y1 = bounds.unbind(-1)
    x0 = torch.where(stairs, x0, torch.full_like(x0, -2.0))
    x1 = torch.where(stairs, x1, torch.full_like(x1, 22.0))
    y0 = torch.where(stairs, y0, torch.full_like(y0, -4.0))
    y1 = torch.where(stairs, y1, torch.full_like(y1, 4.0))
    out_of_bounds = ((local[:, 0] < x0) | (local[:, 0] > x1)
                     | (local[:, 1] < y0) | (local[:, 1] > y1))
    masks = dict(
        side=side,
        bounds=out_of_bounds,
        body=body_contact,
        nonfoot=nonfoot,
        posture=posture,
        fall=fall,
        contact_overflow=contact_overflow.to(torch.bool),
        nonfinite=nonfinite.to(torch.bool),
    )
    return masks


def failure_reason(masks) -> torch.Tensor:
    """Failure reason codes (later masks win, like the frozen lifecycle)."""
    reason = torch.zeros_like(next(iter(masks.values())), dtype=torch.long)
    for name in FAILURE_NAMES:
        reason = torch.where(masks[name], torch.full_like(reason, FAILURE_CODES[name]),
                             reason)
    return reason


@dataclass(frozen=True)
class ClimbResult:
    success: torch.Tensor
    climbed: torch.Tensor
    current_support: torch.Tensor
    recent_support: torch.Tensor
    consecutive_steps: torch.Tensor
    first_step_acquired: torch.Tensor


class PlatformSupportState:
    """Debounced per-foot platform support + five-frame safe-arrival latch."""

    def __init__(self, num_envs, device="cpu"):
        self.last_surface = torch.full((num_envs, 4), -1, dtype=torch.long, device=device)
        self.last_tile = torch.full_like(self.last_surface, -1)
        self.support_count = torch.zeros_like(self.last_surface)
        self.last_support_step = torch.full_like(self.last_surface, -1000000)
        self.consecutive_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.climbed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.climb_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.current_support = torch.zeros_like(self.last_surface, dtype=torch.bool)

    def reset(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.last_surface.device)
        for value in (self.last_surface, self.last_tile):
            value[ids] = -1
        self.support_count[ids] = 0
        self.last_support_step[ids] = -1000000
        self.consecutive_steps[ids] = 0
        self.climbed[ids] = False
        self.climb_step[ids] = -1
        self.current_support[ids] = False

    def update(self, step, stairs, tile_ids, surface_ids, candidate, feet_safe,
               body_x, x_top, body_y, failure):
        """One control-sample advance; ``surface_ids`` identify the platform geom."""
        valid = (candidate & feet_safe & stairs[:, None] & (surface_ids >= 0))
        same = valid & (surface_ids == self.last_surface) & (tile_ids[:, None] == self.last_tile)
        self.support_count = torch.where(same, self.support_count + 1, valid.long())
        self.last_surface = torch.where(valid, surface_ids, -1)
        self.last_tile = torch.where(valid, tile_ids[:, None], -1)
        self.current_support = valid & (self.support_count >= SINGLE_FOOT_FRAMES)
        self.last_support_step = torch.where(self.current_support, step[:, None],
                                             self.last_support_step)
        recent = (step[:, None] - self.last_support_step) <= SUPPORT_WINDOW
        current = (stairs & (body_x >= x_top + PLATFORM_BODY_OFFSET)
                   & (body_y.abs() <= LATERAL_LIMIT)
                   & feet_safe.all(-1) & recent.all(-1)
                   & (self.current_support.sum(-1) >= MIN_SUPPORT_FEET)
                   & ~failure)
        self.consecutive_steps = torch.where(current, self.consecutive_steps + 1, 0)
        event = (self.consecutive_steps >= SUCCESS_FRAMES) & ~self.climbed & ~failure
        self.climbed |= event
        self.climb_step = torch.where(event, step, self.climb_step)
        return ClimbResult(event.clone(), self.climbed.clone(),
                           self.current_support.clone(), recent,
                           self.consecutive_steps.clone(), torch.zeros_like(event))


def platform_candidate(top_forces, geom_tile, geom_step, geom_bounds, tile_ids,
                       feet_safe):
    """Actual ``> 5 N`` top-platform contact of safe feet on the assigned tile."""
    n, h, d, _ = tile_parameters(tile_ids)
    is_platform = ((geom_step[None, :] == n[:, None].long())
                   & (geom_tile[None, :] == tile_ids[:, None])
                   & ((geom_bounds[:, 1] - geom_bounds[:, 0])[None, :] > _spec.PLATFORM_CONTACT_MIN_LENGTH))
    force = torch.where(is_platform[:, None, :], top_forces, -torch.inf)
    upward, surface_ids = force.max(-1)
    candidate = (upward > SUPPORT_FORCE_GT) & feet_safe & (tile_ids > 0)[:, None]
    return candidate, surface_ids


def feet_safe_platform(feet, tile_ids):
    """Current foot positions inside the platform's real 0.042 m inset."""
    n, h, d, _ = tile_parameters(tile_ids)
    half = feet.new_tensor(BODY_HALF_WIDTH)
    x0 = n * d + SAFE_INSET
    x1 = n * d + feet.new_tensor(2.0) - SAFE_INSET
    y = half - SAFE_INSET
    in_platform = ((feet[..., 0] >= x0[:, None]) & (feet[..., 0] <= x1[:, None])
                   & (feet[..., 1].abs() <= y))
    return in_platform, in_platform.all(-1)


@dataclass(frozen=True)
class TaskState:
    """One control sample's decision, captured before any reset."""

    local: torch.Tensor
    roll: torch.Tensor
    pitch: torch.Tensor
    yaw: torch.Tensor
    region: torch.Tensor
    n: torch.Tensor
    h: torch.Tensor
    d: torch.Tensor
    feet_local: torch.Tensor
    feet_safe: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    reason: torch.Tensor
    failure_masks: dict
    first_step: torch.Tensor
    first_step_acquired: torch.Tensor = None


def evaluate(*, episode_length, local, roll, pitch, yaw, feet_local, tile_ids,
             control_body_hit, body_link_force, body_link_mask, nonfoot_duration,
             control_long_hit, contact_overflow, nonfinite, top_forces,
             geom_tile, geom_step, geom_bounds, support_state, first_step_state,
             start_command):
    """Full SDS 6.2 decision for one control sample."""
    stairs = tile_ids > 0
    region = stairs & (local[:, 0] >= 0)
    n, h, d, _ = tile_parameters(tile_ids)
    feet = feet_local
    feet_safe_matrix, feet_safe = feet_safe_platform(feet, tile_ids)
    masks = failure_masks(
        local=local, region=region, stairs=stairs, tile_ids=tile_ids, roll=roll,
        pitch=pitch, control_body_hit=control_body_hit,
        body_link_force=body_link_force, body_link_mask=body_link_mask,
        nonfoot_duration=nonfoot_duration, control_long_hit=control_long_hit,
        contact_overflow=contact_overflow, nonfinite=nonfinite)
    failure = torch.zeros_like(region)
    for mask in masks.values():
        failure |= mask
    candidate, surface_ids = platform_candidate(
        top_forces, geom_tile, geom_step, geom_bounds, tile_ids, feet_safe_matrix)
    result = support_state.update(episode_length, stairs, tile_ids, surface_ids,
                                  candidate, feet_safe_matrix, local[:, 0], n * d,
                                  local[:, 1], failure)
    success = result.success & ~failure          # failure priority (SDS 3.1)
    timeout = episode_length >= timeout_steps(tile_ids, start_command)
    timeout = timeout & ~failure & ~success
    reason = failure_reason(masks)
    reason = torch.where(timeout, torch.full_like(reason, TIMEOUT_CODE), reason)
    reason = torch.where(success, torch.full_like(reason, SUCCESS_CODE), reason)
    first = first_step_contact(top_forces, geom_tile, geom_step, tile_ids)
    first_step_state |= first.any(-1)
    state = TaskState(local=local, roll=roll, pitch=pitch, yaw=yaw, region=region,
                      n=n, h=h, d=d, feet_local=feet, feet_safe=feet_safe_matrix,
                      success=success, failure=failure, timeout=timeout,
                      reason=reason, failure_masks=masks, first_step=first,
                      first_step_acquired=first_step_state.clone())
    return state


def climb_success(state: TaskState) -> torch.Tensor:
    """Success tensor of a :class:`TaskState` (failure priority already applied)."""
    return state.success


def first_step_acquired_from_geoms(top_forces, geom_tile, geom_step, tile_ids):
    """Standalone first-step detector (diagnostic, never a payment)."""
    return first_step_contact(top_forces, geom_tile, geom_step, tile_ids).any(-1)


def bottom_arrival_update(last_support_step,consecutive_steps,step,local,feet_local,ground_support):
    """Actual assigned-bottom top-force support; no foot-position proxy.

    Each foot must have a valid record in the latest 50 control frames, at
    least two must support now, and the complete spatial condition lasts five
    consecutive frames. The caller supplies contact-reduced >5 N booleans.
    """
    support=ground_support&(feet_local[:,:,0]<0)
    last=torch.where(support,step[:,None],last_support_step)
    recent=(step[:,None]-last)<_spec.SUPPORT_HISTORY_STEPS
    qualified=(local[:,0]<=_spec.BOTTOM_BODY_X)&(local[:,1].abs()<=_spec.BOTTOM_LATERAL_MAX)&(feet_local[:,:,0]<0).all(-1)&recent.all(-1)&(support.sum(-1)>=_spec.BOTTOM_MIN_SUPPORT_FEET)
    count=torch.where(qualified,consecutive_steps+1,0)
    return last,count,count>=_spec.BOTTOM_ARRIVAL_FRAMES
