"""Pure-torch SDS 0.2 revision 1.1 reward and success contracts.

Only call the state updates once per *control* sample (50 Hz).  Event state is
independent of the existing generic two-sample support detector.  All positions
and geometric bounds supplied to one call must use the same coordinate frame.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FootfallResult:
    reward: torch.Tensor
    events: torch.Tensor
    selected_step: torch.Tensor
    qualified_count: torch.Tensor


@dataclass(frozen=True)
class ParkingResult:
    stop_latched: torch.Tensor
    parking_steps: torch.Tensor
    success: torch.Tensor
    parking: torch.Tensor


def step_candidates(tile_ids, foot_xy, top_forces, geom_tile, geom_step,
                    geom_step_bounds, max_steps):
    """Select actual top-contact candidates by (assigned tile, 1-based step).

    ``top_forces[B,4,G]`` contains only actual contact-point / normal-qualified
    upward forces (5 mm, 30 degrees). Side/underside/calf contacts must already
    have been excluded by contact extraction, never inferred from foot height.
    ``geom_step[G]`` is 0 for ground, -1 for nonterrain, 1..N for treads. The last
    tread and its coplanar platform have the same step id and identical *union*
    bounds. Bounds are [xmin,xmax,ymin,ymax], not separately inset rectangles.
    Forces aggregate before applying the strict >5 N threshold. Ties resolve
    to the smaller 1-based step id, independent of geom order.
    """
    if top_forces.ndim != 3 or top_forces.shape[1] != 4:
        raise ValueError('top_forces must be [batch,4,geom]')
    batch, _, geoms = top_forces.shape
    if foot_xy.shape != (batch, 4, 2) or geom_step_bounds.shape != (geoms, 4):
        raise ValueError('foot_xy or geom_step_bounds shape mismatch')
    step = geom_step.long()
    assigned = ((geom_tile[None, :] == tile_ids[:, None]) &
                (step[None, :] > 0) & (step[None, :] <= max_steps))
    index = step.clamp(0, max_steps)[None, None, :].expand(batch, 4, -1)
    force = top_forces.new_zeros(batch, 4, max_steps + 1)
    force.scatter_add_(2, index, torch.where(assigned[:, None, :], top_forces, 0.))
    x, y = foot_xy[..., 0, None], foot_xy[..., 1, None]
    bounds = geom_step_bounds
    # Inclusive 42 mm safe margin. The input union removes the coplanar seam.
    safe = ((x >= bounds[None, None, :, 0] + .042) &
            (x <= bounds[None, None, :, 1] - .042) &
            (y >= bounds[None, None, :, 2] + .042) &
            (y <= bounds[None, None, :, 3] - .042) & assigned[:, None, :])
    safe_step = torch.zeros(batch, 4, max_steps + 1, dtype=torch.long, device=foot_xy.device)
    safe_step.scatter_reduce_(2, index, safe.long(), reduce='amax', include_self=True)
    eligible = (force[..., 1:] > 5.) & safe_step[..., 1:].bool()
    ranked = torch.where(eligible, force[..., 1:], -torch.inf)
    maximum, chosen = ranked.max(dim=-1)
    return torch.where(torch.isfinite(maximum), chosen + 1, -1), maximum


class FirstFootfallState:
    """Separate same-foot, same-(tile,step), two-control-sample debounce."""

    def __init__(self, num_envs, max_steps, device='cpu'):
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.foot_step_rewarded = torch.zeros(num_envs, 4, self.max_steps, dtype=torch.bool, device=device)
        self.last_qualified_step = torch.full((num_envs, 4), -1, dtype=torch.long, device=device)
        self.last_qualified_tile = torch.full_like(self.last_qualified_step, -1)
        self.qualified_count = torch.zeros_like(self.last_qualified_step)

    def reset(self, ids):
        self.foot_step_rewarded[ids] = False
        self.last_qualified_step[ids] = -1
        self.last_qualified_tile[ids] = -1
        self.qualified_count[ids] = 0

    def update(self, tile_ids, foot_xy, top_forces, geom_tile, geom_step,
               geom_step_bounds, failure, enabled=True):
        if not enabled:
            events = torch.zeros_like(self.foot_step_rewarded[..., 0])
            return FootfallResult(top_forces.new_zeros(tile_ids.shape), events,
                                  torch.full_like(self.last_qualified_step, -1), self.qualified_count.clone())
        selected, _ = step_candidates(tile_ids, foot_xy, top_forces, geom_tile,
                                       geom_step, geom_step_bounds, self.max_steps)
        valid = selected > 0
        same = valid & (selected == self.last_qualified_step) & (tile_ids[:, None] == self.last_qualified_tile)
        self.qualified_count = torch.where(same, self.qualified_count + 1, valid.long())
        self.last_qualified_step.copy_(selected)
        self.last_qualified_tile.copy_(torch.where(valid, tile_ids[:, None], -1))
        indices = (selected - 1).clamp_min(0)[..., None]
        already = self.foot_step_rewarded.gather(2, indices).squeeze(-1)
        events = valid & (self.qualified_count >= 2) & ~already & ~failure[:, None]
        self.foot_step_rewarded.scatter_(2, indices, (already | events)[..., None])
        return FootfallResult(events.sum(-1).to(top_forces.dtype) * .25,
                              events, selected, self.qualified_count.clone())


class ProgressState:
    """High-water displacement reward with no deferred/catch-up credit."""

    def __init__(self, num_envs, device='cpu'):
        self.x_max = torch.zeros(num_envs, device=device)

    def reset(self, ids, body_x):
        self.x_max[ids] = body_x

    def update(self, body_x, x_stop, vx_cmd, support, active, stairs,
               stop_latched, failure, enabled=True, dt=.02):
        x_eff = torch.minimum(body_x, x_stop)
        delta = torch.minimum((x_eff - self.x_max).clamp_min(0), vx_cmd.clamp_min(0) * dt)
        eligible = stairs & active & (support.sum(-1) >= 2) & ~stop_latched & ~failure
        reward = torch.where(eligible, 2 * delta, 0.) if enabled else torch.zeros_like(delta)
        # Update even after jumps, failure, ineligible support, parking or clipping.
        self.x_max.copy_(torch.maximum(self.x_max, x_eff))
        return reward


class ParkingState:
    """Shared A/B/C/D exact parking contract; reset on every new episode."""

    def __init__(self, num_envs, device='cpu'):
        self.stop_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.parking_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids):
        self.stop_latched[ids] = False
        self.parking_steps[ids] = 0

    def update(self, stairs, body_x, x_stop, all_feet_safe,
               all_platform_support_safe, body_y, speed, failure):
        # all_feet_safe is projection only; actual platform top support is a
        # distinct requirement for counting. The latch itself is irreversible.
        self.stop_latched |= stairs & (body_x >= x_stop) & all_feet_safe
        parking = (stairs & self.stop_latched & (body_x >= x_stop) &
                   all_platform_support_safe & (body_y.abs() <= .20) &
                   (speed < .05) & ~failure)
        self.parking_steps = torch.where(parking, self.parking_steps + 1, 0)
        success = (self.parking_steps >= 150) & ~failure
        return ParkingResult(self.stop_latched.clone(), self.parking_steps.clone(), success, parking)


def baseline_adjustments(raw_linear, raw_angular, raw_posture, command, enabled=True):
    """Return raw/base/adjusted terms; caller retains weights (1,1,.2) and dt.

    Only C/D enable subtraction, only while ||command|| > .1. Unmodified
    parking terms and A/B terms are therefore bit-for-bit the original inputs.
    """
    active = (command.norm(dim=-1) > .1) if enabled else torch.zeros_like(raw_linear, dtype=torch.bool)
    raw = {'track_linear_velocity': raw_linear, 'track_angular_velocity': raw_angular, 'pose': raw_posture}
    base = {'track_linear_velocity': torch.where(active, torch.exp(-command[:, :2].square().sum(-1) / .25), 0.),
            'track_angular_velocity': active.to(raw_angular.dtype), 'pose': active.to(raw_posture.dtype)}
    return {'raw': raw, 'base': base, 'adjusted': {key: raw[key] - base[key] for key in raw}}
