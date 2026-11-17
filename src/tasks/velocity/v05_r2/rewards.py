"""Pure v0.5-r2 (0.5-r2) reward functions and event state machines.

This module is a line-for-line derivative of ``src/tasks/velocity/v05/rewards.py``
(SDS 1.2 C9 / 8.1): only the relative ``from .spec`` import target — which now
resolves to the v0.5-r2 module with the same frozen constants — and the version
labels in this docstring changed.  Every formula, number, threshold, field name,
state transition, export name and error message is the unchanged r1 one.  The r2
version adds one observation channel outside this module and deliberately never
reads it here (SDS 3.9), and the three r1 anchors reproduce on the r2 package.
``tests/test_v05_r2_rewards.py`` proves the bit-for-bit equality against the r1
module and freezes the exported symbol set.

Nothing in this module reads a simulator, writes files or adds observations; all
inputs are simulation truth supplied by the caller.  Every continuous function
returns the **per-control-step amount** with ``dt`` applied exactly once, while
events (footfall/swing/success/failure) are one-time amounts that never multiply
``dt``.  SDS 3.3's truth table is a per-second *rate*, so
``speed_reward(...) / dt`` is the value to compare against it (the ``0.02``
printed in SDS 3.3/3.4 is that same control dt, not an extra weight).

SDS 4 fixes the batch interface used by ``v05/env.py``::

    speed_reward(vx_b, vy_b, executed_c, positive_gate, *, enabled, dt=0.02)
    StallState(num_envs, device)
    StallState.reset(ids, initial_x)
    StallState.update(local_x, executed_c, climbed, enabled) -> StallResult

The swing state keeps the v0.4 ``SwingSample``/``SwingResult`` field contract and
the exact physical recognition/anchor/de-duplication semantics; only the payment
coefficient (0.10 -> 0.15), the target window (0.30 -> 0.60 m) and the legal
``N == 0`` flat path differ (SDS 3.6).  All new reward state is reward/diagnostic
only and is never part of an observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional

import torch

from .spec import (COMMAND_MIN, CONTINUOUS_WEIGHTS, CONTROL_DT, FAILURE_AMOUNT, FOOTFALL_AMOUNT,
                   FOOT_RADIUS, MAX_SWING_STEPS, SAFE_INSET, SPEED_BACKWARD_CAP,
                   SPEED_BACKWARD_COEFFICIENT, SPEED_EXP_COEFFICIENT,
                   SPEED_EXP_OFFSET, SPEED_EXP_SCALE, SPEED_OVER_CAP, SPEED_OVER_COEFFICIENT,
                   SPEED_OVER_DEADBAND, SPEED_SIDE_CAP, SPEED_SIDE_COEFFICIENT, SPEED_SIDE_SCALE,
                   STALL_CONSECUTIVE_SAMPLES, STALL_DX_FRACTION_PER_SECOND,
                   STALL_DX_THRESHOLD_SCALE, STALL_HIGHWATER_DX, STALL_HISTORY_SAMPLES,
                   STALL_RATE, STALL_REWARD, SUCCESS_AMOUNT, SUPPORT_DEBOUNCE,
                   SUPPORT_FORCE_GT, SWING_AMOUNT_PER_TARGET, SWING_BODY_HEIGHT_SCALE,
                   SWING_MAX_FRACTION, SWING_TARGET_HEIGHT_MARGIN, SWING_TARGET_WINDOW_M)

GATE_SAFE_SUPPORTS = 2
GATE_ROLL_MAX = math.radians(30.)
GATE_PITCH_MIN = math.radians(-10.)
GATE_PITCH_MAX = math.radians(55.)
# SDS 3.7 orientation deadbands, which are *not* the SDS 3.2 gate bounds.
ORIENTATION_PITCH_MIN = math.radians(-5.)
ORIENTATION_PITCH_MAX = math.radians(55.)
NONFOOT_FORCE_GT = 10.


# ------------------------------------------------------------------ validation
def _validate_dt(dt):
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    return float(dt)


def _validate_vector(value, name, *, dtype=None, size=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError(f'{name} must be a one-dimensional torch tensor')
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f'{name} must be {dtype}')
    if not (value.dtype.is_floating_point or value.dtype == torch.bool):
        raise TypeError(f'{name} must be a floating/bool tensor')
    if size is not None and value.shape != (size,):
        raise ValueError(f'{name} must have shape [{size}]')
    return value


def _validate_bool(value, name, size):
    if not isinstance(value, torch.Tensor) or value.shape != (size,) or value.dtype != torch.bool:
        raise ValueError(f'{name} must be a bool tensor of shape [{size}]')
    return value


def _require_finite(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a torch tensor')
    if not torch.isfinite(value).all():
        raise ValueError(f'nonfinite {name}')
    return value


def _require_float_tensor(value, name, shape=None):
    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
        raise TypeError(f'{name} must be a floating torch tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(value.shape)}')
    return _require_finite(value, name)


def _share_device(values):
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError('reward tensors must share a device')
    return devices.pop()


# ---------------------------------------------------------------------- speed
def _speed_rate(vx_b, vy_b, executed_c, positive_gate, enabled):
    """Unweighted SDS 3.3 ``speed_rate``; the value the truth table lists per second."""
    c = torch.where(enabled, executed_c, torch.ones_like(executed_c))
    u = vx_b / c
    ey = vy_b / SPEED_SIDE_SCALE
    a = (SPEED_EXP_COEFFICIENT * torch.exp(-SPEED_EXP_SCALE * (u - 1.).square() - ey.square())
         + SPEED_EXP_OFFSET)
    over = (SPEED_OVER_COEFFICIENT * (u - SPEED_OVER_DEADBAND).clamp_min(0.).square()
            .clamp_max(SPEED_OVER_CAP))
    back = SPEED_BACKWARD_COEFFICIENT * (-u).clamp_min(0.).square().clamp_max(SPEED_BACKWARD_CAP)
    side = SPEED_SIDE_COEFFICIENT * ey.square().clamp_max(SPEED_SIDE_CAP)
    positive = torch.where(positive_gate, a.clamp_min(0.), torch.zeros_like(a))
    return positive + a.clamp_max(0.) - over - back - side


def speed_reward(vx_b, vy_b, executed_c, positive_gate, *, enabled, dt=CONTROL_DT):
    """0.5-r2 speed term (SDS 3.3), exactly ``dt * speed_rate`` per control step.

    ``u = vx_b/executed_c``, ``ey = vy_b/0.2``::

        a          = 1.5*exp(-4*(u-1)^2 - ey^2) - 0.5
        over       = 0.5 * min(relu(u-1.2)^2, 4)
        back       = 0.5 * min(relu(-u)^2, 4)
        side       = 0.25 * min(ey^2, 4)
        speed_rate = g*relu(a) + min(a,0) - over - back - side

    ``dt`` is applied exactly once (SDS 3.1, 9.1-1).  The ``0.02`` printed in SDS
    3.3 is that control dt, not an extra weight: all three SDS 9.3 scalar checks
    (stationary -2.482749, matched +3.973384, early failure -6.839384) reproduce
    only with ``dt * speed_rate`` (see ``spec.REWARD_CONFIG['speed']
    ['discount_checks']``).  ``speed_rate`` exports the unweighted per-second rate
    for analysis scripts.

    Only the positive exponential part is gated; every negative term survives a
    closed gate.  A disabled (parking-diagnostic) sample must use ``executed_c == 0``
    and returns exactly zero while its inputs must still be finite.  An enabled
    sample with ``executed_c <= 0`` is a protocol error, never ``nan_to_num``.
    """
    dt = _validate_dt(dt)
    b = _validate_vector(vx_b, 'vx_b')
    _validate_vector(vy_b, 'vy_b', size=b.numel())
    _validate_vector(executed_c, 'executed_c', size=b.numel())
    _validate_vector(positive_gate, 'positive_gate', dtype=torch.bool, size=b.numel())
    _validate_vector(enabled, 'enabled', dtype=torch.bool, size=b.numel())
    if not (vx_b.dtype.is_floating_point and vy_b.dtype.is_floating_point
            and executed_c.dtype.is_floating_point):
        raise TypeError('speed reward values must be floating tensors')
    _share_device((vx_b, vy_b, executed_c, positive_gate, enabled))
    _require_finite(vx_b, 'vx_b')
    _require_finite(vy_b, 'vy_b')
    _require_finite(executed_c, 'executed_c')
    if (enabled & (executed_c <= 0)).any():
        raise ValueError('active executed command must be positive')
    amount = dt * _speed_rate(vx_b, vy_b, executed_c, positive_gate, enabled)
    return torch.where(enabled, amount, torch.zeros_like(amount))


def speed_rate(vx_b, vy_b, executed_c, positive_gate, *, enabled):
    """The SDS 3.3 speed rate alone (per second, unweighted, no ``dt``).

    This is exactly the quantity in the SDS 3.3 truth table, so
    ``speed_reward(...) / dt`` must equal it; a disabled sample returns 0.
    """
    b = _validate_vector(vx_b, 'vx_b')
    _validate_vector(vy_b, 'vy_b', size=b.numel())
    _validate_vector(executed_c, 'executed_c', size=b.numel())
    _validate_vector(positive_gate, 'positive_gate', dtype=torch.bool, size=b.numel())
    _validate_vector(enabled, 'enabled', dtype=torch.bool, size=b.numel())
    if not (vx_b.dtype.is_floating_point and vy_b.dtype.is_floating_point
            and executed_c.dtype.is_floating_point):
        raise TypeError('speed reward values must be floating tensors')
    _share_device((vx_b, vy_b, executed_c, positive_gate, enabled))
    _require_finite(vx_b, 'vx_b')
    _require_finite(vy_b, 'vy_b')
    _require_finite(executed_c, 'executed_c')
    if (enabled & (executed_c <= 0)).any():
        raise ValueError('active executed command must be positive')
    rate = _speed_rate(vx_b, vy_b, executed_c, positive_gate, enabled)
    return torch.where(enabled, rate, torch.zeros_like(rate))


def positive_gate(safe_support, roll, pitch, nonfoot_hit, failure, caller_enabled, *,
                  minimum_supports=GATE_SAFE_SUPPORTS):
    """SDS 3.2 positive-reward gate ``g`` as a pure function.

    All five physical requirements are mandatory: at least two feet with real
    safe top support **in this frame**, ``|roll| <= 30 deg``,
    ``pitch in [-10, 55] deg``, no monitored non-foot link above 10 N and no
    failure; the caller may additionally disable the gate.
    """
    if not isinstance(safe_support, torch.Tensor) or safe_support.ndim != 2 or safe_support.shape[1] != 4:
        raise ValueError('safe_support must be a [batch,4] tensor')
    size = safe_support.shape[0]
    if safe_support.dtype != torch.bool:
        raise ValueError('safe_support must be bool')
    _validate_vector(roll, 'roll', size=size)
    _validate_vector(pitch, 'pitch', size=size)
    _validate_bool(nonfoot_hit, 'nonfoot_hit', size)
    _validate_bool(failure, 'failure', size)
    _validate_bool(caller_enabled, 'caller_enabled', size)
    if not roll.dtype.is_floating_point or not pitch.dtype.is_floating_point:
        raise TypeError('roll/pitch must be floating tensors')
    _share_device((safe_support, roll, pitch, nonfoot_hit, failure, caller_enabled))
    _require_finite(roll, 'roll')
    _require_finite(pitch, 'pitch')
    if isinstance(minimum_supports, bool) or not isinstance(minimum_supports, Integral) \
            or int(minimum_supports) < 1:
        raise ValueError('minimum_supports must be a positive integer')
    support_ok = safe_support.sum(-1) >= int(minimum_supports)
    posture = ((roll.abs() <= GATE_ROLL_MAX) & (pitch >= GATE_PITCH_MIN)
               & (pitch <= GATE_PITCH_MAX))
    return caller_enabled & support_ok & posture & ~nonfoot_hit & ~failure


# ------------------------------------------------------------ sustained stall
@dataclass(frozen=True)
class StallResult:
    reward: torch.Tensor
    dx_1s: torch.Tensor
    highwater_dx_1s: torch.Tensor
    window_ready: torch.Tensor
    qualifying: torch.Tensor
    consecutive_count: torch.Tensor
    penalty_on: torch.Tensor


class StallState:
    """Exactly 51 position samples / 51 high-water samples per environment (SDS 3.4).

    The window holds 50 control intervals = 1.00 s, so ``x[-1] - x[0]`` compares
    ``x_t`` with ``x_(t-50)``.  The per-step penalty is ``dt * stall_rate``
    (``-0.005``), applied once per control step while the condition holds.
    ``reset`` writes the initial position ``x0`` and
    ``M0 = x0`` with a history count of 1; every control step appends exactly one
    ``x_t`` and one ``M_t = max(M_(t-1), x_t)``.  Partial resets never touch other
    environments.  ``executed_c`` is recorded per environment: a changed command
    inside an unfinished episode raises instead of silently re-baselining.
    """

    def __init__(self, num_envs, device='cpu'):
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        b = self.num_envs
        self.x_history = torch.zeros(b, STALL_HISTORY_SAMPLES, dtype=torch.float32, device=self.device)
        self.m_history = torch.zeros(b, STALL_HISTORY_SAMPLES, dtype=torch.float32, device=self.device)
        self.count = torch.zeros(b, dtype=torch.long, device=self.device)
        self.consecutive_count = torch.zeros(b, dtype=torch.long, device=self.device)
        self.executed_c = torch.full((b,), float('nan'), dtype=torch.float32, device=self.device)
        self.dx_1s = torch.zeros(b, dtype=torch.float32, device=self.device)
        self.highwater_dx_1s = torch.zeros(b, dtype=torch.float32, device=self.device)
        self.window_ready = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.qualifying = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.penalty_on = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.reward = torch.zeros(b, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ reset
    def _ids(self, ids):
        if ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(ids, torch.Tensor):
            if ids.device != self.device:
                raise TypeError('ids must be on the state device')
            if ids.dtype == torch.bool:
                if ids.shape != (self.num_envs,):
                    raise ValueError('boolean ids must have shape [num_envs]')
                ids = torch.nonzero(ids, as_tuple=True)[0]
            elif ids.dtype != torch.int64 or ids.ndim != 1:
                raise TypeError('ids must be an int64 vector or bool mask')
        else:
            ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        if ids.numel() and ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('ids out of range')
        return ids

    def reset(self, ids, initial_x):
        ids = self._ids(ids)
        _require_float_tensor(initial_x, 'initial_x', shape=(ids.numel(),))
        if initial_x.device != self.device:
            raise TypeError('initial_x must be on the state device')
        with torch.no_grad():
            self.x_history[ids] = initial_x.to(self.x_history.dtype)[:, None]
            self.m_history[ids] = initial_x.to(self.m_history.dtype)[:, None]
            self.count[ids] = 1
            self.consecutive_count[ids] = 0
            self.executed_c[ids] = float('nan')
            self.dx_1s[ids] = 0.
            self.highwater_dx_1s[ids] = 0.
            self.window_ready[ids] = False
            self.qualifying[ids] = False
            self.penalty_on[ids] = False
            self.reward[ids] = 0.

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, local_x, executed_c, climbed, enabled):
        b = self.num_envs
        _validate_vector(local_x, 'local_x', size=b)
        _validate_vector(executed_c, 'executed_c', size=b)
        _validate_bool(climbed, 'climbed', b)
        _validate_bool(enabled, 'enabled', b)
        if not local_x.dtype.is_floating_point or not executed_c.dtype.is_floating_point:
            raise TypeError('local_x/executed_c must be floating tensors')
        _share_device((local_x, executed_c, climbed, enabled))
        _require_finite(local_x, 'local_x')
        _require_finite(executed_c, 'executed_c')
        active = enabled & (executed_c > COMMAND_MIN)
        if (enabled & (executed_c <= 0)).any():
            raise ValueError('active executed command must be positive')
        known = torch.isfinite(self.executed_c)
        if (known & (self.executed_c != executed_c.to(self.executed_c.dtype))).any():
            raise ValueError('executed command changed inside an episode; reset the stall state')
        with torch.no_grad():
            self.executed_c = torch.where(
                known, self.executed_c, executed_c.to(self.executed_c.dtype))
            x = local_x.to(self.x_history.dtype)
            m = torch.maximum(self.m_history[:, -1], x)
            # Exactly one append per control step.
            self.x_history = torch.cat((self.x_history[:, 1:], x[:, None]), dim=1)
            self.m_history = torch.cat((self.m_history[:, 1:], m[:, None]), dim=1)
            self.count = (self.count + 1).clamp_max(STALL_HISTORY_SAMPLES)
            window_ready = self.count >= STALL_HISTORY_SAMPLES
            dx = self.x_history[:, -1] - self.x_history[:, 0]
            dm = self.m_history[:, -1] - self.m_history[:, 0]
            threshold = STALL_DX_FRACTION_PER_SECOND * executed_c * STALL_DX_THRESHOLD_SCALE
            qualifying = (active & ~climbed & window_ready
                          & (dx < threshold.to(dx.dtype)) & (dm < STALL_HIGHWATER_DX))
            self.consecutive_count = torch.where(
                qualifying, self.consecutive_count + 1, torch.zeros_like(self.consecutive_count))
            penalty_on = qualifying & (self.consecutive_count >= STALL_CONSECUTIVE_SAMPLES)
            # SDS 3.4: r_stall = dt * stall_rate = 0.02 * (-0.25) = -0.005 per step.
            reward = torch.where(penalty_on, torch.full_like(dx, STALL_REWARD),
                                 torch.zeros_like(dx))
            self.dx_1s = dx
            self.highwater_dx_1s = dm
            self.window_ready = window_ready
            self.qualifying = qualifying
            self.penalty_on = penalty_on
            self.reward = reward
        return self.result()

    def result(self):
        return StallResult(reward=self.reward.clone(), dx_1s=self.dx_1s.clone(),
                           highwater_dx_1s=self.highwater_dx_1s.clone(),
                           window_ready=self.window_ready.clone(),
                           qualifying=self.qualifying.clone(),
                           consecutive_count=self.consecutive_count.clone(),
                           penalty_on=self.penalty_on.clone())


# ------------------------------------------------------------------- footfall
def step_candidates(tile_ids, foot_xy, top_forces, geom_tile, geom_step,
                    geom_step_bounds, max_steps):
    """Select the actual top-contact step per foot inside the assigned tile.

    ``top_forces[B,4,G]`` must already contain only actual contact-point,
    normal-qualified (<=30 deg) and 5 mm top-surface upward forces; side,
    underside or calf contacts are never inferred from foot height.  Forces of
    the same ``(tile, step)`` aggregate before the strict ``> 5 N`` test, the foot
    center must be inside the union rectangle inset by 42 mm, and ties resolve to
    the smaller 1-based step id.  Ground is step 0 and is never a candidate.
    """
    if not isinstance(top_forces, torch.Tensor) or top_forces.ndim != 3 or top_forces.shape[1] != 4:
        raise ValueError('top_forces must be [batch,4,geom]')
    _require_float_tensor(top_forces, 'top_forces')
    batch, _, geoms = top_forces.shape
    if not isinstance(geom_step_bounds, torch.Tensor) or tuple(geom_step_bounds.shape) != (geoms, 4):
        raise ValueError('geom_step_bounds shape mismatch')
    _require_float_tensor(geom_step_bounds, 'geom_step_bounds')
    if not isinstance(foot_xy, torch.Tensor) or tuple(foot_xy.shape) != (batch, 4, 2):
        raise ValueError('foot_xy must be [batch,4,2]')
    _require_float_tensor(foot_xy, 'foot_xy')
    for value, name in ((tile_ids, 'tile_ids'), (geom_tile, 'geom_tile'), (geom_step, 'geom_step')):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int64:
            raise ValueError(f'{name} must be an int64 tensor')
    _share_device((top_forces, geom_step_bounds, foot_xy, tile_ids, geom_tile, geom_step))
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or int(max_steps) < 1:
        raise ValueError('max_steps must be a positive integer')
    max_steps = int(max_steps)
    step = geom_step.long()
    assigned = ((geom_tile[None, :] == tile_ids[:, None])
                & (step[None, :] > 0) & (step[None, :] <= max_steps))
    index = step.clamp(0, max_steps)[None, None, :].expand(batch, 4, -1)
    force = top_forces.new_zeros(batch, 4, max_steps + 1)
    force.scatter_add_(2, index, torch.where(assigned[:, None, :], top_forces, 0.))
    x, y = foot_xy[..., 0, None], foot_xy[..., 1, None]
    bounds = geom_step_bounds
    safe = ((x >= bounds[None, None, :, 0] + SAFE_INSET)
            & (x <= bounds[None, None, :, 1] - SAFE_INSET)
            & (y >= bounds[None, None, :, 2] + SAFE_INSET)
            & (y <= bounds[None, None, :, 3] - SAFE_INSET) & assigned[:, None, :])
    safe_step = torch.zeros(batch, 4, max_steps + 1, dtype=torch.long, device=foot_xy.device)
    safe_step.scatter_reduce_(2, index, safe.long(), reduce='amax', include_self=True)
    eligible = (force[..., 1:] > SUPPORT_FORCE_GT) & safe_step[..., 1:].bool()
    ranked = torch.where(eligible, force[..., 1:], -torch.inf)
    maximum, chosen = ranked.max(dim=-1)
    return torch.where(torch.isfinite(maximum), chosen + 1, -1), maximum


@dataclass(frozen=True)
class FootfallResult:
    reward: torch.Tensor
    events: torch.Tensor
    selected_step: torch.Tensor
    qualified_count: torch.Tensor


def footfall_reward(events):
    """Recompute the 0.5-r2 footfall amount: ``events.sum(-1) * 0.50`` (SDS 3.5)."""
    if not isinstance(events, torch.Tensor):
        raise TypeError('events must be a torch tensor')
    if events.dtype == torch.bool:
        events = events.to(torch.float32)
    elif not events.dtype.is_floating_point:
        raise TypeError('events must be bool or floating')
    if events.ndim < 1:
        raise ValueError('events must have a batch dimension')
    if not torch.isfinite(events).all():
        raise ValueError('nonfinite footfall events')
    return events.sum(-1) * FOOTFALL_AMOUNT


class FootfallState:
    """Per-foot, per-(tile, step), per-episode first safe-support debounce.

    A payment needs two consecutive qualified top-support control samples on the
    same ``(tile, step)``, ``step in [1, N]`` and no failure in this frame; the
    ground never pays and the coplanar top platform shares the last step id, so
    it cannot pay twice (SDS 3.5).  The state advances exactly once per control
    step and the caller must use :func:`footfall_reward` (or ``result.reward``,
    which applies the same 0.50 amount) for the total; the old 0.25 amount is
    never reused.
    """

    def __init__(self, num_envs, max_steps=MAX_SWING_STEPS, device='cpu'):
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
            raise ValueError('num_envs must be a positive integer')
        if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or int(max_steps) < 1:
            raise ValueError('max_steps must be a positive integer')
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.device = torch.device(device)
        self.foot_step_rewarded = torch.zeros(self.num_envs, 4, self.max_steps,
                                              dtype=torch.bool, device=self.device)
        self.last_qualified_step = torch.full((self.num_envs, 4), -1, dtype=torch.long,
                                              device=self.device)
        self.last_qualified_tile = torch.full((self.num_envs, 4), -1, dtype=torch.long,
                                              device=self.device)
        self.qualified_count = torch.zeros(self.num_envs, 4, dtype=torch.long,
                                           device=self.device)
        self.qualified_step = torch.full((self.num_envs, 4), -1, dtype=torch.long,
                                         device=self.device)

    def reset(self, ids):
        ids = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(
            list(ids), dtype=torch.long, device=self.device)
        if ids.dtype == torch.bool:
            if ids.shape != (self.num_envs,):
                raise ValueError('boolean ids must have shape [num_envs]')
            ids = torch.nonzero(ids, as_tuple=True)[0]
        elif ids.dtype != torch.int64 or ids.ndim != 1:
            raise TypeError('ids must be an int64 vector or bool mask')
        if ids.numel() and ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('ids out of range')
        self.foot_step_rewarded[ids] = False
        self.last_qualified_step[ids] = -1
        self.last_qualified_tile[ids] = -1
        self.qualified_count[ids] = 0
        self.qualified_step[ids] = -1

    def update(self, tile_ids, foot_xy, top_forces, geom_tile, geom_step,
               geom_step_bounds, failure, enabled=True):
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be a bool')
        if not isinstance(failure, torch.Tensor) or failure.dtype != torch.bool:
            raise ValueError('failure must be a bool tensor')
        b = self.num_envs
        if not isinstance(tile_ids, torch.Tensor) or tuple(tile_ids.shape) != (b,):
            raise ValueError(f'tile_ids must have shape [{b}]')
        if tile_ids.dtype != torch.int64:
            raise ValueError('tile_ids must be int64')
        if failure.shape != (b,):
            raise ValueError(f'failure must have shape [{b}]')
        if not enabled:
            return FootfallResult(reward=torch.zeros(b, device=self.device),
                                  events=torch.zeros(b, 4, dtype=torch.bool, device=self.device),
                                  selected_step=torch.full_like(self.last_qualified_step, -1),
                                  qualified_count=self.qualified_count.clone())
        selected, _ = step_candidates(tile_ids, foot_xy, top_forces, geom_tile, geom_step,
                                      geom_step_bounds, self.max_steps)
        valid = selected > 0
        same = valid & (selected == self.last_qualified_step) \
            & (tile_ids[:, None] == self.last_qualified_tile)
        self.qualified_count = torch.where(same, self.qualified_count + 1,
                                           valid.long())
        self.last_qualified_step.copy_(selected)
        self.last_qualified_tile.copy_(torch.where(valid, tile_ids[:, None],
                                                   torch.full_like(selected, -1)))
        indices = (selected - 1).clamp_min(0)[..., None]
        already = self.foot_step_rewarded.gather(2, indices).squeeze(-1)
        events = valid & (self.qualified_count >= SUPPORT_DEBOUNCE) & ~already & ~failure[:, None]
        self.foot_step_rewarded.scatter_(2, indices, (already | events)[..., None])
        self.qualified_step = selected.clone()
        return FootfallResult(reward=footfall_reward(events), events=events.clone(),
                              selected_step=selected, qualified_count=self.qualified_count.clone())


# ---------------------------------------------------------------------- swing
@dataclass(frozen=True)
class SwingSample:
    tile_ids: torch.Tensor
    num_steps: torch.Tensor
    foot_pos_w: torch.Tensor
    foot_pos_b: torch.Tensor
    contact_found: torch.Tensor
    top_forces: torch.Tensor
    geom_tile: torch.Tensor
    geom_step: torch.Tensor
    geom_top: torch.Tensor
    geom_union_bounds: torch.Tensor
    executed_c: torch.Tensor
    caller_enabled: torch.Tensor
    failure: torch.Tensor
    roll: torch.Tensor
    pitch: torch.Tensor
    nonfoot_hit: torch.Tensor


@dataclass(frozen=True)
class SwingResult:
    reward: torch.Tensor
    safe_support: torch.Tensor
    selected_step: torch.Tensor
    target_step: torch.Tensor
    phi: torch.Tensor
    increment: torch.Tensor
    cumulative_paid: torch.Tensor
    consumed_fraction: torch.Tensor
    active: torch.Tensor
    positive_gate: torch.Tensor
    paid_amount: Optional[torch.Tensor] = None
    gate_details: Optional[dict] = None


class SwingClearanceState:
    """Bounded per-foot swing support/anchor/target/payment state (SDS 3.6).

    The physical recognition, anchor and anti-double-payment semantics are the
    v0.4 ones.  Only the payment coefficient is 0.15 per foot per target step and
    the target window is 0.60 m.  ``num_steps == 0`` (flat tile) is legal: real
    safe support and the gate still work, but there is no candidate step, no
    ``step 1`` geometry query and no stepped event.  Ordering is the ten-step
    SDS 3.6 sequence: support aggregation -> anchor refresh -> two real
    contact-free frames -> frozen takeoff anchor -> single immutable target lock
    -> H/phi -> delta consumption (even when the gate is closed) -> payment ->
    close-frame final-phi consumption without payment.

    ``paid_fraction`` is consumed whenever a target is locked, including
    gated/failed/contact frames, so a later gate reopening cannot claim the same
    height twice.  ``actual_paid_amount`` is stored independently and only grows
    when reward is really paid.
    """

    def __init__(self, num_envs, max_steps=MAX_SWING_STEPS, device='cpu'):
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
            raise ValueError('num_envs must be a positive integer')
        if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or int(max_steps) < 1:
            raise ValueError('max_steps must be a positive integer')
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.device = torch.device(device)
        b, f, k = self.num_envs, 4, self.max_steps

        def z(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.ready = z(b, f, dtype=torch.bool)
        self.active = z(b, f, dtype=torch.bool)
        self.contact_count = z(b, f, dtype=torch.long)
        self.no_contact_count = z(b, f, dtype=torch.long)
        self.qualified_step = torch.full((b, f), -1, dtype=torch.long, device=self.device)
        self._last_tile = torch.full((b, f), -1, dtype=torch.long, device=self.device)
        self.qualified_count = z(b, f, dtype=torch.long)
        self.highest_safe_step = torch.full((b, f), -1, dtype=torch.long, device=self.device)
        self.target_step = torch.full((b, f), -1, dtype=torch.long, device=self.device)
        self.target_locked = z(b, f, dtype=torch.bool)
        self.last_contact_bottom_z = z(b, f)
        self.last_contact_body_z = z(b, f)
        self.last_contact_support_z = z(b, f)
        self.takeoff_bottom_z = z(b, f)
        self.takeoff_body_z = z(b, f)
        self.takeoff_support_z = z(b, f)
        self.target_top_z = z(b, f)
        self.target_edge_x = z(b, f)
        self.paid_fraction = z(b, f, k)
        self.actual_paid_amount = z(b, f, k)
        self.phi = z(b, f)
        self.increment = z(b, f)
        self.paid_amount = z(b, f)
        self.safe_support = z(b, f, dtype=torch.bool)
        self.selected_step = torch.full((b, f), -1, dtype=torch.long, device=self.device)
        self.positive_gate = z(b, dtype=torch.bool)

    # ------------------------------------------------------------------ reset
    def reset(self, ids=None):
        """Clear exactly ``ids``; all other environments remain byte-identical."""
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(ids, torch.Tensor):
            if ids.device != self.device:
                raise TypeError('ids must be on the state device')
            if ids.dtype == torch.bool:
                if ids.shape != (self.num_envs,):
                    raise ValueError('boolean ids must have shape [num_envs]')
                ids = torch.nonzero(ids, as_tuple=True)[0]
            elif ids.dtype != torch.int64 or ids.ndim != 1:
                raise TypeError('ids must be an int64 vector or bool mask')
        else:
            ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        if ids.numel() and ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('ids out of range')
        for name in ('ready', 'active', 'contact_count', 'no_contact_count', 'qualified_step',
                     '_last_tile', 'qualified_count', 'highest_safe_step', 'target_step',
                     'target_locked', 'last_contact_bottom_z', 'last_contact_body_z',
                     'last_contact_support_z', 'takeoff_bottom_z', 'takeoff_body_z',
                     'takeoff_support_z', 'target_top_z', 'target_edge_x', 'paid_fraction',
                     'actual_paid_amount', 'phi', 'increment', 'paid_amount', 'safe_support',
                     'selected_step', 'positive_gate'):
            tensor = getattr(self, name)
            if name in ('qualified_step', '_last_tile', 'highest_safe_step', 'target_step',
                        'selected_step'):
                tensor[ids] = -1
            else:
                tensor[ids] = False if tensor.dtype == torch.bool else 0

    # --------------------------------------------------------------- helpers
    def _validate_sample(self, sample):
        if not isinstance(sample, SwingSample):
            raise TypeError('sample must be SwingSample')
        b, f = self.num_envs, 4
        g = (sample.top_forces.shape[-1]
             if isinstance(sample.top_forces, torch.Tensor) and sample.top_forces.ndim == 3 else -1)
        required_vectors = {
            'tile_ids': (torch.int64, b), 'num_steps': (torch.int64, b),
            'executed_c': (torch.float32, b), 'caller_enabled': (torch.bool, b),
            'failure': (torch.bool, b), 'roll': (torch.float32, b),
            'pitch': (torch.float32, b), 'nonfoot_hit': (torch.bool, b),
        }
        for name, (dtype, size) in required_vectors.items():
            value = getattr(sample, name)
            if not isinstance(value, torch.Tensor) or value.shape != (size,) or value.dtype != dtype:
                raise ValueError(f'{name} must be {dtype} shape [{size}]')
            if value.device != self.device:
                raise ValueError(f'{name} must be on {self.device}')
        for name in ('foot_pos_w', 'foot_pos_b'):
            value = getattr(sample, name)
            if (not isinstance(value, torch.Tensor) or value.shape != (b, f, 3)
                    or value.dtype != torch.float32 or value.device != self.device
                    or not torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite float32 [{b},4,3]')
        value = sample.contact_found
        if (not isinstance(value, torch.Tensor) or value.shape != (b, f)
                or value.dtype != torch.bool or value.device != self.device):
            raise ValueError(f'contact_found must be bool [{b},4]')
        if (not isinstance(sample.top_forces, torch.Tensor) or sample.top_forces.ndim != 3
                or sample.top_forces.shape[:2] != (b, f) or sample.top_forces.dtype != torch.float32
                or sample.top_forces.device != self.device
                or not torch.isfinite(sample.top_forces).all()):
            raise ValueError(f'top_forces must be finite float32 [{b},4,G]')
        for name, dtype in (('geom_tile', torch.int64), ('geom_step', torch.int64),
                            ('geom_top', torch.float32)):
            value = getattr(sample, name)
            if (not isinstance(value, torch.Tensor) or value.shape != (g,)
                    or value.dtype != dtype or value.device != self.device):
                raise ValueError(f'{name} must be {dtype} [{g}]')
        if (not isinstance(sample.geom_union_bounds, torch.Tensor)
                or sample.geom_union_bounds.shape != (g, 4)
                or sample.geom_union_bounds.dtype != torch.float32
                or sample.geom_union_bounds.device != self.device
                or not torch.isfinite(sample.geom_union_bounds).all()):
            raise ValueError(f'geom_union_bounds must be finite float32 [{g},4]')
        if (not torch.isfinite(sample.geom_top).all() or not torch.isfinite(sample.executed_c).all()
                or not torch.isfinite(sample.roll).all() or not torch.isfinite(sample.pitch).all()):
            raise ValueError('nonfinite swing state input')
        # N == 0 (flat) is a legal, explicitly supported configuration.
        if (sample.num_steps < 0).any() or (sample.num_steps > self.max_steps).any():
            raise ValueError('num_steps outside [0,max_steps]')

    @staticmethod
    def _target_geom(sample, tile_ids, target, required):
        """Return per-foot (top_z, bounds, edge_x); metadata errors are explicit.

        Only ``required`` rows are validated.  A mixed-slot batch always contains
        rows that have no target at all -- flat ``N == 0`` slots never produce a
        candidate, and feet that are not in the approach window keep the
        placeholder step.  Those rows must not be matched against, or asserted
        over, real stair geometry, otherwise one flat slot would fail the whole
        batch (SDS 3.6's N=0 rule).
        """
        target = target.long()
        tile = tile_ids[:, None, None]
        match = ((sample.geom_tile[None, None, :] == tile)
                 & (sample.geom_step[None, None, :] == target[:, :, None]))
        present = match.any(-1)
        bad = required & ~present
        if bad.any():
            env, foot = bad.nonzero(as_tuple=True)
            raise ValueError(
                f'missing target-step metadata for env {env[0].item()} foot {foot[0].item()} '
                f'target {target[env[0], foot[0]].item()}')
        tops = torch.where(match, sample.geom_top[None, None, :],
                           torch.full_like(sample.geom_top[None, None, :], torch.nan))
        finite = torch.isfinite(tops)
        # Restrict the finiteness assertion to the rows that must have a target.
        # Non-required rows may legitimately have no matching geom (flat tile,
        # placeholder step, untargeted foot) and stay -inf; they are masked out of
        # every lock/payment decision by ``usable``.
        bad_finite = required & ~finite.any(-1)
        if bad_finite.any():
            env, foot = bad_finite.nonzero(as_tuple=True)
            raise ValueError(
                f'target top metadata is not finite for env {env[0].item()} '
                f'foot {foot[0].item()} target {target[env[0], foot[0]].item()}')
        top = torch.where(finite, tops, torch.full_like(tops, -torch.inf)).amax(-1)
        first = finite.float().argmax(-1)
        bounds = sample.geom_union_bounds[first]
        return top, bounds, bounds[:, :, 0]

    # ---------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, sample):
        self._validate_sample(sample)
        b, f, k = self.num_envs, 4, self.max_steps
        found = sample.contact_found
        tile = sample.tile_ids[:, None]
        steps = sample.geom_step
        g = steps.numel()

        # 1. Aggregate real contact force per (tile, step) and update safe support.
        assigned = (sample.geom_tile[None, :] == tile) & (steps[None, :] >= 0)
        valid_range = assigned & (steps[None, :] <= sample.num_steps[:, None])
        index = steps.clamp(0, k)[None, None, :].expand(b, f, g)
        force = sample.top_forces.new_zeros(b, f, k + 1)
        force.scatter_add_(2, index, torch.where(valid_range[:, None, :], sample.top_forces, 0.))
        fx = sample.foot_pos_w[:, :, 0, None]
        fy = sample.foot_pos_w[:, :, 1, None]
        bounds = sample.geom_union_bounds
        safe_geom = ((fx >= bounds[None, None, :, 0] + SAFE_INSET)
                     & (fx <= bounds[None, None, :, 1] - SAFE_INSET)
                     & (fy >= bounds[None, None, :, 2] + SAFE_INSET)
                     & (fy <= bounds[None, None, :, 3] - SAFE_INSET)
                     & valid_range[:, None, :])
        safe_step = torch.zeros(b, f, k + 1, dtype=torch.int64, device=self.device)
        safe_step.scatter_reduce_(2, index, safe_geom.to(torch.int64),
                                  reduce='amax', include_self=True)
        safe_step = safe_step.bool()
        eligible = (force > SUPPORT_FORCE_GT) & safe_step
        ranked = torch.where(eligible, force, torch.full_like(force, -torch.inf))
        maximum = ranked.max(-1).values
        selected = torch.where(torch.isfinite(maximum),
                               (eligible & (force == maximum[..., None])).float().argmax(-1), -1)
        self.selected_step = selected
        valid = selected >= 0
        same = valid & (selected == self.qualified_step) & (tile == self._last_tile)
        self.qualified_count = torch.where(same, self.qualified_count + 1, valid.long())
        self.qualified_step.copy_(selected)
        self._last_tile = torch.where(valid, tile, torch.full_like(valid, -1,
                                                                   dtype=torch.long))
        safe_support = valid & (self.qualified_count >= SUPPORT_DEBOUNCE)
        self.safe_support = safe_support

        chosen = selected.clamp_min(0)
        same_step = (valid_range[:, None, :] & (steps[None, None, :] == chosen[:, :, None])
                     & (chosen[:, :, None] >= 0))
        individual = torch.where(same_step, sample.top_forces,
                                 torch.full_like(sample.top_forces, -torch.inf))
        geom_index = individual.argmax(-1)
        support_top = torch.where(safe_support, sample.geom_top[geom_index],
                                  torch.zeros_like(sample.geom_top[geom_index]))
        bottom = sample.foot_pos_w[:, :, 2] - FOOT_RADIUS
        body_z = sample.foot_pos_b[:, :, 2]
        # 2. Refresh the anchor only on a real safe-support frame.
        self.last_contact_bottom_z = torch.where(safe_support, bottom, self.last_contact_bottom_z)
        self.last_contact_body_z = torch.where(safe_support, body_z, self.last_contact_body_z)
        self.last_contact_support_z = torch.where(safe_support, support_top,
                                                  self.last_contact_support_z)
        self.highest_safe_step = torch.maximum(
            self.highest_safe_step, torch.where(safe_support, selected,
                                                torch.full_like(selected, -1)))

        ready_before = self.ready.clone()
        active_before = self.active.clone()
        contact_count = torch.where(found, self.contact_count + 1,
                                    torch.zeros_like(self.contact_count))
        no_contact_count = torch.where(~found, self.no_contact_count + 1,
                                       torch.zeros_like(self.no_contact_count))
        self.contact_count, self.no_contact_count = contact_count, no_contact_count
        # Any real contact that is not a safe top support invalidates the anchor.
        ready = torch.where(found, safe_support, ready_before)

        command_active = sample.caller_enabled & (sample.executed_c > COMMAND_MIN)
        gate = positive_gate(safe_support, sample.roll, sample.pitch, sample.nonfoot_hit,
                             sample.failure, sample.caller_enabled)
        self.positive_gate = gate

        active = active_before.clone()
        target_locked = self.target_locked.clone()
        target_step = self.target_step.clone()
        target_top = self.target_top_z.clone()
        target_edge = self.target_edge_x.clone()
        takeoff_support = self.takeoff_support_z.clone()
        phi = torch.zeros_like(self.phi)
        increment = torch.zeros_like(self.increment)
        reward = torch.zeros(b, device=self.device)
        cumulative = torch.zeros_like(self.paid_amount)

        # 3. Start only from a prior ready anchor after two real contact-free frames;
        # the takeoff anchor is frozen, never refreshed inside the swing.
        start = (ready_before & ~active_before & (no_contact_count >= 2) & ~found
                 & ~sample.failure[:, None])
        if start.any():
            self.takeoff_bottom_z = torch.where(start, self.last_contact_bottom_z,
                                                self.takeoff_bottom_z)
            self.takeoff_body_z = torch.where(start, self.last_contact_body_z,
                                              self.takeoff_body_z)
            self.takeoff_support_z = torch.where(start, self.last_contact_support_z,
                                                 self.takeoff_support_z)
            active |= start
            ready &= ~start
            target_locked &= ~start
            target_step = torch.where(start, torch.full_like(target_step, -1), target_step)
            target_top = torch.where(start, torch.zeros_like(target_top), target_top)
            target_edge = torch.where(start, torch.zeros_like(target_edge), target_edge)
            takeoff_support = self.takeoff_support_z

        # 10. Close only after two real contacts; a still-locked target's final phi
        # is consumed on that frame but never paid.
        close = active & found & (contact_count >= 2)
        close_locked = close & target_locked
        close_target_step = target_step.clone()
        active &= ~close
        target_locked &= ~close
        target_step = torch.where(close, torch.full_like(target_step, -1), target_step)

        # 4. Candidate is exactly highest real safe step + 1, inside [1, N]; for
        # N == 0 (flat) this is never a candidate and no step-1 geom is queried.
        candidate = self.highest_safe_step + 1
        usable = active & ~target_locked & (candidate >= 1) \
            & (candidate <= sample.num_steps[:, None])
        if usable.any():
            safe_candidate = torch.where(usable, candidate, torch.ones_like(candidate))
            top, target_bounds, edge = self._target_geom(
                sample, sample.tile_ids, safe_candidate, usable)
            foot_x = sample.foot_pos_w[:, :, 0]
            foot_y = sample.foot_pos_w[:, :, 1]
            in_window = (((edge - foot_x) >= 0.)
                         & ((edge - foot_x) <= SWING_TARGET_WINDOW_M))
            lateral = ((foot_y >= target_bounds[:, :, 2] + SAFE_INSET)
                       & (foot_y <= target_bounds[:, :, 3] - SAFE_INSET))
            higher = top > takeoff_support
            lock = usable & in_window & lateral & higher
            target_locked |= lock
            target_step = torch.where(lock, candidate, target_step)
            target_top = torch.where(lock, top, target_top)
            target_edge = torch.where(lock, edge, target_edge)

        locked = active & target_locked
        consume_locked = locked | close_locked
        effective_target_step = torch.where(close_locked, close_target_step, target_step)
        if consume_locked.any():
            # 6/7. World bottom uses the 32 mm foot radius; body height subtracts
            # the root position before rotating into the body frame.
            h_world = (sample.foot_pos_w[:, :, 2] - FOOT_RADIUS) - self.takeoff_bottom_z
            h_body = sample.foot_pos_b[:, :, 2] - self.takeoff_body_z
            height = target_top - takeoff_support + SWING_TARGET_HEIGHT_MARGIN
            bad = consume_locked & (~torch.isfinite(height) | (height <= 0.))
            if bad.any():
                env, foot = bad.nonzero(as_tuple=True)
                raise ValueError(
                    f'invalid swing target height H<=0 or nonfinite for env {env[0].item()} '
                    f'foot {foot[0].item()} target '
                    f'{effective_target_step[env[0], foot[0]].item()}')
            ratio = torch.where(consume_locked,
                                torch.minimum(h_world / height, h_body / SWING_BODY_HEIGHT_SCALE),
                                torch.zeros_like(h_world))
            phi = torch.where(consume_locked, ratio.clamp(0., SWING_MAX_FRACTION),
                              torch.zeros_like(ratio))
            target_index = (effective_target_step - 1).clamp(0, k - 1)
            target_index3 = target_index[..., None]
            old_fraction = self.paid_fraction.gather(2, target_index3).squeeze(-1)
            delta = torch.where(consume_locked, (phi - old_fraction).clamp_min(0.),
                                torch.zeros_like(phi))
            # 9. Payment additionally needs an airborne frame, an active command,
            # an open gate and no failure.
            pay_mask = (locked & ~found & command_active[:, None] & gate[:, None]
                        & ~sample.failure[:, None])
            increment = torch.where(pay_mask, SWING_AMOUNT_PER_TARGET * delta,
                                    torch.zeros_like(delta))
            old_amount = self.actual_paid_amount.gather(2, target_index3).squeeze(-1)
            new_amount = old_amount + increment
            self.actual_paid_amount.scatter_(
                2, target_index3, torch.where(consume_locked, new_amount, old_amount)[..., None])
            new_fraction = torch.maximum(old_fraction, phi)
            self.paid_fraction.scatter_(
                2, target_index3, torch.where(consume_locked, new_fraction, old_fraction)[..., None])
            reward = increment.sum(-1)
            cumulative = torch.where(locked, new_amount, torch.zeros_like(new_amount))
            consumed = torch.where(locked, new_fraction, torch.zeros_like(new_fraction))
        else:
            consumed = torch.zeros_like(self.phi)
        self.phi = torch.where(locked, phi, torch.zeros_like(phi))
        self.increment = torch.where(locked, increment, torch.zeros_like(increment))
        self.paid_amount = torch.where(locked, cumulative, torch.zeros_like(cumulative))
        self.active = active
        self.ready = torch.where(active, torch.where(found, safe_support, ready), ready)
        self.target_locked = target_locked
        self.target_step = target_step
        self.target_top_z = target_top
        self.target_edge_x = target_edge

        posture = gate_details_posture(sample)
        support_ok = safe_support.sum(-1) >= GATE_SAFE_SUPPORTS
        gate_details = dict(
            caller_enabled=sample.caller_enabled.clone(),
            command_active=command_active,
            support_count=safe_support.sum(-1),
            posture=posture,
            nonfoot_hit=sample.nonfoot_hit.clone(),
            failure=sample.failure.clone(),
            reason=torch.where(~sample.caller_enabled,
                               torch.full((b,), 1, dtype=torch.long, device=self.device),
                   torch.where(sample.failure,
                               torch.full((b,), 2, dtype=torch.long, device=self.device),
                   torch.where(sample.nonfoot_hit,
                               torch.full((b,), 3, dtype=torch.long, device=self.device),
                   torch.where(~posture,
                               torch.full((b,), 4, dtype=torch.long, device=self.device),
                   torch.where(~support_ok,
                               torch.full((b,), 5, dtype=torch.long, device=self.device),
                               torch.zeros(b, dtype=torch.long, device=self.device)))))))
        return SwingResult(reward=reward, safe_support=safe_support.clone(),
                           selected_step=selected.clone(), target_step=target_step.clone(),
                           phi=self.phi.clone(), increment=self.increment.clone(),
                           cumulative_paid=cumulative.clone(),
                           consumed_fraction=consumed.clone(), active=active.clone(),
                           positive_gate=gate.clone(), paid_amount=self.paid_amount.clone(),
                           gate_details=gate_details)


def gate_details_posture(sample):
    return ((sample.roll.abs() <= GATE_ROLL_MAX) & (sample.pitch >= GATE_PITCH_MIN)
            & (sample.pitch <= GATE_PITCH_MAX))


# ------------------------------------------------------- remaining pure terms
def orientation_reward(roll, pitch, *, dt=CONTROL_DT):
    """Weighted orientation penalty: ``roll^2 + relu(pitch-55)^2 + 4*relu(-5-pitch)^2``."""
    dt = _validate_dt(dt)
    b = _validate_vector(roll, 'roll')
    _validate_vector(pitch, 'pitch', size=b.numel())
    for value, name in ((roll, 'roll'), (pitch, 'pitch')):
        if not value.dtype.is_floating_point:
            raise TypeError(f'{name} must be floating')
        _require_finite(value, name)
    _share_device((roll, pitch))
    upper = (pitch - ORIENTATION_PITCH_MAX).clamp_min(0.).square()
    lower = (ORIENTATION_PITCH_MIN - pitch).clamp_min(0.).square()
    raw = roll.square() + upper + 4. * lower
    return dt * CONTINUOUS_WEIGHTS['orientation'] * raw


def nonfoot_reward(link_forces, *, dt=CONTROL_DT):
    """Weighted ``-dt * any(norm(link_force) > 10 N)``; never averaged per link."""
    dt = _validate_dt(dt)
    if not isinstance(link_forces, torch.Tensor) or link_forces.ndim != 3 \
            or link_forces.shape[-1] != 3:
        raise ValueError('link_forces must be [batch, links, 3]')
    if link_forces.shape[1] < 1:
        raise ValueError('at least one monitored link is required')
    _require_finite(link_forces, 'link_forces')
    hit = (link_forces.norm(dim=-1) > NONFOOT_FORCE_GT).any(-1)
    return dt * CONTINUOUS_WEIGHTS['nonfoot'] * hit.to(link_forces.dtype)


def event_reward(success_event, failure):
    """One-time ``+10`` safe arrival and ``-10`` real failure; failure wins."""
    b = _validate_vector(success_event, 'success_event', dtype=torch.bool)
    _validate_vector(failure, 'failure', dtype=torch.bool, size=b.numel())
    _share_device((success_event, failure))
    paid = (success_event & ~failure).to(torch.float32) * SUCCESS_AMOUNT
    return paid + failure.to(torch.float32) * FAILURE_AMOUNT


def pose_reward(joint_pos, default_joint_pos, speed, *, dt=CONTROL_DT):
    """``+0.05`` weighted pose term; active commands subtract the baseline 1."""
    dt = _validate_dt(dt)
    if not isinstance(joint_pos, torch.Tensor) or joint_pos.ndim != 2:
        raise ValueError('joint_pos must be [batch, joints]')
    b, joints = joint_pos.shape
    _require_float_tensor(joint_pos, 'joint_pos', (b, joints))
    _require_float_tensor(default_joint_pos, 'default_joint_pos', (b, joints))
    _validate_vector(speed, 'speed', size=b)
    _require_float_tensor(speed, 'speed', (b,))
    _share_device((joint_pos, default_joint_pos, speed))
    active = speed > COMMAND_MIN
    sigma = torch.where(
        active[:, None],
        joint_pos.new_tensor(([.15, .35, .5] * ((joints + 2) // 3))[:joints]),
        joint_pos.new_tensor(([.05, .1, .15] * ((joints + 2) // 3))[:joints]))
    raw = torch.exp(-(((joint_pos - default_joint_pos) / sigma).square()).mean(-1))
    return dt * CONTINUOUS_WEIGHTS['pose'] * (raw - active.to(raw.dtype))


def action_rate_reward(action, previous_action, *, dt=CONTROL_DT):
    """``-0.02 * sum((action_t - action_(t-1))^2) * dt`` (0.4 used -0.05)."""
    dt = _validate_dt(dt)
    if not isinstance(action, torch.Tensor) or action.ndim != 2:
        raise ValueError('action must be [batch, joints]')
    shape = tuple(action.shape)
    _require_float_tensor(action, 'action', shape)
    _require_float_tensor(previous_action, 'previous_action', shape)
    _share_device((action, previous_action))
    raw = (action - previous_action).square().sum(-1)
    return dt * CONTINUOUS_WEIGHTS['action_rate'] * raw


def track_angular_velocity_reward(body_ang_vel_b, command, *, dt=CONTROL_DT):
    """Weight ``+1`` exponential yaw tracking with the active-command ``-1`` baseline."""
    dt = _validate_dt(dt)
    if not isinstance(body_ang_vel_b, torch.Tensor) or body_ang_vel_b.ndim != 2 \
            or body_ang_vel_b.shape[-1] != 3:
        raise ValueError('body_ang_vel_b must be [batch,3]')
    b = body_ang_vel_b.shape[0]
    _require_float_tensor(body_ang_vel_b, 'body_ang_vel_b', (b, 3))
    if not isinstance(command, torch.Tensor) or command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError('command must be [batch,3]')
    _require_float_tensor(command, 'command', (b, 3))
    _share_device((body_ang_vel_b, command))
    raw = torch.exp(-(((body_ang_vel_b[:, 2] - command[:, 2]).square()
                       + .05 * body_ang_vel_b[:, :2].square().sum(-1)) / .5))
    active = command.norm(dim=-1) > COMMAND_MIN
    return dt * CONTINUOUS_WEIGHTS['track_angular_velocity'] * (raw - active.to(raw.dtype))


def stand_still_reward(joint_pos, default_joint_pos, command, *, dt=CONTROL_DT):
    """``-1.0 * sum((q-q_default)^2) * I(||command|| <= 0.1)``; not this version's stall."""
    dt = _validate_dt(dt)
    if not isinstance(joint_pos, torch.Tensor) or joint_pos.ndim != 2:
        raise ValueError('joint_pos must be [batch, joints]')
    shape = tuple(joint_pos.shape)
    _require_float_tensor(joint_pos, 'joint_pos', shape)
    _require_float_tensor(default_joint_pos, 'default_joint_pos', shape)
    if not isinstance(command, torch.Tensor):
        raise TypeError('command must be a torch tensor')
    if command.ndim == 1:
        norm = command
    elif command.ndim == 2:
        norm = command.norm(dim=-1)
    else:
        raise ValueError('command must be [batch] or [batch,3]')
    _require_float_tensor(norm, 'command', (shape[0],))
    _share_device((joint_pos, default_joint_pos, norm))
    still = (norm <= COMMAND_MIN).to(joint_pos.dtype)
    raw = (joint_pos - default_joint_pos).square().sum(-1) * still
    return dt * CONTINUOUS_WEIGHTS['stand_still'] * raw


def center_reward(local_y, active, *, dt=CONTROL_DT):
    """``-0.5 * min((local_y/0.2)^2,4) * I(active)``; active covers flat/approach."""
    dt = _validate_dt(dt)
    b = _validate_vector(local_y, 'local_y')
    _validate_bool(active, 'active', b.numel())
    _require_float_tensor(local_y, 'local_y', (b.numel(),))
    _share_device((local_y, active))
    raw = (local_y / .2).square().clamp_max(4.) * active.to(local_y.dtype)
    return dt * CONTINUOUS_WEIGHTS['center'] * raw


def heading_reward(local_yaw, active, *, dt=CONTROL_DT):
    """``-0.5 * (1-cos(local_yaw)) * I(active)`` in radians."""
    dt = _validate_dt(dt)
    b = _validate_vector(local_yaw, 'local_yaw')
    _validate_bool(active, 'active', b.numel())
    _require_float_tensor(local_yaw, 'local_yaw', (b.numel(),))
    _share_device((local_yaw, active))
    raw = (1. - local_yaw.cos()) * active.to(local_yaw.dtype)
    return dt * CONTINUOUS_WEIGHTS['heading'] * raw


def slip_reward(foot_vel_world_xy, contact, active, *, dt=CONTROL_DT):
    """``-0.25 * sum_f(I(contact_f)*(vx^2+vy^2)) * I(active)``; no foot averaging."""
    dt = _validate_dt(dt)
    if not isinstance(foot_vel_world_xy, torch.Tensor) or foot_vel_world_xy.ndim != 3 \
            or foot_vel_world_xy.shape[1:] != (4, 2):
        raise ValueError('foot_vel_world_xy must be [batch,4,2]')
    b = foot_vel_world_xy.shape[0]
    _require_float_tensor(foot_vel_world_xy, 'foot_vel_world_xy', (b, 4, 2))
    if not isinstance(contact, torch.Tensor) or tuple(contact.shape) != (b, 4) \
            or contact.dtype != torch.bool:
        raise ValueError('contact must be bool [batch,4]')
    _validate_bool(active, 'active', b)
    _share_device((foot_vel_world_xy, contact, active))
    per_foot = foot_vel_world_xy.square().sum(-1)
    raw = (per_foot * contact.to(per_foot.dtype)).sum(-1) * active.to(per_foot.dtype)
    return dt * CONTINUOUS_WEIGHTS['slip'] * raw


def soft_landing_reward(foot_forces, contact, contact_duration, active, *, dt=CONTROL_DT):
    """``-0.001 * sum_f(|force_f| * I(contact_f AND duration_f <= dt+1e-7)) * I(active)``."""
    dt = _validate_dt(dt)
    if not isinstance(foot_forces, torch.Tensor) or foot_forces.ndim != 3 \
            or foot_forces.shape[1:] != (4, 3):
        raise ValueError('foot_forces must be [batch,4,3]')
    b = foot_forces.shape[0]
    _require_float_tensor(foot_forces, 'foot_forces', (b, 4, 3))
    if not isinstance(contact, torch.Tensor) or tuple(contact.shape) != (b, 4) \
            or contact.dtype != torch.bool:
        raise ValueError('contact must be bool [batch,4]')
    if not isinstance(contact_duration, torch.Tensor) or tuple(contact_duration.shape) != (b, 4):
        raise ValueError('contact_duration must be [batch,4]')
    _require_float_tensor(contact_duration, 'contact_duration', (b, 4))
    _validate_bool(active, 'active', b)
    _share_device((foot_forces, contact, contact_duration, active))
    fresh = contact & (contact_duration <= dt + 1e-7)
    raw = (foot_forces.norm(dim=-1) * fresh.to(foot_forces.dtype)).sum(-1) \
        * active.to(foot_forces.dtype)
    return dt * CONTINUOUS_WEIGHTS['soft_landing'] * raw


def edge_reward(foot_xy, support, union_bounds, *, dt=CONTROL_DT):
    """``-1.0 * mean_4feet(I(support_f) * relu((0.042-distance_f)/0.042)^2) * dt``.

    ``union_bounds[B,4,4]`` must already be the merged rectangle for a coplanar
    last tread plus platform, so the seam is never punished twice.
    """
    dt = _validate_dt(dt)
    if not isinstance(foot_xy, torch.Tensor) or foot_xy.ndim != 3 \
            or foot_xy.shape[1:] != (4, 2):
        raise ValueError('foot_xy must be [batch,4,2]')
    b = foot_xy.shape[0]
    _require_float_tensor(foot_xy, 'foot_xy', (b, 4, 2))
    if not isinstance(support, torch.Tensor) or tuple(support.shape) != (b, 4) \
            or support.dtype != torch.bool:
        raise ValueError('support must be bool [batch,4]')
    if not isinstance(union_bounds, torch.Tensor) or tuple(union_bounds.shape) != (b, 4, 4):
        raise ValueError('union_bounds must be [batch,4,4]')
    _require_float_tensor(union_bounds, 'union_bounds', (b, 4, 4))
    _share_device((foot_xy, support, union_bounds))
    x, y = foot_xy[..., 0], foot_xy[..., 1]
    distance = torch.stack((x - union_bounds[..., 0], union_bounds[..., 1] - x,
                            y - union_bounds[..., 2], union_bounds[..., 3] - y),
                           dim=-1).amin(-1)
    raw = (support.to(foot_xy.dtype) * ((SAFE_INSET - distance) / SAFE_INSET)
           .clamp_min(0.).square()).mean(-1)
    return dt * CONTINUOUS_WEIGHTS['edge'] * raw


def joint_limits_reward(joint_pos, soft_limits, *, dt=CONTROL_DT):
    """``-10.0 * (sum of soft-limit distances) * dt``."""
    dt = _validate_dt(dt)
    if not isinstance(joint_pos, torch.Tensor) or joint_pos.ndim != 2:
        raise ValueError('joint_pos must be [batch, joints]')
    b, joints = joint_pos.shape
    _require_float_tensor(joint_pos, 'joint_pos', (b, joints))
    if not isinstance(soft_limits, torch.Tensor) or tuple(soft_limits.shape) != (b, joints, 2):
        raise ValueError('soft_limits must be [batch, joints, 2]')
    _require_float_tensor(soft_limits, 'soft_limits', (b, joints, 2))
    _share_device((joint_pos, soft_limits))
    raw = ((soft_limits[..., 0] - joint_pos).clamp_min(0.)
           + (joint_pos - soft_limits[..., 1]).clamp_min(0.)).sum(-1)
    return dt * CONTINUOUS_WEIGHTS['joint_limits'] * raw


def joint_acc_reward(joint_acc, *, dt=CONTROL_DT):
    """``-2.5e-7 * sum(joint_acc^2) * dt``."""
    dt = _validate_dt(dt)
    if not isinstance(joint_acc, torch.Tensor) or joint_acc.ndim != 2:
        raise ValueError('joint_acc must be [batch, joints]')
    shape = tuple(joint_acc.shape)
    _require_float_tensor(joint_acc, 'joint_acc', shape)
    raw = joint_acc.square().sum(-1)
    return dt * CONTINUOUS_WEIGHTS['joint_acc'] * raw


def body_ang_vel_reward(root_ang_vel_world, *, dt=CONTROL_DT):
    """``-0.05 * sum(wx^2 + wy^2) * dt`` (world roll/pitch rate only)."""
    dt = _validate_dt(dt)
    if not isinstance(root_ang_vel_world, torch.Tensor) or root_ang_vel_world.ndim != 2 \
            or root_ang_vel_world.shape[-1] != 3:
        raise ValueError('root_ang_vel_world must be [batch,3]')
    b = root_ang_vel_world.shape[0]
    _require_float_tensor(root_ang_vel_world, 'root_ang_vel_world', (b, 3))
    raw = root_ang_vel_world[:, :2].square().sum(-1)
    return dt * CONTINUOUS_WEIGHTS['body_ang_vel'] * raw


def angular_momentum_reward(root_angmom, *, dt=CONTROL_DT):
    """``-0.025 * sum(root_angmom^2) * dt``."""
    dt = _validate_dt(dt)
    if not isinstance(root_angmom, torch.Tensor) or root_angmom.ndim != 2 \
            or root_angmom.shape[-1] != 3:
        raise ValueError('root_angmom must be [batch,3]')
    b = root_angmom.shape[0]
    _require_float_tensor(root_angmom, 'root_angmom', (b, 3))
    raw = root_angmom.square().sum(-1)
    return dt * CONTINUOUS_WEIGHTS['angular_momentum'] * raw
