"""Pure v0.6 reward functions, air-time state and the batch reward computer.

Implements SDS 3 exactly (``SDS_A2_IdealHeightMap_Stairs40_v0.6.md`` revision
1.0) on the frozen interfaces of ``coordination/W0_FREEZE.md`` section 4.2:

* per control step ``r_t = dt*sum_i(w_i*f_i) + 0.4*(x_t-x_(t-1))
  + 10*I_success - 10*I_failure``;
* every continuous term is ``dt * weight * unweighted_rate`` with ``dt`` applied
  **once**; terminal events are one-shot amounts never scaled by ``dt``;
* failure wins over success in the same frame and a timeout adds exactly ``0``;
* ``feet_air_time`` settles the accumulated air time *before* clearing it, only
  initialises on the very first contact, never pays while standing and never
  pays twice after a one-frame contact dropout; partial resets only touch the
  listed environments;
* ``foot_clearance`` uses the surface height under the current foot position
  (``h_i = z_site - 0.032 - H(x_i, y_i)``) with no smoothing and no look-ahead;
* ``forward_progress`` is ``0.4*(x_t-x_(t-1))`` with no extra ``dt`` and uses the
  spawn position as the new ``previous_local_x`` after a reset;
* all differenced terms (action, qdot, local_x) never cross an episode boundary.

Nothing here reads a simulator, writes files, or invents observed state.  As the
frozen W0 frame has no per-foot surface tensor, ``RewardFrame`` carries the
required ``foot_surface_height`` (SDS 3.2 ``H(x_foot,y_foot)``); the caller (v07
``env.py``) fills it from the one v07 ``TerrainSpec`` query.  That additive field
is recorded in ``coordination/QA_spec_rewards.md`` for Codex adjudication.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional

import torch

from .spec import (A2_DEFAULT_JOINT_POS, A2_SOFT_JOINT_LIMITS, AIR_TIME_TARGET,
                   BASE_HEIGHT_TARGET, CENTER_LATERAL_SCALE, CENTER_SATURATION,
                   CONTROL_DT, FAILURE_REWARD, FLAT_CLEARANCE_TARGET, FLAT_TILE_ID,
                   FOOT_CLEARANCE_HEIGHT_MARGIN, FOOT_CLEARANCE_MIN_TARGET,
                   FOOT_RADIUS, FORWARD_PROGRESS_PER_M, LIN_VEL_Z_DEADBAND,
                   REWARD_TERMS, REWARD_WEIGHTS, ROUTE_YAW_AXIS, SUCCESS_REWARD,
                   TERRAIN_TABLE, TRACKING_EXP_SCALE)

#: One shared array of the SDS 5.2 step heights indexed by tile id; it is built
#: once from ``spec.TERRAIN_TABLE`` so the reward never duplicates a number.
_STEP_HEIGHTS = torch.tensor([entry.h for entry in TERRAIN_TABLE], dtype=torch.float64)
_TILE_COUNT = len(TERRAIN_TABLE)


# ------------------------------------------------------------------ validation
def _validate_dt(dt):
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    return float(dt)


def _require_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a torch tensor')
    return value


def _require_float(value, name, shape=None):
    _require_tensor(value, name)
    if not value.dtype.is_floating_point:
        raise TypeError(f'{name} must be a floating tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(value.shape)}')
    if not torch.isfinite(value).all():
        raise ValueError(f'nonfinite {name}')
    return value


def _require_bool(value, name, shape=None):
    _require_tensor(value, name)
    if value.dtype != torch.bool:
        raise TypeError(f'{name} must be a bool tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(value.shape)}')
    return value


def _require_int(value, name, shape=None):
    _require_tensor(value, name)
    if value.dtype != torch.int64:
        raise TypeError(f'{name} must be an int64 tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(value.shape)}')
    return value


def _share_device(values, names):
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError(f'{names} must share one device')
    return devices.pop()


def _batch(value, name):
    if value.ndim < 1:
        raise ValueError(f'{name} must have a batch dimension')
    return value.shape[0]


def _default_joint_pos(q):
    default = q.new_tensor(A2_DEFAULT_JOINT_POS)
    return default.expand(q.shape[0], default.shape[0])


def _default_soft_limits(q):
    limits = q.new_tensor(A2_SOFT_JOINT_LIMITS)
    return limits.expand(q.shape[0], *limits.shape)


def _clearance_targets(tile_ids):
    """SDS 3.3 target per environment, read from the one terrain table."""
    if (tile_ids < 0).any() or (tile_ids >= _TILE_COUNT).any():
        raise ValueError('tile_ids outside the v0.6 terrain table')
    heights = _STEP_HEIGHTS.to(tile_ids.device).to(torch.float32)[tile_ids]
    stair = torch.maximum(torch.full_like(heights, FOOT_CLEARANCE_MIN_TARGET),
                          heights + FOOT_CLEARANCE_HEIGHT_MARGIN)
    flat = tile_ids == FLAT_TILE_ID
    return torch.where(flat, torch.full_like(stair, FLAT_CLEARANCE_TARGET), stair)


# ------------------------------------------------------------- continuous rates
def tracking_lin_vel_rate(body_lin_vel_b, executed_command):
    """``exp(-((vx-c)^2 + vy^2)/0.25)`` (SDS 3.2)."""
    b = _batch(_require_float(body_lin_vel_b, 'body_lin_vel_b'), 'body_lin_vel_b')
    if tuple(body_lin_vel_b.shape) != (b, 3):
        raise ValueError('body_lin_vel_b must be [batch,3]')
    _require_float(executed_command, 'executed_command', (b,))
    _share_device([body_lin_vel_b, executed_command], 'tracking inputs')
    error = (body_lin_vel_b[:, 0] - executed_command).square() + body_lin_vel_b[:, 1].square()
    return torch.exp(-error / TRACKING_EXP_SCALE)


def tracking_ang_vel_rate(body_ang_vel_b):
    """``exp(-wz^2/0.25)`` (SDS 3.2)."""
    b = _batch(_require_float(body_ang_vel_b, 'body_ang_vel_b'), 'body_ang_vel_b')
    if tuple(body_ang_vel_b.shape) != (b, 3):
        raise ValueError('body_ang_vel_b must be [batch,3]')
    return torch.exp(-body_ang_vel_b[:, 2].square() / TRACKING_EXP_SCALE)


def lin_vel_z_rate(body_lin_vel_b):
    """``max(abs(vz)-0.15,0)^2``; exactly zero at v_z = 0.15 m/s (SDS 3.2)."""
    b = _batch(_require_float(body_lin_vel_b, 'body_lin_vel_b'), 'body_lin_vel_b')
    if tuple(body_lin_vel_b.shape) != (b, 3):
        raise ValueError('body_lin_vel_b must be [batch,3]')
    return (body_lin_vel_b[:, 2].abs() - LIN_VEL_Z_DEADBAND).clamp_min(0.).square()


def orientation_roll_rate(gravity_b):
    """``g_y^2``: roll only, pure pitch is never punished (SDS 3.2/9.1-7)."""
    b = _batch(_require_float(gravity_b, 'gravity_b'), 'gravity_b')
    if tuple(gravity_b.shape) != (b, 3):
        raise ValueError('gravity_b must be [batch,3]')
    return gravity_b[:, 1].square()


def base_height_rate(base_height_above_surface, *, target=BASE_HEIGHT_TARGET):
    """``(z_base - H(base_xy) - 0.40)^2`` (SDS 3.2/3.3)."""
    b = _batch(_require_float(base_height_above_surface, 'base_height_above_surface'),
               'base_height_above_surface')
    if tuple(base_height_above_surface.shape) != (b,):
        raise ValueError('base_height_above_surface must be [batch]')
    return (base_height_above_surface - float(target)).square()


def action_rate_rate(action, previous_action):
    """``sum((a_t - a_(t-1))^2)`` (SDS 3.2)."""
    b = _batch(_require_float(action, 'action'), 'action')
    _require_float(previous_action, 'previous_action', tuple(action.shape))
    _share_device([action, previous_action], 'action_rate inputs')
    return (action - previous_action).square().sum(-1)


def similar_to_default_rate(q, q_default=None):
    """``sum(abs(q - q0))`` with the existing A2 default pose (SDS 3.2)."""
    b = _batch(_require_float(q, 'q'), 'q')
    if q.ndim != 2:
        raise ValueError('q must be [batch, joints]')
    if q_default is None:
        q_default = _default_joint_pos(q)
    _require_float(q_default, 'q_default', tuple(q.shape))
    _share_device([q, q_default], 'similar_to_default inputs')
    return (q - q_default).abs().sum(-1)


def dof_acc_rate(qdot, previous_qdot, *, dt=CONTROL_DT):
    """``sum(((qdot_t - qdot_(t-1))/dt)^2)`` (SDS 3.2)."""
    dt = _validate_dt(dt)
    b = _batch(_require_float(qdot, 'qdot'), 'qdot')
    if qdot.ndim != 2:
        raise ValueError('qdot must be [batch, joints]')
    _require_float(previous_qdot, 'previous_qdot', tuple(qdot.shape))
    _share_device([qdot, previous_qdot], 'dof_acc inputs')
    return ((qdot - previous_qdot) / dt).square().sum(-1)


def dof_vel_rate(qdot):
    """``sum(qdot^2)`` (SDS 3.2)."""
    b = _batch(_require_float(qdot, 'qdot'), 'qdot')
    if qdot.ndim != 2:
        raise ValueError('qdot must be [batch, joints]')
    return qdot.square().sum(-1)


def ang_vel_xy_rate(body_ang_vel_b):
    """``wx^2 + wy^2`` (SDS 3.2)."""
    b = _batch(_require_float(body_ang_vel_b, 'body_ang_vel_b'), 'body_ang_vel_b')
    if tuple(body_ang_vel_b.shape) != (b, 3):
        raise ValueError('body_ang_vel_b must be [batch,3]')
    return body_ang_vel_b[:, :2].square().sum(-1)


def foot_slip_rate(foot_velocity_world, contact_filtered):
    """``sum_f(C_f * ||u_f||^2)`` with filtered contact (SDS 3.2)."""
    b = _batch(_require_float(foot_velocity_world, 'foot_velocity_world'),
               'foot_velocity_world')
    if tuple(foot_velocity_world.shape) != (b, 4, 3):
        raise ValueError('foot_velocity_world must be [batch,4,3]')
    _require_bool(contact_filtered, 'contact_filtered', (b, 4))
    _share_device([foot_velocity_world, contact_filtered], 'foot_slip inputs')
    per_foot = foot_velocity_world[..., :2].square().sum(-1)
    return (per_foot * contact_filtered.to(per_foot.dtype)).sum(-1)


def foot_clearance_rate(foot_position_world, foot_velocity_world, foot_surface_height,
                        contact_filtered, tile_ids, *, foot_radius=FOOT_RADIUS):
    """``sum_f((1-C_f)*(h_f-h_target)^2*||u_f||)`` (SDS 3.2/3.3).

    ``h_f = z_site - 0.032 - H(x_foot,y_foot)`` uses the surface height under the
    current foot position; the caller must not pre-smooth it.  ``h_target`` is
    ``max(0.17, h_step+0.02)`` for stairs and ``0.075`` on flat.
    """
    b = _batch(_require_float(foot_position_world, 'foot_position_world'),
               'foot_position_world')
    if tuple(foot_position_world.shape) != (b, 4, 3):
        raise ValueError('foot_position_world must be [batch,4,3]')
    _require_float(foot_velocity_world, 'foot_velocity_world', (b, 4, 3))
    _require_float(foot_surface_height, 'foot_surface_height', (b, 4))
    _require_bool(contact_filtered, 'contact_filtered', (b, 4))
    _require_int(tile_ids, 'tile_ids', (b,))
    _share_device([foot_position_world, foot_velocity_world, foot_surface_height,
                   contact_filtered, tile_ids], 'foot_clearance inputs')
    clearance = (foot_position_world[..., 2] - float(foot_radius)
                 - foot_surface_height)
    target = _clearance_targets(tile_ids)[:, None]
    error = (clearance - target).square()
    speed = foot_velocity_world[..., :2].norm(dim=-1)
    return (error * speed * (~contact_filtered).to(error.dtype)).sum(-1)


def joint_tracking_rate(q_target_executed, q):
    """``sum((q_target_executed - q)^2)``: the target actually executed (SDS 3.2)."""
    b = _batch(_require_float(q_target_executed, 'q_target_executed'), 'q_target_executed')
    if q_target_executed.ndim != 2:
        raise ValueError('q_target_executed must be [batch, joints]')
    _require_float(q, 'q', tuple(q_target_executed.shape))
    _share_device([q_target_executed, q], 'joint_tracking inputs')
    return (q_target_executed - q).square().sum(-1)


def center_rate(local_y):
    """``min((local_y/0.20)^2, 4)`` (SDS 3.2)."""
    b = _batch(_require_float(local_y, 'local_y'), 'local_y')
    if tuple(local_y.shape) != (b,):
        raise ValueError('local_y must be [batch]')
    return (local_y / CENTER_LATERAL_SCALE).square().clamp_max(CENTER_SATURATION)


def heading_rate(yaw_world):
    """``1 - cos(e_world)`` with the route yaw axis (SDS 3.2)."""
    b = _batch(_require_float(yaw_world, 'yaw_world'), 'yaw_world')
    if tuple(yaw_world.shape) != (b,):
        raise ValueError('yaw_world must be [batch]')
    return 1. - (yaw_world - ROUTE_YAW_AXIS).cos()


def joint_limits_rate(q, soft_limits=None):
    """``sum(max(q_min_soft-q,0) + max(q-q_max_soft,0))`` (SDS 3.2)."""
    b = _batch(_require_float(q, 'q'), 'q')
    if q.ndim != 2:
        raise ValueError('q must be [batch, joints]')
    if soft_limits is None:
        soft_limits = _default_soft_limits(q)
    _require_float(soft_limits, 'soft_limits', (b, q.shape[1], 2))
    _share_device([q, soft_limits], 'joint_limits inputs')
    lower = (soft_limits[..., 0] - q).clamp_min(0.)
    upper = (q - soft_limits[..., 1]).clamp_min(0.)
    return (lower + upper).sum(-1)


# ------------------------------------------------------------------ air time
def feet_air_time_bonus(first_contact, air_time_before_reset, has_contacted_once):
    """``sum_f(I_first_contact,f * (air_time_before_reset,f - 0.10))``.

    Unweighted seconds; the caller applies ``dt*weight``.  A foot that has never
    touched the ground only initialises its state and is excluded.
    """
    b = _batch(_require_bool(first_contact, 'first_contact'), 'first_contact')
    if tuple(first_contact.shape) != (b, 4):
        raise ValueError('first_contact must be [batch,4]')
    _require_float(air_time_before_reset, 'air_time_before_reset', (b, 4))
    _require_bool(has_contacted_once, 'has_contacted_once', (b, 4))
    _share_device([first_contact, air_time_before_reset, has_contacted_once],
                  'air_time inputs')
    paid = first_contact & has_contacted_once
    return ((air_time_before_reset - AIR_TIME_TARGET) * paid.to(air_time_before_reset.dtype)).sum(-1)


class AirTimeState:
    """Per-foot contact/air-time state with the SDS 3.4 settle-then-clear order.

    ``air_time`` accumulates on every raw ``False`` frame once the foot has
    contacted the ground; ``previous_raw_contact`` is the previous raw contact,
    ``previous_filtered_contact`` caches ``filtered = raw | previous_raw`` so a
    single-frame dropout never creates a filtered rising edge.  A landing pays
    ``air_time-0.10`` only when ``raw & filtered & ~previous_filtered`` and
    ``air_time > 0``; every raw recovery clears ``air_time`` whether or not it
    paid (Q001-1).  ``update`` returns the unweighted bonus sum, i.e. the
    ``f_i`` of SDS 3.2 that the computer multiplies by ``dt*weight`` exactly
    once.
    """

    def __init__(self, num_envs, device='cpu'):
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        b = self.num_envs
        self.previous_raw_contact = torch.zeros(b, 4, dtype=torch.bool, device=self.device)
        self.previous_filtered_contact = torch.zeros(b, 4, dtype=torch.bool,
                                                     device=self.device)
        self.has_contacted_once = torch.zeros(b, 4, dtype=torch.bool, device=self.device)
        self.air_time = torch.zeros(b, 4, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ ids
    def _ids(self, ids):
        if ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(ids, torch.Tensor):
            if ids.device != self.device:
                raise TypeError('ids must be on the state device')
            if ids.dtype == torch.bool:
                if ids.shape != (self.num_envs,):
                    raise ValueError('boolean ids must have shape [num_envs]')
                return torch.nonzero(ids, as_tuple=True)[0]
            if ids.dtype != torch.int64 or ids.ndim != 1:
                raise TypeError('ids must be an int64 vector or bool mask')
            ids = ids
        else:
            ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        if ids.numel() and ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('ids out of range')
        return ids

    def reset(self, ids=None):
        ids = self._ids(ids)
        with torch.no_grad():
            self.previous_raw_contact[ids] = False
            self.previous_filtered_contact[ids] = False
            self.has_contacted_once[ids] = False
            self.air_time[ids] = 0.

    # --------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, contact_raw, dt):
        """Advance one control step and return the unweighted air-time bonus."""
        dt = _validate_dt(dt)
        b = self.num_envs
        _require_bool(contact_raw, 'contact_raw', (b, 4))
        if contact_raw.device != self.device:
            raise TypeError('contact_raw must be on the state device')
        # Q001-1 (binding addendum, verbatim pseudocode, all right-hand sides
        # are the state before this update):
        #   filtered = raw | previous_raw_contact
        #   landing  = raw & filtered & ~previous_filtered_contact
        #              & has_contacted_once & (air_time > 0)
        #   bonus    = where(landing, air_time - 0.10, 0)
        #   next_air = where(raw, 0, where(has_contacted_once, air_time + dt, 0))
        #   next_has = has_contacted_once | raw
        # Raw air time accumulates even on the first lost frame; any raw
        # recovery clears it whether or not it pays, so a short dropout can
        # never be added to a later real air phase.  The landing frame never
        # appends dt and the filtered falling edge never pays.
        raw = contact_raw
        filtered = raw | self.previous_raw_contact
        landing = (raw & filtered & ~self.previous_filtered_contact
                   & self.has_contacted_once & (self.air_time > 0))
        bonus = torch.where(landing, self.air_time - AIR_TIME_TARGET,
                            torch.zeros_like(self.air_time))
        next_air = torch.where(raw, torch.zeros_like(self.air_time),
                               torch.where(self.has_contacted_once,
                                           self.air_time + dt, torch.zeros_like(self.air_time)))
        next_has = self.has_contacted_once | raw
        self.previous_raw_contact = raw.clone()
        self.previous_filtered_contact = filtered.clone()
        self.has_contacted_once = next_has
        self.air_time = next_air
        return bonus.sum(-1)

    def state_dict(self):
        return dict(previous_raw_contact=self.previous_raw_contact.clone(),
                    previous_filtered_contact=self.previous_filtered_contact.clone(),
                    has_contacted_once=self.has_contacted_once.clone(),
                    air_time=self.air_time.clone())

    def load_state_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError('air-time state must be a dict')
        b = self.num_envs
        expected = dict(previous_raw_contact=(b, 4), previous_filtered_contact=(b, 4),
                        has_contacted_once=(b, 4), air_time=(b, 4))
        for name, shape in expected.items():
            if name not in value:
                raise ValueError(f'air-time state missing {name}')
            tensor = value[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f'air-time state {name} must be a tensor {shape}')
            if name == 'air_time':
                _require_float(tensor, name, shape)
            else:
                _require_bool(tensor, name, shape)
            setattr(self, name, tensor.to(self.device).clone())


# ---------------------------------------------------------------- frame types
@dataclass
class RewardFrame:
    executed_command: torch.Tensor
    local_x: torch.Tensor
    local_x_prev: torch.Tensor
    body_lin_vel_b: torch.Tensor
    body_ang_vel_b: torch.Tensor
    gravity_b: torch.Tensor
    base_height_above_surface: torch.Tensor
    q: torch.Tensor
    qdot: torch.Tensor
    q_target_executed: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    foot_position_world: torch.Tensor
    foot_velocity_world: torch.Tensor
    foot_force_world: torch.Tensor
    contact_raw: torch.Tensor
    contact_filtered: torch.Tensor
    tile_ids: torch.Tensor
    local_y: torch.Tensor
    yaw_world: torch.Tensor
    resets: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    # Additive W0 frame field: SDS 3.2 H(x_foot,y_foot) is not carried by any
    # frozen tensor.  See coordination/QA_spec_rewards.md.
    foot_surface_height: torch.Tensor


@dataclass
class RewardResult:
    raw_terms: dict
    weighted_terms: dict
    total: torch.Tensor


def validate_frame(frame):
    """Strict shape/dtype/finiteness check shared by the computer and tests."""
    if not isinstance(frame, RewardFrame):
        raise TypeError('frame must be a RewardFrame')
    command = _require_float(frame.executed_command, 'executed_command')
    b = _batch(command, 'executed_command')
    if tuple(command.shape) != (b,):
        raise ValueError('executed_command must be [batch]')
    checks = {
        'local_x': ((b,), 'float'), 'local_x_prev': ((b,), 'float'),
        'base_height_above_surface': ((b,), 'float'), 'local_y': ((b,), 'float'),
        'yaw_world': ((b,), 'float'),
        'body_lin_vel_b': ((b, 3), 'float'), 'body_ang_vel_b': ((b, 3), 'float'),
        'gravity_b': ((b, 3), 'float'),
        'q': ((b, 12), 'float'), 'qdot': ((b, 12), 'float'),
        'q_target_executed': ((b, 12), 'float'),
        'action': ((b, 12), 'float'), 'previous_action': ((b, 12), 'float'),
        'foot_position_world': ((b, 4, 3), 'float'),
        'foot_velocity_world': ((b, 4, 3), 'float'),
        'foot_force_world': ((b, 4, 3), 'float'),
        'foot_surface_height': ((b, 4), 'float'),
        'contact_raw': ((b, 4), 'bool'), 'contact_filtered': ((b, 4), 'bool'),
        'tile_ids': ((b,), 'int'),
        'resets': ((b,), 'bool'), 'success': ((b,), 'bool'), 'failure': ((b,), 'bool'),
    }
    tensors = [command]
    for name, (shape, kind) in checks.items():
        value = getattr(frame, name)
        if kind == 'float':
            _require_float(value, name, shape)
        elif kind == 'bool':
            _require_bool(value, name, shape)
        else:
            _require_int(value, name, shape)
        tensors.append(value)
    _share_device(tensors, 'reward frame tensors')
    if ((frame.tile_ids < 0) | (frame.tile_ids >= _TILE_COUNT)).any():
        raise ValueError('tile_ids outside the v0.6 terrain table')
    return b


# ---------------------------------------------------------------- state types
class RewardState:
    """Per-environment reward differencing state plus the air-time state."""

    def __init__(self, num_envs, *, action_dim=12, joint_dim=12, device='cpu'):
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.joint_dim = int(joint_dim)
        self.device = torch.device(device)
        b = self.num_envs
        nan = float('nan')
        self.previous_action = torch.full((b, self.action_dim), nan,
                                          dtype=torch.float32, device=self.device)
        self.previous_qdot = torch.full((b, self.joint_dim), nan,
                                        dtype=torch.float32, device=self.device)
        self.previous_local_x = torch.full((b,), nan, dtype=torch.float32,
                                           device=self.device)
        self.initialized = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.air_time = AirTimeState(b, device=self.device)

    def _ids(self, ids):
        if ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(ids, torch.Tensor):
            if ids.device != self.device:
                raise TypeError('ids must be on the state device')
            if ids.dtype == torch.bool:
                if ids.shape != (self.num_envs,):
                    raise ValueError('boolean ids must have shape [num_envs]')
                return torch.nonzero(ids, as_tuple=True)[0]
            if ids.dtype != torch.int64 or ids.ndim != 1:
                raise TypeError('ids must be an int64 vector or bool mask')
        else:
            ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        if ids.numel() and ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('ids out of range')
        return ids

    def reset(self, ids=None):
        ids = self._ids(ids)
        with torch.no_grad():
            self.previous_action[ids] = float('nan')
            self.previous_qdot[ids] = float('nan')
            self.previous_local_x[ids] = float('nan')
            self.initialized[ids] = False
        self.air_time.reset(ids)

    def state_dict(self):
        return dict(previous_action=self.previous_action.clone(),
                    previous_qdot=self.previous_qdot.clone(),
                    previous_local_x=self.previous_local_x.clone(),
                    initialized=self.initialized.clone(),
                    air_time=self.air_time.state_dict())

    def load_state_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError('reward state must be a dict')
        b = self.num_envs
        for name, shape, kind in (('previous_action', (b, self.action_dim), 'float'),
                                  ('previous_qdot', (b, self.joint_dim), 'float'),
                                  ('previous_local_x', (b,), 'float'),
                                  ('initialized', (b,), 'bool')):
            if name not in value:
                raise ValueError(f'reward state missing {name}')
            tensor = value[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f'reward state {name} must be a tensor {shape}')
            if kind == 'float':
                _require_float(tensor, name, shape)
            else:
                _require_bool(tensor, name, shape)
            setattr(self, name, tensor.to(self.device).clone())
        if 'air_time' not in value:
            raise ValueError('reward state missing air_time')
        self.air_time.load_state_dict(value['air_time'])


# ------------------------------------------------------------- forward/events
def forward_progress_reward(local_x, previous_local_x, *, resets):
    """``0.4*(x_t-x_(t-1))`` in metres; reset rows are exactly zero (SDS 3.5)."""
    b = _batch(_require_float(local_x, 'local_x'), 'local_x')
    _require_float(previous_local_x, 'previous_local_x', (b,))
    _require_bool(resets, 'resets', (b,))
    _share_device([local_x, previous_local_x, resets], 'forward inputs')
    delta = torch.where(resets, torch.zeros_like(local_x), local_x - previous_local_x)
    return FORWARD_PROGRESS_PER_M * delta


def event_rewards(success, failure):
    """One-shot ``+10`` safe summit / ``-10`` failure; failure wins (SDS 3.1)."""
    b = _batch(_require_bool(success, 'success'), 'success')
    _require_bool(failure, 'failure', (b,))
    _share_device([success, failure], 'event inputs')
    return (SUCCESS_REWARD * (success & ~failure).to(torch.float32)
            + FAILURE_REWARD * failure.to(torch.float32))


# --------------------------------------------------------------- the computer
class RewardComputer:
    """Stateful SDS 3 reward composition on the frozen W0 4.2 interface."""

    def __init__(self, weights=None, dt=CONTROL_DT):
        self.dt = _validate_dt(dt)
        if weights is None:
            weights = REWARD_WEIGHTS
        if not isinstance(weights, dict) and not hasattr(weights, 'items'):
            raise TypeError('weights must be a mapping')
        resolved = {name: 0. for name in REWARD_TERMS}
        for name, value in dict(weights).items():
            if name not in resolved:
                raise ValueError(f'unknown reward term: {name!r}')
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError(f'reward weight {name} must be finite')
            resolved[name] = float(value)
        self.weights = resolved
        self.state: Optional[RewardState] = None
        self._pending_resets = []

    # ------------------------------------------------------------------ reset
    def reset(self, ids=None):
        if self.state is None:
            self._pending_resets.append(ids)
            return
        self.state.reset(ids)

    def _ensure_state(self, batch, device):
        if self.state is None:
            self.state = RewardState(batch, device=device)
            pending, self._pending_resets = self._pending_resets, []
            for ids in pending:
                self.state.reset(ids)
        elif self.state.num_envs != batch or self.state.device != device:
            raise ValueError('reward batch size/device changed; reset the computer')

    # ---------------------------------------------------------------- call
    @torch.no_grad()
    def __call__(self, frame):
        b = validate_frame(frame)
        self._ensure_state(b, frame.executed_command.device)
        state = self.state
        dt = self.dt
        resets = frame.resets
        if resets.any():
            state.reset(torch.nonzero(resets, as_tuple=True)[0])

        # Register the frame's own contact first so the air-time state settles
        # against the pre-reset accumulation and only then clears it.
        air_bonus = state.air_time.update(frame.contact_raw, dt)

        # After the reset pass above, ``initialized`` is false exactly for the
        # reset rows.  First-call rows use the frame's own pre-step snapshot; a
        # reset row never differences across the episode boundary.
        has_previous = ~resets
        initialized = state.initialized
        previous_action = torch.where(initialized[:, None], state.previous_action,
                                      frame.previous_action)
        previous_qdot = torch.where(initialized[:, None], state.previous_qdot,
                                    frame.qdot)
        previous_x = torch.where(initialized, state.previous_local_x,
                                 frame.local_x_prev)

        raw = {}
        raw['tracking_lin_vel'] = tracking_lin_vel_rate(frame.body_lin_vel_b,
                                                        frame.executed_command)
        raw['tracking_ang_vel'] = tracking_ang_vel_rate(frame.body_ang_vel_b)
        raw['lin_vel_z'] = lin_vel_z_rate(frame.body_lin_vel_b)
        raw['orientation_roll'] = orientation_roll_rate(frame.gravity_b)
        raw['base_height'] = base_height_rate(frame.base_height_above_surface)
        raw['action_rate'] = action_rate_rate(frame.action, previous_action) * has_previous
        raw['similar_to_default'] = similar_to_default_rate(frame.q)
        raw['dof_acc'] = dof_acc_rate(frame.qdot, previous_qdot, dt=dt) * has_previous
        raw['dof_vel'] = dof_vel_rate(frame.qdot)
        raw['ang_vel_xy'] = ang_vel_xy_rate(frame.body_ang_vel_b)
        raw['feet_air_time'] = air_bonus
        raw['foot_slip'] = foot_slip_rate(frame.foot_velocity_world,
                                          frame.contact_filtered)
        raw['foot_clearance'] = foot_clearance_rate(
            frame.foot_position_world, frame.foot_velocity_world,
            frame.foot_surface_height, frame.contact_filtered, frame.tile_ids)
        raw['joint_tracking'] = joint_tracking_rate(frame.q_target_executed, frame.q)
        raw['center'] = center_rate(frame.local_y)
        raw['heading'] = heading_rate(frame.yaw_world)
        raw['joint_limits'] = joint_limits_rate(frame.q)

        weighted = {}
        for name in REWARD_TERMS:
            weighted[name] = dt * self.weights[name] * raw[name]
        delta_x = torch.where(resets, torch.zeros_like(frame.local_x),
                              frame.local_x - previous_x)
        raw['forward_progress'] = delta_x
        weighted['forward_progress'] = FORWARD_PROGRESS_PER_M * delta_x
        success_paid = (frame.success & ~frame.failure)
        raw['success'] = success_paid.to(torch.float32)
        raw['failure'] = frame.failure.to(torch.float32)
        weighted['success'] = SUCCESS_REWARD * raw['success']
        weighted['failure'] = FAILURE_REWARD * raw['failure']

        continuous = torch.stack([weighted[name] for name in REWARD_TERMS]).sum(0)
        total = (continuous + weighted['forward_progress'] + weighted['success']
                 + weighted['failure'])

        state.previous_action = frame.action.detach().clone()
        state.previous_qdot = frame.qdot.detach().clone()
        state.previous_local_x = frame.local_x.detach().clone()
        state.initialized = torch.ones_like(state.initialized)
        return RewardResult(raw_terms=raw, weighted_terms=weighted, total=total)
