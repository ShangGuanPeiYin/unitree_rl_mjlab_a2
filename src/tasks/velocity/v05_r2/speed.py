"""v0.5-r2 speed windows: F0 time window, P position window, S legacy position window.

SDS 0.5-r2 revision 1.0 sections 6.1, 6.2 and 7.6 define three *different*
measurement windows.  They are implemented here as three bounded,
per-environment accumulators (float64 statistics, raw evidence stays float32):

* ``f0_time``      -- control steps 101..500 of the observed 500-step episode
                      ((2,10] s, exactly 400 samples once complete).
* ``p_position``   -- first body ``x >= -0.5`` up to and including the first
                      ``x >= N*0.2+0.5``; the window is latched, so every
                      consecutive control sample after entry is counted even if
                      the body later moves back below the entry line.
* ``s_position``   -- the frozen v0.3/v0.4 stair oracle: first ``x >= 0`` up to
                      and including first ``x >= N*0.2``.

Derivation (SDS 0.5-r2 section 8.1): this file is a line-by-line derivative of
``v05/speed.py``.  Only the version label, the docstring and the new *identity*
record fields (C1/C8 of the r2 change list) differ; every window boundary,
threshold, counter, dtype and ``SpeedResult`` semantic is unchanged.  ``route``
information is never consumed here (SDS 4.8/9.3-20: ``route/*`` is read-only
diagnostics, not a speed criterion).

Per-slot routing is by the environment's **real tile** (:class:`SlotSpeedRouter`),
never by the supervisor stage and never by a whole-batch stage assignment.  A
flat slot (N=0) is never passed to a position window, so the legacy
``SegmentSpeedState``/CPU oracle never receives ``N=0``.

``cpu_speed_oracle(raw, kind, c, N)`` is the independent NumPy float64
recomputation of one raw episode (SDS 9.2 item 13 requires <=1e-6 against the old
S oracle); it deliberately uses direct indexing instead of the online ring
arithmetic so the two implementations can disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional

import numpy as np
import torch

from . import spec as _spec
from .spec import (CONTROL_DT, DEVIATION_THRESHOLD, F0_OBSERVE_STEPS,
                   F0_WINDOW_STEPS, LONGEST_DEVIATION_MAX, SPEED_EPSILON,
                   SPEED_WINDOW, STAGE_BY_TILE, FLAT_TILE_ID)

#: SDS 6.1: the two remaining thresholds are mean_relative_error <= 0.20 and
#: smooth_mae_relative <= 0.30, both with the +1e-6 epsilon.
MEAN_RELATIVE_ERROR_MAX = .20
SMOOTH_MAE_MAX = .30
MAX_LONGEST_DEVIATION = LONGEST_DEVIATION_MAX

WINDOW_KINDS = ('f0_time', 'p_position', 's_oracle')
P_ENTRY_X = -.5
P_EXIT_MARGIN = .5
STAIR_STEP_DEPTH = .2
F0_LATERAL_MAX = .20
F0_HEADING_MAX = math.radians(10.)

#: SDS 6.1 boundary rule: every threshold comparison uses the +1e-6 epsilon.
THRESHOLD_EPSILON = SPEED_EPSILON


# ----------------------------------------------------------- r2 identity record
def _require(name):
    """Read one r2 spec constant, failing closed instead of inventing a value."""
    try:
        return getattr(_spec, name)
    except AttributeError:
        raise RuntimeError(
            f'v0.5-r2 spec exports no {name}; the r2 identity record cannot be '
            'built from a default (SDS 0.4: single numeric source)') from None


def r2_identity():
    """The r2 version identity attached to every retained speed record.

    The values come from :mod:`v05_r2.spec` only: a speed record that carried only
    the numbers could be mistaken for an r1 record, which SDS 12.3 forbids.
    """
    architecture = getattr(_spec, 'ARCHITECTURE', None)
    if architecture is None:
        architecture = _require('ROUTE_STATE')['architecture']
    return dict(task=_require('TASK'), code_version=_require('CODE_VERSION'),
                recipe=_require('RECIPE_VERSION'), sds_revision=_require('SDS_REVISION'),
                checkpoint_schema=_require('SCHEMA_VERSION'),
                architecture=architecture,
                stair_protocol=_require('PROTOCOL_ID'),
                flat_protocol=_require('FLAT_PROTOCOL_ID'))


R2_IDENTITY = r2_identity()


# ------------------------------------------------- route_state contract binding
def _route_contract():
    """Bind the SDS 4.2 ``route_state`` constants from the single spec source.

    Fail closed: the formula's scale, saturation point and heading axis may not be
    re-invented in an implementation module (SDS 0.4/1.2, Q003-r2 decision 2.2).
    The nested ``spec.ROUTE_STATE`` mapping is preferred; individual
    ``spec.ROUTE_*`` scalars are accepted for the same values.  This lives here --
    in the only r2 module that needs neither the simulator nor a model -- so that
    ``env``, ``evaluation`` and any offline recomputation share one binding.
    """
    state = getattr(_spec, 'ROUTE_STATE', None)
    state = state if isinstance(state, dict) else {}
    wanted = ((('lateral_scale', 'lateral_scale_m'), ('ROUTE_LATERAL_SCALE', 'ROUTE_SCALE')),
              (('clip', 'lateral_clip'), ('ROUTE_CLIP', 'ROUTE_LATERAL_CLIP')),
              (('yaw_axis',), ('ROUTE_YAW_AXIS', 'YAW_AXIS')))
    values, missing = {}, []
    for keys, aliases in wanted:
        key = keys[0]
        if any(name in state for name in keys):
            values[key] = float(next(state[name] for name in keys if name in state))
            continue
        for alias in aliases:
            if hasattr(_spec, alias):
                values[key] = float(getattr(_spec, alias))
                break
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(
            'v0.5-r2 spec exports no route_state constants for '
            f'{missing}; route_error may not be built from a local default '
            '(SDS 4.2: single numeric source)')
    for key, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(f'v0.5-r2 route_state {key} must be finite, got {value!r}')
    if values['lateral_scale'] <= 0 or values['clip'] <= 0:
        raise RuntimeError('route_state scale/clip must be strictly positive')
    if values['yaw_axis'] != 0.0:
        raise RuntimeError(
            'route_state yaw_axis must be 0.0: the current atlas only translates its '
            'tiles and a rotated atlas needs an upgraded contract (SDS 4.2/13.5), '
            f'got {values["yaw_axis"]!r}')
    return dict(values, components=('clip(local_y/0.20,-4,4)', 'sin(e_world)',
                                    'cos(e_world)'), units='radians', size=3,
                normalized=False, noise=False, per_environment_tile=True,
                flat_branch='none_tile_0_uses_the_same_formula',
                yaw_axis_premise='current atlas tiles are translations only; '
                                 'no rotation field',
                e_world='wrap_to_pi(yaw_world - yaw_axis)',
                wrap='atan2(sin(.), cos(.))',
                deployment_label='ideal_route_state')


ROUTE_STATE = _route_contract()
ROUTE_LATERAL_SCALE = ROUTE_STATE['lateral_scale']
ROUTE_CLIP = ROUTE_STATE['clip']
ROUTE_YAW_AXIS = ROUTE_STATE['yaw_axis']
ROUTE_COMPONENTS = ROUTE_STATE['components']


def route_state_contract():
    """JSON-safe copy of the bound SDS 4.2 route contract (identity/reporting)."""
    return dict(ROUTE_STATE, lateral_scale=ROUTE_LATERAL_SCALE, clip=ROUTE_CLIP,
                yaw_axis=ROUTE_YAW_AXIS)



def tile_window_kind(tile_id):
    """Map one environment's *real* atlas tile to its measurement window."""
    if isinstance(tile_id, bool) or not isinstance(tile_id, Integral):
        raise TypeError('tile_id must be an integer')
    tile = int(tile_id)
    if tile == FLAT_TILE_ID:
        return 'f0_time'
    if tile not in STAGE_BY_TILE:
        raise ValueError(f'tile {tile} is not a v0.5-r2 stage tile')
    stage = STAGE_BY_TILE[tile]
    if stage.N <= 0:
        return 'f0_time'
    if stage.id.startswith('P'):
        return 'p_position'
    if stage.id.startswith('S'):
        return 's_oracle'
    raise ValueError(f'stage {stage.id} has no v0.5-r2 speed window')


@dataclass(frozen=True)
class SpeedResult:
    """One environment's window metrics (SDS 6.1) as plain Python scalars.

    Semantics are identical to the r1 ``SpeedResult`` (SDS 6.1: an unavailable
    metric is ``None``, never NaN, so every consumer can write JSON with
    ``allow_nan=False``).
    """

    samples: int
    #: ``None`` whenever the window has no usable sample; never NaN, so every
    #: consumer can write JSON with ``allow_nan=False``.
    mean_vx: Optional[float]
    mean_relative_error: Optional[float]
    smooth_mae_relative: Optional[float]
    longest_deviation: Optional[int]
    valid: bool
    passed: bool
    window_kind: str
    coverage_complete: bool

    def as_dict(self):
        return dict(samples=self.samples, mean_vx=self.mean_vx,
                    mean_relative_error=self.mean_relative_error,
                    smooth_mae_relative=self.smooth_mae_relative,
                    longest_deviation=self.longest_deviation,
                    valid=self.valid, passed=self.passed,
                    window_kind=self.window_kind,
                    coverage_complete=self.coverage_complete,
                    identity=dict(R2_IDENTITY))


def _window_bounds(kind):
    if kind not in WINDOW_KINDS:
        raise ValueError(f'window kind must be one of {WINDOW_KINDS}')
    return kind


def _thresholds(N):
    if isinstance(N, bool) or not isinstance(N, Integral):
        raise TypeError('N must be an integer')
    if int(N) < 1:
        raise ValueError('a position window needs N >= 1; flat(N=0) uses the F0 time window')
    return float(int(N)) * STAIR_STEP_DEPTH


class SpeedWindowState:
    """Bounded per-environment window accumulator for one window kind.

    ``update`` consumes exactly one control sample per environment and returns
    cloned per-environment sample masks.  Only ``enabled`` rows advance; a
    disabled row (parking diagnostic) neither binds a command nor enters any
    window.  Statistics are accumulated in float64 while every input stays the
    float32 physical value, so the offline oracle can be compared at 1e-8.
    """

    def __init__(self, num_envs, kind, device='cpu'):
        _window_bounds(kind)
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = int(num_envs)
        self.kind = kind
        self.device = torch.device(device)
        b = self.num_envs

        def zeros(*shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.bound = zeros(b, dtype=torch.bool)
        self.command = zeros(b, dtype=torch.float32)
        self.num_steps = zeros(b, dtype=torch.int64)
        self.entered = zeros(b, dtype=torch.bool)
        self.ended = zeros(b, dtype=torch.bool)
        self.start_step = torch.full((b,), -1, dtype=torch.int64, device=self.device)
        self.end_step = self.start_step.clone()
        self.control_samples = zeros(b, dtype=torch.int64)
        self.window_samples = zeros(b, dtype=torch.int64)
        self.sum_vx = zeros(b, dtype=torch.float64)
        self.ring = zeros(b, SPEED_WINDOW, dtype=torch.float64)
        self.window_sum = zeros(b, dtype=torch.float64)
        self.window_count = zeros(b, dtype=torch.int64)
        self.sum_smooth_abs = zeros(b, dtype=torch.float64)
        self.current_deviation = zeros(b, dtype=torch.int64)
        self.longest_deviation = zeros(b, dtype=torch.int64)
        self.max_abs_y = zeros(b, dtype=torch.float64)
        self.max_abs_yaw = zeros(b, dtype=torch.float64)
        self.last_step = torch.full((b,), -1, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------- validation
    def _tensor(self, value, name, dtype):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype:
            raise TypeError(f'{name} must be a {dtype} torch tensor')
        if value.shape != (self.num_envs,) or value.device != self.device:
            raise ValueError(f'{name} must have shape [{self.num_envs}] on {self.device}')

    def _check(self, step, local_x, body_vx, executed_c, num_steps, enabled, local_y, yaw):
        for name, value, dtype in (('step', step, torch.int64),
                                   ('local_x', local_x, torch.float32),
                                   ('body_vx', body_vx, torch.float32),
                                   ('executed_c', executed_c, torch.float32),
                                   ('num_steps', num_steps, torch.int64)):
            self._tensor(value, name, dtype)
        self._tensor(enabled, 'enabled', torch.bool)
        if local_y is not None:
            self._tensor(local_y, 'local_y', torch.float32)
        if yaw is not None:
            self._tensor(yaw, 'yaw', torch.float32)
        finite = [('local_x', local_x), ('body_vx', body_vx), ('executed_c', executed_c)]
        if local_y is not None:
            finite.append(('local_y', local_y))
        if yaw is not None:
            finite.append(('yaw', yaw))
        for name, value in finite:
            if not torch.isfinite(value).all():
                raise ValueError(f'nonfinite {name} is a numerical fault')
        if (step < 0).any():
            raise ValueError('step must be nonnegative')
        if (enabled & (executed_c <= 0)).any():
            raise ValueError('an enabled sample needs a strictly positive executed command')
        advancing = enabled & self.bound & (self.last_step >= 0)
        if (advancing & (step != self.last_step + 1)).any():
            raise ValueError('an enabled episode must supply every consecutive control sample')
        if self.kind != 'f0_time':
            if (enabled & (num_steps < 1)).any():
                raise ValueError('position windows never accept N=0; flat slots are routed to F0')
        if (enabled & self.bound & (executed_c != self.command)).any():
            raise ValueError('the executed command must stay constant within one episode')
        if self.kind != 'f0_time' and (enabled & self.bound & (num_steps != self.num_steps)).any():
            raise ValueError('the real tile N must stay constant within one episode')

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, *, step, local_x, body_vx, executed_c, num_steps,
               enabled=None, local_y=None, yaw=None):
        """Consume one control result for every environment row."""
        if enabled is None:
            enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._check(step, local_x, body_vx, executed_c, num_steps, enabled, local_y, yaw)
        if self.kind != 'f0_time' and (enabled & (num_steps < 1)).any():
            raise ValueError('position windows never accept N=0; flat slots are routed to F0')

        entered_before = self.entered.clone()
        ended_before = self.ended.clone()
        new_binding = enabled & ~self.bound
        self.command[new_binding] = executed_c[new_binding]
        self.num_steps[new_binding] = num_steps[new_binding]
        self.bound |= enabled

        if self.kind == 'f0_time':
            upper = int(F0_WINDOW_STEPS[1])
            lower = int(F0_WINDOW_STEPS[0])
            sample = enabled & (step >= lower) & (step <= upper)
            entered_now = enabled & (step >= lower) & ~self.entered
            ended_now = enabled & (step >= upper) & ~self.ended
            self.start_step = torch.where(entered_now, step, self.start_step)
            self.end_step = torch.where(ended_now, step, self.end_step)
        else:
            steps = num_steps.to(torch.float64)
            if self.kind == 'p_position':
                enter_x = torch.full_like(local_x.to(torch.float64), P_ENTRY_X)
                exit_x = steps * STAIR_STEP_DEPTH + P_EXIT_MARGIN
            else:
                enter_x = torch.zeros_like(local_x.to(torch.float64))
                exit_x = steps * STAIR_STEP_DEPTH
            x = local_x.to(torch.float64)
            entered_now = enabled & ~self.entered & (x >= enter_x)
            ended_now = (enabled & (self.entered | entered_now) & ~self.ended
                         & (x >= exit_x))
            self.start_step = torch.where(entered_now, step, self.start_step)
            self.end_step = torch.where(ended_now, step, self.end_step)
            sample = enabled & (self.entered | entered_now) & ~ended_before
        self.entered |= entered_now
        self.ended |= ended_now
        self.control_samples += enabled.long()

        if self.kind == 'f0_time' and local_y is not None and yaw is not None:
            observed = enabled & (step <= int(F0_OBSERVE_STEPS))
            self.max_abs_y = torch.where(observed, torch.maximum(
                self.max_abs_y, local_y.to(torch.float64).abs()), self.max_abs_y)
            self.max_abs_yaw = torch.where(observed, torch.maximum(
                self.max_abs_yaw, yaw.to(torch.float64).abs()), self.max_abs_yaw)

        vx = body_vx.to(torch.float64)
        rows = torch.nonzero(sample, as_tuple=True)[0]
        if rows.numel():
            columns = self.window_samples[rows] % SPEED_WINDOW
            oldest = self.ring[rows, columns]
            self.window_sum[rows] += vx[rows] - oldest
            self.ring[rows, columns] = vx[rows]
            self.window_samples[rows] += 1
            self.sum_vx[rows] += vx[rows]
            ready = rows[self.window_samples[rows] >= SPEED_WINDOW]
            if ready.numel():
                abs_error = (self.window_sum[ready] / SPEED_WINDOW
                             - self.command[ready].to(torch.float64)).abs()
                self.window_count[ready] += 1
                self.sum_smooth_abs[ready] += abs_error
                relative = abs_error / self.command[ready].to(torch.float64)
                deviating = relative > DEVIATION_THRESHOLD + SPEED_EPSILON
                self.current_deviation[ready] = torch.where(
                    deviating, self.current_deviation[ready] + 1,
                    torch.zeros_like(self.current_deviation[ready]))
                self.longest_deviation[ready] = torch.maximum(
                    self.longest_deviation[ready], self.current_deviation[ready])

        self.last_step = torch.where(enabled, step, self.last_step)
        return dict(sample=sample.clone(), entered_now=entered_now.clone(),
                    ended_now=ended_now.clone(), entered=self.entered.clone(),
                    ended=self.ended.clone(),
                    r2_identity=dict(R2_IDENTITY))

    # --------------------------------------------------------------- readout
    @torch.no_grad()
    def reset(self, env_ids=None):
        """Clear exactly the selected rows; other environments stay byte-identical."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor) or env_ids.device != self.device:
            raise TypeError('env_ids must be a device-local int64 vector or bool mask')
        elif env_ids.dtype == torch.bool:
            self._tensor(env_ids, 'env_ids', torch.bool)
            ids = torch.nonzero(env_ids, as_tuple=True)[0]
        elif env_ids.dtype == torch.int64 and env_ids.ndim == 1:
            ids = env_ids
            if ((ids < 0) | (ids >= self.num_envs)).any():
                raise ValueError('env_ids out of range')
        else:
            raise TypeError('env_ids must be an int64 vector or bool mask')
        negative = ('start_step', 'end_step', 'last_step')
        for name, tensor in vars(self).items():
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor[ids] = -1 if name in negative else 0

    @torch.no_grad()
    def snapshot(self):
        """Copy full-batch finite metric tensors together with validity masks."""
        safe_c = torch.where(self.bound, self.command.to(torch.float64), torch.ones_like(self.sum_vx))
        mean = self.sum_vx / self.window_samples.clamp_min(1).to(torch.float64)
        mean_error = torch.where(self.window_samples > 0, (mean - safe_c).abs() / safe_c,
                                 torch.zeros_like(mean))
        mae = self.sum_smooth_abs / self.window_count.clamp_min(1).to(torch.float64) / safe_c
        smooth_valid = self.window_count > 0
        coverage = self.coverage_complete()
        valid = coverage & (self.window_samples >= SPEED_WINDOW) & self.entered & self.ended
        passed = (valid & (mean_error <= MEAN_RELATIVE_ERROR_MAX + SPEED_EPSILON)
                  & (mae <= SMOOTH_MAE_MAX + SPEED_EPSILON)
                  & (self.longest_deviation <= MAX_LONGEST_DEVIATION))
        lateral = (self.max_abs_y <= F0_LATERAL_MAX) & (self.max_abs_yaw <= F0_HEADING_MAX + 1e-6)
        return dict(window_kind=self.kind, command_bound=self.bound, executed_c=self.command,
                    num_steps=self.num_steps, entered=self.entered, ended=self.ended,
                    start_step=self.start_step, end_step=self.end_step,
                    control_samples=self.control_samples, samples=self.window_samples,
                    window_count=self.window_count, mean_vx=mean,
                    mean_relative_error=mean_error, smooth_mae_relative=mae,
                    longest_deviation=self.longest_deviation, mean_valid=self.window_samples > 0,
                    smooth_valid=smooth_valid, coverage_complete=coverage,
                    valid=valid, passed=passed, max_abs_y=self.max_abs_y,
                    max_abs_yaw=self.max_abs_yaw, lateral_heading_pass=lateral,
                    r2_identity=dict(R2_IDENTITY))

    def coverage_complete(self):
        if self.kind == 'f0_time':
            complete = int(F0_WINDOW_STEPS[1]) - int(F0_WINDOW_STEPS[0]) + 1
            return (self.entered & self.ended
                    & (self.window_samples >= complete))
        return self.entered & self.ended

    def result(self, env_id) -> SpeedResult:
        """Per-environment :class:`SpeedResult` (SDS 6.1) before any reset."""
        if isinstance(env_id, bool) or not isinstance(env_id, Integral) or not 0 <= int(env_id) < self.num_envs:
            raise ValueError('env_id out of range')
        i = int(env_id)
        data = self.snapshot()
        smooth_valid = bool(data['smooth_valid'][i])
        return SpeedResult(
            samples=int(data['samples'][i]),
            mean_vx=float(data['mean_vx'][i]) if bool(data['mean_valid'][i]) else None,
            mean_relative_error=(float(data['mean_relative_error'][i])
                                 if bool(data['mean_valid'][i]) else None),
            smooth_mae_relative=float(data['smooth_mae_relative'][i]) if smooth_valid else None,
            longest_deviation=int(data['longest_deviation'][i]) if smooth_valid else None,
            valid=bool(data['valid'][i]), passed=bool(data['passed'][i]),
            window_kind=self.kind, coverage_complete=bool(data['coverage_complete'][i]))


class SlotSpeedRouter:
    """Route every environment to its own real-tile window (SDS 7.6).

    Flat slots use the F0 time window, single-step (P) tiles the P position
    window and multi-step (S) tiles the frozen S position window -- including
    *previous* slots, whose N/h are their own, not the supervisor stage's.
    """

    def __init__(self, num_envs, tile_ids, device='cpu'):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.flat = SpeedWindowState(num_envs, 'f0_time', device)
        self.p = SpeedWindowState(num_envs, 'p_position', device)
        self.s = SpeedWindowState(num_envs, 's_oracle', device)
        self.set_tiles(tile_ids)

    def set_tiles(self, tile_ids):
        tiles = torch.as_tensor(tile_ids, dtype=torch.int64, device=self.device).flatten()
        if tiles.numel() != self.num_envs:
            raise ValueError('tile ids must cover every environment')
        self.tile_ids = tiles.clone()
        kinds = [tile_window_kind(int(t)) for t in tiles.detach().cpu().tolist()]
        self.kind_names = kinds
        self.flat_mask = torch.tensor([k == 'f0_time' for k in kinds], device=self.device)
        self.p_mask = torch.tensor([k == 'p_position' for k in kinds], device=self.device)
        self.s_mask = torch.tensor([k == 's_oracle' for k in kinds], device=self.device)
        if not (self.flat_mask | self.p_mask | self.s_mask).all():
            raise RuntimeError('every environment must route to exactly one window')
        self.N = torch.tensor(
            [0 if int(t) == FLAT_TILE_ID else int(STAGE_BY_TILE[int(t)].N)
             for t in tiles.detach().cpu().tolist()],
            dtype=torch.int64, device=self.device)

    def update(self, *, step, local_x, body_vx, executed_c, enabled=None,
               local_y=None, yaw=None, tile_ids=None):
        if tile_ids is not None:
            self.set_tiles(tile_ids)
        if enabled is None:
            enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._tensor(enabled, 'enabled', torch.bool)
        self._tensor(step, 'step', torch.int64)
        self._tensor(local_x, 'local_x', torch.float32)
        self._tensor(body_vx, 'body_vx', torch.float32)
        self._tensor(executed_c, 'executed_c', torch.float32)
        N = self.N
        masks = {}
        self.flat.update(step=step, local_x=local_x, body_vx=body_vx,
                         executed_c=executed_c, num_steps=N,
                         enabled=enabled & self.flat_mask,
                         local_y=local_y, yaw=yaw)
        self.p.update(step=step, local_x=local_x, body_vx=body_vx,
                      executed_c=executed_c, num_steps=N,
                      enabled=enabled & self.p_mask)
        self.s.update(step=step, local_x=local_x, body_vx=body_vx,
                      executed_c=executed_c, num_steps=N,
                      enabled=enabled & self.s_mask)
        for i, kind in enumerate(self.kind_names):
            masks[i] = kind
        return masks

    def _tensor(self, value, name, dtype):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype:
            raise TypeError(f'{name} must be a {dtype} torch tensor')
        if value.shape != (self.num_envs,) or value.device != self.device:
            raise ValueError(f'{name} must have shape [{self.num_envs}] on {self.device}')

    def state_for(self, kind):
        if kind == 'f0_time':
            return self.flat
        if kind == 'p_position':
            return self.p
        if kind == 's_oracle':
            return self.s
        raise ValueError(f'unknown speed window {kind!r}')

    def reset(self, env_ids=None):
        for state in (self.flat, self.p, self.s):
            state.reset(env_ids)

    def results(self):
        """Recompute :class:`SpeedResult` for every environment."""
        snapshots = {kind: self.state_for(kind).snapshot() for kind in WINDOW_KINDS}
        out = []
        for i, kind in enumerate(self.kind_names):
            data = snapshots[kind]
            smooth_valid = bool(data['smooth_valid'][i])
            out.append(SpeedResult(
                samples=int(data['samples'][i]),
                mean_vx=float(data['mean_vx'][i]) if bool(data['mean_valid'][i]) else None,
                mean_relative_error=(float(data['mean_relative_error'][i])
                                     if bool(data['mean_valid'][i]) else None),
                smooth_mae_relative=(float(data['smooth_mae_relative'][i])
                                     if smooth_valid else None),
                longest_deviation=int(data['longest_deviation'][i]) if smooth_valid else None,
                valid=bool(data['valid'][i]), passed=bool(data['passed'][i]),
                window_kind=kind, coverage_complete=bool(data['coverage_complete'][i])))
        return out

    def snapshot(self):
        return {kind: self.state_for(kind).snapshot() for kind in WINDOW_KINDS}

    def episode_metrics(self, env_id, *, failure=False, climb_success=False):
        """JSON-safe per-environment terminal metrics (pre-reset snapshot)."""
        i = int(env_id)
        kind = self.kind_names[i]
        data = self.state_for(kind).snapshot()
        result = self.results()[i]
        reasons = []
        if not bool(data['entered'][i]):
            reasons.append('not_entered')
        if not bool(data['ended'][i]):
            reasons.append('not_ended')
        if not bool(data['coverage_complete'][i]):
            reasons.append('incomplete_coverage')
        if int(data['samples'][i]) < SPEED_WINDOW:
            reasons.append('insufficient_samples')
        smooth_valid = bool(data['smooth_valid'][i])
        lateral = bool(data['lateral_heading_pass'][i]) if kind == 'f0_time' else None
        return dict(
            window_kind=kind, executed_c=(float(data['executed_c'][i])
                                          if bool(data['command_bound'][i]) else None),
            num_steps=int(data['num_steps'][i]) if bool(data['command_bound'][i]) else None,
            samples=result.samples, window_count=int(data['window_count'][i]),
            entered=bool(data['entered'][i]), ended=bool(data['ended'][i]),
            coverage_complete=result.coverage_complete,
            start_step=int(data['start_step'][i]) if bool(data['entered'][i]) else None,
            end_step=int(data['end_step'][i]) if bool(data['ended'][i]) else None,
            control_samples=int(data['control_samples'][i]),
            mean_vx=result.mean_vx, mean_relative_error=result.mean_relative_error,
            smooth_mae_relative=result.smooth_mae_relative,
            longest_deviation=result.longest_deviation,
            longest_deviation_seconds=(None if result.longest_deviation is None
                                       else result.longest_deviation * CONTROL_DT),
            max_abs_y=float(data['max_abs_y'][i]) if kind == 'f0_time' else None,
            max_abs_yaw=float(data['max_abs_yaw'][i]) if kind == 'f0_time' else None,
            lateral_heading_pass=lateral,
            speed_valid=result.valid, speed_pass=result.passed,
            invalid_reasons=reasons, failure=bool(failure), climb_success=bool(climb_success),
            joint_success=bool(climb_success and result.passed and not failure),
            identity=dict(R2_IDENTITY))


# --------------------------------------------------------------- CPU oracle
def _as_vector(value, name, dtype):
    """Coerce one evidence vector to the oracle dtype without accepting objects."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if dtype == np.int64:
        if array.dtype.kind not in 'iu':
            raise TypeError(f'{name} must be an integer vector, got {array.dtype}')
        array = array.astype(np.int64, copy=False)
    else:
        if array.dtype.kind != 'f':
            raise TypeError(f'{name} must be a float vector, got {array.dtype}')
        array = array.astype(np.float64, copy=False)
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f'nonfinite {name} is a numerical fault')
    return array


def _smooth_metrics(body_vx, c):
    """Direct float64 25-sample sliding-window metrics (no ring arithmetic)."""
    values = np.asarray(body_vx, dtype=np.float64)
    count = len(values)
    if count == 0:
        return None, None, None
    if count < SPEED_WINDOW:
        return None, None, None
    means = np.array([np.sum(values[j - SPEED_WINDOW + 1:j + 1], dtype=np.float64) / SPEED_WINDOW
                      for j in range(SPEED_WINDOW - 1, count)], dtype=np.float64)
    errors = np.abs(means - float(c))
    mae = float(np.mean(errors, dtype=np.float64) / float(c))
    longest = 0
    current = 0
    for relative in errors / float(c) > DEVIATION_THRESHOLD + SPEED_EPSILON:
        current = current + 1 if relative else 0
        longest = max(longest, current)
    return mae, int(longest), int(len(means))


def cpu_speed_oracle(raw, kind, c, N):
    """Recompute one raw episode's window metrics with NumPy float64.

    ``raw`` maps ``body_vx``/``executed_c``/``local_x``/``step`` (and optionally
    ``local_y``/``yaw``) to one-dimensional lossless arrays of equal length; the
    steps must be consecutive.  ``kind`` is ``'f0_time'``, ``'p_position'`` or
    ``'s_oracle'``.  The return value is JSON-safe and contains the
    :class:`SpeedResult` fields plus boundary/coverage details and (F0 only) the
    lateral/heading constraint result.  ``route_error`` is deliberately not an
    input: SDS 4.8/9.3-20 keep route evidence out of every speed criterion.
    """
    if kind == 's_position':
        kind = 's_oracle'
    _window_bounds(kind)
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(float(c)) or float(c) <= 0:
        raise ValueError('c must be a positive finite executed command')
    if isinstance(N, bool) or not isinstance(N, Integral) or int(N) < 0:
        raise ValueError('N must be a nonnegative integer')
    if kind != 'f0_time' and int(N) < 1:
        raise ValueError('a position window needs N >= 1; flat(N=0) uses f0_time')
    if not isinstance(raw, dict):
        raise TypeError('raw must be a mapping of lossless evidence arrays')
    for key in ('body_vx', 'executed_c', 'local_x', 'step'):
        if key not in raw:
            raise ValueError('raw evidence is missing ' + key)
    body_vx = _as_vector(raw['body_vx'], 'body_vx', np.float32)
    executed_c = _as_vector(raw['executed_c'], 'executed_c', np.float32)
    local_x = _as_vector(raw['local_x'], 'local_x', np.float32)
    step = _as_vector(raw['step'], 'step', np.int64)
    total = len(step)
    if len(body_vx) != total or len(executed_c) != total or len(local_x) != total:
        raise ValueError('raw evidence arrays must have equal length')
    if total:
        if np.any(step < 0) or np.any(step[1:] - step[:-1] != 1):
            raise ValueError('raw evidence must be consecutive control samples')
        if np.any(executed_c <= 0) or np.any(executed_c != executed_c[0]):
            raise ValueError('the executed command must be positive and exactly constant')
        if float(executed_c[0]) != float(np.float32(c)):
            raise ValueError('the executed command differs from the requested float32 c')
    # The executed command is the float32 physical value, exactly as retained in
    # the evidence; the requested ``c`` is only a float32 consistency check.
    command = float(np.float64(executed_c[0])) if total else float(c)
    if total and float(np.float32(c)) != float(executed_c[0]):
        raise ValueError('the requested c is not the retained float32 executed command')
    x = local_x.astype(np.float64)
    start = stop = None
    coverage_complete = False
    if kind == 'f0_time':
        lower, upper = int(F0_WINDOW_STEPS[0]), int(F0_WINDOW_STEPS[1])
        indices = np.flatnonzero((step >= lower) & (step <= upper))
        complete = upper - lower + 1
        coverage_complete = bool(len(indices) >= complete)
        if len(indices):
            start, stop = int(step[indices[0]]), int(step[indices[-1]])
            window = body_vx[indices]
        else:
            window = np.zeros(0, dtype=np.float32)
    else:
        enter_x = P_ENTRY_X if kind == 'p_position' else 0.
        exit_x = float(int(N)) * STAIR_STEP_DEPTH + (P_EXIT_MARGIN if kind == 'p_position' else 0.)
        entries = np.flatnonzero(x >= enter_x)
        if len(entries):
            first = int(entries[0])
            exits = np.flatnonzero(x[first:] >= exit_x)
            last = first + int(exits[0]) if len(exits) else None
            end = total if last is None else last + 1
            window = body_vx[first:end]
            start, stop = int(step[first]), int(step[end - 1])
            coverage_complete = last is not None
        else:
            window = np.zeros(0, dtype=np.float32)
    samples = len(window)
    if samples:
        mean_vx = float(np.sum(window, dtype=np.float64) / samples)
        mean_error = abs(mean_vx - command) / command
    else:
        mean_vx, mean_error = None, None
    smooth, longest, window_count = _smooth_metrics(window, command)
    reasons = []
    if start is None:
        reasons.append('not_entered')
    elif stop is None or (kind != 'f0_time' and not coverage_complete):
        reasons.append('not_ended')
    if not coverage_complete:
        reasons.append('incomplete_coverage')
    if samples < SPEED_WINDOW:
        reasons.append('insufficient_samples')
    valid = not reasons
    passed = bool(valid and mean_error is not None
                  and mean_error <= MEAN_RELATIVE_ERROR_MAX + SPEED_EPSILON
                  and smooth is not None and smooth <= SMOOTH_MAE_MAX + SPEED_EPSILON
                  and longest is not None and longest <= MAX_LONGEST_DEVIATION)
    lateral = heading = lateral_pass = None
    if kind == 'f0_time':
        if 'local_y' in raw:
            lateral = float(np.max(np.abs(_as_vector(raw['local_y'], 'local_y', np.float32)
                                          .astype(np.float64))))
        if 'yaw' in raw:
            heading = float(np.max(np.abs(_as_vector(raw['yaw'], 'yaw', np.float32)
                                          .astype(np.float64))))
        if lateral is not None and heading is not None:
            lateral_pass = bool(lateral <= F0_LATERAL_MAX and heading <= F0_HEADING_MAX + 1e-6)
            if not lateral_pass and 'lateral_heading' not in reasons:
                reasons.append('lateral_heading')
                valid = False
                passed = False
    entered = start is not None
    ended = bool(coverage_complete)
    return dict(
        window_kind=kind, samples=int(samples), sample_count=int(samples),
        window_count=int(window_count or 0),
        entered=entered, ended=ended,
        start_step=start, end_step=stop, coverage_complete=bool(coverage_complete),
        mean_vx=mean_vx, mean_relative_error=mean_error,
        smooth_mae_relative=smooth, longest_deviation=longest,
        longest_deviation_seconds=(None if longest is None else longest * CONTROL_DT),
        max_abs_y=lateral, max_abs_yaw=heading, lateral_heading_pass=lateral_pass,
        valid=bool(valid), passed=bool(passed), speed_valid=bool(valid),
        speed_pass=bool(passed), invalid_reasons=reasons,
        identity=dict(R2_IDENTITY))


def cpu_speed_result(raw, kind, c, N) -> SpeedResult:
    """``cpu_speed_oracle`` projected onto the cross-package :class:`SpeedResult`."""
    data = cpu_speed_oracle(raw, kind, c, N)
    mean = data['mean_vx']
    return SpeedResult(
        samples=int(data['samples']),
        mean_vx=None if mean is None else float(mean),
        mean_relative_error=(None if data['mean_relative_error'] is None
                             else float(data['mean_relative_error'])),
        smooth_mae_relative=data['smooth_mae_relative'],
        longest_deviation=data['longest_deviation'],
        valid=bool(data['valid']), passed=bool(data['passed']),
        window_kind=data['window_kind'], coverage_complete=bool(data['coverage_complete']))


def recompute_npz(path, *, kind, c, N, case_id=None):
    """Offline recomputation of one case slice of a retained raw NPZ archive.

    The archive must contain ``offsets`` plus the four (or six) one-dimensional
    evidence arrays written by :mod:`v05_r2.evaluation`.  ``route_error`` is
    retained in the archive as a read-only ``[steps,3]`` array (SDS 4.8) and is
    ignored here: speed metrics never consume route state.
    """
    with np.load(path, allow_pickle=False) as storage:
        files = set(storage.files)
        required = {'body_vx', 'executed_c', 'local_x', 'step'}
        if not required <= files:
            raise ValueError('raw NPZ is missing required evidence arrays')
        raw = {key: storage[key] for key in required}
        for key in ('local_y', 'yaw'):
            if key in files:
                raw[key] = storage[key]
        if 'offsets' in files:
            offsets = storage['offsets']
            row = 0 if case_id is None else int(case_id)
            if not 0 <= row < len(offsets) - 1:
                raise ValueError('case index outside the retained archive')
            lo, hi = int(offsets[row]), int(offsets[row + 1])
            raw = {key: value[lo:hi] for key, value in raw.items()}
    result = cpu_speed_oracle(raw, kind, c, N)
    result['case_id'] = None if case_id is None else int(case_id)
    return result


def window_kind_for_stage(stage_id):
    """The window kind of a *stage's current tile* (supervisor metadata only)."""
    if stage_id not in _spec.STAGE_BY_ID:
        raise ValueError(f'unknown v0.5-r2 stage: {stage_id!r}')
    return tile_window_kind(_spec.STAGE_BY_ID[stage_id].tile_id)
