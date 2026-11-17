"""v0.5-r2 environment: ``route_error`` chain, flat spawn coverage and pre-reset evidence.

SDS 0.5-r2 revision 1.0 sections 4, 5 and 8.1.  The physical control step, contact
extraction, safe-arrival state machine and timeout arithmetic follow
``v02_formal/env.py::FormalEnv`` (the frozen benchmark).  Everything that carries a
0.5-r2 identity is implemented *here*, not inherited:

* the ten-stage table and fixed slot layout come from :mod:`v05_r2.spec` and are
  bound to real atlas tiles; every slot keeps its own N/h/c (a ``previous`` slot is
  the immediate predecessor with **its own** configuration, never the supervisor's
  current height);
* ``route_error`` is a per-control-step, per-environment function of the *same*
  frame and the *same* tile that the reward/gate use (SDS 4.2)::

      e_world     = wrap_to_pi(yaw_world - yaw_axis)          # yaw_axis == 0 here
      route_error = [clip(local_y / 0.20, -4.0, 4.0),
                     sin(e_world), cos(e_world)]              # float32 [B,3]

  It is **not** normalised, **not** clipped to [-1,1], **not** noised and never
  stored in a normalizer.  A missing/invalid tile reference is a hard error; a flat
  slot (N=0) still has its own tile-0 frame origin and therefore takes the very
  same formula (no special branch, no zero fill);
* every actual ``N == 0`` flat slot (all of F0, plus the flat slots of the P1..S5
  mixed layouts) samples its training reset pose from the r2 flat distribution
  ``x ~ U[-1.1,-0.9]``, ``y ~ U[-0.05,0.05]``, ``yaw ~ U[-5deg,5deg]`` in the fixed
  ``x -> y -> yaw -> command`` generator order.  ``initial_cases`` (evaluation and
  playback) takes precedence and consumes no reset RNG.  ``N>0`` slots keep the r1
  fixed ``[-1,0,0]`` pose and the r1 S5 400-update ramp;
* one ``SwingClearanceState``, one ``FirstFootfallState`` and one ``StallState``
  advance exactly once per control sample, *after* the failure/climb/timeout
  decision and *before* the reward and the automatic reset;
* the reward is composed locally from the unchanged 0.5-r2 weights (identical to
  r1).  ``route_error`` never appears in any reward term (SDS 3.9);
* feet/support/stall/swing/gate/first-step diagnostics, the pre-reset
  ``terminal_observation`` (including its pre-reset ``route_error``) and the
  terminal ``episodes`` records are snapshotted before the automatic reset.

Observations: ``proprio 45``, ``history (5,45)``, ``heightmap (3,48,32)``,
``privileged 35`` are unchanged; ``route_error (3,)`` is a fifth, separate key.  12
actions with a single 0.25 scale and the same
``(obs, reward, terminated, truncated, info)`` / ``(obs, info)`` shapes as v0.5-r1.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

# The frozen v0.2 environment config imports the A2 registry; importing it first
# reproduces the normal v0.2/v0.3/v0.4/v0.5 load order and breaks the legacy
# config <-> v02.env_cfg cycle when src.tasks is the bootstrap namespace.
import src.tasks.velocity.config.a2  # noqa: F401
from ..v02.terrain import atlas_metadata
from ..v02_formal.env import FormalEnv as FrozenEnv, NumericalFailure, euler_heading, timeout_steps
from . import spec as _spec
from .rewards import (FootfallState, StallState, SwingClearanceState, SwingSample,
                      footfall_reward, nonfoot_reward, orientation_reward, speed_reward)
from .speed import (ROUTE_CLIP, ROUTE_COMPONENTS, ROUTE_LATERAL_SCALE, ROUTE_YAW_AXIS,
                    SlotSpeedRouter, route_state_contract as _route_state_contract)

# --------------------------------------------------------------- spec binding
_REQUIRED_WEIGHTS = ('track_angular_velocity', 'pose', 'orientation', 'body_ang_vel',
                     'angular_momentum', 'joint_acc', 'joint_limits', 'action_rate',
                     'slip', 'soft_landing', 'stand_still', 'center', 'heading',
                     'edge', 'nonfoot')


def _bind_weights():
    for name in ('REWARD_WEIGHTS', 'CONTINUOUS_WEIGHTS'):
        table = getattr(_spec, name, None)
        if isinstance(table, dict):
            missing = [key for key in _REQUIRED_WEIGHTS if key not in table]
            if missing:
                raise RuntimeError(f'v0.5-r2 spec {name} is missing reward weights: {missing}')
            return dict(table), name
    raise RuntimeError('v0.5-r2 spec exports neither REWARD_WEIGHTS nor CONTINUOUS_WEIGHTS')


_FLAT_SPAWN_FROM_CONTRACT = []


def _user_flat_spawn():
    """Bind the r2 flat training spawn distribution (SDS 5.3) from the spec.

    Preferred: a module-level ``spec.SPAWN_FLAT`` mapping (or ``spec.FLAT_SPAWN_*``
    range scalars).  Fallback: the frozen ``spec.contract()['spawn']['flat']``
    table, read once, which is the same single source.  There is no local default:
    a missing distribution is a hard error, because B1/B2 are contract values.
    """
    table = getattr(_spec, 'SPAWN_FLAT', None)
    table = dict(table) if isinstance(table, dict) else {}
    if not table:
        frozen = getattr(_spec, 'contract', None)
        if frozen is not None:
            try:
                spawn = frozen()['spawn']['flat']
            except Exception:  # the contract table is optional here; fall through
                spawn = None
            if isinstance(spawn, dict):
                table = dict(spawn)
                _FLAT_SPAWN_FROM_CONTRACT.append(True)
    wanted = (('x', ('x', 'x_range'), ('FLAT_SPAWN_X', 'FLAT_SPAWN_X_RANGE', 'R2_FLAT_SPAWN_X')),
              ('y', ('y', 'y_range'), ('FLAT_SPAWN_Y', 'FLAT_SPAWN_Y_RANGE', 'R2_FLAT_SPAWN_Y')),
              ('yaw', ('yaw', 'yaw_range', 'yaw_radians'),
               ('FLAT_SPAWN_YAW', 'FLAT_SPAWN_YAW_RANGE', 'R2_FLAT_SPAWN_YAW')))
    ranges, missing = {}, []
    for key, table_keys, aliases in wanted:
        value = next((table[name] for name in table_keys if name in table), None)
        if value is None:
            for alias in aliases:
                if hasattr(_spec, alias):
                    value = getattr(_spec, alias)
                    break
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, dict):
            value = value.get('range') or value.get('radians') or value.get('values')
        low, high = float(value[0]), float(value[1])
        ranges[key] = (low, high)
    if missing:
        raise RuntimeError(
            'v0.5-r2 spec exports no flat training spawn range for '
            f'{missing}; the r2 flat distribution may not be re-invented locally '
            '(SDS 5.3)')
    for key, (low, high) in ranges.items():
        if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
            raise RuntimeError(f'v0.5-r2 flat spawn range {key} is not a real interval')
    return ranges


def _check_translation_only_atlas(metadata):
    """The ``yaw_axis = 0`` premise (SDS 4.2 / 13.5) as an explicit assertion.

    Every atlas tile is a pure translation along its own frame origin.  A tile
    metadata type that grows a rotation/yaw field would silently invalidate the
    route axis, so it is rejected here instead of being approximated.
    """
    forbidden = ('yaw', 'yaw_axis', 'rotation', 'rotation_deg', 'heading', 'heading_axis')
    for entry in metadata:
        if not is_dataclass(entry):
            continue
        for field in fields(entry):
            if field.name in forbidden:
                raise RuntimeError(
                    'the atlas metadata carries a rotation field '
                    f'{field.name!r}; yaw_axis=0 is only valid for a translating '
                    'atlas (SDS 4.2/13.5)')
    origin = np.asarray([entry.frame_origin_world for entry in metadata], dtype=np.float64)
    if origin.shape != (len(metadata), 3) or not np.isfinite(origin).all():
        raise RuntimeError('atlas frame origins are not a finite [tiles,3] table')


REWARD_WEIGHTS, REWARD_WEIGHTS_SOURCE = _bind_weights()
FLAT_SPAWN = _user_flat_spawn()
_atlas_premise_metadata = atlas_metadata()
_check_translation_only_atlas(_atlas_premise_metadata)

FLAT_SPAWN_SOURCE = ('spec.contract()["spawn"]["flat"]'
                     if _FLAT_SPAWN_FROM_CONTRACT else 'spec.SPAWN_FLAT')

SDS_REVISION = _spec.SDS_REVISION
CODE_VERSION = _spec.CODE_VERSION
RECIPE_VERSION = _spec.RECIPE_VERSION
PROTOCOL_ID = _spec.PROTOCOL_ID
FLAT_PROTOCOL_ID = _spec.FLAT_PROTOCOL_ID
TASK = _spec.TASK
STAGES = _spec.STAGES
STAGE_BY_ID = _spec.STAGE_BY_ID
STAGE_BY_TILE = _spec.STAGE_BY_TILE
SLOT_CLASSES = _spec.SLOT_CLASSES
CONTROL_DT = _spec.CONTROL_DT
FLAT_EPISODE_STEPS = _spec.F0_EPISODE_STEPS
MAX_SWING_STEPS = _spec.MAX_SWING_STEPS
FOOT_RADIUS = _spec.FOOT_RADIUS
COMMAND_MIN = _spec.COMMAND_MIN
FOOTFALL_AMOUNT = _spec.FOOTFALL_AMOUNT
SUCCESS_AMOUNT = _spec.SUCCESS_AMOUNT
FAILURE_AMOUNT = _spec.FAILURE_AMOUNT
SAFE_INSET = _spec.SAFE_INSET
FLAT_TILE_ID = _spec.FLAT_TILE_ID
COMMAND_LOW, COMMAND_HIGH = .2, .6

#: SDS 5.3: the fixed generator consumption order recorded in every episode.
#: Bound from the single source and asserted, so a spec edit cannot drift from the
#: order this file actually consumes.
INITIAL_SAMPLE_ORDER = getattr(_spec, 'SPAWN_FLAT_SAMPLE_ORDER', 'x,y,yaw,command')
if INITIAL_SAMPLE_ORDER != 'x,y,yaw,command':
    raise RuntimeError(
        f'v0.5-r2 flat sample order drifted to {INITIAL_SAMPLE_ORDER!r}; the r2 reset '
        'consumes x, then y, then yaw, then the command (SDS 5.3)')
S5_RAMP_UPDATES = int(getattr(_spec, 'S5_RAMP_UPDATES', 400))
if S5_RAMP_UPDATES != 400:
    raise RuntimeError('the S5 perturbation ramp must stay at the r1 value of 400 updates '
                       '(SDS 5.5)')
_s5_tiles = [entry for entry in _atlas_premise_metadata if entry.stage == 'S5']
if _s5_tiles and int(_s5_tiles[0].perturb_ramp_updates) != S5_RAMP_UPDATES:
    raise RuntimeError('the atlas S5 perturb_ramp_updates no longer matches '
                       f'spec.S5_RAMP_UPDATES ({S5_RAMP_UPDATES})')
#: SDS 5.3 ``initial_pose_source`` labels, per slot's real tile and path.
SOURCE_FLAT = 'r2_flat_uniform'
SOURCE_FIXED = 'r1_fixed'
SOURCE_S5_RAMP = 'r1_s5_ramp'
SOURCE_INITIAL_CASES = 'initial_cases'

#: Reason codes, unchanged from the frozen v0.2 lifecycle (failure codes win).
REASON_NAMES = ('running', 'timeout', 'success', 'bounds', 'below_support', 'side_fall',
                'orientation', 'nonfoot_persistent', 'body_contact', 'numerical_failure')

TERMINAL_KEYS = ('failure', 'climb', 'timeout', 'first_step_acquired', 'stall_on',
                 'terminal_observation', 'reset_mask', 'episodes', 'diagnostics',
                 'reward_terms', 'step_reward_tensors', 'command_executed', 'route_error')

#: ``info`` keys that expose route state; the two formal positions are
#: ``terminal_observation['route_error']`` (pre-reset) and ``obs['route_error']``
#: (post-reset).  The aliases are diagnostics only (SDS 4.5).
ROUTE_INFO_KEYS = ('route_error', 'route_error_terminal', 'route_error_next')

#: SDS 4.8 per-update aggregate names produced from :meth:`V05R2Env.route_metrics`.
ROUTE_AGGREGATE_KEYS = ('route/abs_y_mean', 'route/abs_y_max',
                        'route/normalized_abs_y_max', 'route/saturation_rate',
                        'route/heading_deg_mean', 'route/heading_deg_max',
                        'route/nonfinite_count')


#: SDS 2.1: the observation dictionary has exactly these five keys and no aliases
#: (``route``/``ideal_route_state`` are deployment labels, never simulation keys).
OBSERVATION_KEYS = tuple(_spec.OBSERVATION_KEYS)
OBSERVATION_SHAPES = {'proprio': (45,), 'history': (5, 45), 'heightmap': (3, 48, 32),
                      'privileged': (35,), 'route_error': (3,)}


def validate_observations(observations, *, consumer='v05_r2.env'):
    """Strictly validate one five-key observation dictionary (SDS 2.1/4.7).

    Every consumer (evaluation, playback, runner, PPO) must be able to reject a
    dictionary before any tensor operation: a missing ``route_error``, a wrong
    shape, a non-``float32``/non-finite value, a mixed batch or a foreign device is
    a hard error.  Nothing is zero-filled and no key is renamed.  Returns the batch
    size so callers can cross-check their own expectations.
    """
    if not isinstance(observations, dict):
        raise TypeError(f'{consumer} observation must be a dict, got {type(observations).__name__}')
    missing = [key for key in OBSERVATION_KEYS if key not in observations]
    if missing:
        raise ValueError(f'{consumer} observation is missing {missing}; the r2 contract '
                         'requires all five keys and never zero-fills route_error')
    extra = sorted(set(observations) - set(OBSERVATION_KEYS))
    if extra:
        raise ValueError(f'{consumer} observation has unexpected keys {extra}; the r2 key '
                         'set is fixed (SDS 2.1)')
    batch = None
    device = None
    for key in OBSERVATION_KEYS:
        value = observations[key]
        if not torch.is_tensor(value):
            raise TypeError(f'{consumer} observation {key} must be a tensor')
        expected = OBSERVATION_SHAPES[key]
        if value.dtype != torch.float32:
            raise ValueError(f'{consumer} observation {key} must be float32, '
                             f'got {value.dtype}')
        if value.ndim != len(expected) + 1 or tuple(value.shape[1:]) != expected:
            raise ValueError(f'{consumer} observation {key} must be [B,{expected}], '
                             f'got {tuple(value.shape)}')
        if batch is None:
            batch = int(value.shape[0])
        elif int(value.shape[0]) != batch:
            raise ValueError(f'{consumer} observation {key} has batch {value.shape[0]}, '
                             f'expected {batch}')
        if device is None:
            device = value.device
        elif value.device != device:
            raise ValueError(f'{consumer} observation {key} lives on {value.device}, '
                             f'expected {device}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{consumer} observation {key} contains NaN/Inf')
    return batch


def wrap_to_pi(angle):
    """Wrap radians to ``(-pi, pi]`` via ``atan2(sin, cos)`` (SDS 4.2)."""
    if not torch.is_tensor(angle):
        angle = torch.as_tensor(angle)
    return torch.atan2(torch.sin(angle), torch.cos(angle))


#: SDS 4.2 saturation point in metres: ``|local_y| >= 0.80 m`` (exactly 0.80 m
#: saturates because the comparison is ``>=``, and the value never grows past 4).
ROUTE_LATERAL_SATURATION_M = ROUTE_CLIP * ROUTE_LATERAL_SCALE


@dataclass(frozen=True)
class RouteComponents:
    """The one and only decomposition of the SDS 4.2 route state.

    Every consumer in this module (the observation vector, the read-only
    ``route/*`` diagnostics and the snapshot) reads these fields instead of
    re-deriving a ratio, an absolute value or a saturation predicate, so the three
    can never disagree:

    * ``e_world`` -- ``wrap_to_pi(yaw_world - yaw_axis)`` in radians;
    * ``abs_y`` -- ``|local_y|`` in metres;
    * ``normalized_abs_y_raw`` -- ``|local_y| / 0.20``, **not** clipped (this is the
      quantity the saturation test uses);
    * ``saturated`` -- ``normalized_abs_y_raw >= 4``, i.e. ``|local_y| >= 0.80 m``;
    * ``route_error`` -- ``[clip(local_y/0.20,-4,4), sin(e_world), cos(e_world)]``
      ``float32[B,3]``, whose first component is ``|normalized_abs_y|`` clipped.
    """

    e_world: torch.Tensor
    abs_y: torch.Tensor
    normalized_abs_y_raw: torch.Tensor
    normalized_abs_y: torch.Tensor
    saturated: torch.Tensor
    route_error: torch.Tensor


def route_components(local_y, yaw_world):
    """The single implementation of the SDS 4.2 route decomposition.

    ``local_y`` is the world/tile lateral offset in metres (the same expression as
    the ``center`` reward and the F0 gate); ``yaw_world`` is the world yaw in
    radians.  Both must be tensors of the same shape.
    """
    if not torch.is_tensor(local_y) or not torch.is_tensor(yaw_world):
        raise TypeError('route_components needs tensors')
    if local_y.shape != yaw_world.shape:
        raise ValueError(f'route inputs must share a shape, got {tuple(local_y.shape)} '
                         f'and {tuple(yaw_world.shape)}')
    lateral = local_y.to(torch.float32)
    e_world = wrap_to_pi(yaw_world.to(torch.float32) - ROUTE_YAW_AXIS)
    ratio = lateral / ROUTE_LATERAL_SCALE
    abs_y = lateral.abs()
    normalized_abs_y_raw = ratio.abs()
    saturated = normalized_abs_y_raw >= ROUTE_CLIP
    first = ratio.clamp(-ROUTE_CLIP, ROUTE_CLIP)
    route = torch.stack((first, torch.sin(e_world), torch.cos(e_world)), -1).to(torch.float32)
    return RouteComponents(e_world=e_world, abs_y=abs_y,
                           normalized_abs_y_raw=normalized_abs_y_raw,
                           normalized_abs_y=first.abs(), saturated=saturated,
                           route_error=route)


def route_error_vector(local_y, yaw_world):
    """The exact SDS 4.2 route vector for one frame.

    Returns ``(route_error[B,3] float32, e_world[B] float32, saturation[B] bool)``.
    Thin view of :func:`route_components`; the formula lives in exactly one place.
    """
    components = route_components(local_y, yaw_world)
    return components.route_error, components.e_world, components.saturated


def route_state_contract():
    """JSON-safe route contract plus the r2 flat spawn coverage (identity only).

    The route formula fields come from the single binding in :mod:`v05_r2.speed`;
    this wrapper only adds the SDS 5.3 flat spawn facts that live with the
    environment.
    """
    return dict(_route_state_contract(), flat_spawn={key: list(value)
                                                    for key, value in FLAT_SPAWN.items()},
                flat_spawn_source=FLAT_SPAWN_SOURCE,
                initial_sample_order=INITIAL_SAMPLE_ORDER,
                initial_pose_sources=[SOURCE_FIXED, SOURCE_FLAT, SOURCE_S5_RAMP,
                                      SOURCE_INITIAL_CASES])


@dataclass(frozen=True)
class TaskState:
    """Everything one control sample decided, before any reset."""

    local: torch.Tensor
    roll: torch.Tensor
    pitch: torch.Tensor
    yaw: torch.Tensor
    region: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    truncated: torch.Tensor
    reason: torch.Tensor
    n: torch.Tensor
    h: torch.Tensor
    safe: torch.Tensor
    #: ``route_error`` and its derived fields, computed once from exactly this
    #: frame's ``local``/``yaw`` and each environment's own tile by
    #: :func:`route_components` (SDS 4.2/9.3-23).
    route_error: torch.Tensor
    e_world: torch.Tensor
    abs_y: torch.Tensor
    saturated: torch.Tensor


class V05R2Env(FrozenEnv):
    """v0.5-r2 climbing environment over the frozen physical lifecycle."""

    def __init__(self, num_envs=512, device='cuda:0', seed=42, stage='F0',
                 stage_iteration=0, evaluation=False, fixed_vx=None,
                 initial_cases=None, render_mode=None, parking_diagnostic=False):
        if isinstance(fixed_vx, (int, float, np.floating)) and not isinstance(fixed_vx, bool):
            value = float(fixed_vx)
            if not math.isfinite(value) or not (COMMAND_LOW <= value <= COMMAND_HIGH):
                raise ValueError('v0.5-r2 fixed_vx must be within [0.2,0.6]')
        elif fixed_vx is not None:
            raise TypeError('fixed_vx must be a float in [0.2,0.6] or None')
        if stage not in STAGE_BY_ID:
            raise ValueError(f'unknown v0.5-r2 stage: {stage!r}')
        # Bounded task state must exist before the inherited constructor's
        # set_stage -> reset path reaches _reset_ids.
        self._allocate_task_state(num_envs, device)
        super().__init__(num_envs=num_envs, device=device, seed=seed, stage=stage,
                         stage_iteration=stage_iteration, evaluation=evaluation,
                         fixed_vx=fixed_vx, initial_cases=initial_cases,
                         render_mode=render_mode, parking_diagnostic=parking_diagnostic)
        # Identity labels for terminal records; the physical adapter stays frozen.
        self.recipe = RECIPE_VERSION
        self.reward_version = RECIPE_VERSION
        self.protocol = FLAT_PROTOCOL_ID if STAGE_BY_ID[self.stage].N == 0 else PROTOCOL_ID

    # ------------------------------------------------------------- allocation
    def _allocate_task_state(self, num_envs, device):
        metadata = atlas_metadata()
        self._metadata = metadata
        self.atlas_N = torch.tensor([m.N for m in metadata], dtype=torch.int64, device=device)
        self.atlas_H = torch.tensor([m.h for m in metadata], dtype=torch.float32, device=device)
        b = int(num_envs)
        self.swing = SwingClearanceState(b, max_steps=MAX_SWING_STEPS, device=device)
        self.footfall_state = FootfallState(b, max_steps=MAX_SWING_STEPS, device=device)
        self.stall = StallState(b, device)
        self.speed_router = SlotSpeedRouter(b, np.zeros(b, dtype=np.int64), device=device)
        self.tile_N = torch.zeros(b, dtype=torch.int64, device=device)
        self.previous_stage = None
        self._swing_result = None
        self._stall_result = None
        self._footfall_result = None
        self._footfall_amount = torch.zeros(b, device=device)
        self._last_reward_terms = {}
        self._last_step_diagnostics = {}
        self.step_reward_tensors = {}
        self.reward_terms = {}
        self.first_step_acquired = torch.zeros(b, dtype=torch.bool, device=device)
        self.episode_initial_pose = torch.zeros(b, 3, device=device)
        self.episode_command = torch.zeros(b, device=device)
        self.episode_initial_source = [SOURCE_FIXED] * b
        self.episode_rng_sha = [''] * b
        self.sum_vx = torch.zeros(b, device=device)
        self.sum_speed = torch.zeros(b, device=device)
        self.max_lateral = torch.zeros(b, device=device)
        self.max_abs_yaw = torch.zeros(b, device=device)
        self.max_body_x = torch.zeros(b, device=device)
        self.peak_torque = torch.zeros(b, device=device)
        self.footfall_total = torch.zeros(b, device=device)
        self.swing_total = torch.zeros(b, device=device)
        self.stall_samples = torch.zeros(b, dtype=torch.long, device=device)
        self.active_samples = torch.zeros(b, dtype=torch.long, device=device)
        self.stall_active_samples = torch.zeros(b, dtype=torch.long, device=device)
        self.control_body_hit = torch.zeros(b, dtype=torch.bool, device=device)
        self.control_long_hit = torch.zeros(b, dtype=torch.bool, device=device)
        self.climb_event = torch.zeros(b, dtype=torch.bool, device=device)
        self.current_platform = torch.zeros(b, 4, dtype=torch.bool, device=device)
        self.last_termination = None
        self.sample_command = torch.zeros(b, 3, device=device)
        # ---- r2 route state (diagnostics and audit only; never an input to reward)
        self.episode_route_initial = torch.zeros(b, 3, device=device)
        self.episode_route_final = torch.zeros(b, 3, device=device)
        self.route_max_abs_y = torch.zeros(b, device=device)
        self.route_max_abs_heading = torch.zeros(b, device=device)
        self.sample_route = torch.zeros(b, 3, device=device)
        self.sample_e_world = torch.zeros(b, device=device)

    # ---------------------------------------------------------------- staging
    @property
    def stage_slots(self):
        return list(_spec.slot_layout(self.stage, self.num_envs))

    @property
    def tile_n(self):
        return self.atlas_N[self.tile_ids]

    def set_stage(self, stage, stage_iteration=0, previous_stage=None):
        if stage not in STAGE_BY_ID:
            raise ValueError(f'unknown v0.5-r2 stage: {stage!r}')
        expected = _spec.previous_stage_id(stage)
        # A caller may restate the immediate predecessor, never another version's.
        if previous_stage is not None and previous_stage != expected:
            raise ValueError(f'{stage} previous stage must be {expected!r}')
        changed = stage != self.stage
        self.stage = stage
        self.stage_iteration = int(stage_iteration)
        self.previous_stage = expected
        if changed:
            if self.evaluation:
                tiles = [STAGE_BY_ID[stage].tile_id] * self.num_envs
                classes = ['current'] * self.num_envs
            else:
                tiles = _spec.slot_tiles(stage, self.num_envs)
                classes = list(_spec.slot_layout(stage, self.num_envs))
            self.tile_ids.copy_(torch.as_tensor(tiles, dtype=torch.int64, device=self.device))
            self.slot_class_names = classes
            self.slot_classes.copy_(torch.as_tensor(
                [SLOT_CLASSES.index(name) for name in classes],
                dtype=torch.int64, device=self.device))
            self.tile_N.copy_(self.atlas_N[self.tile_ids])
            self.speed_router.set_tiles(self.tile_ids)
            self.protocol = (FLAT_PROTOCOL_ID if STAGE_BY_ID[stage].N == 0 else PROTOCOL_ID)
            self.reset()

    def _parameters(self):
        # Atlas indexing by real tile (never a list position of an old stage table).
        return self.atlas_N[self.tile_ids], self.atlas_H[self.tile_ids]

    def _slot_name(self, env_id):
        names = getattr(self, 'slot_class_names', None)
        if names is None or not 0 <= int(env_id) < len(names):
            return None
        return names[int(env_id)]

    # ------------------------------------------------------------- route state
    def route_error(self):
        """``float32[B,3]`` route vector of the current, same-frame robot state.

        Uses each environment's **own** atlas tile (``self.tile_ids``; never the
        supervisor stage for the whole batch) and the exact ``local = pos - origin``
        expression of ``_task_state``.  A ``[B,3]`` result that is not finite, a tile
        outside the atlas or a stale/wrong batch size is a hard error: the route is
        never zero-filled and never falls back to flat tile 0 (SDS 4.7).
        """
        data = self.robot.data
        return self._route_error(self.tile_ids, data.root_link_pos_w,
                                 data.root_link_quat_w)

    def _route_frame(self, tile_ids, root_pos_w, root_quat_w):
        """Validated ``(local, yaw_world)`` of the current same frame.

        ``local`` is the exact ``root_pos_w - origins[tile_ids]`` expression of
        ``_task_state``; a stale/wrong batch, a foreign device, a non-finite pose or
        a tile outside the atlas is a hard error (never a flat-0 fallback).
        """
        tiles = tile_ids
        if not torch.is_tensor(tiles) or tiles.dtype != torch.int64:
            raise TypeError('route tile_ids must be an int64 tensor')
        if tiles.shape != (self.num_envs,):
            raise ValueError(f'route tile_ids must have shape [{self.num_envs}], '
                             f'got {tuple(tiles.shape)}')
        device = torch.device(self.device)
        if tiles.device != device:
            raise ValueError(f'route tile_ids live on {tiles.device}, expected {device}')
        if not torch.is_tensor(root_pos_w) or root_pos_w.shape != (self.num_envs, 3):
            raise ValueError('route root positions must be a [num_envs,3] tensor')
        if root_quat_w.shape != (self.num_envs, 4):
            raise ValueError('route root quaternions must be a [num_envs,4] tensor')
        if root_pos_w.device != device or root_quat_w.device != device:
            raise ValueError('route pose tensors must live on the environment device')
        if not (torch.isfinite(root_pos_w).all() and torch.isfinite(root_quat_w).all()):
            raise NumericalFailure(
                'Nonfinite pose before route_error',
                {'root_pos_w': root_pos_w.detach().cpu(),
                 'root_quat_w': root_quat_w.detach().cpu(),
                 'tile_ids': tiles.detach().cpu(),
                 'stage': self.stage})
        count = len(self._metadata)
        out_of_range = (tiles < 0) | (tiles >= count)
        if bool(out_of_range.any()):
            i = int(out_of_range.nonzero(as_tuple=True)[0][0])
            raise RuntimeError(
                'route_error tile reference is invalid: '
                f'stage={self.stage} slot_class={self._slot_name(i)} env_id={i} '
                f'tile_id={int(tiles[i])} atlas_tiles={count}')
        if not torch.is_tensor(self.origins) or self.origins.shape != (count, 3):
            raise RuntimeError('route_error has no usable atlas frame origins '
                               f'(stage={self.stage}, slots={getattr(self, "slot_class_names", None)})')
        local = root_pos_w - self.origins[tiles]
        return local, euler_heading(root_quat_w)[2]

    def _route_components(self, tile_ids, root_pos_w, root_quat_w):
        """Validated frame plus the single :func:`route_components` decomposition."""
        local, yaw_world = self._route_frame(tile_ids, root_pos_w, root_quat_w)
        components = route_components(local[:, 1], yaw_world)
        if not torch.isfinite(components.route_error).all():
            i = int((~torch.isfinite(components.route_error).all(-1))
                    .nonzero(as_tuple=True)[0][0])
            raise NumericalFailure(
                'Nonfinite route_error',
                {'stage': self.stage, 'slot_class': self._slot_name(i), 'env_id': i,
                 'tile_id': int(tile_ids[i]), 'local': local.detach().cpu(),
                 'yaw': yaw_world.detach().cpu(),
                 'route_error': components.route_error.detach().cpu()})
        return local, yaw_world, components

    def _route_error(self, tile_ids, root_pos_w, root_quat_w):
        return self._route_components(tile_ids, root_pos_w, root_quat_w)[2].route_error

    def route_metrics(self):
        """Read-only per-sample ``route/*`` quantities (SDS 4.8 names).

        Every field comes from the one :func:`route_components` decomposition of
        the same frame, so ``abs_y``/``normalized_abs_y``/``saturation``/
        ``route_error`` can never disagree (SDS 4.2: 0.80 m saturates with ``>=``
        and the clamped value never grows past 4).  These tensors only feed
        telemetry/episodes aggregation; they are never an observation key, never
        enter a normalizer, a reward term or an advantage.
        """
        d = self.robot.data
        local, yaw_world, components = self._route_components(
            self.tile_ids, d.root_link_pos_w, d.root_link_quat_w)
        e_world = components.e_world
        return dict(
            local_y=local[:, 1].detach().clone(),
            abs_y=components.abs_y.detach().clone(),
            # SDS 4.8: ``route/normalized_abs_y`` is the clipped route component.
            normalized_abs_y=components.normalized_abs_y.detach().clone(),
            normalized_abs_y_raw=components.normalized_abs_y_raw.detach().clone(),
            saturation=components.saturated.detach().clone(),
            e_world=e_world.detach().clone(),
            heading_deg=(e_world.abs() * (180. / math.pi)).detach().clone(),
            yaw_world=yaw_world.detach().clone(),
            route_error=components.route_error.detach().clone(),
            tile_ids=self.tile_ids.detach().clone(),
            nonfinite_count=int((~torch.isfinite(components.route_error)).sum()),
            yaw_axis=ROUTE_YAW_AXIS,
            lateral_scale=ROUTE_LATERAL_SCALE,
            saturation_m=ROUTE_LATERAL_SATURATION_M)

    # ------------------------------------------------------------------ reset
    def _rng_state_sha256(self):
        """Stable SHA256 of the environment generator state (SDS 5.3 audit)."""
        state = self.rng.get_state()
        payload = (state.detach().cpu().numpy().tobytes()
                   if torch.is_tensor(state) else bytes(state))
        return hashlib.sha256(payload).hexdigest()

    def _sample_commands(self, count):
        """Command for one reset subset: fixed value, otherwise one uniform draw."""
        if self.fixed_vx is not None:
            return torch.full((count,), float(self.fixed_vx), device=self.device)
        return COMMAND_LOW + (COMMAND_HIGH - COMMAND_LOW) * torch.rand(
            count, device=self.device, generator=self.rng)

    def _reset_ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device=self.device) if not isinstance(ids, torch.Tensor) else ids
        if ids.dtype == torch.bool:
            ids = torch.nonzero(ids, as_tuple=True)[0]
        ids = ids.to(torch.int64).flatten()
        if ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('reset ids are outside the environment batch')
        # SDS 5.3: a subset reset consumes RNG in ascending id order only.
        ids, _ = torch.sort(ids)
        self.sim.reset(ids)
        self.scene.reset(ids)
        self.action_manager.reset(ids)
        robot = self.robot
        state = robot.data.default_root_state[ids].clone()
        count = len(ids)
        local = torch.zeros(count, 3, device=self.device)
        local[:, 0] = self.spawn_x[self.tile_ids[ids]]
        yaw = torch.zeros(count, device=self.device)
        sources = [SOURCE_FIXED] * count
        # SDS 5.3: the digest of the generator state *before* any reset sampling.
        rng_sha = self._rng_state_sha256()
        flat = self.atlas_N[self.tile_ids[ids]] == 0
        vx = torch.zeros(count, device=self.device)
        if self.initial_cases is not None:
            # SDS 5.3: explicit cases win and no initial-state sampling happens.
            local[:, :2] = self.initial_cases[ids, :2]
            yaw = self.initial_cases[ids, 2].clone()
            sources = [SOURCE_INITIAL_CASES] * count
            vx = self._sample_commands(count)
        else:
            flat_rows = torch.nonzero(flat, as_tuple=True)[0]
            stair_rows = torch.nonzero(~flat, as_tuple=True)[0]
            if flat_rows.numel():
                # SDS 5.3 fixed order: one [n_flat,3] draw, then one command draw.
                draw = torch.rand((flat_rows.numel(), 3), device=self.device,
                                  generator=self.rng)
                x_low, x_high = FLAT_SPAWN['x']
                y_low, y_high = FLAT_SPAWN['y']
                yaw_low, yaw_high = FLAT_SPAWN['yaw']
                local[flat_rows, 0] = x_low + (x_high - x_low) * draw[:, 0]
                local[flat_rows, 1] = y_low + (y_high - y_low) * draw[:, 1]
                yaw[flat_rows] = yaw_low + (yaw_high - yaw_low) * draw[:, 2]
                vx[flat_rows] = self._sample_commands(flat_rows.numel())
                for row in flat_rows.tolist():
                    sources[row] = SOURCE_FLAT
            if stair_rows.numel():
                # r1 behaviour for N>0 slots: the S5 ramp is the only non-zero
                # amplitude, everything else keeps the fixed [-1,0,0] pose.
                amplitude = self.initial_noise[self.tile_ids[ids][stair_rows]]
                if amplitude.any():
                    ramp = (torch.ones(len(stair_rows), device=self.device) if self.evaluation else
                            (self.stage_iteration / self.perturb_ramp[self.tile_ids[ids][stair_rows]]
                             .clamp_min(1)).clamp_max(1.))
                    noise = (torch.rand((len(stair_rows), 3), device=self.device,
                                        generator=self.rng) * 2 - 1) * amplitude * ramp[:, None]
                    local[stair_rows, :2] += noise[:, :2]
                    yaw[stair_rows] = noise[:, 2]
                    for row in stair_rows.tolist():
                        if float(self.perturb_ramp[self.tile_ids[ids][row]]) > 0.:
                            sources[row] = SOURCE_S5_RAMP
                vx[stair_rows] = self._sample_commands(len(stair_rows))
        local[:, 2] = state[:, 2]
        state[:, :3] = self.origins[self.tile_ids[ids]] + local
        state[:, 3:7] = 0
        state[:, 3] = torch.cos(yaw / 2)
        state[:, 6] = torch.sin(yaw / 2)
        state[:, 7:] = 0
        robot.write_root_state_to_sim(state, ids)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids],
                                       torch.zeros_like(robot.data.joint_vel[ids]), env_ids=ids)
        self.command[ids] = 0
        self.command[ids, 0] = vx
        self.start_vx[ids] = vx
        self.episode_length_buf[ids] = 0
        self.stop[ids] = False
        self.stop_steps[ids] = 0
        self.last_contact[ids] = False
        self.contacts.reset(ids)
        self.parking.reset(ids)
        self.footfall_state.reset(ids)
        self.climb.reset(ids)
        self.parking_diagnostics.reset(ids)
        self.swing.reset(ids)
        self.stall.reset(ids, local[:, 0].detach().clone())
        self.speed_router.reset(ids)
        for tensor in (self.first_step_acquired, self.control_body_hit, self.control_long_hit,
                       self.climb_event):
            tensor[ids] = False
        self.current_platform[ids] = False
        for tensor in (self.sum_vx, self.sum_speed, self.max_lateral, self.max_abs_yaw,
                       self.max_body_x, self.peak_torque, self.footfall_total, self.swing_total,
                       self.stall_samples, self.active_samples, self.stall_active_samples,
                       self.route_max_abs_y, self.route_max_abs_heading):
            tensor[ids] = 0
        for value in self.episode_reward_sums.values():
            value[ids] = 0
        self.episode_initial_pose[ids] = torch.stack((local[:, 0], local[:, 1], yaw), -1)
        self.episode_command[ids] = vx
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.sense()
        obs = self._proprio()
        self.history_cache.reset(obs, ids)
        self.map_cache.reset(robot.data.root_link_pos_w,
                             euler_heading(robot.data.root_link_quat_w)[2], self.tile_ids, ids)
        # SDS 4.5: the reset route is computed *after* the initial state write,
        # sim.forward() and sim.sense() -- never from a default pose.  One
        # :func:`route_components` decomposition supplies the vector, |local_y| and
        # |e_world|, so the audit maxima cannot drift from the route formula.
        d = robot.data
        _, _, components = self._route_components(self.tile_ids, d.root_link_pos_w,
                                                  d.root_link_quat_w)
        route = components.route_error
        self.episode_route_initial[ids] = route[ids]
        self.episode_route_final[ids] = route[ids]
        self.route_max_abs_y[ids] = components.abs_y[ids]
        self.route_max_abs_heading[ids] = components.e_world[ids].abs()
        self.sample_route[ids] = route[ids]
        self.sample_e_world[ids] = components.e_world[ids]
        for row, env_id in enumerate(ids.tolist()):
            self.episode_initial_source[env_id] = sources[row]
            self.episode_rng_sha[env_id] = rng_sha

    # --------------------------------------------------------------- task step
    def _motion_parking_mask(self):
        """Arrival belongs to moving; only subsequent physical samples park."""
        return self.climb.climbed & (self.episode_length_buf > self.climb.climb_step) & self.parking_diagnostic

    def _task_state(self):
        """Failure/climb/timeout, then exactly one footfall/swing/stall advance."""
        # The command executed during this physical sample, read after the
        # physics and before a parking diagnostic can zero it.
        self.sample_command = self.command.clone()
        d = self.robot.data
        n, h = self._parameters()
        local = d.root_link_pos_w - self.origins[self.tile_ids]
        feet = d.site_pos_w[:, self.foot_ids] - self.origins[self.tile_ids, None]
        roll, pitch, yaw = euler_heading(d.root_link_quat_w)
        stairs = self.tile_ids > 0
        region = stairs & (local[:, 0] >= 0)
        safe = ((feet[:, :, 0] >= n[:, None] * .2 + SAFE_INSET)
                & (feet[:, :, 0] <= n[:, None] * .2 + 2 - SAFE_INSET)
                & (feet[:, :, 1].abs() <= .4 - SAFE_INSET))
        expected = torch.minimum((torch.floor(local[:, 0] / .2) + 1).clamp_min(0), n) * h
        body = self.control_body_hit | (self.contacts.link_force[:, self.contacts.body_link_mask]
                                        .norm(dim=-1) > 0).any(-1)
        long = self.control_long_hit | (self.contacts.nonfoot_duration >= .1 - 1e-6).any(-1)
        fallen = (roll.abs() > math.radians(70)) | (pitch.abs() > math.radians(70))
        side = region & (local[:, 1].abs() > .4)
        below = region & (local[:, 2] < expected - .5)
        xmax = torch.where(stairs, n * .2 + 2, 22.)
        ymax = torch.where(stairs, 2., 4.)
        bounds = (local[:, 0] < -2) | (local[:, 0] > xmax) | (local[:, 1].abs() > ymax)
        numeric = ~torch.isfinite(torch.cat((d.root_link_pos_w, d.root_link_quat_w,
                                             d.joint_pos, d.joint_vel), -1)).all(-1)
        failure = body | long | fallen | side | below | bounds | numeric
        # The landing is the two-metre geom; the same-height final tread is
        # deliberately excluded so the final tread cannot mint platform support.
        is_landing = ((self.geom_step[None, :] == n[:, None])
                      & (self.geom_tile[None, :] == self.tile_ids[:, None])
                      & ((self.bounds[:, 1] - self.bounds[:, 0])[None, :] > 1.99))
        force = torch.where(is_landing[:, None, :], self.contacts.top_forces, -torch.inf)
        upward, surface_ids = force.max(-1)
        candidate = (upward > 5.) & safe & stairs[:, None]
        result = self.climb.update(self.episode_length_buf, stairs, self.tile_ids, surface_ids,
                                   candidate, safe, local[:, 0], n * .2, local[:, 1], failure)
        self.climb_event = result.success
        self.current_platform = result.current_support
        self.stop = self.climb.climbed.clone() if self.parking_diagnostic else torch.zeros_like(failure)
        if self.parking_diagnostic:
            self.command[self.stop] = 0
            parking_success, parking_timeout = self.parking_diagnostics.update(
                self.episode_length_buf, self.climb.climb_step, self.climb.climbed,
                self.current_platform, local[:, 1], d.root_link_lin_vel_b.norm(dim=-1), failure)
            self.stop_steps = self.parking_diagnostics.parking_steps.clone()
            success = parking_success & ~failure
        else:
            success = result.success
            parking_timeout = torch.zeros_like(failure)
            self.stop_steps.zero_()
        base_timeout = self.episode_length_buf >= torch.where(
            stairs, timeout_steps(n, h, self.start_vx), torch.full_like(self.episode_length_buf, FLAT_EPISODE_STEPS))
        timeout = (torch.where(self.climb.climbed, parking_timeout, base_timeout)
                   if self.parking_diagnostic else base_timeout)
        truncated = timeout & ~failure & ~success
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for code, mask in ((1, truncated), (2, success), (3, bounds), (4, below), (5, side),
                           (6, fallen), (7, long), (8, body), (9, numeric)):
            reason[mask] = code
        # SDS 4.2/9.3-23: the route vector is produced from this frame's own
        # ``local``/``yaw`` and this environment's own tile -- the same tensors the
        # reward/gate used, with zero tolerance, through the single
        # :func:`route_components` decomposition.
        components = route_components(local[:, 1], yaw)
        self.sample_route = components.route_error
        self.sample_e_world = components.e_world
        state = TaskState(local=local, roll=roll, pitch=pitch, yaw=yaw, region=region,
                          success=success, failure=failure, truncated=truncated,
                          reason=reason, n=n, h=h, safe=safe,
                          route_error=components.route_error,
                          e_world=components.e_world, abs_y=components.abs_y,
                          saturated=components.saturated)
        # One advance per control sample, after the decision and before reward,
        # in the SDS section 4 order: footfall, swing, stall.  The three state
        # machines share no mutable state, so the order cannot change values.
        self._advance_footfall(state)
        self._advance_swing(state)
        self._advance_stall(state)
        return state

    # --------------------------------------------------------------- events
    def _build_swing_sample(self, state):
        d = self.robot.data
        foot_pos_w = d.site_pos_w[:, self.foot_ids].detach().clone().to(torch.float32)
        root_pos = d.root_link_pos_w.detach()
        root_quat = d.root_link_quat_w.detach()
        relative = foot_pos_w - root_pos[:, None]
        foot_pos_b = quat_apply_inverse(
            root_quat[:, None].expand(-1, 4, -1).reshape(-1, 4),
            relative.reshape(-1, 3)).reshape_as(foot_pos_w)
        n, _ = self._parameters()
        return SwingSample(
            tile_ids=self.tile_ids.detach().clone().to(torch.int64),
            # Flat stays N=0: the swing state must accept it instead of pretending
            # N=1, and must never query a step-1 geom for a flat slot.
            num_steps=n.detach().to(torch.int64),
            foot_pos_w=foot_pos_w,
            foot_pos_b=foot_pos_b.contiguous(),
            contact_found=self.contacts.found.detach().clone().to(torch.bool),
            top_forces=self.contacts.top_forces.detach(),
            geom_tile=self.geom_tile,
            geom_step=self.geom_step,
            geom_top=self.geom_top,
            geom_union_bounds=self.geom_step_bounds,
            executed_c=self.sample_command[:, 0].detach().clone().to(torch.float32),
            caller_enabled=(~self._motion_parking_mask()).detach().clone(),
            failure=state.failure.detach().clone().to(torch.bool),
            roll=state.roll.detach().clone().to(torch.float32),
            pitch=state.pitch.detach().clone().to(torch.float32),
            nonfoot_hit=(self.contacts.link_force.norm(dim=-1) > 10.).any(-1).detach().clone(),
        )

    def _advance_swing(self, state):
        sample = self._build_swing_sample(state)
        result = self.swing.update(sample)
        self._swing_result = result
        self._swing_aux = dict(
            foot_pos_w=sample.foot_pos_w.clone(),
            foot_pos_b=sample.foot_pos_b.clone(),
            executed_c=sample.executed_c.clone(),
            caller_enabled=sample.caller_enabled.clone(),
            nonfoot_hit=sample.nonfoot_hit.clone(),
            highest_safe_step=self.swing.highest_safe_step.detach().clone(),
            target_top_z=self.swing.target_top_z.detach().clone(),
            target_edge_x=self.swing.target_edge_x.detach().clone(),
            takeoff_support_z=self.swing.takeoff_support_z.detach().clone(),
            foot_radius=FOOT_RADIUS)

    def _advance_footfall(self, state):
        d = self.robot.data
        result = self.footfall_state.update(
            self.tile_ids, d.site_pos_w[:, self.foot_ids, :2], self.contacts.top_forces,
            self.geom_tile, self.geom_step, self.geom_step_bounds, state.failure, enabled=True)
        self._footfall_result = result
        # SDS 3.5: the amount is always recomputed from the event count (0.50 per
        # event), never taken from any pre-multiplied internal value.
        self._footfall_amount = footfall_reward(result.events)
        self.first_step_acquired |= (result.events & (result.selected_step == 1)).any(-1)

    def _advance_stall(self, state):
        # A parking diagnostic zeroes the command after arrival; the stall state
        # requires one constant command per episode, so it keeps the episode's
        # sampled c while `enabled` (and therefore `active`) is already false.
        parking = self._motion_parking_mask()
        command = torch.where(parking, self.episode_command, self.sample_command[:, 0])
        result = self.stall.update(state.local[:, 0], command, self.climb.climbed, ~parking)
        self._stall_result = result

    # ---------------------------------------------------------------- reward
    def _reward_terms(self, state):
        d = self.robot.data
        command = self.sample_command
        speed_norm = command.norm(dim=-1)
        active = speed_norm > COMMAND_MIN
        executed = command[:, 0]
        jp = d.joint_pos[:, self.joint_ids]
        default = d.default_joint_pos[:, self.joint_ids]
        v = d.root_link_lin_vel_b
        w = d.root_link_ang_vel_b
        dt = self.step_dt
        gate = self._swing_result.positive_gate
        # SDS 3.3/3.1: one ``dt`` per control step, no second multiply and no
        # compensating factor.  The returned value is the final step amount.
        speed_amount = speed_reward(v[:, 0], v[:, 1], executed, gate, dt=dt,
                                    enabled=active)
        active_f = active.to(torch.float32)
        std = torch.where(active[:, None], jp.new_tensor([.15, .35, .5] * 4),
                          jp.new_tensor([.05, .1, .15] * 4))
        contact = self.contacts.found
        foot_vel = d.site_lin_vel_w[:, self.foot_ids, :2].square().sum(-1)
        slip = (foot_vel * contact).sum(-1) * active_f
        impact = ((self.contacts.foot_force.norm(dim=-1)
                   * (contact & (self.contacts.contact_duration <= self.step_dt + 1e-7))).sum(-1)
                  * active_f)
        b = self.bounds[self.contacts.support_geom.clamp_min(0)].clone()
        n, h = self._parameters()
        support_top = self.geom_top[self.contacts.support_geom.clamp_min(0)]
        # Last tread and platform form one coplanar support rectangle.
        joined = (self.tile_ids[:, None] > 0) & ((support_top - n[:, None] * h[:, None]).abs() < .001)
        b[:, :, 0] = torch.where(joined, self.origins[self.tile_ids, 0, None] + (n[:, None] - 1) * .2, b[:, :, 0])
        b[:, :, 1] = torch.where(joined, self.origins[self.tile_ids, 0, None] + n[:, None] * .2 + 2, b[:, :, 1])
        p = d.site_pos_w[:, self.foot_ids]
        distance = torch.stack((p[:, :, 0] - b[:, :, 0], b[:, :, 1] - p[:, :, 0],
                                p[:, :, 1] - b[:, :, 2], b[:, :, 3] - p[:, :, 1]), -1).amin(-1)
        edge = (self.contacts.support * ((SAFE_INSET - distance) / SAFE_INSET).clamp_min(0).square()).mean(-1)
        limits = d.soft_joint_pos_limits[:, self.joint_ids]
        limit = ((limits[:, :, 0] - jp).clamp_min(0) + (jp - limits[:, :, 1]).clamp_min(0)).sum(-1)
        angmom = self.scene['robot/root_angmom'].data.square().sum(-1)
        raw = {
            'track_angular_velocity': torch.exp(-((w[:, 2] - command[:, 2]).square()
                                                  + .05 * w[:, :2].square().sum(-1)) / .5),
            'pose': torch.exp(-((jp - default) / std).square().mean(-1)),
            'body_ang_vel': d.root_link_ang_vel_w[:, :2].square().sum(-1),
            'angular_momentum': angmom,
            'joint_acc': d.joint_acc[:, self.joint_ids].square().sum(-1),
            'joint_limits': limit,
            'action_rate': (self.action_manager.action - self._previous_action).square().sum(-1),
            'slip': slip,
            'soft_landing': impact,
            'stand_still': (jp - default).square().sum(-1) * (speed_norm <= COMMAND_MIN),
            'center': (state.local[:, 1] / .2).square().clamp_max(4) * active_f,
            'heading': (1 - state.e_world.cos()) * active_f,
            'edge': edge,
        }
        amounts = {}
        for name, value in raw.items():
            amounts[name] = value * REWARD_WEIGHTS[name] * dt
        # Baseline-subtracted active terms (SDS 3.7: track_angular_velocity and pose
        # subtract their exact active baseline; the weight is applied separately).
        for name in ('track_angular_velocity', 'pose'):
            amounts[name] = (raw[name] - active_f) * REWARD_WEIGHTS[name] * dt
        amounts['orientation'] = orientation_reward(state.roll, state.pitch, dt=dt)
        amounts['nonfoot'] = nonfoot_reward(self.contacts.link_force, dt=dt)
        raw['orientation'] = -amounts['orientation'] / (REWARD_WEIGHTS['orientation'] * dt)
        raw['nonfoot'] = -amounts['nonfoot'] / (REWARD_WEIGHTS['nonfoot'] * dt)
        amounts['speed'] = speed_amount
        amounts['stall'] = self._stall_result.reward
        success_event = self.climb_event & ~state.failure
        amounts['footfall'] = self._footfall_amount
        amounts['swing'] = self._swing_result.reward
        amounts['success'] = success_event.to(torch.float32) * SUCCESS_AMOUNT
        amounts['failure'] = state.failure.to(torch.float32) * FAILURE_AMOUNT
        return raw, amounts

    def _reward(self, state, previous_action):
        """Compose 0.5-r2 once (numerically identical to 0.5-r1); no route input."""
        self._previous_action = previous_action
        raw, amounts = self._reward_terms(state)
        total = torch.zeros(self.num_envs, device=self.device)
        tensors = {}
        for name, value in raw.items():
            tensors['reward_raw/' + name] = value
        for name, value in amounts.items():
            tensors['reward_weighted/' + name] = value
            total = total + value
        tensors['reward_total'] = total
        # Continuously false by construction: the legacy body progress rate is
        # diagnostic only (SDS 3.1).  It never enters the total or episode sums.
        tensors['diagnostic/old_body_progress'] = torch.zeros_like(total)
        tensors['diagnostic/old_body_progress_rate'] = torch.zeros_like(total)
        # Only actual rewarded terms accumulate; raw/diagnostic namespaces never do.
        for key, term in tensors.items():
            if key.startswith('diagnostic/'):
                continue
            if key not in self.episode_reward_sums:
                self.episode_reward_sums[key] = torch.zeros_like(term)
            self.episode_reward_sums[key] += term
        self.step_reward_tensors = tensors
        self._last_reward_terms = dict(amounts)
        self.reward_terms = {key: float(value.mean().detach().cpu()) for key, value in tensors.items()}
        return total

    # ------------------------------------------------------------ diagnostics
    def _diagnostics_batch(self):
        swing = self._swing_result
        stall = self._stall_result
        footfall = self._footfall_result
        d = self.robot.data
        local = d.root_link_pos_w - self.origins[self.tile_ids]
        route = self.route_metrics()
        route_error = route['route_error'].clone()
        data = dict(
            stage=self.stage, tile_N=self.tile_N.detach().clone(),
            command_c=self.sample_command[:, 0].detach().clone(),
            control_step=self.episode_length_buf.detach().clone(),
            local_position=local.detach().clone(),
            body_vx=d.root_link_lin_vel_b[:, 0].detach().clone(),
            heading=euler_heading(d.root_link_quat_w)[2].detach().clone(),
            gate_open=swing.positive_gate.detach().clone(),
            climb_success=self.climb_event.detach().clone(),
            failure=(self.last_termination[1] if self.last_termination is not None
                     else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)),
            timeout=(self.last_termination[2] if self.last_termination is not None
                     else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)),
            first_step_acquired=self.first_step_acquired.detach().clone(),
            stall_on=stall.penalty_on.detach().clone(),
            stall_penalty=stall.reward.detach().clone(),
            stall_dx_1s=stall.dx_1s.detach().clone(),
            stall_highwater_dx_1s=stall.highwater_dx_1s.detach().clone(),
            stall_window_ready=stall.window_ready.detach().clone(),
            stall_qualifying=stall.qualifying.detach().clone(),
            stall_consecutive_count=stall.consecutive_count.detach().clone(),
            swing_paid_fraction=swing.consumed_fraction.detach().clone(),
            swing_paid_amount=swing.cumulative_paid.detach().clone(),
            swing_safe_support=swing.safe_support.detach().clone(),
            swing_selected_step=swing.selected_step.detach().clone(),
            swing_target_step=swing.target_step.detach().clone(),
            swing_active=swing.active.detach().clone(),
            swing_paid_fraction_all=self.swing.paid_fraction.detach().clone(),
            swing_paid_amount_all=self.swing.actual_paid_amount.detach().clone(),
            footfall_count=footfall.events.sum(-1).detach().clone(),
            footfall_qualified_count=footfall.qualified_count.detach().clone(),
            footfall_selected_step=footfall.selected_step.detach().clone(),
            footfall_amount=self._footfall_amount.detach().clone(),
            # SDS 6.3 diagnostic denominators, pre-reset and per environment.
            active_samples=self.active_samples.detach().clone(),
            stall_active_samples=self.stall_active_samples.detach().clone(),
            diagnostic_denominator_valid=(self.active_samples > 0).detach().clone(),
            stall_fraction=torch.where(
                self.active_samples > 0,
                self.stall_active_samples.to(torch.float32)
                / self.active_samples.clamp_min(1).to(torch.float32),
                torch.zeros(self.num_envs, device=self.device)),
            speed_rate=(self._last_reward_terms['speed'] / self.step_dt
                        if 'speed' in self._last_reward_terms else torch.zeros(self.num_envs, device=self.device)),
            speed_reward=self._last_reward_terms.get(
                'speed', torch.zeros(self.num_envs, device=self.device)).detach().clone(),
            reward_terms={key: value.detach().clone()
                          for key, value in self._last_reward_terms.items()},
            old_body_progress=torch.zeros(self.num_envs, device=self.device),
            slot_class=list(getattr(self, 'slot_class_names', ['flat'] * self.num_envs)),
            # ---- r2 route diagnostics: read-only, never a network input.
            route_error=route_error,
            route=dict(
                local_y=route['local_y'], abs_y=route['abs_y'],
                normalized_abs_y=route['normalized_abs_y'],
                normalized_abs_y_raw=route['normalized_abs_y_raw'],
                saturation=route['saturation'], e_world=route['e_world'],
                heading_deg=route['heading_deg'], yaw_world=route['yaw_world'],
                tile_ids=route['tile_ids'], yaw_axis=route['yaw_axis'],
                nonfinite_count=route['nonfinite_count'],
                lateral_scale=ROUTE_LATERAL_SCALE, clip=ROUTE_CLIP,
                saturation_m=ROUTE_LATERAL_SATURATION_M,
                components=list(ROUTE_COMPONENTS)),
        )
        return data

    @property
    def diagnostics(self):
        return self._last_step_diagnostics

    @property
    def last_reward_terms(self):
        return self._last_reward_terms

    def _swing_payload(self):
        result = self._swing_result
        if result is None:
            return {}
        return dict(target_step=result.target_step.detach().clone(),
                    selected_step=result.selected_step.detach().clone(),
                    phi=result.phi.detach().clone(),
                    increment=result.increment.detach().clone(),
                    consumed_fraction=result.consumed_fraction.detach().clone(),
                    cumulative_paid=result.cumulative_paid.detach().clone(),
                    safe_support=result.safe_support.detach().clone(),
                    active=result.active.detach().clone(),
                    positive_gate=result.positive_gate.detach().clone(),
                    gate_reason=(result.gate_details or {}).get('reason'))

    def diagnostics_snapshot(self, env_id=0):
        """Pre-reset per-environment diagnostics for evidence and video overlays."""
        i = int(env_id)
        d = self.robot.data
        batch = self._last_step_diagnostics
        swing = self._swing_payload()
        route = self.route_metrics()
        values = {
            'env_id': i, 'stage': self.stage, 'protocol_id': self.protocol,
            'code_version': CODE_VERSION, 'recipe': RECIPE_VERSION,
            'step': int(self.episode_length_buf[i]), 'tile_id': int(self.tile_ids[i]),
            'tile_N': int(self.tile_N[i]), 'slot_class': list(getattr(self, 'slot_class_names', []))[i],
            'quaternion': d.root_link_quat_w[i].detach().clone(),
            'position': (d.root_link_pos_w[i] - self.origins[self.tile_ids[i]]).detach().clone(),
            'joint_position': d.joint_pos[i, self.joint_ids].detach().clone(),
            'joint_velocity': d.joint_vel[i, self.joint_ids].detach().clone(),
            'action': self.action_manager.action[i].detach().clone(),
            'linear_velocity_body': d.root_link_lin_vel_b[i].detach().clone(),
            'angular_velocity_body': d.root_link_ang_vel_b[i].detach().clone(),
            'command': self.command[i].detach().clone(),
            'command_executed': self.sample_command[i].detach().clone(),
            'foot_position': (d.site_pos_w[i, self.foot_ids]
                              - self.origins[self.tile_ids[i]]).detach().clone(),
            'foot_force': self.contacts.foot_force[i].detach().clone(),
            'nonfoot_force': self.contacts.link_force[i].detach().clone(),
            'support': self.contacts.support[i].detach().clone(),
            'heading': float(euler_heading(d.root_link_quat_w)[2][i]),
            'speed_metrics': self.speed_router.episode_metrics(
                i, failure=bool(self.last_termination[1][i]) if self.last_termination is not None else False,
                climb_success=bool(self.climb.climbed[i])),
            'reward_terms': self.reward_terms,
            'route_error': route['route_error'][i].detach().clone(),
            'route': dict(local_y=float(route['local_y'][i]),
                          abs_y=float(route['abs_y'][i]),
                          normalized_abs_y=float(route['normalized_abs_y'][i]),
                          normalized_abs_y_raw=float(route['normalized_abs_y_raw'][i]),
                          saturation=bool(route['saturation'][i]),
                          e_world=float(route['e_world'][i]),
                          heading_deg=float(route['heading_deg'][i]),
                          yaw_world=float(route['yaw_world'][i]),
                          yaw_axis=ROUTE_YAW_AXIS,
                          lateral_scale=ROUTE_LATERAL_SCALE, clip=ROUTE_CLIP,
                          saturation_m=ROUTE_LATERAL_SATURATION_M),
            'initial_pose_source': self.episode_initial_source[i],
            'initial_sample_order': INITIAL_SAMPLE_ORDER,
            'rng_state_sha256_before_sample': self.episode_rng_sha[i],
        }
        for name in ('gate_open', 'climb_success', 'failure', 'timeout', 'stall_on',
                     'stall_qualifying', 'stall_window_ready', 'first_step_acquired'):
            if name in batch:
                values[name] = bool(batch[name][i])
        for name in ('stall_dx_1s', 'stall_highwater_dx_1s', 'stall_penalty', 'speed_reward',
                     'footfall_count', 'swing_paid_amount', 'swing_paid_fraction'):
            if name in batch:
                values[name] = batch[name][i].detach().clone()
        if swing:
            for key in ('target_step', 'selected_step', 'phi', 'increment', 'cumulative_paid',
                        'safe_support', 'active'):
                values['swing_' + key] = swing[key][i].detach().clone()
        values.update(success=bool(self.last_termination[0][i]) if self.last_termination is not None else False,
                      truncated=bool(self.last_termination[2][i]) if self.last_termination is not None else False,
                      reason=(REASON_NAMES[int(self.last_termination[3][i])]
                              if self.last_termination is not None else 'running'))
        return values

    # ------------------------------------------------------------------- step
    def step(self, action):
        if not torch.isfinite(action).all():
            raise NumericalFailure('Nonfinite action', {
                'action': action.detach().cpu(),
                'qpos': self.sim.data.qpos.detach().cpu(),
                'qvel': self.sim.data.qvel.detach().cpu()})
        previous = self.action_manager.action.clone()
        self.action_manager.process_action(action)
        self.control_body_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.control_long_hit = torch.zeros_like(self.control_body_hit)
        for substep in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            self.contacts.sample(self.tile_ids, physics_dt=self.physics_dt)
            self.control_body_hit |= (self.contacts.link_force[:, self.contacts.body_link_mask]
                                      .norm(dim=-1) > 0).any(-1)
            self.control_long_hit |= (self.contacts.nonfoot_duration >= .1 - 1e-6).any(-1)
        self.contacts.sample(self.tile_ids, control=True)
        self.sim.sense()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        state = self._task_state()
        reward = self._reward(state, previous)
        # Executed command and window statistics for this control sample, before
        # any reset or parking zeroing can change them.
        self.speed_router.update(
            step=self.episode_length_buf.detach().clone(),
            local_x=state.local[:, 0].detach().to(torch.float32),
            body_vx=self.robot.data.root_link_lin_vel_b[:, 0].detach().to(torch.float32),
            executed_c=self.sample_command[:, 0].detach().clone().to(torch.float32),
            enabled=~self._motion_parking_mask(),
            local_y=state.local[:, 1].detach().to(torch.float32),
            # The r1 speed windows consumed the raw euler yaw; keep it bit-identical
            # (e_world == local_yaw because yaw_axis == 0, and the route vector is
            # the only new consumer of e_world).
            yaw=state.yaw.detach().to(torch.float32),
            tile_ids=self.tile_ids)
        self._update_episode_stats(state, reward)
        # SDS 4.1/4.5: the terminal observation (with its pre-reset route_error) is
        # built from this physical sample, *before* the automatic reset.
        terminal = self._observations(update=True)
        if not torch.isfinite(terminal['route_error']).all():
            raise NumericalFailure(
                'Nonfinite route_error before reset',
                {'stage': self.stage,
                 'route_error': terminal['route_error'].detach().cpu(),
                 'tile_ids': self.tile_ids.detach().cpu(),
                 'slot_class': list(getattr(self, 'slot_class_names', [])),
                 'reward': reward.detach().cpu(),
                 'qpos': self.sim.data.qpos.detach().cpu(),
                 'qvel': self.sim.data.qvel.detach().cpu()})
        self.episode_route_final = terminal['route_error'].detach().clone()
        self.last_termination = (state.success, state.failure, state.truncated, state.reason)
        self._last_step_diagnostics = self._diagnostics_batch()
        terminal_observation = {key: value.clone() for key, value in terminal.items()}
        done = state.success | state.failure | state.truncated
        episodes = self._episode_records(state, done)
        info = dict(
            terminal_observation=terminal_observation, episodes=episodes,
            failure=state.failure.detach().clone(),
            climb=self.climb_event.detach().clone(),
            timeout=state.truncated.detach().clone(),
            first_step_acquired=self.first_step_acquired.detach().clone(),
            stall_on=self._stall_result.penalty_on.detach().clone(),
            reset_mask=done.detach().clone(),
            reward_terms=self.reward_terms,
            step_reward_tensors=self.step_reward_tensors,
            diagnostics=self._last_step_diagnostics,
            command_executed=self.sample_command.detach().clone(),
            protocol_id=self.protocol, code_version=CODE_VERSION,
            recipe_version=RECIPE_VERSION, reward_version=RECIPE_VERSION,
            speed_snapshot=self.speed_router.snapshot(),
            swing=self._swing_payload(),
            # SDS 4.5: the formal pre-reset values; never a replaced position.
            route_error=terminal_observation['route_error'],
            route_error_terminal=terminal_observation['route_error'],
        )
        if (state.reason == 9).any() or not torch.isfinite(reward).all():
            raise NumericalFailure('Numerical state/reward failure before reset', {
                'episodes': episodes,
                'terminal_observation': {key: value.detach().cpu() for key, value in terminal.items()},
                'reward': reward.detach().cpu(),
                'diagnostics': {key: value.detach().cpu() for key, value in self._last_step_diagnostics.items() if torch.is_tensor(value)},
                'qpos': self.sim.data.qpos.detach().cpu(),
                'qvel': self.sim.data.qvel.detach().cpu()})
        if bool(done.any()):
            self._reset_ids(done.nonzero().flatten())
        obs = self._observations(update=False)
        # SDS 4.5: the returned new-episode observation carries the post-reset route.
        info['route_error_next'] = obs['route_error']
        return obs, reward, state.success | state.failure, state.truncated, info

    # --------------------------------------------------------------- records
    def _update_episode_stats(self, state, reward):
        d = self.robot.data
        count = self.episode_length_buf.clamp_min(1).to(torch.float32)
        self.sum_vx += d.root_link_lin_vel_b[:, 0]
        self.sum_speed += d.root_link_lin_vel_b.norm(dim=-1)
        self.max_lateral = torch.maximum(self.max_lateral, state.abs_y)
        self.max_abs_yaw = torch.maximum(self.max_abs_yaw, state.yaw.abs())
        self.max_body_x = torch.maximum(self.max_body_x, state.local[:, 0])
        self.route_max_abs_y = torch.maximum(self.route_max_abs_y, state.abs_y)
        self.route_max_abs_heading = torch.maximum(self.route_max_abs_heading,
                                                   state.e_world.abs())
        torque = self.sim.data.qfrc_actuator[:, self.robot.indexing.joint_v_adr][:, self.joint_ids]
        self.peak_torque = torch.maximum(self.peak_torque, torque.abs().amax(-1))
        self.footfall_total += self._footfall_result.events.sum(-1)
        self.swing_total += self._swing_result.reward
        self.stall_samples += self._stall_result.penalty_on.long()
        # SDS 6.3 diagnostic denominator: returned control samples that had an
        # active command and had not yet achieved a safe arrival (the arrival
        # frame itself already counts as arrived and is excluded).
        active_now = (self.sample_command[:, 0] > COMMAND_MIN) & ~self.climb.climbed
        self.active_samples += active_now.long()
        self.stall_active_samples += (active_now & self._stall_result.penalty_on).long()

    def _episode_records(self, state, done):
        records = []
        if not bool(done.any()):
            return records
        d = self.robot.data
        count = self.episode_length_buf.clamp_min(1).to(torch.float32)
        for i in done.nonzero().flatten().tolist():
            n = int(self.tile_N[i])
            speed = self.speed_router.episode_metrics(
                i, failure=bool(state.failure[i]), climb_success=bool(self.climb.climbed[i]))
            records.append(dict(
                env_id=int(i), seed=int(self.run_seed), stage=self.stage,
                tile_id=int(self.tile_ids[i]), N=n, h=float(self.atlas_H[self.tile_ids[i]]),
                vx=float(self.episode_command[i]),
                slot_class=list(getattr(self, 'slot_class_names', ['flat'] * self.num_envs))[i],
                protocol_id=self.protocol, code_version=CODE_VERSION,
                recipe=RECIPE_VERSION, reward_version=RECIPE_VERSION,
                reward_weights_source=REWARD_WEIGHTS_SOURCE,
                checkpoint=getattr(self, 'checkpoint_identity', None),
                initial_pose=self.episode_initial_pose[i].detach().cpu().tolist(),
                initial_pose_source=self.episode_initial_source[i],
                initial_sample_order=INITIAL_SAMPLE_ORDER,
                rng_state_sha256_before_sample=self.episode_rng_sha[i],
                success=bool(self.climb.climbed[i]), climb_success=bool(self.climb.climbed[i]),
                climb_event=bool(self.climb_event[i]),
                climb_step=(int(self.climb.climb_step[i]) if bool(self.climb.climbed[i]) else None),
                failure=bool(state.failure[i]), timeout=bool(state.truncated[i]),
                truncated=bool(state.truncated[i]),
                reason=REASON_NAMES[int(state.reason[i])],
                terminal_step=int(self.episode_length_buf[i]),
                duration=float(self.episode_length_buf[i]) * self.step_dt,
                control_samples=int(self.episode_length_buf[i]),
                mean_vx=float(self.sum_vx[i] / count[i]),
                mean_speed=float(self.sum_speed[i] / count[i]),
                max_lateral=float(self.max_lateral[i]), max_abs_yaw=float(self.max_abs_yaw[i]),
                max_body_x=float(self.max_body_x[i]), peak_torque=float(self.peak_torque[i]),
                first_step_acquired=bool(self.first_step_acquired[i]),
                footfall_events=float(self.footfall_total[i]),
                swing_paid=float(self.swing_total[i]),
                stall_samples=int(self.stall_samples[i]),
                active_samples=int(self.active_samples[i]),
                stall_active_samples=int(self.stall_active_samples[i]),
                stall_fraction=(float(self.stall_active_samples[i])
                                / float(self.active_samples[i])
                                if int(self.active_samples[i]) > 0 else None),
                diagnostic_denominator_valid=bool(int(self.active_samples[i]) > 0),
                gate_open=bool(self._swing_result.positive_gate[i]),
                stall_on=bool(self._stall_result.penalty_on[i]),
                # SDS 4.8 episode-level route record (pre-reset final value).
                route_final=self.episode_route_final[i].detach().cpu().tolist(),
                route_max_abs_y=float(self.route_max_abs_y[i]),
                route_max_abs_heading_deg=float(self.route_max_abs_heading[i] * (180. / math.pi)),
                route_initial=self.episode_route_initial[i].detach().cpu().tolist(),
                route_yaw_axis=ROUTE_YAW_AXIS,
                reward_sums={key: float(value[i]) for key, value in self.episode_reward_sums.items()},
                speed_metrics=speed,
            ))
        return records

    # --------------------------------------------------------------- obs entry
    def _observations(self, update=True):
        """The five r2 observation keys; ``route_error`` is refreshed every call."""
        d = self.robot.data
        prop = self._proprio()
        if update:
            self.history_cache.update(prop)
            self.map_cache.update(d.root_link_pos_w, euler_heading(d.root_link_quat_w)[2],
                                  self.tile_ids)
        h, missing = self._foot_heights()
        force = quat_apply_inverse(
            d.root_link_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4),
            self.contacts.foot_force.reshape(-1, 3)).reshape(self.num_envs, 12)
        # qfrc_actuator is joint generalized actuation force, already in joint order.
        torque = self.sim.data.qfrc_actuator[:, self.robot.indexing.joint_v_adr][:, self.joint_ids]
        privileged = torch.cat((d.root_link_lin_vel_b,
                                (self.contacts.foot_force.norm(dim=-1) > 5).float(),
                                force, h, torque), -1)
        return {'proprio': prop.clone(), 'history': self.history_cache.history.clone(),
                'heightmap': self.map_cache.map.clone(), 'privileged': privileged,
                'route_error': self.route_error()}
