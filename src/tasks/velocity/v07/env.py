"""v0.7 environment: real 23-tile terrain, schedule commits and per-env outcomes.

``V07Env`` implements the frozen W0 4.3 signature on top of the same physical
lifecycle as ``V05R2Env`` (frozen control step, contact sampling, pre-reset
terminal snapshot, partial reset).  It deliberately re-initialises the scene from
:func:`src.tasks.velocity.v07.env_cfg.env_cfg` instead of the frozen v0.2 cfg, so
the v0.7 collision geometry is the only terrain that can exist in this scene
(SDS 7.1: no implicit ``v02.spec.STAGES`` binding).

Key v0.7 properties:

* ``set_schedule`` applies a curriculum schedule **only at an update boundary**
  and resets exactly the environments whose phase/substage/role/tile changed;
* roles follow SDS 5.3 (flat 102 / frontier 164 / near 123 / easy 123 on 512
  envs, largest remainder otherwise), near/easy tiles are resampled at every
  episode reset from an independent, resumable sampler RNG, and only frontier
  episodes are emitted with ``role='frontier'`` for the curriculum;
* spawn and command follow SDS 5.4, including the V command ramp and the R spawn
  ramp (flat environments keep the smaller range in R);
* the reward is composed by ``v07.rewards.RewardComputer`` (agent A); this file
  only builds ``RewardFrame`` snapshots and never duplicates a reward formula;
* every control step exposes per-environment outcome tensors and every finished
  episode is returned in ``info['episodes']`` with the pre-reset state.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

import src.tasks.velocity.config.a2  # noqa: F401  (A2 registry load order)
from . import curriculum as curriculum_module
from . import events, spec
from . import rewards as reward_module
from . import terrain
from .contacts import ContactState
from .curriculum import ScheduleAssignment
from .env_cfg import env_cfg
from .events import euler_heading
from .map import HeightMap, ProprioHistory

__all__ = ['V07Env', 'ScheduleAssignment', 'NumericalFailure']


class NumericalFailure(FloatingPointError):
    """Fail-fast exception carrying pre-reset evidence for the runner.

    The frozen v0.2/v0.5 lifecycle raises this type; v0.7 owns its copy so that
    the production line never imports an older version at runtime.
    """

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics

CONTROL_DT = float(spec.CONTROL_DT)
DECIMATION = int(spec.DECIMATION)
ACTION_DIM = int(spec.N_JOINTS)
FOOT_ORDER = tuple(spec.FOOT_ORDER)
FLAT_TILE_ID = int(spec.FLAT_TILE_ID)
PHASE_ORDER = tuple(spec.PHASE_ORDER)
REASON_NAMES = tuple(events.REASON_NAMES)
OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', 'privileged', 'route_error')

#: SDS 5.4 command / spawn ranges, bound from the single frozen source.
COMMAND_INITIAL_VX = tuple(float(v) for v in spec.COMMAND_INITIAL_VX)
COMMAND_FINAL_VX = tuple(float(v) for v in spec.COMMAND_FINAL_VX)
V_RAMP_UPDATES = int(spec.V_RAMP_UPDATES)
SPAWN_X = tuple(float(v) for v in spec.SPAWN_X)
SPAWN_Y = tuple(float(v) for v in spec.SPAWN_Y)
SPAWN_YAW = tuple(float(v) for v in spec.SPAWN_YAW_RADIANS)
R_SPAWN_X = tuple(float(v) for v in spec.R_SPAWN_X)
R_SPAWN_Y = tuple(float(v) for v in spec.R_SPAWN_Y)
R_SPAWN_YAW = tuple(float(v) for v in spec.R_SPAWN_YAW_RADIANS)
R_SPAWN_RAMP_UPDATES = int(spec.R_SPAWN_RAMP_UPDATES)
SAFE_INSET = float(spec.SAFE_INSET)
FLAT_EPISODE_STEPS = int(spec.F0_EPISODE_STEPS)
ROUTE_LATERAL_SCALE = float(spec.ROUTE_LATERAL_SCALE)
ROUTE_LATERAL_CLIP = float(spec.ROUTE_LATERAL_CLIP)

#: Independent sampler stream offset (SDS 5.3: role/terrain sampling has its own
#: resumable RNG, never the spawn/command generator).
TERRAIN_RNG_OFFSET = 7919
SOURCE_SAMPLED = 'v07_sampled'
SOURCE_CASES = 'initial_cases'


#: Inherited r2 stall-diagnostic constants, read from the single frozen source
#: (SDS 7.1: spec.py is the only numeric source; Q001-7 keeps the r2 detection).
#: They are diagnostics only and are never paid (SDS 3.6/6.3).
STALL_HISTORY_SAMPLES = int(spec.STALL_HISTORY_SAMPLES)
STALL_DX_FRACTION_PER_SECOND = float(spec.STALL_DX_FRACTION_PER_SECOND)
STALL_DX_THRESHOLD_SCALE = float(spec.STALL_DX_THRESHOLD_SCALE)
STALL_HIGHWATER_DX = float(spec.STALL_HIGHWATER_DX)
STALL_CONSECUTIVE_SAMPLES = int(spec.STALL_CONSECUTIVE_SAMPLES)
STALL_ACTIVE_COMMAND = float(spec.STALL_ACTIVE_COMMAND)
#: Frozen inherited r2 values; a spec edit that changes the diagnostic semantics
#: must be an explicit decision, never a silent drift.
_EXPECTED_STALL = (51, 0.15, 1.0, 0.03, 50, 0.1)
if (STALL_HISTORY_SAMPLES, STALL_DX_FRACTION_PER_SECOND,
        STALL_DX_THRESHOLD_SCALE, STALL_HIGHWATER_DX,
        STALL_CONSECUTIVE_SAMPLES, STALL_ACTIVE_COMMAND) != _EXPECTED_STALL:
    raise RuntimeError(
        'v07 env stall diagnostics drifted from the inherited r2 definition: '
        f'{_EXPECTED_STALL} != spec values')


class StallDiagnostics:
    """Inherited r2 stall state: **diagnostic only**, never a reward term.

    Exactly one 51-sample position/high-water window per environment (50 control
    intervals = 1.00 s).  A sample is *active* while the episode has a positive
    command and has not reached the platform; it is a *stall* sample once the
    first step has been acquired and the 1 s forward/high-water progress is below
    the r2 thresholds for ``STALL_CONSECUTIVE_SAMPLES`` consecutive samples.
    There is no payment attached (SDS 3.6/6.3, Q001-7): ``stall_samples`` is a
    count for the diagnostic collapse decision only.
    """

    def __init__(self, num_envs, device='cpu'):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        b = self.num_envs
        self.x_history = torch.zeros(b, STALL_HISTORY_SAMPLES, device=device)
        self.m_history = torch.zeros(b, STALL_HISTORY_SAMPLES, device=device)
        self.count = torch.zeros(b, dtype=torch.long, device=device)
        self.consecutive_count = torch.zeros(b, dtype=torch.long, device=device)
        self.executed_c = torch.full((b,), float('nan'), device=device)
        self.dx_1s = torch.zeros(b, device=device)
        self.highwater_dx_1s = torch.zeros(b, device=device)
        self.window_ready = torch.zeros(b, dtype=torch.bool, device=device)
        self.qualifying = torch.zeros(b, dtype=torch.bool, device=device)
        self.penalty_on = torch.zeros(b, dtype=torch.bool, device=device)
        self.active_samples = torch.zeros(b, dtype=torch.long, device=device)
        self.stall_samples = torch.zeros(b, dtype=torch.long, device=device)

    def reset(self, ids, initial_x):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        initial_x = torch.as_tensor(initial_x, device=self.device).to(torch.float32)
        self.x_history[ids] = initial_x[:, None]
        self.m_history[ids] = initial_x[:, None]
        self.count[ids] = 1
        self.consecutive_count[ids] = 0
        self.executed_c[ids] = float('nan')
        self.dx_1s[ids] = 0.0
        self.highwater_dx_1s[ids] = 0.0
        self.window_ready[ids] = False
        self.qualifying[ids] = False
        self.penalty_on[ids] = False
        self.active_samples[ids] = 0
        self.stall_samples[ids] = 0

    def update(self, local_x, executed_c, climbed, first_step_acquired, enabled):
        active = enabled & (executed_c > STALL_ACTIVE_COMMAND) & ~climbed
        known = torch.isfinite(self.executed_c)
        changed = known & (self.executed_c != executed_c.to(self.executed_c.dtype))
        if bool(changed.any()):
            raise ValueError('executed command changed inside an episode; '
                             'reset the stall diagnostics')
        x = local_x.to(self.x_history.dtype)
        m = torch.maximum(self.m_history[:, -1], x)
        self.executed_c = torch.where(known, self.executed_c,
                                      executed_c.to(self.executed_c.dtype))
        self.x_history = torch.cat((self.x_history[:, 1:], x[:, None]), dim=1)
        self.m_history = torch.cat((self.m_history[:, 1:], m[:, None]), dim=1)
        self.count = (self.count + 1).clamp_max(STALL_HISTORY_SAMPLES)
        window_ready = self.count >= STALL_HISTORY_SAMPLES
        dx = self.x_history[:, -1] - self.x_history[:, 0]
        dm = self.m_history[:, -1] - self.m_history[:, 0]
        threshold = (STALL_DX_FRACTION_PER_SECOND * executed_c
                     * STALL_DX_THRESHOLD_SCALE).to(dx.dtype)
        qualifying = (active & first_step_acquired & ~climbed & window_ready
                      & (dx < threshold) & (dm < STALL_HIGHWATER_DX))
        self.consecutive_count = torch.where(
            qualifying, self.consecutive_count + 1,
            torch.zeros_like(self.consecutive_count))
        self.penalty_on = qualifying & (self.consecutive_count
                                        >= STALL_CONSECUTIVE_SAMPLES)
        self.dx_1s = dx
        self.highwater_dx_1s = dm
        self.window_ready = window_ready
        self.qualifying = qualifying
        self.active_samples += active.long()
        self.stall_samples += self.penalty_on.long()
        return self

    def state_dict(self):
        return dict(active_samples=self.active_samples.detach().to('cpu').tolist(),
                    stall_samples=self.stall_samples.detach().to('cpu').tolist(),
                    dx_1s=self.dx_1s.detach().to('cpu').tolist(),
                    highwater_dx_1s=self.highwater_dx_1s.detach().to('cpu').tolist())


class V07Env(ManagerBasedRlEnv):
    """v0.7 environment over the real 23-tile v0.7 terrain.

    ``ManagerBasedRlEnv`` is the same physical base class the frozen
    ``v02_formal.env.FormalEnv`` uses; this class implements the frozen
    lifecycle contract (constructor arguments, ``reset``/``step`` shapes,
    pre-reset terminal snapshot, partial reset) itself so the v0.7 production
    line stays self-closed and never imports ``v02``/``v02_formal`` at runtime.
    """

    def __init__(self, num_envs=512, device='cuda:0', seed=42, phase='F0',
                 substage=None, stage_iteration=0, evaluation=False,
                 fixed_vx=None, initial_cases=None, render_mode=None,
                 parking_diagnostic=False, level_tick: int = 0,
                 frontier_tile_id=None):
        # The frozen base hard-codes the v0.2 cfg; v0.7 initialises the scene from
        # the v0.7 cfg and owns the same lifecycle methods below.
        phase = str(phase).upper()
        if phase not in PHASE_ORDER:
            raise ValueError(f'unknown v0.7 phase: {phase!r}')
        spec.phase_spec(phase)
        substage = _normalise_substage(phase, substage)
        self.run_seed = int(seed)
        self.recipe = spec.RECIPE_VERSION
        self.reward_version = spec.RECIPE_VERSION
        self.protocol = spec.PROTOCOL_ID
        self.parking_diagnostic = bool(parking_diagnostic)
        self.evaluation = bool(evaluation)
        self.fixed_vx = _validate_fixed_vx(fixed_vx)
        self.checkpoint_identity = None
        self.phase = phase
        self.substage = substage
        self.stage = phase
        self.stage_iteration = int(stage_iteration)
        self.previous_stage = None
        self.initial_cases, self.initial_command = _case_tensor(
            initial_cases, int(num_envs), device)
        # SDS 6.1: a gate/eval environment must use the *current* frontier, never
        # a hard-coded H00.  ``level_tick`` selects the H row; an explicit
        # ``frontier_tile_id`` pins every environment to one terrain.
        self.level_tick = _validate_level_tick(level_tick)
        self._tile_override = _validate_frontier_tile(
            frontier_tile_id, phase=phase, substage=substage,
            level_tick=self.level_tick, evaluation=bool(evaluation))
        if self._tile_override is not None:
            self.frontier_tile_id = int(self._tile_override)
        else:
            self.frontier_tile_id = int(curriculum_module.frontier_tile_for_state(
                phase, substage, self.level_tick))
        if bool(evaluation) and self._tile_override is None:
            # An evaluation environment always measures one real terrain (the
            # current frontier), never a mixed near/easy sample.
            self._tile_override = int(self.frontier_tile_id)
        self.generation = 0
        self.roles: list[str] = list(curriculum_module.allocate_roles(int(num_envs), phase))

        ManagerBasedRlEnv.__init__(self, env_cfg(int(num_envs), int(seed)), device,
                                   render_mode)

        # -------------------------------------------------------- generators
        self.rng = torch.Generator(device=device).manual_seed(int(seed))
        self._tile_sampler = np.random.default_rng(int(seed) + TERRAIN_RNG_OFFSET)
        # --------------------------------------------------------- entities
        self.robot = self.scene['robot']
        expected = [f'{leg}_{j}_joint' for leg in FOOT_ORDER
                    for j in ('hip', 'thigh', 'calf')]
        self.joint_ids, _ = self.robot.find_joints(expected, preserve_order=True)
        self.foot_ids, _ = self.robot.find_sites(list(FOOT_ORDER), preserve_order=True)
        self.action_joint_ids = self.action_manager.get_term('joint_pos')._target_ids
        if self.action_joint_ids.tolist() != list(self.joint_ids):
            raise ValueError('v07 action targets differ from the ETH joint order')
        self.action_term = self.action_manager.get_term('joint_pos')
        # -------------------------------------------------- v0.7 terrain
        self.specs = terrain.build_terrain_specs()
        self.origins = torch.tensor([s.frame_origin_world for s in self.specs],
                                    device=device, dtype=torch.float32)
        self.spawn_x = torch.tensor(
            [0.0 if int(s.N) == 0 else -1.0 for s in self.specs],
            device=device, dtype=torch.float32)
        self.tile_N = torch.tensor([int(s.N) for s in self.specs], device=device)
        self.tile_h = torch.tensor([float(s.h) for s in self.specs], device=device)
        self.tile_d = torch.tensor([float(s.step_depth) for s in self.specs],
                                   device=device)
        geometry = terrain.index_geometry(self.sim.mj_model, self.specs)
        tensors = terrain.tensorize_geometry(geometry, device)
        self.bounds = tensors['bounds']
        self.geom_top = tensors['top']
        self.geom_tile = tensors['tile']
        self.geom_step = tensors['step']
        self.geom_step_bounds = tensors['support_bounds']
        self.geom_ids = tuple(geometry.geom_ids)
        self.contacts = ContactState(self.sim, geometry.tile, geometry.top, device)
        self.map_cache = HeightMap(int(num_envs), device)
        self.history_cache = ProprioHistory(int(num_envs), device)
        self.command = self.command_manager.get_command('twist')
        self.reward_computer = reward_module.RewardComputer()
        self.reward_computer.reset()
        # ------------------------------------------------------ v0.7 state
        b = int(num_envs)
        self.tile_ids = torch.zeros(b, dtype=torch.long, device=device)
        self.episode_command = torch.zeros(b, device=device)
        self.start_vx = torch.zeros(b, device=device)
        self.sample_command = torch.zeros(b, 3, device=device)
        self.climb = events.PlatformSupportState(b, device)
        self.stall = StallDiagnostics(b, device)
        self.first_step_acquired = torch.zeros(b, dtype=torch.bool, device=device)
        self.control_body_hit = torch.zeros(b, dtype=torch.bool, device=device)
        self.control_long_hit = torch.zeros(b, dtype=torch.bool, device=device)
        self.stop = torch.zeros(b, dtype=torch.bool, device=device)
        self.stop_steps = torch.zeros(b, dtype=torch.long, device=device)
        self.episode_initial_pose = torch.zeros(b, 3, device=device)
        self.episode_initial_source = [SOURCE_SAMPLED] * b
        self.episode_rng_sha = [''] * b
        self._local_x_prev = torch.zeros(b, device=device)
        self._tracking_sum = torch.zeros(b, device=device)
        self._tracking_count = torch.zeros(b, dtype=torch.long, device=device)
        self.sum_vx = torch.zeros(b, device=device)
        self.sum_speed = torch.zeros(b, device=device)
        self.max_lateral = torch.zeros(b, device=device)
        self.max_abs_yaw = torch.zeros(b, device=device)
        self.max_body_x = torch.zeros(b, device=device)
        # Q003: signed net forward displacement in the case tile frame.
        self.episode_start_local_x = torch.zeros(b, device=device)
        self.forward_distance_buf = torch.zeros(b, device=device)
        self.forward_distance_end_buf = torch.zeros(b, device=device)
        self._last_flat_pass = torch.zeros(b, dtype=torch.bool, device=device)
        self.frozen_forward_distance = torch.full((b,), float('nan'), device=device)
        self.frozen_forward_start = torch.full((b,), float('nan'), device=device)
        self.frozen_forward_end = torch.full((b,), float('nan'), device=device)
        self.frozen_forward_steps = torch.zeros(b, dtype=torch.long, device=device)
        self.episode_reward_sums = {}
        self._last_reward_terms = {}
        self._last_step_diagnostics = {}
        self.reward_terms = {}
        self.step_reward_tensors = {}
        self.last_termination = None
        self.current_platform = torch.zeros(b, 4, dtype=torch.bool, device=device)
        self._validate_scene()
        # First full reset: applies the initial roles/tiles and spawn.
        self.set_schedule(self._default_schedule(), int(stage_iteration))
        self.reset()

    # ------------------------------------------------------------- identity
    def _validate_scene(self):
        if self.geom_tile.max().item() >= len(self.specs):
            raise RuntimeError('v07 scene holds geometry outside the v0.7 table')
        expected = sum(1 if int(s.N) == 0 else int(s.N) + 2 for s in self.specs)
        if len(self.geom_ids) != expected:
            raise RuntimeError('v07 scene geometry count does not match the table')

    def _default_schedule(self):
        roles = curriculum_module.allocate_roles(self.num_envs, self.phase)
        if self._tile_override is not None:
            frontier = int(self._tile_override)
            tiles = (frontier,) * self.num_envs
        else:
            frontier = int(curriculum_module.frontier_tile_for_state(
                self.phase, self.substage, self.level_tick))
            tiles = curriculum_module.role_tiles(roles, frontier, self._tile_sampler)
        return ScheduleAssignment(phase=self.phase, substage=self.substage,
                                  frontier_tile_id=frontier, roles=roles,
                                  tile_ids=tiles, generation=self.generation)

    # ------------------------------------------------------------- schedule
    def set_schedule(self, schedule: ScheduleAssignment,
                     stage_iteration: int = 0) -> None:
        """Apply a curriculum schedule at an update boundary (SDS 5.5/5.6)."""
        if not isinstance(schedule, ScheduleAssignment):
            raise TypeError('set_schedule needs the v0.7 ScheduleAssignment')
        phase = str(schedule.phase).upper()
        if phase not in PHASE_ORDER:
            raise ValueError(f'unknown schedule phase: {phase!r}')
        substage = _normalise_substage(phase, schedule.substage)
        roles = tuple(str(role) for role in schedule.roles)
        tiles = tuple(int(v) for v in schedule.tile_ids)
        if len(roles) != self.num_envs or len(tiles) != self.num_envs:
            raise ValueError('schedule roles/tiles must cover every environment')
        if any(role not in curriculum_module.ROLE_NAMES for role in roles):
            raise ValueError('schedule has an unknown role')
        if any(not 0 <= tile < len(self.specs) for tile in tiles):
            raise ValueError('schedule tile id outside the 23-tile table')
        _check_schedule_frontier(phase, substage, int(schedule.frontier_tile_id))
        if self._tile_override is not None:
            if int(schedule.frontier_tile_id) != int(self._tile_override):
                raise ValueError('schedule frontier tile conflicts with the explicit '
                                 'frontier_tile_id of this environment')
            tiles = (int(self._tile_override),) * self.num_envs
        old_roles = tuple(self.roles)
        old_tiles = tuple(int(v) for v in self.tile_ids.tolist())
        changed = [index for index in range(self.num_envs)
                   if roles[index] != old_roles[index]
                   or tiles[index] != old_tiles[index]
                   or phase != self.phase or substage != self.substage]
        self.phase = phase
        self.stage = phase
        self.substage = substage
        self.stage_iteration = int(stage_iteration)
        self.frontier_tile_id = int(schedule.frontier_tile_id)
        self.generation = int(getattr(schedule, 'generation', self.generation))
        self.roles = list(roles)
        self.tile_ids.copy_(torch.as_tensor(tiles, dtype=torch.long, device=self.device))
        if changed:
            ids = torch.as_tensor(changed, dtype=torch.long, device=self.device)
            self._reset_ids(ids, resample_tiles=False)

    @property
    def phase_spec(self):
        return spec.phase_spec(self.phase)

    # ------------------------------------------------------------ parameters
    def _parameters(self):
        return (self.tile_N[self.tile_ids], self.tile_h[self.tile_ids],
                self.tile_d[self.tile_ids])

    def _tile_parameters(self):
        return events.tile_parameters(self.tile_ids)

    # ----------------------------------------------------------------- reset
    def _rng_state_sha256(self):
        state = self.rng.get_state()
        payload = state.cpu().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def _spawn_ranges(self):
        """Per-phase spawn/command ranges (SDS 5.4)."""
        if self.phase == 'R':
            p = 1.0 if self.evaluation else min(
                max(self.stage_iteration, 0) / R_SPAWN_RAMP_UPDATES, 1.0)
            x = _lerp(SPAWN_X, R_SPAWN_X, p)
            y = _lerp(SPAWN_Y, R_SPAWN_Y, p)
            yaw = _lerp(SPAWN_YAW, R_SPAWN_YAW, p)
        else:
            x, y, yaw = SPAWN_X, SPAWN_Y, SPAWN_YAW
        if self.phase == 'R':
            low, high = COMMAND_FINAL_VX
        elif self.phase == 'V':
            p = 1.0 if self.evaluation else min(
                max(self.stage_iteration, 0) / V_RAMP_UPDATES, 1.0)
            low, high = _lerp(COMMAND_INITIAL_VX, COMMAND_FINAL_VX, p)
        else:
            low, high = COMMAND_INITIAL_VX
        return dict(x=x, y=y, yaw=yaw, vx=(low, high))

    def _sample_commands(self, count):
        if self.fixed_vx is not None:
            return torch.full((count,), float(self.fixed_vx), device=self.device)
        low, high = self._spawn_ranges()['vx']
        return low + (high - low) * torch.rand(count, device=self.device,
                                               generator=self.rng)

    def _sample_tiles(self, ids):
        ids = ids.to(torch.int64)
        if self._tile_override is not None:
            # Explicit frontier (evaluation): every environment keeps the same
            # single terrain, never a random near/easy resample.
            self.tile_ids[ids] = int(self._tile_override)
            return
        roles = [self.roles[int(index)] for index in ids.tolist()]
        tiles = curriculum_module.role_tiles(roles, self.frontier_tile_id,
                                             self._tile_sampler)
        self.tile_ids[ids] = torch.as_tensor(tiles, dtype=torch.long,
                                             device=self.device)

    def _reset_ids(self, ids, *, resample_tiles=True):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if ids.dtype == torch.bool:
            ids = torch.nonzero(ids, as_tuple=True)[0]
        ids = ids.to(torch.int64).flatten()
        if ids.numel() == 0:
            return
        if ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError('reset ids are outside the environment batch')
        ids, _ = torch.sort(ids)
        self.sim.reset(ids)
        self.scene.reset(ids)
        self.action_manager.reset(ids)
        if resample_tiles:
            self._sample_tiles(ids)
        robot = self.robot
        state = robot.data.default_root_state[ids].clone()
        count = len(ids)
        local = torch.zeros(count, 3, device=self.device)
        rng_sha = self._rng_state_sha256()
        ranges = self._spawn_ranges()
        vx = torch.zeros(count, device=self.device)
        sources = [SOURCE_SAMPLED] * count
        if self.initial_cases is not None:
            local[:, :2] = self.initial_cases[ids, :2]
            yaw = self.initial_cases[ids, 2].clone()
            sources = [SOURCE_CASES] * count
            if self.initial_command is not None:
                vx = self.initial_command[ids].clone()
        else:
            draw = torch.rand((count, 3), device=self.device, generator=self.rng)
            x_low = torch.full((count,), ranges['x'][0], device=self.device)
            x_high = torch.full((count,), ranges['x'][1], device=self.device)
            y_low = torch.full((count,), ranges['y'][0], device=self.device)
            y_high = torch.full((count,), ranges['y'][1], device=self.device)
            yaw_low = torch.full((count,), ranges['yaw'][0], device=self.device)
            yaw_high = torch.full((count,), ranges['yaw'][1], device=self.device)
            if self.phase == 'R':
                # SDS 5.4: the R flat environments keep the smaller spawn range.
                flat = torch.tensor([self.roles[int(index)] == 'flat'
                                     for index in ids.tolist()], dtype=torch.bool,
                                    device=self.device)
                if bool(flat.any()):
                    fill = [(x_low, SPAWN_X[0]), (x_high, SPAWN_X[1]),
                            (y_low, SPAWN_Y[0]), (y_high, SPAWN_Y[1]),
                            (yaw_low, SPAWN_YAW[0]), (yaw_high, SPAWN_YAW[1])]
                    for tensor, value in fill:
                        tensor[flat] = float(value)
            local[:, 0] = x_low + (x_high - x_low) * draw[:, 0]
            local[:, 1] = y_low + (y_high - y_low) * draw[:, 1]
            yaw = yaw_low + (yaw_high - yaw_low) * draw[:, 2]
        if self.fixed_vx is not None:
            vx = torch.full((count,), float(self.fixed_vx), device=self.device)
        elif self.initial_cases is None or self.initial_command is None:
            vx = self._sample_commands(count)
        local[:, 2] = state[:, 2]
        state[:, :3] = self.origins[self.tile_ids[ids]] + local
        state[:, 3:7] = 0
        state[:, 3] = torch.cos(yaw / 2)
        state[:, 6] = torch.sin(yaw / 2)
        state[:, 7:] = 0
        robot.write_root_state_to_sim(state, ids)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids],
                                       torch.zeros_like(robot.data.joint_vel[ids]),
                                       env_ids=ids)
        self.command[ids] = 0
        self.command[ids, 0] = vx
        self.start_vx[ids] = vx
        self.episode_command[ids] = vx
        self.episode_length_buf[ids] = 0
        self.stop[ids] = False
        self.stop_steps[ids] = 0
        self.contacts.reset(ids)
        self.climb.reset(ids)
        self.stall.reset(ids, local[:, 0])
        self.reward_computer.reset(ids)
        for tensor in (self.first_step_acquired, self.control_body_hit,
                       self.control_long_hit):
            tensor[ids] = False
        self.current_platform[ids] = False
        for tensor in (self._tracking_sum, self.sum_vx, self.sum_speed,
                       self.max_lateral, self.max_abs_yaw, self.max_body_x):
            tensor[ids] = 0.0
        self._tracking_count[ids] = 0
        for value in self.episode_reward_sums.values():
            value[ids] = 0
        self.episode_initial_pose[ids] = torch.stack((local[:, 0], local[:, 1], yaw), -1)
        self._local_x_prev[ids] = local[:, 0]
        self.max_body_x[ids] = local[:, 0]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.sense()
        # Q003: the start is the *actual* position after the spawn write,
        # sim.forward()/sense() and before any action -- never the command
        # constant and never a post-first-step position.
        start_local = (robot.data.root_link_pos_w[ids, 0]
                       - self.origins[self.tile_ids[ids], 0])
        self.episode_start_local_x[ids] = start_local
        self.forward_distance_buf[ids] = 0.0
        self.forward_distance_end_buf[ids] = start_local
        # The frozen terminal snapshot is intentionally NOT cleared here: a new
        # episode must never overwrite the case result already frozen for the
        # evaluator (Q003).  It is only replaced by the next terminal event.
        obs = self._proprio()
        self.history_cache.reset(obs, ids)
        self.map_cache.reset(robot.data.root_link_pos_w,
                             euler_heading(robot.data.root_link_quat_w)[2],
                             self.tile_ids, ids)
        for row, env_id in enumerate(ids.tolist()):
            self.episode_initial_source[env_id] = sources[row]
            self.episode_rng_sha[env_id] = rng_sha

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.manual_seed(int(seed))
            self._tile_sampler = np.random.default_rng(int(seed) + TERRAIN_RNG_OFFSET)
        self._reset_ids(torch.arange(self.num_envs, device=self.device))
        return self._observations(update=False), {}

    # ---------------------------------------------------------------- policy
    def route_error(self):
        """SDS 2.3 ``[clip(local_y/0.20,-4,4), sin(e_world), cos(e_world)]``."""
        data = self.robot.data
        local = data.root_link_pos_w - self.origins[self.tile_ids]
        _, _, yaw = euler_heading(data.root_link_quat_w)
        e_world = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        lateral = (local[:, 1] / ROUTE_LATERAL_SCALE).clamp(-ROUTE_LATERAL_CLIP,
                                                            ROUTE_LATERAL_CLIP)
        return torch.stack((lateral, torch.sin(e_world), torch.cos(e_world)), -1)

    def _foot_surface(self):
        """``(H(x_foot,y_foot), valid)`` from the exact v0.7 terrain query."""
        feet = self.robot.data.site_pos_w[:, self.foot_ids, :2]
        height, valid = terrain.query_surface(feet, self.tile_ids)
        return height, valid

    def _foot_heights(self):
        feet = self.robot.data.site_pos_w[:, self.foot_ids]
        bottom = feet[:, :, 2] - float(getattr(spec, 'FOOT_RADIUS', 0.032))
        height, valid = self._foot_surface()
        h = torch.where(valid, bottom - height, torch.full_like(bottom, 1.5))
        return h.clamp(-0.05, 1.5), ~valid

    def _proprio(self):
        data = self.robot.data
        return torch.cat((data.root_link_ang_vel_b, data.projected_gravity_b,
                          self.command,
                          data.joint_pos[:, self.joint_ids]
                          - data.default_joint_pos[:, self.joint_ids],
                          data.joint_vel[:, self.joint_ids],
                          self.action_manager.action), -1)

    def _observations(self, update=True):
        data = self.robot.data
        prop = self._proprio()
        if update:
            self.history_cache.update(prop)
            self.map_cache.update(data.root_link_pos_w,
                                  euler_heading(data.root_link_quat_w)[2],
                                  self.tile_ids)
        h, _ = self._foot_heights()
        force = quat_apply_inverse(
            data.root_link_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4),
            self.contacts.foot_force.reshape(-1, 3)).reshape(self.num_envs, 12)
        torque = self.sim.data.qfrc_actuator[:, self.robot.indexing.joint_v_adr][
            :, self.joint_ids]
        privileged = torch.cat((data.root_link_lin_vel_b,
                                (self.contacts.foot_force.norm(dim=-1) > 5).float(),
                                force, h, torque), -1)
        return {'proprio': prop.clone(),
                'history': self.history_cache.history.clone(),
                'heightmap': self.map_cache.map.clone(),
                'privileged': privileged,
                'route_error': self.route_error()}

    # ------------------------------------------------------------ task state
    def _task_state(self):
        data = self.robot.data
        self.sample_command = self.command.clone()
        local = data.root_link_pos_w - self.origins[self.tile_ids]
        feet_local = (data.site_pos_w[:, self.foot_ids]
                      - self.origins[self.tile_ids, None])
        roll, pitch, yaw = euler_heading(data.root_link_quat_w)
        nonfinite = ~torch.isfinite(torch.cat((
            data.root_link_pos_w, data.root_link_quat_w, data.joint_pos,
            data.joint_vel), -1)).all(-1)
        state = events.evaluate(
            episode_length=self.episode_length_buf,
            local=local, roll=roll, pitch=pitch, yaw=yaw, feet_local=feet_local,
            tile_ids=self.tile_ids,
            control_body_hit=self.control_body_hit,
            body_link_force=self.contacts.link_force,
            body_link_mask=self.contacts.body_link_mask,
            nonfoot_duration=self.contacts.nonfoot_duration,
            control_long_hit=self.control_long_hit,
            contact_overflow=self.contacts.overflow,
            nonfinite=nonfinite,
            top_forces=self.contacts.top_forces,
            geom_tile=self.geom_tile, geom_step=self.geom_step,
            geom_bounds=self.geom_step_bounds,
            support_state=self.climb,
            first_step_state=self.first_step_acquired,
            start_command=self.episode_command)
        self.current_platform = self.climb.current_support.clone()
        self.stop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return state

    # ---------------------------------------------------------------- reward
    def _reward_frame(self, state, previous_action):
        data = self.robot.data
        local = state.local
        base_height, base_valid = terrain.query_surface(
            data.root_link_pos_w[:, :2], self.tile_ids)
        base_height = torch.where(base_valid, base_height,
                                  torch.zeros_like(base_height))
        foot_height, foot_valid = self._foot_surface()
        foot_height = torch.where(foot_valid, foot_height,
                                  torch.zeros_like(foot_height))
        done = state.success | state.failure | state.timeout
        fields = dict(
            executed_command=self.sample_command[:, 0],
            local_x=local[:, 0],
            local_x_prev=self._local_x_prev.clone(),
            body_lin_vel_b=data.root_link_lin_vel_b,
            body_ang_vel_b=data.root_link_ang_vel_b,
            gravity_b=data.projected_gravity_b,
            base_height_above_surface=data.root_link_pos_w[:, 2] - base_height,
            q=data.joint_pos[:, self.joint_ids],
            qdot=data.joint_vel[:, self.joint_ids],
            q_target_executed=self.action_term._processed_actions,
            action=self.action_manager.action,
            previous_action=previous_action,
            foot_position_world=data.site_pos_w[:, self.foot_ids],
            foot_velocity_world=data.site_lin_vel_w[:, self.foot_ids],
            foot_force_world=self.contacts.foot_force,
            foot_surface_height=foot_height,
            contact_raw=self.contacts.raw,
            contact_filtered=self.contacts.filtered,
            tile_ids=self.tile_ids,
            local_y=local[:, 1],
            yaw_world=state.yaw,
            resets=done,
            success=state.success,
            failure=state.failure,
        )
        return reward_module.RewardFrame(**fields)

    def _reward(self, state, previous_action):
        frame = self._reward_frame(state, previous_action)
        result = self.reward_computer(frame)
        total = result.total
        tensors = {}
        for name, value in result.raw_terms.items():
            tensors['reward_raw/' + name] = value
        for name, value in result.weighted_terms.items():
            tensors['reward_weighted/' + name] = value
        tensors['reward_total'] = total
        for key, term in tensors.items():
            if key not in self.episode_reward_sums:
                self.episode_reward_sums[key] = torch.zeros_like(term)
            self.episode_reward_sums[key] += term
        self.step_reward_tensors = tensors
        self._last_reward_terms = dict(result.weighted_terms)
        self.reward_terms = {key: float(value.mean().detach().cpu())
                             for key, value in result.weighted_terms.items()}
        self.reward_terms['total'] = float(total.mean().detach().cpu())
        return total

    # ------------------------------------------------------------------ step
    def step(self, action):
        if not torch.isfinite(action).all():
            raise NumericalFailure('Nonfinite action', {
                'action': action.detach().cpu(),
                'qpos': self.sim.data.qpos.detach().cpu(),
                'qvel': self.sim.data.qvel.detach().cpu()})
        previous = self.action_manager.action.clone()
        self.action_manager.process_action(action)
        self.control_body_hit = torch.zeros(self.num_envs, dtype=torch.bool,
                                            device=self.device)
        self.control_long_hit = torch.zeros_like(self.control_body_hit)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            self.contacts.sample(self.tile_ids, physics_dt=self.physics_dt)
            self.control_body_hit |= (
                self.contacts.link_force[:, self.contacts.body_link_mask]
                .norm(dim=-1) > 0).any(-1)
            self.control_long_hit |= (
                self.contacts.nonfoot_duration >= 0.1 - 1e-6).any(-1)
        self.contacts.sample(self.tile_ids, control=True)
        self.sim.sense()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        state = self._task_state()
        # Q003/Q005: the end is the *observed* same-frame local x, never
        # reconstructed as start + distance; the distance is end - start.
        self.forward_distance_end_buf = state.local[:, 0].clone()
        self.forward_distance_buf = (self.forward_distance_end_buf
                                     - self.episode_start_local_x)
        self._last_flat_pass = self._flat_pass(state)
        self.stall.update(state.local[:, 0], self.sample_command[:, 0],
                          self.climb.climbed, self.first_step_acquired,
                          torch.ones(self.num_envs, dtype=torch.bool,
                                     device=self.device))
        reward = self._reward(state, previous)
        terminal = self._observations(update=True)
        data = self.robot.data
        local = state.local
        score = reward_module.tracking_lin_vel_rate(data.root_link_lin_vel_b,
                                                    self.sample_command[:, 0])
        self._tracking_sum += torch.where(torch.isfinite(score), score,
                                          torch.zeros_like(score))
        self._tracking_count += 1
        self.sum_vx += data.root_link_lin_vel_b[:, 0]
        self.sum_speed += data.root_link_lin_vel_b.norm(dim=-1)
        self.max_lateral = torch.maximum(self.max_lateral, local[:, 1].abs())
        self.max_abs_yaw = torch.maximum(self.max_abs_yaw, state.yaw.abs())
        self.max_body_x = torch.maximum(self.max_body_x, local[:, 0])
        done = state.success | state.failure | state.timeout
        if bool(done.any()):
            # Immutable per-environment terminal snapshot: frozen before any
            # automatic reset, so a new episode can never overwrite the case
            # result that the evaluator reads from this step's info.
            self.frozen_forward_end = torch.where(
                done, self.forward_distance_end_buf, self.frozen_forward_end)
            self.frozen_forward_start = torch.where(
                done, self.episode_start_local_x, self.frozen_forward_start)
            self.frozen_forward_distance = torch.where(
                done, self.frozen_forward_end - self.frozen_forward_start,
                self.frozen_forward_distance)
            self.frozen_forward_steps = torch.where(
                done, self.episode_length_buf, self.frozen_forward_steps)
        self.last_termination = (state.success.clone(), state.failure.clone(),
                                 state.timeout.clone(), state.reason.clone())
        self._last_step_diagnostics = self._diagnostics_batch(state)
        terminal_observation = {key: value.clone() for key, value in terminal.items()}
        episodes = self._episode_records(state, done)
        outcomes = self._outcomes(state, done)
        info = dict(
            terminal_observation=terminal_observation,
            episodes=episodes,
            outcomes=outcomes,
            failure=state.failure.detach().clone(),
            success=state.success.detach().clone(),
            climb=state.success.detach().clone(),
            timeout=state.timeout.detach().clone(),
            reason=state.reason.detach().clone(),
            failure_masks={name: mask.detach().clone()
                           for name, mask in state.failure_masks.items()},
            first_step_acquired=self.first_step_acquired.detach().clone(),
            # Q003: per-environment tensors, never an env-0 scalar masquerading
            # as the whole evaluation batch.  ``forward_distance`` is the signed
            # net displacement of this control sample; terminal cases also read
            # the frozen pre-reset value/start/end below.
            first_step_valid=self.first_step_acquired.detach().clone(),
            active_samples=self.stall.active_samples.detach().clone(),
            stall_samples=self.stall.stall_samples.detach().clone(),
            max_abs_y=self.max_lateral.detach().clone(),
            max_abs_yaw_deg=(self.max_abs_yaw * (180.0 / math.pi)).detach().clone(),
            forward_distance=(self.forward_distance_end_buf
                              - self.episode_start_local_x).detach().clone(),
            forward_distance_start=self.episode_start_local_x.detach().clone(),
            forward_distance_end=self.forward_distance_end_buf.detach().clone(),
            forward_distance_terminal=self.frozen_forward_distance.detach().clone(),
            forward_distance_terminal_start=self.frozen_forward_start.detach().clone(),
            forward_distance_terminal_end=self.frozen_forward_end.detach().clone(),
            max_forward_excursion=(self.max_body_x
                                   - self.episode_start_local_x).detach().clone(),
            flat_pass=self._last_flat_pass.detach().clone(),
            reset_mask=done.detach().clone(),
            reward_terms=dict(self.reward_terms),
            step_reward_tensors=self.step_reward_tensors,
            diagnostics=self._last_step_diagnostics,
            command_executed=self.sample_command.detach().clone(),
            tracking_score=(self._tracking_sum
                            / self._tracking_count.clamp_min(1).to(torch.float32)),
            protocol_id=self.protocol, code_version=spec.CODE_VERSION,
            recipe_version=spec.RECIPE_VERSION, reward_version=self.reward_version,
            phase=self.phase, substage=self.substage,
            generation=int(self.generation),
            frontier_tile_id=int(self.frontier_tile_id),
            level_tick=int(self.level_tick),
            roles=list(self.roles),
            route_error=terminal_observation['route_error'],
            route_error_terminal=terminal_observation['route_error'],
        )
        if bool((state.reason == 9).any() or not torch.isfinite(reward).all()):
            raise NumericalFailure('Numerical state/reward failure before reset', {
                'episodes': episodes,
                'terminal_observation': {key: value.detach().cpu()
                                         for key, value in terminal.items()},
                'reward': reward.detach().cpu(),
                'qpos': self.sim.data.qpos.detach().cpu(),
                'qvel': self.sim.data.qvel.detach().cpu(),
                'tile_ids': self.tile_ids.detach().cpu(),
                'phase': self.phase, 'substage': self.substage})
        self._local_x_prev.copy_(local[:, 0].detach())
        if bool(done.any()):
            self._reset_ids(done.nonzero().flatten())
        obs = self._observations(update=False)
        info['route_error_next'] = obs['route_error']
        return obs, reward, state.success | state.failure, state.timeout, info

    # --------------------------------------------------------------- records
    def _diagnostics_batch(self, state):
        data = self.robot.data
        local = state.local
        return dict(
            phase=self.phase, substage=self.substage,
            generation=int(self.generation),
            frontier_tile_id=int(self.frontier_tile_id),
            level_tick=int(self.level_tick),
            roles=list(self.roles),
            tile_ids=self.tile_ids.detach().clone(),
            command_c=self.sample_command[:, 0].detach().clone(),
            control_step=self.episode_length_buf.detach().clone(),
            local_position=local.detach().clone(),
            body_vx=data.root_link_lin_vel_b[:, 0].detach().clone(),
            heading=state.yaw.detach().clone(),
            success=state.success.detach().clone(),
            failure=state.failure.detach().clone(),
            timeout=state.timeout.detach().clone(),
            failure_masks={name: mask.detach().clone()
                           for name, mask in state.failure_masks.items()},
            first_step_acquired=self.first_step_acquired.detach().clone(),
            platform_support=self.current_platform.detach().clone(),
            contact_raw=self.contacts.raw.detach().clone(),
            contact_filtered=self.contacts.filtered.detach().clone(),
            foot_force_world_z=self.contacts.foot_force[..., 2].detach().clone(),
            reward_terms={key: value.detach().clone()
                          for key, value in self._last_reward_terms.items()},
            # Q001-7 per-env diagnostic counters (never a payment) and Q003
            # per-env net displacement, all as [B] tensors.  Per-case scalars
            # are available from ``diagnostics_cases()``/``diagnostics_snapshot``
            # and from the immutable terminal snapshot.
            active_samples=self.stall.active_samples.detach().clone(),
            stall_samples=self.stall.stall_samples.detach().clone(),
            first_step_valid=self.first_step_acquired.detach().clone(),
            active_samples_by_env=self.stall.active_samples.detach().clone(),
            stall_samples_by_env=self.stall.stall_samples.detach().clone(),
            first_step_valid_by_env=self.first_step_acquired.detach().clone(),
            stall_penalty_on=self.stall.penalty_on.detach().clone(),
            stall_dx_1s=self.stall.dx_1s.detach().clone(),
            stall_highwater_dx_1s=self.stall.highwater_dx_1s.detach().clone(),
            max_abs_y=self.max_lateral.detach().clone(),
            max_abs_yaw_deg=(self.max_abs_yaw * (180.0 / math.pi)).detach().clone(),
            forward_distance=(self.forward_distance_end_buf
                              - self.episode_start_local_x).detach().clone(),
            forward_distance_start=self.episode_start_local_x.detach().clone(),
            forward_distance_end=self.forward_distance_end_buf.detach().clone(),
            forward_distance_terminal=self.frozen_forward_distance.detach().clone(),
            max_forward_excursion=(self.max_body_x
                                   - self.episode_start_local_x).detach().clone(),
            flat_pass=self._last_flat_pass.detach().clone(),
        )

    def forward_distance(self):
        """Signed net displacement ``x_t - x_start`` of the current episode.

        Q003: this is the only meaning of ``forward_distance``; it can be
        negative, is never clamped/abs'd and uses the actual post-reset start.
        Terminal cases read :meth:`terminal_forward_distance` instead, which is
        frozen before the automatic reset.
        """
        return self.forward_distance_buf.clone()

    def terminal_forward_distance(self):
        """Frozen pre-reset ``D_net`` of the last terminal case (NaN if none)."""
        return self.frozen_forward_distance.clone()

    def max_forward_excursion(self):
        """Presentation-only ``max_body_x - x_start``; never used by a gate."""
        return self.max_body_x - self.episode_start_local_x

    def _flat_pass(self, state):
        """SDS 6.1 / Q003 flat gate verdict for the current (F0) episode."""
        required = (float(spec.F0_PROGRESS_FRACTION) * self.sample_command[:, 0]
                    * float(spec.F0_PROGRESS_SECONDS))
        return ((self.episode_length_buf == int(spec.F0_OBSERVE_STEPS))
                & ~state.failure
                & (self.max_lateral <= float(spec.F0_LATERAL_MAX) + 1e-12)
                & (self.max_abs_yaw <= float(spec.F0_HEADING_MAX_RADIANS) + 1e-12)
                & (self.forward_distance_buf >= required - 1e-9))

    def diagnostics_cases(self):
        """Per-case diagnostic list (Q001-7): counters for every environment."""
        count = self.episode_length_buf.clamp_min(1).to(torch.float32)
        return [dict(env_id=index, role=self.roles[index],
                     tile_id=int(self.tile_ids[index]),
                     frontier_tile_id=int(self.frontier_tile_id),
                     vx=float(self.episode_command[index]),
                     step=int(self.episode_length_buf[index]),
                     mean_vx=float(self.sum_vx[index] / count[index]),
                     mean_speed=float(self.sum_speed[index] / count[index]),
                     active_samples=int(self.stall.active_samples[index]),
                     stall_samples=int(self.stall.stall_samples[index]),
                     first_step_valid=bool(self.first_step_acquired[index]),
                     stall_penalty_on=bool(self.stall.penalty_on[index]),
                     max_abs_y=float(self.max_lateral[index]),
                     max_abs_yaw_deg=float(self.max_abs_yaw[index] * (180.0 / math.pi)),
                     forward_distance=float(self.forward_distance_end_buf[index]
                                            - self.episode_start_local_x[index]),
                     forward_distance_start=float(self.episode_start_local_x[index]),
                     forward_distance_end=float(self.forward_distance_end_buf[index]),
                     forward_distance_terminal=float(
                         self.frozen_forward_distance[index]),
                     max_forward_excursion=float(
                         self.max_body_x[index] - self.episode_start_local_x[index]),
                     flat_pass=bool(self._last_flat_pass[index]))
                for index in range(self.num_envs)]

    def _tracking_score(self):
        return self._tracking_sum / self._tracking_count.clamp_min(1).to(torch.float32)

    def _outcomes(self, state, done):
        if not bool(done.any()):
            return []
        scores = self._tracking_score()
        rows = []
        for index in done.nonzero().flatten().tolist():
            rows.append(dict(
                env_id=int(index),
                role=self.roles[index],
                tile_id=int(self.tile_ids[index]),
                generation=int(self.generation),
                success=bool(state.success[index]),
                failure=bool(state.failure[index]),
                timeout=bool(state.timeout[index]),
                reason=REASON_NAMES[int(state.reason[index])],
                tracking_score=float(scores[index]),
                command=float(self.episode_command[index]),
                step=int(self.common_step_counter),
                terminal_step=int(self.episode_length_buf[index]),
            ))
        return rows

    def _episode_records(self, state, done):
        if not bool(done.any()):
            return []
        data = self.robot.data
        count = self.episode_length_buf.clamp_min(1).to(torch.float32)
        scores = self._tracking_score()
        records = []
        flat_pass = self._last_flat_pass
        for index in done.nonzero().flatten().tolist():
            tile_id = int(self.tile_ids[index])
            tile = self.specs[tile_id]
            records.append(dict(
                env_id=int(index), seed=int(self.run_seed),
                phase=self.phase, substage=self.substage,
                generation=int(self.generation),
                frontier_tile_id=int(self.frontier_tile_id),
                role=self.roles[index], tile_id=tile_id,
                lesson_id=tile.lesson_id, N=int(tile.N), h=float(tile.h),
                step_depth=float(tile.step_depth),
                command=float(self.episode_command[index]),
                tracking_score=float(scores[index]),
                success=bool(state.success[index]),
                failure=bool(state.failure[index]),
                timeout=bool(state.timeout[index]),
                truncated=bool(state.timeout[index]),
                reason=REASON_NAMES[int(state.reason[index])],
                failure_masks={name: bool(mask[index])
                               for name, mask in state.failure_masks.items()},
                terminal_step=int(self.episode_length_buf[index]),
                # Global control step at termination (FIFO order in SDS 5.5).
                end_control_step=int(self.common_step_counter),
                duration=float(self.episode_length_buf[index]) * CONTROL_DT,
                control_samples=int(self.episode_length_buf[index]),
                initial_pose=[float(v) for v in self.episode_initial_pose[index]],
                initial_pose_source=self.episode_initial_source[index],
                rng_state_sha256_before_sample=self.episode_rng_sha[index],
                mean_vx=float(self.sum_vx[index] / count[index]),
                mean_speed=float(self.sum_speed[index] / count[index]),
                max_lateral=float(self.max_lateral[index]),
                max_abs_yaw=float(self.max_abs_yaw[index]),
                max_body_x=float(self.max_body_x[index]),
                first_step_acquired=bool(self.first_step_acquired[index]),
                first_step_valid=bool(self.first_step_acquired[index]),
                active_samples=int(self.stall.active_samples[index]),
                stall_samples=int(self.stall.stall_samples[index]),
                stall_penalty_on=bool(self.stall.penalty_on[index]),
                max_abs_y=float(self.max_lateral[index]),
                max_abs_yaw_deg=float(self.max_abs_yaw[index] * (180.0 / math.pi)),
                # Q003 signed net displacement, frozen pre-reset.
                forward_distance=float(self.forward_distance_end_buf[index]
                                       - self.episode_start_local_x[index]),
                forward_distance_start=float(self.episode_start_local_x[index]),
                forward_distance_end=float(self.forward_distance_end_buf[index]),
                forward_distance_terminal=float(
                    self.frozen_forward_distance[index]),
                max_forward_excursion=float(self.max_body_x[index]
                                            - self.episode_start_local_x[index]),
                flat_pass=bool(flat_pass[index]),
                platform_support=[bool(v) for v in self.current_platform[index]],
                route_final=[float(v) for v in self.terminal_route(index)],
                reward_sums={key: float(value[index])
                             for key, value in self.episode_reward_sums.items()},
                protocol_id=self.protocol, code_version=spec.CODE_VERSION,
                recipe=spec.RECIPE_VERSION, reward_version=self.reward_version,
                checkpoint=self.checkpoint_identity,
            ))
        return records

    def terminal_route(self, index):
        data = self.robot.data
        local = data.root_link_pos_w[index] - self.origins[self.tile_ids[index]]
        _, _, yaw = euler_heading(data.root_link_quat_w[index:index + 1])
        e_world = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        lateral = (local[1] / ROUTE_LATERAL_SCALE).clamp(-ROUTE_LATERAL_CLIP,
                                                         ROUTE_LATERAL_CLIP)
        return torch.stack((lateral, torch.sin(e_world[0]), torch.cos(e_world[0])))

    # ------------------------------------------------------------- interface
    @property
    def diagnostics(self):
        return self._last_step_diagnostics

    def last_reward_terms(self):
        return dict(self._last_reward_terms)

    def diagnostics_snapshot(self, env_id=0):
        i = int(env_id)
        data = self.robot.data
        local = data.root_link_pos_w[i] - self.origins[self.tile_ids[i]]
        return dict(
            env_id=i, phase=self.phase, substage=self.substage,
            generation=int(self.generation), tile_id=int(self.tile_ids[i]),
            frontier_tile_id=int(self.frontier_tile_id),
            level_tick=int(self.level_tick),
            role=self.roles[i], step=int(self.episode_length_buf[i]),
            position=local.detach().clone(),
            quaternion=data.root_link_quat_w[i].detach().clone(),
            joint_position=data.joint_pos[i, self.joint_ids].detach().clone(),
            joint_velocity=data.joint_vel[i, self.joint_ids].detach().clone(),
            action=self.action_manager.action[i].detach().clone(),
            command=self.command[i].detach().clone(),
            command_executed=self.sample_command[i].detach().clone(),
            foot_position=(data.site_pos_w[i, self.foot_ids]
                           - self.origins[self.tile_ids[i]]).detach().clone(),
            foot_force=self.contacts.foot_force[i].detach().clone(),
            route_error=self.route_error()[i].detach().clone(),
            active_samples=int(self.stall.active_samples[i]),
            stall_samples=int(self.stall.stall_samples[i]),
            first_step_valid=bool(self.first_step_acquired[i]),
            stall_penalty_on=bool(self.stall.penalty_on[i]),
            max_abs_y=float(self.max_lateral[i]),
            max_abs_yaw_deg=float(self.max_abs_yaw[i] * (180.0 / math.pi)),
            forward_distance=float(self.forward_distance_end_buf[i]
                                   - self.episode_start_local_x[i]),
            forward_distance_start=float(self.episode_start_local_x[i]),
            forward_distance_end=float(self.forward_distance_end_buf[i]),
            forward_distance_terminal=float(self.frozen_forward_distance[i]),
            max_forward_excursion=float(self.max_body_x[i]
                                        - self.episode_start_local_x[i]),
            reward_terms={key: float(value[i])
                          for key, value in self._last_reward_terms.items()},
        )

    def motion_terms(self):
        data = self.robot.data
        local = data.root_link_pos_w - self.origins[self.tile_ids]
        values = dict(
            command_vx=self.sample_command[:, 0],
            actual_vx_body=data.root_link_lin_vel_b[:, 0],
            actual_vx_world=data.root_link_lin_vel_w[:, 0],
            velocity_error_body=data.root_link_lin_vel_b[:, 0]
            - self.sample_command[:, 0],
            lateral=local[:, 1].abs(),
            tracking_score=self._tracking_score(),
        )
        return dict(sample_count=self.num_envs,
                    **{key: float(value.mean().detach().cpu())
                       for key, value in values.items()})

    # ------------------------------------------------------------- RNG I/O
    def rng_state(self):
        return dict(torch=self.rng.get_state().cpu())

    def load_rng_state(self, value):
        if not isinstance(value, dict) or 'torch' not in value:
            raise ValueError('environment RNG state must hold the torch generator')
        state = value['torch']
        self.rng.set_state(torch.as_tensor(state, dtype=torch.uint8).cpu())

    def sampler_rng_state(self):
        return dict(self._tile_sampler.bit_generator.state)

    def load_sampler_rng_state(self, value):
        if not isinstance(value, dict) or 'state' not in value:
            raise ValueError('sampler RNG state must be a numpy bit-generator state')
        self._tile_sampler.bit_generator.state = dict(value)


# ----------------------------------------------------------------- helpers
def flat_case_verdict(*, x_start, x_end, command, steps, max_abs_y=0.0,
                      max_abs_yaw_deg=0.0, failure=False):
    """Q003 F0 case verdict from the signed net displacement.

    ``D_net = x_end - x_start`` in the case tile frame; ``required = 0.5*c*10``.
    The peak/max forward excursion is deliberately *not* used.  Mirrors the
    vectorized :meth:`V07Env._flat_pass` and C's ``speed.flat_case_pass``.
    """
    required = (float(spec.F0_PROGRESS_FRACTION) * float(command)
                * float(spec.F0_PROGRESS_SECONDS))
    d_net = float(x_end) - float(x_start)
    passed = (int(steps) == int(spec.F0_OBSERVE_STEPS)
              and not bool(failure)
              and float(max_abs_y) <= float(spec.F0_LATERAL_MAX) + 1e-12
              and float(max_abs_yaw_deg) <= float(spec.F0_HEADING_MAX_DEGREES) + 1e-12
              and d_net >= required - 1e-9)
    return dict(passed=bool(passed), d_net=d_net, required=required,
                steps=int(steps), failure=bool(failure))


def _validate_level_tick(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('level_tick must be an integer in [0, 100]')
    if not 0 <= int(value) <= int(spec.LEVEL_TICK_MAX):
        raise ValueError('level_tick must be in [0, 100]')
    return int(value)


def _validate_frontier_tile(tile_id, *, phase, substage, level_tick, evaluation):
    """Resolve/validate the explicit single-terrain override (SDS 6.1)."""
    if tile_id is None:
        if evaluation and phase == 'H' and int(level_tick) == 0:
            raise ValueError(
                'evaluation on H must specify frontier_tile_id (or a non-zero '
                'level_tick); silently evaluating H00 is forbidden')
        return None
    if isinstance(tile_id, bool) or not isinstance(tile_id, int):
        raise TypeError('frontier_tile_id must be an integer or None')
    _check_schedule_frontier(phase, substage, int(tile_id))
    return int(tile_id)


def _check_schedule_frontier(phase, substage, tile_id):
    """Check that ``tile_id`` is a legal frontier for this phase/substage."""
    from . import spec as _spec
    phase = str(phase).upper()
    tile = int(tile_id)
    if phase == 'F0':
        if tile != FLAT_TILE_ID:
            raise ValueError('F0 frontier must be the flat tile')
        return
    if phase == 'H':
        if not 1 <= tile <= _spec.tile_id_for('H15'):
            raise ValueError('H frontier must be T0..T15 (tile 1..16)')
        return
    if phase in ('D', 'L'):
        expected = int(_spec.tile_id_for(substage))
        if tile != expected:
            raise ValueError(f'{phase}/{substage} frontier must be tile {expected}')
        return
    if tile != int(_spec.V_R_FRONTIER_TILE_ID):
        raise ValueError(f'{phase} frontier must be tile {int(_spec.V_R_FRONTIER_TILE_ID)}')


def _lerp(a, b, p):
    return (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)


def _normalise_substage(phase, substage):
    spec.phase_spec(phase)
    allowed = tuple(spec.phase_spec(phase).substages)
    if not allowed:
        if substage is not None:
            raise ValueError(f'{phase} has no substage, got {substage!r}')
        return None
    if substage is None:
        return allowed[0]
    if substage not in allowed:
        raise ValueError(f'unknown {phase} substage: {substage!r}')
    return substage


def _validate_fixed_vx(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError('fixed_vx must be a float or None')
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError('fixed_vx must be a finite value in [0, 1]')
    return value


def _case_tensor(initial_cases, num_envs, device):
    """Normalise evaluation cases to ``([num_envs,3] poses, [num_envs] speeds)``."""
    if initial_cases is None:
        return None, None
    if torch.is_tensor(initial_cases):
        poses = initial_cases.to(device=device, dtype=torch.float32)
        if poses.ndim != 2 or poses.shape[1] < 3:
            raise ValueError('initial_cases tensor must be [cases, >=3]')
        if poses.shape[1] >= 4:
            command = poses[:, 3].clone()
        else:
            command = None
        poses = poses[:, :3]
    else:
        rows = list(initial_cases)
        if not rows:
            raise ValueError('initial_cases must not be empty')
        poses = []
        command = []
        for row in rows:
            if isinstance(row, dict):
                pose = row.get('initial_pose', row.get('pose'))
                if pose is None:
                    pose = [row['x'], row['y'], row.get('yaw', 0.0)]
                poses.append([float(v) for v in pose[:3]])
                command.append(float(row.get('vx', row.get('command', float('nan')))))
            else:
                values = [float(v) for v in row]
                if len(values) < 3:
                    raise ValueError('initial case rows need at least x, y, yaw')
                poses.append(values[:3])
                command.append(values[3] if len(values) > 3 else float('nan'))
        poses = torch.tensor(poses, dtype=torch.float32, device=device)
        command_t = torch.tensor(command, dtype=torch.float32, device=device)
        command = command_t if torch.isfinite(command_t).all() else None
    if poses.shape[0] == 1 and num_envs > 1:
        poses = poses.expand(num_envs, -1).clone()
        if command is not None:
            command = command.expand(num_envs).clone()
    elif poses.shape[0] != num_envs:
        raise ValueError('initial_cases must hold 1 or num_envs entries')
    return poses, command
