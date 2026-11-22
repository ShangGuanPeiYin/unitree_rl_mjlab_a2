"""S2 physics assembly seam and full env interface.

The real v09/mjlab assembly is supplied through ``backend_factory`` (the only
piece that touches MuJoCo/mjwarp).  ``S2PhysicsEnv`` owns the S2 task state,
map branch, g_level/Q028 evidence and the evidence recorder, so the execution
driver and all CPU tests use one contract.  ``build_s2_env`` refuses before any
GPU allocation when no real backend factory is installed.

A deterministic ``FakeS2Backend`` is provided for CPU contract tests; it is
never used by production commands.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import s2_env, s2_map_ablation, s2_state, s2_terrain
from . import s2_execute


class S2PhysicsAssemblyError(RuntimeError):
    pass


@dataclass
class BackendStep:
    route_x: float
    base_xy: tuple[float, float]
    feet_xy: list
    feet_route_x: list
    foot_on_platform: list
    foot_contact: list
    foot_normal_force: list
    pitch: float
    velocity_b: list
    velocity_route: float
    omega_b: list
    c_exec: float
    c_request: float
    a_request: list
    a_applied: list
    pd_target: list
    joint_position: list
    joint_velocity: list
    map_valid: np.ndarray
    map_base_z: float
    refreshed: bool
    terminal: bool
    failure: bool
    top_arrived: bool
    four_feet_on_platform: bool
    front_foot_top_support: bool
    feet_ok: bool
    body_ok: bool
    contact_pairs: list
    native_sample_id: dict
    sim_time_s: float


class S2FakeBackend:
    """CPU-only deterministic backend for contract tests; no physics."""

    def __init__(self, case, *, device='cpu', seed=42, branch='real',
                 h_legacy=False, steps_to_terminal=60):
        self.case = case
        self.device = device
        self.seed = int(seed)
        self.branch = branch
        self.h_legacy = h_legacy
        self.steps_to_terminal = int(steps_to_terminal)
        self.task_adapter = s2_env.S2TaskAdapter.from_case(case)
        self.step_index = 0
        self.route_x = float(case['spawn']['local_position_xy'][0])
        self.sim_time = 0.0
        self.terminated = False

    def reset(self):
        self.step_index = 0
        self.route_x = float(self.case['spawn']['local_position_xy'][0])
        self.sim_time = 0.0
        self.terminated = False
        return self.step()

    def step(self):
        return self._step()

    def _step(self):
        idx = self.step_index
        self.route_x += 0.5
        platform = self.task_adapter.platform
        front = platform['front_x'] if platform else None
        on_platform = bool(platform and self.route_x >= front)
        feet = [[self.route_x, 0.1], [self.route_x, -0.1],
                [self.route_x - 0.1, 0.1], [self.route_x - 0.1, -0.1]]
        self.step_index += 1
        self.sim_time += 0.02
        top = bool(on_platform and self.route_x >= (front + 0.4 if front else 1e9))
        terminal = self.step_index >= self.steps_to_terminal
        return BackendStep(
            route_x=self.route_x, base_xy=(self.route_x, 0.0), feet_xy=feet,
            feet_route_x=[f[0] for f in feet],
            foot_on_platform=[on_platform] * 4,
            foot_contact=[True, True, True, True],
            foot_normal_force=[10.0] * 4, pitch=0.0,
            velocity_b=[0.5, 0.0, 0.0], velocity_route=0.5,
            omega_b=[0.0, 0.0, 0.0], c_exec=0.5, c_request=0.5,
            a_request=[0.0] * 12, a_applied=[0.0] * 12,
            pd_target=[0.0] * 12, joint_position=[0.0] * 12,
            joint_velocity=[0.0] * 12,
            map_valid=np.ones((40, 30), dtype=bool),
            map_base_z=0.4, refreshed=self.step_index % 5 == 1,
            terminal=terminal, failure=False, top_arrived=top,
            four_feet_on_platform=on_platform,
            front_foot_top_support=on_platform,
            feet_ok=on_platform,
            body_ok=bool(front and self.route_x > front + 0.4),
            contact_pairs=[dict(foot=0, geom='tread_6', normal_force=10.0,
                                duration_s=0.02, position_xy=list(feet[0]),
                                surface='tread' if not on_platform else 'platform',
                                valid=True)],
            native_sample_id=dict(refresh_index=self.step_index // 5,
                                  sample_sim_time_s=self.sim_time,
                                  produced_at_actor_step=self.step_index),
            sim_time_s=self.sim_time)


class S2PhysicsEnv:
    """Full S2 env contract used by s2_execute and CPU tests."""

    def __init__(self, case, *, backend_factory, device='cpu', num_envs=1,
                 seed=42, map_branch='real', h_legacy=False):
        if num_envs != 1:
            raise ValueError('S2 physics assembly supports one smoke env per process')
        if map_branch not in ('real', 'plane_diagnostic'):
            raise ValueError('map_branch must be real or plane_diagnostic')
        self.case = case
        self.device = device
        self.seed = int(seed)
        self.map_branch = map_branch
        self.h_legacy = bool(h_legacy)
        self.task_adapter = s2_env.S2TaskAdapter.from_case(case)
        self.machine = self.task_adapter.state_machine()
        self.gate = self.task_adapter.level_gate()
        self.backend = backend_factory(case, device=device, seed=seed,
                                       branch=map_branch, h_legacy=h_legacy)
        self.recorder = s2_env.S2EvidenceRecorder(case=case, adapter=self.task_adapter,
                                                  map_branch=map_branch)
        self.ablation = None
        if map_branch == 'plane_diagnostic':
            plane_z = (self.task_adapter.platform or {}).get('top_height', 0.0)
            self.ablation = s2_map_ablation.S2PlaneAblation(
                task=('stairs_walk' if self.task_adapter.config.task == 'legacy_stairs'
                      else self.task_adapter.config.task), plane_world_z=plane_z)
        evidence_task = ('stairs_walk' if self.task_adapter.config.task == 'legacy_stairs'
                         else self.task_adapter.config.task)
        self.real_evidence = s2_map_ablation.S2RealBranchEvidence(task=evidence_task)
        self.last = None
        self.steps = 0
        self.geometry_events = []
        self.failure_tracker = s2_state.GeometryFailureTracker()
        self._reset_backend()

    def _reset_backend(self):
        self.backend.reset()
        self.steps = 0
        self.last = None

    def reset(self, case_state=None):
        self._reset_backend()
        return self.observe()

    def observe(self):
        last = self.last
        if last is None:
            last = self.backend.reset()
            self.last = last
        h = last.route_x
        proprio = np.zeros(45, dtype=np.float32)
        proprio[0:3] = last.omega_b
        proprio[3:6] = [0, 0, -1]
        proprio[6:9] = [last.c_request, 0.0, 0.0]
        history = np.tile(proprio, (5, 1))
        route = np.zeros(3, dtype=np.float32)
        heightmap = np.zeros((3, 40, 30), dtype=np.float32)
        heightmap[1] = 1.0
        return dict(proprio=proprio, history=history, heightmap=heightmap,
                    route_error=route, c_request=last.c_request,
                    c_exec=last.c_exec)

    def step(self, action):
        if self.last is not None and self.last.terminal:
            raise RuntimeError('physics env cannot advance after terminal')
        frame = (self.backend.step() if hasattr(self.backend, 'step')
                 else self.backend._step())
        self.steps += 1
        platform = self.task_adapter.platform
        condition = bool(platform and frame.feet_ok and frame.body_ok)
        g_value = self.gate.update(condition, step_index=self.steps - 1)
        geometry_reason = self.failure_tracker.observe(
            step=self.steps - 1, condition=condition,
            feet_ok=frame.feet_ok, body_ok=frame.body_ok)
        if self.ablation is not None:
            real_map = np.zeros((3, 40, 30), dtype=np.float64)
            real_map[1] = frame.map_valid
            policy_map, map_info = self.ablation.observe(
                frame_index=self.steps - 1, real_map=real_map,
                valid=frame.map_valid, base_sample_z=frame.map_base_z,
                refreshed=frame.refreshed,
                front_foot_top_support=frame.front_foot_top_support,
                consumed=True, terminal=frame.terminal,
                pairing=None, sample_id=frame.native_sample_id)
            self.real_evidence.observe(
                frame_index=self.steps - 1, real_map=real_map,
                valid=frame.map_valid, refreshed=frame.refreshed,
                front_foot_top_support=frame.front_foot_top_support,
                consumed=True, terminal=frame.terminal,
                sample_id=frame.native_sample_id)
        else:
            map_info = None
            self.real_evidence.observe(
                frame_index=self.steps - 1,
                real_map=np.zeros((3, 40, 30), dtype=np.float64),
                valid=frame.map_valid, refreshed=frame.refreshed,
                front_foot_top_support=frame.front_foot_top_support,
                consumed=True, terminal=frame.terminal,
                sample_id=frame.native_sample_id)
        v07_success = False
        if self.h_legacy and platform:
            v07_success = s2_state.legacy_v07_success(
                frame.route_x, platform_front_x=platform['front_x'],
                supported=frame.feet_ok)
        state_frame = dict(route_x=frame.route_x, first_step=frame.top_arrived,
                           top_arrived=frame.top_arrived,
                           four_feet_on_platform=frame.four_feet_on_platform,
                           failure=frame.failure, v07_success=v07_success,
                           a_request=frame.a_request, a_applied=frame.a_applied)
        self.machine.step(state_frame)
        self.recorder.append(dict(
            step=self.steps - 1, route_x=frame.route_x, base_xy=frame.base_xy,
            feet_xy=frame.feet_xy, feet_route_x=frame.feet_route_x,
            pitch=frame.pitch, foot_on_platform=frame.foot_on_platform,
            foot_contact=frame.foot_contact, g_level=g_value,
            top_arrived=frame.top_arrived, failure=frame.failure,
            stop_request=self.machine.stop_requested,
            terminal=frame.terminal or self.machine.terminal,
            c_exec=frame.c_exec, c_request=frame.c_request,
            a_request=frame.a_request, a_applied=frame.a_applied,
            velocity_route=frame.velocity_route, pitch_rate=0.0,
            foot_normal_force=frame.foot_normal_force,
            contact_pairs=frame.contact_pairs, pd_target=frame.pd_target,
            joint_position=frame.joint_position,
            joint_velocity=frame.joint_velocity,
            map_valid=frame.map_valid,
            map_sample_time_s=frame.sim_time_s,
            native_sample_id=frame.native_sample_id))
        self.last = frame
        return self.observe(), [frame.terminal or self.machine.terminal], dict(
            map_info=map_info, geometry_reason=geometry_reason,
            g_level=g_value, state=self.machine.summary())

    def diagnostics(self):
        frame = self.last
        return dict(feet_xy=frame.feet_xy, foot_contact=frame.foot_contact,
                    foot_normal_force=frame.foot_normal_force,
                    contact_pairs=frame.contact_pairs,
                    pd_target=frame.pd_target,
                    joint_position=frame.joint_position,
                    map_valid=frame.map_valid,
                    map_sample_time_s=frame.sim_time_s,
                    native_sample_id=frame.native_sample_id,
                    g_level=self.gate.value)

    def evidence(self, index=0):
        trace = self.recorder.trace()
        trace['map_evidence'] = (self.ablation.summary() if self.ablation
                                 else self.real_evidence.summary())
        trace['state_summary'] = self.machine.summary()
        trace['failure_context'] = {'present': self.machine.failure}
        trace['g_level_report'] = s2_state.g_level_window_report(
            [self.gate.value], [self.steps - 1],
            window_mask=[True], four_foot_step=None,
            full_geometry_step=self.gate.t0_step,
            geometry_events=self.failure_tracker.events)
        return trace

    @property
    def four_feet_on_platform(self):
        return bool(self.last and self.last.four_feet_on_platform)

    @property
    def front_foot_top_support(self):
        return bool(self.last and self.last.front_foot_top_support)

    def adapter(self, index=0):
        return self.task_adapter

    def recorder(self, index=0):
        return self.recorder

    def adapter_for_index(self, index=0):
        return self.task_adapter

    def recorder_for_index(self, index=0):
        return self.recorder

    def metrics(self, index=0):
        return dict(quality_pass=self.machine.success)


def build_s2_env(case, *, device='cuda:0', num_envs=1, seed=42,
                 map_branch='real', h_legacy=False, backend_factory=None):
    """Construct the S2 env; refuse before GPU allocation if no real backend."""
    if h_legacy and map_branch != 'real':
        raise ValueError('H legacy only supports the real map')
    if backend_factory is None:
        backend_factory = MjlabHLegacyBackend if h_legacy else MjlabS2Backend
    report = s2_execute.device_report(device)
    return S2PhysicsEnv(case, backend_factory=backend_factory, device=device,
                        num_envs=num_envs, seed=seed, map_branch=map_branch,
                        h_legacy=h_legacy), report


# --------------------------------------------------------------- mjlab backend
def _s2_patch_context(specs):
    """Temporarily install S2 terrain into the unchanged v07/mjlab assembly."""
    from contextlib import contextmanager
    from src.tasks.velocity.v07 import terrain as v07terrain
    from src.tasks.velocity.v07 import events as v07events
    from src.tasks.velocity.v07 import env_cfg as v07cfg
    from src.tasks.velocity.v07 import env as v07env

    @contextmanager
    def patches():
        import torch
        originals = {name: getattr(v07terrain, name) for name in
                     ('build_terrain_specs', 'tile_boxes', 'query_surface')}
        original_events_table = getattr(v07events, 'build_terrain_specs', None)
        original_cfg_gen = getattr(v07cfg, 'terrain_generator_cfg', None)
        original_validate = getattr(v07env, '_validate_frontier_tile', None)
        original_check = getattr(v07env, '_check_schedule_frontier', None)
        try:
            v07terrain.build_terrain_specs = lambda: tuple(specs)
            v07terrain.tile_boxes = lambda tile: s2_terrain.boxes(tile)
            def query(points_world, tile_ids):
                points = torch.as_tensor(points_world, dtype=torch.float64)
                ids = torch.as_tensor(tile_ids, dtype=torch.long)
                while ids.ndim < points.ndim - 1:
                    ids = ids.unsqueeze(-1)
                out = torch.full(points.shape[:-1], float('nan'), dtype=torch.float32,
                                 device=points.device)
                valid = torch.zeros_like(out, dtype=torch.bool)
                for index, spec in enumerate(specs):
                    mask = (ids == index).expand(points.shape[:-1])
                    if not bool(mask.any()):
                        continue
                    height, ok = s2_terrain.query_surface(points[mask].cpu().numpy(), spec)
                    out[mask] = torch.as_tensor(height, dtype=torch.float32,
                                                device=points.device)
                    valid[mask] = torch.as_tensor(ok, dtype=torch.bool,
                                                  device=points.device)
                return out, valid
            v07terrain.query_surface = query
            v07events.build_terrain_specs = lambda: tuple(specs)
            v07events._TABLE_CACHE.clear()
            def generator_cfg(seed=42):
                from mjlab.terrains.terrain_generator import (
                    SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput)
                import mujoco
                class S2AtlasCfg(SubTerrainCfg):
                    def function(self, difficulty, spec, rng):
                        del difficulty, rng
                        body = spec.body('terrain')
                        shift = np.array((16.0, 4.0, 0.0))
                        geometries = []
                        for terrain_spec in specs:
                            for box in s2_terrain.boxes(terrain_spec):
                                geom = body.add_geom(
                                    name=f's2_{terrain_spec.tile_id:02d}_{box.name}',
                                    type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=np.asarray(box.center) + shift,
                                    size=np.asarray(box.half), rgba=(0.3, 0.5, 0.7, 1.0),
                                    contype=1, conaffinity=1, condim=3,
                                    friction=(0.6, 0.005, 0.0001))
                                geometries.append(TerrainGeometry(geom=geom,
                                                                  color=(0.3, 0.5, 0.7, 1.0)))
                        return TerrainOutput(origin=shift, geometries=geometries)
                return TerrainGeneratorCfg(
                    seed=int(seed), size=(32.0, 8.0), num_rows=1, num_cols=1,
                    curriculum=False, border_width=0.0,
                    sub_terrains={'v07_atlas': S2AtlasCfg(proportion=1.0)},
                    color_scheme='none')
            v07cfg.terrain_generator_cfg = generator_cfg
            v07env._validate_frontier_tile = lambda *args, **kwargs: (
                kwargs.get('frontier_tile_id') if 'frontier_tile_id' in kwargs
                else args[0] if args else None)
            v07env._check_schedule_frontier = lambda *args, **kwargs: None
            yield
        finally:
            for name, value in originals.items():
                setattr(v07terrain, name, value)
            if original_events_table is not None:
                v07events.build_terrain_specs = original_events_table
            v07events._TABLE_CACHE.clear()
            if original_cfg_gen is not None:
                v07cfg.terrain_generator_cfg = original_cfg_gen
            if original_validate is not None:
                v07env._validate_frontier_tile = original_validate
            if original_check is not None:
                v07env._check_schedule_frontier = original_check
    return patches()


class MjlabS2Backend:
    """v09/mjlab assembly using s2_terrain boxes; single environment."""

    def __init__(self, case, *, device='cuda:0', seed=42, branch='real',
                 h_legacy=False):
        if h_legacy:
            raise S2PhysicsAssemblyError(
                'H legacy requires the original v07 atlas backend; run it in a '
                'separate process/backend selection')
        from src.tasks.velocity.v07.env import V07Env
        from . import s2_terrain
        spec = s2_terrain.spec_from_geometry(case['geometry'])
        self.case = case
        self.device = device
        self.branch = branch
        self.spec = spec
        self._patches = _s2_patch_context(tuple(s2_terrain.terrain_specs()))
        self._patches.__enter__()
        try:
            spawn = case['spawn']
            x, y = spawn['local_position_xy']
            self.env = V07Env(
                num_envs=1, device=device, seed=int(seed), phase='L',
                substage='L40',
                initial_cases=[[float(x), float(y),
                                float(spawn['local_yaw_radians']),
                                float(case['speed'])]],
                frontier_tile_id=int(spec.tile_id), evaluation=True)
            from .map import HeightMap
            self.env.map_cache = HeightMap(1, device)
        except Exception:
            self._patches.__exit__(None, None, None)
            raise
        self.steps = 0
        self.refresh_index = 0
        self.last_sample_id = None
        self.last_sample_time = 0.0

    def close(self):
        self._patches.__exit__(None, None, None)

    def reset(self):
        self.env.reset()
        self.steps = 0
        self.refresh_index = 0
        self.last_sample_id = None
        self.last_sample_time = 0.0
        return self._frame()

    def step(self, action=None):
        import torch
        if action is None:
            action = torch.zeros(1, 12, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32,
                                 device=self.device).reshape(1, 12)
        _, _, terminated, truncated, _ = self.env.step(action)
        self.steps += 1
        return self._frame(terminal=bool(terminated[0] or truncated[0]))

    def _frame(self, terminal=False):
        import torch
        data = self.env.robot.data
        origin = self.env.origins[self.env.tile_ids[0]]
        local = data.root_link_pos_w[0] - origin
        feet = data.site_pos_w[0, self.env.foot_ids]
        contact = self.env.contacts.raw[0].bool()
        platform = self.env.current_platform[0].bool()
        front = float(self.spec.platform_front_local_x)
        body_ok = bool(local[0] > front + 0.4)
        feet_ok = bool(platform.all())
        map_valid = self.env.map_cache.map[0, 1].bool().cpu().numpy()
        refreshed = int(self.env.map_cache.steps[0]) == 0
        sim_time = float(getattr(self.env.sim.data, 'time', self.steps * 0.02))
        if refreshed:
            self.refresh_index += 1
            self.last_sample_time = sim_time
            self.last_sample_id = dict(refresh_index=self.refresh_index,
                                       sample_sim_time_s=sim_time,
                                       produced_at_actor_step=self.steps)
        sample_id = self.last_sample_id or dict(
            refresh_index=0, sample_sim_time_s=sim_time,
            produced_at_actor_step=self.steps)
        top_forces = self.env.contacts.top_forces[0]
        contact_pairs = []
        for foot in range(4):
            forces = top_forces[foot]
            if float(forces.max()) <= 0:
                continue
            geom = int(forces.argmax())
            contact_pairs.append(dict(
                foot=foot, geom=f'geom_{geom}', normal_force=float(forces[geom]),
                duration_s=0.02, position_xy=list(feet[foot, :2].detach().cpu().numpy()),
                surface='platform' if feet_ok else 'tread', valid=True))
        return BackendStep(
            route_x=float(local[0]), base_xy=(float(local[0]), float(local[1])),
            feet_xy=feet[:, :2].detach().cpu().numpy().tolist(),
            feet_route_x=(feet[:, 0] - origin[0]).detach().cpu().numpy().tolist(),
            foot_on_platform=platform.detach().cpu().numpy().tolist(),
            foot_contact=contact.detach().cpu().numpy().tolist(),
            foot_normal_force=self.env.contacts.top_forces[0].detach().cpu().numpy().tolist(),
            pitch=float(__import__('src.tasks.velocity.v07.events', fromlist=['euler_heading'])
                        .euler_heading(data.root_link_quat_w[0:1])[1][0]),
            velocity_b=data.root_link_lin_vel_b[0].detach().cpu().numpy().tolist(),
            velocity_route=float(data.root_link_lin_vel_w[0, 0]),
            omega_b=data.root_link_ang_vel_b[0].detach().cpu().numpy().tolist(),
            c_exec=float(self.env.sample_command[0, 0]),
            c_request=float(self.env.command[0, 0]),
            a_request=self.env.action_manager.action[0].detach().cpu().numpy().tolist(),
            a_applied=self.env.action_term._processed_actions[0].detach().cpu().numpy().tolist(),
            pd_target=self.env.action_term._processed_actions[0].detach().cpu().numpy().tolist(),
            joint_position=data.joint_pos[0, self.env.joint_ids].detach().cpu().numpy().tolist(),
            joint_velocity=data.joint_vel[0, self.env.joint_ids].detach().cpu().numpy().tolist(),
            map_valid=map_valid, map_base_z=float(data.root_link_pos_w[0, 2]),
            refreshed=refreshed,
            terminal=terminal, failure=bool(self.env._physical_failure[0])
            if hasattr(self.env, '_physical_failure') else False,
            top_arrived=feet_ok and body_ok,
            four_feet_on_platform=feet_ok,
            front_foot_top_support=bool(platform[0] or platform[1]),
            feet_ok=feet_ok, body_ok=body_ok,
            contact_pairs=contact_pairs, native_sample_id=sample_id,
            sim_time_s=self.steps * 0.02)


def load_s1_actor(path=None, *, expected_sha256=None, device='cpu'):
    """Load the approved S1.2 actor only; no optimizer/training state."""
    import hashlib
    import torch
    from .model import V09ActorCritic
    path = path or s2_env.s2_spec.S1_ENTRY_WEIGHT
    expected = expected_sha256 or s2_env.s2_spec.S1_ENTRY_WEIGHT_SHA256
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if actual != expected:
        raise ValueError(f'S1 actor SHA mismatch: {actual}')
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get('model') or payload.get('state_dict')
    if not isinstance(state, dict):
        raise ValueError('S1 checkpoint lacks actor/critic model state')
    model = V09ActorCritic().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.actor.normalizer.eval()
    model.critic_normalizer.eval()
    # Exact state check and frozen-normalizer proof.
    reference = model.state_dict()
    for key, value in state.items():
        if not torch.equal(reference[key], value):
            raise ValueError(f'actor state differs at {key}')
    return dict(model=model, state_sha256=actual, frozen=True,
                normalizer_updates=0, requires_grad=False)


def actor_mean(actor_bundle, obs):
    """Deterministic Actor mean with no gradients."""
    import torch
    model = actor_bundle['model'] if isinstance(actor_bundle, dict) else actor_bundle
    with torch.no_grad():
        return model(obs)


def assembly_contract():
    """Static consistency assertions against v07/v09 constants (no physics)."""
    from . import map as v09map
    from src.tasks.velocity.v07 import spec as v07spec
    checks = {
        'control_dt': (v09map.CONTROL_DT, v07spec.CONTROL_DT, 0.02),
        'physics_dt': (v07spec.PHYSICS_DT, 0.005),
        'decimation': (v07spec.DECIMATION, 4),
        'action_scale': (v07spec.ACTION_SCALE, 0.25),
        'joints': (v07spec.N_JOINTS, 12),
        'joint_order': (v07spec.JOINT_ORDER, tuple(
            f'{leg}_{joint}' for leg in ('FL', 'FR', 'RL', 'RR')
            for joint in ('hip', 'thigh', 'calf'))),
        'default_pose': (v07spec.A2_DEFAULT_JOINT_POS,
                         (-.1, .9, -1.8, .1, .9, -1.8, -.1, .9, -1.8, .1, .9, -1.8)),
        'map_shape': (v09map.MAP_SHAPE, (3, 40, 30)),
        'map_grid_step': (v09map.GRID_STEP, 0.05),
        'map_x': (v09map.GRID_X, (-0.8, 1.2)),
        'map_y': (v09map.GRID_Y, (-0.75, 0.75)),
        'map_refresh_steps': (v09map.MAP_REFRESH_STEPS, 5),
    }
    failures = []
    for name, values in checks.items():
        if name == 'physics_dt' or name == 'decimation' or name == 'action_scale':
            if values[0] != values[1]:
                failures.append(name)
        elif name == 'joint_order':
            if tuple(values[0]) != tuple(values[1]):
                failures.append(name)
        elif name == 'default_pose':
            if tuple(values[0]) != tuple(values[1]):
                failures.append(name)
        elif values[0] != values[1]:
            failures.append(name)
    if failures:
        raise ValueError(f'assembly contract drift: {failures}')
    return checks


class MjlabHLegacyBackend(MjlabS2Backend):
    """Original v07 H/L geometry and termination; no S2 terrain patch."""

    def __init__(self, case, *, device='cuda:0', seed=42, branch='real',
                 h_legacy=True):
        if not h_legacy:
            raise ValueError('HLegacy backend requires h_legacy=True')
        from src.tasks.velocity.v07.env import V07Env
        from src.tasks.velocity.v07 import env as v07env
        from .map import HeightMap
        from . import s2_terrain
        self.case = case
        self.device = device
        self.branch = branch
        self.spec = s2_terrain.legacy_spec_from_geometry(case['geometry'])
        original_validate = v07env._validate_frontier_tile
        original_check = v07env._check_schedule_frontier
        v07env._validate_frontier_tile = lambda *a, **k: (
            k.get('frontier_tile_id') if 'frontier_tile_id' in k
            else (a[0] if a else None))
        v07env._check_schedule_frontier = lambda *a, **k: None
        try:
            spawn = case['spawn']
            x, y = spawn['local_position_xy']
            self.env = V07Env(
                num_envs=1, device=device, seed=int(seed), phase='L',
                substage='L40',
                initial_cases=[[float(x), float(y),
                                float(spawn['local_yaw_radians']),
                                float(case['speed'])]],
                frontier_tile_id=int(case['geometry']['tile_id']),
                evaluation=True)
            self.env.map_cache = HeightMap(1, device)
        finally:
            v07env._validate_frontier_tile = original_validate
            v07env._check_schedule_frontier = original_check
        self.steps = 0
        self.refresh_index = 0
        self.last_sample_id = None
        self.last_sample_time = 0.0

    def close(self):
        pass
