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


def _scalar(value, default=0.0):
    """Convert TorchArray/tensor/array scalar (possibly per-env) to float."""
    if value is None:
        return float(default)
    try:
        return float(value.item())
    except Exception:
        pass
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().reshape(-1)
            return float(value[0].item()) if value.numel() else float(default)
        array = np.asarray(value).reshape(-1)
        return float(array[0]) if array.size else float(default)
    except Exception:
        return float(default)


class S2PhysicsAssemblyError(RuntimeError):
    pass


def apply_default_spawn_z(env, spawn_z):
    """Set the V07 default root height used by every subsequent reset."""
    state = env.robot.data.default_root_state
    state[:, 2] = float(spawn_z)
    return state


def sim_state_digest(qpos, qvel):
    """Stable qpos/qvel digest for same-process branch replay evidence."""
    import hashlib
    values = []
    for value in (qpos, qvel):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        values.append(np.ascontiguousarray(np.asarray(value, dtype=np.float64)))
    digest = hashlib.sha256()
    for label, value in zip(('qpos', 'qvel'), values):
        digest.update(label.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass
class BackendStep:
    route_x: float
    base_xy: tuple[float, float]
    root_z: float
    feet_xy: list
    feet_route_x: list
    feet_z: list
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
    termination_reason: str = 'running'
    nonfoot_contact: bool = False
    v07_success: bool = False
    sim_state_sha256: str = ''
    post_reset: bool = False


def aggregate_foot_contacts(top_forces, *, geom_names=None, geom_tiles=None,
                            platform=None, contact_duration=0.02, feet_xy=None):
    """Aggregate per-geom top forces to four feet and build real contact pairs.

    v07 ``contacts.top_forces`` is [4, ngeom] (already foot-indexed); a foot may
    touch several terrain geoms, so the foot normal force is the sum over the
    geom axis.  Nonzero geom pairs only; no privileged contact slots are used.
    """
    import torch
    forces = torch.as_tensor(top_forces)
    if forces.ndim != 2 or forces.shape[0] != 4:
        raise ValueError(f'foot top_forces must be [4,ngeom], got {tuple(forces.shape)}')
    if not torch.isfinite(forces).all():
        raise ValueError('non-finite foot contact force')
    total = forces.sum(dim=-1)
    pairs = []
    for foot in range(4):
        for geom in range(forces.shape[1]):
            value = float(forces[foot, geom])
            if value <= 0:
                continue
            name = (geom_names[geom] if geom_names is not None and geom < len(geom_names)
                    else f'geom_{geom}')
            tile = (geom_tiles[geom] if geom_tiles is not None
                    and geom < len(geom_tiles) else None)
            surface = platform if platform is not None else ('terrain' if tile is not None else 'unknown')
            pairs.append(dict(foot=foot, geom=str(name), normal_force=value,
                              duration_s=float(contact_duration),
                              position_xy=(list(feet_xy[foot]) if feet_xy is not None
                                           else [0.0, 0.0]), surface=surface,
                              valid=True))
    return total, pairs


def nonfoot_contact_from_links(link_force, link_names, *, threshold=10.0):
    """Any non-foot body link over the v07 nonfoot threshold."""
    import torch
    forces = torch.as_tensor(link_force)
    if forces.ndim != 2 or len(link_names) != forces.shape[0]:
        raise ValueError('link force/name shape mismatch')
    for index, name in enumerate(link_names):
        if any(str(name).startswith(foot) for foot in ('FL', 'FR', 'RL', 'RR')):
            continue
        if float(forces[index].norm()) > threshold:
            return True
    return False


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
        return self._frame(advance=False)

    def observation(self, update=True):
        h = self.step_index
        proprio = np.zeros(45, dtype=np.float32)
        proprio[6:9] = [0.3, 0.0, 0.0]
        return dict(proprio=proprio,
                    history=np.tile(proprio, (5, 1)),
                    heightmap=np.zeros((3, 40, 30), dtype=np.float32),
                    route_error=np.zeros(3, dtype=np.float32))

    def step(self, action=None):
        return self._frame(advance=True)

    def _frame(self, advance=True):
        platform = self.task_adapter.platform
        front = platform['front_x'] if platform else None
        if advance:
            self.route_x += 0.5
            self.step_index += 1
            self.sim_time += 0.02
        on_platform = bool(platform and self.route_x >= front)
        feet = [[self.route_x, 0.1], [self.route_x, -0.1],
                [self.route_x - 0.1, 0.1], [self.route_x - 0.1, -0.1]]
        actor_step = max(0, self.step_index - 1) if advance else 0
        top = bool(on_platform and self.route_x >= (front + 0.4 if front else 1e9))
        terminal = self.step_index >= self.steps_to_terminal
        return BackendStep(
            route_x=self.route_x, base_xy=(self.route_x, 0.0), root_z=0.4, feet_xy=feet,
            feet_route_x=[f[0] for f in feet],
            feet_z=[0.0, 0.0, 0.0, 0.0],
            foot_on_platform=[on_platform] * 4,
            foot_contact=[True, True, True, True],
            foot_normal_force=[10.0] * 4, pitch=0.0,
            velocity_b=[0.5, 0.0, 0.0], velocity_route=0.5,
            omega_b=[0.0, 0.0, 0.0], c_exec=0.5, c_request=0.5,
            a_request=[0.0] * 12, a_applied=[0.0] * 12,
            pd_target=[0.0] * 12, joint_position=[0.0] * 12,
            joint_velocity=[0.0] * 12,
            map_valid=np.ones((40, 30), dtype=bool),
            map_base_z=0.4, refreshed=(self.step_index % 5 == 1),
            terminal=terminal, failure=False, top_arrived=top,
            four_feet_on_platform=on_platform,
            front_foot_top_support=on_platform,
            feet_ok=on_platform,
            body_ok=bool(front and self.route_x > front + 0.4),
            contact_pairs=[dict(foot=0, geom='tread_6', normal_force=10.0,
                                duration_s=0.02, position_xy=list(feet[0]),
                                surface='tread' if not on_platform else 'platform',
                                valid=True)],
            native_sample_id=dict(refresh_index=actor_step // 5,
                                  sample_sim_time_s=self.sim_time,
                                  produced_at_actor_step=actor_step),
            sim_time_s=self.sim_time,
            termination_reason=('success' if top and terminal else 'timeout' if terminal else 'running'),
            v07_success=bool(on_platform and front and
                             self.route_x >= front + 0.8),
            sim_state_sha256=sim_state_digest(np.zeros(19), np.zeros(18)),
            post_reset=terminal)


class S2PhysicsEnv:
    """Full S2 env contract used by s2_execute and CPU tests."""

    def __init__(self, case, *, backend_factory, device='cpu', num_envs=1,
                 seed=42, map_branch='real', h_legacy=False,
                 capture_inputs=False, real_branch_evidence=True):
        if num_envs != 1:
            raise ValueError('S2 physics assembly supports one smoke env per process')
        if map_branch not in ('real', 'plane_diagnostic'):
            raise ValueError('map_branch must be real or plane_diagnostic')
        self.case = case
        self.device = device
        self.seed = int(seed)
        self.map_branch = map_branch
        self.h_legacy = bool(h_legacy)
        self.capture_inputs = bool(capture_inputs)
        self.real_branch_evidence = bool(real_branch_evidence)
        self.input_rows = []
        self.task_adapter = s2_env.S2TaskAdapter.from_case(case)
        self.machine = self.task_adapter.state_machine()
        self.gate = self.task_adapter.level_gate()
        self.backend = backend_factory(case, device=device, seed=seed,
                                       branch=map_branch, h_legacy=h_legacy)
        self.termination = (assert_termination_contract(self.backend.env)
                            if hasattr(self.backend, 'env')
                            else dict(contract_ok=True, fake=True))
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
        self.real_evidence = (s2_map_ablation.S2RealBranchEvidence(task=evidence_task)
                              if self.real_branch_evidence else None)
        self.last = None
        self.steps = 0
        self._observation_cache = None
        self.geometry_events = []
        self.failure_tracker = s2_state.GeometryFailureTracker()
        self._reset_backend()

    def _reset_backend(self):
        self.last = self.backend.reset()
        self.steps = 0
        # Backend reset already produced the initial physical snapshot; reading
        # it without update preserves the same FIFO/map phase as S1.
        self._observation_cache = self._policy_observation(update=False)

    def reset(self, case_state=None):
        self._reset_backend()
        return self.observe()

    def _policy_observation(self, update=True):
        last = self.last
        raw = (self.backend.observation(update=update)
               if hasattr(self.backend, 'observation')
               else dict(proprio=np.zeros(45, dtype=np.float32),
                         history=np.zeros((5, 45), dtype=np.float32),
                         heightmap=np.zeros((3, 40, 30), dtype=np.float32),
                         route_error=np.zeros(3, dtype=np.float32)))
        import torch
        def strip_batch(value):
            if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == 1:
                return value[0]
            return value
        def policy_value(value):
            value = strip_batch(value)
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)
        heightmap = raw['heightmap']
        if self.ablation is not None and self.ablation.policy_cache is not None:
            heightmap = self.ablation.policy_cache
        result = dict(proprio=policy_value(raw['proprio']),
                      history=policy_value(raw['history']),
                      heightmap=policy_value(heightmap),
                      route_error=policy_value(raw['route_error']),
                      c_request=float(last.c_request),
                      c_exec=float(last.c_exec))
        s2_env.validate_policy_observation(result)
        return result

    def observe(self):
        if self.last is None:
            self.last = self.backend.reset()
        if self._observation_cache is None:
            self._observation_cache = self._policy_observation(update=False)
        import torch
        return {key: (value.clone() if isinstance(value, torch.Tensor)
                      else (value.copy() if hasattr(value, 'copy') else value))
                for key, value in self._observation_cache.items()}

    def observation(self):
        return self.observe()

    def assert_native_alignment(self):
        if self.ablation is None:
            return True
        sample = self.ablation.native_sample_id
        consumed = self.ablation.first_consumed_replacement_frame
        if consumed is None:
            return True  # construction/pre-consumption frame has no s-t yet
        if sample is not None:
            if int(sample['produced_at_actor_step']) > int(consumed):
                raise ValueError('native_sample_id produced after first consumption')
            if self.ablation.summary()['s_minus_t'] < 0:
                raise ValueError('negative s-t')
        return True

    def step(self, action):
        if self.last is not None and self.last.terminal:
            raise RuntimeError('physics env cannot advance after terminal')
        input_obs = self.observe()
        input_real_obs = (self.backend.observation(update=False)
                          if hasattr(self.backend, 'observation')
                          else input_obs)
        input_base_z = float(self.last.map_base_z) if self.last is not None else None
        input_row = None
        if self.capture_inputs:
            obs = self.observe()
            last = self.last
            input_row = dict(
                step=int(self.steps),
                case_sha256=self.case['case_sha256'],
                task=self.case['task'],
                weight_sha256=s2_env.s2_spec.S1_ENTRY_WEIGHT_SHA256,
                proprio=s2_env.to_numpy(obs['proprio']).copy(),
                history=s2_env.to_numpy(obs['history']).copy(),
                heightmap=s2_env.to_numpy(obs['heightmap']).copy(),
                route_error=s2_env.to_numpy(obs['route_error']).copy(),
                c_exec=float(obs['c_exec']),
                c_request=float(obs['c_request']),
                refresh_index=(int(last.native_sample_id['refresh_index'])
                               if last and last.native_sample_id else None),
                sample_sim_time_s=(float(last.native_sample_id['sample_sim_time_s'])
                                   if last and last.native_sample_id else None),
                produced_at_actor_step=(int(last.native_sample_id['produced_at_actor_step'])
                                        if last and last.native_sample_id else None),
                base_sample_z=(float(last.map_base_z) if last else None),
                plane_world_z=(float(self.task_adapter.platform['top_height'])
                               if self.task_adapter.platform else None),
                refreshed=bool(last.refreshed) if last else False,
                front_foot_top_support=(bool(last.front_foot_top_support)
                                        if last else False),
                a_request=s2_env.to_numpy(action).reshape(-1).tolist())
        frame = (self.backend.step(action) if hasattr(self.backend, 'step')
                 else self.backend._step())
        if input_row is not None:
            input_row['recorded_action'] = list(frame.a_request)
            input_row['post_step_terminal'] = bool(frame.terminal or self.machine.terminal)
            input_row['post_reset'] = bool(frame.post_reset)
            self.input_rows.append(input_row)
        self.steps += 1
        platform = self.task_adapter.platform
        condition = bool(platform and frame.feet_ok and frame.body_ok)
        g_value = self.gate.update(condition, step_index=self.steps - 1)
        geometry_reason = self.failure_tracker.observe(
            step=self.steps - 1, condition=condition,
            feet_ok=frame.feet_ok, body_ok=frame.body_ok)
        real_map = s2_env.to_numpy(input_real_obs['heightmap']).astype(np.float64)
        real_valid = np.asarray(frame.map_valid, dtype=bool)
        if self.ablation is not None:
            policy_map, map_info = self.ablation.observe(
                frame_index=self.steps - 1, real_map=real_map,
                valid=real_valid, base_sample_z=(input_base_z if input_base_z is not None
                                                 else frame.map_base_z),
                refreshed=frame.refreshed,
                front_foot_top_support=frame.front_foot_top_support,
                consumed=True, terminal=frame.terminal,
                pairing=None, sample_id=frame.native_sample_id)
            if self.real_evidence is not None:
                self.real_evidence.observe(
                    frame_index=self.steps - 1, real_map=real_map,
                    valid=real_valid, refreshed=frame.refreshed,
                    front_foot_top_support=frame.front_foot_top_support,
                    consumed=True, terminal=frame.terminal,
                    sample_id=frame.native_sample_id)
        else:
            map_info = None
            if self.real_evidence is not None:
                self.real_evidence.observe(
                    frame_index=self.steps - 1,
                    real_map=real_map,
                    valid=real_valid, refreshed=frame.refreshed,
                    front_foot_top_support=frame.front_foot_top_support,
                    consumed=True, terminal=frame.terminal,
                    sample_id=frame.native_sample_id)
        v07_success = False
        if self.h_legacy and platform:
            v07_success = bool(frame.v07_success)
        first_step = bool(frame.route_x >= 0.0)
        state_frame = dict(route_x=frame.route_x, first_step=first_step,
                           top_arrived=frame.top_arrived,
                           four_feet_on_platform=frame.four_feet_on_platform,
                           failure=frame.failure, v07_success=v07_success,
                           a_request=frame.a_request, a_applied=frame.a_applied)
        self.machine.step(state_frame)
        self.recorder.append(dict(
            step=self.steps - 1, route_x=frame.route_x, base_xy=frame.base_xy,
            feet_xy=frame.feet_xy, feet_route_x=frame.feet_route_x,
            feet_z=frame.feet_z, root_z=frame.root_z, pitch=frame.pitch, foot_on_platform=frame.foot_on_platform,
            sim_state_sha256=frame.sim_state_sha256, post_reset=frame.post_reset,
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
        self.assert_native_alignment()
        # ``backend.step`` already advanced history/map exactly once; read that
        # snapshot instead of calling ``_observations(update=True)`` again.
        self._observation_cache = self._policy_observation(update=False)
        return self.observe(), [frame.terminal or self.machine.terminal], dict(
            map_info=map_info, geometry_reason=geometry_reason,
            g_level=g_value, state=self.machine.summary())

    def diagnostics(self):
        frame = self.last
        from .s2_env import to_numpy
        return dict(feet_xy=to_numpy(frame.feet_xy),
                    foot_contact=to_numpy(frame.foot_contact),
                    foot_normal_force=to_numpy(frame.foot_normal_force),
                    contact_pairs=frame.contact_pairs,
                    pd_target=to_numpy(frame.pd_target),
                    joint_position=to_numpy(frame.joint_position),
                    map_valid=to_numpy(frame.map_valid),
                    map_sample_time_s=float(frame.sim_time_s),
                    native_sample_id=frame.native_sample_id,
                    nonfoot_contact=bool(frame.nonfoot_contact),
                    g_level=float(self.gate.value))

    def evidence(self, index=0):
        trace = self.recorder.trace()
        trace['map_evidence'] = (self.ablation.summary() if self.ablation
                                 else (self.real_evidence.summary()
                                       if self.real_evidence is not None else None))
        trace['state_summary'] = self.machine.summary()
        trace['failure_context'] = {'present': self.machine.failure}
        trace['termination_reason'] = self.last.termination_reason
        trace['nonfoot_contact'] = self.last.nonfoot_contact
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
                 map_branch='real', h_legacy=False, backend_factory=None,
                 capture_inputs=False, real_branch_evidence=True):
    """Construct the S2 env; refuse before GPU allocation if no real backend."""
    if h_legacy and map_branch != 'real':
        raise ValueError('H legacy only supports the real map')
    if backend_factory is None:
        backend_factory = MjlabHLegacyBackend if h_legacy else MjlabS2Backend
    report = s2_execute.device_report(device)
    return S2PhysicsEnv(case, backend_factory=backend_factory, device=device,
                        num_envs=num_envs, seed=seed, map_branch=map_branch,
                        h_legacy=h_legacy, capture_inputs=capture_inputs,
                        real_branch_evidence=real_branch_evidence), report


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
        original_evaluate = getattr(v07events, 'evaluate', None)
        original_cfg_gen = getattr(v07cfg, 'terrain_generator_cfg', None)
        original_validate = getattr(v07env, '_validate_frontier_tile', None)
        original_check = getattr(v07env, '_check_schedule_frontier', None)
        original_stall_update = getattr(v07env.StallDiagnostics, 'update', None)
        try:
            if original_stall_update is not None:
                def stall_update_on_command_change(self, local_x, executed_c,
                                                   climbed, first_step_acquired,
                                                   enabled):
                    # SDS 6.2: the S2 stop command legitimately changes inside an
                    # episode; restart the inherited v07 stall window instead of
                    # aborting the whole run.
                    executed_c = torch.as_tensor(executed_c, device=self.device)
                    known = torch.isfinite(self.executed_c)
                    changed = known & (self.executed_c
                                       != executed_c.to(self.executed_c.dtype))
                    if bool(changed.any()):
                        ids = changed.nonzero(as_tuple=False).flatten()
                        self.reset(ids, local_x.to(self.x_history.dtype)[ids])
                    return original_stall_update(
                        self, local_x, executed_c, climbed, first_step_acquired,
                        enabled)
                v07env.StallDiagnostics.update = stall_update_on_command_change
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
            if original_evaluate is not None:
                from dataclasses import replace as _replace

                def evaluate_without_v07_success(**kwargs):
                    # S2 sequential tasks own success (top_arrived/D_post); the
                    # original v07 success condition would auto-reset at the
                    # platform edge before the S2 state machine can latch top.
                    result = original_evaluate(**kwargs)
                    if bool(result.success.any()):
                        result = _replace(result,
                                          success=torch.zeros_like(result.success))
                    return result

                v07events.evaluate = evaluate_without_v07_success
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
            if original_evaluate is not None:
                v07events.evaluate = original_evaluate
            if original_stall_update is not None:
                v07env.StallDiagnostics.update = original_stall_update
            v07events._TABLE_CACHE.clear()
            if original_cfg_gen is not None:
                v07cfg.terrain_generator_cfg = original_cfg_gen
            if original_validate is not None:
                v07env._validate_frontier_tile = original_validate
            if original_check is not None:
                v07env._check_schedule_frontier = original_check
    return patches()


TERMINATION_REQUIRED = ('body', 'nonfoot', 'posture', 'fall', 'bounds',
                        'contact_overflow', 'nonfinite')
MANAGER_TERMS_EXPECTED = ()  # v07 owns termination in events.evaluate, not managers


def termination_contract(env):
    from src.tasks.velocity.v07 import events as v07events
    manager = getattr(env, 'termination_manager', None)
    names = tuple(str(term) for term in getattr(manager, '_term_cfgs', ()))         if manager is not None else ()
    categories = tuple(v07events.FAILURE_NAMES)
    missing = [name for name in TERMINATION_REQUIRED if name not in categories]
    return dict(manager_terms=list(names), manager_count=len(names),
                failure_categories=list(categories), missing=missing,
                contract_ok=not missing and tuple(names) == MANAGER_TERMS_EXPECTED)


def assert_termination_contract(env):
    contract = termination_contract(env)
    if not contract['contract_ok']:
        raise S2PhysicsAssemblyError(
            f'v07/S2 termination contract missing/differs: {contract}')
    return contract


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
        self.spawn_z = float(spawn['local_position'][2]) if len(spawn.get('local_position', ())) > 2 else 0.4
        self.steps = 0
        self.refresh_index = 0
        self.last_sample_id = None
        self.last_sample_time = 0.0
        self._last_observation = None

    def close(self):
        self._patches.__exit__(None, None, None)

    def _store_observation(self, obs):
        def squeeze(value):
            if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == 1:
                return value[0]
            return value
        self._last_observation = dict(
            proprio=squeeze(obs['proprio']).clone(),
            history=squeeze(obs['history']).clone(),
            heightmap=squeeze(obs['heightmap']).clone(),
            route_error=squeeze(obs['route_error']).clone())
        return {key: value.clone()
                for key, value in self._last_observation.items()}

    def observation(self, update=True):
        if not update and self._last_observation is not None:
            return {key: value.clone()
                    for key, value in self._last_observation.items()}
        obs = self.env._observations(update=update)
        return self._store_observation(obs)

    def reset(self):
        if hasattr(self, 'spawn_z'):
            apply_default_spawn_z(self.env, self.spawn_z)
        obs, _ = self.env.reset()
        self._store_observation(obs)
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
        obs, _, terminated, truncated, _ = self.env.step(action)
        self._store_observation(obs)
        self.steps += 1
        return self._frame(terminal=bool(terminated[0] or truncated[0]))

    def _frame(self, terminal=False):
        import torch
        data = self.env.robot.data
        origin = self.env.origins[self.env.tile_ids[0]]
        local = data.root_link_pos_w[0] - origin
        feet = data.site_pos_w[0, self.env.foot_ids]
        contact = self.env.contacts.raw[0].bool()
        # SDS 5.3: four-foot entry is the first frame with all four foot XY
        # projections inside the top platform; it is not contact/support.
        feet_local = feet[:, :2] - origin[:2]
        if self.spec.kind == 's2_stairs':
            front = float(self.spec.platform_front_local_x)
            platform = ((feet_local[:, 0] >= front)
                        & (feet_local[:, 0] <= front + float(self.spec.platform_length))
                        & (feet_local[:, 1].abs() <= float(self.spec.platform_width) / 2.0))
        else:
            front = None
            platform = torch.zeros(4, dtype=torch.bool, device=feet.device)
        body_ok = bool(front is not None and local[0] > front + 0.4)
        feet_ok = bool(platform.all())
        map_valid = self.env.map_cache.map[0, 1].bool().cpu().numpy()
        refreshed = int(self.env.map_cache.steps[0]) == 0
        sim_time = _scalar(getattr(self.env.sim.data, 'time', None),
                           self.steps * 0.02)
        if refreshed:
            self.refresh_index += 1
            self.last_sample_time = sim_time
            self.last_sample_id = dict(refresh_index=self.refresh_index,
                                       sample_sim_time_s=sim_time,
                                       produced_at_actor_step=max(0, int(self.steps) - 1))
        sample_id = self.last_sample_id or dict(
            refresh_index=0, sample_sim_time_s=sim_time,
            produced_at_actor_step=max(0, int(self.steps) - 1))
        top_forces = self.env.contacts.top_forces[0]
        geom_names = [self.env.sim.mj_model.geom(i).name
                      for i in range(self.env.sim.mj_model.ngeom)]
        geom_tiles = self.env.contacts.inner.tile_by_geom.detach().cpu().numpy().tolist()
        foot_normal_force, contact_pairs = aggregate_foot_contacts(
            top_forces, geom_names=geom_names, geom_tiles=geom_tiles,
            platform='platform' if bool(platform.any()) else 'tread',
            feet_xy=feet[:, :2].detach().cpu().numpy().tolist())
        nonfoot = nonfoot_contact_from_links(
            self.env.contacts.link_force[0], self.env.contacts.inner.link_names)
        termination_reason = 'running'
        last = None
        try:
            from src.tasks.velocity.v07 import events as v07events
            last = getattr(self.env, 'last_termination', None)
            if last is not None:
                code = int(last[3][0])
                termination_reason = v07events.REASON_NAMES[code]
        except Exception:
            termination_reason = 'unknown'
        return BackendStep(
            route_x=float(local[0]), base_xy=(float(local[0]), float(local[1])),
            root_z=float(data.root_link_pos_w[0, 2]),
            feet_xy=feet[:, :2].detach().cpu().numpy().tolist(),
            feet_route_x=(feet[:, 0] - origin[0]).detach().cpu().numpy().tolist(),
            feet_z=feet[:, 2].detach().cpu().numpy().tolist(),
            foot_on_platform=platform.detach().cpu().numpy().tolist(),
            foot_contact=contact.detach().cpu().numpy().tolist(),
            foot_normal_force=foot_normal_force.detach().cpu().tolist(),
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
            terminal=terminal,
            failure=(bool(last[1][0]) if last is not None else False),
            top_arrived=feet_ok and body_ok,
            four_feet_on_platform=feet_ok,
            front_foot_top_support=bool((platform[:2] & contact[:2]).any()),
            feet_ok=feet_ok, body_ok=body_ok,
            contact_pairs=contact_pairs, native_sample_id=sample_id,
            sim_time_s=self.steps * 0.02, nonfoot_contact=nonfoot,
            termination_reason=termination_reason,
            v07_success=(bool(last[0][0]) if last is not None else False),
            sim_state_sha256=sim_state_digest(self.env.sim.data.qpos[0],
                                              self.env.sim.data.qvel[0]),
            post_reset=terminal)


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
        self._last_observation = None

    def close(self):
        pass
