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
    if backend_factory is None:
        raise S2PhysicsAssemblyError(
            'no real v09/mjlab S2 backend_factory is installed; provide '
            'backend_factory and re-run after the V7 GPU audit')
    report = s2_execute.device_report(device)
    return S2PhysicsEnv(case, backend_factory=backend_factory, device=device,
                        num_envs=num_envs, seed=seed, map_branch=map_branch,
                        h_legacy=h_legacy), report
