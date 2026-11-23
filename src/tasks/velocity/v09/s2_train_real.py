"""S2.1 real multi-env training assembly seam (CPU contract, GPU launched later).

The 1024-env environment is a V07Env-derived class with S2 terrain/map patches.
The static plan is CPU-testable; constructing the simulator is deliberately
separate from the plan.
"""
from __future__ import annotations

import copy

import numpy as np

from . import (s2_lifecycle, s2_physics, s2_quality, s2_rewards, s2_spec,
               s2_terrain, s2_train_env)
from .map import _SampleFrameMap


OBSERVATION_CONTRACT = dict(
    proprio=(45,), history=(5, 45), heightmap=(3, 40, 30), route_error=(3,),
    privileged=(35,), critic=211)
MAP_REFRESH_STEPS = 5
CONTROL_DT = 0.02


def _geometry_for(row, course='S2.1'):
    task = row.get('task', row.get('role'))
    if task in ('flat_walk', 'flat_stop'):
        return dict(kind='flat', terrain_key='s2_flat_v07_tile0', tile_id=0,
                    lesson_id='FLAT', N=0, h=0.0, step_depth=0.2, width=8.0,
                    tread_thickness=0.01, platform_length=0.0,
                    platform_width=8.0, platform_present=False,
                    platform_height=0.0, platform_front_local_x=None,
                    frame_origin_world=[0.0, 0.0, 0.0],
                    valid_bounds_local=[-2.0, 22.0, -4.0, 4.0])
    if task in ('stairs_walk', 'stairs_stop', 'legacy_l40'):
        levels = int(row['stair_count'])
        width = float(row.get('platform_width', 0.8))
        # S2.1 keeps the historical 20 m L40 review row; S2.2/S2.3 use the
        # S2 40-level row and the 10/20 review rows.
        if task == 'legacy_l40' or (course == 'S2.1' and levels == 40):
            return dict(kind='s2_stairs',
                        terrain_key='s2_stairs_h0.20_d0.20_w0.80_pw0.8',
                        tile_id=None, lesson_id='S2_N40_PW0.8',
                        N=40, h=0.2, step_depth=0.2, width=0.8,
                        tread_thickness=0.01, platform_length=20.0,
                        platform_width=0.8, platform_present=True,
                        platform_height=8.0, platform_front_local_x=8.0,
                        frame_origin_world=[0.0, 12.0, 0.0],
                        valid_bounds_local=[-2.0, 28.0, -2.0, 2.0])
        return dict(kind='s2_stairs',
                    terrain_key=f's2_stairs_h0.20_d0.20_w0.80_pw{width:g}',
                    tile_id=None, lesson_id=f'S2_N{levels}_PW{width:g}',
                    N=levels, h=0.2, step_depth=0.2, width=0.8,
                    tread_thickness=0.01, platform_length=20.0,
                    platform_width=width, platform_present=True,
                    platform_height=levels * 0.2,
                    platform_front_local_x=levels * 0.2,
                    frame_origin_world=[0.0, 12.0, 0.0],
                    valid_bounds_local=[-2.0, levels * 0.2 + 20.0, -2.0, 2.0])
    if task in ('platform_walk', 'platform_stop'):
        height = float(row['platform_height'])
        levels = int(round(height / 0.2))
        width = float(row['platform_width'])
        return dict(kind='s2_stairs',
                    terrain_key=f's2_stairs_h0.20_d0.20_w0.80_pw{width:g}',
                    tile_id=None, lesson_id=f'S2_N{levels}_PW{width:g}',
                    N=levels, h=0.2, step_depth=0.2, width=0.8,
                    tread_thickness=0.01, platform_length=20.0,
                    platform_width=width, platform_present=True,
                    platform_height=height,
                    platform_front_local_x=levels * 0.2,
                    frame_origin_world=[0.0, 12.0, 0.0],
                    valid_bounds_local=[-2.0, levels * 0.2 + 20.0, -2.0, 2.0])
    # historical role fallback
    return dict(kind='s2_stairs',
                terrain_key='s2_stairs_h0.20_d0.20_w0.80_pw0.8',
                tile_id=None, lesson_id='S2_N40_PW0.8',
                N=40, h=0.2, step_depth=0.2, width=0.8,
                tread_thickness=0.01, platform_length=20.0,
                platform_width=0.8, platform_present=True,
                platform_height=8.0, platform_front_local_x=8.0,
                frame_origin_world=[0.0, 12.0, 0.0],
                valid_bounds_local=[-2.0, 28.0, -2.0, 2.0])


def apply_train_spawn_z(env, plan):
    import torch
    env.robot.data.default_root_state[:, 2] = torch.tensor(
        [row['spawn_z'] for row in plan['rows']],
        dtype=env.robot.data.default_root_state.dtype,
        device=env.robot.data.default_root_state.device)
    return env.robot.data.default_root_state[:, 2]


def build_env_plan(num_envs=1024, course='S2.1'):
    role_plan = s2_train_env.build_role_plan(num_envs, course=course)
    rows = []
    for index, role in enumerate(role_plan['envs']):
        geometry = _geometry_for(role, course=course)
        # terrain_key alone is shared by every stair count, so the frozen row
        # must be matched on (N, width, h, depth, platform length), not the key.
        spec = s2_terrain.spec_from_geometry(geometry)
        geometry['tile_id'] = spec.tile_id
        front = geometry.get('platform_front_local_x')
        if str(role.get('task', '')).startswith('platform_'):
            spawn_x = float(front) + 1.0
            spawn_y = 0.0
            yaw = 0.0
            spawn_z = float(geometry['platform_height']) + 0.4
        else:
            spawn_x = -1.0
            spawn_y = 0.0
            yaw = 0.0
            spawn_z = 0.4
        rows.append(dict(
            env_index=index, role=role['role'], speed=float(role['speed']),
            geometry=geometry, tile_id=int(spec.tile_id),
            spawn=[spawn_x, spawn_y, yaw, float(role['speed'])],
            spawn_z=spawn_z,
            task=role['task'],
            stop_mode=role.get('stop_mode'),
            post_distance=role.get('post_distance'),
            platform_width=role.get('platform_width'),
            stair_count=role.get('stair_count')))
    return dict(num_envs=num_envs, course=course, role_counts=role_plan['counts'],
                rows=rows, observation=OBSERVATION_CONTRACT,
                map_refresh_steps=MAP_REFRESH_STEPS, control_dt=CONTROL_DT)


def assert_v07_contract(env):
    """Physical/PD/action/frequency identity for a constructed V07-derived env."""
    import torch
    expected = s2_physics.assembly_contract()
    checks = {}
    for key in ('control_dt', 'physics_dt', 'decimation', 'action_scale',
                'joints', 'map_shape', 'map_grid_step', 'map_x', 'map_y',
                'map_refresh_steps'):
        left, right = expected[key][0], expected[key][1]
        checks[key] = bool(left == right)
    if not all(checks.values()):
        raise ValueError(f'v07/S2 contract drift: {checks}')
    if tuple(env.map_cache.map.shape[1:]) != (3, 40, 30):
        raise ValueError('training map is not 3x40x30')
    if int(env.map_cache.steps.numel()) != int(env.num_envs):
        raise ValueError('map refresh state length differs from env count')
    return checks


class S2TrainEnv:
    """Constructed lazily; requires mjlab/warp and a real device."""

    def __init__(self, *, num_envs=1024, device='cuda:0', seed=42, course='S2.1'):
        import torch
        from src.tasks.velocity.v07.env import V07Env
        from .map import HeightMap
        self.plan = build_env_plan(num_envs, course=course)
        self.num_envs = int(num_envs)
        self.role_plan = dict(num_envs=self.num_envs, course=course,
                              counts=dict(self.plan['role_counts']),
                              order=list(s2_train_env.ROLE_ORDER),
                              envs=list(self.plan['rows']))
        self.device = device
        self.seed = int(seed)
        self._tile_by_env = [row['tile_id'] for row in self.plan['rows']]
        self._role_by_env = [row['role'] for row in self.plan['rows']]
        self._s2_top_latched = None
        self._s2_success_latched = None
        self._s2_plan_tensors = None
        self._patch = s2_physics._s2_patch_context(s2_terrain.terrain_specs())
        self._top_capture = getattr(self._patch, 'raw_v07_success', None)
        self._patch.__enter__()
        try:
            self.env = V07Env(
                num_envs=self.num_envs, device=device, seed=seed, phase='L',
                substage='L40',
                initial_cases=[row['spawn'] for row in self.plan['rows']],
                frontier_tile_id=0, evaluation=False)
            self.env.map_cache = HeightMap(self.num_envs, device)
            apply_train_spawn_z(self.env, self.plan)
            import types
            tile_by = list(self._tile_by_env)
            def _sample_tiles(_self, ids):
                ids = ids.to(torch.long)
                self.env.tile_ids[ids] = torch.as_tensor(
                    [tile_by[int(i)] for i in ids.tolist()], dtype=torch.long,
                    device=device)
            self.env._sample_tiles = types.MethodType(_sample_tiles, self.env)
            self.env.roles = list(self._role_by_env)
            self._v07_task_state = self.env._task_state
            def _s2_task_state(_env):
                return self._s2_task_state(_env)
            self.env._task_state = types.MethodType(_s2_task_state, self.env)
            def _s2_reward(_env, state, previous_action):
                return self._s2_reward(_env, state, previous_action)
            self.env._reward = types.MethodType(_s2_reward, self.env)
            self.env.reset()
            from src.tasks.velocity.v07 import terrain as _v07terrain
            self._teacher_map = _SampleFrameMap(
                self.num_envs, device, (3, 48, 32), -0.8, -0.8,
                _v07terrain.query_surface)
            self._update_teacher_map()
            assert_v07_contract(self.env)
        except Exception:
            self._patch.__exit__(None, None, None)
            raise

    def _s2_task_state(self, env):
        """S2 termination: top is an event; success waits for post distance."""
        import torch
        state = self._v07_task_state()
        n = int(env.num_envs)
        if not hasattr(self, '_s2_top_event') or self._s2_top_event is None:
            self._s2_top_event = torch.zeros(n, dtype=torch.bool, device=env.device)
            self._s2_top_latched = torch.zeros(n, dtype=torch.bool, device=env.device)
            self._s2_success_latched = torch.zeros(n, dtype=torch.bool, device=env.device)
            self._s2_spawn_x = torch.tensor(
                [float(r['spawn'][0]) for r in self.plan['rows']],
                dtype=torch.float32, device=env.device)
            self._s2_front = torch.tensor(
                [float(r['geometry'].get('platform_front_local_x') or 0.0)
                 for r in self.plan['rows']], dtype=torch.float32, device=env.device)
            self._s2_post = torch.tensor(
                [float(r.get('post_distance') or 5.0) for r in self.plan['rows']],
                dtype=torch.float32, device=env.device)
            self._s2_legacy = torch.tensor(
                [str(r.get('task', '')) == 'legacy_l40' for r in self.plan['rows']],
                dtype=torch.bool, device=env.device)
            self._s2_stairs = torch.tensor(
                ['stairs' in str(r.get('task', '')) for r in self.plan['rows']],
                dtype=torch.bool, device=env.device)
            self._s2_platform = torch.tensor(
                [str(r.get('task', '')).startswith('platform_') for r in self.plan['rows']],
                dtype=torch.bool, device=env.device)
            self._s2_speed = torch.tensor(
                [float(r.get('speed') or 0.5) for r in self.plan['rows']],
                dtype=torch.float32, device=env.device)
            self._s2_stop = torch.tensor(
                [str(r.get('task', '')).endswith('_stop') for r in self.plan['rows']],
                dtype=torch.bool, device=env.device)
            self._s2_c_request = self._s2_speed.clone()
            self._s2_c_exec = self._s2_speed.clone()
        if not hasattr(self, '_s2_stop_requested'):
            z = lambda *shape, dtype=torch.float32: torch.zeros(
                *shape, dtype=dtype, device=env.device)
            self._s2_stop_requested = z(n, dtype=torch.bool)
            self._s2_stop_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
            self._s2_hold_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
            self._s2_g_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
            self._s2_air_time = z(n, 4)
            self._s2_prev_contact = z(n, 4, dtype=torch.bool)
            self._s2_contacted = z(n, 4, dtype=torch.bool)
            self._s2_prev_qdot = z(n, 12)
            self._s2_g_level = z(n)
            self._s2_steps = z(n, dtype=torch.long)
        if not hasattr(self, '_lifecycle'):
            rows = self.plan['rows']
            tasks = [str(r.get('task') or r.get('role')) for r in rows]
            stop_modes = [str(r.get('stop_mode') or 'none') for r in rows]
            fronts = [float(r['geometry'].get('platform_front_local_x') or 0.0)
                      for r in rows]
            spawns = [float(r['spawn'][0]) for r in rows]
            post = [float(r.get('post_distance') or 5.0) for r in rows]
            speeds = [float(r.get('speed') or 0.5) for r in rows]
            legacy = [t == 'legacy_l40' for t in tasks]
            # Planned endpoints in completed 0.02 s control intervals: flat_walk
            # 1000 (20 s), flat_stop 1400 (instant) / 1500 (ramp 2 s).  The
            # +1 hard-protection limit lives in S2Lifecycle, never here.
            completion = [
                1000 if t == 'flat_walk' else
                (1500 if (t == 'flat_stop' and stop_modes[i] == 'ramp_2s') else
                 (1400 if t == 'flat_stop' else -1))
                for i, t in enumerate(tasks)]
            self._lifecycle = s2_lifecycle.S2Lifecycle(
                task=tasks, post_distance=post, speed=speeds,
                stop_mode=stop_modes, legacy=legacy, front=fronts,
                spawn_x=spawns, completion_steps=completion)
            self._quality = s2_quality.S2Quality(
                task=tasks, speed=speeds, front=fronts, spawn_x=spawns,
                post_distance=post,
                platform_length=[float(r['geometry'].get('platform_length') or 0.0)
                                 for r in rows],
                platform_width=[float(r['geometry'].get('platform_width') or 8.0)
                                for r in rows],
                stop_mode=stop_modes, legacy=legacy)
            self._s2_flat = torch.tensor(
                [t in ('flat_walk', 'flat_stop') for t in tasks],
                dtype=torch.bool, device=env.device)
            self._s2_top_tasks = torch.tensor(
                [t in ('stairs_walk', 'stairs_stop', 'legacy_l40')
                 for t in tasks], dtype=torch.bool, device=env.device)
            self._s2_platform_width = torch.tensor(
                [float(r['geometry'].get('platform_width') or 8.0) for r in rows],
                dtype=torch.float32, device=env.device)
            self._s2_platform_length = torch.tensor(
                [float(r['geometry'].get('platform_length') or 0.0) for r in rows],
                dtype=torch.float32, device=env.device)
        first = env.episode_length_buf <= 1
        first_np = first.detach().cpu().numpy()
        self._lifecycle.reset(first_np)
        self._quality.reset(first_np)
        self._s2_top_latched[first] = False
        self._s2_success_latched[first] = False
        self._s2_top_event[first] = False
        self._s2_stop_requested[first] = False
        self._s2_stop_step[first] = -1
        self._s2_hold_step[first] = -1
        self._s2_g_step[first] = -1
        self._s2_air_time[first] = 0
        self._s2_prev_contact[first] = False
        self._s2_contacted[first] = False
        self._s2_prev_qdot[first] = 0
        self._s2_g_level[first] = 0
        self._s2_steps[first] = 0
        self._s2_c_request[first] = self._s2_speed[first]
        self._s2_c_exec[first] = self._s2_speed[first]
        if hasattr(self, '_s2_support_count'):
            self._s2_support_count[first] = 0
        local = state.local[:, 0]
        failure = state.failure.clone()
        timeout = state.timeout.clone()
        # DSHS-Q036: four-foot XY entry + body > front+0.4 is the §5.2 gating
        # start, never ``top_arrived``.  The safe summit is the raw v07 success
        # captured by ``s2_physics._s2_patch_context`` *before* the S2 patch
        # suppresses v07 termination (actual tile geometry, 0.8 m body margin,
        # 0.042 m foot inset, real support and 5-frame confirmation).
        raw_top = (self._top_capture.take() if getattr(
            self, '_top_capture', None) is not None else None)
        if raw_top is None:
            raw_top = torch.zeros(n, dtype=torch.bool, device=env.device)
        else:
            raw_top = raw_top.to(env.device).reshape(-1)
        top_signal = raw_top & self._s2_top_tasks
        # §5.2 gating geometry, recorded separately from the safe summit.
        feet_local = state.feet_local
        in_platform = (
            (feet_local[:, :, 0] >= self._s2_front[:, None])
            & (feet_local[:, :, 0]
               <= self._s2_front[:, None] + self._s2_platform_length[:, None])
            & (feet_local[:, :, 1].abs()
               <= self._s2_platform_width[:, None] / 2.0))
        four_feet = in_platform.all(-1)
        body_ok = local > self._s2_front + 0.4
        out = self._s2_quality_update(env, state, top_signal, four_feet)
        self._lifecycle_out = out
        self._s2_phase = out['phase']
        self._s2_c_request = torch.as_tensor(out['c_request'], dtype=torch.float32,
                                             device=env.device)
        self._s2_c_exec = torch.as_tensor(out['c_exec'], dtype=torch.float32,
                                          device=env.device)
        self._s2_top_event = torch.as_tensor(out['top_event'], dtype=torch.bool,
                                             device=env.device)
        env.command[:, 0] = self._s2_c_exec.detach()
        import dataclasses
        return dataclasses.replace(
            state,
            success=torch.as_tensor(out['success'], dtype=torch.bool,
                                    device=env.device),
            failure=torch.as_tensor(out['failure'], dtype=torch.bool,
                                    device=env.device),
            timeout=torch.as_tensor(out['timeout'], dtype=torch.bool,
                                    device=env.device))

    def _s2_foot_support(self, env, feet_local):
        """Per-foot §5.3 valid support: real top-face >5 N inside the 0.042 m
        inset of the actual support rectangle (platform/tread/ground)."""
        import torch
        tops = env.contacts.top_forces
        same_tile = env.geom_tile[None, None, :] == env.tile_ids[:, None, None]
        force = torch.where(same_tile, tops,
                            torch.zeros_like(tops)).sum(-1)
        force_ok = force > 5.0
        x = feet_local[..., 0]
        y = feet_local[..., 1]
        inset = 0.042
        front = self._s2_front[:, None]
        width = self._s2_platform_width[:, None]
        length = self._s2_platform_length[:, None]
        platform = ((x >= front + inset)
                    & (x <= front + length - inset)
                    & (y.abs() <= width / 2.0 - inset))
        tread = ((x >= inset) & (x <= front - inset)
                 & (y.abs() <= 0.4 - inset))
        ground = ((x >= -2.0 + inset) & (x < inset)
                  & (y.abs() <= 2.0 - inset))
        flat_ground = ((x >= -2.0 + inset) & (x <= 22.0 - inset)
                       & (y.abs() <= 4.0 - inset))
        region = torch.where(self._s2_stairs[:, None],
                             platform | tread | ground, flat_ground)
        raw = force_ok & region
        # §5.3: two consecutive control frames of real top-face support.
        if not hasattr(self, '_s2_support_count'):
            self._s2_support_count = torch.zeros(
                raw.shape, dtype=torch.long, device=raw.device)
        self._s2_support_count = torch.where(
            raw, self._s2_support_count + 1,
            torch.zeros_like(self._s2_support_count))
        return self._s2_support_count >= 2

    def _s2_quality_update(self, env, state, top_signal, four_feet):
        """Measure this frame and advance the shared S2 quality state machine."""
        import math
        import torch
        from src.tasks.velocity.v07 import events as _v07events
        from src.tasks.velocity.v07 import terrain as _v07terrain
        n = int(env.num_envs)
        data = env.robot.data
        local = state.local
        roll, pitch, _ = _v07events.euler_heading(data.root_link_quat_w)
        surface_z, surface_valid = _v07terrain.query_surface(
            data.root_link_pos_w[:, :2], env.tile_ids)
        surface_z = torch.where(surface_valid, surface_z,
                                torch.zeros_like(surface_z))
        foot_support = self._s2_foot_support(env, state.feet_local)
        foot_speed = data.site_lin_vel_w[:, env.foot_ids, :2].norm(dim=-1)
        expected = (state.failure | state.timeout)
        quality = self._quality.update(
            local_x=local[:, 0].detach().cpu().numpy(),
            local_y=local[:, 1].detach().cpu().numpy(),
            pitch_deg=(pitch * (180.0 / math.pi)).detach().cpu().numpy(),
            roll_deg=(roll * (180.0 / math.pi)).detach().cpu().numpy(),
            horizontal_speed=(data.root_link_lin_vel_w[:, :2]
                              .norm(dim=-1)).detach().cpu().numpy(),
            v_heading=data.root_link_lin_vel_w[:, 0].detach().cpu().numpy(),
            c_exec=env.sample_command[:, 0].detach().cpu().numpy(),
            four_feet_entry=four_feet.detach().cpu().numpy(),
            top_arrived=top_signal.detach().cpu().numpy(),
            base_height=data.root_link_pos_w[:, 2].detach().cpu().numpy(),
            surface_height=surface_z.detach().cpu().numpy(),
            foot_support=foot_support.detach().cpu().numpy(),
            foot_speed_xy=foot_speed.detach().cpu().numpy(),
            angular_speed=data.root_link_ang_vel_b.norm(dim=-1)
            .detach().cpu().numpy(),
            failure=state.failure.detach().cpu().numpy(),
            # DSHS-Q036: flat hard protection is the S2 1001/1401/1501 limit,
            # never the raw v07 1500-step flat timeout.
            timeout=(state.timeout & ~self._s2_flat).detach().cpu().numpy())
        self._quality_out = quality
        out = self._lifecycle.update(
            local_x=local[:, 0].detach().cpu().numpy(),
            top_signal=top_signal.detach().cpu().numpy(),
            four_feet=four_feet.detach().cpu().numpy(),
            body_ok=(local[:, 0] > self._s2_front + 0.4)
            .detach().cpu().numpy(),
            quality_ok=quality['quality_ok'],
            quality_failure=quality['quality_failure'],
            quality_reason=quality['quality_reason'],
            failure=expected.detach().cpu().numpy(),
            # DSHS-Q036: flat protection limits are 1001/1401/1501 inside the
            # lifecycle; the v07 1500-step flat timeout must not pre-empt them.
            timeout=(state.timeout & ~self._s2_flat).detach().cpu().numpy())
        return out

    def _s2_reward(self, env, state, previous_action):
        """Vectorized S2 §5 reward; never the v07 RewardComputer."""
        import torch
        n = int(env.num_envs)
        if self._s2_plan_tensors is None:
            rows = self.plan['rows']
            device = env.device
            self._s2_front = torch.tensor(
                [float(r['geometry'].get('platform_front_local_x') or 0.0)
                 for r in rows], dtype=torch.float32, device=device)
            self._s2_post = torch.tensor(
                [float(r.get('post_distance') or 5.0) for r in rows],
                dtype=torch.float32, device=device)
            self._s2_h = torch.tensor(
                [float(r['geometry'].get('h') or 0.0) for r in rows],
                dtype=torch.float32, device=device)
            self._s2_stairs = torch.tensor(
                ['stairs' in str(r.get('task', '')) for r in rows],
                dtype=torch.bool, device=device)
            self._s2_stop = torch.tensor(
                [str(r.get('task', '')).endswith('_stop') for r in rows],
                dtype=torch.bool, device=device)
            self._s2_ramp = torch.tensor(
                [str(r.get('stop_mode', 'none')) == 'ramp_2s' for r in rows],
                dtype=torch.bool, device=device)
            self._s2_plan_tensors = True
        if not hasattr(self, '_s2_g_level'):
            device = env.device
            z = lambda *shape, dtype=torch.float32: torch.zeros(
                *shape, dtype=dtype, device=device)
            self._s2_top_latched = z(n, dtype=torch.bool)
            self._s2_success_latched = z(n, dtype=torch.bool)
            self._s2_air_time = z(n, 4)
            self._s2_prev_contact = z(n, 4, dtype=torch.bool)
            self._s2_contacted = z(n, 4, dtype=torch.bool)
            self._s2_prev_qdot = z(n, 12)
            self._s2_stop_requested = z(n, dtype=torch.bool)
            self._s2_stop_step = torch.full((n,), -1, dtype=torch.long,
                                            device=device)
            self._s2_hold_step = torch.full((n,), -1, dtype=torch.long,
                                            device=device)
            self._s2_g_step = torch.full((n,), -1, dtype=torch.long,
                                         device=device)
            self._s2_g_level = z(n)
            self._s2_steps = torch.zeros(n, dtype=torch.long, device=device)
        data = env.robot.data
        local = data.root_link_pos_w - env.origins[env.tile_ids]
        feet_local = data.site_pos_w[:, env.foot_ids] - env.origins[env.tile_ids, None]
        feet_xy = data.site_pos_w[:, env.foot_ids, :2]
        from src.tasks.velocity.v07 import terrain as _v07terrain
        surface_z, surface_valid = _v07terrain.query_surface(
            data.root_link_pos_w[:, :2], env.tile_ids)
        surface_z = torch.where(surface_valid, surface_z,
                                torch.zeros_like(surface_z))
        foot_h, _ = env._foot_heights()
        contact = env.contacts.filtered.bool()
        foot_vel = data.site_lin_vel_w[:, env.foot_ids, :2]
        failure = state.failure | state.timeout
        # §5.3/§4 g_level: single source of truth is the lifecycle state machine
        # (top & four-foot support & body margin, 0.5 s = 25-step ramp).
        g_level = torch.as_tensor(
            self._lifecycle_out.get('g_level', self._s2_g_level),
            dtype=torch.float32, device=env.device)
        self._s2_g_level = g_level
        # Command chain: the S2Lifecycle is the only source of truth for the
        # request, ramp, first zero and HOLD.  Never recompute the stop timeline
        # here (DSH-Q036; the old duplicate 400/1000 logic is gone).
        out = self._lifecycle_out
        # DSHS-Q037.2: the reward is settled with the phase/command that this
        # control step actually consumed (``*_at_actor``), never with the new
        # step-end command that is only consumed next tick.  One source: the
        # same S2Lifecycle update that produced success/failure.
        c_request = torch.as_tensor(out['c_request_at_actor'],
                                    dtype=torch.float32, device=env.device)
        c_exec = torch.as_tensor(out['c_exec_at_actor'],
                                 dtype=torch.float32, device=env.device)
        self._s2_c_request = torch.as_tensor(out['c_request'], dtype=torch.float32,
                                             device=env.device)
        self._s2_c_exec = torch.as_tensor(out['c_exec'], dtype=torch.float32,
                                          device=env.device)
        phase_names = list(out['phase'])
        top_now = torch.as_tensor(out['top_event'], dtype=torch.bool,
                                  device=env.device)
        success_now = torch.as_tensor(out['success_event'], dtype=torch.bool,
                                      device=env.device)
        self._s2_top_event = top_now
        # air-time accumulated from filtered contacts (reward-only state)
        self._s2_contacted |= contact
        landing = contact & ~self._s2_prev_contact & self._s2_contacted & (self._s2_air_time > 0)
        air_bonus = (landing.float() * (self._s2_air_time - 0.10)).sum(-1)
        self._s2_air_time = torch.where(
            contact, torch.zeros_like(self._s2_air_time),
            torch.where(self._s2_contacted, self._s2_air_time + 0.02,
                        self._s2_air_time))
        self._s2_prev_contact = contact
        # target clearance by real surface and per-row h; platform/ground 0.075.
        on_platform = feet_local[:, :, 0] >= self._s2_front[:, None]
        targets = torch.where(
            on_platform, torch.full_like(foot_h, 0.075),
            torch.where(self._s2_stairs[:, None],
                        torch.maximum(torch.full_like(foot_h, 0.17),
                                      self._s2_h[:, None] + 0.02),
                        torch.full_like(foot_h, 0.075)))
        qdot = data.joint_vel[:, env.joint_ids]
        route_error = env.route_error()
        yaw_error = torch.atan2(route_error[:, 1], route_error[:, 2])
        stable_stand = (np.asarray([p == 'HOLD' for p in phase_names], dtype=bool)
                        & (c_exec.detach().cpu().numpy() <= 1e-7))
        frame = dict(
            c_request=c_request.detach().cpu().numpy(),
            c_exec=c_exec.detach().cpu().numpy(),
            phase=phase_names,
            gravity_b=data.projected_gravity_b.detach().cpu().numpy(),
            velocity_b=data.root_link_lin_vel_b.detach().cpu().numpy(),
            omega_b=data.root_link_ang_vel_b.detach().cpu().numpy(),
            joint_position=data.joint_pos[:, env.joint_ids].detach().cpu().numpy(),
            default_joint_position=data.default_joint_pos[:, env.joint_ids].detach().cpu().numpy(),
            joint_velocity=qdot.detach().cpu().numpy(),
            joint_velocity_prev=self._s2_prev_qdot.detach().cpu().numpy(),
            action=env.action_manager.action.detach().cpu().numpy(),
            previous_action=previous_action.detach().cpu().numpy(),
            q_target_applied=env.action_term._processed_actions.detach().cpu().numpy(),
            contact_filtered=contact.float().detach().cpu().numpy(),
            foot_velocity_xy=foot_vel.detach().cpu().numpy(),
            foot_height=foot_h.detach().cpu().numpy(),
            target_clearance=targets.detach().cpu().numpy(),
            g_level=g_level.detach().cpu().numpy(),
            base_height=data.root_link_pos_w[:, 2].detach().cpu().numpy(),
            surface_height=surface_z.detach().cpu().numpy(),
            air_time_reward=air_bonus.detach().cpu().numpy(),
            local_y=local[:, 1].detach().cpu().numpy(),
            yaw_error=yaw_error.detach().cpu().numpy(),
            joint_limit_violation=torch.zeros(n, device=env.device).cpu().numpy(),
            v_heading=data.root_link_lin_vel_w[:, 0].detach().cpu().numpy(),
            dx=(local[:, 0] - env._local_x_prev).detach().cpu().numpy(),
            failure=failure.detach().cpu().numpy(),
            timeout=state.timeout.detach().cpu().numpy(),
            top_once=(top_now & ~failure).detach().cpu().numpy(),
            success_once=success_now.detach().cpu().numpy(),
            stable_stand=stable_stand)
        result = s2_rewards.compute_reward_batch(frame)
        self.reward_term_values = dict(result['terms'])
        totals = torch.as_tensor(result['total'], dtype=torch.float32,
                                 device=env.device)
        self._s2_prev_qdot = qdot.detach().clone()
        # The lifecycle already set env.command to c_exec for the next Actor
        # tick; never overwrite the command timeline from the reward path.
        self.step_reward_tensors = {'reward_total': totals}
        self.reward_terms = {'total': float(totals.mean().detach().cpu())}
        return totals

    def __getattr__(self, name):
        return getattr(self.env, name)

    def clean_flags(self):
        """§5.4 five flags per env: ordinary flat x nominal x no noise/delay/force."""
        tasks = [str(row.get('task', '')) for row in self.plan['rows']]
        n = len(tasks)
        # v07 L-phase has no noise, action delay or external force randomisation
        # (s2_physics._s2_patch_context installs the terrain only), so the three
        # nominal flags are True by construction; only ordinary_flat is selective.
        return dict(ordinary_flat=[task == 'flat_walk' for task in tasks],
                    nominal_dynamics=[True] * n, no_noise=[True] * n,
                    no_delay=[True] * n, no_force=[True] * n)

    def _update_teacher_map(self):
        from src.tasks.velocity.v07.events import euler_heading
        data = self.env.robot.data
        yaw = euler_heading(data.root_link_quat_w)[2]
        self._teacher_map.update(data.root_link_pos_w, yaw, self.env.tile_ids)

    def teacher_observation(self):
        self._update_teacher_map()
        raw = self.env._observations(update=False)
        return dict(proprio=raw['proprio'].clone(),
                    history=raw['history'].clone(),
                    heightmap=self._teacher_map.map.clone(),
                    route_error=raw['route_error'].clone(),
                    privileged=raw['privileged'].clone())

    def reset(self):
        obs, _info = self.env.reset()
        self._update_teacher_map()
        return obs

    def step(self, actions):
        import torch
        obs, reward, terminated, truncated, info = self.env.step(actions)
        info = dict(info)
        failure = info.get('failure', False)
        timeout = info.get('timeout', False)
        failure = bool(failure.any()) if hasattr(failure, 'any') else bool(failure)
        timeout = bool(timeout.any()) if hasattr(timeout, 'any') else bool(timeout)
        failure_mask = info.get('failure', False)
        timeout_mask = info.get('timeout', False)
        n = int(self.num_envs)
        quality_out = getattr(self, '_quality_out', None) or {}
        quality_ok = torch.as_tensor(
            quality_out.get('quality_ok', np.ones(n, dtype=bool)),
            dtype=torch.bool, device=self.device)
        quality_failure_mask = torch.as_tensor(
            quality_out.get('quality_failure', np.zeros(n, dtype=bool)),
            dtype=torch.bool, device=self.device)
        quality_reason = list(quality_out.get('quality_reason',
                                              [''] * n))
        lifecycle_out = getattr(self, '_lifecycle_out', None) or {}
        physical = lifecycle_out.get('physical_failure')
        physical_mask = (torch.as_tensor(physical, dtype=torch.bool,
                                         device=self.device)
                         if physical is not None else failure_mask)
        failure_class_mask = None
        termination = getattr(self.env, 'last_termination', None)
        if termination is not None and len(termination) >= 4:
            from src.tasks.velocity.v07.events import FAILURE_CODES
            names = {code: name for name, code in FAILURE_CODES.items()}
            reasons = termination[3].detach().cpu().tolist()
            physical_list = physical_mask.detach().cpu().tolist() \
                if hasattr(physical_mask, 'detach') else list(physical_mask or [])
            timeout_list = timeout_mask.detach().cpu().tolist() \
                if hasattr(timeout_mask, 'detach') else list(timeout_mask or [])
            quality_list = quality_failure_mask.detach().cpu().tolist() \
                if hasattr(quality_failure_mask, 'detach') else []
            ok = terminated.detach().cpu().tolist() \
                if hasattr(terminated, 'detach') else list(terminated or [])
            failure_class_mask = []
            for i, code in enumerate(reasons):
                if i < len(physical_list) and physical_list[i]:
                    failure_class_mask.append(names.get(code, f'v07_code_{code}'))
                elif i < len(quality_list) and quality_list[i]:
                    failure_class_mask.append(
                        's2_quality_failure:' + str(quality_reason[i]))
                elif i < len(timeout_list) and timeout_list[i]:
                    failure_class_mask.append('timeout')
                elif i < len(ok) and ok[i]:
                    failure_class_mask.append('s2_success')
                else:
                    failure_class_mask.append(None)
        classes = failure_class_mask or []
        if any(str(v).startswith('s2_quality_failure') for v in classes):
            failure_class = 'quality_failure'
        elif any(v == 'timeout' for v in classes):
            failure_class = 'timeout'
        elif any(v == 's2_success' for v in classes):
            failure_class = 'success'
        elif failure:
            failure_class = 'failure'
        elif timeout:
            failure_class = 'timeout'
        else:
            failure_class = None
        info.update(
            terminated=bool(terminated.any()) if hasattr(terminated, 'any') else bool(terminated),
            truncated=bool(truncated.any()) if hasattr(truncated, 'any') else bool(truncated),
            terminated_mask=terminated.detach().clone() if hasattr(terminated, 'detach') else terminated,
            truncated_mask=truncated.detach().clone() if hasattr(truncated, 'detach') else truncated,
            failure_mask=failure_mask,
            timeout_mask=timeout_mask,
            physical_failure_mask=physical_mask,
            quality_ok=quality_ok,
            quality_failure_mask=quality_failure_mask,
            quality_reason=list(quality_reason),
            quality_cruise_state=list(quality_out.get(
                'cruise_state', [''] * n)),
            quality_stop_state=list(quality_out.get('stop_state', [''] * n)),
            cruise_entry_step=quality_out.get('four_feet_entry_step'),
            cruise_start_step=quality_out.get('cruise_start_step'),
            cruise_samples=quality_out.get('cruise_samples'),
            hold_samples=quality_out.get('hold_samples'),
            stop_mean_speed=quality_out.get('stop_mean_speed'),
            stop_peak_speed=quality_out.get('stop_peak_speed'),
            stop_max_displacement=quality_out.get('stop_max_displacement'),
            stop_support_ratio=quality_out.get('stop_support_ratio'),
            failure_class=failure_class,
            failure_class_mask=failure_class_mask,
            teacher_observation=self.teacher_observation(),
            a_request=self.env.action_manager.action.detach().clone(),
            a_applied=self.env.action_term._processed_actions.detach().clone(),
            phase=getattr(self, '_s2_phase', None),
            c_request=getattr(self, '_s2_c_request', None),
            c_exec=getattr(self, '_s2_c_exec', None),
            c_exec_consumed=self.env.sample_command[:, 0].detach().clone(),
            top_arrived=(getattr(self, '_lifecycle_out', {}) or {}).get('top'),
            top_event=(getattr(self, '_lifecycle_out', {}) or {}).get('top_event'),
            local_x=(self.env.robot.data.root_link_pos_w
                     - self.env.origins[self.env.tile_ids])[:, 0],
            g_level=(getattr(self, '_s2_g_level', None)),
            episode_length=self.env.episode_length_buf.detach().clone())
        return obs, reward, terminated | truncated, info

    def close(self):
        self._patch.__exit__(None, None, None)


def training_contract_report():
    return dict(
        observation=OBSERVATION_CONTRACT, map_refresh_steps=MAP_REFRESH_STEPS,
        control_dt=CONTROL_DT, physical=s2_physics.assembly_contract(),
        note='static contract only; simulator construction is separate')
