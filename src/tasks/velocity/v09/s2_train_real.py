"""S2.1 real multi-env training assembly seam (CPU contract, GPU launched later).

The 1024-env environment is a V07Env-derived class with S2 terrain/map patches.
The static plan is CPU-testable; constructing the simulator is deliberately
separate from the plan.
"""
from __future__ import annotations

import copy

from . import s2_lifecycle, s2_physics, s2_rewards, s2_spec, s2_terrain, s2_train_env
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
    specs = {spec.terrain_key: spec for spec in s2_terrain.terrain_specs()}
    rows = []
    for index, role in enumerate(role_plan['envs']):
        geometry = _geometry_for(role, course=course)
        # use the frozen S2 table entry where present
        key = geometry['terrain_key']
        spec = specs.get(key)
        if spec is None:
            spec = s2_terrain.spec_from_geometry(geometry)
            geometry['tile_id'] = spec.tile_id
        else:
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
            self.env._task_state = _s2_task_state
            def _s2_reward(_env, state, previous_action):
                return self._s2_reward(_env, state, previous_action)
            self.env._reward = _s2_reward
            self.env.reset()
            self._teacher_map = _SampleFrameMap(
                self.num_envs, device, (3, 48, 32), -0.8, -0.8,
                s2_terrain.query_surface)
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
            self._s2_steps = z(n, dtype=torch.long)
        if not hasattr(self, '_lifecycle'):
            rows = self.plan['rows']
            tasks = [str(r.get('task') or r.get('role')) for r in rows]
            self._lifecycle = s2_lifecycle.S2Lifecycle(
                task=tasks,
                post_distance=[float(r.get('post_distance') or 5.0) for r in rows],
                speed=[float(r.get('speed') or 0.5) for r in rows],
                stop_mode=[str(r.get('stop_mode') or 'none') for r in rows],
                legacy=[str(t) == 'legacy_l40' for t in tasks],
                front=[float(r['geometry'].get('platform_front_local_x') or 0.0)
                       for r in rows],
                spawn_x=[float(r['spawn'][0]) for r in rows],
                completion_steps=[1001 if t == 'flat_walk' else
                                  (1401 if t == 'flat_stop' else 1000)
                                  for t in tasks])
        first = env.episode_length_buf <= 1
        self._lifecycle.reset(first.detach().cpu().numpy())
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
        self._s2_steps[first] = 0
        self._s2_c_request[first] = self._s2_speed[first]
        self._s2_c_exec[first] = self._s2_speed[first]
        original_success = state.success.clone()
        local = state.local[:, 0]
        failure = state.failure.clone()
        timeout = state.timeout.clone()
        support = env.climb.current_support.bool()
        four_feet = support.all(-1)
        body_ok = local > self._s2_front + 0.4
        quality_ok = torch.ones(n, dtype=torch.bool, device=env.device)
        out = self._lifecycle.update(
            local_x=local.detach().cpu().numpy(),
            top_signal=original_success.detach().cpu().numpy(),
            four_feet=four_feet.detach().cpu().numpy(),
            body_ok=body_ok.detach().cpu().numpy(),
            quality_ok=quality_ok.detach().cpu().numpy(),
            failure=failure.detach().cpu().numpy(),
            timeout=timeout.detach().cpu().numpy())
        self._lifecycle_out = out
        self._s2_phase = out['phase']
        self._s2_c_request = torch.as_tensor(out['c_request'], dtype=torch.float32,
                                             device=env.device)
        self._s2_c_exec = torch.as_tensor(out['c_exec'], dtype=torch.float32,
                                          device=env.device)
        self._s2_top_event = torch.as_tensor(out['top_event'], dtype=torch.bool,
                                             device=env.device)
        state.success = torch.as_tensor(out['success'], dtype=torch.bool,
                                        device=env.device)
        state.failure = torch.as_tensor(out['failure'], dtype=torch.bool,
                                        device=env.device)
        state.timeout = torch.as_tensor(out['timeout'], dtype=torch.bool,
                                        device=env.device)
        env.command[:, 0] = self._s2_c_exec.detach()
        return state
        return state

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
        if self._s2_top_latched is None:
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
        surface_z, surface_valid = s2_terrain.query_surface(
            data.root_link_pos_w[:, :2], env.tile_ids)
        surface_z = torch.where(surface_valid, surface_z,
                                torch.zeros_like(surface_z))
        foot_h, _ = env._foot_heights()
        contact = env.contacts.filtered.bool()
        foot_vel = data.site_lin_vel_w[:, env.foot_ids, :2]
        failure = state.failure | state.timeout
        # §5.3/§4: g_level t0 from four-foot support + body margin; 0.5 s ramp.
        support = env.climb.current_support.bool()
        g_ready = (support.all(-1) & (local[:, 0] > self._s2_front + 0.4)
                   & ~failure)
        t0 = self._s2_g_step
        t0 = torch.where(g_ready & (t0 < 0), self._s2_steps, t0)
        t0 = torch.where(~g_ready, torch.full_like(t0, -1), t0)
        self._s2_g_step = t0
        g_level = torch.where(
            t0 < 0, torch.zeros_like(self._s2_g_level),
            ((self._s2_steps - t0).float() / 25.0).clamp(0.0, 1.0))
        self._s2_g_level = g_level
        # Command chain: request is external speed; c_exec is the post-ramp
        # command actually consumed next. S2.1 has no stop.
        speed = self._s2_c_request
        dpost = local[:, 0] - self._s2_front
        request_stop = self._s2_stop & ~self._s2_stop_requested & (dpost >= self._s2_post) & ~failure
        self._s2_stop_requested |= request_stop
        self._s2_stop_step = torch.where(
            request_stop, self._s2_steps, self._s2_stop_step)
        instant_zero = self._s2_stop_requested & ~self._s2_ramp & (self._s2_hold_step < 0)
        ramp_elapsed = (self._s2_steps - self._s2_stop_step).clamp_min(0).float()
        ramp_exec = speed * (1.0 - ramp_elapsed / 100.0).clamp(0.0, 1.0)
        c_exec = torch.where(self._s2_stop_requested,
                             torch.where(instant_zero, torch.zeros_like(speed), ramp_exec),
                             speed)
        c_exec = torch.where(c_exec <= 1e-12, torch.zeros_like(c_exec), c_exec)
        self._s2_c_exec = c_exec
        first_zero = self._s2_stop_requested & (self._s2_hold_step < 0) & (c_exec <= 1e-12)
        self._s2_hold_step = torch.where(first_zero, self._s2_steps,
                                         self._s2_hold_step)
        phase = torch.where(
            self._s2_stop_requested & (c_exec > 1e-7), torch.ones_like(self._s2_steps),
            torch.where(self._s2_hold_step >= 0, torch.full_like(self._s2_steps, 2),
                        torch.zeros_like(self._s2_steps)))
        # phase codes: 0 APPROACH/CLIMB/POST_WALK, 1 DECEL, 2 HOLD
        # top comes from the v07 climb event; success from distance/stop contract.
        top_now = self._s2_top_event
        if self._s2_stop.any():
            hold_ok = ((self._s2_hold_step >= 0)
                       & ((self._s2_steps - self._s2_hold_step) >= 1000))
            success_now = self._s2_stop & hold_ok
        else:
            success_now = ((~self._s2_stop) & state.success
                           & ~self._s2_success_latched)
        success_now &= ~self._s2_success_latched & ~failure
        self._s2_success_latched |= success_now
        # air-time accumulated from filtered contacts
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
        phase_names = ['DECEL' if p.item() == 1 else 'HOLD' if p.item() == 2
                       else 'POST_WALK' for p in phase]
        frame = dict(
            c_request=speed.detach().cpu().numpy(),
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
            stable_stand=((phase > 0) & (c_exec <= 1e-7)).detach().cpu().numpy())
        result = s2_rewards.compute_reward_batch(frame)
        totals = torch.as_tensor(result['total'], dtype=torch.float32,
                                 device=env.device)
        self._s2_prev_qdot = qdot.detach().clone()
        self._s2_steps = self._s2_steps + 1
        # Actor consumes c_exec on the next control step.
        env.command[:, 0] = c_exec.detach()
        self.step_reward_tensors = {'reward_total': totals}
        self.reward_terms = {'total': float(totals.mean().detach().cpu())}
        return totals

    def __getattr__(self, name):
        return getattr(self.env, name)

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
        obs, reward, terminated, truncated, info = self.env.step(actions)
        info = dict(info)
        failure = info.get('failure', False)
        timeout = info.get('timeout', False)
        failure = bool(failure.any()) if hasattr(failure, 'any') else bool(failure)
        timeout = bool(timeout.any()) if hasattr(timeout, 'any') else bool(timeout)
        failure_mask = info.get('failure', False)
        timeout_mask = info.get('timeout', False)
        info.update(
            terminated=bool(terminated.any()) if hasattr(terminated, 'any') else bool(terminated),
            truncated=bool(truncated.any()) if hasattr(truncated, 'any') else bool(truncated),
            terminated_mask=terminated.detach().clone() if hasattr(terminated, 'detach') else terminated,
            truncated_mask=truncated.detach().clone() if hasattr(truncated, 'detach') else truncated,
            failure_mask=failure_mask,
            timeout_mask=timeout_mask,
            failure_class='failure' if failure else 'timeout' if timeout else None,
            teacher_observation=self.teacher_observation(),
            a_request=self.env.action_manager.action.detach().clone(),
            a_applied=self.env.action_term._processed_actions.detach().clone(),
            phase=getattr(self, '_s2_phase', None),
            c_request=getattr(self, '_s2_c_request', None),
            c_exec=getattr(self, '_s2_c_exec', None),
            top_arrived=(getattr(self, '_lifecycle_out', {}) or {}).get('top'),
            local_x=(self.env.robot.data.root_link_pos_w
                     - self.env.origins[self.env.tile_ids])[:, 0])
        return obs, reward, terminated | truncated, info

    def close(self):
        self._patch.__exit__(None, None, None)


def training_contract_report():
    return dict(
        observation=OBSERVATION_CONTRACT, map_refresh_steps=MAP_REFRESH_STEPS,
        control_dt=CONTROL_DT, physical=s2_physics.assembly_contract(),
        note='static contract only; simulator construction is separate')
