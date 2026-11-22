"""S2.1 real multi-env training assembly seam (CPU contract, GPU launched later).

The 1024-env environment is a V07Env-derived class with S2 terrain/map patches.
The static plan is CPU-testable; constructing the simulator is deliberately
separate from the plan.
"""
from __future__ import annotations

import copy

from . import s2_physics, s2_spec, s2_terrain, s2_train_env


OBSERVATION_CONTRACT = dict(
    proprio=(45,), history=(5, 45), heightmap=(3, 40, 30), route_error=(3,),
    privileged=(35,), critic=211)
MAP_REFRESH_STEPS = 5
CONTROL_DT = 0.02


def _geometry_for(row):
    if row['role'] == 'flat_walk' or row['role'] == 'flat_stop':
        return dict(kind='flat', terrain_key='s2_flat_v07_tile0', tile_id=0,
                    lesson_id='FLAT', N=0, h=0.0, step_depth=0.2, width=8.0,
                    tread_thickness=0.01, platform_length=0.0,
                    platform_width=8.0, platform_present=False,
                    platform_height=0.0, platform_front_local_x=None,
                    frame_origin_world=[0.0, 0.0, 0.0],
                    valid_bounds_local=[-2.0, 22.0, -4.0, 4.0])
    if row['role'] == 'stairs_walk':
        levels = int(row['stair_count'])
        width = float(row['platform_width'])
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
    if row['role'] == 'platform_walk':
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
    # legacy L40 review
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


def build_env_plan(num_envs=1024):
    role_plan = s2_train_env.build_role_plan(num_envs)
    specs = {spec.terrain_key: spec for spec in s2_terrain.terrain_specs()}
    rows = []
    for index, role in enumerate(role_plan['envs']):
        geometry = _geometry_for(role)
        # use the frozen S2 table entry where present
        key = geometry['terrain_key']
        spec = specs.get(key)
        if spec is None:
            spec = s2_terrain.spec_from_geometry(geometry)
            geometry['tile_id'] = spec.tile_id
        else:
            geometry['tile_id'] = spec.tile_id
        front = geometry.get('platform_front_local_x')
        if role['role'] == 'platform_walk':
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
    return dict(num_envs=num_envs, role_counts=role_plan['counts'],
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

    def __init__(self, *, num_envs=1024, device='cuda:0', seed=42):
        import torch
        from src.tasks.velocity.v07.env import V07Env
        from .map import HeightMap
        self.plan = build_env_plan(num_envs)
        self.num_envs = int(num_envs)
        self.role_plan = dict(num_envs=self.num_envs,
                              counts=dict(self.plan['role_counts']),
                              order=list(s2_train_env.ROLE_ORDER))
        self.device = device
        self.seed = int(seed)
        self._tile_by_env = [row['tile_id'] for row in self.plan['rows']]
        self._role_by_env = [row['role'] for row in self.plan['rows']]
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
            self.env.reset()
            assert_v07_contract(self.env)
        except Exception:
            self._patch.__exit__(None, None, None)
            raise

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs, _info = self.env.reset()
        return obs

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return obs, reward, terminated | truncated, info

    def close(self):
        self._patch.__exit__(None, None, None)


def training_contract_report():
    return dict(
        observation=OBSERVATION_CONTRACT, map_refresh_steps=MAP_REFRESH_STEPS,
        control_dt=CONTROL_DT, physical=s2_physics.assembly_contract(),
        note='static contract only; simulator construction is separate')
