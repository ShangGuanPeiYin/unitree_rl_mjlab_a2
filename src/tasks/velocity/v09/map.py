"""Native grids sampled from real geometry; no interpolation between grids.

The map holds its sample frame between 10 Hz refreshes. Valid age increases
with elapsed control time, while invalid height/valid/age are always 0/0/1.
"""
from __future__ import annotations

import torch

from . import spec

MAP_SHAPE = spec.MAP_SHAPE
CONTROL_DT = spec.CONTROL_DT
MAP_HZ = 10
MAP_REFRESH_STEPS = 5
GRID_STEP = 0.05
GRID_X = (-0.8, 1.2)
GRID_Y = (-0.75, 0.75)
HEIGHT_SCALE = 1.5


def query_surface(points, tile_ids):
    """Use the unchanged geometry/collision source, imported only when needed."""
    from src.tasks.velocity.v07.terrain import query_surface as query
    return query(points, tile_ids)


class _SampleFrameMap:
    def __init__(self, num_envs, device, shape, x_min, y_min, surface_query):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError('num_envs must be positive')
        self.device = torch.device(device)
        self.surface_query = surface_query or query_surface
        _, nx, ny = shape
        x = x_min + (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * GRID_STEP
        y = y_min + (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * GRID_STEP
        self.grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.map = torch.zeros(self.num_envs, *shape, device=device)
        self.map[:, 2] = 1
        self.steps = torch.full((self.num_envs,), MAP_REFRESH_STEPS, dtype=torch.long, device=device)
        self.sample_position = torch.zeros(self.num_envs, 3, device=device)
        self.sample_yaw = torch.zeros(self.num_envs, device=device)
        self.sample_tile = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.sample_points_world = torch.zeros(self.num_envs, nx, ny, 2, device=device)

    def _ids(self, env_ids):
        return (torch.arange(self.num_envs, device=self.device) if env_ids is None
                else torch.as_tensor(env_ids, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def update(self, base_pos, yaw, tile_ids, env_ids=None, force=False):
        ids = self._ids(env_ids)
        if not ids.numel():
            return self.map
        if not force:
            self.steps[ids] += 1
            self.map[ids, 2] = torch.where(
                self.map[ids, 1].bool(),
                (self.steps[ids, None, None] * CONTROL_DT / 0.5).clamp(0, 1), 1.0)
        refresh = ids if force else ids[self.steps[ids] >= MAP_REFRESH_STEPS]
        if refresh.numel():
            pos, angle = base_pos[refresh], yaw[refresh]
            c, s = angle.cos()[:, None, None], angle.sin()[:, None, None]
            gx, gy = self.grid[..., 0], self.grid[..., 1]
            points = torch.stack((pos[:, None, None, 0] + c * gx - s * gy,
                                  pos[:, None, None, 1] + s * gx + c * gy), -1)
            z, valid = self.surface_query(points, tile_ids[refresh])
            if z.shape != points.shape[:-1] or valid.shape != z.shape or valid.dtype != torch.bool:
                raise ValueError('surface query must return matching height and boolean validity grids')
            if not torch.isfinite(z[valid]).all():
                raise ValueError('non-finite valid surface height')
            self.map[refresh, 0] = torch.where(
                valid, ((z - pos[:, None, None, 2]) / HEIGHT_SCALE).clamp(-1, 1), 0.0)
            self.map[refresh, 1] = valid.float()
            self.map[refresh, 2] = (~valid).float()
            self.steps[refresh] = 0
            self.sample_position[refresh] = pos
            self.sample_yaw[refresh] = angle
            self.sample_tile[refresh] = tile_ids[refresh]
            self.sample_points_world[refresh] = points
        return self.map

    def reset(self, base_pos, yaw, tile_ids, env_ids=None):
        ids = self._ids(env_ids)
        self.steps[ids] = MAP_REFRESH_STEPS
        self.map[ids] = 0
        self.map[ids, 2] = 1
        return self.update(base_pos, yaw, tile_ids, ids, force=True)


class HeightMap(_SampleFrameMap):
    """Student's native 3x40x30 cache in the current true body/yaw frame."""
    def __init__(self, num_envs, device='cpu', surface_query=None):
        super().__init__(num_envs, device, MAP_SHAPE, GRID_X[0], GRID_Y[0], surface_query)


class PairedHeightMaps:
    """Two real queries of one physical snapshot, sharing refresh/reset timing."""
    def __init__(self, num_envs, device='cpu', surface_query=None):
        self.teacher = _SampleFrameMap(num_envs, device, (3, 48, 32), -0.8, -0.8, surface_query)
        self.student = HeightMap(num_envs, device, surface_query)

    def _apply(self, method, base_pos, yaw, tile_ids, env_ids, **kwargs):
        # Copy once, before either query: both grids really see the same state.
        snapshot = (base_pos.detach().clone(), yaw.detach().clone(), tile_ids.detach().clone())
        return tuple(getattr(cache, method)(*snapshot, env_ids=env_ids, **kwargs)
                     for cache in (self.teacher, self.student))

    def update(self, base_pos, yaw, tile_ids, env_ids=None, force=False):
        return self._apply('update', base_pos, yaw, tile_ids, env_ids, force=force)

    def reset(self, base_pos, yaw, tile_ids, env_ids=None):
        return self._apply('reset', base_pos, yaw, tile_ids, env_ids)

    def observations(self, shared):
        """Use one executed proprio/history/route/privileged trajectory for both."""
        required = ('proprio', 'history', 'route_error', 'privileged')
        if any(key not in shared for key in required):
            raise ValueError('paired observations require the complete shared physical snapshot')
        return tuple(dict(shared, heightmap=cache.map) for cache in (self.teacher, self.student))


class ProprioHistory:
    """Unchanged five-frame history of the single actual executed trajectory."""
    def __init__(self, num_envs, device='cpu'):
        self.history = torch.zeros(num_envs, 5, 45, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, observation):
        fresh = ~self.initialized
        self.history = torch.cat((self.history[:, 1:], observation[:, None]), 1)
        self.history[fresh] = observation[fresh, None].expand(-1, 5, -1)
        self.initialized[:] = True
        return self.history.flatten(1)

    def reset(self, observation, env_ids):
        ids = torch.as_tensor(env_ids, device=self.history.device, dtype=torch.long)
        self.history[ids] = observation[ids, None].expand(-1, 5, -1)
        self.initialized[ids] = True
        return self.history.flatten(1)
