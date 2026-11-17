"""v0.7 ideal 3x48x32 height map sampled at 10 Hz from the real v0.7 terrain.

SDS v0.7 section 2.3: ``heightmap[3,48,32]`` updates at ``10 Hz`` (one refresh
every 5 control steps at ``dt = 0.02 s``).  Every sample height comes from
:func:`src.tasks.velocity.v07.terrain.query_surface`, i.e. the *same*
``TerrainSpec`` table that builds the collision boxes; there is no implicit
v0.2 atlas, no stage table and no height-field quantisation here.

Channels (identical semantics to the frozen observation contract):

* ``0``: ``(surface_z - base_z) / 1.5`` clipped to ``[-1, 1]`` (0 outside);
* ``1``: validity (1 where the sample lies inside the tile's real bounds);
* ``2``: ageing channel -- 1 while a sample was invalid at the last refresh and
  decays by ``dt / 0.5`` per control step until the next refresh.
"""
from __future__ import annotations

import torch

from . import spec as _spec
from .terrain import query_surface

MAP_HZ = int(getattr(_spec, "MAP_HZ", 10))
CONTROL_DT = float(_spec.CONTROL_DT)
#: 10 Hz on a 50 Hz control loop -> one refresh every 5 control samples.
MAP_REFRESH_STEPS = int(round(1.0 / (CONTROL_DT * MAP_HZ)))
MAP_SHAPE = tuple(int(v) for v in _spec.HEIGHTMAP_SHAPE)
HISTORY_LEN = int(getattr(_spec, "HISTORY_LEN", 5))
PROPRIO_DIM = int(_spec.PROPRIO_DIM)
GRID_STEP = 0.05
GRID_X = (-0.8, 0.8)
GRID_Y = (-0.8, 0.8)
HEIGHT_SCALE = 1.5


class HeightMap:
    """Batched ideal map cache with sample-frame hold (no motion compensation)."""

    def __init__(self, num_envs: int, device="cpu"):
        if tuple(MAP_SHAPE) != (3, 48, 32):
            raise RuntimeError(f"v07 height map shape must be (3,48,32), got {MAP_SHAPE}")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        x = GRID_X[0] + (torch.arange(48, device=device, dtype=torch.float32) + 0.5) * GRID_STEP
        y = GRID_Y[0] + (torch.arange(32, device=device, dtype=torch.float32) + 0.5) * GRID_STEP
        self.grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)
        self.map = torch.zeros(self.num_envs, 3, 48, 32, device=device)
        self.map[:, 2] = 1.0
        self.steps = torch.full((self.num_envs,), MAP_REFRESH_STEPS, device=device,
                                dtype=torch.long)
        self.sample_position = torch.zeros(self.num_envs, 3, device=device)
        self.sample_yaw = torch.zeros(self.num_envs, device=device)
        self.sample_tile = torch.full((self.num_envs,), -1, dtype=torch.long,
                                      device=device)
        self.sample_points_world = torch.zeros(self.num_envs, 48, 32, 2, device=device)

    def update(self, base_pos, yaw, tile_ids, env_ids=None, force=False):
        ids = (torch.arange(self.num_envs, device=self.device)
               if env_ids is None else torch.as_tensor(env_ids, device=self.device,
                                                       dtype=torch.long))
        if ids.numel() == 0:
            return self.map
        if not force:
            self.steps[ids] += 1
            valid = self.map[ids, 1].bool()
            self.map[ids, 2] = torch.where(
                valid, (self.steps[ids, None, None] * CONTROL_DT / 0.5).clamp(0, 1), 1.0)
        refresh = ids if force else ids[self.steps[ids] >= MAP_REFRESH_STEPS]
        if len(refresh):
            pos, angle = base_pos[refresh], yaw[refresh]
            c, s = angle.cos()[:, None, None], angle.sin()[:, None, None]
            gx, gy = self.grid[..., 0], self.grid[..., 1]
            points = torch.stack((pos[:, None, None, 0] + c * gx - s * gy,
                                  pos[:, None, None, 1] + s * gx + c * gy), -1)
            z, valid = query_surface(points, tile_ids[refresh])
            self.map[refresh, 0] = torch.where(
                valid, ((z - pos[:, None, None, 2]) / HEIGHT_SCALE).clamp(-1, 1), 0.0)
            self.map[refresh, 1] = valid.to(torch.float32)
            self.map[refresh, 2] = (~valid).to(torch.float32)
            self.steps[refresh] = 0
            self.sample_position[refresh] = pos
            self.sample_yaw[refresh] = angle
            self.sample_tile[refresh] = tile_ids[refresh]
            self.sample_points_world[refresh] = points
        return self.map

    def reset(self, base_pos, yaw, tile_ids, env_ids=None):
        ids = (torch.arange(self.num_envs, device=self.device)
               if env_ids is None else env_ids)
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self.steps[ids] = MAP_REFRESH_STEPS
        self.map[ids] = 0.0
        self.map[ids, 2] = 1.0
        return self.update(base_pos, yaw, tile_ids, ids, force=True)


class ProprioHistory:
    """FIFO history of ``HISTORY_LEN`` proprioceptive samples (reset per subset)."""

    def __init__(self, num_envs: int, device="cpu"):
        self.history = torch.zeros(num_envs, HISTORY_LEN, PROPRIO_DIM, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, observation):
        fresh = ~self.initialized
        self.history = torch.cat((self.history[:, 1:], observation[:, None]), 1)
        self.history[fresh] = observation[fresh, None].expand(-1, HISTORY_LEN, -1)
        self.initialized[:] = True
        return self.history.flatten(1)

    def reset(self, observation, env_ids):
        ids = torch.as_tensor(env_ids, device=self.history.device, dtype=torch.long)
        self.history[ids] = observation[ids, None].expand(-1, HISTORY_LEN, -1)
        self.initialized[ids] = True
        return self.history.flatten(1)
