"""Open-riser stairs made from independent, collidable thin boxes.

SubTerrainCfg follows mjlab's tile-local convention. build_open_stairs_spec
recenters a single tile so the first tread's front edge is x=0, y=0.
No policy, reward, curriculum, or sensing configuration is defined here.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainGenerator, TerrainGeneratorCfg,
    TerrainGeometry, TerrainOutput,
)


@dataclass(kw_only=True)
class OpenStairsTerrainCfg(SubTerrainCfg):
    step_count: int = 40
    step_height: float = 0.20
    step_depth: float = 0.20
    stair_width: float = 0.80
    tread_thickness: float = 0.01
    approach_length: float = 2.0
    landing_length: float = 2.0
    ground_width: float = 4.0
    ground_thickness: float = 0.10

    def __post_init__(self):
        if not isinstance(self.step_count, int) or isinstance(self.step_count, bool) or self.step_count < 1:
            raise ValueError('step_count must be a positive integer')
        for name in ('step_height', 'step_depth', 'stair_width', 'tread_thickness',
                     'approach_length', 'landing_length', 'ground_width', 'ground_thickness'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.tread_thickness >= self.step_height:
            raise ValueError('tread_thickness must leave an open riser')
        if self.ground_width < self.stair_width:
            raise ValueError('ground_width must cover stair_width')
        self.size = (self.approach_length + self.step_count * self.step_depth + self.landing_length,
                     self.ground_width)

    def function(self, difficulty, spec, rng):
        del difficulty, rng
        body = spec.body('terrain')
        geometries = []

        def box(name, center, dimensions, color):
            geom = body.add_geom(
                name=f'open_stairs_{len(list(body.geoms))}_{name}', type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=center, size=tuple(v / 2 for v in dimensions),
                rgba=color, contype=1, conaffinity=1, condim=3,
                friction=(0.6, 0.005, 0.0001),
            )
            geometries.append(TerrainGeometry(geom=geom, color=color))

        y = self.size[1] / 2
        box('ground', (self.size[0] / 2, y, -self.ground_thickness / 2),
            (self.size[0], self.ground_width, self.ground_thickness), (0.30, 0.33, 0.36, 1))
        for i in range(1, self.step_count + 1):
            color = (0.30, 0.55, 0.75, 1) if i % 2 else (0.42, 0.65, 0.82, 1)
            box(f'tread_{i:02d}',
                (self.approach_length + (i - 0.5) * self.step_depth, y,
                 i * self.step_height - self.tread_thickness / 2),
                (self.step_depth, self.stair_width, self.tread_thickness), color)
        box('top_landing',
            (self.approach_length + self.step_count * self.step_depth + self.landing_length / 2,
             y, self.step_count * self.step_height - self.tread_thickness / 2),
            (self.landing_length, self.stair_width, self.tread_thickness), (0.35, 0.70, 0.45, 1))
        return TerrainOutput(
            origin=np.array((self.approach_length / 2, y, 0.0)), geometries=geometries,
        )


def build_open_stairs_spec(cfg=None, *, seed=42):
    """Return a standalone MjSpec and spawn ground point, using native mjlab generation.

    For grid generation, use OpenStairsTerrainCfg directly as a sub_terrain;
    its returned origin is in the approach region. No border walls are added.
    """
    cfg = cfg or OpenStairsTerrainCfg()
    generator = TerrainGenerator(TerrainGeneratorCfg(
        seed=seed, size=cfg.size, num_rows=1, num_cols=1, curriculum=False,
        border_width=0.0, color_scheme='height', sub_terrains={'open_stairs': cfg},
        add_lights=True,
    ))
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    generator.compile(spec)
    shift = np.array((cfg.size[0] / 2 - cfg.approach_length, 0.0, 0.0))
    names = ['ground'] + [f'tread_{i:02d}' for i in range(1, cfg.step_count + 1)] + ['top_landing']
    for geom, name in zip(spec.body('terrain').geoms, names, strict=True):
        geom.pos = np.asarray(geom.pos) + shift
        geom.name = name
    spawn = generator.terrain_origins[0, 0] + shift
    return spec, spawn
