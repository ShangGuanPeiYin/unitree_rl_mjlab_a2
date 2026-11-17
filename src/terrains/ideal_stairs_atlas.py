"""Explicit static six-tile atlas for the ideal-map experiment only.

The generator overwrites subterrain size; every function below deliberately
uses the same 24 x 8 layout. Flat and stair frames are distinct contracts.
"""
from dataclasses import dataclass
import numpy as np
import mujoco
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from src.terrains.open_stairs import OpenStairsTerrainCfg

STAGES = ((0, 0.0), (4, .08), (8, .12), (12, .16), (20, .20), (40, .20))

@dataclass(frozen=True)
class TileMetadata:
    tile_id: int
    stage: int
    step_count: int
    step_height: float
    terrain_origin_world: tuple
    frame_origin_world: tuple
    valid_bounds_local: tuple
    step_depth: float = .20
    stair_width: float = .80
    tread_thickness: float = .01
    approach_length: float = 2.
    landing_length: float = 2.
    @property
    def stair_frame_origin_world(self):
        return self.frame_origin_world
    def world_to_local(self, points):
        if isinstance(points, torch.Tensor):
            return points - points.new_tensor(self.frame_origin_world[:points.shape[-1]])
        return np.asarray(points) - np.asarray(self.frame_origin_world[:np.asarray(points).shape[-1]])


def atlas_metadata():
    result = []
    for stage, (n, h) in enumerate(STAGES):
        y = -24. + stage * 8. + 4.
        frame = (-10., y, 0.)
        origin = frame if stage == 0 else (-11., y, 0.)
        bounds = (-2., 22., -4., 4.) if stage == 0 else (-2., n*.2+2., -2., 2.)
        result.append(TileMetadata(stage, stage, n, h, origin, frame, bounds))
    return tuple(result)

@dataclass(kw_only=True)
class AtlasFlatCfg(SubTerrainCfg):
    def function(self, difficulty, spec, rng):
        geom = spec.body('terrain').add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(12.,4.,-.05), size=(12.,4.,.05), contype=1, conaffinity=1,
            condim=3, friction=(.6,.005,.0001))
        return TerrainOutput(origin=np.array((2.,4.,0.)), geometries=[TerrainGeometry(geom=geom, color=(.3,.33,.36,1.))])

@dataclass(kw_only=True)
class AtlasStairsCfg(OpenStairsTerrainCfg):
    """Keep bottom support limited to actual approach/stairs/platform length."""
    def function(self, difficulty, spec, rng):
        output = super().function(difficulty, spec, rng)
        ground = output.geometries[0].geom
        length = self.approach_length+self.step_count*self.step_depth+self.landing_length
        ground.pos[0] = length/2
        ground.size[0] = length/2
        return output


def atlas_generator_cfg(seed=42):
    configs = {'S0': AtlasFlatCfg(proportion=1.)}
    for stage, (n,h) in enumerate(STAGES[1:],1):
        configs[f'S{stage}'] = AtlasStairsCfg(step_count=n, step_height=h, proportion=1.)
    return TerrainGeneratorCfg(seed=seed, size=(24.,8.), num_rows=1, num_cols=6,
        curriculum=True, border_width=0., sub_terrains=configs, color_scheme='height')


def allocate_stage_slots(num_envs, stage):
    """Largest remainder, deterministic lower-stage tie break; no RNG side effects."""
    if num_envs < 1 or not 0 <= stage <= 5:
        raise ValueError('positive num_envs and stage 0..5 required')
    weights = {0:1.} if stage == 0 else ({0:.1,1:.9} if stage == 1 else {0:.1,stage-1:.2,stage:.7})
    raw = {k:num_envs*v for k,v in weights.items()}
    counts = {k:int(np.floor(v)) for k,v in raw.items()}
    order = sorted(raw, key=lambda k:(-(raw[k]-counts[k]),k))
    for k in order[:num_envs-sum(counts.values())]: counts[k] += 1
    return np.concatenate([np.full(counts[k], k, dtype=np.int64) for k in sorted(counts)])


def query_top_surface(points_world, tile_ids):
    """Analytic vertical first hit of precisely the atlas axis-aligned boxes.

    Rays conceptually start above max(N*h)+1, independent of base height.
    Only the assigned tile is tested; robots and neighboring tiles never enter.
    Invalid heights are zero and must be interpreted using the validity mask.
    """
    points = torch.as_tensor(points_world)
    ids = torch.as_tensor(tile_ids, device=points.device, dtype=torch.long)
    while ids.ndim < points.ndim-1: ids = ids.unsqueeze(-1)
    if torch.any((ids<0)|(ids>5)): raise ValueError('tile id outside atlas')
    x = points[...,0]+10.
    y = points[...,1]-(-20.+ids*8.)
    ns = points.new_tensor([s[0] for s in STAGES])[ids]
    hs = points.new_tensor([s[1] for s in STAGES])[ids]
    end = torch.where(ids==0,22.,ns*.2+2.)
    width = torch.where(ids==0,4.,2.)
    valid = (x>=-2.)&(x<=end)&(y.abs()<=width)
    index = torch.floor(x/.2)+1
    stair = (ids>0)&(x>=0)&(x<ns*.2)&(y.abs()<=.4)
    landing = (ids>0)&(x>=ns*.2)&(x<=ns*.2+2.)&(y.abs()<=.4)
    z = torch.where(stair,index*hs,torch.where(landing,ns*hs,0.))
    return torch.where(valid,z,0.), valid
