"""SDS 0.2 stage metadata, thin-board atlas and actual geom indexing.

Every consumer uses the same first-tread coordinate frame.  The native terrain
origin is the spawn point and is deliberately one metre behind that frame.
"""
from dataclasses import dataclass
import numpy as np
import mujoco
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from .spec import STAGES, STAGE_BY_ID


@dataclass(frozen=True)
class TileMetadata:
    tile_id: int
    stage: str
    N: int
    h: float
    budget: int
    terrain_origin_world: tuple
    frame_origin_world: tuple
    valid_bounds_local: tuple
    step_depth: float = .20
    stair_width: float = .80
    tread_thickness: float = .01
    approach_length: float = 2.
    landing_length: float = 2.
    comparison_proportions: tuple = (.1,.9)
    formal_proportions: tuple = (.1,.2,.7)
    initial_distance_range: tuple = (1.,1.)
    initial_y_range: tuple = (0.,0.)
    initial_yaw_range: tuple = (0.,0.)
    perturb_ramp_updates: int = 0
    @property
    def step_count(self): return self.N
    @property
    def step_height(self): return self.h
    @property
    def stair_origin_world(self): return self.frame_origin_world
    def world_to_local(self, points):
        if isinstance(points,torch.Tensor):
            return points-points.new_tensor(self.frame_origin_world[:points.shape[-1]])
        points=np.asarray(points)
        return points-np.asarray(self.frame_origin_world[:points.shape[-1]])


def atlas_metadata():
    result=[]
    for stage in STAGES:
        y=-len(STAGES)*8./2+stage.tile_id*8.+4.
        frame=(-10.,y,0.)
        origin=frame if stage.N==0 else (-11.,y,0.)
        bounds=(-2.,22.,-4.,4.) if stage.N==0 else (-2.,stage.N*.2+2.,-2.,2.)
        noise=dict(initial_distance_range=(.8,1.2),initial_y_range=(-.1,.1),initial_yaw_range=(-np.pi/18,np.pi/18),perturb_ramp_updates=400) if stage.id=='S5' else {}
        result.append(TileMetadata(stage.tile_id,stage.id,stage.N,stage.h,stage.budget,origin,frame,bounds,**noise))
    return tuple(result)


@dataclass(kw_only=True)
class AtlasTileCfg(SubTerrainCfg):
    N: int = 0
    h: float = 0.
    def function(self,difficulty,spec,rng):
        del difficulty,rng
        body=spec.body('terrain'); geometry=[]
        def box(name,pos,size,color):
            geom=body.add_geom(name=f'v02_{len(list(body.geoms))}_{name}',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,size=size,contype=1,conaffinity=1,condim=3,friction=(.6,.005,.0001))
            geometry.append(TerrainGeometry(geom=geom,color=color))
        if not self.N:
            box('flat',(12.,4.,-.05),(12.,4.,.05),(.3,.33,.36,1.))
            origin=(2.,4.,0.)
        else:
            if not .01 < self.h: raise ValueError('Thin tread must leave an open riser')
            length=4.+self.N*.2
            box('ground',(length/2,4.,-.05),(length/2,2.,.05),(.3,.33,.36,1.))
            for i in range(1,self.N+1):
                box(f'tread_{i}',(2.+(i-.5)*.2,4.,i*self.h-.005),(.1,.4,.005),(.3,.55,.75,1.))
            box('landing',(2.+self.N*.2+1.,4.,self.N*self.h-.005),(1.,.4,.005),(.35,.7,.45,1.))
            origin=(1.,4.,0.)
        return TerrainOutput(origin=np.array(origin),geometries=geometry)


def atlas_generator_cfg(seed=42):
    return TerrainGeneratorCfg(seed=seed,size=(24.,8.),num_rows=1,num_cols=len(STAGES),
        curriculum=True,border_width=0.,sub_terrains={s.id:AtlasTileCfg(N=s.N,h=s.h,proportion=1.) for s in STAGES},color_scheme='height')


def allocate_stage_slots(num_envs,stage,mode='comparison',previous_stage=None):
    if num_envs<1 or mode not in ('comparison','formal'): raise ValueError('invalid allocation')
    current=STAGE_BY_ID[stage]
    weights={0:1.} if not current.N else {0:.1,current.tile_id:.9}
    if current.N and mode=='formal' and previous_stage is not None:
        previous=STAGE_BY_ID[previous_stage]
        if previous.tile_id==current.tile_id or not previous.N: raise ValueError('invalid previous stage')
        weights={0:.1,previous.tile_id:.2,current.tile_id:.7}
    raw={k:num_envs*v for k,v in weights.items()}
    counts={k:int(np.floor(v)) for k,v in raw.items()}
    order=sorted(raw,key=lambda k:(-(raw[k]-counts[k]),k))
    for k in order[:num_envs-sum(counts.values())]: counts[k]+=1
    return np.concatenate([np.full(counts[k],k,dtype=np.int64) for k in sorted(counts)])


def query_top_surface(points_world,tile_ids):
    points=torch.as_tensor(points_world)
    ids=torch.as_tensor(tile_ids,device=points.device,dtype=torch.long)
    while ids.ndim<points.ndim-1: ids=ids.unsqueeze(-1)
    if torch.any((ids<0)|(ids>=len(STAGES))): raise ValueError('tile id outside v02 atlas')
    meta=atlas_metadata()
    origin=points.new_tensor([m.frame_origin_world for m in meta])[ids]
    x,y=points[...,0]-origin[...,0],points[...,1]-origin[...,1]
    n=points.new_tensor([m.N for m in meta])[ids];h=points.new_tensor([m.h for m in meta])[ids]
    flat=n==0
    valid=(x>=-2)&(x<=torch.where(flat,22.,n*.2+2.))&(y.abs()<=torch.where(flat,4.,2.))
    tread=(~flat)&(x>=0)&(x<n*.2)&(y.abs()<=.4)
    landing=(~flat)&(x>=n*.2)&(x<=n*.2+2.)&(y.abs()<=.4)
    z=torch.where(tread,(torch.floor(x/.2)+1)*h,torch.where(landing,n*h,0.))
    return torch.where(valid,z,0.),valid


@dataclass(frozen=True)
class GeometryIndex:
    tile: np.ndarray
    step: np.ndarray
    bounds: np.ndarray
    top: np.ndarray
    support_bounds: np.ndarray


def index_geometry(model):
    """Index generated boxes by measured world coordinates, then validate all faces.

    Names identify terrain membership only; order and fixed geom counts play no
    role. Counts are checked against each generated metadata description.
    """
    tile=np.full(model.ngeom,-1,dtype=np.int64);step=np.full(model.ngeom,-1,dtype=np.int64)
    bounds=np.zeros((model.ngeom,4),dtype=np.float32);top=np.zeros(model.ngeom,dtype=np.float32)
    support_bounds=bounds.copy(); metadata=atlas_metadata()
    for g in range(model.ngeom):
        if not (model.geom(g).name or '').startswith('terrain_'): continue
        p,s=model.geom_pos[g],model.geom_size[g]
        matches=[m for m in metadata if abs(p[1]-m.frame_origin_world[1])<1e-5]
        if len(matches)!=1: raise ValueError(f'Unassigned terrain geom {g}')
        m=matches[0];ox,oy,_=m.frame_origin_world
        b=np.array([p[0]-s[0],p[0]+s[0],p[1]-s[1],p[1]+s[1]])
        z=p[2]+s[2];local=(p[0]-ox,p[1]-oy)
        if b[0]<ox+m.valid_bounds_local[0]-1e-5 or b[1]>ox+m.valid_bounds_local[1]+1e-5:
            raise ValueError('Terrain geom outside metadata')
        k=0 if abs(z)<1e-6 else int(round(z/m.h))
        if k<0 or k>m.N or (k and abs(z-k*m.h)>1e-5): raise ValueError('Invalid tread height')
        if k:
            is_landing=abs(local[0]-(m.N*.2+1.))<1e-5
            expected_x=m.N*.2+1. if is_landing else (k-.5)*.2
            expected_size=(1. if is_landing else .1,.4,.005)
            if abs(local[0]-expected_x)>1e-5 or not np.allclose(s,expected_size,atol=1e-6): raise ValueError('Invalid thin board')
        tile[g],step[g],bounds[g],top[g]=m.tile_id,k,b,z
        support_bounds[g]=b
        if k==m.N and k:
            support_bounds[g]=[ox+(m.N-1)*.2,ox+m.N*.2+2.,oy-.4,oy+.4]
    for m in metadata:
        expected=1 if not m.N else m.N+2
        if int((tile==m.tile_id).sum())!=expected: raise ValueError(f'Missing/extra geometry for {m.stage}')
        if m.N and sorted(step[tile==m.tile_id].tolist())!=[0]+list(range(1,m.N+1))+[m.N]:
            raise ValueError(f'Incorrect step metadata {m.stage}')
    return GeometryIndex(tile,step,bounds,top,support_bounds)
