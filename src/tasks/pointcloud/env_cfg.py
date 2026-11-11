"""Independent SDS scene; original task factories are never mutated."""
from dataclasses import dataclass
import numpy as np
import mujoco
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput, TerrainGeneratorCfg
from src.tasks.velocity.config.a2.eth_env_cfg import unitree_a2_eth_flat_env_cfg
from .curriculum import STAGES

@dataclass(kw_only=True)
class IsolatedCurriculumTerrain(SubTerrainCfg):
    def function(self, difficulty, spec, rng):
        del difficulty, rng
        body=spec.body('terrain'); geometries=[]
        def box(name, center, size):
            g=body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=center,size=size,
                contype=1,conaffinity=1,condim=3,friction=(.6,.005,.0001),rgba=(.4,.55,.7,1))
            geometries.append(TerrainGeometry(geom=g,color=(.4,.55,.7,1)))
        # Generator subtracts size/2. Compensate once; all first edges x=0.
        ox,oy=self.size[0]/2,self.size[1]/2
        box('pc_flat',(ox+10,oy,-.05),(15,5,.05))
        for k,(_,n,h,*_) in enumerate(STAGES[1:8],1):
            y=oy+40*k; end=n*.2
            box(f'pc_ground_{k}',(ox+(end)/2,y,-.05),((end+4)/2,2,.05))
            for i in range(n):
                box(f'pc_tread_{k}_{i}',(ox+(i+.5)*.2,y,(i+1)*h-.005),(.1,.4,.005))
            box(f'pc_top_{k}',(ox+end+1,y,n*h-.005),(1,.4,.005))
        return TerrainOutput(geometries=geometries,origin=np.array([ox,oy,0.]))


def pointcloud_env_cfg(num_envs=1,seed=42):
    cfg=unitree_a2_eth_flat_env_cfg()
    cfg.scene.num_envs=num_envs
    cfg.seed=seed
    cfg.episode_length_s=90
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=seed,size=(40.,600.),num_rows=1,num_cols=1,
        border_width=0.,color_scheme='none',sub_terrains={'isolated':IsolatedCurriculumTerrain()})
    cfg.curriculum={}
    cfg.events['reset_base'].params['pose_range']={'x':(-.2,.2),'y':(-.05,.05),'z':(0.,0.),'yaw':(-.0872664626,.0872664626)}
    cfg.commands['twist'].resampling_time_range=(1e9,1e9)
    for name in ('track_linear_velocity','body_orientation_l2','foot_clearance','is_terminated'):
        cfg.rewards.pop(name)
    return cfg
