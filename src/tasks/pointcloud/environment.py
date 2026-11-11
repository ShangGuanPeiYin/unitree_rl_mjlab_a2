"""Explicit no-auto-reset collection boundary for the independent pointcloud task."""
import math
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ObjRef
from mjlab.utils.lab_api.math import matrix_from_quat, euler_xyz_from_quat
from .env_cfg import pointcloud_env_cfg
from .geometry import query_top, stair_surfaces, fit_normal
from .rewards import velocity_tracking, alignment, edge_penalty, swing_penalty, SwingTargets
from .terminations import EpisodeEnds
from .curriculum import STAGES
from .perception import explicit_lidar_cfg, DualLidarScanAdapter, PointCloudHistory


class PointCloudEnv:
    privileged_dim=63
    privileged_names={'eth_observation':45,'body_velocity':3,'sole_clearance':4,'support_mask':4,'contact':4,'friction_strength_delay':3}

    def __init__(self,num_envs=1,device='cuda:0',seed=42,stage='P1',evaluation=False,command_speed=None,terrain_kind=None,scan_seed=42,perception_degraded=False,record_video=False):
        self.record_video=record_video
        self.num_envs=num_envs; self.device=device; self.evaluation=evaluation
        self.perception_degraded=perception_degraded
        self.command_speed=command_speed; self.terrain_kind=terrain_kind
        cfg=pointcloud_env_cfg(num_envs,seed)
        frame=ObjRef(type='body',name='base_link',entity='robot')
        cfg.scene.sensors += (explicit_lidar_cfg('pc_front',frame,seed=scan_seed),explicit_lidar_cfg('pc_rear',frame,rear=True,seed=scan_seed))
        self.env=ManagerBasedRlEnv(cfg,device=device,render_mode='rgb_array' if record_video else None)
        self.robot=self.env.scene['robot']
        self._strength_fields=('actuator_gainprm','actuator_biasprm','actuator_forcerange')
        self._nominal_actuator={name:self.env.sim.get_default_field(name).clone() for name in self._strength_fields}
        self.env.sim.expand_model_fields(('geom_friction',)+self._strength_fields)
        self.history=PointCloudHistory(num_envs,device=device)
        self.obs_history=torch.zeros(num_envs,5,45,device=device)
        self.ends=EpisodeEnds(num_envs,device); self.swing=SwingTargets(num_envs,device)
        self.scanner=DualLidarScanAdapter(self.env.sim,[self.env.scene['pc_front'],self.env.scene['pc_rear']],self.robot.indexing.geom_ids)
        self.foot_ids=[self.robot.geom_names.index(f'{leg}_foot_collision') for leg in ('FL','FR','RL','RR')]
        self.origin=torch.zeros(num_envs,3,device=device)
        self.flat=torch.ones(num_envs,dtype=torch.bool,device=device)
        self.level=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.friction=torch.full((num_envs,),.6,device=device); self.strength=torch.ones(num_envs,device=device)
        self.delay=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.contact_capacity_hits=0; self.constraint_capacity_hits=0
        self.set_stage(stage)

    def set_stage(self,stage):
        self.stage=next(s for s in STAGES if s[0]==stage)

    def reset(self):
        self._reset(torch.arange(self.num_envs,device=self.device))
        return self._observation(False)

    def _reset(self,ids):
        n=len(ids); idx=STAGES.index(self.stage); current=min(idx,7)
        level=torch.full((n,),current,device=self.device,dtype=torch.long)
        if self.evaluation:
            if self.terrain_kind=='flat' or idx==0: level.zero_()
        elif idx>0:
            u=torch.rand(n,device=self.device); level[u<.1]=0
            if idx>=4:
                replay=(u>=.1)&(u<.3)
                level[replay]=torch.randint(1,min(idx,7),(int(replay.sum()),),device=self.device)
        self.level[ids]=level; self.flat[ids]=level==0
        self.origin[ids,1]=40*level.float()
        self.env.scene.env_origins[ids]=self.origin[ids]
        self.env.scene.env_origins[ids,0]=torch.where(level==0,0.,-1.)
        self.env._reset_idx(ids)
        self.friction[ids]=.4+torch.rand(n,device=self.device)*.4 if idx==8 and not self.evaluation else .6
        self.env.sim.model.geom_friction[ids,:,0]=self.friction[ids,None]
        # Approved P5 effective PD/force-limit scaling: one scalar per episode,
        # always applied to immutable nominal values, never compounded.
        self.strength[ids]=.9+.2*torch.rand(n,device=self.device) if idx==8 and not self.evaluation else 1.
        for name in self._strength_fields:
            getattr(self.env.sim.model,name)[ids]=self._nominal_actuator[name][None]*self.strength[ids,None,None]
        cmd=self.env.command_manager.get_term('twist').command
        cmd[ids]=0
        vmax=.6 if idx==0 or idx>=7 else (.3 if idx<=3 else .4)
        upper=torch.where(level==0,.6,vmax)
        speed=.2+torch.rand(n,device=self.device)*(upper-.2)
        if not self.evaluation: speed[(level==0)&(torch.rand(n,device=self.device)<.1)]=0
        if self.command_speed is not None: speed.fill_(self.command_speed)
        cmd[ids,0]=speed
        self.ends.reset(ids); self.swing.reset(ids)
        self.delay[ids]=torch.randint(2,4,(n,),device=self.device) if (idx==8 and not self.evaluation) or self.perception_degraded else 0
        self.history.reset(ids,delay_steps=self.delay[ids])
        self.env.scene.write_data_to_sim(); self.env.sim.forward(); self.env.sim.sense()
        obs=self.env.observation_manager.compute(update_history=False)['actor']
        self.obs_history[ids]=obs[ids,None,:]

    def _state(self):
        data=self.robot.data
        body=data.root_link_pos_w-self.origin
        feet=data.geom_pos_w[:,self.foot_ids].clone(); feet[...,2]-=.032
        feet-=self.origin[:,None]
        surfaces=torch.zeros(self.num_envs,42,5,device=self.device); surfaces[...,0]=1; surfaces[...,1]=-1
        end=torch.zeros(self.num_envs,device=self.device); height=end.clone(); sh=end.clone()
        for k in range(8):
            chosen=self.level==k
            if k==0: surf=torch.tensor([[-5.,25.,-5.,5.,0.]],device=self.device)
            else:
                _,count,h,*_=STAGES[k]; surf=stair_surfaces(count,h,device=self.device)
                end[chosen]=count*.2; height[chosen]=count*h; sh[chosen]=h
            surfaces[chosen,:len(surf)]=surf
        z,valid,sid=query_top(feet,surfaces)
        contact=self.env.scene['feet_ground_contact'].data
        # mjlab netforce is the force exerted by primary foot onto secondary terrain.
        force=(-contact.force.reshape(self.num_envs,4,-1,3).sum(2)[:,[1,0,3,2],2]).clamp(min=0)
        stance=force>5
        roll,pitch,yaw=euler_xyz_from_quat(data.root_link_quat_w)
        return dict(body=body,feet=feet,surfaces=surfaces,z=z,valid=valid,sid=sid,force=force,stance=stance,
                    end=end,height=height,sh=sh,roll_pitch=torch.stack((roll,pitch),-1),yaw=yaw,
                    world_velocity=data.root_link_lin_vel_w.clone(),terrain_yaw=torch.zeros_like(yaw),
                    elapsed=self.env.episode_length_buf*.02,body_world=data.root_link_pos_w.clone(),body_terrain=body.clone(),stair_end=end.clone())

    def _observation(self,advance=True,state=None):
        obs=self.env.observation_manager.compute(update_history=False)['actor']
        if advance:
            self.obs_history=torch.cat((self.obs_history[:,1:],obs[:,None]),1)
            points,mask=self.scanner.scan(self.history.scan_group,self.robot.data.root_link_pos_w,matrix_from_quat(self.robot.data.root_link_quat_w))
            if (STAGES.index(self.stage)>0 and not self.evaluation) or self.perception_degraded:
                points=points+torch.randn_like(points)*.02
                outlier=(torch.rand_like(mask.float())<.02)&mask
                lo=torch.tensor([-1.2,-.6,-1.],device=self.device); hi=torch.tensor([1.2,.6,.5],device=self.device)
                points=torch.where(outlier[...,None],lo+torch.rand_like(points)*(hi-lo),points)
                mask=mask&(torch.rand_like(mask.float())>=.02)
            self.history.advance(points,mask)
        points,mask,ages=self.history.read()
        state=self._state() if state is None else state
        clear=torch.where(state['valid'],state['feet'][...,2]-state['z'],0)
        velocity=self.robot.data.root_link_lin_vel_b
        privileged=torch.cat((obs,velocity,clear,state['valid'].float(),state['stance'].float(),torch.stack((self.friction,self.strength,self.delay.float()*.02),-1)),1)
        return {'actor':{'obs_history':self.obs_history.clone(),'points':points.clone(),'mask':mask.clone(),'ages':ages.clone()},
                'privileged':privileged,'velocity':velocity.clone(),'clearance':clear,'clearance_mask':state['valid'].clone()}

    def step(self,actions):
        self.contact_capacity_hits=0; self.constraint_capacity_hits=0
        e=self.env; e.action_manager.process_action(actions)
        for _ in range(e.cfg.decimation):
            e._sim_step_counter+=1; e.action_manager.apply_action(); e.scene.write_data_to_sim(); e.sim.step(); e.scene.update(dt=e.physics_dt)
            wp=e.sim.wp_data
            import warp as wp_api
            self.contact_capacity_hits+=int((wp_api.to_torch(wp.nacon)>=wp.naconmax).any())
            self.constraint_capacity_hits+=int((wp_api.to_torch(wp.nefc)>=wp.njmax).any())
        e.episode_length_buf+=1; e.common_step_counter+=1
        e.sim.forward(); e.sim.sense()
        s=self._state()
        illegal=e.termination_manager.get_term_cfg('illegal_contact')
        illegal_mask=illegal.func(e,**illegal.params)
        failure,success,timeout=self.ends.update(s['body'],s['feet'],s['roll_pitch'],s['force']*((s['z']-s['height'][:,None]).abs()<.01)*s['valid'],illegal_mask,self.flat,s['elapsed'],s['end'],s['height'],step_height=s['sh'])
        s['failure_reasons']=torch.stack((illegal_mask,(torch.cos(s['roll_pitch'][:,0])*torch.cos(s['roll_pitch'][:,1])<math.cos(math.radians(70))),self.ends.side>=5,self.ends.drop>=5,torch.where(self.flat,(s['body'][:,0]<-5)|(s['body'][:,0]>25)|(s['body'][:,1].abs()>5),(s['body'][:,0]<-2)|(s['body'][:,0]>s['end']+2)|(s['body'][:,1].abs()>2))),-1)
        base=e.reward_manager.compute(dt=.02)
        cmd=e.command_manager.get_command('twist')
        cy,sy=torch.cos(s['yaw']),torch.sin(s['yaw']); vel=s['world_velocity']
        heading=torch.stack((cy*vel[:,0]+sy*vel[:,1],-sy*vel[:,0]+cy*vel[:,1]),-1)
        gx,gy=torch.meshgrid(torch.linspace(-.6,.6,13,device=self.device),torch.linspace(-.3,.3,7,device=self.device),indexing='ij')
        dx,dy=gx.flatten(),gy.flatten()
        grid=s['body'][:,None,:].expand(-1,len(dx),-1).clone()
        grid[...,0]+=cy[:,None]*dx-sy[:,None]*dy; grid[...,1]+=sy[:,None]*dx+cy[:,None]*dy
        gz,gv,_=query_top(grid,s['surfaces'],.05,.65); grid[...,2]=gz
        normal,nvalid=fit_normal(grid+self.origin[:,None],gv)
        up=matrix_from_quat(self.robot.data.root_link_quat_w)[...,2]
        phase=(s['elapsed'][:,None]/.6+torch.tensor([.5,0.,0.,.5],device=self.device))%1
        swing=phase>=.56; progress=((phase-.56)/.44).clamp(0,1)
        target=s['feet'].clone(); d=(cmd[:,0]*.264).clamp(.10,.20)
        target[...,0]+=cy[:,None]*d[:,None]; target[...,1]+=sy[:,None]*d[:,None]; target[...,2]=self.swing.last_support
        tz,tv,_=query_top(target,s['surfaces'],-.22,.02)
        h0,h1,sv=self.swing.update(s['stance'],s['z'],s['valid'],swing,tz,tv)
        theta=torch.arange(11,device=self.device)*(2*math.pi/11)
        edge=s['feet'][:,:,None,:].expand(-1,-1,11,-1).clone()
        edge[...,0]+=.042*torch.cos(theta); edge[...,1]+=.042*torch.sin(theta)
        _,ev,_=query_top(edge,s['surfaces'],-.01,.02)
        terms={'track_linear_velocity':velocity_tracking(heading,cmd),'local_alignment':alignment(up,normal,nvalid),'swing_clearance':swing_penalty(s['feet'][...,2],h0,h1,progress,sv),'edge':.5*edge_penalty(ev,s['stance']),'failure':-200*failure.float()}
        s['edge_ratio']=((~ev).float()*s['stance'][...,None]).sum((-1,-2))/(11*s['stance'].sum(-1)).clamp(min=1)
        s['foot_slip']=(self.robot.data.geom_lin_vel_w[:,self.foot_ids,:2].norm(dim=-1)*s['stance']).sum(-1)/s['stance'].sum(-1).clamp(min=1)
        s['highest_step']=torch.minimum((s['body'][:,0]/.2).floor().clamp(min=0),s['end']/.2)
        limits=self.env.sim.model.actuator_forcerange[:,self.robot.indexing.ctrl_ids]
        force=self.robot.data.actuator_force
        s['torque_saturation']=((force<=limits[...,0]*.999)|(force>=limits[...,1]*.999)).float().mean(-1)
        s['joint_limit']=((self.robot.data.joint_pos<self.robot.data.soft_joint_pos_limits[...,0])|(self.robot.data.joint_pos>self.robot.data.soft_joint_pos_limits[...,1])).float().mean(-1)
        reward=base+.02*sum(terms.values())
        for i,name in enumerate(e.reward_manager.active_terms):
            terms[name]=e.reward_manager._step_reward[:,i].clone()
        terminal=self._observation(True,s)
        info={'terminal_observation':terminal,'terminal_state':{k:v.clone() for k,v in s.items() if isinstance(v,torch.Tensor)},'reward_terms':terms,'contact_capacity_hits':self.contact_capacity_hits,'constraint_capacity_hits':self.constraint_capacity_hits}
        info['diagnostics']={'support_missing':float((~s['valid']).float().mean()),'normal_missing':float((~nvalid).float().mean()),'swing_target_missing':float((swing & ~sv).float().sum()/swing.sum().clamp_min(1))}
        if self.record_video: info['terminal_frame']=self.render()
        ids=(failure|success|timeout).nonzero().flatten()
        if len(ids):
            self._reset(ids); obs=self._observation(False)
        else: obs=terminal
        return obs,reward,failure,success,timeout,info

    def render(self): return self.env.render()

    def close(self): self.env.close()
