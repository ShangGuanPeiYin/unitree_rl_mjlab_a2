"""Dedicated SDS environment: actual contact, terminal snapshots and local RNG."""
import math
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from ..v02.env_cfg import v02_env_cfg
from ..v02.spec import STAGES
from .spec import STAGE_BY_ID, STAGE_BY_TILE, recipe_config, PROTOCOL_ID, CODE_VERSION
from .success import ClimbState, ParkingDiagnosticState, allocate_stage_slots
from ..v02.terrain import atlas_metadata, index_geometry
from ..v02.rewards import ParkingState, FirstFootfallState, ProgressState, baseline_adjustments
from ..v02.map import IdealHeightMap, ProprioHistory
from ..v02.contacts import ActualContacts, LEGS


def euler_heading(q):
    w,x,y,z = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    # Positive rotation about +y points the robot nose downward; SDS uses nose up.
    pitch = -torch.asin((2*(w*y-z*x)).clamp(-1,1))
    yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return roll,pitch,yaw


def timeout_steps(n,h,vx):
    return torch.ceil((2*(2.2+n*torch.sqrt(h*h+.2**2))/vx.clamp_min(.01)+10)/.02).long()


def event_reward(success, failure):
    return success.float()*10-failure.float()*4


class NumericalFailure(FloatingPointError):
    """Fail-fast exception carrying pre-reset evidence for the runner to persist."""
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


class FormalEnv(ManagerBasedRlEnv):
    def __init__(self,num_envs=1024,device='cuda:0',seed=42,stage='R0',stage_iteration=0,evaluation=False,fixed_vx=None,initial_cases=None,render_mode=None,parking_diagnostic=False):
        self.run_seed=int(seed)
        self.recipe,self.mode='C','formal'
        self.recipe_options=recipe_config('C')
        self.parking_diagnostic=bool(parking_diagnostic)
        self.initial_cases=None if initial_cases is None else torch.as_tensor(initial_cases,device=device,dtype=torch.float32)
        if self.initial_cases is not None and self.initial_cases.shape != (num_envs,3): raise ValueError('initial_cases must be [num_envs,3]')
        self.checkpoint_identity=None
        self.previous_stage=None
        self.evaluation,self.fixed_vx = evaluation,fixed_vx
        self.stage,self.stage_iteration = None,0
        super().__init__(v02_env_cfg(num_envs,seed),device,render_mode)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.robot = self.scene['robot']
        expected = [f'{leg}_{j}_joint' for leg in LEGS for j in ('hip','thigh','calf')]
        self.joint_ids,_ = self.robot.find_joints(expected,preserve_order=True)
        self.foot_ids,_ = self.robot.find_sites(LEGS,preserve_order=True)
        self.action_joint_ids = self.action_manager.get_term('joint_pos')._target_ids
        if self.action_joint_ids.tolist() != list(self.joint_ids):
            raise ValueError('Action targets differ from ETH joint order')
        # Action manager actuator matching determines joint order; reject unnoticed drift.
        actual = [self.robot.joint_names[i] for i in self.joint_ids]
        if actual != expected: raise ValueError('ETH joint order mismatch')
        self.tile_ids = torch.zeros(num_envs,dtype=torch.long,device=device)
        metadata=atlas_metadata()
        self.origins = torch.tensor([m.frame_origin_world for m in metadata],device=device)
        self.spawn_x=torch.tensor([m.terrain_origin_world[0]-m.frame_origin_world[0] for m in metadata],device=device)
        self.initial_noise=torch.tensor([[(m.initial_distance_range[1]-m.initial_distance_range[0])/2,
            (m.initial_y_range[1]-m.initial_y_range[0])/2,(m.initial_yaw_range[1]-m.initial_yaw_range[0])/2] for m in metadata],device=device)
        self.perturb_ramp=torch.tensor([m.perturb_ramp_updates for m in metadata],device=device)
        geometry=index_geometry(self.sim.mj_model)
        self.bounds=torch.tensor(geometry.bounds,device=device)
        self.geom_top=torch.tensor(geometry.top,device=device)
        self.geom_tile=torch.tensor(geometry.tile,device=device)
        self.geom_step=torch.tensor(geometry.step,device=device)
        self.geom_step_bounds=torch.tensor(geometry.support_bounds,device=device)
        self.contacts=ActualContacts(self.sim,geometry.tile,geometry.top,device)
        self.parking=ParkingState(num_envs,device)
        self.footfall=FirstFootfallState(num_envs,max(s.N for s in STAGES),device)
        self.progress=ProgressState(num_envs,device)
        self.map_cache=IdealHeightMap(num_envs,device)
        self.history_cache=ProprioHistory(num_envs,device)
        self.command=self.command_manager.get_command('twist')
        self.start_vx=torch.zeros(num_envs,device=device)
        self.stop=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.stop_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.last_contact=torch.zeros(num_envs,4,dtype=torch.bool,device=device)
        self.initial_pose=torch.zeros(num_envs,3,device=device)
        self.sum_vx=torch.zeros(num_envs,device=device)
        self.sum_speed=torch.zeros(num_envs,device=device)
        self.max_lateral=torch.zeros(num_envs,device=device)
        self.highest=torch.zeros(num_envs,device=device)
        self.slip_sum=torch.zeros(num_envs,device=device)
        self.nonfoot_sum=torch.zeros(num_envs,device=device)
        self.limit_sum=torch.zeros(num_envs,device=device)
        self.peak_torque=torch.zeros(num_envs,device=device)
        self.height_defaults=torch.zeros(num_envs,device=device)
        self.first_foot_times=torch.full((num_envs,4),-1.,device=device)
        self.foot_step_times=torch.full((num_envs,4,max(s.N for s in STAGES)),-1.,device=device)
        self.first_step_support_time=torch.full((num_envs,),-1.,device=device)
        self.first_nonfoot_time=torch.full((num_envs,),-1.,device=device)
        self.first_nonfoot_link=torch.full((num_envs,),-1,device=device,dtype=torch.long)
        self.first_nonfoot_force=torch.zeros(num_envs,3,device=device)
        self.before_first_foot_height=torch.zeros(num_envs,2,device=device)
        self.max_body_x=torch.zeros(num_envs,device=device)
        self.stagnation_total=torch.zeros(num_envs,device=device)
        self.stagnation_current=torch.zeros(num_envs,device=device)
        self.stagnation_longest=torch.zeros(num_envs,device=device)
        self.body_crossed_first=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.all_feet_on_platform=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.episode_reward_sums={}
        self.climb=ClimbState(num_envs,device)
        self.parking_diagnostics=ParkingDiagnosticState(num_envs,device)
        self.slot_classes=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.max_heading=torch.zeros(num_envs,device=device)
        self.sum_vx_world=torch.zeros(num_envs,device=device)
        self.sum_horizontal_speed_world=torch.zeros(num_envs,device=device)
        self.moving_count=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.moving_sum_vx=torch.zeros(num_envs,device=device)
        self.moving_sum_speed=torch.zeros(num_envs,device=device)
        self.set_stage(stage,stage_iteration)

    def set_stage(self,stage,stage_iteration=0,previous_stage=None):
        if stage not in STAGE_BY_ID: raise ValueError('Unknown v02 stage')
        if stage==self.stage and previous_stage is None: previous_stage=self.previous_stage
        changed=stage!=self.stage or previous_stage!=self.previous_stage
        self.stage,self.stage_iteration=stage,int(stage_iteration)
        self.previous_stage=previous_stage
        if changed:
            if self.evaluation:
                ids=np.full(self.num_envs,STAGE_BY_ID[stage].tile_id,dtype=np.int64)
                classes=np.full(self.num_envs,3 if stage!='flat' else (1 if self.fixed_vx==0 else 0),dtype=np.int64)
            else:
                ids,classes,_=allocate_stage_slots(self.num_envs,stage,previous_stage)
            self.tile_ids.copy_(torch.as_tensor(ids,device=self.device))
            self.slot_classes.copy_(torch.as_tensor(classes,device=self.device))
            self.reset()

    def _parameters(self):
        ns=self.command.new_tensor([v.N for v in STAGES])[self.tile_ids]
        hs=self.command.new_tensor([v.h for v in STAGES])[self.tile_ids]
        return ns,hs

    def _reset_ids(self,ids):
        self.sim.reset(ids);self.scene.reset(ids);self.action_manager.reset(ids)
        r=self.robot;state=r.data.default_root_state[ids].clone()
        local=torch.zeros(len(ids),3,device=self.device)
        stairs=self.tile_ids[ids]>0
        local[:,0]=self.spawn_x[self.tile_ids[ids]]
        yaw=torch.zeros(len(ids),device=self.device)
        amplitude=self.initial_noise[self.tile_ids[ids]]
        if amplitude.any():
            ramp=torch.ones(len(ids),device=self.device) if self.evaluation else (self.stage_iteration/self.perturb_ramp[self.tile_ids[ids]].clamp_min(1)).clamp_max(1.)
            noise=(torch.rand((len(ids),3),device=self.device,generator=self.rng)*2-1)*amplitude*ramp[:,None]
            local[:,:2]+=noise[:,:2]
            yaw=noise[:,2]
        if self.initial_cases is not None:
            local[:,:2]=self.initial_cases[ids,:2]
            yaw=self.initial_cases[ids,2].clone()
        local[:,2]=state[:,2]
        state[:,:3]=self.origins[self.tile_ids[ids]]+local
        state[:,3:7]=0;state[:,3]=torch.cos(yaw/2);state[:,6]=torch.sin(yaw/2);state[:,7:]=0
        r.write_root_state_to_sim(state,ids)
        r.write_joint_state_to_sim(r.data.default_joint_pos[ids],torch.zeros_like(r.data.joint_vel[ids]),env_ids=ids)
        self.command[ids]=0
        vx=.2+.4*torch.rand(len(ids),device=self.device,generator=self.rng)
        flat=~stairs
        if self.fixed_vx is not None: vx[:]=float(self.fixed_vx)
        else:
            vx[self.slot_classes[ids]==1]=0.
        self.command[ids,0]=vx;self.start_vx[ids]=vx
        self.episode_length_buf[ids]=0;self.stop[ids]=False;self.stop_steps[ids]=0
        self.last_contact[ids]=False;self.contacts.reset(ids)
        self.parking.reset(ids);self.footfall.reset(ids);self.progress.reset(ids,local[:,0])
        self.climb.reset(ids);self.parking_diagnostics.reset(ids)
        for a in (self.max_heading,self.sum_vx_world,self.sum_horizontal_speed_world,self.moving_count,self.moving_sum_vx,self.moving_sum_speed): a[ids]=0
        self.first_foot_times[ids]=-1;self.foot_step_times[ids]=-1;self.first_step_support_time[ids]=-1
        self.first_nonfoot_time[ids]=-1;self.first_nonfoot_link[ids]=-1;self.first_nonfoot_force[ids]=0
        self.before_first_foot_height[ids]=0
        self.max_body_x[ids]=local[:,0]
        for a in (self.stagnation_total,self.stagnation_current,self.stagnation_longest): a[ids]=0
        self.body_crossed_first[ids]=False;self.all_feet_on_platform[ids]=False
        for value in self.episode_reward_sums.values(): value[ids]=0
        for a in (self.sum_vx,self.sum_speed,self.max_lateral,self.highest,self.slip_sum,self.nonfoot_sum,self.limit_sum,self.peak_torque,self.height_defaults): a[ids]=0
        self.initial_pose[ids]=torch.stack((local[:,0],local[:,1],yaw),-1)
        self.scene.write_data_to_sim();self.sim.forward();self.sim.sense()
        obs=self._proprio()
        self.history_cache.reset(obs,ids)
        self.map_cache.reset(r.data.root_link_pos_w,euler_heading(r.data.root_link_quat_w)[2],self.tile_ids,ids)

    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng.manual_seed(seed)
        ids=torch.arange(self.num_envs,device=self.device)
        self._reset_ids(ids)
        return self._observations(update=False),{}

    def _proprio(self):
        d=self.robot.data
        return torch.cat((d.root_link_ang_vel_b,d.projected_gravity_b,self.command,
            d.joint_pos[:,self.joint_ids]-d.default_joint_pos[:,self.joint_ids],
            d.joint_vel[:,self.joint_ids],self.action_manager.action),-1)

    def _foot_heights(self):
        p=self.robot.data.site_pos_w[:,self.foot_ids]
        bottom=p[:,:,2]-.032
        xy=p[:,:,:2,None]
        b=self.bounds
        inside=(xy[:,:,0]>=b[:,0])&(xy[:,:,0]<=b[:,1])&(xy[:,:,1]>=b[:,2])&(xy[:,:,1]<=b[:,3])
        valid=inside&(self.geom_tile[None,None,:]==self.tile_ids[:,None,None])&(self.geom_top[None,None,:]<=bottom[:,:,None]+.005)
        z=torch.where(valid,self.geom_top[None,None,:],-torch.inf).amax(-1)
        chosen=self.geom_top[self.contacts.support_geom.clamp_min(0)]
        z=torch.where(self.contacts.support,chosen,z)
        missing=~torch.isfinite(z)
        heights=torch.where(missing,1.5,(bottom-z).clamp(-.05,1.5))
        return heights,missing

    def _observations(self,update=True):
        d=self.robot.data;prop=self._proprio()
        if update:
            self.history_cache.update(prop)
            self.map_cache.update(d.root_link_pos_w,euler_heading(d.root_link_quat_w)[2],self.tile_ids)
        h,missing=self._foot_heights()
        force=quat_apply_inverse(d.root_link_quat_w[:,None].expand(-1,4,-1).reshape(-1,4),self.contacts.foot_force.reshape(-1,3)).reshape(self.num_envs,12)
        # qfrc_actuator is joint generalized actuation force, already in joint order.
        torque=self.sim.data.qfrc_actuator[:,self.robot.indexing.joint_v_adr][:,self.joint_ids]
        privileged=torch.cat((d.root_link_lin_vel_b,(self.contacts.foot_force.norm(dim=-1)>5).float(),force,h,torque),-1)
        return {'proprio':prop.clone(),'history':self.history_cache.history.clone(),
                'heightmap':self.map_cache.map.clone(),'privileged':privileged}

    def _task_state(self):
        self.sample_command=self.command.clone()  # command executed during this physical sample, before arrival decision
        d=self.robot.data;n,h=self._parameters();local=d.root_link_pos_w-self.origins[self.tile_ids]
        feet=d.site_pos_w[:,self.foot_ids]-self.origins[self.tile_ids,None]
        roll,pitch,yaw=euler_heading(d.root_link_quat_w)
        stairs=self.tile_ids>0;region=stairs&(local[:,0]>=0)
        safe=(feet[:,:,0]>=n[:,None]*.2+.042)&(feet[:,:,0]<=n[:,None]*.2+2-.042)&(feet[:,:,1].abs()<=.4-.042)
        expected=torch.minimum((torch.floor(local[:,0]/.2)+1).clamp_min(0),n)*h
        body=self.control_body_hit | (self.contacts.link_force[:,self.contacts.body_link_mask].norm(dim=-1)>0).any(-1)
        long=self.control_long_hit | (self.contacts.nonfoot_duration>=.1-1e-6).any(-1)
        fallen=(roll.abs()>math.radians(70))|(pitch.abs()>math.radians(70))
        side=region&(local[:,1].abs()>.4)
        below=region&(local[:,2]<expected-.5)
        xmax=torch.where(stairs,n*.2+2,22.)
        ymax=torch.where(stairs,2.,4.)
        bounds=(local[:,0]<-2)|(local[:,0]>xmax)|(local[:,1].abs()>ymax)
        numeric=~torch.isfinite(torch.cat((d.root_link_pos_w,d.root_link_quat_w,d.joint_pos,d.joint_vel),-1)).all(-1)
        failure=body|long|fallen|side|below|bounds|numeric
        # Detect actual safe top support directly from contact forces. The legacy
        # generic support counter does not enforce same-surface consecutive samples.
        # The landing is the two-metre geom; the same-height final tread is
        # deliberately excluded. IDs enforce same physical support surface.
        is_landing=(self.geom_step[None,:]==n[:,None]) & (self.geom_tile[None,:]==self.tile_ids[:,None]) & ((self.bounds[:,1]-self.bounds[:,0])[None,:]>1.99)
        force=torch.where(is_landing[:,None,:],self.contacts.top_forces,-torch.inf)
        upward,surface_ids=force.max(-1)
        candidate=(upward>5.) & safe & stairs[:,None]
        result=self.climb.update(self.episode_length_buf,stairs,self.tile_ids,surface_ids,candidate,safe,
            local[:,0],n*.2,local[:,1],failure)
        self.climb_event=result.success
        self.current_platform=result.current_support
        self.stop=self.climb.climbed.clone() if self.parking_diagnostic else torch.zeros_like(failure)
        if self.parking_diagnostic:
            self.command[self.stop]=0
            parking_success,parking_timeout=self.parking_diagnostics.update(self.episode_length_buf,
                self.climb.climb_step,self.climb.climbed,self.current_platform,local[:,1],d.root_link_lin_vel_b.norm(dim=-1),failure)
            self.stop_steps=self.parking_diagnostics.parking_steps.clone()
            success=parking_success & ~failure
        else:
            success=result.success
            parking_timeout=torch.zeros_like(failure)
            self.stop_steps.zero_()
        base_timeout=self.episode_length_buf>=torch.where(stairs,timeout_steps(n,h,self.start_vx),1500)
        # Arrival keeps its original climb record and receives up to ten extra
        # seconds, even if the ordinary climb timeout was reached after arrival.
        timeout=torch.where(self.climb.climbed,parking_timeout,base_timeout) if self.parking_diagnostic else base_timeout
        truncated=timeout&~failure&~success
        reason=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        for code,mask in [(1,truncated),(2,success),(3,bounds),(4,below),(5,side),(6,fallen),(7,long),(8,body),(9,numeric)]:reason[mask]=code
        return local,roll,pitch,yaw,region,success,failure,truncated,reason

    def _reward(self,state,previous_action):
        local,roll,pitch,yaw,region,success,failure,_,_=state
        success=self.climb_event
        d=self.robot.data;v=d.root_link_lin_vel_b;w=d.root_link_ang_vel_b
        jp=d.joint_pos[:,self.joint_ids];default=d.default_joint_pos[:,self.joint_ids]
        speed=self.command.norm(dim=-1);active=speed>.1
        std=torch.where((speed<.1)[:,None],jp.new_tensor([.05,.1,.15]*4),jp.new_tensor([.15,.35,.5]*4))
        contact=self.contacts.found
        vel=d.site_lin_vel_w[:,self.foot_ids,:2].square().sum(-1)
        slip=(vel*contact).sum(-1)*(speed>.1)
        impact=(self.contacts.foot_force.norm(dim=-1)*(contact&(self.contacts.contact_duration<=self.step_dt+1e-7))).sum(-1)*active
        self.last_contact=contact
        b=self.bounds[self.contacts.support_geom.clamp_min(0)].clone()
        n,h=self._parameters()
        # Last tread and platform form one coplanar support rectangle.
        z=self.geom_top[self.contacts.support_geom.clamp_min(0)]
        joined=(self.tile_ids[:,None]>0)&((z-n[:,None]*h[:,None]).abs()<.001)
        b[:,:,0]=torch.where(joined,self.origins[self.tile_ids,0,None]+(n[:,None]-1)*.2,b[:,:,0])
        b[:,:,1]=torch.where(joined,self.origins[self.tile_ids,0,None]+n[:,None]*.2+2,b[:,:,1])
        p=d.site_pos_w[:,self.foot_ids]
        distance=torch.stack((p[:,:,0]-b[:,:,0],b[:,:,1]-p[:,:,0],p[:,:,1]-b[:,:,2],b[:,:,3]-p[:,:,1]),-1).amin(-1)
        edge=(self.contacts.support*((.042-distance)/.042).clamp_min(0).square()).mean(-1)
        limits=d.soft_joint_pos_limits[:,self.joint_ids]
        limit=((limits[:,:,0]-jp).clamp_min(0)+(jp-limits[:,:,1]).clamp_min(0)).sum(-1)
        nonfoot=(self.contacts.link_force.norm(dim=-1)>10).float().mean(-1)
        angmom=self.scene['robot/root_angmom'].data.square().sum(-1)
        raw={'track_linear_velocity':torch.exp(-(v[:,:2]-self.command[:,:2]).square().sum(-1)/.25),
            'track_angular_velocity':torch.exp(-((w[:,2]-self.command[:,2]).square()+.05*w[:,:2].square().sum(-1))/.5),
            'orientation':roll.square()+(pitch-math.radians(55)).clamp_min(0).square()+(-math.radians(15)-pitch).clamp_min(0).square(),
            'pose':torch.exp(-((jp-default)/std).square().mean(-1)),
            'body_ang_vel':d.root_link_ang_vel_w[:,:2].square().sum(-1),'angular_momentum':angmom,
            'joint_acc':d.joint_acc[:,self.joint_ids].square().sum(-1),'joint_limits':limit,
            'action_rate':(self.action_manager.action-previous_action).square().sum(-1),'slip':slip,'soft_landing':impact,
            'stand_still':(jp-default).square().sum(-1)*(speed<=.1),'center':(local[:,1]/.2).square().clamp_max(4)*region,
            'heading':(1-yaw.cos())*region,'edge':edge,'nonfoot':nonfoot}
        weights=dict(zip(raw,(1,1,-1,.2,-.05,-.025,-2.5e-7,-10,-.05,-.25,-.001,-1,-.5,-.5,-1,-1)))
        adjustments=baseline_adjustments(raw['track_linear_velocity'],raw['track_angular_velocity'],raw['pose'],self.command,enabled=self.recipe_options['new_rewards'])
        adjusted=dict(raw);adjusted.update(adjustments['adjusted'])
        self.footfall_result=self.footfall.update(self.tile_ids,d.site_pos_w[:,self.foot_ids,:2],self.contacts.top_forces,self.geom_tile,self.geom_step,self.geom_step_bounds,failure,enabled=self.recipe_options['new_rewards'])
        progress_reward=self.progress.update(local[:,0],n*.2+.8,self.command[:,0],self.contacts.support,active,n>0,self.stop,failure,enabled=self.recipe_options['new_rewards'],dt=self.step_dt)
        tensors={'reward_raw/'+k:x for k,x in raw.items()}
        tensors.update({'reward_baseline/'+k:x for k,x in adjustments['base'].items()})
        tensors.update({'reward_weighted/'+k:x*weights[k]*self.step_dt for k,x in adjusted.items()})
        tensors.update({'reward_event/success':success.float()*10,'reward_event/failure':failure.float()*-4,
                        'reward_event/footfall':self.footfall_result.reward,'reward_event/footfall_count':self.footfall_result.events.sum(-1).float(),
                        'reward_progress':progress_reward})
        reward=self.step_dt*sum(adjusted[k]*weights[k] for k in raw)+event_reward(success,failure)+self.footfall_result.reward+progress_reward
        tensors['reward_total']=reward
        for key,value in tensors.items():
            if key not in self.episode_reward_sums: self.episode_reward_sums[key]=torch.zeros_like(value)
            self.episode_reward_sums[key]+=value
        self.step_reward_tensors=tensors
        self.slip_sum+=slip;self.nonfoot_sum+=nonfoot;self.limit_sum+=limit
        self.reward_terms=dict(zip(tensors,torch.stack([value.mean() for value in tensors.values()]).detach().cpu().tolist()))
        return reward

    def step(self,action):
        if not torch.isfinite(action).all():
            raise NumericalFailure('Nonfinite action',{'action':action.detach().cpu(),'qpos':self.sim.data.qpos.detach().cpu(),'qvel':self.sim.data.qvel.detach().cpu()})
        previous=self.action_manager.action.clone()
        self.action_manager.process_action(action)
        self.control_body_hit=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.control_long_hit=torch.zeros_like(self.control_body_hit)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter+=1
            self.action_manager.apply_action();self.scene.write_data_to_sim();self.sim.step()
            self.sim.forward();self.scene.update(dt=self.physics_dt)
            self.contacts.sample(self.tile_ids,physics_dt=self.physics_dt)
            hit=self.contacts.link_force.norm(dim=-1)>0
            fresh=(self.first_nonfoot_time<0)&hit.any(-1)
            link=hit.long().argmax(-1)
            self.first_nonfoot_time[fresh]=(self.episode_length_buf[fresh]*self.step_dt+(_+1)*self.physics_dt)
            self.first_nonfoot_link[fresh]=link[fresh]
            self.first_nonfoot_force[fresh]=self.contacts.link_force[torch.arange(self.num_envs,device=self.device),link][fresh]
            self.control_body_hit|=(self.contacts.link_force[:,self.contacts.body_link_mask].norm(dim=-1)>0).any(-1)
            self.control_long_hit|=(self.contacts.nonfoot_duration>=.1-1e-6).any(-1)
        self.contacts.sample(self.tile_ids,control=True)
        self.sim.sense()
        self.episode_length_buf+=1;self.common_step_counter+=1
        state=self._task_state();reward=self._reward(state,previous)
        local,_,_,_,_,success,failure,truncated,reason=state
        terminal=self._observations(update=True)
        self._update_diagnostics(local)
        self.height_defaults+=self._foot_heights()[1].sum(-1)
        d=self.robot.data
        self.max_heading=torch.maximum(self.max_heading,state[3].abs())
        self.sum_vx_world+=d.root_link_lin_vel_w[:,0]
        self.sum_horizontal_speed_world+=d.root_link_lin_vel_w[:,:2].norm(dim=-1)
        moving=~self._motion_parking_mask()
        self.moving_count+=moving.long()
        self.moving_sum_vx+=d.root_link_lin_vel_b[:,0]*moving
        self.moving_sum_speed+=d.root_link_lin_vel_b.norm(dim=-1)*moving
        self.sum_vx+=self.robot.data.root_link_lin_vel_b[:,0]
        self.sum_speed+=self.robot.data.root_link_lin_vel_b.norm(dim=-1)
        self.max_lateral=torch.maximum(self.max_lateral,local[:,1].abs())
        n,h=self._parameters()
        valid_step=torch.where(self.footfall_result.qualified_count>=2,self.footfall_result.selected_step,0).amax(-1)
        self.highest=torch.maximum(self.highest,valid_step.clamp_min(0).to(self.highest.dtype))
        self.peak_torque=torch.maximum(self.peak_torque,terminal['privileged'][:,-12:].abs().amax(-1))
        names=['running','timeout','success','bounds','below_support','side_fall','orientation','nonfoot_persistent','body_contact','numerical_failure']
        done=success|failure|truncated;ids=done.nonzero().flatten();episodes=[]
        for i in ids.tolist():
            count=int(self.episode_length_buf[i])
            episodes.append(dict(env_id=i,seed=self.run_seed,vx=float(self.start_vx[i]),stage=STAGE_BY_TILE[int(self.tile_ids[i])].id,recipe=self.recipe,mode=self.mode,checkpoint=self.checkpoint_identity,N=int(n[i]),h=float(h[i]),
                protocol_id=PROTOCOL_ID,code_version=CODE_VERSION,slot_class=('flat_walk','flat_stand','previous_stairs','current_stairs')[int(self.slot_classes[i])],
                initial_pose=self.initial_pose[i].tolist(),success=bool(self.climb.climbed[i]),climb_success=bool(self.climb.climbed[i]),
                parking_entered=bool(self.parking_diagnostic and self.climb.climbed[i]),
                parking_success=(bool(success[i]) if self.parking_diagnostic and self.climb.climbed[i] else None),
                post_climb_failure=bool(self.climb.climbed[i] and failure[i]),
                climb_time=(float(self.climb.climb_step[i])*.02 if self.climb.climbed[i] else None),
                parking_time=((count-int(self.climb.climb_step[i]))*.02 if self.parking_diagnostic and self.climb.climbed[i] else None),
                reason=names[int(reason[i])],highest_valid_step=float(self.highest[i]),
                duration=count*self.step_dt,completion_time=(float(self.climb.climb_step[i])*.02 if self.climb.climbed[i] else None),parking_duration=float(self.stop_steps[i])*.02,mean_vx=float(self.sum_vx[i])/count,
                mean_speed=float(self.sum_speed[i])/count,max_lateral=float(self.max_lateral[i]),max_heading=float(self.max_heading[i]),
                velocity_frame='body',speed_definition='3D norm',mean_vx_world=float(self.sum_vx_world[i])/count,
                mean_horizontal_speed_world=float(self.sum_horizontal_speed_world[i])/count,
                moving_samples=int(self.moving_count[i]),moving_mean_vx=float(self.moving_sum_vx[i])/max(int(self.moving_count[i]),1),
                moving_mean_speed=float(self.moving_sum_speed[i])/max(int(self.moving_count[i]),1),slip=float(self.slip_sum[i])/count,
                nonfoot_contact=float(self.nonfoot_sum[i])/count,joint_limits=float(self.limit_sum[i])/count,peak_torque=float(self.peak_torque[i]),foot_height_default_count=int(self.height_defaults[i]),
                first_nonfoot_time=float(self.first_nonfoot_time[i]),first_nonfoot_link=(self.contacts.link_names[int(self.first_nonfoot_link[i])] if self.first_nonfoot_link[i]>=0 else None),
                first_nonfoot_force=self.first_nonfoot_force[i].tolist(),first_step_front_foot_center_height=self.before_first_foot_height[i].tolist(),
                first_step_support_time=float(self.first_step_support_time[i]),foot_valid_step_times=self.first_foot_times[i].tolist(),foot_step_times=self.foot_step_times[i,:,:int(n[i])].tolist(),
                body_crossed_first=bool(self.body_crossed_first[i]),all_feet_on_platform=bool(self.all_feet_on_platform[i]),max_body_x=float(self.max_body_x[i]),
                stagnation_total=float(self.stagnation_total[i]),stagnation_longest=float(self.stagnation_longest[i]),stop_latched=bool(self.stop[i]),
                reward_sums={key:float(value[i]) for key,value in self.episode_reward_sums.items()}))
        self.last_termination=(success,failure,truncated,reason)
        info={'terminal_observation':terminal,'episodes':episodes,'reward_terms':self.reward_terms,'diagnostics':self.diagnostics_snapshot(0),
            'motion_terms':self.motion_terms(),'step_reward_tensors':self.step_reward_tensors}
        if (reason==9).any() or not torch.isfinite(reward).all():
            raise NumericalFailure('Numerical state/reward failure before reset',{'episodes':episodes,'terminal_observation':{k:v.detach().cpu() for k,v in terminal.items()},'reward':reward.detach().cpu(),'qpos':self.sim.data.qpos.detach().cpu(),'qvel':self.sim.data.qvel.detach().cpu()})
        if len(ids):self._reset_ids(ids)
        obs=self._observations(update=False)
        return obs,reward,success|failure,truncated,info

    def _update_diagnostics(self,local):
        t=self.episode_length_buf*self.step_dt
        actual_step=self.footfall_result.selected_step
        qualified=(self.footfall_result.qualified_count>=2)&(actual_step>0)
        index=(actual_step-1).clamp_min(0)[...,None]
        previous=self.foot_step_times.gather(2,index).squeeze(-1)
        first_by_step=qualified&(previous<0)
        self.foot_step_times.scatter_(2,index,torch.where(first_by_step,t[:,None],previous)[...,None])
        fresh=(self.first_foot_times<0)&qualified
        self.first_foot_times=torch.where(fresh,t[:,None],self.first_foot_times)
        first=(self.first_step_support_time<0)&(qualified&(actual_step==1)).any(-1)
        self.first_step_support_time=torch.where(first,t,self.first_step_support_time)
        before=(local[:,0]<0)&(self.first_step_support_time<0)
        footz=self.robot.data.site_pos_w[:,self.foot_ids[:2],2]-self.origins[self.tile_ids,2,None]
        self.before_first_foot_height[before]=footz[before]
        self.body_crossed_first|=(self.tile_ids>0)&(local[:,0]>=0)
        self.all_feet_on_platform|=self.current_platform.all(-1)
        self.max_body_x=torch.maximum(self.max_body_x,local[:,0])
        # World horizontal speed; diagnostic only, never a reward or termination.
        stagnant=(self.sample_command[:,0]>.1)&(self.robot.data.root_link_lin_vel_w[:,:2].norm(dim=-1)<.05)
        self.stagnation_total+=stagnant*self.step_dt
        self.stagnation_current=torch.where(stagnant,self.stagnation_current+self.step_dt,0.)
        self.stagnation_longest=torch.maximum(self.stagnation_longest,self.stagnation_current)

    def diagnostics_snapshot(self,env_id=0):
        """Detached pre-reset device tensors; caller chooses serialization cadence."""
        i=env_id;d=self.robot.data
        value=dict(env_id=i,stage=STAGE_BY_TILE[int(self.tile_ids[i])].id,recipe=self.recipe,
            step=self.episode_length_buf[i],quaternion=d.root_link_quat_w[i],joint_position=d.joint_pos[i,self.joint_ids],
            joint_velocity=d.joint_vel[i,self.joint_ids],action=self.action_manager.action[i],
            success=self.last_termination[0][i],failure=self.last_termination[1][i],truncated=self.last_termination[2][i],reason_code=self.last_termination[3][i],position=d.root_link_pos_w[i]-self.origins[self.tile_ids[i]],
            command=self.command[i],command_executed=self.sample_command[i],command_timing='command is post-decision; command_executed applied to this physical sample',linear_velocity=d.root_link_lin_vel_w[i],angular_velocity=d.root_link_ang_vel_w[i],
            linear_velocity_body=d.root_link_lin_vel_b[i],angular_velocity_body=d.root_link_ang_vel_b[i],
            velocity_error=d.root_link_lin_vel_b[i,0]-self.sample_command[i,0],velocity_error_post_decision=d.root_link_lin_vel_b[i,0]-self.command[i,0],heading=euler_heading(d.root_link_quat_w)[2][i],
            climb_success=self.climb.climbed[i],climb_event=self.climb_event[i],climb_step=self.climb.climb_step[i],
            platform_support=self.current_platform[i],recent_support_steps=self.climb.last_support_step[i],climb_consecutive_steps=self.climb.consecutive_steps[i],
            foot_position=d.site_pos_w[i,self.foot_ids]-self.origins[self.tile_ids[i]],foot_force=self.contacts.foot_force[i],
            support=self.contacts.support[i],support_step=self.geom_step[self.contacts.support_geom[i].clamp_min(0)],
            nonfoot_force=self.contacts.link_force[i],stop_latched=self.stop[i],parking_steps=self.stop_steps[i],
            footfall_selected=self.footfall_result.selected_step[i],footfall_count=self.footfall_result.qualified_count[i],
            footfall_events=self.footfall_result.events[i],max_body_x=self.max_body_x[i],
            stagnation_total=self.stagnation_total[i],stagnation_longest=self.stagnation_longest[i],
            torque=self.sim.data.qfrc_actuator[i,self.robot.indexing.joint_v_adr][self.joint_ids],
            rewards={k:v[i].detach().clone() for k,v in self.step_reward_tensors.items()})
        return {k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in value.items()}

    def _motion_parking_mask(self):
        """Arrival belongs to moving; only subsequent physical samples park."""
        return self.climb.climbed & (self.episode_length_buf>self.climb.climb_step) & self.parking_diagnostic

    def motion_terms(self):
        """Measured per-control-sample means, separate from episode aggregates."""
        d=self.robot.data
        local=d.root_link_pos_w-self.origins[self.tile_ids]
        yaw=euler_heading(d.root_link_quat_w)[2]
        torque=self.sim.data.qfrc_actuator[:,self.robot.indexing.joint_v_adr][:,self.joint_ids]
        parking=self._motion_parking_mask()
        values=dict(command_vx=self.sample_command[:,0],command_vx_post_decision=self.command[:,0],actual_vx_body=d.root_link_lin_vel_b[:,0],
            actual_vx_world=d.root_link_lin_vel_w[:,0],velocity_error_body=d.root_link_lin_vel_b[:,0]-self.sample_command[:,0],
            lateral=local[:,1].abs(),heading=yaw.abs(),nonfoot=(self.contacts.link_force.norm(dim=-1)>10).float().mean(-1),
            slip=self.robot.data.site_lin_vel_w[:,self.foot_ids,:2].square().sum(-1).mul(self.contacts.found).sum(-1),
            peak_torque=torque.abs().amax(-1),parking=parking.float())
        for j,leg in enumerate(LEGS): values['foot_force_'+leg]=self.contacts.foot_force[:,j].norm(dim=-1)
        for code,name in [(3,'bounds'),(4,'below_support'),(5,'side_fall'),(6,'orientation'),(7,'nonfoot_persistent'),(8,'body_contact'),(9,'numerical_failure')]:
            values['failure_'+name]=(self.last_termination[3]==code).float()
        maximums={name+'_max':values[name] for name in ('peak_torque','lateral','heading')}
        groups={}
        for class_id,class_name in enumerate(('flat_walk','flat_stand','previous_stairs','current_stairs')):
            for phase,phase_mask in (('moving',~parking),('parking',parking)):
                mask=(self.slot_classes==class_id)&phase_mask
                count=int(mask.sum())
                if count:
                    groups[class_name+'/'+phase]=dict(sample_count=count,**{k:float(v[mask].mean()) for k,v in values.items()},**{k:float(v[mask].max()) for k,v in maximums.items()})
        return dict(sample_count=self.num_envs,groups=groups,command_timing='command_vx is executed pre-decision; command_vx_post_decision is next command',
            aggregate_semantics={'peak_torque':'mean of each env maximum absolute joint torque in this sample','*_max':'maximum over sampled environments; accumulate maximum over control samples'},coordinate_frames={'actual_vx_body':'body','actual_vx_world':'world','lateral':'stair local','heading':'world yaw'},
            **{k:float(v.mean()) for k,v in values.items()},**{k:float(v.max()) for k,v in maximums.items()})
