"""Real MuJoCo v08 episodes with causal commands and continuous endpoint stops.

The collision model, PD control, integration, observation and contact extraction
are frozen v07 behavior. Only the SDS task/command/reward layer is replaced.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
import src.tasks.velocity.config.a2  # registry ordering
from . import spec,terrain,events,rewards
from .commands import CommandState
from .contacts import ContactState
from .env_cfg import env_cfg
from .map import HeightMap,ProprioHistory
from .standing import StandWindow
from .speed import StallDiagnostics
from .tasks import TaskMachine,canonical_case
class NumericalFailure(FloatingPointError):
    """Fail-fast engineering exception with actual pre-reset state evidence."""
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics

class V08Env(ManagerBasedRlEnv):
    def __init__(self,num_envs=512,device='cuda:0',seed=42,initial_cases=None,
                 phase='A',stage_iteration=0,evaluation=False,fixed_vx=None,
                 render_mode=None,pre_reset_callback=None,**legacy):
        if str(phase).upper() not in spec.PHASE_ORDER:raise ValueError('unknown v08 phase')
        self.phase=self.stage=str(phase).upper();self.substage=None;self.stage_iteration=int(stage_iteration)
        self.pre_reset_callback=pre_reset_callback;self.run_seed=int(seed);self.evaluation=bool(evaluation);self.fixed_vx=fixed_vx
        if fixed_vx is not None and not spec.SPEED_MIN<=float(fixed_vx)<=spec.SPEED_MAX:raise ValueError('fixed_vx is cruise amplitude [.3,.6]')
        self.recipe=self.reward_version=spec.RECIPE_VERSION;self.protocol=spec.PROTOCOL_ID;self.checkpoint_identity=None;self.generation=0
        ManagerBasedRlEnv.__init__(self,env_cfg(num_envs,seed),device,render_mode)
        self.rng=torch.Generator(device=device).manual_seed(seed)
        # Legacy sampler is restored as source provenance; v08 task choices use
        # a separate documented stream so source restoration cannot overwrite it.
        self._tile_sampler=np.random.default_rng(seed+spec.TERRAIN_RNG_OFFSET)
        self._task_sampler=np.random.default_rng(np.random.SeedSequence([seed,spec.SAMPLER_SEED_OFFSET]))
        self.robot=self.scene['robot'];order=[f'{leg}_{j}_joint' for leg in spec.FOOT_ORDER for j in ('hip','thigh','calf')]
        self.joint_ids,_=self.robot.find_joints(order,preserve_order=True);self.foot_ids,_=self.robot.find_sites(list(spec.FOOT_ORDER),preserve_order=True)
        self.action_term=self.action_manager.get_term('joint_pos');self.action_joint_ids=self.action_term._target_ids
        if self.action_joint_ids.tolist()!=list(self.joint_ids):raise ValueError('action order changed')
        self.specs=terrain.build_terrain_specs();self.origins=torch.tensor([s.frame_origin_world for s in self.specs],device=device,dtype=torch.float32)
        for name,values in (('tile_N',[s.N for s in self.specs]),('tile_h',[s.h for s in self.specs]),('tile_d',[s.step_depth for s in self.specs])):setattr(self,name,torch.tensor(values,device=device))
        geometry=terrain.index_geometry(self.sim.mj_model,self.specs);t=terrain.tensorize_geometry(geometry,device)
        self.bounds=t['bounds'];self.geom_top=t['top'];self.geom_tile=t['tile'];self.geom_step=t['step'];self.geom_step_bounds=t['support_bounds'];self.geom_ids=tuple(geometry.geom_ids)
        self.contacts=ContactState(self.sim,geometry.tile,geometry.top,device);self.map_cache=HeightMap(num_envs,device);self.history_cache=ProprioHistory(num_envs,device)
        self.command=self.command_manager.get_command('twist');self.commands=self.command;self.command_state=CommandState(num_envs,device)
        self.stall=StallDiagnostics(num_envs,device);self.climb=events.PlatformSupportState(num_envs,device);self.standing=StandWindow(num_envs,device);self.reward_computer=rewards.RewardComputer()
        self.tile_ids=torch.zeros(num_envs,dtype=torch.long,device=device)
        for name in ('first_step_acquired','control_body_hit','control_long_hit'):setattr(self,name,torch.zeros(num_envs,dtype=torch.bool,device=device))
        self.parking_location=torch.full((num_envs,),-1,dtype=torch.long,device=device)
        self.bottom_last=torch.full((num_envs,4),-spec.SUPPORT_HISTORY_STEPS-1,dtype=torch.long,device=device);self.bottom_count=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.episode_initial_pose=torch.zeros(num_envs,3,device=device);self._local_x_prev=torch.zeros(num_envs,device=device)
        self.episode_start_local_x=torch.zeros_like(self._local_x_prev);self.episode_command=torch.zeros_like(self._local_x_prev)
        self.max_lateral=torch.zeros_like(self._local_x_prev);self.max_abs_yaw=torch.zeros_like(self._local_x_prev);self.max_stair_abs_yaw=torch.zeros_like(self._local_x_prev)
        self.flat_walk_end=torch.zeros_like(self._local_x_prev);self.flat_max_y=torch.zeros_like(self._local_x_prev);self.flat_max_yaw=torch.zeros_like(self._local_x_prev)
        self.sample_command=self.command.clone();self.current_platform=torch.zeros(num_envs,4,dtype=torch.bool,device=device)
        self.roles=['task']*num_envs;self.frontier_tile_id=spec.tile_id_for('L40');self.level_tick=0
        self.episode_reward_sums={};self.reward_terms={};self.step_reward_tensors={};self._last_reward_terms={};self._last_step_diagnostics={}
        self.stop_windows=[[] for _ in range(num_envs)];self.trace=[[] for _ in range(num_envs)];self.recovery_mode='new_physical_episode';self.recovered_logical_state=None
        if initial_cases is None:
            from .curriculum import CurriculumManager
            schedule=CurriculumManager(num_envs,seed).schedule();self.case_templates=list(schedule.cases);self.roles=list(schedule.roles)
        else:
            if len(initial_cases)!=num_envs:raise ValueError('initial_cases must cover the batch')
            self.case_templates=deepcopy(list(initial_cases))
        self.tasks=TaskMachine(self.case_templates,device)
        self.reset()
    @property
    def executed_c(self):return self.command_state.executed
    @property
    def requested_c(self):return self.command_state.requested
    def _parameters(self):return self.tile_N[self.tile_ids],self.tile_h[self.tile_ids],self.tile_d[self.tile_ids]
    def _rng_state_sha256(self):return hashlib.sha256(self.rng.get_state().cpu().numpy().tobytes()).hexdigest()
    def _sample_case(self,template):
        c=deepcopy(template)
        from .curriculum import spawn_bounds
        bounds=spawn_bounds(self.phase,self.stage_iteration,evaluation=self.evaluation)
        # Bounds from a new schedule apply only when that slot starts a new
        # episode, never by resetting every environment at each B update.
        for key in ('x','y','yaw'):
            explicit=c.get('spawn_'+key,c.get(key))
            if explicit is None:
                lo,hi=c.get('spawn_'+key+'_bounds',bounds[key]);explicit=float(self._task_sampler.uniform(lo,hi))
            c[key]=float(explicit)
        if c.get('spawn_mode')=='top' or canonical_case(c)['initial_direction']<0:
            c['x']=float(c.get('spawn_x') if c.get('spawn_x') is not None else self.specs[int(c.get('tile_id',21))].N*self.specs[int(c.get('tile_id',21))].step_depth+spec.TOP_SPAWN_OFFSET)
        amplitude=c.get('cruise_speed',c.get('speed',c.get('vx',self.fixed_vx)))
        if amplitude is None:
            low,high=c.get('speed_bounds',(spec.SPEED_MIN,spec.SPEED_MAX));amplitude=float(self._task_sampler.uniform(low,high))
        c['speed']=float(amplitude)
        if c.get('trigger_p') is None and c.get('task') in spec.TRIGGER_RANGES:
            c['trigger_p']=float(self._task_sampler.uniform(*c.get('trigger_range',spec.TRIGGER_RANGES[c['task']])))
        return canonical_case(c)
    def _reset_ids(self,ids,**unused):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long).flatten()
        if not len(ids):return
        self.sim.reset(ids);self.scene.reset(ids);self.action_manager.reset(ids)
        cases=[self._sample_case(self.case_templates[i]) for i in ids.tolist()]
        for i,c in zip(ids.tolist(),cases):self.tasks.cases[i]=c;self.stop_windows[i]=[];self.trace[i]=[]
        self.tile_ids[ids]=torch.tensor([c['tile_id'] for c in cases],device=self.device)
        local=torch.tensor([[c['x'],c['y'],0.] for c in cases],device=self.device)
        state=self.robot.data.default_root_state[ids].clone();local[:,2]=state[:,2]
        top=torch.tensor([c['initial_direction']<0 for c in cases],device=self.device)
        local[:,2]+=torch.where(top,self.tile_N[self.tile_ids[ids]]*self.tile_h[self.tile_ids[ids]],0.)
        state[:,:3]=self.origins[self.tile_ids[ids]]+local
        yaw=torch.tensor([c['yaw'] for c in cases],device=self.device);state[:,3:7]=0;state[:,3]=torch.cos(yaw/2);state[:,6]=torch.sin(yaw/2);state[:,7:]=0
        self.robot.write_root_state_to_sim(state,ids);self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[ids],torch.zeros_like(self.robot.data.joint_vel[ids]),env_ids=ids)
        initial=torch.tensor([0. if c['initial_direction']<0 else c['speed'] for c in cases],device=self.device)
        self.command_state.reset(ids,initial)
        top_ids=ids[top]
        if len(top_ids):self.command_state.request(top_ids,-torch.tensor([c['speed'] for c in cases],device=self.device)[top])
        # Prepare k1 for the next Actor observation without changing executed=0.
        active=self.command_state.k<spec.TRANSITION_STEPS
        if len(top_ids):
            self.command_state.k[top_ids]=1;self.command_state.current[top_ids]=self.command_state.start[top_ids]+(self.command_state.requested[top_ids]-self.command_state.start[top_ids])/spec.TRANSITION_STEPS
        self.command[ids]=0;self.command[ids,0]=self.command_state.current[ids];self.episode_command[ids]=initial
        self.parking_location[ids]=-1;self.episode_length_buf[ids]=0;self.bottom_last[ids]=-spec.SUPPORT_HISTORY_STEPS-1;self.bottom_count[ids]=0
        self.contacts.reset(ids);self.stall.reset(ids);self.climb.reset(ids);self.standing.reset(ids);self.reward_computer.reset(ids)
        for name in ('first_step_acquired','control_body_hit','control_long_hit','max_lateral','max_abs_yaw','max_stair_abs_yaw','flat_max_y','flat_max_yaw'):getattr(self,name)[ids]=0
        for value in self.episode_reward_sums.values():value[ids]=0
        self.episode_initial_pose[ids]=torch.stack((local[:,0],local[:,1],yaw),-1)
        self.scene.write_data_to_sim();self.sim.forward();self.sim.sense()
        x=self.robot.data.root_link_pos_w[ids,0]-self.origins[self.tile_ids[ids],0]
        self._local_x_prev[ids]=x;self.episode_start_local_x[ids]=x;self.flat_walk_end[ids]=x;self.tasks.reset(ids,x)
        self.history_cache.reset(self._proprio(),ids);self.map_cache.reset(self.robot.data.root_link_pos_w,events.euler_heading(self.robot.data.root_link_quat_w)[2],self.tile_ids,ids)
        if self.evaluation:
            actual_local=self.robot.data.root_link_pos_w[ids]-self.origins[self.tile_ids[ids]];actual_yaw=events.euler_heading(self.robot.data.root_link_quat_w)[2][ids]
            rows=torch.stack((actual_local[:,0],actual_local[:,1],actual_yaw,self.executed_c[ids],self.requested_c[ids],self.command[ids,0]),-1).detach().cpu().tolist()
            for i,row in zip(ids.detach().cpu().tolist(),rows):
                self.trace[i]=[dict(control_step=0,boundary=True,local_x=row[0],local_y=row[1],yaw=row[2],executed_c=row[3],requested_c=row[4],next_actor_c=row[5],next_requested_c=row[4],transition=False,segment='initial',request_event=row[4] if self.tasks.cases[i]['initial_direction']<0 else None,position_semantics='after_action_boundary')]

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            self.rng.manual_seed(int(seed));self._task_sampler=np.random.default_rng(np.random.SeedSequence([int(seed),spec.SAMPLER_SEED_OFFSET]))
        self._reset_ids(torch.arange(self.num_envs,device=self.device));return self._observations(update=False),{}
    def set_schedule(self,schedule,stage_iteration=0):
        def get(name,default=None):return schedule.get(name,default) if isinstance(schedule,dict) else getattr(schedule,name,default)
        phase=str(get('phase',self.phase)).upper()
        if phase not in spec.PHASE_ORDER:raise ValueError('invalid schedule phase')
        cases=list(get('cases',get('assignments',self.case_templates)))
        if len(cases)!=self.num_envs:raise ValueError('schedule must cover every slot')
        def identity(c):return tuple(c.get(k) for k in ('task','tile_id','script','initial_direction','start_direction','resume_direction'))
        changed=[i for i in range(self.num_envs) if phase!=self.phase or identity(cases[i])!=identity(self.case_templates[i])]
        self.phase=self.stage=phase;self.stage_iteration=int(stage_iteration);self.generation=int(get('generation',self.generation));self.roles=list(get('roles',self.roles));self.case_templates=deepcopy(cases)
        if changed:self._reset_ids(changed)
    def _set_parking_location(self,ids):
        local=self.robot.data.root_link_pos_w[ids]-self.origins[self.tile_ids[ids]];n=self.tile_N[self.tile_ids[ids]];end=n*self.tile_d[self.tile_ids[ids]]
        self.parking_location[ids]=torch.where(n==0,0,torch.where(local[:,0]<0,2,torch.where(local[:,0]<end,3,1)))
    def request_speed(self,ids,target):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long).flatten()
        old=self.command_state.requested[ids].clone()
        self.command_state.request(ids,target)
        stopping=(old.abs()>spec.COMMAND_TOLERANCE)&(self.command_state.requested[ids].abs()<=spec.COMMAND_TOLERANCE)
        if stopping.any():
            stop_ids=ids[stopping];self._set_parking_location(stop_ids);self.standing.reset(stop_ids)
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long).flatten();changed=self.command_state.k[ids]==0;ids=ids[changed]
        if len(ids):
            self.command_state.k[ids]=1;self.command_state.current[ids]=self.command_state.start[ids]+(self.command_state.requested[ids]-self.command_state.start[ids])/spec.TRANSITION_STEPS
            self.command[ids,0]=self.command_state.current[ids]
            if self.evaluation:
                rows=torch.stack((self.command_state.requested[ids],self.command_state.current[ids]),-1).detach().cpu().tolist()
                for i,row in zip(ids.detach().cpu().tolist(),rows):
                    if self.trace[i]:self.trace[i][-1].update(next_requested_c=row[0],next_actor_c=row[1],request_event=row[0])
                    self.tasks.events[i].append(dict(event='request',step=int(self.episode_length_buf[i]),target=row[0]))
        return self.command
    def route_error(self):
        local=self.robot.data.root_link_pos_w-self.origins[self.tile_ids];yaw=events.euler_heading(self.robot.data.root_link_quat_w)[2]
        return torch.stack(((local[:,1]/spec.ROUTE_LATERAL_SCALE).clamp(-spec.ROUTE_LATERAL_CLIP,spec.ROUTE_LATERAL_CLIP),yaw.sin(),yaw.cos()),-1)
    def _foot_surface(self):return terrain.query_surface(self.robot.data.site_pos_w[:,self.foot_ids,:2],self.tile_ids)
    def _foot_heights(self):
        height,valid=self._foot_surface();z=self.robot.data.site_pos_w[:,self.foot_ids,2]-spec.FOOT_RADIUS
        return torch.where(valid,z-height,torch.full_like(z,spec.FOOT_HEIGHT_INVALID)).clamp(spec.FOOT_HEIGHT_MIN,spec.FOOT_HEIGHT_INVALID),~valid
    def _proprio(self):
        d=self.robot.data
        return torch.cat((d.root_link_ang_vel_b,d.projected_gravity_b,self.command,d.joint_pos[:,self.joint_ids]-d.default_joint_pos[:,self.joint_ids],d.joint_vel[:,self.joint_ids],self.action_manager.action),-1)
    def _observations(self,update=True):
        d=self.robot.data;prop=self._proprio()
        if update:self.history_cache.update(prop);self.map_cache.update(d.root_link_pos_w,events.euler_heading(d.root_link_quat_w)[2],self.tile_ids)
        h,_=self._foot_heights();force=quat_apply_inverse(d.root_link_quat_w[:,None].expand(-1,4,-1).reshape(-1,4),self.contacts.foot_force.reshape(-1,3)).reshape(self.num_envs,12)
        torque=self.sim.data.qfrc_actuator[:,self.robot.indexing.joint_v_adr][:,self.joint_ids]
        privileged=torch.cat((d.root_link_lin_vel_b,(self.contacts.foot_force.norm(dim=-1)>spec.SUPPORT_FORCE_GT).float(),force,h,torque),-1)
        return dict(proprio=prop.clone(),history=self.history_cache.history.clone(),heightmap=self.map_cache.map.clone(),privileged=privileged,route_error=self.route_error())
    def get_observations(self):return self._observations(update=False)
    def _task_state(self):
        d=self.robot.data;local=d.root_link_pos_w-self.origins[self.tile_ids];feet=d.site_pos_w[:,self.foot_ids]-self.origins[self.tile_ids,None];roll,pitch,yaw=events.euler_heading(d.root_link_quat_w)
        finite=torch.isfinite(torch.cat((d.root_link_pos_w,d.root_link_quat_w,d.joint_pos,d.joint_vel),-1)).all(-1)
        state=events.evaluate(episode_length=self.episode_length_buf,local=local,roll=roll,pitch=pitch,yaw=yaw,feet_local=feet,tile_ids=self.tile_ids,control_body_hit=self.control_body_hit,body_link_force=self.contacts.link_force,body_link_mask=self.contacts.body_link_mask,nonfoot_duration=self.contacts.nonfoot_duration,control_long_hit=self.control_long_hit,contact_overflow=self.contacts.overflow,nonfinite=~finite,top_forces=self.contacts.top_forces,geom_tile=self.geom_tile,geom_step=self.geom_step,geom_bounds=self.geom_step_bounds,support_state=self.climb,first_step_state=self.first_step_acquired,start_command=torch.tensor([c['speed'] for c in self.tasks.cases],device=self.device))
        return state
    def _standing_frame(self,state):
        d=self.robot.data;n,h,depth=self._parameters();x_top=n*depth;feet=state.feet_local
        # The reducer restricts contacts to the assigned terrain and actual top
        # faces. Ground contact is never inferred from foot position alone.
        ground=((self.geom_tile[None,:]==self.tile_ids[:,None])&(self.geom_step[None,:]==0))
        bottom_support=torch.where(ground[:,None,:],self.contacts.top_forces,0.).sum(-1)>spec.SUPPORT_FORCE_GT
        self.bottom_last,self.bottom_count,_=events.bottom_arrival_update(self.bottom_last,self.bottom_count,self.episode_length_buf,state.local,feet,bottom_support)
        heights,valid=self._foot_surface();actual=self.contacts.top_forces.sum(-1)>spec.SUPPORT_FORCE_GT
        support=actual&valid
        actual_height=self.geom_top[self.contacts.top_forces.argmax(-1)]
        median,known=self.standing.support_height(actual_height,support)
        mid=self.parking_location==3
        relative=d.root_link_pos_w[:,2]-torch.where(mid,median,terrain.query_surface(d.root_link_pos_w[:,:2],self.tile_ids)[0].nan_to_num())
        surface=torch.where(mid,(state.local[:,0]>0)&(state.local[:,0]<x_top)&known,torch.where(self.parking_location==1,state.feet_safe.all(-1),torch.where(self.parking_location==2,(feet[:,:,0]<0).all(-1),valid.all(-1))))
        speed=d.root_link_lin_vel_w[:,:2].norm(dim=-1);omega=d.root_link_ang_vel_w.norm(dim=-1);foot_speed=d.site_lin_vel_w[:,self.foot_ids].norm(dim=-1).mean(-1);four=support.all(-1)
        stable=self.standing.update(d.root_link_pos_w[:,:2],speed,state.roll,state.pitch,state.yaw,omega,relative,foot_speed,four,surface,mid)
        instant=(speed<=spec.STAND_MEAN_SPEED)&(state.roll.abs()<=torch.where(mid,speed.new_tensor(spec.MID_ROLL),speed.new_tensor(spec.LEVEL_ROLL)))&(state.pitch.abs()<=torch.where(mid,speed.new_tensor(spec.MID_PITCH),speed.new_tensor(spec.LEVEL_PITCH)))&(omega<=spec.STAND_OMEGA)&(relative>=torch.where(mid,speed.new_tensor(spec.MID_HEIGHT),speed.new_tensor(spec.LEVEL_HEIGHT)))&four&surface
        instant&=~mid|(state.yaw.abs()<=spec.MID_YAW)
        return stable,instant,mid,surface
    def _reward_frame(self,state,previous):
        d=self.robot.data;base,valid=terrain.query_surface(d.root_link_pos_w[:,:2],self.tile_ids);feet,fvalid=self._foot_surface();false=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        return rewards.RewardFrame(executed_command=self.executed_c,local_x=state.local[:,0],local_x_prev=self._local_x_prev.clone(),body_lin_vel_b=d.root_link_lin_vel_b,body_ang_vel_b=d.root_link_ang_vel_b,gravity_b=d.projected_gravity_b,base_height_above_surface=d.root_link_pos_w[:,2]-torch.where(valid,base,0.),q=d.joint_pos[:,self.joint_ids],qdot=d.joint_vel[:,self.joint_ids],q_target_executed=self.action_term._processed_actions,action=self.action_manager.action,previous_action=previous,foot_position_world=d.site_pos_w[:,self.foot_ids],foot_velocity_world=d.site_lin_vel_w[:,self.foot_ids],foot_force_world=self.contacts.foot_force,contact_raw=self.contacts.raw,contact_filtered=self.contacts.filtered,tile_ids=self.tile_ids,local_y=state.local[:,1],yaw_world=state.yaw,resets=false,success=false,failure=false,foot_surface_height=torch.where(fvalid,feet,0.))
    def _export_window(self,i):
        mid=bool(self.tasks.stop_kind[i]==1);w=spec.MID_WINDOW_STEPS if mid else spec.ENDPOINT_WINDOW_STEPS
        n=min(int(self.standing.count[i]),w)
        c=self.tasks.cases[i];kind='mid' if mid else ('flat' if c['task']=='flat_stop' else 'top' if self.tasks.goal[i]>0 else 'bottom')
        return dict(kind=kind,count=int(self.tasks.hold_steps[i]),xy=self.standing.xy[i,-n:].detach().cpu().tolist(),values=self.standing.values[i,-n:].detach().cpu().tolist())
    def _records(self,state,result,done):
        records=[]
        terminal_ids=done.nonzero().flatten().detach().cpu().tolist()
        if not terminal_ids:return records
        mask_names=tuple(state.failure_masks)
        mask_rows=torch.stack([state.failure_masks[name] for name in mask_names],-1).detach().cpu()
        for i in terminal_ids:
            failure_masks=dict(zip(mask_names,mask_rows[i].tolist()))
            physical_failure=bool(result['failure'][i]) and any(failure_masks.get(name,False) for name in ('side','bounds','body','nonfoot','posture','fall'))
            c=self.tasks.cases[i];windows=deepcopy(self.stop_windows[i]);
            if self.tasks.hold_steps[i]>0:windows.append(self._export_window(i))
            records.append(dict(env_id=i,case_id=c.get('case_id',f'env-{i}'),task=c['task'],script=c.get('script'),tile_id=int(self.tile_ids[i]),phase=self.phase,role=self.roles[i],seed=c.get('seed',self.run_seed),command=c['speed'],cruise_speed=c['speed'],success=bool(result['success'][i]),failure=bool(result['failure'][i]),physical_failure=physical_failure,failure_kind='physical' if physical_failure else 'task' if bool(result['failure'][i]) else None,failure_masks=failure_masks,timeout=result['reason'][i]=='timeout',reason=result['reason'][i],segment=self._record_segments[i],complete=True,raw_complete=self.evaluation,finite=True,terminal_step=int(self.episode_length_buf[i]),control_samples=int(self.episode_length_buf[i]),duration=float(self.episode_length_buf[i])*spec.CONTROL_DT,initial_pose=self.episode_initial_pose[i].cpu().tolist(),flat_progress=dict(net_dx=float(self.flat_walk_end[i]-self.episode_start_local_x[i]),max_abs_y=float(self.flat_max_y[i]),max_abs_yaw=float(self.flat_max_yaw[i]),walk_steps=min(int(self.episode_length_buf[i]),spec.FLAT_WALK_STEPS)),direction_checks=deepcopy(self.tasks.direction_checks[i]),max_stair_abs_yaw=float(self.max_stair_abs_yaw[i]),stop_windows=windows,time_budget_events=deepcopy(self.tasks.budget_events[i]),events=deepcopy(self.tasks.events[i]),trace=deepcopy(self.trace[i]),first_step_valid=bool(self.first_step_acquired[i]),active_samples=int(self.stall.active_samples[i]),stall_samples=int(self.stall.stall_samples[i]),forward_distance=float(state.local[i,0]-self.episode_start_local_x[i]),max_abs_y=float(self.max_lateral[i]),max_abs_yaw_deg=float(self.max_abs_yaw[i])*180/math.pi,reward_sums={k:float(v[i]) for k,v in self.episode_reward_sums.items()},protocol_id=self.protocol,recipe=spec.RECIPE_VERSION))
        return records
    def _record_action_trace(self,state,request):
        raw=torch.stack((state.local[:,0],state.local[:,1],state.yaw,self.executed_c,self.command_state.executed_requested,self.command_state.executed_transition.float(),self.requested_c,self.command[:,0],request),-1).detach().cpu().tolist()
        steps=self.episode_length_buf.detach().cpu().tolist()
        for i,(step,row) in enumerate(zip(steps,raw)):
            event=row[8] if math.isfinite(row[8]) else None
            self.trace[i].append(dict(control_step=int(step),boundary=False,local_x=row[0],local_y=row[1],yaw=row[2],executed_c=row[3],requested_c=row[4],transition=bool(row[5]),next_requested_c=row[6],next_actor_c=row[7],request_event=event,segment=self._record_segments[i],position_semantics='after_action_boundary'))
            if event is not None:self.tasks.events[i].append(dict(event='request',step=int(step),target=event))
    def segment(self,i):
        mode=int(self.tasks.mode[i]);return ('walk','deceleration','hold','done')[mode]
    def step(self,action):
        if not torch.isfinite(action).all():raise NumericalFailure('Nonfinite action',{'action':action.detach().cpu()})
        self.command_state.latch();self.sample_command=self.command.clone();self.sample_command[:,0]=self.executed_c
        previous=self.action_manager.action.clone();self.action_manager.process_action(action)
        self.control_body_hit.zero_();self.control_long_hit.zero_()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter+=1;self.action_manager.apply_action();self.scene.write_data_to_sim();self.sim.step();self.sim.forward();self.scene.update(dt=self.physics_dt);self.contacts.sample(self.tile_ids,physics_dt=self.physics_dt)
            self.control_body_hit|=(self.contacts.link_force[:,self.contacts.body_link_mask].norm(dim=-1)>0).any(-1);self.control_long_hit|=(self.contacts.nonfoot_duration>=spec.NONFOOT_LIMIT_S-1e-6).any(-1)
        self.contacts.sample(self.tile_ids,control=True);self.sim.sense();self.episode_length_buf+=1;self.common_step_counter+=1
        state=self._task_state()
        if state.failure_masks['nonfinite'].any():raise NumericalFailure('Nonfinite physical state',{'qpos':self.sim.data.qpos.detach().cpu()})
        stable,instant,mid,surface=self._standing_frame(state)
        self.max_lateral=torch.maximum(self.max_lateral,state.local[:,1].abs());self.max_abs_yaw=torch.maximum(self.max_abs_yaw,state.yaw.abs())
        stair=(state.local[:,0]>=0)&(state.local[:,0]<=state.n*state.d)&(state.n>0)
        self.max_stair_abs_yaw=torch.maximum(self.max_stair_abs_yaw,torch.where(stair,state.yaw.abs(),0.))
        reverse_case=torch.tensor([c['task'] in ('down_stop','E1','E2','E3','E4') for c in self.tasks.cases],device=self.device)
        yaw_fail=reverse_case&stair&(state.yaw.abs()>spec.REVERSE_YAW)
        top_parking=self.command_state.parking&(self.parking_location==1)
        top_boundary=top_parking&((state.local[:,0]<state.n*state.d)|(state.local[:,0]>state.n*state.d+spec.terrain_spec(21).platform_length)|(state.local[:,1].abs()>spec.BODY_HALF_WIDTH))
        mid_departure=self.command_state.parking&(self.parking_location==3)&(self.executed_c.abs()<=spec.COMMAND_TOLERANCE)&((state.local[:,0]<=0)|(state.local[:,0]>=state.n*state.d))
        physical=state.failure|yaw_fail|top_boundary|mid_departure
        flat_walk=torch.tensor([c['task']=='flat_stop' for c in self.tasks.cases],device=self.device)&(self.episode_length_buf<=spec.FLAT_WALK_STEPS)
        self.flat_walk_end=torch.where(flat_walk,state.local[:,0],self.flat_walk_end);self.flat_max_y=torch.where(flat_walk,torch.maximum(self.flat_max_y,state.local[:,1].abs()),self.flat_max_y);self.flat_max_yaw=torch.where(flat_walk,torch.maximum(self.flat_max_yaw,state.yaw.abs()),self.flat_max_yaw)
        self.tasks.flat_valid&=((~flat_walk)|((state.local[:,1].abs()<=spec.BOTTOM_LATERAL_MAX)&(state.yaw.abs()<=spec.LEVEL_ROLL))).cpu()
        old_mode=self.tasks.mode.clone();old_hold=self.tasks.hold_steps.clone();old_kind=self.tasks.stop_kind.clone()
        result=self.tasks.advance(state.local[:,0],self.climb.consecutive_steps>=spec.BOTTOM_ARRIVAL_FRAMES,self.bottom_count>=spec.BOTTOM_ARRIVAL_FRAMES,stable,physical,self.executed_c,torch.zeros_like(physical),transition=self.command_state.executed_transition,requested=self.command_state.executed_requested)
        self._record_segments=list(self.tasks.action_segments)
        for i in result['failure'].nonzero().flatten().detach().cpu().tolist():
            if bool(state.failure[i]):result['reason'][i]=events.REASON_NAMES[int(state.reason[i])]
            elif bool(yaw_fail[i]):result['reason'][i]='wrong_direction_yaw'
            elif bool(top_boundary[i]):result['reason'][i]='platform_boundary'
            elif bool(mid_departure[i]):result['reason'][i]='mid_stop_drift'
        parking=self.command_state.parking
        self.stall.update(state.local[:,0],self.executed_c,self.command_state.executed_transition,parking,self.first_step_acquired,self.climb.consecutive_steps>=spec.SUCCESS_CONSECUTIVE_FRAMES)
        level=~mid
        reward_result=self.reward_computer(self._reward_frame(state,previous),transition=self.command_state.executed_transition,parking=parking,level=level,stand=instant,arrival=result['arrival'],success=result['success'],failure=result['failure'])
        reward=reward_result.total
        self._last_reward_terms=reward_result.weighted_terms;means=torch.stack([v.mean() for v in self._last_reward_terms.values()]).detach().cpu().tolist();self.reward_terms=dict(zip(self._last_reward_terms,means))
        self.step_reward_tensors={'reward_weighted/'+k:v for k,v in self._last_reward_terms.items()}
        for k,v in self.step_reward_tensors.items():
            if k not in self.episode_reward_sums:self.episode_reward_sums[k]=torch.zeros_like(v)
            self.episode_reward_sums[k]+=v
        if not torch.isfinite(reward).all():raise NumericalFailure('Nonfinite reward',{'reward':reward.detach().cpu()})
        # Snapshot intermediate windows before a restart resets the window.
        restarting=(old_mode!=TaskMachine.WALK)&(self.tasks.mode==TaskMachine.WALK)
        for i in restarting.nonzero().flatten().tolist():
            # The machine has already reset its hold counter; export the actual
            # completed dwell from the pre-advance snapshot.
            window=self._export_window(i);window['count']=int(old_hold[i])+1;window['kind']='mid' if old_kind[i]==1 else 'top';self.stop_windows[i].append(window)
        request=result['request'];ids=torch.isfinite(request).nonzero().flatten()
        self.command_state.request(ids,request[ids])
        new_stop=(request==0)&torch.isfinite(request)
        if new_stop.any():self._set_parking_location(new_stop.nonzero().flatten())
        clear=new_stop|restarting.to(self.device)
        if clear.any():self.standing.reset(clear.nonzero().flatten())
        self.command[:,0]=self.command_state.advance();self.command[:,1:]=0
        if self.evaluation:self._record_action_trace(state,request)

        terminal=self._observations(update=True);done=result['success']|result['failure'];episodes=self._records(state,result,done)
        self._last_step_diagnostics=dict(executed_c=self.executed_c.clone(),requested_c=self.requested_c.clone(),mode=self.tasks.mode.clone(),hold_steps=self.tasks.hold_steps.clone(),active_samples=self.stall.active_samples.clone(),stall_samples=self.stall.stall_samples.clone(),stall_dx_1s=self.stall.dx_1s.clone())
        info=dict(episodes=episodes,terminal_observation={k:v.clone() for k,v in terminal.items()},success=result['success'].clone(),failure=result['failure'].clone(),timeout=torch.tensor([r=='timeout' for r in result['reason']],device=self.device),reason=result['reason'],command_executed=self.sample_command.clone(),command_requested=self.command_state.executed_requested.clone(),reset_mask=done.clone(),reward_terms=dict(self.reward_terms),step_reward_tensors=self.step_reward_tensors,phase=self.phase,generation=self.generation,route_error_terminal=terminal['route_error'],diagnostics=self._last_step_diagnostics)
        self._local_x_prev.copy_(state.local[:,0]);self.last_termination=(result['success'].clone(),result['failure'].clone(),info['timeout'].clone(),result['reason'])
        if done.any():
            if self.pre_reset_callback is not None:self.pre_reset_callback(self,info)
            self._reset_ids(done.nonzero().flatten())
        return self._observations(update=False),reward,done,torch.zeros_like(done),info
    def last_reward_terms(self):return self._last_reward_terms
    def diagnostics(self):return self._last_step_diagnostics
    def diagnostics_snapshot(self,env_id=0):return {k:v[env_id].detach().cpu().tolist() for k,v in self._last_step_diagnostics.items()}
    def reset_task_sampler(self,seed=42):
        # No physics mutation: discard provisional construction draws only.
        self._task_sampler=np.random.default_rng(np.random.SeedSequence([int(seed),spec.SAMPLER_SEED_OFFSET]))
    def rng_state(self):return {'torch':self.rng.get_state().clone()}
    def load_rng_state(self,value):
        state=value['torch'] if isinstance(value,dict) else value;self.rng.set_state(state.cpu())
    def sampler_rng_state(self):return deepcopy(self._tile_sampler.bit_generator.state)
    def load_sampler_rng_state(self,value):self._tile_sampler.bit_generator.state=deepcopy(value)
    def state_dict(self):
        return dict(version=spec.CODE_VERSION,phase=self.phase,stage_iteration=self.stage_iteration,generation=self.generation,roles=list(self.roles),common_step_counter=int(self.common_step_counter),case_templates=deepcopy(self.case_templates),rng=self.rng_state(),sampler_rng=self.sampler_rng_state(),task_sampler_rng=deepcopy(self._task_sampler.bit_generator.state),command=self.command_state.state_dict(),tasks=self.tasks.state_dict(),standing=self.standing.state_dict(),reward=self.reward_computer.state_dict(),tensors={k:v.clone() for k,v in vars(self).items() if k in ('parking_location','tile_ids','episode_length_buf','bottom_last','bottom_count','_local_x_prev','episode_start_local_x','flat_walk_end','flat_max_y','flat_max_yaw','max_stair_abs_yaw','max_lateral','max_abs_yaw','episode_initial_pose','episode_command','current_platform')},contacts={k:v.clone() for k,v in vars(self.contacts).items() if isinstance(v,torch.Tensor)},contact_inner={k:v.clone() for k,v in vars(self.contacts.inner).items() if isinstance(v,torch.Tensor)},stall=self.stall.state_dict(),climb={k:v.clone() for k,v in vars(self.climb).items() if isinstance(v,torch.Tensor)},history=self.history_cache.history.clone(),history_initialized=self.history_cache.initialized.clone(),map={k:v.clone() for k,v in vars(self.map_cache).items() if isinstance(v,torch.Tensor)},stop_windows=deepcopy(self.stop_windows),trace=deepcopy(self.trace),reward_sums={k:v.clone() for k,v in self.episode_reward_sums.items()},physics_restorable=False)
    def load_state_dict(self,state,restore_physics=False):
        self.phase=self.stage=state['phase'];self.stage_iteration=int(state['stage_iteration']);self.generation=int(state['generation']);self.roles=list(state['roles']);self.common_step_counter=int(state['common_step_counter']);self.case_templates=deepcopy(state['case_templates']);self.load_rng_state(state['rng']);self.load_sampler_rng_state(state['sampler_rng']);self._task_sampler.bit_generator.state=deepcopy(state['task_sampler_rng'])
        if not restore_physics:
            self.recovered_logical_state=deepcopy(state);self.recovery_mode='fresh_physical_episode';self.reset();return
        # Only a caller that separately restored matching physical arrays may
        # request this branch. It restores logical state, never the simulator.
        self.command_state.load_state_dict(state['command']);self.tasks.load_state_dict(state['tasks']);self.standing.load_state_dict(state['standing']);self.reward_computer.load_state_dict(state['reward'],self.num_envs,self.device)
        for k,v in state['tensors'].items():getattr(self,k).copy_(v.to(self.device))
        for obj,key in ((self.contacts,'contacts'),(self.contacts.inner,'contact_inner'),(self.climb,'climb'),(self.map_cache,'map')):
            for k,v in state[key].items():getattr(obj,k).copy_(v.to(self.device))
        self.stall.load_state_dict(state['stall']);self.history_cache.history.copy_(state['history'].to(self.device));self.history_cache.initialized.copy_(state['history_initialized'].to(self.device));self.stop_windows=deepcopy(state['stop_windows']);self.trace=deepcopy(state['trace']);self.episode_reward_sums={k:v.to(self.device).clone() for k,v in state['reward_sums'].items()};self.command[:,0]=self.command_state.current;self.command[:,1:]=0;self.recovery_mode='logical_state_with_caller_physics'
