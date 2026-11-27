"""Independent v1.0 vector environment over the real nominal MuJoCo/Warp asset.

Only the asset/config and low-level contact force extraction are reused. No old
runner, course, termination, reward or episode state enters this environment.
"""
import copy
import hashlib
import json
import math
import random
import numpy as np
import torch
from . import cases as case_module
from .geometry import Terrain, primitives, query_local, safe_top, route_height, corridor, terrain_from_case
from .tasks import TaskState, allocate_roles, combinations, CRUISE
from .contacts import ContactFilter

FOOT_ORDER=('FL','FR','RL','RR')


def _generator(terrains,origins,seed):
    import mujoco
    from mjlab.terrains.terrain_generator import SubTerrainCfg,TerrainGeneratorCfg,TerrainGeometry,TerrainOutput
    class Atlas(SubTerrainCfg):
        def function(self,difficulty,spec,rng):
            body=spec.body('terrain');geometries=[]
            shift=np.array((16.,6.*len(terrains),0.))
            for tid,t in enumerate(terrains):
                for p in primitives(t):
                    g=body.add_geom(name=f'v10_{tid}_{p.name}',type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=np.array(p.center)+origins[tid]+shift,size=p.half,
                                    contype=1,conaffinity=1,condim=3,friction=(.6,.005,.0001),
                                    rgba=(.3,.5,.7,1.))
                    geometries.append(TerrainGeometry(geom=g,color=(.3,.5,.7,1.)))
            return TerrainOutput(origin=shift,geometries=geometries)
    return TerrainGeneratorCfg(seed=seed,size=(32.,12.*len(terrains)),num_rows=1,num_cols=1,
        curriculum=False,border_width=0.,sub_terrains={'v07_atlas':Atlas(proportion=1.)},color_scheme='none')


class V10Env:
    def __init__(self,num_envs=1024,course='D1',seed=42,device='cuda:0',cases=None,auto_reset=True):
        from mjlab.envs import ManagerBasedRlEnv
        from src.tasks.velocity.v07.env_cfg import env_cfg
        from src.tasks.velocity.v07.contacts import ActualContacts
        from src.tasks.velocity.v09.map import HeightMap,ProprioHistory
        from .rewards import AirtimeTracker
        self.num_envs=int(num_envs);self.course=course;self.seed=int(seed);self.device=torch.device(device)
        self.auto_reset=bool(auto_reset);self.terminal_done=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self.terminal_records={}
        self.rng=random.Random(seed);self.fixed_cases=copy.deepcopy(cases)
        if cases is not None and len(cases)!=num_envs:raise ValueError('cases must cover every environment')
        counts=allocate_roles(self.num_envs)
        self.roles=([c['task'] for c in cases] if cases is not None else [role for role,n in counts.items() for _ in range(n)])
        self.role_cursors={r:0 for r in counts};self.episode_ids=[-1]*num_envs
        source=case_module.groups('D3' if course in ('P0','P1') else course)
        self.terrains=list(dict.fromkeys([terrain_from_case(c) for c in source]+[terrain_from_case(c) for c in (cases or [])]))
        self._origins=np.asarray([(0.,12.*i,0.) for i in range(len(self.terrains))])
        self.origins=torch.tensor(self._origins,dtype=torch.float32,device=self.device)
        cfg=env_cfg(num_envs,seed);cfg.scene.terrain.terrain_generator=_generator(self.terrains,self._origins,seed)
        self.physics=ManagerBasedRlEnv(cfg,str(device));self.cfg=self.physics.cfg
        self.sim=self.physics.sim;self.scene=self.physics.scene;self.action_manager=self.physics.action_manager
        self.robot=self.scene['robot'];self.action_term=self.action_manager.get_term('joint_pos')
        names=[f'{leg}_{j}_joint' for leg in FOOT_ORDER for j in ('hip','thigh','calf')]
        self.joint_ids,_=self.robot.find_joints(names,preserve_order=True)
        self.foot_ids,_=self.robot.find_sites(list(FOOT_ORDER),preserve_order=True)
        if self.action_term._target_ids.tolist()!=list(self.joint_ids):raise RuntimeError('action joint order mismatch')
        model=self.sim.mj_model
        geom_tile=np.full(model.ngeom,-1,dtype=np.int64);geom_top=np.zeros(model.ngeom)
        self.primitive_geom={};self.platform_geom={}
        for tid,t in enumerate(self.terrains):
            for p in primitives(t):
                expected=np.asarray(p.center)+self._origins[tid]
                found=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('terrain_') and np.allclose(model.geom_pos[g],expected,atol=1e-5) and np.allclose(model.geom_size[g],p.half,atol=1e-5)]
                if len(found)!=1:raise RuntimeError('missing/duplicate real primitive '+p.name)
                gid=found[0];geom_tile[gid]=tid;geom_top[gid]=model.geom_pos[gid,2]+model.geom_size[gid,2]
                expected=np.asarray(p.center)+self._origins[tid]
                if not np.allclose(model.geom_pos[gid],expected,atol=1e-5):
                    raise RuntimeError(f'collision/map primitive pose mismatch: {model.geom_pos[gid]} != {expected}')
                self.primitive_geom[(tid,p.name)]=gid
                if p.name=='platform':self.platform_geom[tid]=gid
        terrain_body=model.body('terrain').id
        if int((model.geom_bodyid==terrain_body).sum())!=len(self.primitive_geom):raise RuntimeError('unexpected collision primitive')
        self.contacts=ActualContacts(self.sim,geom_tile,geom_top,str(device))
        self.filter=ContactFilter(num_envs,self.device);self.airtime=AirtimeTracker(num_envs,self.device)
        self.map_cache=HeightMap(num_envs,self.device,self.query_surface)
        self.history_cache=ProprioHistory(num_envs,self.device)
        self.tile_ids=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.command=torch.zeros(num_envs,3,device=self.device)
        self.states=[None]*num_envs;self.current_cases=[None]*num_envs
        self.metrics=[None]*num_envs
        self.previous_qdot=torch.zeros(num_envs,12,device=self.device)
        self.control_steps=0;self._observation=None
        self.contact_state={k:torch.zeros(num_envs,4,dtype=torch.bool,device=self.device) for k in ('contact_raw','reward_contact','safe_support')}
        self.reset()

    def __getattr__(self,name):
        if name=='physics':raise AttributeError(name)
        return getattr(self.physics,name)

    def query_surface(self,points,tile_ids):
        ids=tile_ids
        while ids.ndim<points.ndim-1:ids=ids.unsqueeze(-1)
        ids=ids.expand(points.shape[:-1]);out=torch.zeros(points.shape[:-1],device=points.device);valid=torch.zeros_like(out,dtype=torch.bool)
        for tid,t in enumerate(self.terrains):
            mask=ids==tid
            if bool(mask.any()):
                z,v=query_local(points[mask]-self.origins[tid,:2],t);out[mask]=z;valid[mask]=v
        return out,valid

    def _new_case(self,i):
        if self.fixed_cases is not None:return copy.deepcopy(self.fixed_cases[i])
        role=self.roles[i];variants=combinations(role,self.course)
        row=variants[self.role_cursors[role]%len(variants)].copy();self.role_cursors[role]+=1
        row['spawn_x']=(-1.+self.rng.uniform(-.1,.1) if role=='S_ROUTE' else 1.+self.rng.uniform(-.05,.05))
        row['initial_joint_offsets']=[self.rng.uniform(-.01,.01) for _ in range(12)]
        row['max_steps']=case_module.max_steps(row);row['episode_id']=self.episode_ids[i]+1
        row['case_sha256']=case_module.digest(row)
        return row

    def _reset_ids(self,ids):
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        if not ids.numel():return
        self.terminal_done[ids]=False
        for i in ids.tolist():self.terminal_records.pop(i,None)
        self.sim.reset(ids);self.scene.reset(ids);self.action_manager.reset(ids)
        root=self.robot.data.default_root_state[ids].clone();root[:,3:]=0;root[:,3]=1.
        joint=self.robot.data.default_joint_pos[ids].clone()
        for j,i in enumerate(ids.tolist()):
            row=self._new_case(i);t=terrain_from_case(row);tid=self.terrains.index(t)
            self.tile_ids[i]=tid;self.current_cases[i]=row;self.episode_ids[i]+=1
            x=float(row['spawn_x']);z=0. if t.kind in ('flat','stairs') else t.H
            root[j,:3]=self.origins[tid]+torch.tensor([x,0.,z+.4],device=self.device)
            joint[j,self.joint_ids]+=torch.tensor(row['initial_joint_offsets'],device=self.device)
            self.states[i]=TaskState(row,self.course,x)
            self.command[i]=0;self.command[i,0]=self.states[i].command
            # Compact numeric traces avoid millions of Python dicts while retaining exact windows.
            from .evaluation import EpisodeMetrics
            self.metrics[i]=EpisodeMetrics(row)
        self.robot.write_root_state_to_sim(root,ids)
        self.robot.write_joint_state_to_sim(joint,torch.zeros_like(joint),env_ids=ids)
        self.contacts.reset(ids);self.filter.reset(ids);self.airtime.reset(ids)
        for value in self.contact_state.values():value[ids]=False
        self.previous_qdot[ids]=0
        self.scene.write_data_to_sim();self.sim.forward();self.sim.sense()
        prop=self._proprio();self.history_cache.reset(prop,ids)
        from src.tasks.velocity.v07.events import euler_heading
        yaw=euler_heading(self.robot.data.root_link_quat_w)[2]
        self.map_cache.reset(self.robot.data.root_link_pos_w,yaw,self.tile_ids,ids)
        local_feet=(self.robot.data.site_pos_w[:,self.foot_ids]-self.origins[self.tile_ids,None]).detach().cpu().numpy()
        for i in ids.tolist():
            if self.roles[i]=='S_ROUTE':
                f=local_feet[i]
                self.states[i].g_level=float(bool(((f[:,0]>=-2)&(f[:,0]<0)&(abs(f[:,1])<=2)).all()))

    def reset(self):
        self._reset_ids(torch.arange(self.num_envs,device=self.device));self._build_observation(False)
        return self.observe()

    def _proprio(self):
        d=self.robot.data
        return torch.cat((d.root_link_ang_vel_b,d.projected_gravity_b,self.command,
            d.joint_pos[:,self.joint_ids]-d.default_joint_pos[:,self.joint_ids],d.joint_vel[:,self.joint_ids],self.action_manager.action),-1)

    def _build_observation(self,advance):
        from mjlab.utils.lab_api.math import quat_apply_inverse
        from src.tasks.velocity.v07.events import euler_heading
        d=self.robot.data;prop=self._proprio();yaw=euler_heading(d.root_link_quat_w)[2]
        if advance:
            self.history_cache.update(prop);self.map_cache.update(d.root_link_pos_w,yaw,self.tile_ids)
        feet=d.site_pos_w[:,self.foot_ids];h,valid=self.query_surface(feet[:,:,:2],self.tile_ids)
        clearance=torch.where(valid,feet[:,:,2]-.032-h,1.5).clamp(-.05,1.5)
        force=quat_apply_inverse(d.root_link_quat_w[:,None].expand(-1,4,-1).reshape(-1,4),self.contacts.foot_force.reshape(-1,3)).reshape(self.num_envs,12)
        torque=self.sim.data.qfrc_actuator[:,self.robot.indexing.joint_v_adr][:,self.joint_ids]
        privileged=torch.cat((d.root_link_lin_vel_b,self.contact_state['contact_raw'].float(),force,clearance,torque),-1)
        local=d.root_link_pos_w-self.origins[self.tile_ids]
        route=torch.stack(((local[:,1]/.2).clamp(-4,4),yaw.sin(),yaw.cos()),-1)
        self._observation=dict(proprio=prop,history=self.history_cache.history,heightmap=self.map_cache.map,route_error=route,privileged=privileged)
        if not all(torch.isfinite(v).all() for v in self._observation.values()):raise FloatingPointError('nonfinite observation')

    def observe(self):return {k:v.detach().clone() for k,v in self._observation.items()}
    def metadata(self):
        return dict(task=list(self.roles),teacher=[s.teacher for s in self.states],case=[c['case_sha256'] for c in self.current_cases],
                    episode=list(self.episode_ids),control_step=[s.step for s in self.states])
    def state_dict(self):
        return dict(rng=self.rng.getstate(),role_cursors=dict(self.role_cursors),episode_ids=list(self.episode_ids),control_steps=self.control_steps)
    def load_state_dict(self,state,allow_num_env_change=False):
        changed=len(state['episode_ids'])!=self.num_envs
        if changed and not allow_num_env_change:raise ValueError('environment count migration must be explicit')
        self.rng.setstate(state['rng']);self.role_cursors=dict(state['role_cursors']);self.episode_ids=([-1]*self.num_envs if changed else list(state['episode_ids']));self.control_steps=state['control_steps']
        return self.reset()
    def close(self):self.physics.close()

    @torch.no_grad()
    def step(self,actions):
        from src.tasks.velocity.v07.events import euler_heading
        from src.tasks.velocity.v07.contacts import support_from_top_forces
        from .rewards import compute_reward
        if actions.shape!=(self.num_envs,12) or not torch.isfinite(actions).all():raise FloatingPointError('invalid actions')
        previous=self.action_manager.action.detach().clone()
        d=self.robot.data
        before_x=(d.root_link_pos_w-self.origins[self.tile_ids])[:,0].clone()
        phase=[s.phase for s in self.states]
        metadata=self.metadata()
        command=self.command[:,0].clone()
        level=torch.tensor([s.g_level for s in self.states],device=self.device)
        cruise=torch.tensor([p in CRUISE for p in phase],device=self.device)
        padding=self.terminal_done.clone()
        actions=torch.where(padding[:,None],torch.zeros_like(actions),actions)
        self.action_manager.process_action(actions)
        actual=self.action_manager.action.detach().clone()
        body_hit=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);long_hit=body_hit.clone()
        for _ in range(4):
            self.physics._sim_step_counter+=1
            self.action_manager.apply_action();self.scene.write_data_to_sim();self.sim.step();self.sim.forward()
            self.scene.update(dt=.005)
            self.contacts.sample(self.tile_ids,physics_dt=.005)
            body_hit|=(self.contacts.link_force[:,self.contacts.body_link_mask].norm(dim=-1)>0).any(-1)
            long_hit|=(self.contacts.nonfoot_duration>=.1-1e-6).any(-1)
        self.contacts.sample(self.tile_ids,control=True);self.sim.sense()
        self.control_steps+=1;self.physics.common_step_counter+=1
        if not torch.isfinite(self.sim.data.qpos).all() or not torch.isfinite(self.sim.data.qvel).all():raise FloatingPointError('nonfinite physics state')
        d=self.robot.data;local=d.root_link_pos_w-self.origins[self.tile_ids]
        feet=d.site_pos_w[:,self.foot_ids];feet_local=feet-self.origins[self.tile_ids,None]
        roll,pitch,yaw=euler_heading(d.root_link_quat_w)
        base_h,base_valid=self.query_surface(d.root_link_pos_w[:,:2],self.tile_ids)
        foot_h,foot_valid=self.query_surface(feet[:,:,:2],self.tile_ids)
        upward,gid=support_from_top_forces(self.contacts.top_forces,self.contacts.geom_top,self.contacts.tile_by_geom)
        support_h=self.contacts.geom_top[gid]
        inside=torch.zeros(self.num_envs,4,dtype=torch.bool,device=self.device)
        top_condition=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        platform_projection=torch.zeros_like(top_condition);approach_level=torch.zeros_like(top_condition)
        bounds=torch.ones_like(top_condition);route_h=torch.zeros(self.num_envs,device=self.device)
        foot_stairs=torch.zeros_like(inside)
        for tid,t in enumerate(self.terrains):
            mask=self.tile_ids==tid
            if not bool(mask.any()):continue
            pts=feet_local[mask,:,:2]
            inside[mask]=safe_top(pts,support_h[mask],t)
            bounds[mask]=corridor(local[mask,:2],t)
            route_h[mask]=route_height(local[mask,0],t)
            if t.kind=='stairs':
                front=.2*t.N
                projection=(pts[:,:,0]>=front)&(pts[:,:,0]<=front+14)&(pts[:,:,1].abs()<=t.W/2)
                platform_projection[mask]=projection.all(-1)
                inset=(pts[:,:,0]>=front+.042)&(pts[:,:,0]<=front+14-.042)&(pts[:,:,1].abs()<=t.W/2-.042)
                platform_force=self.contacts.top_forces[mask,:,self.platform_geom[tid]]
                top_condition[mask]=inset.all(-1)&(feet_local[mask,:,2]-.032>=front-.02).all(-1)&((platform_force>5).sum(-1)>=2)
                approach_level[mask]=(pts[:,:,0]<0).all(-1)&(pts[:,:,0]>=-2).all(-1)&(pts[:,:,1].abs()<=2).all(-1)
                # Classify by actual measured surface, not task label or body phase.
                foot_stairs[mask]=foot_valid[mask]&(foot_h[mask]>0)&(foot_h[mask]<=front)&(pts[:,:,0]>=0)&(pts[:,:,0]<front)
        self.contact_state=self.filter.update(self.contacts.foot_force,upward,inside)
        fv=d.site_lin_vel_w[:,self.foot_ids]
        clearance=torch.where(base_valid,d.root_link_pos_w[:,2]-base_h,torch.full_like(base_h,-1.5))
        stable=(d.root_link_lin_vel_w[:,:2].norm(dim=-1)<=.05)&(roll.abs()<=math.radians(10))&(pitch.abs()<=math.radians(10))
        stable&=(d.root_link_ang_vel_b.norm(dim=-1)<=.25)&(clearance>=.30)&self.contact_state['safe_support'].all(-1)&(fv[:,:,:2].norm(dim=-1).mean(-1)<=.05)
        physical_failure=body_hit|long_hit|(roll.abs()>math.radians(70))|(pitch.abs()>math.radians(70))|(d.root_link_pos_w[:,2]<route_h-.5)|~bounds
        # One batched CPU transfer provides all scalar task and quality measurements.
        packed=torch.cat((local,d.root_link_lin_vel_b[:,:2],d.root_link_lin_vel_w[:,:2],
            roll[:,None]*180/math.pi,pitch[:,None]*180/math.pi,clearance[:,None],
            self.contact_state['safe_support'].float(),stable[:,None].float(),
            top_condition[:,None].float(),physical_failure[:,None].float(),platform_projection[:,None].float(),approach_level[:,None].float(),command[:,None]),-1).cpu().numpy()
        tops=[];successes=[];failures=[];timeouts=[];done_list=[];episodes=[];steps=[];next_commands=[]
        for i,s in enumerate(self.states):
            a=packed[i]
            if s.done:
                tops.append(False);successes.append(False);failures.append(False);timeouts.append(False);done_list.append(False);steps.append(s.step);next_commands.append(0.)
                continue
            out=s.advance(float(a[0]),bool(a[15]),bool(a[16]),bool(a[17]),bool(a[18]))
            quality_frame=dict(phase=phase[i],command=float(a[19]),position=a[:2],v_body=a[3:5],v_world=a[5:7],
                roll_deg=float(a[7]),pitch_deg=float(a[8]),clearance=float(a[9]),safe_support=a[10:14].astype(bool),stable=bool(a[14]),
                top_event=out['top'],failure=out['failure'],timeout=out['timeout'])
            self.metrics[i].update(quality_frame)
            quality=None;success=False
            if out['done']:
                quality=self.metrics[i].finish(completed=out['finished'])
                success=out['finished'] and quality['passed']
                if not success:out['failure']=True
                episodes.append(dict(env_id=i,case_sha256=self.current_cases[i]['case_sha256'],task=self.roles[i],
                    episode=self.episode_ids[i],steps=s.step,success=success,failure=out['failure'],timeout=out['timeout'],quality=quality))
                self.terminal_records[i]=copy.deepcopy(episodes[-1])
            tops.append(out['top'] and not out['failure']);successes.append(success);failures.append(out['failure']);timeouts.append(out['timeout']);done_list.append(out['done']);steps.append(s.step)
            next_commands.append(0. if out['done'] and not self.auto_reset else s.command)
        self.command[:,0]=torch.tensor(next_commands,device=self.device)
        tensor=lambda x,dtype=torch.bool:torch.tensor(x,device=self.device,dtype=dtype)
        failure=tensor(failures);top=tensor(tops);success=tensor(successes);done=tensor(done_list)
        self.terminal_done|=done
        hard=torch.as_tensor(self.sim.mj_model.jnt_range[self.robot.indexing.joint_ids.cpu().numpy()],device=self.device,dtype=torch.float32)[self.joint_ids]
        frame=dict(body_velocity=d.root_link_lin_vel_b.clone(),world_velocity=d.root_link_lin_vel_w.clone(),
            angular_velocity=d.root_link_ang_vel_b.clone(),projected_gravity=d.projected_gravity_b.clone(),
            command=command,g_level=level,base_z=d.root_link_pos_w[:,2].clone(),base_surface_height=base_h,base_surface_valid=base_valid,
            q=d.joint_pos[:,self.joint_ids].clone(),q0=d.default_joint_pos[:,self.joint_ids].clone(),qdot=d.joint_vel[:,self.joint_ids].clone(),
            previous_qdot=self.previous_qdot.clone(),action=actual,previous_action=previous,q_applied=self.action_term._processed_actions.clone(),hard_limits=hard,
            **{k:v.clone() for k,v in self.contact_state.items()},foot_velocity=fv.clone(),foot_site_z=feet[:,:,2].clone(),foot_surface_height=foot_h,foot_surface_valid=foot_valid,foot_is_stairs=foot_stairs,
            airtime_reward=self.airtime.update(self.contact_state['reward_contact'],cruise),local_y=local[:,1].clone(),yaw_error=yaw.clone(),
            heading_speed=(d.root_link_lin_vel_w[:,0]*yaw.cos()+d.root_link_lin_vel_w[:,1]*yaw.sin()),stable=stable,cruise=cruise,
            delta_x=local[:,0]-before_x,top=top,success=success,failure=failure,timeout=tensor(timeouts),phase=phase,
            position=local.clone(),v_body=d.root_link_lin_vel_b.clone(),v_world=d.root_link_lin_vel_w.clone(),roll_deg=roll*180/math.pi,pitch_deg=pitch*180/math.pi,
            clearance=clearance,step=tensor(steps,torch.long),top_event=top,feet_local=feet_local.clone(),foot_force=self.contacts.foot_force.clone(),upward_force=upward,
            action_clip=(actual-actions).abs().amax(-1),padding=padding)
        # Recording only: expose already-computed physical signals without changing
        # termination, reward, command, observation, or task state.
        frame.update(yaw=yaw.clone(),foot_position_local=feet_local.clone(),top_upward_force=upward.clone(),
            failure_masks=dict(body_hit=body_hit.clone(),long_hit=long_hit.clone(),
                attitude=(roll.abs()>math.radians(70))|(pitch.abs()>math.radians(70)),
                route_fall=d.root_link_pos_w[:,2]<route_h-.5,bounds=~bounds,timeout=frame['timeout'].clone()))
        reward,terms=compute_reward(frame)
        reward=torch.where(padding,torch.zeros_like(reward),reward)
        terms={kind:{k:torch.where(padding,torch.zeros_like(v),v) for k,v in values.items()} for kind,values in terms.items()}
        self.previous_qdot=frame['qdot'].clone()
        self._build_observation(True)
        terminal=self.observe()
        info=dict(frame=frame,terminal_observation=terminal,metadata=metadata,episodes=episodes,reward_terms=terms,
                  success=success,failure=failure,timeout=frame['timeout'],physical_samples=self.num_envs,
                  failure_masks=dict(body=body_hit.clone(),nonfoot=long_hit.clone(),posture=(roll.abs()>math.radians(70))|(pitch.abs()>math.radians(70)),
                      fall=d.root_link_pos_w[:,2]<route_h-.5,bounds=~bounds))
        if self.auto_reset and bool(done.any()):
            self._reset_ids(done.nonzero().flatten());self._build_observation(False)
        return self.observe(),reward,(done if self.auto_reset else self.terminal_done.clone()),info
