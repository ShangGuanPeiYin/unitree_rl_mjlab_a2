"""Real rollout collection and phase workers; no implicit gate advancement."""
import copy
import hashlib
import json
import os
import uuid
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from . import spec
from .checkpoint import BudgetLedger,atomic_json,checkpoint_load,checkpoint_save,restore_rng,sha256
from .model import V10ActorCritic
from .teachers import DualTeacher
from .replay import TaskReservoir,concat_batches,snapshot
from .dagger import DAggerLearner
from .ppo import PPOLearner,gae

def frozen_teachers(device):
    # Always the SDS-identified immutable source, never a resumed student.
    return DualTeacher(V10ActorCritic.from_stairs(device=device)).to(device)

def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def physical_contract(env):
    m=env.sim.mj_model
    # Ownership is taken from the robot entity, never from global body-array size.
    # Terrain is a static body with inferred mass/inertia and varies by course.
    robot_bodies=[int(i) for i in env.robot.indexing.body_ids]
    robot_body_set=set(robot_bodies)
    joints=[i for i in range(m.njnt) if int(m.jnt_bodyid[i]) in robot_body_set]
    joint_set=set(joints)
    dofs=[i for i in range(m.nv) if int(m.dof_jntid[i]) in joint_set]
    actuators=[int(i) for i in env.robot.indexing.ctrl_ids]
    robot_geoms=[int(i) for i in env.robot.indexing.geom_ids]
    fields=('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','dof_damping','dof_armature','dof_frictionloss',
            'jnt_range','jnt_axis','jnt_pos','jnt_type','actuator_gainprm','actuator_biasprm','actuator_gear',
            'actuator_ctrlrange','actuator_forcerange','actuator_ctrllimited','actuator_forcelimited',
            'geom_size','geom_pos','geom_quat','geom_type','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_contype','geom_conaffinity')
    ownership={'body':robot_bodies,'dof':dofs,'jnt':joints,'actuator':actuators,'geom':robot_geoms}
    arrays={key:np.asarray(getattr(m,key))[ownership[key.split('_')[0]]].tolist() for key in fields}
    options={key:np.asarray(getattr(m.opt,key)).tolist() for key in ('gravity','solver','integrator','iterations','tolerance','ls_iterations','ls_tolerance','cone','jacobian','impratio')}
    contract=dict(arrays=arrays,options=options,robot_body_names=[m.body(i).name for i in robot_bodies],
        physics_dt=float(m.opt.timestep),control_dt=spec.DT,decimation=4,
        joint_ids=list(env.joint_ids),joint_names=[f'{leg}_{j}_joint' for leg in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')],
        action_scale=.25,action_clip=None,actuator_ctrlrange=arrays['actuator_ctrlrange'],map_shape=list(spec.MAP_SHAPE),map_grid=.05,map_x=[-.8,1.2],map_y=[-.75,.75],map_refresh=5,history=5)
    canonical=json.dumps(contract,sort_keys=True,separators=(',',':')).encode()
    terrain_repr=[repr(t) for t in env.terrains]
    return dict(sha256=hashlib.sha256(canonical).hexdigest(),contract=contract,
        terrain_sha256=hashlib.sha256(json.dumps(terrain_repr).encode()).hexdigest())

@torch.no_grad()
def collect_rollout(env,model,teachers,*,phase,ledger=None,kind='learning',budget_limit=None,steps=48):
    """Same o_t labels/means/value; env keeps physical episodes across calls."""
    is_dagger=phase.startswith('D');rows=[];rewards=[];values=[];dones=[];logps=[];executed=[];episodes=[]
    key='PPO' if phase.startswith('P') else phase
    attempt=uuid.uuid4().hex
    for control in range(steps):
        obs=env.observe();meta=env.metadata()
        if is_dagger:
            action=model(obs);label=teachers.label(obs,[t=='stairs' for t in meta['teacher']])
            row=dict(obs=obs,actions=label,**meta)
        else:
            dist=model.action_distribution(obs);action=dist.sample();value=model.evaluate(obs).flatten()
            row=dict(obs=obs,actions=action)
            values.append(value.detach().cpu());logps.append(dist.log_prob(action).sum(-1).detach().cpu())
        reservation=f'{attempt}:{control}'
        if ledger is not None:
            if not ledger.reserve(reservation,key,env.num_envs,limit=budget_limit,kind=kind,details=dict(phase=phase,rollout_control_step=control)):
                raise RuntimeError('attempted replay of reserved control step')
        _,reward,done,info=env.step(action)
        if ledger is not None:ledger.mark_executed(reservation)
        rows.append(snapshot(row));rewards.append(reward.detach().cpu());dones.append(done.detach().cpu())
        executed.append(info['frame']['action'].detach().cpu());episodes.extend(info.get('episodes',[]))
    result=concat_batches(rows)
    metrics=dict(rollout_id=attempt,samples=env.num_envs*steps,episodes=episodes,executed_actions=torch.cat(executed),done=torch.stack(dones))
    if not is_dagger:
        # observe() is read-only; no reset precedes this true final-state bootstrap.
        last_value=model.evaluate(env.observe()).flatten().detach().cpu()
        value=torch.stack(values);adv,ret=gae(torch.stack(rewards),value,torch.stack(dones),last_value)
        result.update(values=value.flatten(),log_prob=torch.stack(logps).flatten(),returns=ret.flatten(),advantages=adv.flatten())
    return result,metrics

def save_worker(path,model,learner,reservoir,env,ledger,rounds,physical_sha,keep=None,extra=None):
    learner_state=(learner.state_dict() if isinstance(learner,PPOLearner) else dict(optimizer=learner.optimizer.state_dict()))
    state=dict(model=model.state_dict(),learner=learner_state,reservoir=reservoir.state_dict(copy_storage=False),env=env.state_dict(),
        budgets=ledger.totals(),accounting=ledger.report(),rounds=dict(rounds),course=env.course,num_envs=env.num_envs,physical_sha256=physical_sha,
        keep=keep.state_dict() if keep is not None else None,extra=extra or {})
    return checkpoint_save(path,state)

def verify_acceptance(path,checkpoint,physical_sha):
    c=json.loads(Path(path).read_text())
    if (c.get('token')!='V10_DAGGER_ACCEPTED' or c.get('passed') is not True or c.get('sds_sha256')!=spec.SDS_SHA256
        or c.get('model_sha256')!=sha256(checkpoint) or c.get('physical_sha256')!=physical_sha
        or not c.get('suite_sha256') or not c.get('result_sha256')):raise ValueError('valid DAgger acceptance certificate required')
    return c

def worker(*,course,num_envs,output_dir,stop_round,resume=None,acceptance=None,device='cuda:0',ledger_path=None,controller_state=None):
    from .env import V10Env
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);seed_all(spec.SEED)
    env=V10Env(num_envs=num_envs,course=course,seed=spec.SEED,device=device)
    try:
        physical=physical_contract(env);atomic_json(output/'physical_contract.json',physical)
        ledger=BudgetLedger(ledger_path or output/'costs.jsonl');ledger.reconcile_uncertain();model=V10ActorCritic.from_stairs(device=device)
        reservoir=TaskReservoir();rounds={};keep=None;state=None;extra={}
        if resume:
            state=checkpoint_load(resume,num_envs=num_envs,physical_sha256=physical['sha256'],restore_random=False)
            ledger.assert_checkpoint_not_ahead(state['budgets']);model.load_state_dict(state['model']);reservoir.load_state_dict(state['reservoir'],copy_storage=False);rounds=dict(state['rounds']);extra=state.get('extra',{})
        if controller_state is not None:
            extra=dict(extra,controller_state=json.loads(Path(controller_state).read_text()) if isinstance(controller_state,(str,Path)) else copy.deepcopy(controller_state))
        if course.startswith('D'):
            if state and state['course'] not in ('D1','D2','D3'):raise ValueError('cannot resume DAgger from PPO')
            model.set_phase(course);learner=DAggerLearner(model);teachers=frozen_teachers(device)
            if state:learner.optimizer.load_state_dict(state['learner']['optimizer'])
        else:
            if not state:raise ValueError('PPO requires accepted DAgger checkpoint')
            if state['course'].startswith('D'):
                if course!='P0' or not acceptance:raise ValueError('PPO entry must be P0 with acceptance')
                extra=dict(extra,dagger_acceptance=verify_acceptance(acceptance,resume,physical['sha256']))
                keep=model.initialize_ppo(seed=spec.SEED)
            else:
                if not state.get('keep') or not extra.get('dagger_acceptance'):raise ValueError('PPO state missing frozen accepted DAgger teacher')
                keep=copy.deepcopy(model);keep.load_state_dict(state['keep']);keep.eval().requires_grad_(False)
            model.set_phase(course);learner=PPOLearner(model)
            if state['course'].startswith('P'):learner.load_state_dict(state['learner'])
            teachers=None
        if state:
            # New episodes; preserve role rotation and training RNG, never stale physical trajectories.
            env.load_state_dict(state['env']);restore_rng(state['rng'])
        budget_key='PPO' if course.startswith('P') else course;cap=spec.BUDGETS[budget_key]
        # External consumed samples determine progress even after interrupted rollouts / old-state rollback.
        charged=ledger.totals().get('learning:'+budget_key,0)
        consumed=ledger.report()['executed'].get('learning:'+budget_key,0);nstep=num_envs*spec.ROLLOUT
        if course=='P0':cap=spec.P0_BUDGET
        max_round=cap//nstep
        if stop_round<0 or stop_round>max_round:raise ValueError('stop-round beyond phase budget')
        if course=='P1' and consumed<spec.P0_BUDGET:raise ValueError('P0 budget incomplete')
        if consumed>stop_round*nstep:raise ValueError('stop-round precedes already consumed budget')
        last_path=None
        last_saved_confirmed=(state.get('accounting',{}).get('executed',state['budgets']).get('learning:'+budget_key,0) if state else 0)
        while consumed<stop_round*nstep and charged+nstep<=cap:
            rollout,metrics=collect_rollout(env,model,teachers,phase=course,ledger=ledger,budget_limit=cap)
            update=(learner.update(rollout,reservoir,num_envs) if course.startswith('D') else learner.update(rollout,keep,reservoir,course))
            ledger.record_update(metrics['rollout_id']+':update',budget_key,nstep)
            charged=ledger.totals().get('learning:'+budget_key,0)
            consumed=ledger.report()['executed'].get('learning:'+budget_key,0);rounds[budget_key]=consumed//nstep
            append_json(output/'updates.jsonl',dict(phase=course,round=rounds[budget_key],consumed=consumed,metrics=update,
                updated_samples=nstep,completed_episodes=len(metrics['episodes']),failures=sum(not e['success'] for e in metrics['episodes'])))
            if consumed//spec.SAVE_SAMPLES>last_save_bucket:
                last_saved_confirmed=(state.get('accounting',{}).get('executed',state['budgets']).get('learning:'+budget_key,0) if state else 0)
                last_path=output/f'{course}_{rounds[budget_key]:06d}.pt'
                save_worker(last_path,model,learner,reservoir,env,ledger,rounds,physical['sha256'],keep,extra)
                retain_latest(output,course)
        # Gate pause always follows complete rollout/update/bootstrap. Entry/exit saves also cover zero-round workers.
        last_path=output/f'{course}_{consumed//nstep:06d}_boundary.pt'
        digest=save_worker(last_path,model,learner,reservoir,env,ledger,rounds,physical['sha256'],keep,extra)
        result=dict(checkpoint=str(last_path),checkpoint_sha256=digest,course=course,round=consumed//nstep,
            consumed=consumed,updated_samples=ledger.report()['updated'].get('learning:'+budget_key,0),physical_sha256=physical['sha256'],budget=ledger.report(),gate_reached=consumed>=stop_round*nstep,training_accepted=False)
        atomic_json(output/'worker_result.json',result);return result
    finally:env.close()

def save_due(previous_confirmed,current_confirmed):
    return current_confirmed//spec.SAVE_SAMPLES>previous_confirmed//spec.SAVE_SAMPLES

def append_json(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()

def retain_latest(output,course):
    # Boundary/certified course states are never removed by regular retention.
    files=sorted(Path(output).glob(f'{course}_*.pt'))
    files=[p for p in files if not p.stem.endswith('_boundary')]
    for path in files[:-5]:path.unlink()

def smoke(*,output_dir,num_envs=6,course='D3',steps=48,device='cuda:0'):
    from .env import V10Env
    from .cases import groups
    seed_all();output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    selected=[next(c for c in groups(course) if c['task']==role) for role in spec.ROLES]
    # env fixed cases require frozen actual spawn and joint offsets.
    import random as _random
    rng=_random.Random(42)
    for row in selected:
        row.update(spawn_x=-1. if row['task']=='S_ROUTE' else 1.,initial_joint_offsets=[rng.uniform(-.01,.01) for _ in range(12)])
        from .cases import digest,max_steps
        row['max_steps']=max_steps(row);row['case_sha256']=digest(row)
    selected=[copy.deepcopy(selected[i%6]) for i in range(num_envs)]
    env=V10Env(num_envs,course,42,device,cases=selected)
    try:
        model=V10ActorCritic.from_stairs(device=device);teachers=frozen_teachers(device)
        cost=BudgetLedger(output/'costs.jsonl');batch,metrics=collect_rollout(env,model,teachers,phase='D3',ledger=cost,kind='engineering',budget_limit=num_envs*steps,steps=steps)
        physical=physical_contract(env);atomic_json(output/'physical_contract.json',physical)
        result=dict(finite=True,physical_sha256=physical['sha256'],samples=metrics['samples'],roles=sorted(set(batch['task'])),budget=cost.totals())
        atomic_json(output/'smoke.json',result);return result
    finally:env.close()
