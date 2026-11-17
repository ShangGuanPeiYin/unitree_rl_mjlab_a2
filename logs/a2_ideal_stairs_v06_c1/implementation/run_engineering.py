"""Engineering observer; product code, RNG calls and numeric recipe stay frozen."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import gc
import json
import os
import sys
import weakref

ROOT=Path('/home/xxs/dog/unitree_rl_mjlab_a2')
os.chdir(ROOT)
for path in [ROOT,ROOT/'scripts']:
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import validate_a2_ideal_stairs_v06_c1 as validation
import torch
import train_a2_ideal_stairs_v06 as worker
from src.tasks.velocity.v06 import c1_runtime,orchestration,recovery,model as model_module,ppo
from src.tasks.velocity.v06 import continuation_c1 as migration
from src.tasks.velocity.v06.curriculum import CurriculumManager,allocate_roles
from src.tasks.velocity.v06.runner import V06Runner

original_build=worker.build_environment

def observed_environment(contract,**kwargs):
    env=original_build(contract,**kwargs)
    events=[]
    original_reset=env._reset_ids
    def reset_ids(ids,**reset_kwargs):
        indices=torch.as_tensor(ids).detach().cpu().to(torch.long).flatten().tolist()
        before=dict(phase=env.phase,substage=env.substage,generation=env.generation,
                    stage_iteration=env.stage_iteration)
        result=original_reset(ids,**reset_kwargs)
        events.append(dict(ids=indices,count=len(indices),resample_tiles=reset_kwargs.get('resample_tiles',True),
                           context=before,learning_control_steps=0,completed_episodes=0))
        return result
    env._reset_ids=reset_ids
    env.c1_observed_reset_events=events
    return env

worker.build_environment=observed_environment
original_prepare=c1_runtime.prepare_runner

def measured_resources(runner,output):
    if runner.resource_guard is not None:
        measured=runner.resource_guard()
    else:
        snapshot=orchestration.resource_snapshot(directory=output,device=0)
        measured=orchestration.check_resources(snapshot,stage='measured')
    validation.write_json(Path(output)/'resources/source_reload_measured.json',measured)
    if not measured['ok']:raise RuntimeError('source-reload measured resources failed')
    return measured


def environment_evidence(runner,original_source):
    env=runner.env
    roles=list(env.roles);tiles=env.tile_ids.detach().cpu().tolist()
    assert env._tile_override is None
    assert env.phase==runner.phase=='D' and env.substage==runner.substage=='D25'
    assert env.stage_iteration==runner.phase_update==1524
    assert env.generation==runner.curriculum.state.generation==68
    assert Counter(roles)==Counter(allocate_roles(512,'D'))==dict(flat=102,frontier=164,near=123,easy=123)
    expected=dict(flat={0},frontier={17},near={15,16},easy=set(range(1,15)))
    for role,tile in zip(roles,tiles):assert tile in expected[role],(role,tile)
    command=env.command.detach().cpu()
    assert .3<=float(command[:,0].min())<=float(command[:,0].max())<=.6
    assert bool((command[:,1:]==0).all())
    fresh=env._observations(update=False)
    checks=dict(
        obs_is_final_episode=all(torch.equal(runner.ppo.obs[k],v) for k,v in fresh.items()),
        all_episode_steps_zero=bool((env.episode_length_buf==0).all()),
        no_terminal=env.last_termination is None,
        no_first_step=not bool(env.first_step_acquired.any()),
        no_stop=not bool(env.stop.any()),
        no_stop_counter=bool((env.stop_steps==0).all()),
        no_contact_history=not bool(env.contacts.raw.any() or env.contacts.filtered.any()),
        no_climb_history=not bool(env.climb.climbed.any()) and bool((env.climb.support_count==0).all()),
        no_tracking_count=bool((env._tracking_count==0).all()),
        no_reward_sums=all(bool((value==0).all()) for value in env.episode_reward_sums.values()),
        no_stall_samples=bool((env.stall.active_samples==0).all() and (env.stall.stall_samples==0).all()),
        no_forward_progress=bool((env.forward_distance_buf==0).all()),
        start_matches_final_pose=bool(torch.equal(env.episode_start_local_x,
            env.robot.data.root_link_pos_w[:,0]-env.origins[env.tile_ids,0])),
        map_matches_final_tile=bool(torch.equal(env.map_cache.sample_tile,env.tile_ids)),
        history_matches_final_proprio=bool(torch.equal(env.history_cache.history,
            env._proprio()[:,None].expand(-1,5,-1))),
        reward_state_fresh=env.reward_computer.state is None,
        no_learning=runner.update==4224 and runner.curriculum.state.generation_updates==0,
        no_frontier_fifo=runner.curriculum.state.episode_fifo==[])
    actual=runner.checkpoint_state()
    for key in ['model','optimizer','normalizer','learning']:
        checks['source_'+key+'_preserved']=migration.tree_digest(actual[key])==migration.tree_digest(original_source[key])
    assert all(checks.values()),checks
    events=env.c1_observed_reset_events
    assert events and events[-1]['ids']==list(range(512))
    assert len(events) in [1,2],events
    # One original schedule call may reset changed slots, followed by one original
    # full reset. These initializations are not learning or completed episodes.
    return dict(phase=env.phase,substage=env.substage,stage_iteration=env.stage_iteration,generation=env.generation,
        role_counts=dict(Counter(roles)),role_to_tile=[dict(env_id=i,role=r,tile_id=t) for i,(r,t) in enumerate(zip(roles,tiles))],
        allowed_tiles={k:sorted(v) for k,v in expected.items()},
        command=dict(min=float(command[:,0].min()),max=float(command[:,0].max()),vy_zero=True,wz_zero=True),
        reset_events=deepcopy(events),reset_calls=len(events),reset_slot_initializations=sum(e['count'] for e in events),
        distinct_initialized_slots=len({i for e in events for i in e['ids']}),
        extra_initialization_slots=sum(e['count'] for e in events)-512,
        rng_audit=runner.c1_initialization_audit,checks=checks,
        pending_reward_reset_records=len(env.reward_computer._pending_resets),
        accepted_extra_rng_consumption=True,full_evaluation_rng_isolation='not_established_Q008')


def second_runner(contract,output,ready):
    state=recovery.preload_checkpoint(ready['path'],ready['sha256'],mode='resume')
    model=model_module.V06ActorCritic().to(contract['device'])
    cur=state['curriculum']
    env=worker.build_environment(contract,phase=cur['phase'],substage=cur['substage'],level_tick=cur['level_tick'])
    trainer=ppo.PPOTrainer(model,env)
    manager=CurriculumManager(num_envs=512,seed=42,state=cur)
    holder={}
    runner=V06Runner(model=model,ppo=trainer,curriculum=manager,env=env,directory=output,
        seed=42,device=contract['device'],mode='resume',source=state['source'],require_resume_evidence=True,
        resource_guard=worker.build_measured_resource_guard(output,contract['device'],holder))
    holder['runner']=runner
    runner.load(ready['path'],ready['sha256'])
    loaded=runner.save_verified('resume_loaded');new_ready=runner.save_verified('resume_ready')
    return runner,dict(resume_loaded=loaded,resume_ready=new_ready,
        source_checkpoint_sha256=ready['sha256'],generation=runner.curriculum.state.generation,
        original_parent_sha256=runner.source['continuation']['parent']['checkpoint_sha256'])


def observed_source_stage(output,device):
    output=Path(output);contract=dict(validation.spec.recipe_config(),device=device)
    original_source=migration.load_designated_source(ROOT/migration.SOURCE_PATH)
    first,receipt=original_prepare(contract,output)
    first_evidence=environment_evidence(first,original_source)
    first_resources=measured_resources(first,output)
    ready=dict(first.resume_ready)
    validation.write_json(output/'first_load_environment.json',first_evidence)
    env_ref=weakref.ref(first.env);model_ref=weakref.ref(first.model)
    first.env.close();del first;gc.collect();torch.cuda.empty_cache()
    released=dict(first_env_released=env_ref() is None,first_model_released=model_ref() is None,
        resource_snapshot=orchestration.resource_snapshot(directory=output,device=0))
    validation.write_json(output/'first_release.json',released)
    assert released['first_env_released'] and released['first_model_released']
    second,second_receipt=second_runner(contract,output/'second_load',ready)
    second_evidence=environment_evidence(second,original_source)
    second_resources=measured_resources(second,output/'second_load')
    assert second.source['continuation']['parent']['checkpoint_sha256']==migration.SOURCE_SHA256
    # Engineering-only demonstration of a later real frontier switch. Saved ready
    # checkpoints above still contain the original D25/gen68 state, not this probe.
    second.curriculum.state.substage='D20';second.curriculum.state.frontier_tile_id=18
    second.curriculum.state.generation=69
    second._reset_environment_schedule()
    env=second.env
    switch=dict(phase=env.phase,substage=env.substage,frontier=env.frontier_tile_id,
        generation=env.generation,stage_iteration=env.stage_iteration,
        role_to_tile=[dict(env_id=i,role=r,tile_id=int(t)) for i,(r,t) in enumerate(zip(env.roles,env.tile_ids.tolist()))],
        engineering_only=True,learning_updates=0)
    expected=dict(flat={0},frontier={18},near={16,17},easy=set(range(1,16)))
    assert env.substage=='D20' and env.frontier_tile_id==18
    assert Counter(env.roles)==Counter(allocate_roles(512,'D'))
    for role,tile in zip(env.roles,env.tile_ids.tolist()):assert tile in expected[role]
    validation.write_json(output/'second_load_environment.json',second_evidence)
    validation.write_json(output/'later_schedule_switch.json',switch)
    result=dict(first=receipt,first_environment=first_evidence,first_resources=first_resources,
        release=released,second=second_receipt,second_environment=second_evidence,
        second_resources=second_resources,later_schedule_switch=switch,
        learning_updates=0,engineering_only=True,resumable_for_production=False,
        full_evaluation_rng_isolation='not_established_Q008')
    validation.write_json(output/'source_reload.json',result)
    second.env.close()
    return result

original_stage=validation.run_stage

def run_stage(stage,output,device):
    if stage=='source-reload':return observed_source_stage(output,device)
    return original_stage(stage,output,device)

validation.run_stage=run_stage

def observer_arguments(arguments):
    args=list(arguments)
    flag='--observer-snapshot-recheck'
    if flag not in args:
        return args,None
    if args.count(flag)!=1 or args.count('--stage')!=1:
        raise ValueError('observer recheck needs exactly one explicit source-reload stage')
    index=args.index('--stage')
    if index+1>=len(args) or args[index+1]!='source-reload' or any(x.startswith('--stage=') for x in args):
        raise ValueError('observer recheck is restricted to source-reload')
    args.remove(flag)
    return args,'observer_snapshot_v1'

arguments,suffix=observer_arguments(sys.argv[1:])
if suffix is not None:
    # Only the observation output root changes; numeric recipe and source stay frozen.
    validation.IMPLEMENTATION=str(Path(validation.IMPLEMENTATION)/suffix)
status=validation.main(arguments)
if '--execute' in arguments and '--stage' in arguments:
    stage=arguments[arguments.index('--stage')+1]
    if stage=='baseline':
        output=ROOT/'logs/a2_ideal_stairs_v06_c1/baseline/stop004224_seed42'
        snapshot=json.loads((output/'baseline.json').read_text())['resource_snapshot']
        measured=orchestration.check_resources(snapshot,stage='measured')
        validation.write_json(output/'measured_resource_check.json',measured)
        if not measured['ok']:raise RuntimeError('baseline measured resources failed')
raise SystemExit(status)
