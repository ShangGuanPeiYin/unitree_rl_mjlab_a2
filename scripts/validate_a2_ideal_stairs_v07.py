"""Native v07 engineering stages; execution requires the common frozen closure.

Without --execute, print a plan and create no output. All learning here is billed
as engineering, outside the formal 48000-update branch.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for entry in (ROOT,ROOT/'scripts'):
    if str(entry) not in sys.path:sys.path.insert(0,str(entry))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec, recovery, orchestration

IMPLEMENTATION=ROOT/spec.LOG_ROOT/'implementation'
FREEZE_PATH=IMPLEMENTATION/'freeze/v2/V07_SOURCE_FREEZE.json'
STAGES=('smoke','probe','source-reload','baseline','retention-protocol')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')
    return dict(path=str(path),sha256=spec.sha256(path))


def verify_engineering_freeze(path=None):
    """Fail closed before any allocation when the frozen product is incomplete."""
    path=Path(path or FREEZE_PATH)
    if spec.pending_source_entries():
        raise RuntimeError('v07 source closure is incomplete: '+str(spec.pending_source_entries()))
    value=json.loads(path.read_text())
    hashes=spec.source_hashes()
    expected=dict(source_hashes=hashes,source_closure_sha256=recovery.digest(hashes),
                  contract_sha256=spec.contract_sha256(),sds_sha256=spec.SDS_SHA256)
    for key,want in expected.items():
        if value.get(key)!=want:raise RuntimeError('engineering freeze mismatch: '+key)
    if spec.sha256(ROOT/spec.SDS_PATH)!=spec.SDS_SHA256:
        raise RuntimeError('v07 approved SDS bytes differ')
    return dict(path=str(path),sha256=spec.sha256(path),**expected)


def smoke_plan():
    tile=spec.terrain_spec(spec.tile_id_for(spec.SMOKE_TILE))
    return dict(plan=dict(seed=spec.SMOKE_SEED,num_envs=spec.SMOKE_ENVS,phase='D',
                         substage=tile.lesson_id,tile=tile.lesson_id,level_tick=0,
                         updates=spec.SMOKE_UPDATES,rollout=spec.PPO_ROLLOUT,
                         epochs=spec.PPO_EPOCHS,minibatches=spec.PPO_MINIBATCHES),
        environment=dict(phase='D',substage='D22',level_tick=0,frontier_tile_id=tile.tile_id,
            lesson_id=tile.lesson_id,N=tile.N,h=tile.h,step_depth=tile.step_depth,
            width=tile.width,tile_id=tile.tile_id),
        cost=dict(learning_updates=spec.SMOKE_UPDATES,control_samples=spec.SMOKE_CONTROL_SAMPLES,
                  physics_substeps=spec.SMOKE_PHYSICS_SUBSTEPS,adam_updates=spec.SMOKE_ADAM_UPDATES),
        container=spec.SMOKE_CONTAINER,resumable=False,counts_against_training=False)


def baseline_plan():
    from src.tasks.velocity.v07 import evaluation
    from src.tasks.velocity.v07 import continuation_v07 as migration
    tables=[]
    for substage in ('D25','D22','D20'):
        table=evaluation.make_cases('D',substage,'validation',42)
        selected=[deepcopy(case) for case in table.cases if int(case['episode_index'])<5]
        if len(selected)!=15:raise RuntimeError('baseline selection must contain three speeds of five cases')
        tables.append(dict(substage=substage,full_validation_table=table.to_json(),
            full_validation_table_sha256=table.sha256,selected_cases=selected,
            selected_cases_sha256=recovery.digest(selected)))
    return dict(schema='a2-v07-zero-learning-baseline-v1',seed=42,
        source_checkpoint=migration.SOURCE_PATH,source_sha256=migration.SOURCE_SHA256,
        speeds=list(spec.GATE_SPEEDS_F0_H_D_L),cases_per_speed=5,terrains=tables,cases_total=45,
        deterministic_mean=True,learning_updates=0,counts_against_training=False,
        curriculum_eligibility=False,training_rng_shared=False,
        full_evaluation_rng_isolation='not_established_Q008')


def retention_protocol_plan():
    from src.tasks.velocity.v07 import evaluation
    from src.tasks.velocity.v07 import continuation_v07 as migration
    sets=[]
    for kind in ('flat_retention','stair_retention'):
        table=evaluation.make_cases('D','D25',kind,42)
        sets.append(dict(kind=kind,case_table=table.to_json(),case_table_sha256=table.sha256))
    return dict(schema='a2-v07-zero-learning-retention-protocol-v1',seed=42,
        source_checkpoint=migration.SOURCE_PATH,source_sha256=migration.SOURCE_SHA256,
        speeds=list(spec.GATE_SPEEDS_F0_H_D_L),cases_per_speed=5,cases_total=30,sets=sets,
        deterministic_mean=True,learning_updates=0,counts_against_training=False,
        curriculum_eligibility=False,training_rng_shared=False,
        full_evaluation_rng_isolation='not_established_Q008')


def evaluation_cost(records):
    """Preserve useful first-episode samples and additional batched work."""
    fields=dict(case_control_steps='steps',actual_eval_control_samples='physical_control_samples',
        extra_eval_control_samples='extra_control_samples_after_case',
        actual_eval_physics_substeps='physical_physics_substeps',
        extra_eval_physics_substeps='extra_physics_substeps_after_case')
    totals={key:0 for key in fields}
    for record in records:
        values={}
        for key,source in fields.items():
            value=record[source]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError('evaluation cost must use nonnegative integer samples')
            values[key]=value;totals[key]+=value
        if (values['actual_eval_control_samples']!=values['case_control_steps']+values['extra_eval_control_samples']
            or values['actual_eval_physics_substeps']!=values['actual_eval_control_samples']*spec.DECIMATION
            or values['extra_eval_physics_substeps']!=values['extra_eval_control_samples']*spec.DECIMATION):
            raise ValueError('evaluation physical/first-episode costs do not balance')
    return totals


def persist_case_batch(output,label,cases,records,summary):
    """Bind the actual ordered first records, selected cases and summary bytes."""
    if len(cases)!=len(records) or not cases:
        raise ValueError('engineering evaluation batch is incomplete')
    if [case['case_id'] for case in cases]!=[record['case_id'] for record in records]:
        raise ValueError('engineering case identities are missing, duplicated or reordered')
    identities=[(case['tile_id'],case['vx'],case['case_id']) for case in cases]
    if len(set(identities))!=len(identities):raise ValueError('duplicate engineering case identity')
    for record in records:
        if (record.get('evidence_kind') not in ('terminal','window_end')
            or not isinstance(record.get('first_episode'),dict) or not record['first_episode']):
            raise ValueError('engineering records lack immutable first episode evidence')
    payload=dict(cases=deepcopy(cases),records=deepcopy(records),summary=deepcopy(summary),
        case_table_sha256=recovery.digest(cases),records_sha256=recovery.digest(records),
        summary_sha256=recovery.digest(summary),learning_updates=0,
        counts_against_training=False,**evaluation_cost(records))
    return write_json(Path(output)/(label+'.json'),payload)


def resource_after_batch(output,label):
    snapshot=orchestration.resource_snapshot(directory=Path(output),device=0)
    check=orchestration.check_resources(snapshot,stage='measured')
    row=dict(batch=label,snapshot=snapshot,check=check,measurement_point='after_batch')
    write_json(Path(output)/'resources'/(label+'.json'),row)
    if not check['ok']:raise RuntimeError('engineering batch resource guard failed: '+str(check['blocking']))
    return row


def source_model(device):
    """Restore original stop4817 weights and normalizers, without ETH reinit."""
    from src.tasks.velocity.v07 import continuation_v07 as migration
    from src.tasks.velocity.v07.model import V07ActorCritic
    source=migration.load_designated_source(ROOT/migration.SOURCE_PATH)
    model=V07ActorCritic().to(device)
    recovery.load_model_state(model,source['model'])
    model.actor.normalizer.load_state_dict(source['normalizer']['actor'])
    model.critic_normalizer.load_state_dict(source['normalizer']['critic'])
    model.eval()
    return model,migration.tree_digest(model.state_dict())


def _summary(records,flat=False):
    reasons={}
    for record in records:
        reason=str(record.get('reason') or ('success' if record['climb_success'] else 'timeout'))
        reasons[reason]=reasons.get(reason,0)+1
    return dict(total=len(records),case_pass_count=sum(bool(record['flat_pass']) if flat
        else bool(record['climb_success'] and not record['failure']) for record in records),
        reasons=reasons,window_end_count=sum(record['evidence_kind']=='window_end' for record in records),
        terminal_count=sum(record['evidence_kind']=='terminal' for record in records))


def baseline(output,device):
    import torch
    import train_a2_ideal_stairs_v07 as worker
    from src.tasks.velocity.v07 import continuation_v07 as migration
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=baseline_plan();write_json(output/'case_selection.json',plan)
    model,before=source_model(device)
    results=[];summaries=[];bindings=[];resources=[]
    with torch.inference_mode():
        for terrain in plan['terrains']:
            run=worker.build_case_runner(model,dict(device=device,num_envs=512,eval_num_envs=8,seed=42),'D','D25')
            for vx in plan['speeds']:
                cases=[deepcopy(case) for case in terrain['selected_cases'] if case['vx']==vx]
                records=run(cases);summary=_summary(records)
                label=f"{terrain['substage']}_vx{vx:.1f}"
                bindings.append(persist_case_batch(output,label,cases,records,summary))
                results.extend(dict(substage=terrain['substage'],vx=vx,case=case,result=record)
                               for case,record in zip(cases,records))
                summaries.append(dict(substage=terrain['substage'],vx=vx,**summary))
                resources.append(resource_after_batch(output,label))
    after=migration.tree_digest(model.state_dict())
    if before!=after:raise RuntimeError('zero-learning baseline changed the evaluated model')
    report=dict(plan=plan,summary=summaries,results=results,batch_evidence=bindings,
        records_sha256=recovery.digest(results),summary_sha256=recovery.digest(summaries),
        model_unchanged=True,model_digest_before=before,model_digest_after=after,
        learning_updates=0,counts_against_training=False,curriculum_eligibility=False,
        behavior_is_not_an_admission_threshold=True,resource_checks=resources,
        resource_snapshot=resources[-1]['snapshot'],**evaluation_cost([row['result'] for row in results]))
    write_json(output/'baseline.json',report)
    return report


def retention_protocol(output,device):
    import torch
    import train_a2_ideal_stairs_v07 as worker
    from src.tasks.velocity.v07 import continuation_v07 as migration
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=retention_protocol_plan();write_json(output/'case_selection.json',plan)
    model,before=source_model(device)
    run=worker.build_case_runner(model,dict(device=device,num_envs=512,eval_num_envs=8,seed=42),'D','D25')
    results=[];summaries=[];bindings=[];resources=[]
    with torch.inference_mode():
        for selected in plan['sets']:
            for vx in plan['speeds']:
                cases=[deepcopy(case) for case in selected['case_table']['cases'] if case['vx']==vx]
                records=run(cases);summary=_summary(records,flat=selected['kind']=='flat_retention')
                label=f"{selected['kind']}_vx{vx:.1f}"
                bindings.append(persist_case_batch(output,label,cases,records,summary))
                results.extend(dict(kind=selected['kind'],case=case,result=record) for case,record in zip(cases,records))
                summaries.append(dict(kind=selected['kind'],vx=vx,**summary))
                resources.append(resource_after_batch(output,label))
    after=migration.tree_digest(model.state_dict())
    if before!=after:raise RuntimeError('zero-learning retention protocol changed model state')
    windows=[row['result'] for row in results if row['kind']=='flat_retention' and row['result']['evidence_kind']=='window_end']
    report=dict(plan=plan,summary=summaries,results=results,batch_evidence=bindings,
        records_sha256=recovery.digest(results),summary_sha256=recovery.digest(summaries),
        model_unchanged=True,model_digest_before=before,model_digest_after=after,
        learning_updates=0,counts_against_training=False,curriculum_eligibility=False,
        resource_checks=resources,resource_snapshot=resources[-1]['snapshot'],
        flat_500_window_count=len(windows),
        flat_500_window_coverage_established=bool(windows) and all(record['steps']==500 for record in windows),
        behavior_is_not_an_admission_threshold=True,**evaluation_cost([row['result'] for row in results]))
    write_json(output/'retention_protocol.json',report)
    return report


def smoke(output,device):
    import torch
    import train_a2_ideal_stairs_v07 as worker
    from src.tasks.velocity.v07.model import V07ActorCritic
    from src.tasks.velocity.v07.ppo import PPOTrainer
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    record=smoke_plan();write_json(output/'smoke_plan.json',record)
    model=V07ActorCritic.migrate_eth_checkpoint(spec.INIT_PATH,spec.INIT_SHA256,seed=42,device=device)
    env=worker.build_environment(dict(num_envs=record['plan']['num_envs'],
        eval_num_envs=spec.EVALUATION_NUM_ENVS,device=device,seed=record['plan']['seed']),
        phase=record['environment']['phase'],substage=record['environment']['substage'],
        level_tick=0,frontier_tile_id=record['environment']['frontier_tile_id'])
    if env.frontier_tile_id!=22 or env.substage!='D22' or not bool((env.tile_ids==22).all()):
        raise RuntimeError('smoke actual environment does not construct D22 tile22')
    trainer=PPOTrainer(model,env);history=[];resources=[]
    before=dict(trainer.learning)
    for index in range(record['plan']['updates']):
        _,metrics=trainer.train_step();after=dict(trainer.learning)
        delta={key:int(after[key])-int(before[key]) for key in ('update','control_samples','physics_substeps','adam_steps')}
        per_control=record['plan']['num_envs']*spec.PPO_ROLLOUT
        if delta!=dict(update=1,control_samples=per_control,
            physics_substeps=per_control*spec.DECIMATION,adam_steps=spec.ADAM_UPDATES_PER_ITERATION):
            raise RuntimeError('smoke actual per-update learning counters differ')
        history.append(dict(update=index+1,delta=delta,learning=after,loss=metrics.get('loss')))
        resources.append(resource_after_batch(output,f'update_{index+1:02d}'))
        before=after
    totals=dict(learning_updates=int(after['update']),control_samples=int(after['control_samples']),
                physics_substeps=int(after['physics_substeps']),adam_updates=int(after['adam_steps']))
    if totals!=record['cost']:raise RuntimeError('smoke learning cost differs from the frozen two updates')
    result=dict(record,history=history,cost_totals=totals,learning=after,
        membership=trainer.membership(),environment_actual=dict(phase=env.phase,substage=env.substage,
            frontier_tile_id=int(env.frontier_tile_id),physical_tile_ids=env.tile_ids.tolist()),
        resource_checks=resources)
    write_json(output/'smoke.json',result)
    return result


def run_stage(stage,output,device):
    output=Path(output)
    if output.exists():raise FileExistsError('engineering stage already exists; preserve and report it')
    if stage=='smoke':return smoke(output,device)
    if stage=='baseline':return baseline(output,device)
    if stage=='retention-protocol':return retention_protocol(output,device)
    if stage=='probe':
        import train_a2_ideal_stairs_v07 as worker
        plan=orchestration.probe_plan(seed=42);plan['device']=device
        output.mkdir(parents=True,exist_ok=False);write_json(output/'probe_plan.json',plan)
        result=worker.build_probe_worker(plan,output)
        orchestration.validate_probe_recovery(result['recovery'],plan)
        measured=orchestration.summarize_resource_samples(result['resources']['samples'])
        if not measured['ok']:raise RuntimeError('v07 probe resources failed: '+str(measured['blocking']))
        result['resource_summary']=measured;result['plan']=plan
        write_json(output/'probe.json',result)
        return result
    if stage=='source-reload':
        from src.tasks.velocity.v07.runtime import prepare_runner
        runner,receipt=prepare_runner(dict(spec.recipe_config(),device=device),output)
        if runner.update!=4817 or runner.curriculum.state.generation!=73:
            raise RuntimeError('source-reload must preserve 4817 and migrate generation exactly once to73')
        receipt=dict(receipt,engineering_only=True,learning_updates=0,counts_against_training=False)
        write_json(output/'source_reload.json',receipt)
        return receipt
    raise ValueError(stage)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=STAGES)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output-dir',default=None)
    args=parser.parse_args(argv)
    if args.seed!=42:raise SystemExit('v07 engineering seed is42')
    output=Path(args.output_dir) if args.output_dir else IMPLEMENTATION/'gpu/v1'/args.stage
    if not output.is_absolute():output=ROOT/output
    production=(ROOT/spec.CLI_DEFAULTS['output_dir']).resolve()
    if output.resolve()==production or production in output.resolve().parents:
        raise ValueError('engineering output must not use the formal continuation directory')
    plan=dict(stage=args.stage,device=args.device,seed=args.seed,output=str(output),
        sds_sha256=spec.SDS_SHA256,source_closure_sha256=recovery.digest(spec.source_hashes()),
        contract_sha256=spec.contract_sha256(),freeze_file=str(FREEZE_PATH),
        resumable_for_production=False,counts_against_training=False)
    if args.stage=='baseline':plan['baseline']=baseline_plan()
    elif args.stage=='retention-protocol':plan['retention_protocol']=retention_protocol_plan()
    elif args.stage=='smoke':plan['smoke']=smoke_plan()
    elif args.stage=='probe':plan['probe']=orchestration.probe_plan(seed=42)
    print(json.dumps(plan,indent=2,sort_keys=True))
    if not args.execute:return 0
    frozen=verify_engineering_freeze()
    if output.exists():raise FileExistsError('engineering stage already exists; preserve and report it')
    pre=orchestration.resource_snapshot(directory=output.parent if output.parent.exists() else ROOT,device=0)
    check=orchestration.check_resources(pre,stage='pre_run')
    if not check['ok']:raise RuntimeError('resources failed before engineering: '+str(check))
    registry=orchestration.worker_registry_path()
    orchestration.acquire_single_worker(args.device,path=registry)
    started=time.time()
    try:
        result=run_stage(args.stage,output,args.device)
        verify_engineering_freeze()
        evidence=dict(plan=plan,freeze=frozen,started=started,finished=time.time(),pre_resources=pre,
            status='engineering_check_only',learning_updates=2 if args.stage=='smoke' else 56 if args.stage=='probe' else 0,
            counts_against_training=False,full_evaluation_rng_isolation='not_established_Q008')
        write_json(output/'engineering_manifest.json',evidence)
        print(json.dumps(evidence,indent=2,sort_keys=True))
    except BaseException as error:
        write_json(output/'engineering_error.json',dict(error_type=type(error).__name__,error=repr(error),
            plan=plan,started=started,failed=time.time()))
        raise
    finally:orchestration.release_single_worker(registry)
    return 0


if __name__=='__main__':raise SystemExit(main())
