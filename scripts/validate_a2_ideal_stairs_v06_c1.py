"""C1 engineering stages, always isolated from the production continuation."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for path in [ROOT,ROOT/'scripts']:
    if str(path) not in sys.path:sys.path.insert(0,str(path))
if 'src.tasks.velocity.v06.spec' in sys.modules:
    raise RuntimeError('C1 engineering requires a clean interpreter')
os.environ['A2_V06_CONTINUATION']='v06-c1-speed030060-budget36000'
import a2_v06_bootstrap
a2_v06_bootstrap.activate(ROOT)

from src.tasks.velocity.v06 import spec, recovery, orchestration
from src.tasks.velocity.v06 import continuation_c1 as migration
from src.tasks.velocity.v06.c1_contract import IMPLEMENTATION, LOG_ROOT, PLAN_SHA256
from src.tasks.velocity.v06.c1_runtime import verify_plan, write_json


def baseline_plan():
    from src.tasks.velocity.v06 import evaluation
    tables=[]
    for substage in ['D25','D20']:
        table=evaluation.make_cases('D',substage,'validation',42)
        selected=[deepcopy(c) for c in table.cases if int(c['episode_index'])<5]
        assert len(selected)==15
        tables.append(dict(substage=substage,full_validation_table=table.to_json(),
            full_validation_table_sha256=table.sha256,selected_cases=selected,
            selected_cases_sha256=recovery.digest(selected)))
    return dict(schema='a2-v06-c1-zero-learning-baseline-v1',seed=42,
        source_checkpoint=migration.SOURCE_PATH,source_sha256=migration.SOURCE_SHA256,
        speeds=[.3,.5,.6],cases_per_speed=5,terrains=tables,cases_total=30,
        deterministic_mean=True,learning_updates=0,counts_against_training=False,
        curriculum_eligibility=False,training_rng_shared=False,
        full_evaluation_rng_isolation='not_established_Q008')


def baseline(output,device):
    import torch
    from src.tasks.velocity.v06.model import V06ActorCritic
    import train_a2_ideal_stairs_v06 as worker
    output=Path(output); plan=baseline_plan();write_json(output/'case_selection.json',plan)
    source=migration.load_designated_source(ROOT/migration.SOURCE_PATH)
    # This is a new model/env, not the candidate training worker or its RNG.
    model=V06ActorCritic().to(device)
    recovery.load_model_state(model,source['model'])
    model.actor.normalizer.load_state_dict(source['normalizer']['actor'])
    model.critic_normalizer.load_state_dict(source['normalizer']['critic'])
    model.eval(); before=migration.tree_digest(model.state_dict())
    results=[]
    with torch.inference_mode():
        for terrain in plan['terrains']:
            run=worker.build_case_runner(model,dict(device=device,num_envs=512,
                eval_num_envs=8,seed=42),'D',terrain['substage'])
            for speed in plan['speeds']:
                cases=[c for c in terrain['selected_cases'] if c['vx']==speed]
                records=run(cases)
                if len(records)!=5:raise RuntimeError('baseline batch incomplete')
                for case,record in zip(cases,records):
                    if record['case_id']!=case['case_id']:raise RuntimeError('baseline case identity mismatch')
                    result=dict(substage=terrain['substage'],vx=speed,case=case,result=record)
                    results.append(result)
                    with (output/'cases.jsonl').open('a') as stream:
                        stream.write(json.dumps(result,sort_keys=True)+'\n')
                write_json(output/f"{terrain['substage']}_vx{speed:.1f}.json",records)
    after=migration.tree_digest(model.state_dict())
    if before!=after:raise RuntimeError('baseline changed the evaluated model')
    summary=[]
    for substage in ['D25','D20']:
        for speed in [.3,.5,.6]:
            rows=[r['result'] for r in results if r['substage']==substage and r['vx']==speed]
            successes=sum(bool(r['climb_success'] and not r['failure']) for r in rows)
            reasons={}
            for r in rows:
                reason=r.get('reason') or ('success' if r['climb_success'] else 'timeout')
                reasons[reason]=reasons.get(reason,0)+1
            summary.append(dict(substage=substage,vx=speed,passed=successes,total=5,reasons=reasons))
    report=dict(plan=plan,summary=summary,results=results,model_unchanged=True,
        model_digest_before=before,model_digest_after=after,learning_updates=0,
        case_control_steps=sum(r['result']['steps'] for r in results),
        actual_eval_control_samples=sum(r['result']['physical_control_samples'] for r in results),
        extra_eval_control_samples=sum(r['result']['extra_control_samples_after_case'] for r in results),
        actual_eval_physics_substeps=sum(r['result']['physical_physics_substeps'] for r in results),
        resource_snapshot=orchestration.resource_snapshot(directory=output,device=0))
    write_json(output/'baseline.json',report)
    return report


def evaluation_cost(records):
    """Separate useful first-episode samples from all batched simulation work."""
    fields = dict(case_control_steps='steps',
        actual_eval_control_samples='physical_control_samples',
        extra_eval_control_samples='extra_control_samples_after_case',
        actual_eval_physics_substeps='physical_physics_substeps',
        extra_eval_physics_substeps='extra_physics_substeps_after_case')
    totals = {name: 0 for name in fields}
    for record in records:
        values = {}
        for name, key in fields.items():
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('evaluation cost must use nonnegative integer samples')
            values[name] = value
            totals[name] += value
        if (values['actual_eval_control_samples'] != values['case_control_steps'] + values['extra_eval_control_samples']
                or values['actual_eval_physics_substeps'] != values['actual_eval_control_samples'] * spec.DECIMATION
                or values['extra_eval_physics_substeps'] != values['extra_eval_control_samples'] * spec.DECIMATION):
            raise ValueError('evaluation physical/first-episode costs do not balance')
    return totals


def retention_protocol_plan():
    from src.tasks.velocity.v06 import evaluation
    sets = []
    for kind in ('flat_retention', 'stair_retention'):
        table = evaluation.make_cases('D', 'D25', kind, 42)
        sets.append(dict(kind=kind, case_table=table.to_json(), case_table_sha256=table.sha256))
    return dict(schema='a2-c1-zero-learning-retention-protocol-v1', seed=42,
        source_checkpoint=migration.SOURCE_PATH, source_sha256=migration.SOURCE_SHA256,
        speeds=[.3,.5,.6], cases_per_speed=5, cases_total=30, sets=sets,
        deterministic_mean=True, learning_updates=0, counts_against_training=False,
        curriculum_eligibility=False, training_rng_shared=False,
        full_evaluation_rng_isolation='not_established_Q008')


def retention_protocol(output, device):
    import torch
    from src.tasks.velocity.v06.model import V06ActorCritic
    import train_a2_ideal_stairs_v06 as worker
    output = Path(output); plan = retention_protocol_plan()
    write_json(output/'case_selection.json', plan)
    source = migration.load_designated_source(ROOT/migration.SOURCE_PATH)
    model = V06ActorCritic().to(device)
    recovery.load_model_state(model, source['model'])
    model.actor.normalizer.load_state_dict(source['normalizer']['actor'])
    model.critic_normalizer.load_state_dict(source['normalizer']['critic'])
    model.eval(); before = migration.tree_digest(model.state_dict())
    # The same long-lived D25 closure serves both protocols; the production
    # router must choose F0/None and H/None from each frozen case batch.
    run = worker.build_case_runner(model, dict(device=device,num_envs=512,eval_num_envs=8,seed=42), 'D','D25')
    results = []; resources = []
    with torch.inference_mode():
        for selected in plan['sets']:
            for vx in plan['speeds']:
                cases = [deepcopy(c) for c in selected['case_table']['cases'] if c['vx']==vx]
                records = run(cases)
                if len(records)!=5 or [r['case_id'] for r in records] != [c['case_id'] for c in cases]:
                    raise RuntimeError('retention engineering cases are incomplete or reordered')
                rows = [dict(kind=selected['kind'],case=c,result=r) for c,r in zip(cases,records)]
                results.extend(rows)
                write_json(output/f"{selected['kind']}_vx{vx:.1f}.json",rows)
                snapshot = orchestration.resource_snapshot(directory=output,device=0)
                check = orchestration.check_resources(snapshot,stage='measured')
                resources.append(dict(kind=selected['kind'],vx=vx,snapshot=snapshot,check=check,
                                      measurement_point='after_batch'))
                write_json(output/'resource_checks.json',resources)
                if not check['ok']:
                    raise RuntimeError('retention protocol resource guard failed: '+str(check['blocking']))
    after = migration.tree_digest(model.state_dict())
    if before!=after: raise RuntimeError('zero-learning retention protocol changed model state')
    summary = []
    for selected in plan['sets']:
        for vx in plan['speeds']:
            rows = [x['result'] for x in results if x['kind']==selected['kind'] and x['case']['vx']==vx]
            flat = selected['kind']=='flat_retention'
            summary.append(dict(kind=selected['kind'],vx=vx,total=len(rows),
                case_pass_count=sum(bool(r['flat_pass']) if flat else bool(r['climb_success'] and not r['failure']) for r in rows),
                window_end_count=sum(r['evidence_kind']=='window_end' for r in rows),
                terminal_count=sum(r['evidence_kind']=='terminal' for r in rows),
                raw_mean_vx=[r['mean_vx'] for r in rows]))
    flat_records = [x['result'] for x in results if x['kind']=='flat_retention']
    windows = [r for r in flat_records if r['evidence_kind']=='window_end']
    report = dict(plan=plan,summary=summary,results=results,model_unchanged=True,
        model_digest_before=before,model_digest_after=after,learning_updates=0,
        curriculum_eligibility=False,resource_checks=resources,
        resource_snapshot=resources[-1]['snapshot'],
        flat_500_window_count=len(windows),
        flat_500_window_coverage_established=bool(windows) and all(r['steps']==500 for r in windows),
        behavior_is_not_an_admission_threshold=True,
        **evaluation_cost([x['result'] for x in results]))
    write_json(output/'retention_protocol.json',report)
    return report


def run_stage(stage,output,device):
    if stage=='baseline':return baseline(output,device)
    if stage=='retention-protocol':return retention_protocol(output,device)
    if stage=='smoke':
        import smoke_a2_ideal_stairs_v06 as smoke
        smoke.main(['--execute','--device',device,'--output-dir',str(output)])
        return json.loads((output/'smoke.json').read_text())
    if stage=='probe':
        import train_a2_ideal_stairs_v06 as worker
        plan=orchestration.probe_plan(seed=42);plan['device']=device
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'probe_plan.json',plan)
        result=worker.build_probe_worker(plan,output)
        orchestration.validate_probe_recovery(result['recovery'],plan)
        measured=orchestration.summarize_resource_samples(result['resources']['samples'])
        if not measured['ok']:raise RuntimeError('C1 probe resources failed: '+str(measured['blocking']))
        result['resource_summary']=measured
        result['plan']=plan
        write_json(output/'probe.json',result)
        return result
    if stage=='source-reload':
        from src.tasks.velocity.v06.c1_runtime import prepare_runner
        runner,receipt=prepare_runner(dict(spec.recipe_config(),device=device),output)
        assert runner.update==4224 and runner.curriculum.state.generation==68
        receipt['engineering_only']=True;receipt['learning_updates']=0
        write_json(output/'source_reload.json',receipt)
        return receipt
    raise ValueError(stage)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['smoke','probe','baseline','source-reload','retention-protocol'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv);verify_plan()
    if args.seed!=42:raise SystemExit('C1 engineering seed is 42')
    output=(ROOT/LOG_ROOT/'baseline/stop004224_seed42' if args.stage=='baseline'
            else ROOT/IMPLEMENTATION/'gpu'/'q010_v3'/args.stage)
    plan=dict(stage=args.stage,device=args.device,seed=args.seed,output=str(output),
        plan_sha256=PLAN_SHA256,source_closure_sha256=recovery.digest(spec.source_hashes()),
        resumable_for_production=False,counts_against_training=False)
    if args.stage=='baseline':plan['baseline']=baseline_plan()
    if args.stage=='retention-protocol':plan['retention_protocol']=retention_protocol_plan()
    print(json.dumps(plan,indent=2,sort_keys=True))
    if not args.execute:return 0
    if output.exists():raise FileExistsError('engineering stage already exists; preserve and report it')
    limits,_=orchestration.resource_limits()
    pre=orchestration.resource_snapshot(directory=output.parent if output.parent.exists() else ROOT,
                                        device=0)
    check=orchestration.check_resources(pre,limits,stage='pre_run')
    if not check['ok']:raise RuntimeError('resources failed before engineering: '+str(check))
    registry=orchestration.worker_registry_path()
    orchestration.acquire_single_worker(args.device,path=registry)
    started=time.time()
    try:
        result=run_stage(args.stage,output,args.device)
        if recovery.digest(spec.source_hashes())!=plan['source_closure_sha256']:
            raise RuntimeError('source closure changed during engineering')
        evidence=dict(plan=plan,started=started,finished=time.time(),pre_resources=pre,
            status='engineering_check_only',learning_updates=(2 if args.stage=='smoke' else
                    56 if args.stage=='probe' else 0),counts_against_training=False,
            full_evaluation_rng_isolation='not_established_Q008')
        write_json(output/'engineering_manifest.json',evidence)
        print(json.dumps(evidence,indent=2,sort_keys=True))
    except BaseException as error:
        write_json(output/'engineering_error.json',dict(error_type=type(error).__name__,
            error=repr(error),plan=plan,started=started,failed=time.time()))
        raise
    finally:orchestration.release_single_worker(registry)
    return 0

if __name__=='__main__':raise SystemExit(main())
