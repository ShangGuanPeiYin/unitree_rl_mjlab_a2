"""Native v07 admission and exact-source / same-version recovery adapter."""
from copy import deepcopy
import json
from pathlib import Path
import time

from . import spec,recovery,orchestration,model as model_module,ppo as ppo_module
from . import continuation_v07 as migration

SOURCE_PATH=migration.SOURCE_PATH
SOURCE_SHA256=migration.SOURCE_SHA256
IMPLEMENTATION='logs/a2_ideal_stairs_v07/implementation'
MANIFEST_PATH=IMPLEMENTATION+'/V07_BRIDGE_MANIFEST.json'
ACCEPTANCE_FILE=IMPLEMENTATION+'/acceptance/INDEPENDENT_ACCEPTANCE.md'
OUTPUT_DIR='logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01'

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')

def verify_plan():
    if spec.SCHEMA_VERSION!=9 or spec.RECIPE_VERSION!='0.7-d22-eval-regression-r1':
        raise orchestration.AdmissionError('native v07 must be selected before imports')
    if spec.sha256(spec.ROOT/spec.SDS_PATH)!=migration.PLAN_SHA256:
        raise orchestration.AdmissionError('approved v07 SDS bytes differ')
    if spec.pending_source_entries():
        raise orchestration.AdmissionError('native v07 source entries are incomplete: '+str(spec.pending_source_entries()))

def verify_manifest(path=MANIFEST_PATH):
    verify_plan();path=Path(path)
    if not path.is_absolute():path=spec.ROOT/path
    if path.resolve()!=(spec.ROOT/MANIFEST_PATH).resolve():raise orchestration.AdmissionError('only the fixed v07 bridge manifest is admitted')
    manifest=json.loads(path.read_text());hashes=spec.source_hashes()
    expected=dict(schema=9,manifest_schema='a2-v07-bridge-manifest-v1',recipe=spec.RECIPE_VERSION,sds_sha256=migration.PLAN_SHA256,
        source_checkpoint_sha256=SOURCE_SHA256,source_closure_sha256=recovery.digest(hashes),
        contract_sha256=spec.contract_sha256())
    if type(manifest.get('schema')) is not int:raise orchestration.AdmissionError('v07 manifest schema must be integer9')
    for key,value in expected.items():
        if manifest.get(key)!=value:raise orchestration.AdmissionError('v07 manifest mismatch: '+key)
    if manifest.get('source_hashes')!=hashes:raise orchestration.AdmissionError('v07 manifest source file closure differs')
    acceptance=spec.ROOT/ACCEPTANCE_FILE;verdict_path=acceptance.with_suffix('.json')
    if manifest.get('independent_acceptance_sha256')!=spec.sha256(acceptance) or manifest.get('independent_verdict_sha256')!=spec.sha256(verdict_path):
        raise orchestration.AdmissionError('v07 acceptance file SHA differs')
    verdict=json.loads(verdict_path.read_text())
    if verdict.get('accepted') is not True or verdict.get('blocking')!=0 or verdict.get('reviewer_role')!='independent' or verdict.get('report_sha256')!=spec.sha256(acceptance) or verdict.get('source_closure_sha256')!=expected['source_closure_sha256'] or verdict.get('sds_sha256')!=migration.PLAN_SHA256:
        raise orchestration.AdmissionError('independent v07 acceptance is not passing for this closure')
    if recovery.sha256(spec.ROOT/SOURCE_PATH)!=SOURCE_SHA256:raise orchestration.AdmissionError('designated source bytes differ')
    return manifest

def verify_native_resume(path,expected_sha=None):
    state=recovery.preload_checkpoint(path,expected_sha,mode='resume')
    recovery.resume_contract(state,seed=spec.TRAINING_SEED,num_envs=spec.TRAINING_NUM_ENVS)
    source=state['source'];continuation=source.get('continuation') or {}
    if source.get('engineering_only') or source.get('counts_against_training') is False:
        raise orchestration.AdmissionError('engineering evidence cannot resume production')
    if continuation.get('continuation_id')!=migration.CONTINUATION_ID or continuation.get('parent',{}).get('checkpoint_sha256')!=SOURCE_SHA256:
        raise orchestration.AdmissionError('native resume lacks the attested stop4817 parent')
    return state

def prepare_runner(contract,output_dir):
    """Bare-model load and independently verified loaded/ready, before learning."""
    verify_plan()
    import train_a2_ideal_stairs_v07 as worker
    from .curriculum import CurriculumManager
    from .runner import V07Runner
    output_dir=Path(output_dir)
    native=contract.get('resume_kind')=='native'
    if native:
        input_path=Path(contract['resume_run']);input_sha=contract.get('resume_sha256')
        preloaded=verify_native_resume(input_path,input_sha);original=preloaded
        input_sha=preloaded['checkpoint_sha256']
    else:
        original=migration.load_designated_source(spec.ROOT/SOURCE_PATH)
        preloaded,diff=migration.migrate_stop_4817(original,source_sha256=SOURCE_SHA256)
        # Engineering source-reload uses this same production load implementation,
        # but its output can never become a formal resume source.
        if output_dir.resolve()!=(spec.ROOT/OUTPUT_DIR).resolve():
            preloaded['source'].update(engineering_only=True,counts_against_training=False)
        write_json(output_dir/'migration_diff.json',diff)
        input_path=output_dir/'checkpoints'/'migration_source_004817.pt'
        input_sha=recovery.atomic_save(input_path,preloaded)
        preloaded=recovery.preload_checkpoint(input_path,input_sha,mode='resume')
    device=contract['device'];cur=preloaded['curriculum']
    model=model_module.V07ActorCritic().to(device)
    # Training must never pass the evaluation-only tile override.
    env=worker.build_environment(contract,phase=cur['phase'],substage=cur['substage'],level_tick=cur['level_tick'])
    trainer=ppo_module.PPOTrainer(model,env)
    manager=CurriculumManager(num_envs=int(contract['num_envs']),seed=int(contract['seed']),state=cur)
    holder={};guard=worker.build_measured_resource_guard(output_dir,device,holder) if str(device).startswith('cuda') else None
    runner=V07Runner(model=model,ppo=trainer,curriculum=manager,env=env,directory=output_dir,
        seed=contract['seed'],device=device,mode='resume',source=deepcopy(preloaded['source']),
        telemetry=worker.build_telemetry(output_dir,int(contract['seed'])),resource_guard=guard,
        acceptance_file=contract.get('acceptance_file'),require_resume_evidence=True)
    from .operator_control import ControlProtocol
    runner.operator_control=ControlProtocol(output_dir/'control.json',output_dir/'control_evidence')
    runner.stop_flag=runner.operator_control.poll
    holder['runner']=runner
    runner.case_runner=worker.build_case_runner(model,contract,cur['phase'],cur['substage'])
    loaded_contract=runner.load(input_path,input_sha)
    loaded=runner.save_verified('resume_loaded')
    if native:
        try:
            if not runner._poll_operator_control():
                runner._complete_pending_evaluations();runner._complete_pending_boundary()
        except Exception as error:
            runner._save_boundary_exception(error)
            raise
    elif runner.pending_boundary is not None or runner.pending_evaluations:
        raise recovery.RecoveryBlocked('designated source unexpectedly contains pending work')
    ready=None
    if runner.stop_state is None and runner.lifecycle_state!='complete':ready=runner.save_verified('resume_ready')
    actual=runner.checkpoint_state()
    checks={key:migration.tree_digest(actual[key])==migration.tree_digest(original[key]) for key in ('model','optimizer','normalizer','learning')}
    checks['metadata']=migration.tree_digest(model.transfer_metadata)==migration.tree_digest(original['initialization']['transfer_metadata'])
    checks['current_lr']=trainer.optimizer.param_groups[0]['lr']==original['learning']['learning_rate']
    checks['source_chain']=actual['source']['continuation']['parent']['checkpoint_sha256']==SOURCE_SHA256
    if not native:checks['generation_once']=actual['curriculum']['generation']==73 and actual['curriculum']['generation_updates']==0
    if not all(checks.values()):raise recovery.RecoveryBlocked('native loaded/ready preservation failed',audit=checks)
    command=getattr(env,'command',None);command_evidence=None
    if command is not None:
        vx=command[:,0].detach().cpu()
        command_evidence=dict(vx_min=float(vx.min()),vx_max=float(vx.max()),configured_bounds=list(spec.COMMAND_INITIAL_VX),
            episodes=int(vx.numel()),vy_zero=bool((command[:,1]==0).all()),wz_zero=bool((command[:,2]==0).all()),stage_iteration=int(env.stage_iteration))
        if not(.3-1e-6<=float(vx.min())<=float(vx.max())<=.6+1e-6) or not command_evidence['vy_zero'] or not command_evidence['wz_zero']:
            raise recovery.RecoveryBlocked('restored native command outside approved bounds')
    receipt=dict(recipe=spec.RECIPE_VERSION,sds_sha256=migration.PLAN_SHA256,original_checkpoint_sha256=SOURCE_SHA256,
        staged_checkpoint_sha256=input_sha,source_closure_sha256=preloaded['source']['source_closure_sha256'],
        resume_kind='native' if native else 'designated_4817',resume_contract=loaded_contract,resume_loaded=loaded,resume_ready=ready,
        preserved_checks=checks,actual_learning=actual['learning'],command_evidence=command_evidence,
        generation=actual['curriculum']['generation'],generation_updates=actual['curriculum']['generation_updates'],
        first_new_update=int(actual['learning']['update'])+1,resume_from_update=int(original['learning']['update']),lost_updates=0,
        replay_billed_updates=False,full_evaluation_rng_isolation='not_established_Q008',new_episode=True,physical_trajectory_continuous=False)
    write_json(output_dir/'recovery_v07'/'receipt.json',receipt)
    return runner,receipt

def build_worker(contract,output_dir):
    runner,receipt=prepare_runner(contract,output_dir)
    result=runner.run(max_updates=int(contract['max_updates']))
    if runner.telemetry is not None:runner.telemetry.finish(result)
    write_json(Path(output_dir)/'result.json',result)
    return result

def build_parser():
    parser=orchestration.build_parser()
    parser.add_argument('--source-checkpoint',default=SOURCE_PATH)
    parser.add_argument('--source-sha256',default=SOURCE_SHA256)
    parser.add_argument('--resume-sha256',default=None)
    parser.add_argument('--v07-manifest',default=MANIFEST_PATH)
    return parser

def resolve_request(args):
    if args.mode!='train' or any(getattr(args,key) for key in ('allow_q005_bridge','allow_q006_bridge','allow_q007_bridge')):
        raise orchestration.AdmissionError('v07 admits native training/recovery only, never legacy bridges')
    supplied=orchestration._provided(args)
    config=spec.recipe_config(**supplied)
    manifest=verify_manifest(args.v07_manifest)
    if args.resume_run:
        if not args.output_dir:raise orchestration.AdmissionError('native resume requires an explicit new segment directory')
        output=Path(args.output_dir);output=output if output.is_absolute() else spec.ROOT/output
        if output.resolve()==(spec.ROOT/OUTPUT_DIR).resolve() or not output.resolve().is_relative_to((spec.ROOT/spec.LOG_ROOT).resolve()):
            raise orchestration.AdmissionError('native resume must use a new directory under the v07 log root')
        resume_path=Path(args.resume_run);resume_path=resume_path if resume_path.is_absolute() else spec.ROOT/resume_path
        state=verify_native_resume(resume_path,args.resume_sha256)
        resume=str(resume_path);sha=state['checkpoint_sha256'];kind='native';at=int(state['learning']['update'])
    else:
        if (spec.ROOT/args.source_checkpoint).resolve()!=(spec.ROOT/SOURCE_PATH).resolve() or args.source_sha256!=SOURCE_SHA256:
            raise orchestration.AdmissionError('first native run accepts only the exact designated stop4817')
        output=spec.ROOT/OUTPUT_DIR
        if args.output_dir and (spec.ROOT/args.output_dir).resolve()!=output.resolve():raise orchestration.AdmissionError('first native output directory is fixed')
        resume=SOURCE_PATH;sha=SOURCE_SHA256;kind='designated_4817';at=4817
    if output.exists():raise orchestration.AdmissionError('output already exists; never reuse a run directory')
    return dict(config,mode='resume',resume_kind=kind,resume_run=resume,resume_sha256=sha,output_dir=str(output),
        source_checkpoint=SOURCE_PATH,source_sha256=SOURCE_SHA256,acceptance_file=str(spec.ROOT/ACCEPTANCE_FILE),
        v07_manifest_sha256=(spec.sha256(spec.ROOT/args.v07_manifest) if (spec.ROOT/args.v07_manifest).is_file() else None),
        segment=dict(run_id='v07_seed42',segment_id=output.name,rollback_from_update=at,resume_from_update=at,lost_updates=0,replay_billed_updates=False))

def main(argv=None):
    args=build_parser().parse_args(argv);contract=resolve_request(args)
    write_json(spec.ROOT/IMPLEMENTATION/'preflight'/'v07_preflight.json',dict(contract=contract,output_directory_created=False,
        cuda_initialized=__import__('torch').cuda.is_initialized(),timestamp=time.time()))
    print(json.dumps(dict(preflight_only=not args.execute,output_dir=contract['output_dir'],output_directory_created=False),indent=2))
    if not args.execute:return 0
    supervisor=orchestration.V07Orchestrator(args,root=spec.ROOT,preflight_dir=spec.ROOT/IMPLEMENTATION/'preflight',
        worker_factory=build_worker,gpu_registry=orchestration.worker_registry_path())
    supervisor.contract=contract;result=supervisor.execute();print(json.dumps(dict(status=result['status']),indent=2));return 0
