"""C1 admission and worker adapter; original v06 training math stays unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import time

from . import spec, recovery, model as model_module, ppo as ppo_module, orchestration
from . import continuation_c1 as migration
from .c1_contract import (CONTINUATION_ID, PLAN_PATH, PLAN_SHA256, SOURCE_PATH,
                          SOURCE_SHA256, OUTPUT_DIR, ACCEPTANCE_FILE, MANIFEST_PATH,
                          IMPLEMENTATION)


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')


def verify_plan():
    if getattr(spec,'CONTINUATION_ID',None)!=CONTINUATION_ID:
        raise recovery.RecoveryBlocked('C1 identity was not selected before imports')
    if spec.sha256(spec.ROOT/PLAN_PATH)!=PLAN_SHA256:
        raise recovery.RecoveryBlocked('C1 plan SHA differs from approved contract')


def verify_manifest(path=MANIFEST_PATH):
    """Fail closed on source, plan, frozen closure and independent acceptance."""
    verify_plan()
    path=Path(path)
    if path.resolve()!=(spec.ROOT/MANIFEST_PATH).resolve():
        raise orchestration.AdmissionError('only the fixed C1 bridge manifest is allowed')
    manifest=json.loads(path.read_text())
    expected=dict(continuation_id=CONTINUATION_ID,plan_sha256=PLAN_SHA256,
                  source_checkpoint_sha256=SOURCE_SHA256,
                  source_closure_sha256=recovery.digest(spec.source_hashes()),
                  contract_sha256=spec.contract_sha256())
    for key,want in expected.items():
        if manifest.get(key)!=want:
            raise orchestration.AdmissionError('C1 manifest mismatch: '+key)
    if manifest.get('source_hashes')!=spec.source_hashes():
        raise orchestration.AdmissionError('C1 manifest file closure mismatch')
    acceptance=spec.ROOT/ACCEPTANCE_FILE
    if manifest.get('independent_acceptance_sha256')!=spec.sha256(acceptance):
        raise orchestration.AdmissionError('C1 independent acceptance SHA mismatch')
    verdict=json.loads(acceptance.with_suffix('.json').read_text())
    if (verdict.get('blocking')!=0 or verdict.get('accepted') is not True
            or verdict.get('source_closure_sha256')!=expected['source_closure_sha256']
            or verdict.get('plan_sha256')!=PLAN_SHA256
            or verdict.get('report_sha256')!=spec.sha256(acceptance)
            or verdict.get('reviewer_role')!='independent'):
        raise orchestration.AdmissionError('independent C1 acceptance is not passing for this closure')
    if manifest.get('independent_verdict_sha256')!=spec.sha256(acceptance.with_suffix('.json')):
        raise orchestration.AdmissionError('C1 independent verdict SHA mismatch')
    if recovery.sha256(spec.ROOT/SOURCE_PATH)!=SOURCE_SHA256:
        raise orchestration.AdmissionError('C1 designated stop source bytes differ')
    return manifest


def prepare_runner(contract, output_dir):
    """Production bare-model preload/load/save/reload path, before new learning."""
    verify_plan()
    import train_a2_ideal_stairs_v06 as worker
    from .curriculum import CurriculumManager
    from .runner import V06Runner
    output_dir=Path(output_dir)
    source=migration.load_designated_source(spec.ROOT/SOURCE_PATH)
    state,diff=migration.migrate_stop_4224(source,source_sha256=SOURCE_SHA256)
    write_json(output_dir/'migration_diff.json',diff)
    staged=output_dir/'checkpoints/migration_source_004224.pt'
    staged_sha=recovery.atomic_save(staged,state)
    # The new contract checkpoint is formally admitted before CUDA/env allocation.
    preloaded=recovery.preload_checkpoint(staged,staged_sha,mode='resume')
    model=model_module.V06ActorCritic().to(contract['device'])
    cur=preloaded['curriculum']
    env=worker.build_environment(contract,phase=cur['phase'],substage=cur['substage'],
        level_tick=cur['level_tick'])
    trainer=ppo_module.PPOTrainer(model,env)
    manager=CurriculumManager(num_envs=int(contract['num_envs']),seed=int(contract['seed']),state=cur)
    holder={}
    guard=(worker.build_measured_resource_guard(output_dir,contract['device'],holder)
           if str(contract['device']).startswith('cuda') else None)
    runner=V06Runner(model=model,ppo=trainer,curriculum=manager,env=env,directory=output_dir,
        seed=contract['seed'],device=contract['device'],mode='resume',
        telemetry=worker.build_telemetry(output_dir,int(contract['seed'])),
        resource_guard=guard,acceptance_file=contract.get('acceptance_file'),
        source=deepcopy(preloaded['source']),require_resume_evidence=True)
    holder['runner']=runner
    runner.case_runner=worker.build_case_runner(model,contract,cur['phase'],cur['substage'])
    loaded_contract=runner.load(staged,staged_sha)
    loaded=runner.save_verified('resume_loaded')
    # The designated stop4224 has no pending transaction. Ready is still required
    # explicitly, so no first learning call can bypass either reload verification.
    if runner.pending_boundary is not None or runner.pending_evaluations:
        raise recovery.RecoveryBlocked('C1 stop4224 unexpectedly has pending work')
    ready=runner.save_verified('resume_ready')
    actual=runner.checkpoint_state()
    checks={key:migration.tree_digest(actual[key])==migration.tree_digest(source[key])
            for key in ['model','optimizer','normalizer','learning']}
    checks['metadata']=migration.tree_digest(model.transfer_metadata)==migration.tree_digest(
        source['initialization']['transfer_metadata'])
    checks['current_lr']=trainer.optimizer.param_groups[0]['lr']==1e-5
    checks['generation_once']=runner.curriculum.state.generation==68
    checks['source_chain']=actual['source']['continuation']['parent']['checkpoint_sha256']==SOURCE_SHA256
    if not all(checks.values()):
        raise recovery.RecoveryBlocked('C1 loaded/ready source preservation failed',audit=checks)
    command=getattr(env,'command',None)
    command_evidence=None
    if command is not None:
        vx=command[:,0].detach().cpu()
        command_evidence=dict(vx_min=float(vx.min()),vx_max=float(vx.max()),
            configured_bounds=list(spec.COMMAND_INITIAL_VX),episodes=int(vx.numel()),
            vy_zero=bool((command[:,1]==0).all()),wz_zero=bool((command[:,2]==0).all()),
            stage_iteration=int(getattr(env,'stage_iteration',0)))
        if not (.3-1e-6<=float(vx.min())<=float(vx.max())<=.6+1e-6):
            raise recovery.RecoveryBlocked('restored C1 environment sampled outside [.3,.6]')
    receipt=dict(continuation_id=CONTINUATION_ID,plan_sha256=PLAN_SHA256,
        original_checkpoint_sha256=SOURCE_SHA256,staged_checkpoint_sha256=staged_sha,
        source_closure_sha256=state['source']['source_closure_sha256'],
        resume_contract=loaded_contract,resume_loaded=loaded,resume_ready=ready,
        preserved_checks=checks,actual_learning=actual['learning'],command_evidence=command_evidence,
        generation=68,first_new_update=4225,rollback_from_update=4224,resume_from_update=4224,
        lost_updates=0,replay_billed_updates=False,
        full_evaluation_rng_isolation='not_established_Q008',
        new_episode=True,physical_trajectory_continuous=False)
    write_json(output_dir/'recovery_c1/receipt.json',receipt)
    return runner,receipt


def build_worker(contract, output_dir):
    runner,receipt=prepare_runner(contract,output_dir)
    result=runner.run(max_updates=int(contract['max_updates']))
    if runner.telemetry is not None:runner.telemetry.finish(result)
    write_json(Path(output_dir)/'result.json',result)
    return result


def main(argv=None):
    parser=orchestration.build_parser()
    parser.add_argument('--source-checkpoint',default=SOURCE_PATH)
    parser.add_argument('--source-sha256',default=SOURCE_SHA256)
    parser.add_argument('--c1-manifest',default=MANIFEST_PATH)
    args=parser.parse_args(argv)
    verify_plan()
    if args.mode!='train' or args.resume_run or any(getattr(args,k) for k in
            ['allow_q005_bridge','allow_q006_bridge','allow_q007_bridge']):
        raise orchestration.AdmissionError('C1 only performs the designated continuation, never fresh/legacy bridges')
    if (Path(args.source_checkpoint).resolve()!=(spec.ROOT/SOURCE_PATH).resolve()
            or args.source_sha256!=SOURCE_SHA256):
        raise orchestration.AdmissionError('C1 designated stop checkpoint mismatch')
    supplied={key:getattr(args,key) for key in orchestration.RECIPE_FLAGS
              if getattr(args,key) is not None}
    config=spec.recipe_config(**supplied)
    if Path(config['output_dir']).resolve()!=(spec.ROOT/OUTPUT_DIR).resolve():
        raise orchestration.AdmissionError('C1 output directory is fixed')
    output=spec.ROOT/OUTPUT_DIR
    if output.exists():raise orchestration.AdmissionError('C1 output already exists; never reuse it')
    manifest=verify_manifest(args.c1_manifest)
    contract=dict(config,mode='resume',resume_run=SOURCE_PATH,
        output_dir=str(output),acceptance_file=str(spec.ROOT/ACCEPTANCE_FILE),
        source_checkpoint=SOURCE_PATH,source_sha256=SOURCE_SHA256,
        continuation_id=CONTINUATION_ID,c1_manifest_sha256=spec.sha256(args.c1_manifest),
        segment=dict(run_id='trial02',segment_id=CONTINUATION_ID,
                     rollback_from_update=4224,resume_from_update=4224,lost_updates=0,
                     replay_billed_updates=False))
    preflight=dict(contract=contract,manifest=manifest,output_directory_created=False,
                   cuda_initialized=__import__('torch').cuda.is_initialized(),timestamp=time.time())
    write_json(spec.ROOT/IMPLEMENTATION/'preflight/c1_preflight.json',preflight)
    print(json.dumps(dict(preflight_only=not args.execute,output_dir=str(output),
                         output_directory_created=False),indent=2))
    if not args.execute:return 0
    # Reuse the original worker lease, resource admission, controls and failures.
    supervisor=orchestration.V06Orchestrator(args,root=spec.ROOT,
        preflight_dir=spec.ROOT/IMPLEMENTATION/'preflight',worker_factory=build_worker,
        gpu_registry=orchestration.worker_registry_path())
    supervisor.contract=contract
    result=supervisor.execute()
    print(json.dumps(dict(status=result['status']),indent=2))
    return 0
