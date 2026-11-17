"""Native CLI separates first designated migration from later native segments."""
from pathlib import Path
import importlib
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec,orchestration

def module():return importlib.import_module('src.tasks.velocity.v07.runtime')

def test_native_first_request_is_fixed_and_preflight_does_not_create_output(monkeypatch,tmp_path):
    m=module();args=m.build_parser().parse_args([])
    monkeypatch.setattr(m,'verify_manifest',lambda path:{'accepted':True})
    c=m.resolve_request(args)
    assert c['resume_kind']=='designated_4817' and c['resume_run']==m.SOURCE_PATH
    assert c['max_updates']==48000 and c['num_envs']==512 and c['eval_num_envs']==8
    assert c['source_sha256']==m.SOURCE_SHA256 and c['execute'] is False
    assert c['output_dir']==str(spec.ROOT/m.OUTPUT_DIR)

@pytest.mark.parametrize('argv',[['--source-sha256','0'*64],['--mode','probe'],['--allow-q007-bridge'],['--num-envs','8'],['--max-updates','36000'],['--output-dir','logs/a2_ideal_stairs_v07/wrong_first']])
def test_first_contract_cannot_choose_another_source_or_recipe(argv,monkeypatch):
    m=module();monkeypatch.setattr(m,'verify_manifest',lambda path:{})
    with pytest.raises((orchestration.AdmissionError,ValueError)):
        m.resolve_request(m.build_parser().parse_args(argv))

def test_native_resume_requests_new_segment_without_requiring_source4817_again(monkeypatch,tmp_path):
    m=module();checkpoint=tmp_path/'native.pt'
    monkeypatch.setattr(m,'verify_manifest',lambda path:{})
    monkeypatch.setattr(m,'verify_native_resume',lambda path,expected_sha=None:dict(learning=dict(update=6100),curriculum=dict(generation=74),checkpoint_sha256='e'*64))
    args=m.build_parser().parse_args(['--resume-run',str(checkpoint),'--output-dir',str(spec.ROOT/spec.LOG_ROOT/'resume_seed42_segment02')])
    c=m.resolve_request(args)
    assert c['resume_kind']=='native' and c['resume_run']==str(checkpoint)
    assert c['segment']['resume_from_update']==6100 and c['segment']['lost_updates']==0
    assert c['segment']['replay_billed_updates'] is False

def test_prepare_runner_actual_bare_adam_evidence_and_engineering_source_refusal(tmp_path,monkeypatch):
    m=module()
    import train_a2_ideal_stairs_v07 as worker
    from v07_runner_support import RecordingEnv,bare_runner,migration,recovery
    monkeypatch.setattr(worker,'build_environment',lambda *a,**k:RecordingEnv())
    monkeypatch.setattr(worker,'build_telemetry',lambda *a,**k:None)
    monkeypatch.setattr(worker,'build_case_runner',lambda *a,**k:None)
    contract=dict(spec.recipe_config(),device='cpu')
    runner,receipt=m.prepare_runner(contract,tmp_path/'engineering')
    assert all(receipt['preserved_checks'].values())
    assert receipt['generation']==73 and receipt['first_new_update']==4818
    assert receipt['resume_loaded']['verification']['aligned'] and receipt['resume_ready']['verification']['aligned']
    assert runner.curriculum.state.last_update==4817 and runner.env.stage_iteration==2117
    ready=receipt['resume_ready']
    with pytest.raises(orchestration.AdmissionError):m.verify_native_resume(ready['path'],ready['sha256'])
    second=bare_runner(tmp_path/'second');second.load(ready['path'],ready['sha256'])
    assert second.curriculum.state.generation==73 and second.curriculum.state.generation_updates==0
    assert second.env.stage_iteration==2117

def test_manifest_binds_native_schema_source_closure_and_both_independent_files(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'verify_plan',lambda:None)
    manifest_path=tmp_path/'V07_BRIDGE_MANIFEST.json';acceptance=tmp_path/'INDEPENDENT_ACCEPTANCE.md'
    acceptance.write_text('independent actual fixture report\n')
    hashes={'source.py':'a'*64};closure=m.recovery.digest(hashes)
    monkeypatch.setattr(m,'MANIFEST_PATH',str(manifest_path));monkeypatch.setattr(m,'ACCEPTANCE_FILE',str(acceptance))
    monkeypatch.setattr(m.spec,'source_hashes',lambda **kwargs:hashes)
    verdict=dict(accepted=True,blocking=0,reviewer_role='independent',report_sha256=m.spec.sha256(acceptance),source_closure_sha256=closure,sds_sha256=m.migration.PLAN_SHA256)
    verdict_path=acceptance.with_suffix('.json');verdict_path.write_text(__import__('json').dumps(verdict))
    manifest=dict(schema=9,manifest_schema='a2-v07-bridge-manifest-v1',recipe=spec.RECIPE_VERSION,
        sds_sha256=m.migration.PLAN_SHA256,source_checkpoint_sha256=m.SOURCE_SHA256,source_closure_sha256=closure,
        contract_sha256=spec.contract_sha256(),source_hashes=hashes,independent_acceptance_sha256=m.spec.sha256(acceptance),independent_verdict_sha256=m.spec.sha256(verdict_path))
    manifest_path.write_text(__import__('json').dumps(manifest));assert m.verify_manifest(manifest_path)==manifest
    for key,value in [('schema',9.0),('manifest_schema','unknown'),('source_checkpoint_sha256','0'*64),('independent_verdict_sha256','0'*64),('contract_sha256','0'*64)]:
        changed=dict(manifest);changed[key]=value;manifest_path.write_text(__import__('json').dumps(changed))
        with pytest.raises(orchestration.AdmissionError):m.verify_manifest(manifest_path)

def test_native_prepare_keeps_existing_generation_progress_and_bad_queue(tmp_path,monkeypatch):
    m=module();import train_a2_ideal_stairs_v07 as worker
    from v07_runner_support import RecordingEnv,migrated_checkpoint,loaded_runner,set_synthetic_boundary
    from src.tasks.velocity.v07 import regression_policy
    from v07_test_support import validation_payload
    checkpoint=migrated_checkpoint(tmp_path/'source.pt');original=loaded_runner(tmp_path/'old',checkpoint)
    set_synthetic_boundary(original,5900,gu=1083)
    payload,identity=validation_payload(5900,73,4)
    original.curriculum.regression_policy_state,_=regression_policy.apply_validation(original.curriculum.regression_policy_state,payload,expected_identity=identity,generation_updates=1083)
    saved=original.save('native_source')
    monkeypatch.setattr(worker,'build_environment',lambda *a,**k:RecordingEnv())
    monkeypatch.setattr(worker,'build_telemetry',lambda *a,**k:None)
    monkeypatch.setattr(worker,'build_case_runner',lambda *a,**k:None)
    contract=dict(spec.recipe_config(),device='cpu',resume_kind='native',resume_run=saved['path'],resume_sha256=saved['sha256'])
    resumed,receipt=m.prepare_runner(contract,tmp_path/'segment')
    assert resumed.curriculum.regression_policy_state==original.curriculum.regression_policy_state
    assert resumed.curriculum.state.generation_updates==1083 and resumed.curriculum.state.generation==73
    assert receipt['resume_kind']=='native' and receipt['first_new_update']==5901
    assert all(receipt['preserved_checks'].values())
