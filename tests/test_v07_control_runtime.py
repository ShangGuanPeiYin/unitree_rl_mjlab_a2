"""Real runtime->runner control file boundary integration; no GPU simulation."""
from copy import deepcopy
import json
import pytest
import torch
from v07_runner_support import *
from src.tasks.velocity.v07 import runtime,operator_control,orchestration,evaluation,spec

def prepare(tmp_path,monkeypatch,raw):
    import train_a2_ideal_stairs_v07 as worker
    monkeypatch.setattr(worker,'build_environment',lambda *a,**k:RecordingEnv())
    monkeypatch.setattr(worker,'build_telemetry',lambda *a,**k:None)
    monkeypatch.setattr(worker,'build_case_runner',lambda *a,**k:None)
    out=tmp_path/'run';out.mkdir();(out/'control.json').write_bytes(raw)
    r,receipt=runtime.prepare_runner(dict(spec.recipe_config(),device='cpu'),out)
    return r,receipt

@pytest.mark.parametrize('action,reason',[('stop','user_requested_stop'),('pause','operator_pause')])
def test_first4818_control_stops_before_any_ppo_resource_or_final(tmp_path,monkeypatch,action,reason):
    raw=json.dumps(dict(action=action,reason='continue',resume_authorized=True)).encode()
    r,_=prepare(tmp_path,monkeypatch,raw)
    def forbidden(*args,**kwargs):raise AssertionError('GPU/resource work after operator request')
    r.ppo.collect=forbidden;r.resource_guard=forbidden;r.final_test=forbidden
    result=r.run(max_updates=4818)
    assert result['state']=='user_requested_stop' and result['stop_reason']==reason
    assert result['failure_class']=='user' and result['update']==4817
    stop=recovery.preload_checkpoint(r.directory/'checkpoints'/'stop_004817.pt',mode='train')
    assert stop['ledger']['stop_reason']==reason and stop['learning']['adam_steps']==96340
    assert (r.directory/'control.json').read_bytes()==raw
    if action=='pause':assert result['stop_evidence']['pause_requested'] is True and result['stop_evidence']['resume_authorized'] is False
    operator_control.write_control(r.directory/'control.json',dict(action='continue'))
    assert r.run(max_updates=4818)['stop_reason']==reason

@pytest.mark.parametrize('raw',[b'{bad json',b'{}',b'[]'])
def test_invalid_control_is_named_engineering_exception_with_original_bytes(tmp_path,monkeypatch,raw):
    r,_=prepare(tmp_path,monkeypatch,raw)
    def forbidden(*args,**kwargs):raise AssertionError('PPO/resource after invalid control')
    r.ppo.collect=forbidden;r.resource_guard=forbidden
    with pytest.raises(operator_control.ControlProtocolError):r.run(max_updates=4818)
    checkpoint=recovery.preload_checkpoint(r.directory/'checkpoints'/'exception_004817.pt',mode='train')
    assert checkpoint['ledger']['error_type']=='ControlProtocolError'
    assert checkpoint['ledger']['stop_state'] is None and r.update==4817
    assert (r.directory/'control.json').read_bytes()==raw
    audit=json.loads((r.directory/'control_error.json').read_text())
    assert __import__('base64').b64decode(audit['raw_base64'])==raw


def twenty_steps(r,write_during_update=None):
    calls=[]
    def collect():calls.append('collect');return []
    def update():
        for i in range(20):
            for parameter in r.model.parameters():
                if parameter.requires_grad:parameter.grad=torch.full_like(parameter,1e-6)
            r.ppo.optimizer.step();r.ppo.optimizer.zero_grad(set_to_none=True)
            r.ppo.learning_state['optimizer_steps']+=1
            if i==0 and write_during_update:write_during_update()
        r.ppo.learning_state['completed_ppo_updates']+=1
        return dict(loss=0.,update=r.ppo.learning_state['completed_ppo_updates'])
    r.ppo.collect=collect;r.ppo.update=update;return calls

def test_continue_with_stop_text_allows_one_update_then_mid_update_stop_blocks_next(tmp_path,monkeypatch):
    r,_=prepare(tmp_path,monkeypatch,b'{"action":"continue","reason":"stop","resume_authorized":true}')
    calls=twenty_steps(r,lambda:operator_control.write_control(r.directory/'control.json',dict(action='stop')))
    def forbidden(*args,**kwargs):raise AssertionError('resource call after observed mid-update stop')
    r.resource_guard=forbidden
    result=r.run(max_updates=4819)
    assert calls==['collect'] and result['update']==4818 and result['stop_reason']=='user_requested_stop'
    assert result['learning']['adam_steps']==96360
    state=recovery.preload_checkpoint(r.directory/'checkpoints'/'stop_004818.pt',mode='train')
    assert state['curriculum']['last_update']==4818 and state['curriculum']['generation_updates']==1
    assert state['ledger']['pending_boundary'] is not None
    assert state['ledger']['pending_boundary']['curriculum_decided'] is False


def test_pending_runtime_control_preserves_unfinished_evaluation_without_gpu_work(tmp_path,monkeypatch):
    import train_a2_ideal_stairs_v07 as worker
    cp=migrated_checkpoint(tmp_path/'source.pt');original=loaded_runner(tmp_path/'old',cp)
    set_synthetic_boundary(original,4900,substage='D25',gu=83)
    table=evaluation.make_cases('D','D25','validation',42,tile_id=17)
    original.pending_boundary=dict(update=4900,phase='D',substage='D25',generation=73,
        tasks=[dict(kind='validation',gate_kind='validation',retention_kind=None,tile=17,phase='D',substage='D25',case_table_sha256=table.sha256)],results=[],curriculum_decided=False,committed=True)
    saved=original.save('pending')
    out=tmp_path/'segment';out.mkdir();operator_control.write_control(out/'control.json',dict(action='pause'))
    monkeypatch.setattr(worker,'build_environment',lambda *a,**k:RecordingEnv())
    monkeypatch.setattr(worker,'build_telemetry',lambda *a,**k:None)
    def forbidden(*args,**kwargs):raise AssertionError('pending case runner was invoked after pause')
    monkeypatch.setattr(worker,'build_case_runner',lambda *a,**k:forbidden)
    contract=dict(spec.recipe_config(),device='cpu',resume_kind='native',resume_run=saved['path'],resume_sha256=saved['sha256'])
    r,receipt=runtime.prepare_runner(contract,out)
    assert r.stop_state=='user_requested_stop' and r.stop_reason=='operator_pause'
    assert r.pending_boundary['results']==[] and r.pending_boundary['curriculum_decided'] is False
    assert receipt['resume_ready'] is None and r.update==4900

# B CORE R1: a new helper and a later continue cannot release an observed latch.
@pytest.mark.parametrize('action',['pause','stop'])
@pytest.mark.parametrize('pending',[False,True])
def test_r1_unresolved_latched_request_refuses_native_before_environment_or_pending(tmp_path,monkeypatch,action,pending):
    import train_a2_ideal_stairs_v07 as worker
    cp=migrated_checkpoint(tmp_path/'source.pt');source=loaded_runner(tmp_path/'old',cp)
    if pending:
        set_synthetic_boundary(source,4900,substage='D25',gu=83)
        table=evaluation.make_cases('D','D25','validation',42,tile_id=17)
        source.pending_boundary=dict(update=4900,phase='D',substage='D25',generation=73,
            tasks=[dict(kind='validation',gate_kind='validation',retention_kind=None,tile=17,phase='D',substage='D25',case_table_sha256=table.sha256)],results=[],curriculum_decided=False,committed=True)
        cp=source.save('native_pending')
    monkeypatch.setattr(worker,'build_environment',lambda *a,**k:RecordingEnv())
    monkeypatch.setattr(worker,'build_telemetry',lambda *a,**k:None)
    monkeypatch.setattr(worker,'build_case_runner',lambda *a,**k:None)
    out=tmp_path/'observed';out.mkdir();raw=json.dumps(dict(action=action)).encode();(out/'control.json').write_bytes(raw)
    contract=dict(spec.recipe_config(),device='cpu',resume_kind='native',resume_run=cp['path'],resume_sha256=cp['sha256'])
    def io_fail(*args,**kwargs):raise OSError('injected control snapshot persistence error')
    with monkeypatch.context() as patch:
        patch.setattr(operator_control.ControlProtocol,'_snapshot',io_fail)
        with pytest.raises(operator_control.ControlProtocolError):runtime.prepare_runner(contract,out)
    at=4900 if pending else 4817;exception=out/'checkpoints'/f'exception_{at:06d}.pt'
    original_sha=recovery.sha256(exception)
    evidence=recovery.preload_checkpoint(exception,original_sha,mode='train')
    audit=evidence['ledger']['terminal']['evidence']['control_error']
    assert audit['latched_request']['action']==action and evidence['ledger']['error_type']=='ControlProtocolError'
    assert evidence['ledger']['stop_state'] is None and (out/'control.json').read_bytes()==raw
    if pending:assert evidence['ledger']['pending_boundary']['results']==[]
    later=tmp_path/'later';later.mkdir();operator_control.write_control(later/'control.json',dict(action='continue',resume_authorized=True))
    calls=[]
    def observed_env(*args,**kwargs):calls.append('environment');return RecordingEnv()
    monkeypatch.setattr(worker,'build_environment',observed_env)
    later_contract=dict(contract,resume_run=str(exception),resume_sha256=original_sha)
    with pytest.raises(recovery.RecoveryBlocked):runtime.prepare_runner(later_contract,later)
    assert calls==[] and not list(later.rglob('resume_ready*'))
    assert recovery.sha256(exception)==original_sha and (out/'control.json').read_bytes()==raw
    assert (later/'control.json').read_text().find('continue')>=0

def test_r1_correctable_bad_json_without_observed_latch_remains_nonterminal_resumable(tmp_path,monkeypatch):
    import train_a2_ideal_stairs_v07 as worker
    cp=migrated_checkpoint(tmp_path/'source.pt')
    monkeypatch.setattr(worker,'build_environment',lambda *a,**k:RecordingEnv())
    monkeypatch.setattr(worker,'build_telemetry',lambda *a,**k:None)
    monkeypatch.setattr(worker,'build_case_runner',lambda *a,**k:None)
    out=tmp_path/'invalid';out.mkdir();raw=b'{bad json';(out/'control.json').write_bytes(raw)
    contract=dict(spec.recipe_config(),device='cpu',resume_kind='native',resume_run=cp['path'],resume_sha256=cp['sha256'])
    with pytest.raises(operator_control.ControlProtocolError):runtime.prepare_runner(contract,out)
    exception=out/'checkpoints'/'exception_004817.pt';original_sha=recovery.sha256(exception)
    state=recovery.preload_checkpoint(exception,original_sha,mode='train')
    assert 'latched_request' not in state['ledger']['terminal']['evidence']['control_error']
    later=tmp_path/'corrected';later.mkdir();operator_control.write_control(later/'control.json',dict(action='continue'))
    restored,receipt=runtime.prepare_runner(dict(contract,resume_run=str(exception),resume_sha256=original_sha),later)
    assert restored.stop_state is None and receipt['resume_ready']['verification']['aligned']
    assert restored._check_resume_evidence(stage='learning') is True
    assert recovery.sha256(exception)==original_sha and (out/'control.json').read_bytes()==raw
