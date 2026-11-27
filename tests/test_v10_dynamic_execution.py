"""Small CPU accounting/dispatch tests; no allocation, probe or training."""
import copy
import json
from pathlib import Path
import pytest
from src.tasks.velocity.v10 import dynamic_execution as d
from src.tasks.velocity.v10.checkpoint import BudgetLedger


@pytest.fixture
def origin(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();cp=tmp_path/'origin50.pt';cp.write_bytes(b'original model Adam replay unchanged')
    ledger=root/'learning_costs.jsonl';b=BudgetLedger(ledger);b.charge('D1',2457600,limit=2457600,kind='learning');b.record_update('original50','D1',2457600)
    metadata=dict(from_num_envs=1024,from_code_sha256='old-exact-source',physical_sha256='a'*64,
                  rounds=dict(D1=50,D2=0,D3=0,PPO=0),accounting=b.report())
    monkeypatch.setattr(d,'_checkpoint_metadata',lambda path:copy.deepcopy(metadata))
    monkeypatch.setattr(d,'code_fingerprint',lambda:'current-profile-source')
    return root,cp,ledger


def profile(origin,n=2048,write=True):
    root,cp,ledger=origin;path=root/f'execution_profile_N{n}.json'
    row=d.build_execution_profile(path,num_envs=n,origin_checkpoint=cp,learning_ledger_path=ledger,write=write)
    return path,row


@pytest.mark.parametrize('n,cap,updates',[(2048,46694400,20),(512,13516800,10),(768,19046400,20)])
def test_fixed_rounds_carried_costs_scaled_future_samples(origin,n,cap,updates):
    root,cp,ledger=origin;oldcp=cp.read_bytes();oldcost=ledger.read_bytes()
    path,row=profile(origin,n)
    assert row['sample_caps']['D1']==cap
    assert row['origin_rounds']['D1']==50 and row['round_budgets']['D1']==500 and row['gates']['D1']==[200,400,500]
    assert row['save_rounds']==50 and row['dagger_updates_per_round']==updates
    assert row['resource_status']=='unverified_for_requested_N' and row['resource_probe_requested'] is False
    assert d.validate_execution_profile(path,num_envs=n)['num_envs']==n
    assert cp.read_bytes()==oldcp and ledger.read_bytes()==oldcost


def test_profile_origin_tamper_and_current_identity_are_not_waived(origin):
    path,row=profile(origin);cp=origin[1];cp.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='reference changed'):d.validate_execution_profile(path)


def test_new_N_after_crash_before_registry_is_rejected_before_profile_creation(origin):
    root,cp,ledger=origin;oldpath,row=profile(origin,2048)
    b=BudgetLedger(ledger);b.charge('D1',2048,limit=row['sample_caps']['D1'],kind='learning')
    assert not Path(str(ledger)+'.dynamic_latest.json').exists()
    target=root/'execution_profile_N512.json';before=ledger.read_bytes()
    with pytest.raises(d.StopTraining,match='second_dynamic_N_switch'):
        d.run_dynamic_execution(run_dir=root,profile_path=target,num_envs=512,check_only=True)
    assert not target.exists() and ledger.read_bytes()==before


def test_missing_profile_check_only_calculates_without_writing(origin,monkeypatch):
    root,cp,ledger=origin;original=d.build_execution_profile
    def build(path,**kwargs):return original(path,origin_checkpoint=cp,**kwargs)
    monkeypatch.setattr(d,'build_execution_profile',build)
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=d.run_dynamic_execution(run_dir=root,profile_path=root/'execution_profile_N2048.json',num_envs=2048,check_only=True)
    assert result['passed'] and result['executed'] is False and result['profile_exists'] is False
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dynamic_D1_starts_at_original50_and_first_absolute200(origin,monkeypatch):
    path,row=profile(origin);checked=d.validate_execution_profile(path)
    o=d.DynamicOrchestrator(origin[0]/'dynamic_N2048',checked)
    o.state.update(learning_checkpoint=str(origin[1]),schedule={'gates':d.GATES})
    calls=[]
    def worker(key,course,target,resume=None,acceptance=None):
        calls.append((key,course,target,resume));raise d.StopTraining('CPU boundary')
    o.worker=worker
    with pytest.raises(d.StopTraining,match='CPU boundary'):o._train_dagger()
    assert calls==[('learning_D1_200','D1',200,str(origin[1]))]


def test_dynamic_worker_validates_retained_round_not_mixed_N_sample_quotient(origin,monkeypatch):
    path,row=profile(origin);checked=d.validate_execution_profile(path);o=d.DynamicOrchestrator(origin[0]/'dynamic_N2048',checked)
    o.state.update(num_envs=2048,physical_sha256='a'*64)
    cp=origin[0]/'new200.pt';cp.write_bytes(b'new checkpoint')
    consumed=2457600+150*2048*48
    assert consumed!=200*2048*48
    result=dict(gate_reached=True,course='D1',round=200,consumed=consumed,physical_sha256='a'*64,checkpoint=str(cp),checkpoint_sha256=d.file_sha(cp),
                budget={'charged':{'learning:D1':consumed}})
    commands=[]
    o._teacher_receipts=lambda:{}
    def launch(key,command,result_path):commands.append(command);return result
    o._launch=launch
    assert o.worker('learning_D1_200','D1',200,str(origin[1]))==str(cp)
    assert '--execution-profile' in commands[0] and '--formal-handoff' not in commands[0] and 'probe' not in commands[0]
    result['budget']['charged']['learning:D1']=row['sample_caps']['D1']+1
    with pytest.raises(d.StopTraining,match='charged_cap'):o.worker('learning_D1_200','D1',200,str(origin[1]))
