"""CPU controller seams only; never creates a physics environment."""
import copy
import json
from pathlib import Path
import pytest
from src.tasks.velocity.v10.orchestration import Orchestrator,StopTraining,execution_plan,file_sha


def test_handoff_resumes50_without_zero_entry_and_uses_absolute200(tmp_path):
    o=Orchestrator(tmp_path/'run',tmp_path/'engineering')
    origin=tmp_path/'trial50.pt';origin.write_text('complete retained trial50')
    o.state.update(learning_checkpoint=str(origin),formal_handoff={'path':'signed'},schedule=execution_plan(1024))
    calls=[]
    def worker(key,course,target,resume=None,acceptance=None):
        calls.append((key,course,target,resume))
        if key=='learning_D1_200':raise StopTraining('CPU boundary reached')
        return str(origin)
    o.worker=worker
    with pytest.raises(StopTraining,match='CPU boundary'):o._train_dagger()
    assert calls==[('learning_D1_200','D1',200,str(origin))]


def test_eval_recovery_needs_gpu_exclusion_and_never_refunds(tmp_path,monkeypatch):
    from src.tasks.velocity.v10.checkpoint import BudgetLedger
    from src.tasks.velocity.v10 import gpu_lock
    from contextlib import contextmanager
    o=Orchestrator(tmp_path/'run',tmp_path/'engineering',formal_handoff_plan=tmp_path/'plan.json')
    o.handoff_plan.write_text('{}');ledger=BudgetLedger(o.evaluation_ledger)
    ledger.reserve('interrupted','ALL',20,limit=100,kind='evaluation')
    before=ledger.report();entered=[]
    @contextmanager
    def locked(*,blocking):
        assert blocking is False;entered.append(True);yield
    monkeypatch.setattr(gpu_lock,'gpu_execution_lock',locked)
    o._recover_evaluation_accounting()
    after=ledger.report()
    assert entered and not after['pending'] and before['charged']==after['charged']=={'evaluation:ALL':20}
    assert after['uncertain']=={'evaluation:ALL':20}
    size=o.events.path.stat().st_size;o._recover_evaluation_accounting();assert o.events.path.stat().st_size==size


def test_source_migration_replay_keeps_new_operation_and_current_checkpoint(tmp_path):
    o=Orchestrator(tmp_path/'run',tmp_path/'engineering')
    o.state['code_sha256']='old'
    o.commit_event('formal-source-transition:compat',dict(code_sha256='new',source_transition={'to_code_sha256':'new'},operations={},teacher_admission={},teacher_admitted=False))
    o.state['operations']['teacher_admission_D1_A']={'status':'succeeded','result_sha256':'newreceipt'}
    o.state['teacher_admission']['teacher_admission_D1_A']={'passed':True}
    o.commit_event('formal-handoff-adopt:cert',dict(formal_handoff={'path':'cert','sha256':'digest'},learning_checkpoint='origin50'))
    o.state['learning_checkpoint']='new100';o.persist()
    restored=Orchestrator(o.root,o.approval)
    assert restored.state['operations']['teacher_admission_D1_A']['result_sha256']=='newreceipt'
    assert restored.state['teacher_admission']['teacher_admission_D1_A']['passed']
    assert restored.state['learning_checkpoint']=='new100'


def test_latest_saved_worker_checkpoint_exact_operation_and_nonrefundable_ledger(tmp_path):
    from src.tasks.velocity.v10.checkpoint import BudgetLedger
    o=Orchestrator(tmp_path/'run',tmp_path/'engineering')
    o.state.update(num_envs=1024,physical_sha256='physics',operations={'learning_D1_200':{'status':'failed'}})
    o.handoff={'path':'signed','sha256':'credential'}
    out=o.root/'learning'/'D1';out.mkdir(parents=True)
    cp=out/'D1_000100.pt';cp.write_text('saved complete100')
    ledger=BudgetLedger(o.learning_ledger);ledger.charge('D1',100*1024*48,limit=500*1024*48,kind='learning')
    manifest=dict(schema='a2-v10-worker-recovery-v1',checkpoint=dict(path=str(cp),sha256=file_sha(cp)),course='D1',num_envs=1024,
        physical_sha256='physics',code_sha256=o.state['code_sha256'],learning_ledger_path=str(o.learning_ledger),operation_id='learning_D1_200',
        target_round=200,confirmed_samples=100*1024*48,round=100,formal_handoff_sha256='credential')
    path=out/'worker_checkpoint_manifest.json';path.write_text(json.dumps(manifest));cost_before=o.learning_ledger.read_bytes()
    assert o._recover_worker_checkpoint('learning_D1_200','D1',200,out,'trial50')==str(cp)
    assert o.learning_ledger.read_bytes()==cost_before
    manifest['operation_id']='old_operation';path.write_text(json.dumps(manifest))
    assert o._recover_worker_checkpoint('learning_D1_200','D1',200,out,'trial50')=='trial50'
    manifest['operation_id']='learning_D1_200';manifest['checkpoint']['sha256']='0'*64;path.write_text(json.dumps(manifest))
    with pytest.raises(StopTraining,match='checkpoint_changed'):o._recover_worker_checkpoint('learning_D1_200','D1',200,out,'trial50')


def test_preflight_is_readonly_before_any_controller_constructor(tmp_path,monkeypatch):
    from src.tasks.velocity.v10 import orchestration,source_compatibility,handoff
    plan=tmp_path/'plan.json';approval=tmp_path/'approval.json';compat=tmp_path/'compat.json'
    for p in (plan,approval,compat):p.write_text('{}')
    bridge={'sha256':'fixed','to_code_sha256':'new'}
    monkeypatch.setattr(source_compatibility,'validate_compatibility',lambda p:bridge)
    monkeypatch.setattr(orchestration,'validate_engineering',lambda p,s:{'passed':True})
    monkeypatch.setattr(handoff,'validate_plan',lambda p:dict(compatibility=bridge,initial_round=50))
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    checked=orchestration.preflight_formal_handoff(plan,approval,compat)
    assert checked['passed'] and checked['executed'] is False and checked['next_D1_gate']==200
    assert {p:p.read_bytes() for p in tmp_path.iterdir()}==before


def test_controller_migration_rejects_nonidentical_snapshot_before_any_child(tmp_path,monkeypatch):
    from src.tasks.velocity.v10 import source_compatibility,orchestration
    plan=tmp_path/'plan';plan.write_text('{}')
    o=Orchestrator(tmp_path/'run',tmp_path/'engineering',formal_handoff_plan=plan,source_compatibility=tmp_path/'compat',runner=lambda *a,**k:pytest.fail('no physical child allowed'))
    o.state['code_sha256']='approved-old'
    snapshot=tmp_path/'snapshot.json';snapshot.write_text('{}')
    bridge=dict(path=str(tmp_path/'compat'),sha256='bridge',from_code_sha256='approved-old',to_code_sha256='approved-new',document=dict(artifacts=[dict(path=str(snapshot),sha256=file_sha(snapshot),purpose='controller_state',operation_ids=['controller_resume'])]))
    monkeypatch.setattr(source_compatibility,'validate_compatibility',lambda p:bridge)
    monkeypatch.setattr(source_compatibility,'validate_compatible_artifact',lambda *a,**k:{'wrong':'snapshot'})
    monkeypatch.setattr(orchestration,'validate_engineering',lambda *a:{'passed':True})
    with pytest.raises(StopTraining,match='origin_snapshot_mismatch'):o._migrate_controller_source()
    assert o.state['code_sha256']=='approved-old' and not o.events.path.exists()


def test_two_phase_teacher_failure_cannot_finalize_or_launch_training(tmp_path,monkeypatch):
    from src.tasks.velocity.v10 import handoff
    o=Orchestrator(tmp_path/'run',tmp_path/'engineering',formal_handoff_plan=tmp_path/'plan',runner=lambda *a,**k:pytest.fail('no physical child allowed'))
    o._migrate_controller_source=lambda:None;o.prepare=lambda:None;o._recover_evaluation_accounting=lambda:None
    def teachers():raise StopTraining('teacher_capability_gap:CPU-case')
    o.teacher_admission=teachers
    monkeypatch.setattr(handoff,'finalize_handoff',lambda *a,**k:pytest.fail('cannot sign handoff without all A/B'))
    o.worker=lambda *a,**k:pytest.fail('cannot train without all A/B')
    result=o.run()
    assert result['passed'] is False and 'teacher_capability_gap' in result['reason']
    assert not o.state.get('formal_handoff') and o.state['learning_checkpoint'] is None
