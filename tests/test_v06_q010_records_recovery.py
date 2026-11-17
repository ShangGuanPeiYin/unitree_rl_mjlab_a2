"""Q010 current-result persistence through real checkpoint/load with real Adam."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import a2_v06_bootstrap
a2_v06_bootstrap.activate(ROOT)
from src.tasks.velocity.v06 import evaluation,recovery,model as model_module
from src.tasks.velocity.v06 import evaluation_records as records_io
from test_v06_q006 import make_runner,passing


def pending_runner(path,calls,telemetry=None):
    def cases(case):calls.append(case['case_id']);return passing(case)
    runner,*_=make_runner(path,update=100,case_runner=cases,telemetry=telemetry)
    table=evaluation.make_cases('F0',None,'validation',42,tile_id=0)
    runner.pending_boundary=dict(update=100,phase='F0',substage=None,generation=0,
        tasks=[dict(kind='validation',gate_kind='validation',retention_kind=None,tile=0,
                    phase='F0',substage=None,case_table_sha256=table.sha256)],
        results=[],curriculum_decided=False,committed=True)
    return runner


def restore_runner(path,checkpoint,calls):
    def forbidden(case):calls.append('unexpected_re_evaluation');raise AssertionError('cached cases re-evaluated')
    resumed,*_=make_runner(path,model=model_module.V06ActorCritic(),require_resume_evidence=True,
                          case_runner=forbidden)
    resumed.load(checkpoint['path'],checkpoint['sha256'])
    resumed.save_verified('resume_loaded')
    return resumed


def test_sidecar_write_failure_reloads_current_result_without_new_case_calls(tmp_path,monkeypatch):
    calls=[];runner=pending_runner(tmp_path/'first',calls)
    original=records_io.write_sidecar
    def io_error(path,payload):
        original(path,payload)  # Actual atomic sidecar exists, reporting fails before apply.
        raise OSError('injected sidecar persistence failure')
    with monkeypatch.context() as patch:
        patch.setattr(records_io,'write_sidecar',io_error)
        with pytest.raises(OSError):runner.run_gate('validation',round=None)
    assert len(calls)==60 and runner.gate_streak==0 and not runner.gates
    assert runner.evaluation_transaction['stage']=='evaluated'
    with pytest.raises(recovery.RecoveryBlocked):runner._check_resume_evidence(stage='learning')
    with pytest.raises(recovery.RecoveryBlocked):runner._finalize_curriculum(runner.pending_boundary,None)
    checkpoint=runner.save('exception')
    state=recovery.preload_checkpoint(checkpoint['path'],checkpoint['sha256'],mode='resume')
    assert state['ledger']['evaluation_transaction']['result']['records']
    resumed=restore_runner(tmp_path/'resumed',checkpoint,calls)
    resumed._complete_pending_boundary()
    assert len(calls)==60 and resumed.update==100
    assert resumed.learning['adam_steps']==2000
    assert resumed.gate_streak==1 and len(resumed.gates)==1
    assert resumed.evaluation_transaction is None and resumed.pending_boundary is None
    saved=resumed.save_verified('resume_ready')
    again=restore_runner(tmp_path/'again',saved,calls)
    assert again._complete_pending_boundary() is None
    assert len(calls)==60 and again.gate_streak==1 and len(again.gates)==1
    sidecars=list((tmp_path/'resumed'/'evaluation_records').glob('*.json'))
    assert len(sidecars)==1
    payload=json.loads(sidecars[0].read_text())
    assert records_io.restore_result(payload,payload['identity']).evidence_sha256==payload['evidence_sha256']


def test_applied_result_with_pending_marker_is_not_applied_twice(tmp_path):
    calls=[]
    class BrokenTelemetry:
        def gate(self,record):raise OSError('post-application telemetry failure')
    runner=pending_runner(tmp_path/'first',calls,telemetry=BrokenTelemetry())
    with pytest.raises(OSError):runner.run_gate('validation',round=None)
    assert runner.gate_streak==1 and runner.evaluation_transaction['stage']=='applied'
    saved=runner.save('exception')
    resumed=restore_runner(tmp_path/'resume',saved,calls)
    resumed._complete_pending_boundary()
    assert len(calls)==60 and resumed.gate_streak==1 and len(resumed.gates)==1
    assert resumed.evaluation_transaction is None


def test_cached_model_task_or_application_incomplete_refuses_automatic_recovery(tmp_path,monkeypatch):
    calls=[];runner=pending_runner(tmp_path/'first',calls)
    with monkeypatch.context() as patch:
        patch.setattr(records_io,'write_sidecar',lambda *a,**k:(_ for _ in ()).throw(OSError('write failed')))
        with pytest.raises(OSError):runner.run_gate('validation',round=None)
    original=deepcopy(runner.evaluation_transaction)
    for key,value in [('update',101),('phase','D'),('substage','D25'),('generation',1),
                      ('kind','stair_retention'),('model_sha256','0'*64),
                      ('case_table_sha256','0'*64),('source_closure_sha256','0'*64),
                      ('contract_sha256','0'*64)]:
        runner.evaluation_transaction=deepcopy(original)
        runner.evaluation_transaction['identity'][key]=value
        with pytest.raises(recovery.RecoveryBlocked):runner._validate_evaluation_transaction()
    runner.evaluation_transaction=deepcopy(original)
    runner.evaluation_transaction['stage']='application_incomplete'
    saved=runner.save('incomplete_application')
    with pytest.raises(recovery.RecoveryBlocked):restore_runner(tmp_path/'refused',saved,calls)
    assert len(calls)==60

@pytest.mark.parametrize('schema',[None,'unknown-evaluation-v9'])
def test_outer_transaction_schema_is_required(tmp_path,monkeypatch,schema):
    calls=[];runner=pending_runner(tmp_path/'first',calls)
    with monkeypatch.context() as patch:
        patch.setattr(records_io,'write_sidecar',lambda *a,**k:(_ for _ in ()).throw(OSError('write failed')))
        with pytest.raises(OSError):runner.run_gate('validation',round=None)
    if schema is None:runner.evaluation_transaction.pop('schema')
    else:runner.evaluation_transaction['schema']=schema
    with pytest.raises(recovery.RecoveryBlocked):runner._validate_evaluation_transaction()


def test_existing_checkpoint_with_conflicting_model_cannot_prove_application(tmp_path):
    import torch
    calls=[]
    class BrokenTelemetry:
        def gate(self,record):raise OSError('after apply')
    runner=pending_runner(tmp_path/'first',calls,telemetry=BrokenTelemetry())
    runner.pending_boundary=None
    with pytest.raises(OSError):runner.run_gate('validation',round=None)
    state=runner.checkpoint_state()
    key=next(key for key in state['model'] if key.endswith('weight'))
    state['model'][key]=state['model'][key]+1
    target=runner.directory/'checkpoints'/'evaluation_validation_1_0_000100.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,target)
    with pytest.raises(recovery.RecoveryBlocked):runner._finish_direct_evaluation_transaction()
    assert runner.evaluation_transaction is not None


def test_final_test_sidecar_failure_resumes_same_300_cases(tmp_path,monkeypatch):
    calls=[]
    def cases(case):calls.append(case['case_id']);return passing(case)
    runner,*_=make_runner(tmp_path/'first',update=25000,case_runner=cases)
    runner.curriculum.phase='R';runner._sync_phase()
    runner.lifecycle_state='ready_for_final_test';runner.curriculum_complete=True
    with monkeypatch.context() as patch:
        patch.setattr(records_io,'write_sidecar',lambda *a,**k:(_ for _ in ()).throw(OSError('write failed')))
        with pytest.raises(OSError):runner.run(max_updates=25000)
    assert len(calls)==300 and runner.evaluation_transaction['stage']=='evaluated'
    path=runner.directory/'checkpoints'/'exception_025000.pt'
    saved=dict(path=str(path),sha256=recovery.sha256(path))
    resumed=restore_runner(tmp_path/'resumed',saved,calls)
    resumed.run(max_updates=25000)
    assert len(calls)==300 and resumed.final['passed'] is True
    assert resumed.lifecycle_state=='complete' and resumed.evaluation_transaction is None
    assert resumed.learning['adam_steps']==500000
    assert resumed.final['case_records']['records_sha256']==resumed.final['evidence']['records_sha256']


def test_two_retention_sets_resume_without_repeating_either_actual_evaluation(tmp_path,monkeypatch):
    calls=[]
    def cases(case):calls.append(case['case_id']);return passing(case)
    runner,*_=make_runner(tmp_path/'first',update=600,case_runner=cases)
    runner.curriculum.phase='H';runner._sync_phase()
    tasks=[]
    for kind,tile,short in [('flat_retention',0,'flat'),('stair_retention',1,'h00')]:
        table=evaluation.make_cases('H',None,kind,42,tile_id=tile)
        tasks.append(dict(kind='retention',gate_kind=kind,retention_kind=short,tile=tile,
            phase='H',substage=None,case_table_sha256=table.sha256))
    runner.pending_boundary=dict(update=600,phase='H',substage=None,generation=0,
        tasks=tasks,results=[],curriculum_decided=False,committed=True)
    original=records_io.write_sidecar
    def second_fails(path,payload):
        if payload['identity']['kind']=='stair_retention':raise OSError('second set persistence failure')
        return original(path,payload)
    with monkeypatch.context() as patch:
        patch.setattr(records_io,'write_sidecar',second_fails)
        with pytest.raises(OSError):runner._run_pending_tasks(runner.pending_boundary)
    assert len(calls)==30 and len(runner.retention['history'])==1
    saved=runner.save('exception')
    resumed=restore_runner(tmp_path/'resumed',saved,calls)
    resumed._complete_pending_boundary()
    assert len(calls)==30 and len(resumed.retention['history'])==2
    assert {row['kind'] for row in resumed.retention['history']}=={'flat_retention','stair_retention'}
    assert resumed.learning['adam_steps']==12000 and resumed.evaluation_transaction is None


def test_applied_final_is_read_only_complete_evidence_and_refuses_resume(tmp_path,monkeypatch):
    calls=[]
    def cases(case):calls.append(case['case_id']);return passing(case)
    runner,*_=make_runner(tmp_path/'first',update=25000,case_runner=cases)
    runner.curriculum.phase='R';runner._sync_phase()
    runner.lifecycle_state='ready_for_final_test';runner.curriculum_complete=True
    with monkeypatch.context() as patch:
        patch.setattr(runner,'_finish_direct_evaluation_transaction',lambda:(_ for _ in ()).throw(OSError('after applied')))
        with pytest.raises(OSError):runner.run(max_updates=25000)
    assert runner.evaluation_transaction['stage']=='applied' and runner.lifecycle_state=='complete'
    path=runner.directory/'checkpoints'/'exception_025000.pt'
    saved=dict(path=str(path),sha256=recovery.sha256(path))
    state=recovery.preload_checkpoint(path,saved['sha256'],mode='train')
    with pytest.raises(recovery.RecoveryBlocked):restore_runner(tmp_path/'refused',saved,calls)
    transaction=state['ledger']['evaluation_transaction']
    assert records_io.verify_sidecar(transaction['case_records'],transaction['result'])
    assert records_io.restore_result(transaction['result'],transaction['identity']).gate['gate_pass']
    assert json.loads((runner.directory/'final_test'/'summary.json').read_text())['passed']
    assert json.loads((runner.directory/'final_test'/'complete.json').read_text())['state']=='complete'
    assert len(calls)==300 and state['learning']['optimizer_step']==500000


def test_applied_checkpoint_refuses_corrupted_bound_sidecar(tmp_path):
    calls=[]
    class BrokenTelemetry:
        def gate(self,record):raise OSError('after apply')
    runner=pending_runner(tmp_path/'first',calls,telemetry=BrokenTelemetry())
    with pytest.raises(OSError):runner.run_gate('validation',round=None)
    saved=runner.save('exception')
    binding=runner.evaluation_transaction['case_records']
    Path(binding['path']).write_text('{}\n')
    with pytest.raises(recovery.RecoveryBlocked):restore_runner(tmp_path/'refused',saved,calls)
    assert len(calls)==60
