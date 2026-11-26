import json
from pathlib import Path
import pytest
from src.tasks.velocity.v10 import spec
from src.tasks.velocity.v10.orchestration import (
    StopTraining, RetentionState, execution_plan, is_retention_failure,
    code_fingerprint, validate_engineering, Orchestrator,
)


def test_exact_schedule_and_budgets():
    a=execution_plan(1024);b=execution_plan(512)
    assert a['gates']=={'D1':[200,400,500],'D2':[400,800,1000],'D3':[600,1200,1500],'PPO':[250,500,750,1000]}
    assert b['p0_rounds']==200 and a['p0_rounds']==100
    assert b['gates']['PPO']==[500,1000,1500,2000]
    assert sum(a['learning_budgets'].values())==196608000


def test_retention_counter_and_nonrefundable_rollback_limit():
    s=RetentionState()
    assert not s.record(False)
    assert s.record(False) and s.rollbacks==1
    assert not s.record(False)
    assert s.record(False) and s.rollbacks==2
    assert not s.record(False)
    with pytest.raises(StopTraining,match='third_rollback'):s.record(False)
    # A retention pass breaks the consecutive failure streak, without refunding failures/rollbacks.
    s=RetentionState();s.record(False);s.record(True)
    assert s.consecutive_failures==0 and s.failures==1 and s.rollbacks==0


def test_relative_failure_is_separate_from_absolute():
    assert not is_retention_failure({'groups':{'g':{'reasons':['absolute_success']}}})
    for reason in ('relative_success','relative_metric:first_meter:rmse','pair_coverage:stand_0'):
        assert is_retention_failure({'groups':{'g':{'reasons':[reason]}}})


def test_admission_is_not_default_green(tmp_path):
    path=tmp_path/'engineering.json'
    with pytest.raises(StopTraining):validate_engineering(path,'abc')
    path.write_text(json.dumps(dict(passed=True,sds_sha256=spec.SDS_SHA256,code_sha256='abc')))
    with pytest.raises(StopTraining):validate_engineering(path,'abc')
    report=tmp_path/'review.md';report.write_text('independent checks')
    import hashlib
    row=dict(passed=True,sds_sha256=spec.SDS_SHA256,code_sha256='abc',independent_review=dict(passed=True,report_path=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    path.write_text(json.dumps(row));assert validate_engineering(path,'abc')['passed']
    with pytest.raises(StopTraining):validate_engineering(path,'different')


def test_failed_existing_final_is_never_reopened(tmp_path):
    o=Orchestrator(tmp_path/'run',tmp_path/'approval.json')
    o.state['finals']['DAGGER_FINAL']={'status':'failed','candidate_sha256':'abc'}
    with pytest.raises(StopTraining,match='already_opened'):o.open_final('DAGGER_FINAL','abc')


def test_fingerprint_deterministic_and_scope(tmp_path):
    (tmp_path/'src/tasks/velocity/v10').mkdir(parents=True)
    (tmp_path/'src/tasks/velocity/v10/a.py').write_text('a')
    a=code_fingerprint(tmp_path)
    (tmp_path/'unrelated.txt').write_text('not recipe')
    assert code_fingerprint(tmp_path)==a
    (tmp_path/'src/tasks/velocity/v10/a.py').write_text('b')
    assert code_fingerprint(tmp_path)!=a


class FakeProtocol(Orchestrator):
    """CPU-only child receipts test control decisions; no fake physical evidence."""
    def __init__(self,path,*,teacher_failure=False,dagger_failure=False,dagger_final_failure=False,ppo_retention_failures=()):
        super().__init__(path,path/'approval.json')
        self.teacher_failure=teacher_failure;self.dagger_failure=dagger_failure
        self.dagger_final_failure=dagger_final_failure;self.ppo_retention_failures=set(ppo_retention_failures)
        self.calls=[];self.learning=[]
        self.state.update(num_envs=1024,physical_sha256='physical',evaluation_max_samples=1685390840,schedule=execution_plan(1024))
    def prepare(self):self.state['frozen_sha256']='frozen';self.persist()
    def worker(self,key,course,target,resume=None,acceptance=None):
        self.calls.append(('worker',key,course,target,resume))
        checkpoint=self.root/'weights'/(key+'.pt');checkpoint.parent.mkdir(exist_ok=True);checkpoint.write_text(key)
        self.state['learning_checkpoint']=str(checkpoint);self.persist()
        self.learning.append(dict(key=key,course=course,target=target,resume=resume))
        return str(checkpoint)
    def evaluate(self,key,course,suite,model_path,reference):
        from src.tasks.velocity.v10.orchestration import file_sha,atomic_json
        self.calls.append(('eval',key,course,suite,reference,model_path))
        final=suite.endswith('FINAL');n=100 if final else 20
        fail=(key.startswith('teacher_admission') and self.teacher_failure or
              reference=='candidate' and course.startswith('D') and self.dagger_failure or
              reference=='candidate' and suite=='DAGGER_FINAL' and self.dagger_final_failure)
        retention=(reference=='candidate' and any(key.startswith(f'gate_PPO_{t}_') for t in self.ppo_retention_failures))
        rows=[dict(case_sha256=str(i),passed=not fail,windows={'walk_0':dict(rmse=.2 if retention else .05,pitch_rms=1.)}) for i in range(n)]
        result=dict(sds_sha256=spec.SDS_SHA256,suite_sha256=course+suite,physical_sha256='physical',replicas=n,
                    model_sha256='teachers' if reference=='teachers' else file_sha(model_path),groups={'g':rows})
        out=self.root/'evaluation'/key/'result.json';atomic_json(out,result)
        return result
    def _launch(self,key,command,result_path):
        from src.tasks.velocity.v10.orchestration import atomic_json
        self.calls.append(('export',key));actor=Path(result_path).parent/'student_actor.pt';actor.parent.mkdir(parents=True,exist_ok=True);actor.write_text('actor')
        from src.tasks.velocity.v10.orchestration import file_sha
        row=dict(sds_sha256=spec.SDS_SHA256,checkpoint_sha256=file_sha(command[command.index('--checkpoint')+1]),actor_sha256=file_sha(actor))
        atomic_json(result_path,row)
        return row


def test_all_gates_pass_then_full_protocol_exports(tmp_path):
    o=FakeProtocol(tmp_path/'run');result=o.run()
    assert result['passed'] and o.state['status']=='complete'
    workers=[x for x in o.calls if x[0]=='worker']
    assert [(x[2],x[3]) for x in workers]==[('D1',0),('D1',200),('D2',0),('D2',400),('D3',0),('D3',600),('P0',0),('P0',100),('P1',100),('P1',250)]
    assert len(o.state['teacher_admission'])==6
    for key,row in o.state['gates'].items():
        assert row['passed'] and set(row['suites'])=={'A','B'}
    assert set(o.state['certificates'])=={'V10_DAGGER_ACCEPTED','V10_DAGGER_PPO_ACCEPTED'}
    assert o.state['finals']['DAGGER_FINAL']['status']=='passed'


def test_teacher_failure_never_opens_learning(tmp_path):
    o=FakeProtocol(tmp_path/'run',teacher_failure=True);result=o.run()
    assert not result['passed'] and 'teacher_capability_gap' in result['reason']
    assert not o.learning and len(o.state['teacher_admission'])==1


def test_course_failure_stops_at_own_cap_without_ppo(tmp_path):
    o=FakeProtocol(tmp_path/'run',dagger_failure=True);result=o.run()
    assert not result['passed'] and 'course_budget_exhausted:D1' in result['reason']
    assert [x['target'] for x in o.learning]==[0,200,400,500]
    assert not o.state['certificates']


def test_dagger_final_failure_cannot_start_ppo_or_new_final(tmp_path):
    o=FakeProtocol(tmp_path/'run',dagger_final_failure=True);result=o.run()
    assert not result['passed'] and 'independent_final_failed:DAGGER_FINAL' in result['reason']
    assert not any(x['course'].startswith('P') for x in o.learning)
    before=sum(x[0]=='eval' and x[3]=='DAGGER_FINAL' for x in o.calls)
    with pytest.raises(StopTraining,match='already_opened'):
        o.final('DAGGER_FINAL','D3',o.state['learning_checkpoint'],{'teachers':None})
    assert sum(x[0]=='eval' and x[3]=='DAGGER_FINAL' for x in o.calls)==before


def test_retention_rollback_resumes_p0_without_refunding_schedule(tmp_path):
    o=FakeProtocol(tmp_path/'run',ppo_retention_failures=(250,500));result=o.run()
    assert result['passed']
    row=next(r for r in o.learning if r['key']=='learning_PPO_750')
    assert row['resume']==o.state['p0_checkpoint']
    assert [r['target'] for r in o.learning if r['key'].startswith('learning_PPO_')]==[250,500,750]
    assert o.state['rollback_history'][0]['refund']==0 and o.state['retention']['rollbacks']==1
    assert o.state['accepted_dagger']


def test_q3_replay_cannot_undo_final_or_refund_rollback_with_old_snapshot(tmp_path):
    import copy
    o=FakeProtocol(tmp_path/'run',ppo_retention_failures=(250,500))
    old=copy.deepcopy(o.state)
    assert o.run()['passed']
    # Restoring an old controller cache cannot erase journaled final use or rollback.
    o.state_path.write_text(json.dumps(old))
    restored=Orchestrator(o.root,o.approval)
    assert restored.state['retention']['rollbacks']==1
    assert restored.state['ppo_checked']==[250,500,750]
    assert restored.state['finals']['PPO_FINAL']['status']=='passed'
    assert restored.state['accepted_dagger'] and restored.state['accepted_ppo']
    with pytest.raises(StopTraining,match='already_opened'):
        restored.open_final('PPO_FINAL',restored.state['finals']['PPO_FINAL']['candidate_sha256'])


def test_q3_checkpoint_immutable_and_tamper_is_rejected(tmp_path):
    from src.tasks.velocity.v10.orchestration import file_sha
    o=FakeProtocol(tmp_path/'run');o.prepare()
    checkpoint=o.worker('candidate','D3',600)
    digest=file_sha(checkpoint)
    result=o.final('DAGGER_FINAL','D3',checkpoint,{'teachers':None})
    assert file_sha(checkpoint)==digest
    before=len(o.events.read())
    o.certificate('V10_DAGGER_ACCEPTED',checkpoint,result)
    assert len(o.events.read())==before
    Path(checkpoint).write_text('wrong candidate')
    with pytest.raises(StopTraining,match='reference_changed'):
        Orchestrator(o.root,o.approval)


def test_q3_crash_after_event_before_certificate_recovers_idempotently(tmp_path,monkeypatch):
    from src.tasks.velocity.v10 import orchestration as m
    o=FakeProtocol(tmp_path/'run');o.prepare();checkpoint=o.worker('candidate','D3',600)
    original=m.atomic_json
    def crash(path,value):
        if Path(path).name=='V10_DAGGER_ACCEPTED.json':raise OSError('simulated power loss after durable event')
        return original(path,value)
    monkeypatch.setattr(m,'atomic_json',crash)
    with pytest.raises(OSError):o.final('DAGGER_FINAL','D3',checkpoint,{'teachers':None})
    assert not (o.root/'V10_DAGGER_ACCEPTED.json').exists()
    monkeypatch.setattr(m,'atomic_json',original)
    restored=Orchestrator(o.root,o.approval)
    assert restored.state['accepted_dagger']==checkpoint
    assert (o.root/'V10_DAGGER_ACCEPTED.json').exists()
    count=len(restored.events.read())
    again=Orchestrator(o.root,o.approval)
    assert len(again.events.read())==count
    assert sum(r['id']=='acceptance:V10_DAGGER_ACCEPTED' for r in again.events.read())==1


def test_partial_confirmed_recovery_allows_only_first_complete_update_overshoot(tmp_path):
    from src.tasks.velocity.v10.orchestration import file_sha
    o=Orchestrator(tmp_path/'run',tmp_path/'approval.json')
    o.state.update(num_envs=1024,physical_sha256='p')
    checkpoint=tmp_path/'model.pt';checkpoint.write_text('checkpoint')
    target=200;threshold=target*1024*48
    receipt=dict(gate_reached=True,course='D1',round=target,physical_sha256='p',checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
                 consumed=threshold+200,budget={'charged':{'learning:D1':threshold+1224}})
    o._launch=lambda *a,**k:receipt
    assert o.worker('recover','D1',target)==str(checkpoint)
    receipt['consumed']=threshold+1024*48
    with pytest.raises(StopTraining,match='sample_mismatch'):o.worker('recover','D1',target)
    receipt['consumed']=threshold;receipt['budget']['charged']['learning:D1']=spec.BUDGETS['D1']+1
    with pytest.raises(StopTraining,match='charged_cap'):o.worker('recover','D1',target)


def test_resource_receipt_cannot_default_green_or_adopt_unadmitted(tmp_path):
    from src.tasks.velocity.v10.orchestration import file_sha
    o=Orchestrator(tmp_path/'run',tmp_path/'approval.json')
    with pytest.raises(StopTraining,match='resource_receipt'):o._validate_receipt('gate_D1_200_A_candidate',{})
    report=tmp_path/'resources.json';report.write_text(json.dumps({'passed':True}))
    receipt=dict(resource_admitted=True,resource_report_path=str(report.resolve()),resource_report_sha256=file_sha(report))
    o._validate_receipt('gate_D1_200_A_candidate',receipt)
    report.write_text(json.dumps({'passed':False}))
    with pytest.raises(StopTraining,match='resource_receipt'):o._validate_receipt('gate_D1_200_A_candidate',receipt)


def test_event_unique_identity_and_hash_chain_reject_cross_suite_merge(tmp_path):
    from src.tasks.velocity.v10.orchestration import EventJournal
    ledger=EventJournal(tmp_path/'events.jsonl')
    ledger.append('acceptance:one',{'candidate':'a','suite':'A'})
    ledger.append('acceptance:one',{'candidate':'a','suite':'A'})
    assert len(ledger.read())==1
    with pytest.raises(StopTraining,match='identity_mismatch'):ledger.append('acceptance:one',{'candidate':'b','suite':'B'})
    row=json.loads(ledger.path.read_text());row['payload']['suite']='B';ledger.path.write_text(json.dumps(row)+'\n')
    with pytest.raises(StopTraining,match='chain_damaged'):ledger.read()


def test_passed_ppo_gate_recovery_cannot_train_next_gate(tmp_path):
    o=FakeProtocol(tmp_path/'run');o.prepare();dagger=o._train_dagger()
    def crash_before_final(*a,**k):raise KeyboardInterrupt('parent stopped after gate event')
    o.final=crash_before_final
    with pytest.raises(KeyboardInterrupt):o._train_ppo(dagger)
    assert o.state['last_passed_ppo'] and o.state['ppo_checked']==[250]
    restored=FakeProtocol(o.root);restored.prepare()
    result=restored._train_ppo(restored.state['accepted_dagger'])
    assert result==restored.state['accepted_ppo']
    assert not restored.learning
    assert restored.state['ppo_checked']==[250]


def test_open_final_completed_candidate_recovery_does_not_train(tmp_path):
    from src.tasks.velocity.v10.orchestration import file_sha
    o=FakeProtocol(tmp_path/'run');o.prepare();dagger=o._train_dagger()
    checkpoint=o.worker('ppo_candidate','P1',250,dagger)
    o.commit_event('ppo-gate:250',dict(last_passed_ppo=checkpoint,ppo_resume=checkpoint,ppo_checked=[250]),{checkpoint:file_sha(checkpoint)})
    o.open_final('PPO_FINAL',file_sha(checkpoint),checkpoint)
    o.evaluate('final_PPO_FINAL_candidate','PPO','PPO_FINAL',checkpoint,'candidate')
    # Simulate the immutable completed receipt surviving a parent hard crash.
    o.state['operations']['final_PPO_FINAL_candidate']={'status':'succeeded'};o.persist()
    restored=FakeProtocol(o.root);restored.prepare()
    assert restored._train_ppo(restored.state['accepted_dagger'])==checkpoint
    assert not restored.learning
    assert restored.state['finals']['PPO_FINAL']['status']=='passed'


def test_first_bad_teacher_group_partial_receipt_stops_before_suite_grade(tmp_path):
    o=FakeProtocol(tmp_path/'run')
    evaluate=o.evaluate
    def partial(*a,**k):
        row=evaluate(*a,**k)
        row.update(aborted_for_teacher_failure=True,complete=False,tested_case_count=20,planned_case_count=3080)
        return row
    o.evaluate=partial
    result=o.run()
    assert not result['passed'] and 'teacher_capability_gap' in result['reason']
    assert not o.learning and len(o.state['teacher_admission'])==1
