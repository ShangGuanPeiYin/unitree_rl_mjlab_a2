"""CPU-only global probe accounting: historical facts never grant future physics."""
import hashlib
import json
from pathlib import Path
import pytest
from src.tasks.velocity.v10 import spec
from src.tasks.velocity.v10.probe_budget import GlobalProbeBudget, ProbeBudgetError, migrate_legacy_ledgers


def legacy(path,n=4):
    rows=[dict(seq=1,op='reserve',id='physical-step',key='probe:D3',samples=n),
          dict(seq=2,op='executed',id='physical-step',key='probe:D3',samples=n),
          dict(seq=3,op='update',id='update0',key='probe:D3',samples=n)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));return path


def authorization(tmp_path,ledger,cap,prior=0):
    from src.tasks.velocity.v10.probe_budget import RULINGS_PATH
    path=tmp_path/'authorization.json'
    row=dict(schema='a2-v10-global-probe-budget-authorization-v1',approved=True,authorization_id='user-test-approval',
             scope='probe_only',sds_sha256=spec.SDS_SHA256,rulings_sha256=hashlib.sha256(RULINGS_PATH.read_bytes()).hexdigest(),
             global_ledger_path=str(ledger.resolve()),max_probe_samples=cap,prior_charged_samples=prior,
             approval_source={'kind':'user','reference':'CPU test fixture, no actual physics'})
    path.write_text(json.dumps(row));return path


def test_migration_idempotent_and_duplicate_copy_not_double_counted(tmp_path):
    source=legacy(tmp_path/'old.jsonl');copy=tmp_path/'copy.jsonl';copy.write_bytes(source.read_bytes())
    path=tmp_path/'global.jsonl'
    a=migrate_legacy_ledgers(path,[source],expected_totals={'charged':4,'executed':4,'uncertain':0,'updated':4})
    b=migrate_legacy_ledgers(path,[source,copy])
    assert a['global_totals']==b['global_totals']==dict(charged=4,executed=4,uncertain=0,updated=4)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ProbeBudgetError):migrate_legacy_ledgers(path,[source])


def test_no_authorization_means_no_sequence_or_step(tmp_path):
    b=GlobalProbeBudget(tmp_path/'global.jsonl')
    with pytest.raises(ProbeBudgetError,match='authorization'):b.begin_sequence('new-run',required_samples=4)
    c=GlobalProbeBudget(b.path,scope_id='new-run:1024:dagger')
    with pytest.raises(ProbeBudgetError,match='authorization'):c.reserve('step','D3',1,kind='probe',limit=4)
    assert b.global_report()['global_totals']['charged']==0


def test_new_run_id_cannot_reset_cumulative_cap(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=6)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('run1',required_samples=4)
    first=GlobalProbeBudget(path,auth,scope_id='run1:1024:dagger');first.assert_authorized('run1');first.acquire_stack_capacity('run1',4)
    assert first.reserve('step','D3',4,limit=4,kind='probe');first.mark_executed('step');first.record_update('update','D3',4,kind='probe')
    first.release_stack_capacity();parent.finish_sequence('run1',passed=False,reason='engineering test')
    with pytest.raises(ProbeBudgetError,match='cap|remaining'):parent.begin_sequence('run2',required_samples=4)
    assert parent.global_report()['global_totals']==dict(charged=4,executed=4,uncertain=0,updated=4)


def test_single_inflight_crash_idempotent_and_uncertain_never_refunded(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=8)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('run',required_samples=8)
    b=GlobalProbeBudget(path,auth,scope_id='run:512:dagger');b.assert_authorized('run');b.acquire_stack_capacity('run',8)
    assert b.reserve('step1','D3',2,limit=8,kind='probe')
    assert not b.reserve('step1','D3',2,limit=8,kind='probe')
    with pytest.raises(ProbeBudgetError,match='in.flight'):b.reserve('step2','D3',2,limit=8,kind='probe')
    b.reconcile_uncertain();b.reconcile_uncertain()
    assert not b.reserve('step1','D3',2,limit=8,kind='probe')
    with pytest.raises(ProbeBudgetError,match='uncertain'):b.mark_executed('step1')
    assert b.reserve('step2','D3',2,limit=8,kind='probe');b.mark_executed('step2');assert not b.mark_executed('step2')
    assert b.record_update('update','D3',2,kind='probe');assert not b.record_update('update','D3',2,kind='probe')
    assert b.global_report()['global_totals']==dict(charged=4,executed=2,uncertain=2,updated=2)
    with pytest.raises(ProbeBudgetError):b.record_update('bad','D3',4,kind='probe')


def test_local_stack_cap_and_non_probe_charges_are_rejected(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=20)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('r',required_samples=20)
    b=GlobalProbeBudget(path,auth,scope_id='r:512:ppo');b.assert_authorized('r');b.acquire_stack_capacity('r',5 if 'ppo' in b.scope_id else 4)
    b.reserve('fill','D3',3,limit=5,kind='probe');b.mark_executed('fill')
    with pytest.raises(ProbeBudgetError,match='scope|stack'):b.reserve('more','PPO',3,limit=5,kind='probe')
    with pytest.raises(ProbeBudgetError):b.reserve('learn','D1',1,limit=20,kind='learning')


def test_streaming_file_sha_never_reads_entire_file(tmp_path,monkeypatch):
    from src.tasks.velocity.v10.orchestration import file_sha
    p=tmp_path/'large';p.write_bytes(b'z'*(9*1024*1024+17))
    expected=hashlib.sha256(p.read_bytes()).hexdigest()
    def no_read_bytes(*a,**k):raise AssertionError('whole-file read forbidden')
    monkeypatch.setattr(Path,'read_bytes',no_read_bytes)
    assert file_sha(p)==expected


@pytest.mark.parametrize('stacks_pass',[True,False])
def test_real_selector_receipt_is_consumed_by_real_controller(tmp_path,monkeypatch,stacks_pass):
    """Actual selector + actual controller validation; only GPU child is a CPU receipt stub.

    All synthetic costs/approvals live under tmp_path. This test does not certify
    physics or resources and never touches the real project global ledger.
    """
    from types import SimpleNamespace
    from src.tasks.velocity.v10 import resources
    from src.tasks.velocity.v10.identity import code_fingerprint
    from src.tasks.velocity.v10.orchestration import Orchestrator,StopTraining,file_sha
    from src.tasks.velocity.v10.probe_budget import FULL_SEQUENCE_DEMAND
    global_path=tmp_path/'global.jsonl';auth=authorization(tmp_path,global_path,FULL_SEQUENCE_DEMAND)
    sequence='receipt-contract-test';calls=[]
    def gpu_child_stub(command,**kwargs):
        # Exercise actual selector CLI forwarding, not a hand-crafted selection.json.
        assert command[2]=='_probe-stack'
        argument=lambda name:command[command.index(name)+1]
        n=int(argument('--num-envs'));stack=argument('--stack');out=Path(argument('--output-dir'));out.mkdir()
        assert Path(argument('--global-probe-ledger'))==global_path
        assert argument('--probe-sequence-id')==sequence
        ledger=GlobalProbeBudget(global_path,argument('--probe-budget-authorization'),scope_id=f'{sequence}:{n}:{stack}')
        grant=ledger.assert_authorized(sequence)
        ledger.acquire_stack_capacity(sequence,26*n*48)
        # Synthetic CPU transaction exercises journal wiring; no physical action occurs.
        course='D3' if stack=='dagger' else 'PPO'
        ledger.reserve('cpu-fixture-step',course,n*48,limit=26*n*48,kind='probe')
        ledger.mark_executed('cpu-fixture-step');ledger.record_update('cpu-fixture-update',course,n*48,kind='probe')
        row=dict(passed=stacks_pass,num_envs=n,stack=stack,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),
                 physical_sha256='cpu-receipt-fixture-not-physics',restore_checked=True,replay_counts={r:32768 for r in spec.ROLES},
                 gpu_total_peak_mib=1000,gpu_task_peak_mib=1000,host_available_min_bytes=8*1024**3,warmup_rounds=5,measurement_rounds=20,
                 global_probe_ledger=str(global_path.resolve()),probe_sequence_id=sequence,probe_budget_authorization_sha256=grant['sha256'],
                 global_probe_totals=ledger.global_report()['global_totals'],cost=ledger.report())
        (out/'result.json').write_text(json.dumps(row));calls.append((n,stack));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(resources.subprocess,'run',gpu_child_stub)
    output=tmp_path/'selector'
    selection=resources.run_probes(output,device='cuda:0',global_probe_ledger=global_path,probe_budget_authorization=auth,probe_sequence_id=sequence)
    assert selection==json.loads((output/'selection.json').read_text())
    report=tmp_path/'review.md';report.write_text('CPU contract test fixture, not engineering approval')
    approval=tmp_path/'engineering.json'
    approval.write_text(json.dumps(dict(passed=True,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),
        independent_review=dict(passed=True,report_path=str(report),report_sha256=file_sha(report)))))
    controller=Orchestrator(tmp_path/'controller',approval,resource_selection=output/'selection.json',
        global_probe_ledger=global_path,probe_budget_authorization=auth,probe_sequence_id=sequence)
    grant=GlobalProbeBudget(global_path,auth).validate_authorization()
    if stacks_pass:
        controller.admit_resource_selection(selection,grant,sequence)
        assert controller.state['num_envs']==1024
        assert calls==[(1024,'dagger'),(1024,'ppo')]
    else:
        with pytest.raises(StopTraining,match='resource_selection_invalid'):
            controller.admit_resource_selection(selection,grant,sequence)
        assert calls==[(1024,'dagger'),(512,'dagger')]
    global_report=GlobalProbeBudget(global_path).global_report()
    assert global_report['active_sequence'] is None
    assert global_report['global_totals']==selection['global_probe_totals']
    assert global_report['global_totals']['charged']==sum(n*48 for n,_ in calls)


def test_real_selector_missing_budget_authorization_never_calls_child(tmp_path,monkeypatch):
    from src.tasks.velocity.v10 import resources
    def forbidden(*a,**k):raise AssertionError('must not start a child without explicit authorization')
    monkeypatch.setattr(resources.subprocess,'run',forbidden)
    with pytest.raises(ProbeBudgetError,match='authorization'):
        resources.run_probes(tmp_path/'probe',global_probe_ledger=tmp_path/'global.jsonl',probe_sequence_id='new-unapproved-run')


def test_explicit_denial_wrong_baseline_and_wrong_ledger_are_not_authorization(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=8)
    original=json.loads(auth.read_text())
    for field,value in [('approved',False),('scope','learning'),('global_ledger_path',str(tmp_path/'new-run.jsonl'))]:
        row=dict(original);row[field]=value;auth.write_text(json.dumps(row))
        with pytest.raises(ProbeBudgetError,match='authorization'):GlobalProbeBudget(path,auth).begin_sequence('r',required_samples=4)
    row=dict(original);row['prior_charged_samples']=4;auth.write_text(json.dumps(row))
    with pytest.raises(ProbeBudgetError,match='baseline'):GlobalProbeBudget(path,auth).begin_sequence('r',required_samples=4)
    assert GlobalProbeBudget(path).global_report()['global_totals']['charged']==0


def test_global_step_cap_shared_between_distinct_scopes(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=5)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('r',required_samples=5)
    first=GlobalProbeBudget(path,auth,scope_id='r:1024:dagger');first.assert_authorized('r');first.acquire_stack_capacity('r',3)
    first.reserve('one','D3',3,limit=20,kind='probe');first.mark_executed('one')
    first.release_stack_capacity()
    second=GlobalProbeBudget(path,auth,scope_id='r:1024:ppo');second.assert_authorized('r')
    with pytest.raises(ProbeBudgetError,match='global probe.*cap'):second.acquire_stack_capacity('r',3)
    assert parent.global_report()['global_totals']['charged']==3
    second.acquire_stack_capacity('r',2)
    second.reserve('two','PPO',2,limit=20,kind='probe');second.mark_executed('two')
    assert parent.global_report()['global_totals']['charged']==5


def test_repeated_migration_after_new_run_cost_does_not_refund_or_double_count(tmp_path):
    path=tmp_path/'global.jsonl';source=legacy(tmp_path/'legacy.jsonl',n=4)
    migrated=migrate_legacy_ledgers(path,[source],expected_totals=dict(charged=4,executed=4,uncertain=0,updated=4))
    auth=authorization(tmp_path,path,cap=8,prior=4)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('r',required_samples=4)
    b=GlobalProbeBudget(path,auth,scope_id='r:512:dagger');b.assert_authorized('r');b.acquire_stack_capacity('r',5 if 'ppo' in b.scope_id else 4)
    b.reserve('new','D3',4,kind='probe',limit=4);b.mark_executed('new');b.record_update('update','D3',4,kind='probe')
    again=migrate_legacy_ledgers(path,[source],expected_totals=dict(charged=4,executed=4,uncertain=0,updated=4))
    assert again['global_totals']==dict(charged=8,executed=8,uncertain=0,updated=8)
    assert again['migration_totals']==migrated['migration_totals']


@pytest.mark.parametrize('ppo_passes',[True,False])
def test_actual_A_cap_history_allows_1024_pair_but_no_unfunded_512(tmp_path,monkeypatch,ppo_passes):
    """Actual A cumulative cap; real selector/journal with CPU-only child receipts."""
    from types import SimpleNamespace
    from src.tasks.velocity.v10 import resources
    from src.tasks.velocity.v10.identity import code_fingerprint
    path=tmp_path/'global.jsonl';source=legacy(tmp_path/'old.jsonl',n=737280)
    rows=[json.loads(line) for line in source.read_text().splitlines()];rows[-1]['samples']=663552
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    migrate_legacy_ledgers(path,[source])
    auth=authorization(tmp_path,path,3833856,prior=737280)
    sequence='A-current-cap';calls=[]
    def cpu_stack_receipt(command,**kwargs):
        arg=lambda key:command[command.index(key)+1]
        n=int(arg('--num-envs'));stack=arg('--stack');calls.append((n,stack))
        assert n==1024,'512 child must not start when its whole stack is unfunded'
        ledger=GlobalProbeBudget(path,auth,scope_id=f'{sequence}:{n}:{stack}');grant=ledger.assert_authorized(sequence)
        ledger.acquire_stack_capacity(sequence,26*n*48)
        # Aggregate synthetic full-stack fixture cost, not physical execution evidence.
        course='D3' if stack=='dagger' else 'PPO';cost=26*n*48
        ledger.reserve('CPU-fixture-full-stack',course,cost,limit=cost,kind='probe');ledger.mark_executed('CPU-fixture-full-stack')
        ledger.record_update('CPU-fixture-updates',course,25*n*48,kind='probe')
        row=dict(passed=stack=='dagger' or ppo_passes,num_envs=n,stack=stack,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),
                 global_probe_ledger=str(path.resolve()),probe_sequence_id=sequence,probe_budget_authorization_sha256=grant['sha256'])
        out=Path(arg('--output-dir'));out.mkdir();(out/'result.json').write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(resources.subprocess,'run',cpu_stack_receipt)
    output=tmp_path/'probe'
    result=resources.run_probes(output,global_probe_ledger=path,probe_budget_authorization=auth,probe_sequence_id=sequence)
    assert calls==[(1024,'dagger'),(1024,'ppo')]
    assert result==json.loads((output/'selection.json').read_text())
    assert result['global_probe_totals']['charged']==3293184
    assert GlobalProbeBudget(path).global_report()['active_sequence'] is None
    if ppo_passes:
        assert result['passed'] and result['num_envs']==1024
    else:
        assert result['passed'] is False and result['num_envs'] is None
        assert result['budget_shortfall']==dict(required=638976,remaining=540672,shortfall=98304)
        assert result['blocked_stack']==dict(num_envs=512,stack='dagger')


def test_first_stack_shortfall_publishes_failed_selection_without_child(tmp_path,monkeypatch):
    from src.tasks.velocity.v10 import resources
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=1)
    def forbidden(*a,**k):raise AssertionError('unfunded first stack must not start')
    monkeypatch.setattr(resources.subprocess,'run',forbidden)
    out=tmp_path/'probe'
    result=resources.run_probes(out,global_probe_ledger=path,probe_budget_authorization=auth,probe_sequence_id='too-small')
    assert not result['passed'] and result['budget_shortfall']==dict(required=1277952,remaining=1,shortfall=1277951)
    assert (out/'selection.json').exists()
    assert GlobalProbeBudget(path).global_report()['global_totals']['charged']==0


def test_stack_capacity_parent_child_reentry_exclusive_and_uncertain_not_refunded(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=10)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('seq',required_samples=8)
    scope='seq:1024:dagger'
    first=parent.acquire_stack_capacity('seq',8,scope_id=scope)
    assert parent.global_report()['global_totals']['charged']==0  # capacity is not a debit
    child=GlobalProbeBudget(path,auth,scope_id=scope);child.assert_authorized('seq',required_samples=8)
    assert child.acquire_stack_capacity('seq',8)==first
    assert child.acquire_stack_capacity('seq',8)==first
    with pytest.raises(ProbeBudgetError,match='exclusive capacity'):
        parent.acquire_stack_capacity('seq',2,scope_id='seq:1024:ppo')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert sum(r['op']=='stack_capacity_acquire' for r in rows)==1
    child.reserve('executed','D3',2,limit=8,kind='probe');child.mark_executed('executed')
    child.reserve('unknown','D3',2,limit=8,kind='probe')
    with pytest.raises(ProbeBudgetError,match='in.flight'):parent.release_stack_capacity(scope)
    child.reconcile_uncertain()
    # A resumed parent/child still shares the same original capacity, not another 8.
    resumed=GlobalProbeBudget(path,auth,scope_id=scope);resumed.assert_authorized('seq',required_samples=8)
    assert resumed.acquire_stack_capacity('seq',8)==first
    assert parent.release_stack_capacity(scope,reason='early stop')==4
    assert parent.release_stack_capacity(scope,reason='duplicate close') is False
    assert parent.global_report()['global_totals']==dict(charged=4,executed=2,uncertain=2,updated=0)
    with pytest.raises(ProbeBudgetError,match='remaining cap'):
        parent.acquire_stack_capacity('seq',7,scope_id='seq:512:dagger')
    lease=parent.acquire_stack_capacity('seq',6,scope_id='seq:512:dagger')
    assert lease['required_samples']==6
    with pytest.raises(ProbeBudgetError,match='capacity lease'):
        child.reserve('after-release','D3',1,limit=8,kind='probe')
    parent.release_stack_capacity('seq:512:dagger');parent.finish_sequence('seq',passed=False)


def test_probe_step_without_capacity_is_rejected_even_when_cap_fits(tmp_path):
    path=tmp_path/'global.jsonl';auth=authorization(tmp_path,path,cap=8)
    parent=GlobalProbeBudget(path,auth);parent.begin_sequence('r',required_samples=8)
    child=GlobalProbeBudget(path,auth,scope_id='r:512:dagger');child.assert_authorized('r')
    with pytest.raises(ProbeBudgetError,match='capacity lease'):
        child.reserve('step','D3',1,limit=8,kind='probe')
    assert parent.global_report()['global_totals']['charged']==0


def test_superseding_A1_authorization_preserves_cap_and_requires_registered_parent(tmp_path):
    import copy
    path=tmp_path/'global.jsonl';parent=authorization(tmp_path,path,cap=8)
    base=json.loads(parent.read_text());parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    child=tmp_path/'A1.json';inherited=copy.deepcopy(base)
    inherited.update(authorization_id='A1-new-id',supersedes=dict(authorization_id=base['authorization_id'],path=str(parent.resolve()),sha256=parent_sha))
    child.write_text(json.dumps(inherited))
    b=GlobalProbeBudget(path,child)
    with pytest.raises(ProbeBudgetError,match='registered'):b.begin_sequence('a1',required_samples=4)
    old=GlobalProbeBudget(path,parent);old.begin_sequence('old',required_samples=4)
    with pytest.raises(ProbeBudgetError,match='active sequence'):b.begin_sequence('a1',required_samples=4)
    old.finish_sequence('old')
    for field,value in [('max_probe_samples',9),('global_ledger_path',str(tmp_path/'different.jsonl'))]:
        bad=copy.deepcopy(inherited);bad[field]=value;child.write_text(json.dumps(bad))
        with pytest.raises(ProbeBudgetError):GlobalProbeBudget(path,child).begin_sequence('a1',required_samples=4)
    bad=copy.deepcopy(inherited);bad['supersedes']['sha256']='0'*64;child.write_text(json.dumps(bad))
    with pytest.raises(ProbeBudgetError,match='supersedes'):GlobalProbeBudget(path,child).begin_sequence('a1',required_samples=4)
    child.write_text(json.dumps(inherited))
    b=GlobalProbeBudget(path,child);first=b.begin_sequence('a1',required_samples=4)
    assert b.begin_sequence('a1',required_samples=4)==first
    assert len(b.global_report()['authorizations'])==2
    assert b.global_report()['global_totals']['charged']==0
    with pytest.raises(ProbeBudgetError,match='active'):old.begin_sequence('concurrent-old',required_samples=4)
    b.finish_sequence('a1')
