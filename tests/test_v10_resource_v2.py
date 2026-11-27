"""Unique fixed v2 resource supplement; CPU receipts are not real admission."""
import copy
import json
import os
import time
from pathlib import Path
import pytest
from test_v10_resource_a1 import association_receipts
from src.tasks.velocity.v10 import resource_a1 as a,spec
from src.tasks.velocity.v10.probe_budget import streaming_sha256


def startup_report(path):
    report=dict(schema='a2-v10-v2-startup-headroom-v1',passed=True,required_bytes=12*1024**3,required_seconds=60,
        observed_seconds=60,min_available_bytes=12*1024**3,owner_pid=os.getpid(),created_unix=time.time(),pre_torch_import=True,
        samples=[dict(elapsed_seconds=i,available_bytes=12*1024**3) for i in range(61)])
    path.write_text(json.dumps(report));return path


def test_v2_fixed_profile_and_probe_function_ast_only():
    assert a.PROFILES[a.A1_V2_PROTOCOL]==dict(warmup=1,measurement=2,stack_samples=196608,total_samples=393216)
    assert a.PROFILES[a.A1_PROTOCOL]['warmup']==2
    r=a._invariance_record('src/tasks/velocity/v10/runtime.py',a.A1_V2_PROTOCOL)
    assert not r['unchanged'] and r['probe_path_ast_equivalent'] is False
    baseline=spec.ROOT/'logs/a2_v10/formal_handoff_baseline'
    historical=a._invariance_record('src/tasks/velocity/v10/runtime.py',a.A1_V2_PROTOCOL,source_root=baseline)
    assert not historical['unchanged'] and historical['probe_path_ast_equivalent'] is True
    assert a._failed_context()['passed'] is False and a._failed_context()['positive_resource_evidence'] is False


def test_direct_v2_without_parent_gate_cannot_claim(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('must reject missing startup gate before contract/claim')
    monkeypatch.setattr(a,'validate_a1_contract',forbidden)
    with pytest.raises(ValueError,match='headroom report'):
        a.run_a1_v2_probes(output_dir=tmp_path/'out',linkage_contract='unused')


def test_v2_certificate_exact_cost_and_archived_headroom_not_expired(association_receipts):
    selection,link,out=association_receipts
    selection.update(resource_protocol=a.A1_V2_PROTOCOL,failed_context=a._failed_context())
    link['contract']['protocol']=a.A1_V2_PROTOCOL
    gate=startup_report(out/'gate.json')
    selection.update(startup_headroom_path=str(gate),startup_headroom_sha256=streaming_sha256(gate))
    for row in selection['runs']:
        row.update(resource_protocol=a.A1_V2_PROTOCOL,warmup_rounds=1,
                   cost=dict(charged={'probe:X':196608},executed={'probe:X':196608},uncertain={'probe:X':0},updated={'probe:X':147456}),
                   prefill_headroom=dict(passed=True,required_bytes=19*1024**3//2,available_bytes=10*1024**3),
                   startup_headroom_sha256=streaming_sha256(gate),memory_stages=[dict(stage=x) for x in sorted(a._required_stages(a.PROFILES[a.A1_V2_PROTOCOL])|{'prefill_headroom_check'})])
        (out/f'n1024_{row["stack"]}/result.json').write_text(json.dumps(row))
    # Verification is historical evidence checking, not a new startup gate.
    archived=json.loads(gate.read_text());archived['created_unix']-=3600;archived['owner_pid']=99999999;gate.write_text(json.dumps(archived))
    selection['startup_headroom_sha256']=streaming_sha256(gate)
    for row in selection['runs']:
        row['startup_headroom_sha256']=streaming_sha256(gate);row['startup_parent_pid']=archived['owner_pid'];(out/f'n1024_{row["stack"]}/result.json').write_text(json.dumps(row))
    a.make_linkage_certificate(link,selection,out)
    cert=out/'resource_linkage_certificate.json';selection['resource_linkage_certificate_sha256']=streaming_sha256(cert)
    verified=a.verify_linkage_selection(selection)
    assert verified['new_warmup']==1 and verified['new_measurement']==2
    assert verified['failed_context']['positive_resource_evidence'] is False
    broken=copy.deepcopy(selection);broken['runs'][0]['cost']['charged']['probe:X']=245760
    with pytest.raises(ValueError):a.verify_linkage_selection(broken)


def test_prefill_failure_in_actual_shared_stack_has_zero_sampling(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.tasks.velocity.v10 import resources,headroom
    from src.tasks.velocity.v10.env import V10Env
    import src.tasks.velocity.v10.env as env_module
    from src.tasks.velocity.v10.probe_budget import GlobalProbeBudget
    ledger=tmp_path/'global.jsonl';auth=tmp_path/'auth.json'
    auth.write_text(json.dumps(dict(schema='a2-v10-global-probe-budget-authorization-v1',approved=True,authorization_id='CPU-prefill-only',scope='probe_only',
        sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(a.RULINGS),global_ledger_path=str(ledger),max_probe_samples=393216,
        prior_charged_samples=0,approval_source=dict(kind='user',reference='CPU fixture, no physics'))))
    parent=GlobalProbeBudget(ledger,auth);parent.begin_sequence('v2',required_samples=393216)
    parent.acquire_stack_capacity('v2',196608,scope_id='v2:1024:dagger')
    class Monitor:
        rows=[];memory_stages=[]
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def memory_stage(self,row):self.memory_stages.append(row)
    monkeypatch.setattr(resources,'ResourceMonitor',Monitor)
    monkeypatch.setattr(resources,'trace_memory',lambda *args:None)
    monkeypatch.setattr(resources,'seed_all',lambda *args:None)
    monkeypatch.setattr(resources.torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(env_module,'V10Env',lambda *args:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(resources,'physical_contract',lambda env:dict(sha256='physical'))
    monkeypatch.setattr(resources.V10ActorCritic,'from_stairs',lambda **kw:object())
    monkeypatch.setattr(resources,'TaskReservoir',lambda:object())
    monkeypatch.setattr(resources,'DualTeacher',lambda model:SimpleNamespace(to=lambda device:object()))
    monkeypatch.setattr(headroom,'prefill_headroom',lambda:dict(passed=False,required_bytes=19*1024**3//2,available_bytes=9*1024**3))
    def forbidden(*a,**kw):raise AssertionError('no rollout may run below prefill threshold')
    monkeypatch.setattr(resources,'collect_rollout',forbidden)
    result=resources._probe_stack_impl(num_envs=1024,stack='dagger',output_dir=tmp_path/'child',global_probe_ledger=ledger,
        probe_budget_authorization=auth,probe_sequence_id='v2',warmup_rounds=1,measurement_rounds=2,protocol=a.A1_V2_PROTOCOL,require_prefill_headroom=True)
    assert not result['passed'] and 'zero physical samples' in result['error']
    assert result['cost']['global_totals']['charged']==0
    parent.release_stack_capacity('v2:1024:dagger');parent.finish_sequence('v2')


def test_v2_parent_uses_exact_393216_and_binds_original_gate_claim(association_receipts,monkeypatch):
    from types import SimpleNamespace
    from src.tasks.velocity.v10.probe_budget import GlobalProbeBudget,migrate_legacy_ledgers,ProbeBudgetInsufficient
    selection,link,out=association_receipts;link['contract']['protocol']=a.A1_V2_PROTOCOL
    startup=startup_report(out/'current_gate.json')
    ledger=out/'v2-global.jsonl';old=out/'oldcost.jsonl'
    old.write_text(json.dumps(dict(seq=1,op='charge',id='old',key='probe:D3',samples=3391488))+'\n')
    migrate_legacy_ledgers(ledger,[old]);auth=out/'v2auth.json'
    auth.write_text(json.dumps(dict(schema='a2-v10-global-probe-budget-authorization-v1',approved=True,authorization_id='CPU-v2-fixture',scope='probe_only',
        sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(a.RULINGS),global_ledger_path=str(ledger),max_probe_samples=3833856,
        prior_charged_samples=3391488,approval_source=dict(kind='user',reference='CPU-only transaction fixture'))))
    sequence='v2-once';calls=[]
    def child(command,**kwargs):
        assert command[2]=='_probe-a1-stack-v2'
        arg=lambda name:command[command.index(name)+1];stack=arg('--stack');calls.append(stack)
        b=GlobalProbeBudget(ledger,auth,scope_id=f'{sequence}:1024:{stack}');grant=b.assert_authorized(sequence,required_samples=196608)
        claim=b.require_protocol_attempt(a.A1_V2_PROTOCOL,link['contract_sha256'],sequence)
        assert claim['admission_evidence']['sha256']==streaming_sha256(startup)
        b.acquire_stack_capacity(sequence,196608);b.reserve('cpu-total','D3' if stack=='dagger' else 'PPO',196608,limit=196608,kind='probe');b.mark_executed('cpu-total')
        b.record_update('cpu-updates','D3' if stack=='dagger' else 'PPO',147456,kind='probe')
        row=copy.deepcopy(next(r for r in selection['runs'] if r['stack']==stack))
        row.update(resource_protocol=a.A1_V2_PROTOCOL,warmup_rounds=1,cost=b.report(),global_probe_ledger=str(ledger),probe_sequence_id=sequence,
            probe_budget_authorization_sha256=grant['sha256'],startup_headroom_sha256=streaming_sha256(startup),startup_parent_pid=os.getpid(),
            prefill_headroom=dict(passed=True,required_bytes=19*1024**3//2,available_bytes=10*1024**3),
            memory_stages=[dict(stage=s) for s in a._required_stages(a.PROFILES[a.A1_V2_PROTOCOL])|{'prefill_headroom_check'}])
        target=Path(arg('--output-dir'));target.mkdir();(target/'result.json').write_text(json.dumps(row));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a.subprocess,'run',child)
    result=a.run_a1_v2_probes(output_dir=out/'fresh-v2',linkage_contract=link['contract_path'],startup_headroom=startup,
        global_probe_ledger=ledger,probe_budget_authorization=auth,probe_sequence_id=sequence)
    assert calls==['dagger','ppo'] and result['passed']
    assert result['global_probe_totals']['charged']==3784704 and 3833856-result['global_probe_totals']['charged']==49152
    assert a.verify_linkage_selection(result)['new_warmup']==1
    with pytest.raises(ProbeBudgetInsufficient):a.run_a1_v2_probes(output_dir=out/'no-retry',linkage_contract=link['contract_path'],startup_headroom=startup,
        global_probe_ledger=ledger,probe_budget_authorization=auth,probe_sequence_id='new-id')
    assert calls==['dagger','ppo']


@pytest.mark.parametrize('mutation',['collect','save','module_side_effect','worker_return_annotation'])
def test_runtime_probe_ast_rejects_changed_execution_or_import_side_effect(monkeypatch,mutation):
    import ast
    path=spec.ROOT/'src/tasks/velocity/v10/runtime.py';original=Path.read_text;tree=ast.parse(original(path))
    if mutation in ('collect','save'):
        name='collect_rollout' if mutation=='collect' else 'save_worker'
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        function.body.insert(0,ast.parse('env.reset()').body[0])
    elif mutation=='module_side_effect':tree.body.append(ast.parse('unapproved_global_call()').body[0])
    else:
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        worker.returns=ast.parse('unapproved_global_call()',mode='eval').body
    text=ast.unparse(ast.fix_missing_locations(tree))
    monkeypatch.setattr(Path,'read_text',lambda p,*args,**kw:text if p==path else original(p,*args,**kw))
    assert a._runtime_probe_equivalence('src/tasks/velocity/v10/runtime.py') is False
