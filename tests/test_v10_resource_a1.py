"""CPU-only A1 resource association; no abbreviated check is a new full probe."""
import json
from pathlib import Path
import shutil
import pytest
from src.tasks.velocity.v10 import spec
from src.tasks.velocity.v10.resource_a1 import (
    A1_PROTOCOL,A1_STACK_SAMPLES,A1_TOTAL_SAMPLES,OLD_BASELINE_DIR,
    verify_old_evidence,build_a1_contract_template,validate_a1_contract,
)


def test_a1_protocol_counts_are_separate_from_full_probe(monkeypatch):
    from src.tasks.velocity.v10 import resources
    captured=[]
    def implementation(**kw):captured.append(kw);return kw
    monkeypatch.setattr(resources,'_probe_stack_impl',implementation)
    resources.probe_stack(num_envs=1024,stack='dagger',output_dir='unused')
    assert captured[-1]['warmup_rounds']==5 and captured[-1]['measurement_rounds']==20
    assert A1_STACK_SAMPLES==245760 and A1_TOTAL_SAMPLES==491520


def test_old_full_results_pinned_and_tampering_rejected(tmp_path):
    evidence=verify_old_evidence(OLD_BASELINE_DIR)
    assert evidence['num_envs']==1024 and set(evidence['runs'])=={'dagger','ppo'}
    for relative in ('selection.json','n1024_dagger/result.json','n1024_ppo/result.json'):
        dest=tmp_path/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD_BASELINE_DIR/relative,dest)
    (tmp_path/'n1024_ppo/result.json').write_text('{}')
    with pytest.raises(ValueError,match='old.*SHA|old.*identity'):verify_old_evidence(tmp_path)


def test_contract_template_never_approves_itself(tmp_path):
    template=build_a1_contract_template(tmp_path/'contract.json',evidence_dir=tmp_path/'evidence')
    assert template['approved'] is False
    with pytest.raises(ValueError,match='approval|approved'):validate_a1_contract(tmp_path/'contract.json')


@pytest.fixture
def association_receipts(tmp_path,monkeypatch):
    """Synthetic new result files test the verifier, never resource admission."""
    from src.tasks.velocity.v10 import resource_a1 as a
    from src.tasks.velocity.v10.identity import code_fingerprint
    from src.tasks.velocity.v10.probe_budget import streaming_sha256
    contract=tmp_path/'approved_fixture.json';contract.write_text('{}')
    report=tmp_path/'review.txt';report.write_text('CPU fixture only, no physical evidence')
    extra={}
    for name in ('invariance_evidence','source_diff','proposal'):
        p=tmp_path/(name+'.json');p.write_text('{}');extra[name]=dict(path=str(p),sha256=streaming_sha256(p))
    old=a.verify_old_evidence()
    link=dict(contract=dict(baseline_dir=str(a.OLD_BASELINE_DIR),independent_review=dict(report_path=str(report),report_sha256=streaming_sha256(report)),**extra),
              contract_path=str(contract),contract_sha256=streaming_sha256(contract),old=old)
    monkeypatch.setattr(a,'validate_a1_contract',lambda path:link)
    selection=dict(passed=True,resource_protocol=a.A1_PROTOCOL,num_envs=1024,sds_sha256=spec.SDS_SHA256,
                   rulings_sha256=streaming_sha256(a.RULINGS),code_sha256=code_fingerprint(),physical_sha256=old['physical_sha256'],
                   global_probe_ledger='CPU-fixture',probe_sequence_id='fixture',probe_budget_authorization_sha256='a'*64,
                   linkage_contract_path=str(contract),linkage_contract_sha256=streaming_sha256(contract),probe_state_discarded=True,full_twenty_measurement_revalidation=False,runs=[])
    for stack in ('dagger','ppo'):
        row=dict(passed=True,stack=stack,num_envs=1024,resource_protocol=a.A1_PROTOCOL,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),
                 physical_sha256=old['physical_sha256'],warmup_rounds=2,measurement_rounds=2,fill_rollouts=1,measured_seconds=[1.,1.],
                 restore_checked=True,replay_counts={r:32768 for r in spec.ROLES},gpu_total_peak_mib=1000,gpu_task_peak_mib=1000,host_available_min_bytes=8*1024**3,
                 cost=dict(charged={'probe:X':245760},executed={'probe:X':245760},uncertain={'probe:X':0},updated={'probe:X':196608}),
                 global_probe_ledger='CPU-fixture',probe_sequence_id='fixture',probe_budget_authorization_sha256='a'*64,linkage_contract_sha256=streaming_sha256(contract),
                 checkpoint_sha256='c'*64,next_sample_manifest='d'*64,restored_replay_manifest=dict(capacity=32768,sha256='e'*64,counts={r:32768 for r in spec.ROLES}),
                 memory_stages=[dict(stage=x) for x in sorted(a.A1_REQUIRED_STAGES)])
        p=tmp_path/f'n1024_{stack}/result.json';p.parent.mkdir();p.write_text(json.dumps(row));selection['runs'].append(row)
    a.make_linkage_certificate(link,selection,tmp_path)
    cert=tmp_path/'resource_linkage_certificate.json'
    selection.update(resource_linkage_certificate_path=str(cert),resource_linkage_certificate_sha256=streaming_sha256(cert))
    return selection,link,tmp_path


def test_association_requires_exact_real_new_results(association_receipts):
    import copy
    from src.tasks.velocity.v10 import resource_a1 as a
    selection,link,out=association_receipts
    assert a.verify_linkage_selection(selection)['full_twenty_measurement_revalidation'] is False
    bad=copy.deepcopy(selection);bad['runs'][0]['gpu_task_peak_mib']=1
    with pytest.raises(ValueError,match='raw targeted result'):a.verify_linkage_selection(bad)
    bad=copy.deepcopy(selection);bad['runs']=[bad['runs'][0],bad['runs'][0]]
    with pytest.raises(ValueError,match='distinct'):a.verify_linkage_selection(bad)
    (out/'n1024_ppo/result.json').unlink()
    with pytest.raises(ValueError,match='raw targeted result'):a.verify_linkage_selection(selection)


def test_association_cannot_drop_required_references_or_change_N(association_receipts):
    from src.tasks.velocity.v10 import resource_a1 as a
    from src.tasks.velocity.v10.probe_budget import streaming_sha256
    selection,link,out=association_receipts
    cert=out/'resource_linkage_certificate.json';row=json.loads(cert.read_text());row['references']={};cert.write_text(json.dumps(row))
    selection['resource_linkage_certificate_sha256']=streaming_sha256(cert)
    with pytest.raises(ValueError,match='certificate.*mismatch'):a.verify_linkage_selection(selection)
    selection['num_envs']=512
    with pytest.raises(ValueError,match='selection identity'):a.verify_linkage_selection(selection)


@pytest.mark.parametrize('dagger_passes',[True,False])
def test_targeted_parent_exact_budget_and_first_failure_stops(association_receipts,monkeypatch,dagger_passes):
    import copy
    from types import SimpleNamespace
    from src.tasks.velocity.v10 import resource_a1 as a
    from src.tasks.velocity.v10.probe_budget import GlobalProbeBudget,ProbeBudgetInsufficient,streaming_sha256,migrate_legacy_ledgers
    selection,link,fixture=association_receipts
    ledger_path=fixture/'global.jsonl';legacy=fixture/'old_costs.jsonl'
    records=[dict(seq=1,op='charge',id='historical-cost',key='probe:D3',samples=3293184),
             dict(seq=2,op='update',id='historical-update',key='probe:D3',samples=3121152)]
    legacy.write_text(''.join(json.dumps(r)+'\n' for r in records));migrate_legacy_ledgers(ledger_path,[legacy])
    auth=fixture/'user-fixture.json'
    auth.write_text(json.dumps(dict(schema='a2-v10-global-probe-budget-authorization-v1',approved=True,authorization_id='CPU-only-A1-fixture',scope='probe_only',
        sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(a.RULINGS),global_ledger_path=str(ledger_path),max_probe_samples=3833856,prior_charged_samples=3293184,
        approval_source={'kind':'user','reference':'CPU test fixture only; no actual physics'})))
    sequence='a1-targeted-test';calls=[]
    def child_fixture(command,**kw):
        assert command[2]=='_probe-a1-stack'
        arg=lambda key:command[command.index(key)+1]
        stack=arg('--stack');calls.append(stack)
        b=GlobalProbeBudget(ledger_path,auth,scope_id=f'{sequence}:1024:{stack}');grant=b.assert_authorized(sequence,required_samples=a.A1_STACK_SAMPLES)
        b.acquire_stack_capacity(sequence,a.A1_STACK_SAMPLES) # parent's same lease, not a second reservation
        b.reserve('CPU-aggregate', 'D3' if stack=='dagger' else 'PPO',a.A1_STACK_SAMPLES,kind='probe',limit=a.A1_STACK_SAMPLES);b.mark_executed('CPU-aggregate')
        b.record_update('CPU-updates','D3' if stack=='dagger' else 'PPO',196608,kind='probe')
        row=copy.deepcopy(next(r for r in selection['runs'] if r['stack']==stack))
        row.update(passed=dagger_passes or stack=='ppo',global_probe_ledger=str(ledger_path),probe_sequence_id=sequence,
                   probe_budget_authorization_sha256=grant['sha256'],cost=b.report())
        out=Path(arg('--output-dir'));out.mkdir();(out/'result.json').write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a.subprocess,'run',child_fixture)
    out=fixture/'new-targeted'
    result=a.run_a1_probes(output_dir=out,linkage_contract=link['contract_path'],global_probe_ledger=ledger_path,probe_budget_authorization=auth,probe_sequence_id=sequence)
    assert calls==(['dagger','ppo'] if dagger_passes else ['dagger'])
    assert result['passed'] is dagger_passes
    expected=3784704 if dagger_passes else 3538944
    assert GlobalProbeBudget(ledger_path).global_report()['global_totals']['charged']==expected
    assert GlobalProbeBudget(ledger_path).global_report()['active_capacity'] is None
    if dagger_passes:
        assert a.verify_linkage_selection(result)['new_measurement']==2
        assert 3833856-expected==49152
    with pytest.raises(ValueError,match='no targeted remeasure'):a.run_a1_probes(output_dir=out,linkage_contract=link['contract_path'],global_probe_ledger=ledger_path,probe_budget_authorization=auth,probe_sequence_id=sequence)
    before=list(calls)
    with pytest.raises(ProbeBudgetInsufficient):a.run_a1_probes(output_dir=fixture/'retry',linkage_contract=link['contract_path'],global_probe_ledger=ledger_path,probe_budget_authorization=auth,probe_sequence_id='new-attempt')
    assert calls==before


def test_real_invariants_keep_env_byte_difference_and_exact_recording_ast(tmp_path):
    from src.tasks.velocity.v10 import resource_a1 as a
    from src.tasks.velocity.v10.probe_budget import streaming_sha256
    path=tmp_path/'contract.json';row=a.build_a1_contract_template(path,tmp_path/'proof')
    proof=json.loads(Path(row['invariance_evidence']['path']).read_text())
    assert proof['invariants_satisfied'] is True
    env=proof['files']['src/tasks/velocity/v10/env.py']
    assert env['unchanged'] is False and env['recording_only_ast_equivalent'] is True
    assert env['baseline_sha256']!=env['current_sha256']
    report=tmp_path/'independent-fixture.md';report.write_text('CPU verifier fixture only; no production approval')
    row.update(approved=True,independent_review=dict(passed=True,report_path=str(report),report_sha256=streaming_sha256(report)))
    path.write_text(json.dumps(row))
    assert a.validate_a1_contract(path)['contract']['approved']
    diff_path=Path(row['source_diff']['path']);diff=json.loads(diff_path.read_text());assert diff['changed_files']
    diff['changed_files'].pop();diff_path.write_text(json.dumps(diff));row['source_diff']['sha256']=streaming_sha256(diff_path);path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='list incomplete'):a.validate_a1_contract(path)


@pytest.mark.parametrize('mutation',['extra_field','extra_expression','duplicate','wrong_position'])
def test_recording_ast_proof_rejects_other_changes(monkeypatch,mutation):
    import ast,copy
    from src.tasks.velocity.v10 import resource_a1 as a
    relative='src/tasks/velocity/v10/env.py';path=spec.ROOT/relative
    original=Path.read_text;tree=ast.parse(original(path))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='V10Env')
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    expected=ast.dump(ast.parse(a._RECORDING_UPDATE).body[0],include_attributes=False)
    index=next(i for i,n in enumerate(step.body) if ast.dump(n,include_attributes=False)==expected)
    if mutation=='extra_field':step.body[index].value.keywords.append(ast.keyword(arg='unapproved',value=ast.Constant(1)))
    elif mutation=='extra_expression':step.body.insert(index,ast.parse('random.random()').body[0])
    elif mutation=='duplicate':step.body.insert(index,copy.deepcopy(step.body[index]))
    else:step.body.insert(0,step.body.pop(index))
    modified=ast.unparse(ast.fix_missing_locations(tree))
    def read(p,*args,**kwargs):return modified if p==path else original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert a._env_recording_equivalence(relative) is False


def test_extremely_early_failure_cannot_change_run_to_remeasure(association_receipts,monkeypatch):
    from types import SimpleNamespace
    from src.tasks.velocity.v10 import resource_a1 as a
    from src.tasks.velocity.v10.probe_budget import GlobalProbeBudget,ProbeBudgetError,streaming_sha256,migrate_legacy_ledgers
    selection,link,fixture=association_receipts
    path=fixture/'global-early.jsonl';old=fixture/'legacy-early.jsonl'
    old.write_text(json.dumps(dict(seq=1,op='charge',id='old',key='probe:D3',samples=3293184))+'\n');migrate_legacy_ledgers(path,[old])
    auth=fixture/'early-auth.json';auth.write_text(json.dumps(dict(schema='a2-v10-global-probe-budget-authorization-v1',approved=True,
        authorization_id='CPU-only-early-failure',scope='probe_only',sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(a.RULINGS),
        global_ledger_path=str(path),max_probe_samples=3833856,prior_charged_samples=3293184,approval_source={'kind':'user','reference':'CPU fixture only'})))
    calls=[]
    def early_failure(command,**kw):
        arg=lambda key:command[command.index(key)+1]
        sequence=arg('--probe-sequence-id');calls.append(sequence)
        ledger=GlobalProbeBudget(path,auth,scope_id=f'{sequence}:1024:dagger');ledger.assert_authorized(sequence)
        ledger.reserve('cpu-first-step','D3',1024,limit=245760,kind='probe');ledger.mark_executed('cpu-first-step')
        out=Path(arg('--output-dir'));out.mkdir();(out/'result.json').write_text(json.dumps({'passed':False,'error':'CPU simulated early resource failure'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a.subprocess,'run',early_failure)
    kwargs=dict(linkage_contract=link['contract_path'],global_probe_ledger=path,probe_budget_authorization=auth)
    assert a.run_a1_probes(output_dir=fixture/'first-early',probe_sequence_id='first-early',**kwargs)['passed'] is False
    assert 3833856-GlobalProbeBudget(path).global_report()['global_totals']['charged']>=491520
    with pytest.raises(ProbeBudgetError,match='already claimed'):
        a.run_a1_probes(output_dir=fixture/'different-dir',probe_sequence_id='different-run',**kwargs)
    assert calls==['first-early']
    report=GlobalProbeBudget(path).global_report()
    assert len(report['protocol_attempts'])==1 and report['global_totals']['charged']==3294208 and report['active_sequence'] is None
