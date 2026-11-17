"""CPU engineering protocol evidence; never starts a learning worker."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import recovery, spec


def test_baseline_cli_dry_run_declares_45_cases_without_creating_output(tmp_path):
    output=tmp_path/'plan_only'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_a2_ideal_stairs_v07.py'),
        '--stage','baseline','--device','cpu','--seed','42','--output-dir',str(output)],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    plan=json.loads(result.stdout)
    assert plan['baseline']['cases_total']==45
    assert plan['baseline']['learning_updates']==0
    assert not output.exists()


def test_d22_smoke_plan_uses_eight_envs_two_updates_and_real_tile_kwargs():
    engine=import_module('validate_a2_ideal_stairs_v07')
    plan=engine.smoke_plan()
    assert plan['environment']==dict(phase='D',substage='D22',level_tick=0,
        frontier_tile_id=22,lesson_id='D22',N=6,h=.2,step_depth=.22,width=.8,tile_id=22)
    assert plan['plan']['num_envs']==8 and plan['plan']['updates']==2
    assert plan['cost']==dict(learning_updates=2,control_samples=768,physics_substeps=3072,adam_updates=40)
    assert plan['resumable'] is False and plan['counts_against_training'] is False


def test_baseline_identity_uses_terrain_speed_case_composites_and_selected_hashes():
    engine=import_module('validate_a2_ideal_stairs_v07')
    plan=engine.baseline_plan()
    assert plan['cases_total']==45
    assert [row['substage'] for row in plan['terrains']]==['D25','D22','D20']
    pairs=[]
    for row,tile in zip(plan['terrains'],[17,22,18]):
        assert len(row['selected_cases'])==15
        assert row['selected_cases_sha256']==recovery.digest(row['selected_cases'])
        assert all(case['tile_id']==tile for case in row['selected_cases'])
        for speed in [.3,.5,.6]:
            assert sum(case['vx']==speed for case in row['selected_cases'])==5
        pairs.extend((row['substage'],case['vx'],case['case_id']) for case in row['selected_cases'])
    assert len(set(pairs))==45
    assert len({case['case_id'] for row in plan['terrains'] for case in row['selected_cases']})<45


def record(case_id='validation:0.3:000'):
    return dict(case_id=case_id,steps=3,physical_control_samples=8,physical_physics_substeps=32,
        extra_control_samples_after_case=5,extra_physics_substeps_after_case=20,
        evidence_kind='terminal',first_episode=dict(env_id=0,control_samples=3),
        climb_success=False,failure=True,timeout=False,reason='body')


def test_evaluation_cost_keeps_first_episode_and_extra_work_separate():
    engine=import_module('validate_a2_ideal_stairs_v07')
    got=engine.evaluation_cost([record(),record()])
    assert got==dict(case_control_steps=6,actual_eval_control_samples=16,
        extra_eval_control_samples=10,actual_eval_physics_substeps=64,extra_eval_physics_substeps=40)
    bad=record();bad['physical_control_samples']=7
    with pytest.raises(ValueError):engine.evaluation_cost([bad])


def test_persisted_batch_binds_actual_first_records_and_summary(tmp_path):
    engine=import_module('validate_a2_ideal_stairs_v07')
    cases=[dict(case_id='validation:0.3:000',tile_id=22,lesson_id='D22',vx=.3)]
    records=[record()]
    summary=dict(total=1,case_pass_count=0)
    receipt=engine.persist_case_batch(tmp_path,'D22_vx0.3',cases,records,summary)
    artifact=json.loads(Path(receipt['path']).read_text())
    assert artifact['records']==records
    assert artifact['summary']==summary
    assert artifact['case_table_sha256']==recovery.digest(cases)
    assert artifact['records_sha256']==recovery.digest(records)
    assert artifact['summary_sha256']==recovery.digest(summary)
    assert receipt['sha256']==spec.sha256(receipt['path'])
    bad=deepcopy(records);bad[0]['case_id']='wrong'
    with pytest.raises(ValueError):engine.persist_case_batch(tmp_path,'bad',cases,bad,summary)
    bad=deepcopy(records);bad[0].pop('first_episode')
    with pytest.raises(ValueError):engine.persist_case_batch(tmp_path,'missing-first',cases,bad,summary)


def diagnostic_sidecar(tmp_path):
    from src.tasks.velocity.v07 import evaluation,evaluation_records
    table=evaluation.make_cases('R',None,'diagnostic',42)
    records=[]
    for i,case in enumerate(table.cases):
        row=record(case['case_id'])
        row.update(vx=case['vx'],mean_vx=0.,measurement='stair')
        row['first_episode']['env_id']=i%5
        if i%5==4:  # one last slot keeps each real five-case batch alive to step8
            row.update(steps=8,extra_control_samples_after_case=0,extra_physics_substeps_after_case=0)
            row['first_episode']['control_samples']=8
        records.append(row)
    isolation=evaluation.compare_snapshots({'model':'unchanged'},{'model':'unchanged'})
    summary=evaluation.summarize(records,table);summary.update(isolation)
    hashes=dict(case_table_sha256=table.sha256,records_sha256=recovery.digest(records),summary_sha256=recovery.digest(summary))
    result=evaluation.EvalResult(table,summary,records,isolation,hashes)
    identity=dict(update=50,phase='R',substage=None,generation=0,kind='diagnostic',
        model_sha256='a'*64,source_closure_sha256='b'*64,contract_sha256='c'*64,case_table_sha256=table.sha256)
    payload=evaluation_records.pack_result(result,identity)
    return evaluation_records.write_sidecar(tmp_path/'diagnostic50.json',payload)


def test_probe_verifies_persisted_update50_records_and_rejects_missing_or_tampered_raw(tmp_path,monkeypatch):
    from src.tasks.velocity.v07 import evaluation,orchestration
    worker=import_module('train_a2_ideal_stairs_v07')
    runner,binding,observer,_=live_probe50_fixture(tmp_path,monkeypatch)
    proof=worker.probe_diagnostic_evidence(runner)
    assert proof['update']==50 and proof['case_count']==15
    assert proof['actual_eval_control_samples']==120
    assert proof['case_control_steps']==60 and proof['extra_eval_control_samples']==60
    payload=json.loads(Path(binding['path']).read_text());payload['records'][0]['first_episode']['control_samples']=999
    Path(binding['path']).write_text(json.dumps(payload))
    with pytest.raises(evaluation.EvaluationError):worker.probe_diagnostic_evidence(runner)
    Path(binding['path']).unlink()
    with pytest.raises(orchestration.ProbeError):worker.probe_diagnostic_evidence(runner)
    runner.diagnostics=[]
    with pytest.raises(orchestration.ProbeError):worker.probe_diagnostic_evidence(runner)


def test_engineering_execute_blocks_on_incomplete_frozen_sources_before_stage(tmp_path,monkeypatch):
    engine=import_module('validate_a2_ideal_stairs_v07')
    monkeypatch.setattr(spec,'pending_source_entries',lambda:['scripts/not-yet-implemented.py'])
    with pytest.raises(RuntimeError,match='incomplete'):
        engine.verify_engineering_freeze(tmp_path/'missing-freeze.json')


@pytest.mark.parametrize('change',['nonterminal_raw','foreign_identity'])
def test_probe_rejects_self_consistent_counterexamples_without_independent_p50(tmp_path,change):
    from types import SimpleNamespace
    from src.tasks.velocity.v07 import evaluation,evaluation_records,orchestration
    worker=import_module('train_a2_ideal_stairs_v07')
    binding=diagnostic_sidecar(tmp_path)
    payload=json.loads(Path(binding['path']).read_text())
    if change=='nonterminal_raw':
        for row in payload['records']:row['first_episode']={'arbitrary_non_terminal_evidence':1}
    else:
        payload['identity']['source_closure_sha256']='0'*64
        payload['identity']['model_sha256']='0'*64
        payload['identity'].pop('optimizer_sha256',None)
    payload['evidence_sha256']['records_sha256']=recovery.digest(payload['records'])
    forged=evaluation_records.write_sidecar(tmp_path/f'{change}.json',payload)
    runner=SimpleNamespace(diagnostics=[dict(update=50,case_records=forged)])
    with pytest.raises((evaluation.EvaluationError,orchestration.ProbeError)):
        worker.probe_diagnostic_evidence(runner)


def live_probe50_fixture(tmp_path,monkeypatch,change=None):
    """Actual tiny torch model+1000 Adam steps; controlled terminal exports for update50."""
    from types import SimpleNamespace
    import torch
    from src.tasks.velocity.v07 import evaluation,evaluation_records,model
    worker=import_module('train_a2_ideal_stairs_v07')
    frozen_sources={'scripts/train_a2_ideal_stairs_v07.py':spec.sha256(ROOT/'scripts/train_a2_ideal_stairs_v07.py')}
    monkeypatch.setattr(spec,'source_hashes',lambda **kwargs:dict(frozen_sources))
    monkeypatch.setattr(spec,'pending_source_entries',lambda **kwargs:[])
    with torch.random.fork_rng(devices=[]):
        network=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(network.parameters(),lr=1e-4)
    for _ in range(50*20):
        optimizer.zero_grad();network(torch.ones(1,2)).square().sum().backward();optimizer.step()
    learning=dict(update=50,control_samples=1_228_800,physics_substeps=4_915_200,adam_steps=1000)
    table=evaluation.make_cases('R',None,'diagnostic',42,tile_id=21)
    records=[]
    for i,case in enumerate(table.cases):
        index=i%5;step=8 if index==4 else 3
        raw=dict(env_id=index,control_samples=step,terminal_step=step,
            success=False,failure=True,timeout=False,first_step_valid=False,
            active_samples=step,stall_samples=step,mean_vx=0.,mean_speed=0.,
            max_abs_y=0.,max_abs_yaw_deg=0.,reason='body',tile_id=21,lesson_id='L40',
            frontier_tile_id=21,phase='R',substage=None,seed=42,generation=0,
            initial_pose=[float(v) for v in torch.tensor(case['initial_pose'],dtype=torch.float32)],
            command=float(torch.tensor(case['vx'],dtype=torch.float32)))
        row=dict(case_id=case['case_id'],vx=case['vx'],measurement='stair',
            first_episode=raw,evidence_kind='terminal',env_index=index,steps=step,
            first_observed_at_batch_step=step,batch_loop_control_steps=8,
            physical_control_samples=8,physical_physics_substeps=32,
            extra_control_samples_after_case=8-step,extra_physics_substeps_after_case=(8-step)*4,
            climb_success=False,**{key:raw[key] for key in ['failure','timeout','reason','active_samples',
                'stall_samples','first_step_valid','mean_vx','mean_speed','max_abs_y','max_abs_yaw_deg']})
        records.append(row)
    source_sha=recovery.digest(frozen_sources)
    expected=dict(update=50,phase='R',substage=None,generation=0,kind='diagnostic',round=None,
        model_sha256=model.state_sha256(network.state_dict()),
        optimizer_sha256=recovery.digest(evaluation._optimizer_digest(optimizer.state_dict())),
        source_closure_sha256=source_sha,contract_sha256=spec.contract_sha256(),case_table_sha256=table.sha256)
    identity=deepcopy(expected)
    if change=='arbitrary_raw':
        for row in records:row['first_episode']={'arbitrary_non_terminal_evidence':1}
    elif change=='foreign_model':identity['model_sha256']='0'*64
    elif change=='foreign_source':identity['source_closure_sha256']='0'*64
    elif change=='missing_optimizer':identity.pop('optimizer_sha256')
    elif change=='wrong_optimizer':identity['optimizer_sha256']='0'*64
    elif change=='wrong_env':records[0]['first_episode']['env_id']=1
    elif change=='wrong_tile':records[0]['first_episode']['tile_id']=22
    elif change=='wrong_step':records[0]['first_episode']['control_samples']=4
    elif change=='wrong_outcome':records[0]['first_episode']['success']=True
    elif change=='wrong_command':records[0]['first_episode']['command']=.8
    elif change=='wrong_pose':records[0]['first_episode']['initial_pose']=[0.,0.,0.]
    elif change=='wrong_generation':records[0]['first_episode']['generation']=99
    snapshot=dict(model=model.state_sha256(network.state_dict()),
        optimizer=recovery.digest(evaluation._optimizer_digest(optimizer.state_dict())),counters=learning)
    isolation=evaluation.compare_snapshots(snapshot,snapshot)
    summary=evaluation.summarize(records,table);summary.update(isolation)
    hashes=dict(case_table_sha256=table.sha256,records_sha256=recovery.digest(records),summary_sha256=recovery.digest(summary))
    result=evaluation.EvalResult(table,summary,records,isolation,hashes)
    payload=evaluation_records.pack_result(result,identity)
    binding=evaluation_records.write_sidecar(tmp_path/'observed_diagnostic50.json',payload)
    row=dict(update=50,phase='R',substage=None,evidence=deepcopy(hashes),case_records=binding)
    if change=='wrong_main_evidence':row['evidence']['records_sha256']='0'*64
    runner=SimpleNamespace(update=50,phase='R',substage=None,seed=42,model=network,
        ppo=SimpleNamespace(optimizer=optimizer,learning=learning),
        source=dict(source_closure_sha256=source_sha),
        curriculum=SimpleNamespace(state_dict=lambda:dict(generation=0)),diagnostics=[row])
    observer=worker.capture_probe50_boundary(runner,tmp_path)
    return runner,binding,observer,expected


def test_probe_uses_real_model_adam_capture50_even_after_current_model_changes_to55(tmp_path,monkeypatch):
    import torch
    from src.tasks.velocity.v07 import model
    worker=import_module('train_a2_ideal_stairs_v07')
    runner,binding,observer,expected=live_probe50_fixture(tmp_path,monkeypatch)
    with torch.no_grad():runner.model.weight.add_(1.)
    runner.update=55
    assert model.state_sha256(runner.model.state_dict())!=expected['model_sha256']
    proof=worker.probe_diagnostic_evidence(runner)
    assert proof['identity']==expected
    assert proof['identity']['optimizer_sha256'] and proof['boundary50']==observer
    assert proof['case_count']==15 and proof['case_control_steps']==60
    assert proof['actual_eval_control_samples']==120 and proof['extra_eval_control_samples']==60


@pytest.mark.parametrize('change',['arbitrary_raw','foreign_model','foreign_source','missing_optimizer',
    'wrong_optimizer','wrong_env','wrong_tile','wrong_step','wrong_outcome','wrong_command','wrong_pose',
    'wrong_generation','wrong_main_evidence'])
def test_probe_rejects_corrupt_raw_or_identity_despite_valid_hashes_and_real_p50_capture(tmp_path,monkeypatch,change):
    from src.tasks.velocity.v07 import evaluation,orchestration
    worker=import_module('train_a2_ideal_stairs_v07')
    runner,_,_,_=live_probe50_fixture(tmp_path,monkeypatch,change)
    with pytest.raises((evaluation.EvaluationError,orchestration.ProbeError)):
        worker.probe_diagnostic_evidence(runner)
