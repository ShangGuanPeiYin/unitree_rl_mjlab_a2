"""Native worker routes real current case tables and bills engineering recovery."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import evaluation, orchestration, spec

CONTRACT=dict(num_envs=512,eval_num_envs=8,device='cpu',seed=42)


def test_restored_worker_closure_routes_d22_and_retention_current_cases():
    worker=import_module('train_a2_ideal_stairs_v07')
    for phase,substage,kind,want_phase,want_substage,want_tile in [
        ('D','D22','validation','D','D22',22),
        ('D','D20','validation','D','D20',18),
        ('D','D22','flat_retention','F0',None,0),
        ('D','D22','stair_retention','H',None,1),
        ('R',None,'test','R',None,21)]:
        table=evaluation.make_cases(phase,substage,kind,42)
        cases=deepcopy(table.cases[:2]); before=deepcopy(cases)
        out=worker.case_environment_kwargs(CONTRACT,cases,'D','D25')
        assert (out['phase'],out['substage'],out['frontier_tile_id'])==(want_phase,want_substage,want_tile)
        assert out['fixed_vx']==.3 and out['num_envs']==2
        assert cases==before and out['initial_cases']==before


@pytest.mark.parametrize('change', ['tile','speed','kind','lesson','substage'])
def test_router_rejects_corrupted_second_case(change):
    worker=import_module('train_a2_ideal_stairs_v07')
    cases=deepcopy(evaluation.make_cases('D','D22','validation',42).cases[:2])
    key,value={'tile':('tile_id',23),'speed':('vx',.8),'kind':('kind','flat_retention'),
               'lesson':('lesson_id','D20'),'substage':('substage','D25')}[change]
    cases[1][key]=value
    with pytest.raises(ValueError):
        worker.case_environment_kwargs(CONTRACT,cases,'D','D25')


def test_probe_cumulative_55_plus_one_counts_are_validated():
    worker=import_module('train_a2_ideal_stairs_v07')
    plan=orchestration.probe_plan(seed=42)
    before=dict(update=55,control_samples=1_351_680,physics_substeps=5_406_720,adam_steps=1100)
    after=dict(update=56,control_samples=1_376_256,physics_substeps=5_505_024,adam_steps=1120)
    counts,evidence=worker.probe_counts(before,after,plan)
    assert counts['learning_updates']==56
    assert counts['counts_against_training'] is False
    assert counts['training_budget_untouched']==48000
    assert evidence['delta']==dict(control_samples=24576,physics_substeps=98304,adam_updates=20)
    after['update']=57
    with pytest.raises(orchestration.ProbeError):
        worker.probe_counts(before,after,plan)


def test_engineering_probe_curriculum_uses_l40_not_largest_physical_id():
    worker=import_module('train_a2_ideal_stairs_v07')
    manager=worker.probe_curriculum(orchestration.probe_plan(seed=42),42)
    assert (manager.state.phase,manager.state.substage,manager.state.frontier_tile_id)==('R',None,21)
    assert manager.state.generation_updates==0
    assert manager.state.retention_enabled==dict(flat=False,h00=False)
    assert sum(manager.state.spent_by_phase.values())==0


def test_probe_case_runner_persists_actual_batch_resource_and_cost_evidence(tmp_path,monkeypatch):
    worker=import_module('train_a2_ideal_stairs_v07')
    import validate_a2_ideal_stairs_v07 as engine
    snapshot=dict(whole_device_mib=1024,process_mib=512,host_available_mib=8192,disk_free_gib=20.)
    monkeypatch.setattr(orchestration,'resource_snapshot',lambda **kwargs:dict(snapshot))
    rows=[]
    records=[dict(case_id='diagnostic:0.3:000',steps=3,physical_control_samples=8,
        physical_physics_substeps=32,extra_control_samples_after_case=5,
        extra_physics_substeps_after_case=20)]
    def simulated_batch(cases):return deepcopy(records)
    run=worker.probe_case_runner(simulated_batch,tmp_path,rows,lambda:50)
    result=run([dict(case_id='diagnostic:0.3:000')])
    assert result==records
    assert rows[0]['update']==50 and rows[0]['cost']['actual_eval_control_samples']==8
    raw=__import__('json').loads(Path(rows[0]['resource_binding']['path']).read_text())
    assert raw['snapshot']==snapshot and raw['check']['ok'] is True
    snapshot['host_available_mib']=4095
    with pytest.raises(RuntimeError):run([dict(case_id='diagnostic:0.3:000')])
