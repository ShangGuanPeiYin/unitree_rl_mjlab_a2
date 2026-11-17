"""The restored D25 closure must never leak into another case's physical stage."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import a2_v06_bootstrap
a2_v06_bootstrap.activate(Path(__file__).resolve().parents[1])
import train_a2_ideal_stairs_v06 as worker
from src.tasks.velocity.v06 import evaluation,spec

CONTRACT=dict(num_envs=512,eval_num_envs=8,device='cpu',seed=42)
CONTEXTS=[('F0',None),('H',None),('D','D25'),('D','D20'),('L','L12'),('L','L20'),('L','L40'),('V',None),('R',None)]


def table(phase,substage,kind):
    extra=dict(tile_id=spec.tile_id_for('H15')) if phase=='H' and kind not in ('flat_retention','stair_retention') else {}
    return evaluation.make_cases(phase,substage,kind,42,**extra)


def batch(value):return deepcopy(value.cases[:2])


@pytest.mark.parametrize('phase,substage',CONTEXTS)
def test_long_lived_d25_closure_routes_ordinary_cases_from_their_actual_tile(phase,substage):
    value=table(phase,substage,'validation');cases=batch(value);before=deepcopy(cases)
    kwargs=worker.case_environment_kwargs(CONTRACT,cases,'D','D25')
    assert kwargs['phase']==phase
    assert kwargs['substage']==substage
    assert kwargs['frontier_tile_id']==value.target_tile_id
    assert kwargs['initial_cases']==before and cases==before


@pytest.mark.parametrize('phase,substage',CONTEXTS)
@pytest.mark.parametrize('kind,physical,target',[('flat_retention','F0','FLAT'),('stair_retention','H','H00')])
def test_both_retention_sets_ignore_the_restored_d25_substage(phase,substage,kind,physical,target):
    cases=batch(table(phase,substage,kind))
    kwargs=worker.case_environment_kwargs(CONTRACT,cases,'D','D25')
    assert (kwargs['phase'],kwargs['substage'],kwargs['frontier_tile_id'])==(physical,None,spec.tile_id_for(target))


@pytest.mark.parametrize('change',['missing_tile','missing_phase','missing_lesson','wrong_lesson','mixed_kind',
    'mixed_phase','mixed_measurement','mixed_speed','explicit_wrong_substage','fractional_tile','bool_tile'])
def test_every_case_is_checked_not_just_the_first(change):
    cases=batch(table('D','D20','validation'))
    if change=='missing_tile':cases[1].pop('tile_id')
    elif change=='missing_phase':cases[1].pop('phase')
    elif change=='missing_lesson':cases[1].pop('lesson_id')
    elif change=='wrong_lesson':cases[1]['lesson_id']='D25'
    elif change=='mixed_kind':cases[1]['kind']='diagnostic'
    elif change=='mixed_phase':cases[1]['phase']='R'
    elif change=='mixed_measurement':cases[1]['measurement_stage']='F0'
    elif change=='mixed_speed':cases[1]['vx']=.5
    elif change=='explicit_wrong_substage':cases[1]['substage']='D25'
    elif change=='fractional_tile':cases[1]['tile_id']=18.5
    elif change=='bool_tile':cases[1]['tile_id']=True
    with pytest.raises(ValueError):worker.case_environment_kwargs(CONTRACT,cases,'D','D25')


def test_final_and_explicit_wrong_retention_substage():
    value=evaluation.make_cases('R',None,'test',42)
    kwargs=worker.case_environment_kwargs(CONTRACT,batch(value),'D','D25')
    assert (kwargs['phase'],kwargs['substage'],kwargs['frontier_tile_id'])==('R',None,spec.FINAL_TILE_ID)
    cases=batch(table('D','D25','flat_retention'));cases[0]['substage']='D25'
    with pytest.raises(ValueError):worker.case_environment_kwargs(CONTRACT,cases,'D','D25')


def test_restored_double_retention_constructs_real_environments():
    from src.tasks.velocity.v06.env import V06Env
    for kind in ('flat_retention','stair_retention'):
        cases=batch(table('D','D25',kind))
        kwargs=worker.case_environment_kwargs(CONTRACT,cases,'D','D25')
        env=V06Env(device='cpu',seed=42,**{k:v for k,v in kwargs.items() if k!='contract'})
        try:
            env.reset()
            assert env.substage is None and set(env.tile_ids.tolist())=={kwargs['frontier_tile_id']}
        finally:
            if hasattr(env,'close'):env.close()


def test_one_actual_long_lived_worker_closure_routes_all_later_calls(monkeypatch):
    class Constructed(Exception):pass
    seen=[]
    def inspect_environment(contract,**kwargs):
        seen.append((kwargs['phase'],kwargs['substage'],kwargs['frontier_tile_id']))
        # Route admission is under test; avoid running an entire GPU/CPU episode.
        raise Constructed
    monkeypatch.setattr(worker,'build_environment',inspect_environment)
    run=worker.build_case_runner(object(),CONTRACT,'D','D25')
    calls=[('D','D20','validation','D','D20'),('L','L12','validation','L','L12'),
           ('L','L20','validation','L','L20'),('L','L40','validation','L','L40'),
           ('V',None,'validation','V',None),('R',None,'validation','R',None),
           ('D','D25','flat_retention','F0',None),('D','D25','stair_retention','H',None)]
    expected=[]
    for phase,substage,kind,physical,physical_substage in calls:
        value=table(phase,substage,kind)
        with pytest.raises(Constructed):run(batch(value))
        expected.append((physical,physical_substage,value.target_tile_id))
    assert seen==expected
