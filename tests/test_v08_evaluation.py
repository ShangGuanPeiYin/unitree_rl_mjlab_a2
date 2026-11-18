from copy import deepcopy
import random
import numpy as np
import pytest
import torch

def test_complete_denominator_missing_raw_nonfinite_and_physical_separate():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(kind='baseline')
    def runner(case):
        i=int(case['case_id'].rsplit('-',1)[1])
        row=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=False,reason='fall',segment='cruise',failure_kind='physical')
        if i==0: row['raw_complete']=False
        elif i==1: row['sample']=float('nan')
        elif i==2: raise RuntimeError('sensor')
        return row
    result=e.evaluate_cases(None,table,runner)
    assert result.summary['expected']==60 and result.summary['success']==0
    assert result.summary['invalid']>0 and result.summary['error']>0
    assert result.summary['physical_failure']>0 and not result.summary['gate_pass']
    assert sum(result.summary[k] for k in ('valid','invalid','error'))==60

def test_success_claim_cannot_hide_stationary_or_wrong_direction():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(phase='E')
    flat=next(x for x in table.cases if x['task']=='flat_stop')
    raw=dict(case_id=flat['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[window('flat')],flat_progress=dict(net_dx=0.,max_abs_y=0.,max_abs_yaw=0.,walk_steps=400))
    row=e.normalize_record(raw,flat)
    assert row['status']=='failure' and row['reason']=='insufficient_flat_progress'
    case=next(x for x in table.cases if x['task']=='E1')
    raw.update(case_id=case['case_id'],stop_windows=[window('bottom')],direction_checks=[dict(direction=-1,net_progress=-1.,elapsed_steps=200)],max_stair_abs_yaw=0.)
    row=e.normalize_record(raw,case)
    assert row['status']=='failure' and row['reason']=='wrong_direction'

def window(kind):
    n=150 if kind=='mid' else 500
    return dict(kind=kind,count=300 if kind=='mid' else 1000,xy=[[0.,0.]]*n,values=[[0.,0.,0.,0.,0.,.4,0.,1.,1.,1.]]*n)

def test_rng_and_model_mode_isolated_even_on_exception():
    from src.tasks.velocity.v08 import evaluation as e
    model=torch.nn.Linear(1,1);model.train()
    before=(random.getstate(),deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    def runner(case):
        random.random();np.random.random();torch.rand(1)
        raise RuntimeError('intentional')
    result=e.evaluate_cases(model,e.make_cases(kind='baseline'),runner)
    assert random.getstate()==before[0]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2]) and model.training
    assert result.isolation['statistics_unchanged']

def test_direction_evidence_binds_request_and_first_cruise_clock():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E1')
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[window('bottom')],max_stair_abs_yaw=0.,direction_checks=[dict(direction=-1,net_progress=.61,elapsed_steps=200,request_step=50,cruise_start_step=151,completion_step=250)])
    assert e.normalize_record(raw,case)['status']=='success'
    raw['direction_checks'][0]['cruise_start_step']=0
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw['direction_checks'][0].update(cruise_start_step=151,completion_step=450,elapsed_steps=200)
    assert e.normalize_record(raw,case)['status']=='invalid'

def test_summary_rejects_unrecognized_status():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(kind='baseline')
    with pytest.raises(e.EvaluationError,match='status'):
        e.summarize([dict(case_id=table.cases[0]['case_id'],status='magic',success=True,reason='success')],table)

def test_model_mutation_is_restored_and_blocks_evaluation():
    from src.tasks.velocity.v08 import evaluation as e
    model=torch.nn.Linear(1,1);before=deepcopy(model.state_dict())
    def runner(c):
        with torch.no_grad():model.weight.add_(1.)
        return None
    with pytest.raises(e.EvaluationError,match='mutated'):
        e.evaluate_cases(model,e.make_cases(kind='baseline'),runner)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())

def test_raw_window_binary_support_and_nonnegative_norms_are_required():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases().cases if c['task']=='up_stop')
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[window('top')])
    raw['stop_windows'][0]['values']=[list(x) for x in raw['stop_windows'][0]['values']]
    raw['stop_windows'][0]['values'][0][7]=4.
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw['stop_windows'][0]['values'][0][7]=1.
    raw['stop_windows'][0]['values'][0][0]=-1.
    assert e.normalize_record(raw,case)['status']=='invalid'


def test_direction_contract_only_reverse_e3_but_full_roundtrip_counts():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(phase='E')
    for c in table.cases:
        expected=c['task'] in ('E1','E2','E4') or (c['task']=='E3' and c['script'].endswith('reverse'))
        assert e.needs_direction_check(c)==expected
    case=next(c for c in table.cases if c['task']=='E3' and c['script']=='up_same')
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[window('mid'),window('top')],max_stair_abs_yaw=0.)
    assert e.normalize_record(raw,case)['status']=='success'
    case=next(c for c in table.cases if c['task']=='E4')
    raw.update(case_id=case['case_id'],stop_windows=[window('top'),window('bottom')])
    assert e.normalize_record(raw,case)['status']=='invalid'
