from copy import deepcopy

def test_final_subgates_and_full_denominator_missing_cases():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(kind='final')
    rows=[dict(case_id=c['case_id'],status='success',success=True,reason='success',segment='hold',task=c['task'],script=c['script'],speed=c['speed']) for c in table.cases]
    assert e.summarize(rows,table)['gate_pass']
    speed=.3
    selected=[x for x in rows if x['task']=='E3' and x['script']=='up_reverse' and x['speed']==speed]
    for row in selected[:3]:row.update(status='failure',success=False,reason='wrong_direction')
    summary=e.summarize(rows,table)
    assert not summary['gate_pass']
    assert summary['groups']['E3@0.3']['success']==97
    assert summary['e3_subgates']['up_reverse@0.3']['success']==22
    missing=e.summarize(rows[:-1],table)
    assert missing['expected']==2100 and missing['received']==2099 and missing['invalid']==1

def test_duplicate_or_unknown_ids_are_engineering_errors():
    from src.tasks.velocity.v08 import evaluation as e
    t=e.make_cases(kind='baseline')
    row=dict(case_id=t.cases[0]['case_id'],status='failure',success=False,reason='fall',segment='walk')
    s=e.summarize([row,row,dict(row,case_id='foreign')],t)
    assert s['engineering_error'] and s['duplicate_ids'] and s['unknown_ids']==['foreign']
