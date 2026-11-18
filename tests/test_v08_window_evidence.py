from copy import deepcopy
import json
import pytest
from test_v08_evaluation import window

def identity(table):
    return dict(update=6500,phase=table.phase,kind=table.kind,substage=None,case_table_sha256=table.sha256,source_sha256='a'*64,recipe_sha256='b'*64)

def test_actual_eval_result_roundtrip_atomic_sidecar_and_tampering(tmp_path):
    from src.tasks.velocity.v08 import evaluation as e, evaluation_records as r
    table=e.make_cases(kind='baseline')
    result=e.evaluate_cases(None,table,lambda c:dict(case_id=c['case_id'],complete=True,raw_complete=True,finite=True,success=False,reason='fall',segment='cruise',failure_kind='physical'))
    frozen=r.pack_result(result,identity(table))
    loaded=r.restore_result(frozen,identity(table))
    assert loaded.records==result.records and loaded.summary==result.summary
    binding=r.write_sidecar(tmp_path/'result.json',frozen)
    assert r.verify_sidecar(binding,frozen)
    assert r.write_sidecar(tmp_path/'result.json',frozen)==binding
    wrong=deepcopy(frozen);wrong['records'][0]['status']='success'
    with pytest.raises(e.EvaluationError):r.restore_result(wrong,identity(table))
    changed=deepcopy(frozen);changed['identity']['update']+=100
    with pytest.raises(e.EvaluationError):r.write_sidecar(tmp_path/'result.json',changed)
    (tmp_path/'result.json').write_text('{}')
    with pytest.raises(e.EvaluationError):r.verify_sidecar(binding,frozen)

def test_window_preserves_exact_samples_and_rejects_fabrication(tmp_path):
    from src.tasks.velocity.v08 import window_evidence as w
    ident=dict(case_id='frozen-case',phase='E',update=6500,window_index=0)
    payload=w.pack_window(window('mid'),ident)
    binding=w.write_window(tmp_path/'window.json',payload)
    assert w.verify_window(binding,payload)
    changed=deepcopy(payload);changed['window']['values'][0][5]=0.
    with pytest.raises(ValueError):w.validate_window(changed)
    wrong=window('mid');wrong['values']=wrong['values'][:-1]
    with pytest.raises(ValueError):w.pack_window(wrong,ident)

def test_failed_partial_is_preserved_and_final_conflict_never_overwritten(tmp_path):
    from src.tasks.velocity.v08 import evaluation as e, evaluation_records as r
    table=e.make_cases(kind='baseline')
    result=e.evaluate_cases(None,table,lambda c:None)
    payload=r.pack_result(result,identity(table));path=tmp_path/'result.json'
    partial=tmp_path/'result.json.partial';partial.write_text('{bad')
    r.write_sidecar(path,payload)
    assert (tmp_path/'result.json.partial.failed.0000').read_text()=='{bad'
    assert not partial.exists()


def test_sidecar_restore_recomputes_raw_trace_even_when_cached_hashes_are_rebound():
    from src.tasks.velocity.v08 import evaluation as e, evaluation_records as r
    from test_v08_evaluation import direction_raw
    table=e.make_cases(phase='E')
    selected=next(c for c in table.cases if c['task']=='E1')
    def runner(case):
        if case['case_id']==selected['case_id']:return direction_raw(case)
        return dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=False,reason='fall',segment='cruise',failure_kind='physical')
    result=e.evaluate_cases(None,table,runner)
    assert result.summary['success']==1
    payload=r.pack_result(result,identity(table))
    row=next(row for row in payload['records'] if row['case_id']==selected['case_id'])
    for frame in row['raw']['trace']:frame['local_x']=4.
    # Rehash the edited payload while preserving its false cached success.
    payload['evidence_sha256']['records_sha256']=e._digest(payload['records'])
    payload['summary']=e.summarize(payload['records'],table);payload['summary'].update(payload['isolation'])
    payload['evidence_sha256']['summary_sha256']=e._digest(payload['summary'])
    with pytest.raises(e.EvaluationError,match='raw evidence'):
        r.restore_result(payload,identity(table))
