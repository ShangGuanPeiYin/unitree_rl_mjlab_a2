"""Q010 durable result data must preserve first evidence and reject corruption."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import a2_v06_bootstrap
a2_v06_bootstrap.activate(Path(__file__).resolve().parents[1])
from src.tasks.velocity.v06 import evaluation


def helper():
    try:return importlib.import_module('src.tasks.velocity.v06.evaluation_records')
    except ImportError:pytest.fail('Q010 durable evaluation result helper is missing')


def digest(value):
    return hashlib.sha256(evaluation._canonical(value).encode()).hexdigest()


def result():
    table=evaluation.make_cases('D','D25','validation',42)
    records=[dict(case_id=c['case_id'],vx=c['vx'],climb_success=True,failure=False,
        mean_vx=c['vx'],reason='success',measurement='stair',steps=100,
        first_episode=dict(env_id=i%8,terminal_step=100,mean_vx=c['vx']))
        for i,c in enumerate(table.cases)]
    isolation=evaluation.compare_snapshots({'model':'x'},{'model':'x'})
    summary=evaluation.summarize(records,table);summary.update(isolation)
    evidence=dict(case_table_sha256=table.sha256,records_sha256=digest(records),summary_sha256=digest(summary))
    return evaluation.EvalResult(table,summary,records,isolation,evidence)


IDENTITY=dict(update=4400,phase='D',substage='D25',generation=68,kind='validation',
              model_sha256='b'*64,source_closure_sha256='c'*64,contract_sha256='d'*64,
              case_table_sha256=evaluation.make_cases('D','D25','validation',42).sha256)


def test_roundtrip_is_immutable_json_and_preserves_literal_twenty_per_speed():
    io=helper();value=result();payload=io.pack_result(value,IDENTITY)
    value.records[0]['first_episode']['mean_vx']=999
    restored=io.restore_result(json.loads(json.dumps(payload)),IDENTITY)
    assert restored.records[0]['first_episode']['mean_vx']==.3
    assert [x['case_pass_count'] for x in restored.summary['per_speed']]==[20,20,20]
    assert restored.gate['gate_pass'] is True
    assert restored.evidence_sha256==payload['evidence_sha256']


@pytest.mark.parametrize('change',['identity','record_sha','summary_with_valid_hash','record_with_valid_hash','table','isolation','schema'])
def test_corruption_never_uses_cached_passed_or_falls_back(change):
    io=helper();payload=io.pack_result(result(),IDENTITY);expected=deepcopy(IDENTITY)
    if change=='identity':expected['model_sha256']='e'*64
    elif change=='record_sha':payload['evidence_sha256']['records_sha256']='0'*64
    elif change=='summary_with_valid_hash':
        payload['summary']['per_speed'][0]['case_pass_count']=19
        payload['evidence_sha256']['summary_sha256']=digest(payload['summary'])
    elif change=='record_with_valid_hash':
        payload['records'][0].update(climb_success=False,failure=True)
        payload['evidence_sha256']['records_sha256']=digest(payload['records'])
    elif change=='table':payload['table']['arrays']['0.3']['x'][0]+=.01
    elif change=='isolation':payload['isolation']['statistics_unchanged']=False
    elif change=='schema':payload['schema']='unknown'
    with pytest.raises(evaluation.EvaluationError):io.restore_result(payload,expected)


def test_sidecar_is_atomic_idempotent_rng_neutral_and_conflicts_do_not_overwrite(tmp_path,monkeypatch):
    io=helper();payload=io.pack_result(result(),IDENTITY);path=tmp_path/'case_records.json'
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    binding=io.write_sidecar(path,payload);original=path.read_bytes();mtime=path.stat().st_mtime_ns
    assert binding['records_sha256']==payload['evidence_sha256']['records_sha256']
    assert binding['sha256']==hashlib.sha256(original).hexdigest()
    assert io.write_sidecar(path,payload)==binding and path.stat().st_mtime_ns==mtime
    assert random.getstate()==before[0]
    now=np.random.get_state();assert now[0]==before[1][0] and np.array_equal(now[1],before[1][1]) and now[2:]==before[1][2:]
    assert torch.equal(torch.get_rng_state(),before[2])
    other=io.pack_result(result(),dict(IDENTITY,update=4500))
    with pytest.raises(evaluation.EvaluationError):io.write_sidecar(path,other)
    assert path.read_bytes()==original


def test_atomic_install_failure_preserves_partial_and_reuses_exact_cached_result(tmp_path,monkeypatch):
    io=helper();payload=io.pack_result(result(),IDENTITY);path=tmp_path/'records.json'
    original=io.os.link
    def fail(*args,**kwargs):raise OSError('injected atomic install failure')
    monkeypatch.setattr(io.os,'link',fail)
    with pytest.raises(OSError):io.write_sidecar(path,payload)
    assert not path.exists() and list(tmp_path.glob('records.json.partial*'))
    monkeypatch.setattr(io.os,'link',original)
    binding=io.write_sidecar(path,payload)
    assert binding['records_sha256']==payload['evidence_sha256']['records_sha256']
    assert io.restore_result(json.loads(path.read_text()),IDENTITY).gate['gate_pass'] is True


def test_incomplete_old_partial_is_preserved_without_becoming_a_completed_result(tmp_path):
    io=helper();payload=io.pack_result(result(),IDENTITY);path=tmp_path/'records.json'
    partial=tmp_path/'records.json.partial';partial.write_bytes(b'{"unfinished":')
    io.write_sidecar(path,payload)
    preserved=list(tmp_path.glob('records.json.partial.failed.*'))
    assert len(preserved)==1 and preserved[0].read_bytes()==b'{"unfinished":'
    assert io.restore_result(json.loads(path.read_text()),IDENTITY).gate['gate_pass'] is True


def test_actual_installed_bytes_are_revalidated_before_a_binding_is_returned(tmp_path,monkeypatch):
    io=helper();payload=io.pack_result(result(),IDENTITY);path=tmp_path/'records.json'
    original=io.os.link
    def corrupted_install(source,destination):
        original(source,destination)
        Path(destination).write_text('{"corrupted":true}')
    monkeypatch.setattr(io.os,'link',corrupted_install)
    with pytest.raises(evaluation.EvaluationError):io.write_sidecar(path,payload)
    assert path.exists() and list(tmp_path.glob('records.json.partial*'))


@pytest.mark.parametrize('extra',['forged_gate','extra_key','missing_snapshot','bad_snapshot'])
def test_isolation_cannot_override_recomputed_outcomes_even_with_valid_hashes(extra):
    io=helper();payload=io.pack_result(result(),IDENTITY)
    if extra=='forged_gate':
        for row in payload['records']:row.update(climb_success=False,failure=True,reason='nonfoot')
        payload['evidence_sha256']['records_sha256']=digest(payload['records'])
        payload['isolation'].update(per_speed=deepcopy(payload['summary']['per_speed']),gate_pass=True)
        payload['summary'].update(payload['isolation'])
    elif extra=='extra_key':
        payload['isolation']['untrusted_extra']='x';payload['summary']['untrusted_extra']='x'
    elif extra=='missing_snapshot':
        payload['isolation'].pop('snapshot_sha');payload['summary'].pop('snapshot_sha')
    else:
        payload['isolation']['snapshot_sha']='bad';payload['summary']['snapshot_sha']='bad'
    payload['evidence_sha256']['summary_sha256']=digest(payload['summary'])
    with pytest.raises(evaluation.EvaluationError):io.restore_result(payload,IDENTITY)


def test_valid_other_task_table_cannot_be_bound_to_current_task_identity():
    io=helper()
    for table in [evaluation.make_cases('D','D20','validation',42),
                  evaluation.make_cases('D','D25','stair_retention',42)]:
        value=result();value.table=table
        value.records=[dict(case_id=c['case_id'],vx=c['vx'],climb_success=True,failure=False,
            mean_vx=c['vx'],reason='success',measurement='stair') for c in table.cases]
        value.summary=evaluation.summarize(value.records,table);value.summary.update(value.isolation)
        value.evidence_sha256=dict(case_table_sha256=table.sha256,
            records_sha256=digest(value.records),summary_sha256=digest(value.summary))
        identity=dict(IDENTITY,case_table_sha256=table.sha256)
        with pytest.raises(evaluation.EvaluationError):io.pack_result(value,identity)


@pytest.mark.parametrize('fail_target',['file','directory'])
def test_retry_repeats_the_failed_fsync_step_before_returning_binding(tmp_path,monkeypatch,fail_target):
    import os
    io=helper();payload=io.pack_result(result(),IDENTITY);path=tmp_path/'retry.json'
    original=io.os.fsync;calls=[];failed={'done':False}
    def injected(fd):
        target=os.readlink(f'/proc/self/fd/{fd}')
        category='directory' if Path(target).is_dir() else 'file'
        calls.append(category)
        if category==fail_target and not failed['done']:
            failed['done']=True;raise OSError('injected '+fail_target+' fsync failure')
        return original(fd)
    monkeypatch.setattr(io.os,'fsync',injected)
    with pytest.raises(OSError):io.write_sidecar(path,payload)
    before=len([x for x in calls if x==fail_target])
    io.write_sidecar(path,payload)
    assert len([x for x in calls if x==fail_target])>before
    assert path.exists()
