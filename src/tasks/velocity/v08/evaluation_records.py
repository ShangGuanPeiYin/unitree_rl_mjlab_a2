"""Immutable v0.8 result envelopes: actual case/raw/summary hashes and durable I/O.

Files are installed without replacement. Retries reuse identical evidence and
retain malformed partial writes; they never select or regenerate policy results.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from . import evaluation as e

SCHEMA='v08-first-episode-result-v1'
_HASH_KEYS={'case_table_sha256','records_sha256','summary_sha256'}
_KEYS={'schema','identity','table','records','summary','isolation','evidence_sha256'}

def _plain(value): return json.loads(e._canonical(value))
def _digest(value): return e._digest(value)
def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)

def restore_result(payload,expected_identity):
    try:
        if not isinstance(payload,dict) or set(payload)!=_KEYS or payload['schema']!=SCHEMA: raise e.EvaluationError('invalid v08 result envelope')
        identity=payload['identity']
        if not isinstance(expected_identity,dict) or not expected_identity or e._canonical(identity)!=e._canonical(expected_identity): raise e.EvaluationError('cached result identity mismatch')
        required={'update','phase','kind','case_table_sha256','source_sha256','recipe_sha256'}
        if not required<=set(identity) or type(identity['update']) is not int or identity['update']<6400: raise e.EvaluationError('result identity missing update/source/recipe/cases')
        if any(not _sha(identity[k]) for k in ('case_table_sha256','source_sha256','recipe_sha256')): raise e.EvaluationError('invalid identity SHA')
        hashes=payload['evidence_sha256']
        if not isinstance(hashes,dict) or set(hashes)!=_HASH_KEYS or not all(_sha(v) for v in hashes.values()): raise e.EvaluationError('invalid evidence hashes')
        table=e.table_from_json(payload['table'],expected_sha=hashes['case_table_sha256'])
        if identity['case_table_sha256']!=table.sha256 or identity['phase']!=table.phase or identity['kind']!=table.kind: raise e.EvaluationError('result identity differs from frozen table')
        records,summary,isolation=_plain(payload['records']),_plain(payload['summary']),_plain(payload['isolation'])
        if not isinstance(records,list) or not isinstance(summary,dict) or not isinstance(isolation,dict): raise e.EvaluationError('invalid result blocks')
        if set(isolation)!={'statistics_unchanged','statistics_changed','snapshot_sha'} or isolation['statistics_unchanged'] is not True or isolation['statistics_changed']!=[] or not _sha(isolation['snapshot_sha']): raise e.EvaluationError('evaluation isolation was not verified')
        if _digest(records)!=hashes['records_sha256'] or _digest(summary)!=hashes['summary_sha256']: raise e.EvaluationError('result evidence SHA mismatch')
        cases={c['case_id']:c for c in table.cases}
        for row in records:
            if row.get('case_id') not in cases: raise e.EvaluationError('unknown cached case')
            if 'raw' in row:
                normalized=e.normalize_record(row['raw'],cases[row['case_id']])
                if e._canonical(normalized)!=e._canonical(row): raise e.EvaluationError('cached outcome differs from actual raw evidence')
            elif row.get('status')!='invalid' or row.get('reason') not in ('missing_raw','nonfinite_raw'):
                raise e.EvaluationError('cached result has no auditable raw evidence')
        recomputed=e.summarize(records,table);recomputed.update(isolation)
        if e._canonical(recomputed)!=e._canonical(summary): raise e.EvaluationError('cached summary differs from actual rows')
        return e.EvalResult(table,summary,records,isolation,deepcopy(hashes))
    except (KeyError,TypeError,AttributeError) as error: raise e.EvaluationError('invalid cached v08 result: '+str(error)) from error

def pack_result(result,identity):
    payload=_plain(dict(schema=SCHEMA,identity=identity,table=result.table.to_json(),records=result.records,
                        summary=result.summary,isolation=result.isolation,evidence_sha256=result.evidence_sha256))
    restore_result(payload,identity)
    return payload

def _preserve_incomplete(path):
    for index in range(10000):
        archive=path.with_name(path.name+f'.failed.{index:04d}')
        try:os.link(path,archive)
        except FileExistsError:continue
        path.unlink();return
    raise e.EvaluationError('too many retained partial attempts')

def _sync_file_and_directory(path):
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)

def _matches(path,payload,validate):
    try:existing=json.loads(Path(path).read_text())
    except (ValueError,UnicodeError) as error:raise e.EvaluationError('invalid existing evidence') from error
    validate(existing)
    if e._canonical(existing)!=e._canonical(payload):raise e.EvaluationError('conflicting immutable evidence')

def _binding(path,payload):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),**deepcopy(payload['evidence_sha256']))

def _install(path,payload,validate):
    """Durable no-replace JSON; shared by results and exact raw stop windows."""
    validate(payload);path=Path(path)
    if path.exists():
        _matches(path,payload,validate);_sync_file_and_directory(path);return
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.name+'.partial')
    if partial.exists():
        try:json.loads(partial.read_text())
        except (ValueError,UnicodeError):_preserve_incomplete(partial)
        else:_matches(partial,payload,validate)
    if not partial.exists():
        with partial.open('xb') as stream:
            stream.write((e._canonical(payload)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    _sync_file_and_directory(partial)
    try:os.link(partial,path)
    except FileExistsError:_matches(path,payload,validate)
    _sync_file_and_directory(path);_matches(path,payload,validate);partial.unlink()

def write_sidecar(path,payload):
    _install(path,payload,lambda p:restore_result(p,p.get('identity')))
    return _binding(path,payload)

def verify_sidecar(binding,payload):
    if not isinstance(binding,dict) or set(binding)!={'path','sha256',*_HASH_KEYS}:raise e.EvaluationError('invalid sidecar binding')
    _matches(binding['path'],payload,lambda p:restore_result(p,p.get('identity')))
    if _binding(binding['path'],payload)!=binding:raise e.EvaluationError('bound sidecar SHA differs')
    return True
