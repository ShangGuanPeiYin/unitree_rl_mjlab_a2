"""Exact physical stop-window sidecars; no adaptive FIFO or v07 EMA decisions."""
from copy import deepcopy
import hashlib
from pathlib import Path
from . import evaluation as e
from .evaluation_records import _install, _matches

SCHEMA='v08-stop-window-v1'

def validate_window(payload):
    if not isinstance(payload,dict) or set(payload)!={'schema','identity','window','window_sha256'} or payload['schema']!=SCHEMA:raise e.EvaluationError('invalid stop-window envelope')
    ident=payload['identity']
    if not isinstance(ident,dict) or not {'case_id','phase','update','window_index'}<=set(ident):raise e.EvaluationError('missing stop-window identity')
    if not isinstance(ident['case_id'],str) or not ident['case_id'] or ident['phase'] not in ('A','B','C','D','E'):raise e.EvaluationError('invalid stop-window case/phase')
    if any(type(ident[k]) is not int or ident[k]<0 for k in ('update','window_index')):raise e.EvaluationError('invalid stop-window boundary')
    window=payload['window']
    if payload['window_sha256']!=e._digest(window):raise e.EvaluationError('stop-window raw SHA mismatch')
    # Failed physical windows are equally valid evidence; validity is not success.
    try:e._stop_window(window)
    except (KeyError,TypeError,ValueError,IndexError) as error:raise e.EvaluationError('invalid raw stop-window samples') from error
    return True

def pack_window(window,identity):
    payload=dict(schema=SCHEMA,identity=deepcopy(identity),window=deepcopy(window),window_sha256=e._digest(window))
    validate_window(payload);return payload

def _binding(path,payload):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),window_sha256=payload['window_sha256'],identity=deepcopy(payload['identity']))

def write_window(path,payload):
    _install(path,payload,validate_window);return _binding(path,payload)

def verify_window(binding,payload):
    validate_window(payload)
    if not isinstance(binding,dict) or set(binding)!={'path','sha256','window_sha256','identity'}:raise e.EvaluationError('invalid stop-window binding')
    _matches(binding['path'],payload,validate_window)
    if _binding(binding['path'],payload)!=binding:raise e.EvaluationError('bound stop-window SHA differs')
    return True
