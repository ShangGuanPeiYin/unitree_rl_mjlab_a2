"""Immutable evidence for the exact 64 FIFO outcomes consumed by curriculum.

No replay, sampling or EMA calculation occurs here. Installation is durable and
idempotent; failure leaves the same raw rows available for exception recovery.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

from . import spec
from .evaluation_records import _sync_file_and_directory, _preserve_incomplete

SCHEMA='v07-frontier-window-v1'
_KEYS={'schema','identity','records','records_sha256','previous_ema','diagnostic_only'}
_ROW={'env_id','step','role','tile_id','success','failure','timeout','reason','tracking_score','command','generation'}

def _canonical(value):return spec.canonical_json(value)
def _digest(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()
def _integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError('invalid window '+name)
    return value

def validate_window(payload):
    if not isinstance(payload,dict) or set(payload)!=_KEYS or payload['schema']!=SCHEMA:
        raise ValueError('unknown or incomplete consumed-window envelope')
    identity=payload['identity']
    if not isinstance(identity,dict) or set(identity)!={'update','phase','substage','generation','window_index'}:
        raise ValueError('incomplete consumed-window identity')
    for key in ('update','generation','window_index'):_integer(identity[key],key,1 if key!='generation' else 0)
    phase=identity['phase'];sub=identity['substage']
    if phase not in spec.PHASE_ORDER:raise ValueError('unknown window phase')
    if phase in ('D','L'):
        if sub not in (spec.D_SUBSTAGES if phase=='D' else spec.L_SUBSTAGES):raise ValueError('invalid window substage')
        tile=int(spec.tile_id_for(sub))
    else:tile=None
    if payload['diagnostic_only'] is not (phase in ('D','L')):raise ValueError('wrong window decision classification')
    rows=payload['records']
    if not isinstance(rows,list) or len(rows)!=int(spec.CURRICULUM_WINDOW):raise ValueError('window must contain the exact 64 outcomes')
    previous=None
    for row in rows:
        if not isinstance(row,dict) or set(row)!=_ROW:raise ValueError('incomplete normalized window row')
        for key in ('env_id','step','tile_id','generation'):_integer(row[key],key)
        if row['generation']!=identity['generation'] or row['role']!='frontier':raise ValueError('window contains a different generation or role')
        spec.terrain_spec(row['tile_id'])
        if tile is not None and row['tile_id']!=tile:raise ValueError('window tile differs from measured substage')
        if tile is None:tile=row['tile_id']
        if row['tile_id']!=tile:raise ValueError('mixed frontier window tiles')
        if any(type(row[key]) is not bool for key in ('success','failure','timeout')) or sum(row[key] for key in ('success','failure','timeout'))!=1:
            raise ValueError('window outcome flags are not exclusive terminal flags')
        if not isinstance(row['reason'],str) or not row['reason']:raise ValueError('window reason missing')
        for key in ('tracking_score','command'):
            if isinstance(row[key],bool) or not isinstance(row[key],(float,int)) or not math.isfinite(row[key]):raise ValueError('nonfinite window '+key)
        order=(row['step'],row['env_id'])
        if previous is not None and order<previous:raise ValueError('window is not in actual FIFO order')
        previous=order
    if payload['records_sha256']!=_digest(rows):raise ValueError('window records SHA mismatch')
    ema=payload['previous_ema']
    if not isinstance(ema,dict) or set(ema)!={'s','f','q','batches'}:raise ValueError('window EMA anchor missing')
    _integer(ema['batches'],'EMA batches')
    if ema['batches']+1!=identity['window_index']:raise ValueError('window index differs from prior EMA')
    for key in ('s','f','q'):
        if ema['batches']==0:
            if ema[key] is not None:raise ValueError('initial EMA must be empty')
        elif isinstance(ema[key],bool) or not isinstance(ema[key],(int,float)) or not math.isfinite(ema[key]):raise ValueError('nonfinite prior EMA')
    return True

def _matches(path,payload):
    try:actual=json.loads(Path(path).read_text())
    except (ValueError,UnicodeError) as error:raise ValueError('invalid existing window file') from error
    validate_window(actual)
    if _canonical(actual)!=_canonical(payload):raise ValueError('conflicting consumed-window evidence')

def _binding(path,payload):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),records_sha256=payload['records_sha256'],identity=deepcopy(payload['identity']))

def verify_window(binding,payload):
    validate_window(payload)
    if not isinstance(binding,dict) or set(binding)!={'path','sha256','records_sha256','identity'}:raise ValueError('window binding incomplete')
    _matches(binding['path'],payload)
    if _binding(binding['path'],payload)!=binding:raise ValueError('bound window file or SHA differs')
    return True

def write_window(path,payload):
    validate_window(payload);path=Path(path)
    if path.exists():
        _matches(path,payload);_sync_file_and_directory(path);_matches(path,payload)
        return _binding(path,payload)
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.partial')
    if temporary.exists():
        try:json.loads(temporary.read_text())
        except (ValueError,UnicodeError):_preserve_incomplete(temporary)
        else:_matches(temporary,payload)
    if not temporary.exists():
        with temporary.open('xb') as stream:
            stream.write((_canonical(payload)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    _sync_file_and_directory(temporary)
    try:os.link(temporary,path)
    except FileExistsError:_matches(path,payload)
    _sync_file_and_directory(path);_matches(path,payload);temporary.unlink()
    return _binding(path,payload)
