"""Atomic operator control with signed requests and durable stop/pause latches."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from .recovery import _atomic_bytes

class ControlProtocolError(ValueError): pass

def validate(value):
    if not isinstance(value,dict) or value.get('action') not in ('continue','pause','stop','request_speed'):
        raise ControlProtocolError('invalid control action')
    if value['action']=='request_speed':
        speed=value.get('speed')
        if isinstance(speed,bool) or not isinstance(speed,(float,int)) or not math.isfinite(speed) or not -.6<=speed<=.6:
            raise ControlProtocolError('speed must be finite in [-0.6,+0.6]')
        ids=value.get('env_ids')
        if ids is not None and (not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids)):
            raise ControlProtocolError('env_ids must be nonnegative integer list')
    return value

def write_control(path,value):
    validate(value);_atomic_bytes(Path(path),(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())

class ControlProtocol:
    def __init__(self,path,evidence_directory):
        self.path=Path(path);self.directory=Path(evidence_directory);self.latched=None;self.last_speed_sha=None;self._latched_raw=None;self._latched_persisted=False
    def _snapshot(self,raw):
        sha=hashlib.sha256(raw).hexdigest();self.directory.mkdir(parents=True,exist_ok=True)
        evidence=self.directory/(sha+'.json')
        if not evidence.exists():_atomic_bytes(evidence,raw)
        elif evidence.read_bytes()!=raw:raise ControlProtocolError('control evidence conflict')
        return dict(evidence_path=str(evidence),raw_sha256=sha)
    def poll(self):
        if self.latched is not None:
            if not self._latched_persisted:
                self.latched.update(self._snapshot(self._latched_raw));self._latched_persisted=True
            return deepcopy(self.latched)
        try:raw=self.path.read_bytes()
        except FileNotFoundError:return None
        sha=hashlib.sha256(raw).hexdigest()
        try:value=validate(json.loads(raw))
        except (ValueError,UnicodeError) as error:
            self._snapshot(raw)
            raise ControlProtocolError('invalid control; raw SHA '+sha) from error
        if value['action'] in ('stop','pause'):
            # Latch before evidence IO; a failed snapshot cannot let a later
            # continue revoke a request already observed by the worker.
            self.latched=dict(value,raw_sha256=sha,resume_authorized=False)
            self._latched_raw=raw
            return self.poll()
        self._snapshot(raw)
        if value['action']=='continue':return None
        if self.last_speed_sha==sha:return None
        self.last_speed_sha=sha
        return deepcopy(value)
