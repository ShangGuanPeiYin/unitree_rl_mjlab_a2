"""Qv07-002: atomic external requests, latched stops, preserved invalid bytes."""
from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile

ACTIONS=('continue','pause','stop')

class ControlProtocolError(RuntimeError):
    def __init__(self,message,*,audit):
        super().__init__(message);self.audit=deepcopy(audit)

def _object(value):
    if not isinstance(value,dict) or not isinstance(value.get('action'),str) or value['action'] not in ACTIONS:
        raise ValueError('control must be an object with action continue/pause/stop')
    return value

def _directory_sync(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)

def write_control(path,value):
    """Publisher helper: replace the full object atomically, never partial JSON."""
    _object(value);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path);_directory_sync(path.parent)
    except BaseException:
        # A failed publisher's partial is evidence, never delete it silently.
        raise
    return path

class ControlProtocol:
    def __init__(self,path,evidence_directory):
        self.path=Path(path);self.evidence_directory=Path(evidence_directory);self.latched=None;self._latched_raw=None;self._latched_persisted=False

    def _snapshot(self,raw):
        self.evidence_directory.mkdir(parents=True,exist_ok=True)
        sha=hashlib.sha256(raw).hexdigest();path=self.evidence_directory/(sha+'.bin')
        try:
            with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes()!=raw:raise OSError('conflicting control evidence bytes')
            with path.open('rb') as stream:os.fsync(stream.fileno())
        _directory_sync(path.parent)
        return dict(evidence_path=str(path),raw_sha256=sha)

    def _error(self,error,raw=None):
        audit=dict(path=str(self.path),error_type='ControlProtocolError',parse_error=repr(error),
            raw_available=raw is not None,raw_base64=(base64.b64encode(raw).decode() if raw is not None else None))
        if self.latched is not None:audit['latched_request']=deepcopy(self.latched)
        if raw is not None:
            try:audit.update(self._snapshot(raw))
            except OSError as failure:audit['evidence_write_error']=repr(failure)
        raise ControlProtocolError('invalid external control: '+str(error),audit=audit) from error

    def poll(self):
        if self.latched is not None:
            if not self._latched_persisted:
                try:self.latched.update(self._snapshot(self._latched_raw))
                except OSError as error:self._error(error,self._latched_raw)
                self._latched_persisted=True
            return deepcopy(self.latched)
        raw=None
        try:raw=self.path.read_bytes()
        except FileNotFoundError:return None
        except OSError as error:self._error(error)
        try:value=_object(json.loads(raw))
        except (ValueError,UnicodeError,TypeError) as error:self._error(error,raw)
        action=value['action']
        if action=='continue':return None
        # Observation latches before evidence IO: a failed write must never let
        # a later continue revoke this already observed pause/stop request.
        self.latched=dict(action=action,reason='operator_pause' if action=='pause' else 'user_requested_stop',
            pause_requested=action=='pause',resume_authorized=False,raw_sha256=hashlib.sha256(raw).hexdigest())
        self._latched_raw=raw
        return self.poll()
