"""Atomic complete learning state and append-only nonrefundable external costs."""
import contextlib
import gc
import zipfile
import time
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from . import spec
from .identity import code_fingerprint
from .teachers import teacher_identity
SCHEMA='a2-v10-state-v1'

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def memory_observation(stage):
    import psutil
    import resource
    return dict(stage=stage,time=time.time(),process_rss_bytes=psutil.Process().memory_info().rss,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        host_available_bytes=psutil.virtual_memory().available)

def release_host_memory():
    """Drop dead Python/native CPU allocations without changing any live storage."""
    gc.collect()
    import ctypes
    try:ctypes.CDLL(None).malloc_trim(0)
    except AttributeError:pass

def trace_memory(callback,stage):
    if callback is not None:callback(memory_observation(stage))

def verify_checkpoint_file(path,expected_sha):
    if sha256(path)!=expected_sha:raise ValueError('checkpoint disk SHA mismatch')
    with zipfile.ZipFile(path) as archive:
        bad=archive.testzip()
    if bad is not None:raise ValueError('checkpoint disk CRC mismatch: '+bad)
    return expected_sha

def state_fingerprint(value):
    """Streaming tensor/tree identity without cloning an entire state tree."""
    digest=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous();digest.update(str((t.dtype,tuple(t.shape))).encode())
            data=memoryview(t.numpy()).cast('B')
            for offset in range(0,len(data),8*1024*1024):digest.update(data[offset:offset+8*1024*1024])
        elif isinstance(v,np.ndarray):
            digest.update(str((v.dtype,v.shape)).encode());digest.update(memoryview(np.ascontiguousarray(v)).cast('B'))
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):digest.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            digest.update(type(v).__name__.encode())
            for x in v:visit(x)
        else:digest.update(repr(v).encode()+b'\0')
    visit(value);return digest.hexdigest()

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        json.dump(data,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno());temp=f.name
    os.replace(temp,path)

def checkpoint_save(path,state,memory_trace=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(state,schema=SCHEMA,sds_sha256=spec.SDS_SHA256,recipe=spec.RECIPE,code_sha256=code_fingerprint(),rng=rng_state(),
               teachers=teacher_identity())
    trace_memory(memory_trace,'save_begin')
    fd,temp=tempfile.mkstemp(dir=path.parent);os.close(fd)
    try:
        with open(temp,'wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    trace_memory(memory_trace,'save_serialized')
    digest=sha256(path)
    verify_checkpoint_file(path,digest)
    trace_memory(memory_trace,'save_disk_validated')
    return digest

def checkpoint_validate(state,*,num_envs,physical_sha256):
    if (state.get('schema')!=SCHEMA or state.get('sds_sha256')!=spec.SDS_SHA256 or state.get('recipe')!=spec.RECIPE
        or state.get('code_sha256')!=code_fingerprint()
        or state.get('teachers')!=teacher_identity()):raise ValueError('checkpoint identity mismatch')
    if state.get('num_envs')!=num_envs or state.get('physical_sha256')!=physical_sha256:raise ValueError('checkpoint N/physics mismatch')
    return state

def checkpoint_load(path,*,num_envs,physical_sha256,restore_random=True,memory_trace=None,formal_handoff=None):
    if formal_handoff is not None:
        from .handoff import validate_handoff
        reference=formal_handoff.get('path') if isinstance(formal_handoff,dict) else formal_handoff
        if not reference:raise ValueError('verified formal handoff credential path required')
        checked=validate_handoff(reference,num_envs=num_envs,physical_sha256=physical_sha256)
        if isinstance(formal_handoff,dict) and formal_handoff.get('sha256')!=checked['sha256']:raise ValueError('formal handoff credential changed')
        formal_handoff=checked
    trace_memory(memory_trace,'load_begin')
    state=torch.load(path,map_location='cpu',weights_only=False)
    try:checkpoint_validate(state,num_envs=num_envs,physical_sha256=physical_sha256)
    except ValueError:
        if formal_handoff is None:raise
        from .handoff import authorize_origin_checkpoint
        authorize_origin_checkpoint(state,formal_handoff,checkpoint_path=path,num_envs=num_envs,physical_sha256=physical_sha256)
    if restore_random:restore_rng(state['rng'])
    trace_memory(memory_trace,'load_complete')
    return state

class BudgetLedger:
    """Durable idempotent reserve/execute journal, with conservative uncertain cost."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self._cache=None;self._pending=set();self._updated={}
    def _scan(self,f):
        stat=os.fstat(f.fileno())
        if self._cache is None:
            offset=0;totals={};confirmed={};entries={};seq=0
        else:
            inode,offset,totals,confirmed,entries,seq=self._cache
            if inode!=stat.st_ino or stat.st_size<offset:raise RuntimeError('append-only ledger replaced or truncated')
        f.seek(offset)
        while True:
            line=f.readline()
            if not line:break
            row=json.loads(line);seq+=1
            if row['seq']!=seq:raise RuntimeError('ledger sequence damaged')
            op=row['op'];key=row['key'];uid=row['id'];samples=row['samples']
            if samples<0:raise RuntimeError('negative journal samples')
            if op=='update':
                if uid in entries or self._updated.get(key,0)+samples>confirmed.get(key,0):raise RuntimeError('invalid completed-update samples')
                entries[uid]=dict(key=key,samples=samples,status='updated')
                self._updated[key]=self._updated.get(key,0)+samples
            elif op in ('reserve','charge'):
                if uid in entries:raise RuntimeError('duplicate debit id')
                entries[uid]=dict(key=key,samples=samples,status='pending' if op=='reserve' else 'executed')
                if op=='reserve':self._pending.add(uid)
                totals[key]=totals.get(key,0)+samples
                if op=='charge':confirmed[key]=confirmed.get(key,0)+samples
            else:
                if uid not in entries or entries[uid]['status']!='pending':raise RuntimeError('invalid journal finalization')
                entry=entries[uid]
                if entry['key']!=key or entry['samples']!=samples:raise RuntimeError('journal identity mismatch')
                if op not in ('executed','uncertain'):raise RuntimeError('invalid journal operation')
                entry['status']=op;self._pending.discard(uid)
                if op=='executed':confirmed[key]=confirmed.get(key,0)+samples
        self._cache=(stat.st_ino,f.tell(),totals,confirmed,entries,seq)
        return totals,confirmed,entries,seq
    def _locked(self,callback):
        import fcntl
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);state=self._scan(f)
            return callback(f,*state)
    @staticmethod
    def _write(f,row):
        f.seek(0,2);f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    def totals(self):return self.report()['charged']
    def report(self):
        def perform(f,totals,confirmed,entries,seq):
            uncertain={k:v-confirmed.get(k,0) for k,v in totals.items()}
            return dict(charged=dict(totals),executed=dict(confirmed),uncertain=uncertain,pending=list(self._pending),updated=dict(self._updated))
        return self._locked(perform)
    def record_update(self,uid,course,samples,*,kind='learning'):
        key=f'{kind}:{course}'
        def perform(f,totals,confirmed,entries,seq):
            if uid in entries:return False
            if self._pending:raise RuntimeError('cannot update while physical step unresolved')
            if samples<=0 or self._updated.get(key,0)+samples>confirmed.get(key,0):raise RuntimeError('update exceeds confirmed samples')
            self._write(f,dict(seq=seq+1,op='update',id=uid,key=key,samples=samples));return True
        return self._locked(perform)
    def charge(self,course,samples,*,limit,kind,details=None):
        import uuid
        uid=uuid.uuid4().hex;key=f'{kind}:{course}'
        def perform(f,totals,confirmed,entries,seq):
            value=totals.get(key,0)+samples
            if not isinstance(samples,int) or samples<0:raise ValueError('nonnegative integer samples required')
            if value>limit:raise RuntimeError(f'budget limit exceeded {key}: {value}>{limit}')
            self._write(f,dict(seq=seq+1,op='charge',id=uid,key=key,samples=samples,details=details or {}));return value
        return self._locked(perform)
    def reserve(self,uid,course,samples,*,limit,kind,details=None):
        key=f'{kind}:{course}'
        def perform(f,totals,confirmed,entries,seq):
            if uid in entries:return False
            if self._pending:raise RuntimeError('one step already in flight; reconcile before proceeding')
            if not isinstance(samples,int) or samples<=0:raise ValueError('positive integer reservation required')
            if totals.get(key,0)+samples>limit:raise RuntimeError('charged budget exhausted')
            self._write(f,dict(seq=seq+1,op='reserve',id=uid,key=key,samples=samples,details=details or {}));return True
        return self._locked(perform)
    def mark_executed(self,uid):
        def perform(f,totals,confirmed,entries,seq):
            if uid not in entries:raise RuntimeError('unknown reservation')
            e=entries[uid]
            if e['status']=='executed':return False
            if e['status']!='pending':raise RuntimeError('uncertain reservation cannot be replayed')
            self._write(f,dict(seq=seq+1,op='executed',id=uid,key=e['key'],samples=e['samples']));return True
        return self._locked(perform)
    def reconcile_uncertain(self):
        def perform(f,totals,confirmed,entries,seq):
            for uid in list(self._pending):
                e=entries[uid]
                if e['status']=='pending':
                    seq+=1;self._write(f,dict(seq=seq,op='uncertain',id=uid,key=e['key'],samples=e['samples']))
        return self._locked(perform)
    def assert_checkpoint_not_ahead(self,checkpoint_totals):
        current=self.totals()
        if any(current.get(k,0)<v for k,v in checkpoint_totals.items()):raise RuntimeError('external ledger behind checkpoint; refusing refund')
