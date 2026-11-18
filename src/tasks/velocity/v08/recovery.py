"""v08 fail-closed file attestation and honest fresh-episode checkpoint recovery."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from . import spec
from .continuation_v08 import SOURCE_SHA256,SOURCE_PATH,SDS_SHA256,tree_digest

ROOT=Path(__file__).resolve().parents[4]
SDS_PATH='doc/sds/SDS_A2_v0.8_Stairs_Walk_Stop_Reverse.md'
RULINGS_PATH='doc/sds/SDS_A2_v0.8_W0_RULINGS.md'
RULINGS_SHA256='f0e9ebec57355805cb44ea774e4a3d23ab4a583033141ab4f36746bf24da3268'
class RecoveryBlocked(ValueError): pass

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def jsonable(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [jsonable(v) for v in value]
    if value is None or isinstance(value,(str,float,int,bool)): return value
    raise TypeError(type(value).__name__)

def numerical_config():
    out={}
    for name,value in vars(spec).items():
        if name.isupper():
            try: out[name]=jsonable(value)
            except TypeError: pass
    return out

def build_freeze(root=ROOT):
    """Conservative superset: every project source/script, all source assets, SDS.

    Old immutable imported helpers are included even when imported only lazily.
    Missing files never receive a placeholder digest. Generated Python caches
    are excluded; they are not source or physics assets.
    """
    root=Path(root).resolve(); paths=set()
    for directory in ('src','scripts'):
        for path in (root/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                paths.add(path)
    for relative in (SDS_PATH,RULINGS_PATH,'setup.py'):
        path=root/relative
        if not path.is_file(): raise RecoveryBlocked('required freeze input missing: '+relative)
        paths.add(path)
    if sha256(root/RULINGS_PATH)!=RULINGS_SHA256:raise RecoveryBlocked('approved W0 rulings SHA differs')
    if sha256(root/SDS_PATH)!=SDS_SHA256: raise RecoveryBlocked('approved SDS SHA differs')
    if sha256(SOURCE_PATH)!=SOURCE_SHA256:raise RecoveryBlocked('actual canonical6400 checkpoint SHA differs')
    files={str(p.relative_to(root)):sha256(p) for p in sorted(paths)}
    config=numerical_config()
    payload=dict(schema='v08-freeze-v1',files=files,config=config,
                 config_sha256=digest(config),sds_sha256=SDS_SHA256,
                 rulings_sha256=sha256(root/RULINGS_PATH),source_checkpoint_sha256=SOURCE_SHA256)
    payload['sha256']=digest(payload)
    return payload

def verify_freeze(frozen,root=ROOT):
    root=Path(root).resolve()
    if not isinstance(frozen,dict) or frozen.get('schema')!='v08-freeze-v1': raise RecoveryBlocked('freeze missing')
    payload={k:v for k,v in frozen.items() if k!='sha256'}
    if frozen.get('sha256')!=digest(payload): raise RecoveryBlocked('freeze envelope SHA differs')
    current=build_freeze(root)
    if frozen!=current: raise RecoveryBlocked('frozen source/config/SDS/assets differ from actual tree')
    return True

def freeze_case_tables():
    from .evaluation import make_cases
    tables={}
    for kind,phases in (('baseline',('A',)),('gate',spec.PHASE_ORDER),('h00',spec.PHASE_ORDER),('final',('E',))):
        for phase in phases:
            table=make_cases(phase,kind=kind,seed=42)
            tables[f'{kind}:{phase}']=dict(sha256=table.sha256,table=table.to_json())
    return json.loads(canonical(tables))

def verify_case_tables(tables):
    if tables!=freeze_case_tables(): raise RecoveryBlocked('frozen case tables differ')
    return True

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(jsonable(value),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    _atomic_bytes(path,raw)
    return sha256(path)

def _atomic_bytes(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(name,path)
    directory=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(directory)
    finally:os.close(directory)

def atomic_save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    with os.fdopen(fd,'wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(name,path)
    directory=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(directory)
    finally:os.close(directory)
    return sha256(path)

def load_verified(path,expected_sha):
    if not isinstance(expected_sha,str) or len(expected_sha)!=64 or sha256(path)!=expected_sha:
        raise RecoveryBlocked('checkpoint SHA differs or not supplied')
    return torch.load(path,map_location='cpu',weights_only=False)

def capture_rng(env=None):
    return dict(python=deepcopy(random.getstate()),numpy=deepcopy(np.random.get_state()),
        torch=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        env=deepcopy(env.rng_state()) if env is not None else None,
        sampler=deepcopy(env.sampler_rng_state()) if env is not None else None)

def restore_rng(state,env=None,*,device='cpu',allow_cpu_cuda_provenance=False):
    if not isinstance(state,dict) or set(state)!=set(('python','numpy','torch','cuda','env','sampler')):
        raise RecoveryBlocked('RNG schema differs')
    if env is not None:
        if not all(hasattr(env,k) for k in ('load_rng_state','load_sampler_rng_state')) or state['env'] is None or state['sampler'] is None:
            raise RecoveryBlocked('environment/sampler RNG is missing or incompatible')
    cuda=state['cuda'];receipt=dict(cuda='empty_source')
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available() or len(cuda)!=torch.cuda.device_count():
            raise RecoveryBlocked('saved CUDA RNG incompatible; owner adjudication required')
        torch.cuda.set_rng_state_all([torch.as_tensor(x,dtype=torch.uint8).cpu() for x in cuda])
        receipt['cuda']='restored'
    elif cuda:
        if not allow_cpu_cuda_provenance: raise RecoveryBlocked('CUDA RNG cannot be restored on CPU')
        receipt['cuda']='source_preserved_cpu_engineering_only'
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(torch.as_tensor(state['torch'],dtype=torch.uint8).cpu())
    if env is not None:
        try:
            env.load_rng_state(deepcopy(state['env']));env.load_sampler_rng_state(deepcopy(state['sampler']))
        except (RuntimeError,ValueError,TypeError,KeyError) as error:
            raise RecoveryBlocked('saved environment/sampler RNG incompatible; owner adjudication required: '+str(error)) from error
    return receipt

def normalizer_state(model):
    return dict(actor=deepcopy(model.actor.normalizer.state_dict()),critic=deepcopy(model.critic_normalizer.state_dict()))

def load_evaluation_reference(entry):
    from .evaluation_records import restore_result,verify_sidecar
    if not isinstance(entry,dict) or not all(k in entry for k in ('binding','identity','summary','control_samples')):
        raise RecoveryBlocked('evaluation reference missing')
    path=Path(entry['binding']['path'])
    payload=json.loads(path.read_text())
    result=restore_result(payload,entry['identity']);verify_sidecar(entry['binding'],payload)
    if result.summary!=entry['summary']:raise RecoveryBlocked('evaluation summary/reference differs')
    if evaluation_control_samples(result)!=entry['control_samples']:raise RecoveryBlocked('evaluation reference cost differs from raw')
    if 'transaction_intent_path' in entry:
        _,actual,_=_verify_evaluation_transaction(entry['transaction_intent_path'],entry['identity'])
        if actual!=entry:raise RecoveryBlocked('evaluation transaction/reference differs')
    return result

def evaluation_control_samples(result):
    batches={}
    for row in result.records:
        raw=row.get('raw') or {};key=raw.get('collector_batch_id');cost=raw.get('collector_batch_control_samples')
        if not isinstance(key,str) or len(key)!=64 or type(cost) is not int or cost<1:
            raise RecoveryBlocked('actual asynchronous collector cost evidence missing')
        if key in batches and batches[key]!=cost:raise RecoveryBlocked('collector cost conflicts within batch')
        batches[key]=cost
    return sum(batches.values())

def _immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=jsonable(value):raise RecoveryBlocked('immutable evaluation transaction conflict: '+str(path))
        return sha256(path)
    return write_json(path,value)

def _evaluation_progress(intent_path,intent):
    directory=intent_path.with_name(intent_path.name.replace('.intent.json','.batches'))
    running=0;hashes={};complete=True;identity_sha=digest(intent['identity'])
    for index,case_ids in enumerate(intent['batches']):
        before=directory/f'batch_{index:04d}.intent.json';after=directory/f'batch_{index:04d}.commit.json'
        expected=dict(schema='v08-evaluation-batch-intent-v1',identity_sha256=identity_sha,
            batch_index=index,case_ids=case_ids,before_control_samples=running)
        if not before.exists():complete=False;break
        if json.loads(before.read_text())!=expected:raise RecoveryBlocked('evaluation batch intent differs')
        hashes[before.name]=sha256(before)
        if not after.exists():complete=False;break
        value=json.loads(after.read_text());cost=value.get('control_samples')
        if type(cost) is not int or cost<0 or value.get('intent_sha256')!=sha256(before) or value.get('before_control_samples')!=running or value.get('after_control_samples')!=running+cost or value.get('physics_substeps')!=cost*4:
            raise RecoveryBlocked('evaluation batch cost/progress differs')
        hashes[after.name]=sha256(after);running+=cost
        if value.get('completed') is not True:complete=False;break
    return running,complete,hashes

def _verify_evaluation_transaction(intent_path,expected_identity=None):
    from .evaluation_records import restore_result,verify_sidecar
    intent_path=Path(intent_path);intent=json.loads(intent_path.read_text())
    if intent.get('schema')!='v08-evaluation-intent-v1' or not isinstance(intent.get('identity'),dict) or not isinstance(intent.get('batches'),list):
        raise RecoveryBlocked('invalid evaluation intent')
    identity=intent['identity']
    if expected_identity is not None and identity!=expected_identity:raise RecoveryBlocked('evaluation intent input identity differs')
    known,complete,hashes=_evaluation_progress(intent_path,intent)
    sidecar=Path(intent['sidecar_path'])
    if not complete or not sidecar.is_file():
        error=RecoveryBlocked(f'unfinished evaluation intent: known_control_samples={known}; remaining physical cost uncertain; owner adjudication required, no automatic re-evaluation')
        error.audit=dict(intent_path=str(intent_path),known_control_samples=known,remaining_physical_cost_unknown=True)
        raise error
    payload=json.loads(sidecar.read_text());result=restore_result(payload,identity)
    if [cid for batch in intent['batches'] for cid in batch]!=[c['case_id'] for c in result.table.cases] or any(not batch or len(batch)>8 for batch in intent['batches']):
        raise RecoveryBlocked('evaluation batch plan differs from complete frozen table')
    if result.summary.get('engineering_error'):raise RecoveryBlocked('evaluation transaction contains engineering errors; owner adjudication required')
    if known!=evaluation_control_samples(result):raise RecoveryBlocked('evaluation raw batch cost differs from durable progress')
    binding=dict(path=str(sidecar),sha256=sha256(sidecar),**result.evidence_sha256)
    verify_sidecar(binding,payload)
    reference=dict(binding=binding,identity=identity,summary=deepcopy(result.summary),control_samples=known,raw_bytes=sidecar.stat().st_size)
    receipt=dict(schema='v08-evaluation-commit-v1',intent_sha256=sha256(intent_path),batch_progress_sha256=hashes,reference=deepcopy(reference))
    commit=intent_path.with_name(intent_path.name.replace('.intent.json','.commit.json'))
    if commit.exists() and json.loads(commit.read_text())!=receipt:raise RecoveryBlocked('evaluation completed cost/binding receipt differs')
    reference.update(transaction_intent_path=str(intent_path),transaction_commit_path=str(commit))
    return result,reference,receipt

def verify_evaluation_journal(directory,freeze_sha256=None):
    entries={};journal=Path(directory)/'evaluation_transactions'
    for evidence in list(journal.glob('*.batches'))+list(journal.glob('*.commit.json')):
        name=evidence.name.replace('.batches','.intent.json').replace('.commit.json','.intent.json')
        if not evidence.with_name(name).is_file():raise RecoveryBlocked('orphan evaluation batch/cost evidence without intent; owner adjudication required')
    for path in sorted((Path(directory)/'evaluation_transactions').glob('*.intent.json')):
        result,entry,_=_verify_evaluation_transaction(path)
        identity=entry['identity']
        if identity.get('source_sha256')!=SOURCE_SHA256 or (freeze_sha256 is not None and identity.get('freeze_sha256')!=freeze_sha256):
            raise RecoveryBlocked('evaluation journal frozen input differs')
        entries[path.name]=entry
    return entries

def verify_evaluation_checkpoint_coverage(state,transactions):
    pending=state['ledger'].get('pending_boundary')
    for entry in transactions.values():
        identity=entry['identity'];key=f"{identity['kind']}:{identity['update']}"
        if key in state['evaluation']:
            if state['evaluation'][key]!=entry:raise RecoveryBlocked('evaluation checkpoint/journal binding or cost differs')
            continue
        remaining=pending['events'][pending['index']:] if pending else []
        if not pending or identity['update']!=state['learning']['update'] or identity['kind'] not in remaining or identity['phase']!=pending['phase']:
            raise RecoveryBlocked('completed evaluation has unaccounted cost outside pending boundary; owner adjudication required')
    return True

class EvaluationTransaction:
    """An incomplete physics attempt is never automatically replayed.

    Batch intents precede factory/physics calls. Completed and interrupted batch
    counter deltas are fsynced separately. A complete immutable result may be
    reused after a missing final receipt; partial physical episodes require the
    owner to adjudicate, with known costs retained and unknown remainder explicit.
    """
    def __init__(self,directory,identity,table,sidecar,case_runner):
        self.directory=Path(directory).resolve();self.identity=deepcopy(identity);self.table=table
        self.sidecar=Path(sidecar).resolve();self.base=case_runner;self.index=0
        self.path=self.directory/'evaluation_transactions'/f'{identity["kind"]}_{identity["phase"]}_{identity["update"]:06d}.intent.json'
        self.progress=self.path.with_name(self.path.name.replace('.intent.json','.batches'))
        if self.path.exists():
            self.intent=json.loads(self.path.read_text())
            if self.intent.get('identity')!=identity or self.intent.get('sidecar_path')!=str(self.sidecar):raise RecoveryBlocked('evaluation intent identity differs')
            planned=[case_id for batch in self.intent.get('batches',[]) for case_id in batch]
            if planned!=[c['case_id'] for c in table.cases]:raise RecoveryBlocked('evaluation intent case plan differs')
        else:
            size=8 if getattr(case_runner,'supports_batch',False) else 1
            cases=[c['case_id'] for c in table.cases]
            self.intent=dict(schema='v08-evaluation-intent-v1',identity=deepcopy(identity),sidecar_path=str(self.sidecar),
                batches=[cases[i:i+size] for i in range(0,len(cases),size)])
        self.supports_batch=getattr(case_runner,'supports_batch',False)
        self.before=int(getattr(case_runner,'control_samples',0))
    def recover(self):
        if not self.path.exists():
            if self.progress.exists() or self.path.with_name(self.path.name.replace('.intent.json','.commit.json')).exists():
                raise RecoveryBlocked('orphan evaluation batch/cost evidence without intent; no automatic replay')
            if self.sidecar.exists():raise RecoveryBlocked('evaluation sidecar lacks durable intent; no missing-file shortcut')
            return None
        result,reference,receipt=_verify_evaluation_transaction(self.path,self.identity)
        _immutable_json(reference['transaction_commit_path'],receipt)
        return result,reference
    def start(self):
        if self.path.exists():raise RecoveryBlocked('evaluation already started; recover or seek owner adjudication')
        self.recover()  # reject orphan cost evidence before starting any physics
        if self.base is None or not hasattr(self.base,'control_samples'):raise RecoveryBlocked('actual evaluation collector counter missing')
        _immutable_json(self.path,self.intent)
    def __call__(self,cases):
        batch=[cases] if isinstance(cases,dict) else list(cases);ids=[c['case_id'] for c in batch]
        if self.index>=len(self.intent['batches']) or ids!=self.intent['batches'][self.index]:raise RecoveryBlocked('evaluation batch differs from durable plan')
        before=int(self.base.control_samples)-self.before
        path=self.progress/f'batch_{self.index:04d}.intent.json'
        value=dict(schema='v08-evaluation-batch-intent-v1',identity_sha256=digest(self.identity),batch_index=self.index,
            case_ids=ids,before_control_samples=before)
        _immutable_json(path,value);completed=False;error_type=None
        try:
            result=self.base(cases);completed=True;return result
        except BaseException as error:
            error_type=type(error).__name__;raise
        finally:
            after=int(self.base.control_samples)-self.before
            receipt=dict(intent_sha256=sha256(path),before_control_samples=before,after_control_samples=after,
                control_samples=after-before,physics_substeps=(after-before)*4,completed=completed,error_type=error_type,
                remaining_physical_cost_unknown=not completed)
            _immutable_json(path.with_name(path.name.replace('.intent.json','.commit.json')),receipt)
            self.index+=1
    def finish(self,result):
        from .evaluation_records import pack_result,write_sidecar
        payload=pack_result(result,self.identity);write_sidecar(self.sidecar,payload)
        verified,reference,receipt=_verify_evaluation_transaction(self.path,self.identity)
        _immutable_json(reference['transaction_commit_path'],receipt)
        return verified,reference

def verify_learning_journal(directory,checkpoint_update):
    directory=Path(directory)/'transactions'
    for intent in directory.glob('update_*.intent.json'):
        value=json.loads(intent.read_text());update=value.get('update')
        if type(update) is not int or update<6401:raise RecoveryBlocked('invalid learning intent')
        commit=intent.with_name(intent.name.replace('.intent.json','.commit.json'))
        aborted=intent.with_name(intent.name.replace('.intent.json','.aborted.json'))
        if commit.exists():
            completed=json.loads(commit.read_text())
            if completed.get('update')!=update:raise RecoveryBlocked('learning commit identity differs')
            if update>checkpoint_update:raise RecoveryBlocked('checkpoint predates spent learning; rollback/replay requires owner adjudication')
        elif aborted.exists():
            record=json.loads(aborted.read_text())
            if record.get('update')!=update or record.get('adam_unchanged') is not True:
                raise RecoveryBlocked('invalid aborted learning audit')
        else:raise RecoveryBlocked('unfinished PPO intent: no automatic replay or budget reset')
    return True

@__import__('functools').lru_cache(maxsize=1)
def source_architecture_contract():
    from .continuation_v08 import load_designated_source
    source=load_designated_source()
    def shapes(value):
        return {k:(str(v.dtype),tuple(v.shape)) for k,v in value.items()}
    return dict(model_shapes=shapes(source['model']),
        optimizer_shapes={k:shapes(v) for k,v in source['optimizer']['state'].items()},
        groups=[{k:deepcopy(v) for k,v in group.items() if k!='lr'} for group in source['optimizer']['param_groups']],
        actor_normalizer=tree_digest(source['normalizer']['actor']))

def validate_environment_snapshot(state,num_envs):
    required=('version','phase','stage_iteration','generation','roles','common_step_counter','case_templates',
        'rng','sampler_rng','task_sampler_rng','command','tasks','standing','reward','tensors','contacts',
        'contact_inner','stall','climb','history','history_initialized','map','stop_windows','trace','reward_sums','physics_restorable')
    if not isinstance(state,dict) or any(k not in state for k in required):raise RecoveryBlocked('complete task logical snapshot missing')
    if state['version']!=spec.CODE_VERSION or state['phase'] not in spec.PHASE_ORDER or state['physics_restorable'] is not False:
        raise RecoveryBlocked('environment identity/recovery claim differs')
    if any(len(state[k])!=num_envs for k in ('roles','case_templates','stop_windows','trace')):
        raise RecoveryBlocked('environment logical batch differs')
    command=state['command']
    for key in ('current','executed','requested','start','k','executed_transition','executed_requested'):
        value=command.get(key)
        if not torch.is_tensor(value) or tuple(value.shape)!=(num_envs,) or not torch.isfinite(value).all():
            raise RecoveryBlocked('command/interpolation state incomplete: '+key)
    if (command['k']<0).any() or (command['k']>100).any():raise RecoveryBlocked('interpolation index outside100step contract')
    tasks=state['tasks']
    for key in ('tensors','cases','events','direction_checks','limits','points','budget_events','leg_ids'):
        if key not in tasks:raise RecoveryBlocked('task milestone/contact/timing snapshot missing: '+key)
    for key in ('steps','mode','hold_steps','leg_steps','direction_steps','direction','goal','stop_kind','request_step',
                'arrival_paid','success_paid','failure_paid','mid_done','top_done'):
        value=tasks['tensors'].get(key)
        if not torch.is_tensor(value) or tuple(value.shape)!=(num_envs,):raise RecoveryBlocked('task latch state missing: '+key)
    expected=dict(count=(num_envs,),xy=(num_envs,spec.ENDPOINT_WINDOW_STEPS,2),
        values=(num_envs,spec.ENDPOINT_WINDOW_STEPS,10),last_height=(num_envs,4),height_valid=(num_envs,4))
    for key,shape in expected.items():
        value=state['standing'].get(key)
        if not torch.is_tensor(value) or tuple(value.shape)!=shape:raise RecoveryBlocked('standing/support history missing: '+key)
    if tuple(state['history'].shape)!=(num_envs,spec.HISTORY_LEN,spec.PROPRIO_DIM):raise RecoveryBlocked('Actor history snapshot missing')
    if any(not isinstance(state[k],dict) or not state[k] for k in ('contacts','contact_inner','climb','map','stall')):
        raise RecoveryBlocked('physical-contact/map diagnostic logical snapshot missing')
    return True

def validate_checkpoint(state,*,mode=None,root=ROOT):
    required=('schema','identity','model','optimizer','normalizer','learning','rng','curriculum',
        'ledger','source','initialization','environment','freeze','case_tables','mode','recipe','resume_mode','evaluation')
    if not isinstance(state,dict) or any(k not in state for k in required): raise RecoveryBlocked('incomplete native checkpoint')
    if state['schema']!=10 or state['identity']!=dict(task=spec.TASK_ID,recipe=spec.RECIPE_VERSION): raise RecoveryBlocked('foreign identity/schema')
    if mode is not None and state['mode']!=mode: raise RecoveryBlocked('engineering/formal container mismatch')
    if state['source'].get('checkpoint_sha256')!=SOURCE_SHA256 or state['source'].get('checkpoint_path')!=SOURCE_PATH:
        raise RecoveryBlocked('source ancestry differs')
    if state['resume_mode']!='fresh_physical_episode' or not isinstance(state['environment'],dict): raise RecoveryBlocked('task logical snapshot missing')
    if type(state.get('recovery_pending_fresh_reset')) is not bool:raise RecoveryBlocked('native reset readiness flag missing')
    validate_environment_snapshot(state['environment'],state['learning']['num_envs'])
    if state['recipe']!=spec.contract(): raise RecoveryBlocked('native recipe differs')
    verify_freeze(state['freeze'],root);verify_case_tables(state['case_tables'])
    from .ledger import CostLedger
    ledger=CostLedger(mode=state['mode'],num_envs=state['learning']['num_envs'],state=state['ledger'])
    learning=state['learning'];update=learning['update']
    if update!=6400+ledger.updates or learning['adam_steps']!=128000+20*ledger.updates or learning['optimizer_step']!=learning['adam_steps']:
        raise RecoveryBlocked('learning/Adam ledger differs')
    if learning['control_samples']!=ledger.learning_samples or learning['physics_substeps']!=ledger.learning_samples*4:
        raise RecoveryBlocked('sample ledger differs')
    expected_envs=8 if state['mode']=='smoke' else 512
    if learning['num_envs']!=expected_envs:raise RecoveryBlocked('native environment count differs from container')
    if learning['rollout']!=48 or learning['epochs']!=5 or learning['minibatches']!=4: raise RecoveryBlocked('PPO recipe differs')
    cur=state['curriculum']['state']
    if cur['last_update']!=update or sum(cur['spent_by_phase'].values())!=ledger.updates: raise RecoveryBlocked('curriculum count differs')
    reference=source_architecture_contract()
    model_shapes={k:(str(v.dtype),tuple(v.shape)) for k,v in state['model'].items()}
    if model_shapes!=reference['model_shapes']:raise RecoveryBlocked('native architecture/model shapes differ')
    if tree_digest(state['normalizer']['actor'])!=reference['actor_normalizer']:
        raise RecoveryBlocked('frozen source Actor normalizer changed')
    optimizer=state['optimizer']
    groups=[{k:v for k,v in group.items() if k!='lr'} for group in optimizer['param_groups']]
    if groups!=reference['groups']:raise RecoveryBlocked('Adam parameter order/hyperparameters differ')
    shapes={k:{name:(str(v.dtype),tuple(v.shape)) for name,v in values.items()} for k,values in optimizer['state'].items()}
    if shapes!=reference['optimizer_shapes']:raise RecoveryBlocked('Adam tensor association/shape differs')
    if len(optimizer['state'])!=45 or any(float(x['step'])!=learning['adam_steps'] for x in optimizer['state'].values()):
        raise RecoveryBlocked('Adam history differs')
    if any(group['lr']!=learning['learning_rate'] for group in optimizer['param_groups']): raise RecoveryBlocked('actual LR differs')
    def finite_tree(value):
        if torch.is_tensor(value):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite_tree(x) for x in value.values())
        if isinstance(value,(list,tuple)):return all(finite_tree(x) for x in value)
        return True
    if state['ledger']['evaluation_control_samples']!=sum(ref['control_samples'] for ref in state['evaluation'].values()):
        raise RecoveryBlocked('evaluation cost ledger differs from complete sidecar references')
    for reference in state['evaluation'].values():
        if 'transaction_intent_path' not in reference:raise RecoveryBlocked('evaluation durable intent reference missing')
        load_evaluation_reference(reference)
    for block in ('model','optimizer','normalizer'):
        tree_digest(state[block])
        if not finite_tree(state[block]):raise RecoveryBlocked('nonfinite checkpoint '+block)
    if state['normalizer']!={}:
        for group,prefix in (('actor','actor.normalizer.'),('critic','critic_normalizer.')):
            derived={k[len(prefix):]:v for k,v in state['model'].items() if k.startswith(prefix)}
            if tree_digest(derived)!=tree_digest(state['normalizer'][group]): raise RecoveryBlocked('normalizer/model mismatch')
    return True
