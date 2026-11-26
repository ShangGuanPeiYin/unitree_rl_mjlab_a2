"""Q4 CPU-only lifecycle checks; no CUDA context or physics execution."""
import copy
import hashlib
from pathlib import Path
import pytest
import torch
from src.tasks.velocity.v10.checkpoint import sha256, verify_checkpoint_file, checkpoint_save,checkpoint_load
from src.tasks.velocity.v10.replay import TaskReservoir,TASKS

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 # Other agents may edit independent files during this CPU lifecycle seam test.
 # Pin only test checkpoint identity; production always hashes real source.
 from src.tasks.velocity.v10 import checkpoint as cp
 monkeypatch.setattr(cp,'code_fingerprint',lambda:'q4-cpu-lifecycle-fixture')

def batch():
 return dict(obs={'proprio':torch.arange(6*45,dtype=torch.float32).reshape(6,45)},actions=torch.arange(72,dtype=torch.float32).reshape(6,12),task=list(TASKS),teacher=['blind']*6,case=['case']*6,episode=[2**40+i for i in range(6)],control_step=list(range(6)))

def test_sha_reads_bounded_8mib_and_matches_reference(tmp_path,monkeypatch):
 p=tmp_path/'blob';payload=b'A'*(8*1024*1024+31);p.write_bytes(payload)
 original=Path.open;reads=[]
 class Reader:
  def __init__(self,f):self.f=f
  def __enter__(self):self.f.__enter__();return self
  def __exit__(self,*a):return self.f.__exit__(*a)
  def read(self,n=-1):reads.append(n);assert n==8*1024*1024;return self.f.read(n)
 def opened(self,*a,**kw):
  f=original(self,*a,**kw)
  return Reader(f) if self==p and a==('rb',) else f
 monkeypatch.setattr(Path,'open',opened)
 assert sha256(p)==hashlib.sha256(payload).hexdigest();assert len(reads)==3

def test_owned_restore_moves_storage_and_preserves_source_file(tmp_path):
 r=TaskReservoir(capacity=3,seed=42)
 for _ in range(3):r.add(batch())
 signature=r.integrity_manifest();expected=r.next_sample_manifest(2)
 path=tmp_path/'c.pt';digest=checkpoint_save(path,dict(model={},num_envs=512,course='D1',physical_sha256='abc',reservoir=r.state_dict(copy_storage=False)))
 verify_checkpoint_file(path,digest);r.release_storage();assert all(v is None for v in r.data.values())
 state=checkpoint_load(path,num_envs=512,physical_sha256='abc',restore_random=False)
 owned=state['reservoir'];ptr=owned['data'][TASKS[0]]['obs']['proprio'].data_ptr()
 r.take_state_dict(owned)
 assert 'data' not in owned and r.data[TASKS[0]]['obs']['proprio'].data_ptr()==ptr
 assert r.integrity_manifest()==signature and r.next_sample_manifest(2)==expected
 assert path.exists() and sha256(path)==digest

def test_failed_load_does_not_delete_source(tmp_path):
 path=tmp_path/'c.pt';checkpoint_save(path,dict(model={},num_envs=512,course='D1',physical_sha256='abc'))
 with pytest.raises(ValueError):checkpoint_load(path,num_envs=512,physical_sha256='wrong',restore_random=False)
 assert path.exists()
 with pytest.raises(ValueError):verify_checkpoint_file(path,'0'*64)
 assert path.exists()

def test_grad_inputs_are_detached_without_source_aliases():
 b=batch();b['obs']['proprio'].requires_grad_(True);b['actions'].requires_grad_(True)
 r=TaskReservoir(capacity=3);r.add(b)
 before=r.data[TASKS[0]]['obs']['proprio'][0].clone()
 assert all(t.grad_fn is None and not t.requires_grad for t in r.data[TASKS[0]]['obs'].values())
 assert r.data[TASKS[0]]['actions'].grad_fn is None
 with torch.no_grad():b['obs']['proprio'].fill_(99)
 assert torch.equal(before,r.data[TASKS[0]]['obs']['proprio'][0])

def test_real_worker_periodic_resume_and_boundary_save(tmp_path,monkeypatch):
 """Exercise actual worker and torch checkpoints using CPU model/env seams.

 The collector substitutes tiny observation schemas and virtual sample accounting;
 this is an engineering persistence test, not physics or behavior evidence.
 """
 import src.tasks.velocity.v10.runtime as rt
 import src.tasks.velocity.v10.env as env_module
 from src.tasks.velocity.v10.checkpoint import BudgetLedger
 class Env:
  def __init__(self,num_envs,course,seed,device):self.num_envs=num_envs;self.course=course
  def state_dict(self):return {'cursor':3}
  def load_state_dict(self,state):assert state=={'cursor':3}
  def close(self):pass
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.actor=torch.nn.Linear(45,12)
  @classmethod
  def from_stairs(cls,device):return cls().to(device)
  def set_phase(self,phase):pass
  def forward(self,obs):return self.actor(obs['proprio'])
 class Learner:
  def __init__(self,model):self.model=model;self.optimizer=torch.optim.Adam(model.parameters(),lr=3e-5)
  def update(self,current,reservoir,num_envs):
   self.optimizer.zero_grad();loss=self.model(current['obs']).square().mean();loss.backward();self.optimizer.step();reservoir.add(current);return {'loss':float(loss.detach())}
 def collect(env,model,teachers,phase,ledger,kind='learning',budget_limit=None,steps=48):
  import uuid
  ledger.charge(phase,env.num_envs*48,limit=budget_limit,kind=kind)
  return batch(),dict(rollout_id=uuid.uuid4().hex,episodes=[])
 monkeypatch.setattr(env_module,'V10Env',Env);monkeypatch.setattr(rt,'V10ActorCritic',Model)
 monkeypatch.setattr(rt,'DAggerLearner',Learner);monkeypatch.setattr(rt,'frozen_teachers',lambda device:None)
 monkeypatch.setattr(rt,'physical_contract',lambda env:{'sha256':'cpu-worker-seam','contract':{}})
 monkeypatch.setattr(rt,'collect_rollout',collect);monkeypatch.setattr(rt,'release_host_memory',lambda:None)
 # Keep actual save period 2,457,600 and actual 512*48 unit; 100 rounds crosses it.
 original_reservoir=rt.TaskReservoir
 monkeypatch.setattr(rt,'TaskReservoir',lambda:original_reservoir(capacity=3))
 ledger=tmp_path/'costs.jsonl'
 first=rt.worker(course='D1',num_envs=512,output_dir=tmp_path/'first',stop_round=100,device='cpu',ledger_path=ledger)
 periodic=tmp_path/'first'/'D1_000100.pt';boundary=Path(first['checkpoint'])
 assert periodic.exists() and boundary.exists() and periodic.stat().st_ino==boundary.stat().st_ino
 assert first['checkpoint_sha256']==sha256(boundary)
 # Resume full worker; next 100 rounds crosses the next actual save boundary.
 second=rt.worker(course='D1',num_envs=512,output_dir=tmp_path/'second',stop_round=200,resume=boundary,device='cpu',ledger_path=ledger)
 assert (tmp_path/'second'/'D1_000200.pt').exists()
 assert second['consumed']==2*2457600 and second['updated_samples']==2*2457600
 assert second['checkpoint_sha256']==sha256(second['checkpoint'])
 assert BudgetLedger(ledger).report()['updated']['learning:D1']==2*2457600

def test_preallocated_rollout_keeps_exact_frame_snapshots_and_integer_metadata():
 from src.tasks.velocity.v10.runtime import allocate_rollout,store_rollout_frame
 from src.tasks.velocity.v10.replay import concat_batches,snapshot
 frames=[];source=batch();source['episode']=torch.arange(6,dtype=torch.int64)+2**40
 storage=allocate_rollout(source,18)
 for i in range(3):
  source['obs']['proprio'].add_(1);source['episode'].add_(1)
  frames.append(snapshot(source));store_rollout_frame(storage,source,i*6,6)
 expected=concat_batches(frames)
 assert torch.equal(storage['obs']['proprio'],expected['obs']['proprio'])
 assert torch.equal(storage['episode'],expected['episode']) and storage['episode'].dtype==torch.int64
 source['obs']['proprio'].zero_();assert storage['obs']['proprio'].abs().sum()>0

def test_task_index_composition_preserves_sampler_rng_and_logical_batch():
 from src.tasks.velocity.v10.replay import index_batch,concat_batches
 r=TaskReservoir(capacity=3,seed=42);b=batch();r.add(b)
 g=torch.Generator();g.set_state(r.rng.get_state());pieces=[]
 for task in TASKS:
  ids=torch.tensor([i for i,t in enumerate(b['task']) if t==task]);task_batch=index_batch(b,ids)
  pieces.append(index_batch(task_batch,torch.randint(len(ids),(256,),generator=g)))
  pieces.append(index_batch(r.data[task],torch.randint(1,(1,),generator=g)))
  pieces.append(index_batch(task_batch,torch.randint(len(ids),(255,),generator=g)))
 expected=concat_batches(pieces);actual=r.sample_logical(b)
 assert torch.equal(actual['obs']['proprio'],expected['obs']['proprio'])
 assert torch.equal(actual['actions'],expected['actions']) and actual['episode']==expected['episode']
 assert torch.equal(r.rng.get_state(),g.get_state())

@pytest.mark.skipif(__import__('os').environ.get('V10_FULL_REPLAY_CPU')!='1',reason='explicit 3GB CPU save/restore equivalence check')
def test_full_196608_real_shape_replay_and_learning_state_roundtrip(tmp_path,monkeypatch):
 from src.tasks.velocity.v10.model import V10ActorCritic
 from src.tasks.velocity.v10.checkpoint import state_fingerprint,rng_state,restore_rng,release_host_memory,memory_observation
 from src.tasks.velocity.v10 import checkpoint as cp
 import json
 monkeypatch.setattr(cp,'code_fingerprint',lambda:'q4-full-cpu-fixture')
 original_threads=torch.get_num_threads();torch.set_num_threads(1)
 trace=[]
 def mark(stage):trace.append(memory_observation(stage))
 r=TaskReservoir();model=V10ActorCritic.from_stairs();model.set_phase('P1')
 optimizer=torch.optim.Adam(model.parameters(),lr=3e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
 x=dict(proprio=torch.zeros(2,45),history=torch.zeros(2,5,45),heightmap=torch.zeros(2,3,40,30),route_error=torch.zeros(2,3),privileged=torch.zeros(2,35))
 loss=model(x).square().mean()+model.evaluate(x).square().mean()+model.std.sum();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
 shapes=dict(proprio=(45,),history=(5,45),heightmap=(3,40,30),route_error=(3,),privileged=(35,))
 mark('before_fill')
 for number,task in enumerate(TASKS):
  n=r.capacity
  r.data[task]=dict(obs={k:torch.full((n,*shape),number/10,dtype=torch.float32) for k,shape in shapes.items()},actions=torch.full((n,12),number/100,dtype=torch.float32),task=[task]*n,teacher=['stairs' if task=='S_ROUTE' else 'blind']*n,case=[f'{task}:{i}' for i in range(n)],episode=torch.arange(n,dtype=torch.int64)+2**40,control_step=torch.arange(n,dtype=torch.int64))
  r.seen[task]=n+1000
 mark('all_196608_slots_filled')
 assert sum(min(r.seen[k],r.capacity) for k in TASKS)==196608
 expected=r.integrity_manifest();next_sample=r.next_sample_manifest();sampler_rng=r.rng.get_state().clone()
 assert torch.equal(sampler_rng,r.rng.get_state())
 model_sha=state_fingerprint(model.state_dict());adam_sha=state_fingerprint(optimizer.state_dict());global_rng=rng_state();rng_sha=state_fingerprint(global_rng)
 path=tmp_path/'full.pt'
 digest=checkpoint_save(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),reservoir=r.state_dict(copy_storage=False),num_envs=512,course='D1',physical_sha256='q4-cpu-full'),memory_trace=trace.append)
 assert path.stat().st_size>3_000_000_000
 r.release_storage();release_host_memory();mark('released_before_load')
 state=checkpoint_load(path,num_envs=512,physical_sha256='q4-cpu-full',memory_trace=trace.append)
 model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);r.take_state_dict(state.pop('reservoir'))
 state_rng=state['rng'];assert state_fingerprint(state_rng)==rng_sha
 assert r.integrity_manifest()==expected and r.next_sample_manifest()==next_sample
 assert torch.equal(r.rng.get_state(),sampler_rng)
 assert state_fingerprint(model.state_dict())==model_sha and state_fingerprint(optimizer.state_dict())==adam_sha
 assert all(torch.equal(t,model.state_dict()[k]) for k,t in state['model'].items() if 'normalizer' in k)
 assert r.data[TASKS[0]]['episode'].dtype==torch.int64 and r.data[TASKS[0]]['episode'][0].item()==2**40
 state.clear();del state,state_rng;release_host_memory();mark('restored_unique_owner')
 restore_rng(global_rng);assert state_fingerprint(rng_state())==rng_sha
 assert sha256(path)==digest
 evidence=Path('logs/a2_v10/q4_cpu_full_replay_memory.json');evidence.parent.mkdir(parents=True,exist_ok=True)
 evidence.write_text(json.dumps(dict(slots=196608,obs_shapes=shapes,dtype='torch.float32',physical_samples=0,checkpoint_bytes=path.stat().st_size,checkpoint_sha256=digest,replay=expected,next_sample_manifest=next_sample,memory_stages=trace),indent=2))
 r.release_storage();del model,optimizer;release_host_memory();torch.set_num_threads(original_threads)

def test_preallocated_collector_matches_old_snapshot_collector_across_resets():
 from src.tasks.velocity.v10.runtime import collect_rollout
 from src.tasks.velocity.v10.replay import snapshot,concat_batches
 class Env:
  num_envs=2
  def __init__(self):self.episode=[0,0];self.steps=[0,0];self.tick=0
  def observe(self):return {'proprio':torch.tensor([[self.episode[i],self.steps[i]] for i in range(2)],dtype=torch.float32)}
  def metadata(self):return dict(task=['F_WALK','S_ROUTE'],teacher=['blind','stairs' if self.steps[1]<1 else 'blind'],case=[f'c{x}' for x in self.episode],episode=list(self.episode),control_step=list(self.steps))
  def step(self,action):
   self.tick+=1;done=torch.tensor([self.tick%2==0,self.tick%3==0])
   for i in range(2):
    if done[i]:self.episode[i]+=1;self.steps[i]=0
    else:self.steps[i]+=1
   return self.observe(),torch.zeros(2),done,{'frame':{'action':action.clone()},'episodes':[]}
 class Model:
  def __call__(self,o):return o['proprio'].sum(-1,keepdim=True).expand(-1,12).clone()
 class Teachers:
  def label(self,o,stairs):return o['proprio'].sum(-1,keepdim=True).expand(-1,12)+torch.tensor(stairs)[:,None]
 a=Env();b=Env();m=Model();t=Teachers();parts=[]
 for _ in range(7):
  o=a.observe();meta=a.metadata();label=t.label(o,[k=='stairs' for k in meta['teacher']]);action=m(o)
  a.step(action);parts.append(snapshot(dict(obs=o,actions=label,**meta)))
 expected=concat_batches(parts);actual,_=collect_rollout(b,m,t,phase='D1',steps=7)
 assert torch.equal(expected['obs']['proprio'],actual['obs']['proprio'])
 assert torch.equal(expected['actions'],actual['actions'])
 assert all(actual[k]==expected[k] for k in ('task','teacher','case','episode','control_step'))
 # Updating history after the rollout must preserve both old/new snapshots.
 for _ in range(4):b.step(torch.zeros(2,12))
 assert torch.equal(expected['obs']['proprio'],actual['obs']['proprio'])

def test_resource_summary_includes_phase_only_low_available_and_rss_hwm(monkeypatch):
 from src.tasks.velocity.v10.resources import ResourceMonitor
 monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
 monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
 monitor=ResourceMonitor();monitor.rows=[dict(gpu_total_mib=1000,gpu_task_mib=900,gpu_task_seen=True,host_available_bytes=8*1024**3,process_rss_bytes=1024**3)]
 monitor.memory_stage(dict(stage='instant-peak',host_available_bytes=4*1024**3-1,process_rss_bytes=2*1024**3,process_peak_rss_bytes=5*1024**3))
 summary=monitor.summary();assert not summary['passed']
 assert summary['host_available_min_bytes']==4*1024**3-1
 assert summary['process_rss_peak_bytes']==5*1024**3
