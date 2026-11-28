"""Minimal CPU migration tests: small state payloads, no physical training."""
import json,random
from pathlib import Path
import pytest,torch
from src.tasks.velocity.v10 import checkpoint as cp
from src.tasks.velocity.v10.replay import TaskReservoir,TASKS
OLD='a'*64;NEW='b'*64;PHYS='c'*64


def test_dynamic_dagger_keeps_twenty_updates_and_legacy_512_ten():
 from src.tasks.velocity.v10.dagger import DAggerLearner
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.actor=torch.nn.Linear(2,12)
  def forward(self,obs):return self.actor(obs['proprio'])
 batch=dict(obs={'proprio':torch.ones(6,2)},actions=torch.zeros(6,12))
 class Replay:
  def sample_logical(self,current):return batch
  def add(self,current):pass
 for n,expected in ((2048,20),(1024,20),(512,10)):
  learner=DAggerLearner(Model());step=learner.optimizer.step;count=[]
  def counted():count.append(1);return step()
  learner.optimizer.step=counted;learner.update(batch,Replay(),n)
  assert len(count)==expected


def test_env_N_migration_is_explicit_and_resets_only_episode_slots():
 from src.tasks.velocity.v10.env import V10Env
 env=V10Env.__new__(V10Env);env.num_envs=8;env.rng=random.Random(1)
 state=dict(rng=random.Random(42).getstate(),role_cursors={t:7 for t in TASKS},episode_ids=list(range(1024)),control_steps=2400)
 env.reset=lambda:None
 with pytest.raises(ValueError):env.load_state_dict(state)
 env.load_state_dict(state,allow_num_env_change=True)
 assert env.episode_ids==[-1]*8 and env.role_cursors==state['role_cursors']
 assert env.rng.getstate()==state['rng'] and env.control_steps==2400

@pytest.fixture
def seam(tmp_path,monkeypatch):
 from src.tasks.velocity.v10 import execution_restore as er,dynamic_execution as de,runtime as rt,handoff
 import src.tasks.velocity.v10.env as em
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False);monkeypatch.setattr(cp,'code_fingerprint',lambda:OLD)
 model=torch.nn.Linear(45,12);opt=torch.optim.Adam(model.parameters(),lr=3e-5)
 model(torch.ones(6,45)).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
 batch=dict(obs={'proprio':torch.ones(6,45)},actions=torch.zeros(6,12),task=list(TASKS),teacher=['blind']*6,case=['c']*6,episode=[0]*6,control_step=[0]*6)
 replay=TaskReservoir(capacity=1);replay.add(batch)
 ledger=tmp_path/'learning.jsonl';account=cp.BudgetLedger(ledger);account.charge('D1',2457600,limit=24576000,kind='learning');account.record_update('old','D1',2457600)
 envstate=dict(rng=random.Random(42).getstate(),role_cursors={t:7 for t in TASKS},episode_ids=list(range(1024)),control_steps=2400)
 origin=tmp_path/'old.pt';cp.checkpoint_save(origin,dict(model=model.state_dict(),learner={'optimizer':opt.state_dict()},reservoir=replay.state_dict(),env=envstate,budgets=account.totals(),accounting=account.report(),rounds={'D1':50},course='D1',num_envs=1024,physical_sha256=PHYS,keep=None,extra={'trial':{'retained_completed_updates':50}}))
 raw=tmp_path/'profile.json';raw.write_text('{}');ref={'path':str(origin),'sha256':cp.sha256(origin),'from_num_envs':1024,'from_code_sha256':OLD}
 doc=dict(origin_checkpoint=ref,origin_rounds={'D1':50,'D2':0,'D3':0,'PPO':0},accounting_origin=account.report(),physical_sha256=PHYS,recovery_registry_path=str(ledger)+'.dynamic_latest.json',num_envs=2048,resource_status='unverified_for_requested_N')
 profile=dict(path=str(raw),sha256=cp.sha256(raw),document=doc,origin_checkpoint=ref,learning_ledger_path=str(ledger),num_envs=2048,round_budgets={'D1':500,'D2':1000,'D3':1500,'PPO':1000,'P0':100},save_rounds=50,gates={'D1':[200,400,500]},sample_caps={'D1':2457600+450*2048*48,'D2':1000*2048*48,'D3':1500*2048*48,'PPO':1000*2048*48,'P0':100*2048*48})
 def validate(path,**kw):
  assert Path(path)==raw
  if kw.get('num_envs') not in (None,2048):raise ValueError('profile N mismatch')
  if kw.get('physical_sha256') not in (None,PHYS):raise ValueError('profile physics mismatch')
  if kw.get('ledger_path') is not None and Path(kw['ledger_path'])!=ledger:raise ValueError('ledger mismatch')
  return profile
 monkeypatch.setattr(de,'validate_execution_profile',validate);monkeypatch.setattr(cp,'code_fingerprint',lambda:NEW);monkeypatch.setattr(er,'code_fingerprint',lambda:NEW)
 monkeypatch.setattr(handoff,'_teachers',lambda receipts,plan:{k:{'passed':True} for k in receipts})
 calls={'updates':0,'collect':0};expected=dict(model=cp.state_fingerprint(model.state_dict()),optimizer=cp.state_fingerprint(opt.state_dict()),replay=replay.integrity_manifest())
 class Env:
  def __init__(self,num_envs,course,**kw):self.num_envs=num_envs;self.course=course
  def load_state_dict(self,state,allow_num_env_change=False):
   assert allow_num_env_change and state==envstate;calls['env_migrated']=True
  def state_dict(self):return dict(envstate,episode_ids=[0]*2048)
  def close(self):pass
 class Model(torch.nn.Linear):
  @classmethod
  def from_stairs(cls,device):return cls(45,12).to(device)
  def set_phase(self,p):pass
 class Learner:
  def __init__(self,m):self.model=m;self.optimizer=torch.optim.Adam(m.parameters(),lr=3e-5)
  def update(self,b,r,n):
   if not calls['updates']:
    assert cp.state_fingerprint(self.optimizer.state_dict())==expected['optimizer'];assert r.integrity_manifest()==expected['replay']
   calls['updates']+=1;self.optimizer.zero_grad();loss=self.model(b['obs']['proprio']).square().mean();loss.backward();self.optimizer.step();r.add(b);return {'loss':loss.item()}
 def collect(env,model,teachers,phase,ledger,budget_limit):
  if not calls['collect']:assert cp.state_fingerprint(model.state_dict())==expected['model']
  calls['collect']+=1;ledger.charge('D1',2048*48,limit=budget_limit,kind='learning')
  return batch,dict(rollout_id=f'new-{calls["collect"]}',episodes=[],per_task={})
 monkeypatch.setattr(em,'V10Env',Env);monkeypatch.setattr(rt,'V10ActorCritic',Model);monkeypatch.setattr(rt,'DAggerLearner',Learner);monkeypatch.setattr(rt,'TaskReservoir',lambda:TaskReservoir(capacity=1))
 monkeypatch.setattr(rt,'physical_contract',lambda e:{'sha256':PHYS,'contract':{}});monkeypatch.setattr(rt,'frozen_teachers',lambda d:None);monkeypatch.setattr(rt,'collect_rollout',collect);monkeypatch.setattr(rt,'release_host_memory',lambda:None)
 from src.tasks.velocity.v10 import identity
 monkeypatch.setattr(identity,'code_fingerprint',lambda:NEW)
 snapshot=dict(worker_operation_id='learning_D1_51',teacher_admitted=True,execution_profile_sha256=profile['sha256'],execution_teacher_receipts={k:{} for k in handoff.TEACHER_KEYS})
 args=dict(course='D1',num_envs=2048,stop_round=51,output_dir=tmp_path/'new',resume=str(origin),ledger_path=str(ledger),device='cpu',execution_profile=str(raw),controller_state=snapshot)
 return er,rt,profile,args,calls


def test_checkpoint_migration_requires_verified_profile(seam):
 er,rt,p,args,calls=seam
 with pytest.raises(ValueError):cp.checkpoint_load(args['resume'],num_envs=2048,physical_sha256=PHYS,restore_random=False)
 state=cp.checkpoint_load(args['resume'],num_envs=2048,physical_sha256=PHYS,restore_random=False,execution_profile=p)
 assert state['num_envs']==1024 and state['rounds']['D1']==50
 with pytest.raises(ValueError):cp.checkpoint_load(args['resume'],num_envs=2048,physical_sha256=PHYS,restore_random=False,execution_profile={'origin_checkpoint':p['origin_checkpoint']})


def test_worker_preserves_fifty_then_adds_one_new_N_round(seam):
 er,rt,p,args,calls=seam;result=rt.worker(**args)
 assert calls['updates']==1 and calls['env_migrated'] and result['round']==51
 assert result['consumed']==2457600+2048*48 and result['gate_reached'] is True
 state=cp.checkpoint_load(result['checkpoint'],num_envs=2048,physical_sha256=PHYS,restore_random=False,execution_profile=p)
 assert state['extra']['trial']['retained_completed_updates']==50 and state['rounds']['D1']==51
 assert Path(p['document']['recovery_registry_path']).is_file()


def test_dynamic_worker_rejects_missing_admission_before_sampling(seam):
 er,rt,p,args,calls=seam;args['controller_state']=dict(args['controller_state'],teacher_admitted=False)
 with pytest.raises(ValueError):rt.worker(**args)
 assert calls['collect']==0


def test_dynamic_low_host_memory_stops_without_new_sampling(seam,monkeypatch):
 er,rt,p,args,calls=seam
 monkeypatch.setattr(rt,'memory_observation',lambda stage:{'host_available_bytes':4*1024**3-1})
 with pytest.raises(RuntimeError,match='below 4GiB'):rt.worker(**args)
 assert calls['collect']==0


def test_fresh_worker_initial_save_then_one_round_and_reload(seam,monkeypatch):
 er,rt,p,args,calls=seam
 from src.tasks.velocity.v10 import env as em
 p['origin_checkpoint']=None;p['document']['origin_rounds']={k:0 for k in p['round_budgets']}
 # The fixture validator checks its existing path, so reset only that fixture's
 # ledger bytes for this CPU fresh-path test (no real/project ledger involved).
 Path(args['ledger_path']).write_text('')
 args.update(resume=None,stop_round=0)
 monkeypatch.setattr(rt,'memory_observation',lambda stage:{'host_available_bytes':16*1024**3})
 first=rt.worker(**args)
 assert first['round']==0 and calls['collect']==0
 state=cp.checkpoint_load(first['checkpoint'],num_envs=2048,physical_sha256=PHYS,restore_random=False,execution_profile=p)
 assert state['extra']['execution_profile']==er.provenance(p)
 assert 'trial' not in state['extra']
 monkeypatch.setattr(em.V10Env,'load_state_dict',lambda self,state,**kw:None)
 class Learner:
  def __init__(self,m):self.optimizer=torch.optim.Adam(m.parameters(),lr=3e-5)
  def update(self,b,r,n):return {'loss':0.}
 monkeypatch.setattr(rt,'DAggerLearner',Learner)
 def collect(env,model,teachers,phase,ledger,budget_limit):
  ledger.charge('D1',2048*48,limit=budget_limit,kind='learning')
  return {},dict(rollout_id='fresh-one',episodes=[],per_task={})
 monkeypatch.setattr(rt,'collect_rollout',collect)
 args.update(resume=first['checkpoint'],stop_round=1)
 result=rt.worker(**args)
 assert result['round']==1 and result['consumed']==2048*48
 restored=cp.checkpoint_load(result['checkpoint'],num_envs=2048,physical_sha256=PHYS,restore_random=False,execution_profile=p)
 assert restored['rounds']['D1']==1
