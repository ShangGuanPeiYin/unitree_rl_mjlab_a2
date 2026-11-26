"""SDS8.1 training telemetry: real collector -> real worker CPU seams only."""
import json
import pytest
import torch
from src.tasks.velocity.v10 import runtime as rt,spec
from src.tasks.velocity.v10.replay import TaskReservoir

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 from src.tasks.velocity.v10 import checkpoint as cp
 monkeypatch.setattr(cp,'code_fingerprint',lambda:'task-logging-cpu-fixture')

class Env:
 instances=[]
 def __init__(self,num_envs=512,course='D1',seed=42,device='cpu'):
  self.num_envs=num_envs;self.course=course;self.calls=0
  self.roles=[role for role,n in zip(spec.ROLES,spec.role_counts(num_envs)) for _ in range(n)]
  self.ticks=[0]*num_envs;self.episodes=[0]*num_envs;Env.instances.append(self)
 def observe(self):return {'proprio':torch.tensor([[step/100]*45 for step in self.ticks])}
 def metadata(self):
  return dict(task=list(self.roles),teacher=['stairs' if role=='S_ROUTE' and self.ticks[i]<30 else 'blind' for i,role in enumerate(self.roles)],case=[f'case:{i}:{self.episodes[i]}' for i in range(self.num_envs)],episode=list(self.episodes),control_step=list(self.ticks))
 def step(self,action):
  self.calls+=1;terminal=[];done=torch.zeros(self.num_envs,dtype=torch.bool)
  for i,role in enumerate(self.roles):
   self.ticks[i]+=1
   if role=='F_WALK' and self.ticks[i]==60:
    done[i]=True;terminal.append(dict(task=role,env_id=i,episode=self.episodes[i],steps=60,success=i%2==0))
    self.ticks[i]=0;self.episodes[i]+=1
  return self.observe(),torch.zeros(self.num_envs),done,dict(frame={'action':action.clone()},episodes=terminal)
 def state_dict(self):return {'ticks':list(self.ticks),'episodes':list(self.episodes)}
 def load_state_dict(self,state):
  # Physical restart must not turn the abandoned partial episode into a failure.
  self.episodes=[e+1 for e in state['episodes']];self.ticks=[0]*self.num_envs
 def close(self):pass

class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.actor=torch.nn.Linear(45,12)
 @classmethod
 def from_stairs(cls,device):return cls().to(device)
 def set_phase(self,course):pass
 def forward(self,obs):return self.actor(obs['proprio'])
 def action_distribution(self,obs):return torch.distributions.Normal(self(obs),.1)
 def evaluate(self,obs):return obs['proprio'][:,0:1]*0

class Teachers:
 def label(self,obs,stairs):return torch.tensor(stairs)[:,None].float().expand(-1,12).clone()

class Learner:
 def __init__(self,model):self.model=model;self.optimizer=torch.optim.Adam(model.parameters(),lr=3e-5)
 def update(self,current,reservoir,num_envs):
  self.optimizer.zero_grad();loss=self.model(current['obs']).square().mean();loss.backward();self.optimizer.step()
  reservoir.add(current);return {'loss':float(loss.detach())}

def test_actual_collector_worker_persists_task_denominators_and_real_episode_lengths(tmp_path,monkeypatch):
 import src.tasks.velocity.v10.env as env_module
 monkeypatch.setattr(env_module,'V10Env',Env);monkeypatch.setattr(rt,'V10ActorCritic',Model)
 monkeypatch.setattr(rt,'DAggerLearner',Learner);monkeypatch.setattr(rt,'frozen_teachers',lambda device:Teachers())
 monkeypatch.setattr(rt,'TaskReservoir',lambda:TaskReservoir(capacity=3));monkeypatch.setattr(rt,'release_host_memory',lambda:None)
 monkeypatch.setattr(rt,'physical_contract',lambda env:{'sha256':'task-log-cpu'})
 ledger=tmp_path/'costs.jsonl'
 first=rt.worker(course='D1',num_envs=512,output_dir=tmp_path/'run',stop_round=2,device='cpu',ledger_path=ledger)
 rows=[json.loads(line) for line in (tmp_path/'run'/'updates.jsonl').read_text().splitlines()]
 assert len(rows)==2 and rows[0]['rollout_id']!=rows[1]['rollout_id']
 assert rows[0]['physical_session_id']==rows[1]['physical_session_id']
 counts=dict(zip(spec.ROLES,spec.role_counts(512)))
 for j,row in enumerate(rows):
  assert row['collection_statistics_scope']=='training_rollout'
  for role,count in counts.items():
   item=row['per_task'][role]
   assert item['collected_samples']==48*count and item['teacher_labels_expected']==48*count
   assert item['teacher_labels_labeled']==48*count and item['teacher_label_coverage']==1
   assert sum(item['teacher_label_sources'].values())==48*count
  walk=row['per_task']['F_WALK']
  assert walk['completed_episodes']==(0 if j==0 else counts['F_WALK'])
  assert walk['completed_episode_length_steps']['mean']==(None if j==0 else 60)
  assert walk['failure_rate']==(None if j==0 else .5)
  assert walk['unfinished_episodes_at_rollout_end']==counts['F_WALK']
  assert walk['unfinished_episode_length_steps']['mean']==(48 if j==0 else 36)
 assert rows[0]['per_task']['S_ROUTE']['teacher_label_sources']=={'blind':18*counts['S_ROUTE'],'stairs':30*counts['S_ROUTE']}
 assert rows[1]['per_task']['S_ROUTE']['teacher_label_sources']=={'blind':48*counts['S_ROUTE'],'stairs':0}
 # Actual worker restore starts new physical episodes. Partial old episodes are gauges, not failures/completions.
 rt.worker(course='D1',num_envs=512,output_dir=tmp_path/'restored',stop_round=3,resume=first['checkpoint'],device='cpu',ledger_path=ledger)
 resumed=json.loads((tmp_path/'restored'/'updates.jsonl').read_text())
 assert resumed['physical_session_id']!=rows[0]['physical_session_id']
 assert resumed['per_task']['F_WALK']['completed_episodes']==0
 assert resumed['per_task']['F_WALK']['failures']==0
 assert resumed['per_task']['F_WALK']['failure_rate'] is None

def test_ppo_teacher_denominators_are_explicitly_not_applicable():
 env=Env();_,metrics=rt.collect_rollout(env,Model(),None,phase='P0',steps=2)
 for role,count in zip(spec.ROLES,spec.role_counts(512)):
  item=metrics['per_task'][role]
  assert item['collected_samples']==count*2 and not item['teacher_labels_applicable']
  assert item['teacher_labels_expected']==item['teacher_labels_labeled']==0
  assert item['teacher_label_coverage'] is None and sum(item['teacher_label_sources'].values())==0

@pytest.mark.parametrize('bad',['shape','nan','not_tensor'])
def test_invalid_teacher_labels_fail_before_step_or_cost(tmp_path,bad):
 from src.tasks.velocity.v10.checkpoint import BudgetLedger
 class Bad:
  def label(self,obs,stairs):
   if bad=='shape':return torch.zeros(len(stairs),11)
   if bad=='nan':return torch.full((len(stairs),12),torch.nan)
   return [[0.]*12]*len(stairs)
 env=Env();ledger=BudgetLedger(tmp_path/'costs')
 with pytest.raises(ValueError,match='teacher labels'):
  rt.collect_rollout(env,Model(),Bad(),phase='D1',ledger=ledger,budget_limit=100000)
 assert env.calls==0 and not ledger.totals()
