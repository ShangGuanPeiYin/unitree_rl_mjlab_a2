"""CPU-only trial accounting/persistence tests; virtual counts are not physics."""
from pathlib import Path
import uuid
import pytest
import torch
from src.tasks.velocity.v10.checkpoint import BudgetLedger, checkpoint_load, sha256

@pytest.fixture
def guard(tmp_path,monkeypatch):
 from src.tasks.velocity.v10 import trial_guard as tg
 monkeypatch.setattr(tg,'CANONICAL_GUARD',tmp_path/'canonical.json')
 return tg

def open_guard(g,tmp_path,**kw):
 args=dict(course='D1',num_envs=1024,stop_round=50,cap=2457600,trial_id='fixed',guard_path=g.CANONICAL_GUARD,ledger_path=tmp_path/'ledger',resume=None)
 args.update(kw);return g.TrialGuard.open(**args)

def test_fixed_binding_and_explicit_restrictions(guard,tmp_path):
 g=open_guard(guard,tmp_path);assert g.retained==0
 assert open_guard(guard,tmp_path).identity==g.identity
 for kw in (dict(trial_id='new'),dict(ledger_path=tmp_path/'new'),dict(guard_path=tmp_path/'newguard'),dict(num_envs=512),dict(course='D2'),dict(stop_round=51),dict(cap=2457601),dict(ledger_path=None)):
  with pytest.raises(ValueError):open_guard(guard,tmp_path,**kw)

@pytest.mark.parametrize('executed_before_crash',[False,True])
def test_uncertain_crash_no_refund_and_whole_rollout_boundary(guard,tmp_path,executed_before_crash):
 g=open_guard(guard,tmp_path);ledger=BudgetLedger(tmp_path/'ledger')
 # Both physical outcomes before durable mark remain uncertain and charged.
 ledger.reserve('crash-step','D1',1024,limit=g.cap,kind='learning')
 if executed_before_crash:pass # Physics happened, mark was never persisted.
 ledger.reconcile_uncertain();ledger.reconcile_uncertain()
 with pytest.raises(ValueError):open_guard(guard,tmp_path)
 again=open_guard(guard,tmp_path,resume='initial.pt')
 assert ledger.totals()['learning:D1']==1024
 assert again.can_collect(2457600-49152)
 assert not again.can_collect(2457600-49152+1024)
 assert again.stop_reason=='trial_budget_exhausted'

@pytest.fixture
def worker_seam(tmp_path,monkeypatch,guard):
 import src.tasks.velocity.v10.runtime as rt
 import src.tasks.velocity.v10.env as em
 from src.tasks.velocity.v10 import checkpoint as cp
 from src.tasks.velocity.v10.replay import TaskReservoir,TASKS
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 monkeypatch.setattr(cp,'code_fingerprint',lambda:'trial-cpu-seam')
 calls={'updates':0,'collect':0,'env':0}
 class Env:
  def __init__(self,num_envs,course,**kw):self.num_envs=num_envs;self.course=course;calls['env']+=1
  def state_dict(self):return {}
  def load_state_dict(self,s):pass
  def close(self):pass
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.actor=torch.nn.Linear(45,12)
  @classmethod
  def from_stairs(cls,device):return cls().to(device)
  def set_phase(self,p):pass
 class Learner:
  def __init__(self,m):self.model=m;self.optimizer=torch.optim.Adam(m.parameters(),lr=3e-5)
  def update(self,b,r,n):
   calls['updates']+=1;self.optimizer.zero_grad();loss=self.model.actor(b['obs']['proprio']).square().mean();loss.backward();self.optimizer.step();r.add(b);return {'loss':loss.item()}
 def collect(env,model,teachers,phase,ledger,budget_limit):
  calls['collect']+=1;uid=uuid.uuid4().hex
  # Virtual full-rollout debit for this persistence seam; crash tests below
  # exercise real Q1 reserve/finalize. Production collector is unchanged.
  ledger.charge('D1',1024*48,kind='learning',limit=budget_limit)
  batch=dict(obs={'proprio':torch.ones(6,45)},actions=torch.zeros(6,12),task=list(TASKS),teacher=['blind']*6,case=['c']*6,episode=[0]*6,control_step=[0]*6)
  return batch,dict(rollout_id=uid,episodes=[],per_task={})
 monkeypatch.setattr(em,'V10Env',Env);monkeypatch.setattr(rt,'V10ActorCritic',Model);monkeypatch.setattr(rt,'DAggerLearner',Learner)
 monkeypatch.setattr(rt,'frozen_teachers',lambda device:None);monkeypatch.setattr(rt,'physical_contract',lambda e:{'sha256':'trial-cpu','contract':{}})
 monkeypatch.setattr(rt,'collect_rollout',collect);monkeypatch.setattr(rt,'TaskReservoir',lambda:TaskReservoir(capacity=1));monkeypatch.setattr(rt,'release_host_memory',lambda:None)
 args=dict(course='D1',num_envs=1024,stop_round=50,output_dir=tmp_path/'out',device='cpu',ledger_path=tmp_path/'ledger',trial_charged_cap=2457600,trial_id='fixed',trial_guard=guard.CANONICAL_GUARD)
 return rt,args,calls

def test_worker_normal_fifty_persists_trial_and_default_cannot_bypass(worker_seam):
 rt,args,calls=worker_seam;result=rt.worker(**args)
 assert calls['updates']==50 and result['stop_reason']=='trial_limit'
 assert result['retained_completed_updates']==50 and result['updated_samples']==2457600
 assert result['historical_updated_samples']==2457600
 assert result['gate_reached'] is False and result['accepted'] is None and result['teacher_admitted'] is False
 state=checkpoint_load(result['checkpoint'],num_envs=1024,physical_sha256='trial-cpu',restore_random=False)
 assert state['extra']['trial']['retained_completed_updates']==50
 assert state['extra']['trial']['pre_gate_adjusted_trial'] is True
 assert sha256(result['checkpoint'])==result['checkpoint_sha256']
 # Repeated recovery neither adds accounting nor updates.
 again=rt.worker(**dict(args,output_dir=Path(args['output_dir']).parent/'repeat',resume=result['checkpoint']))
 assert calls['updates']==50 and again['stop_reason']=='trial_limit'
 plain={k:v for k,v in args.items() if not k.startswith('trial_')}
 with pytest.raises(ValueError,match='trial'):rt.worker(**dict(plain,resume=result['checkpoint']))
 with pytest.raises(ValueError):rt.worker(**args)

def test_old_initial_restore_does_not_claim_historical_updates(worker_seam):
 rt,args,calls=worker_seam;first=rt.worker(**args)
 initial=Path(args['output_dir'])/'trial_initial.pt';assert initial.exists()
 old=rt.worker(**dict(args,resume=initial,output_dir=Path(args['output_dir']).parent/'old'))
 assert old['stop_reason']=='trial_budget_exhausted' and old['retained_completed_updates']==0
 assert old['updated_samples']==0 and old['historical_updated_samples']==2457600
 assert old['round']==0 and old['confirmed_sample_round']==50
 assert calls['updates']==50

def test_initial_save_failure_never_collects(worker_seam,monkeypatch):
 rt,args,calls=worker_seam
 def fail(*a,**kw):raise OSError('disk full')
 monkeypatch.setattr(rt,'save_worker',fail)
 with pytest.raises(OSError,match='disk full'):rt.worker(**args)
 assert calls['updates']==calls['collect']==0
 assert not (Path(args['output_dir'])/'worker_result.json').exists()

@pytest.mark.parametrize('crash_point',['reserved','executed_unmarked','marked'])
def test_uncertain_initial_resume_stops_at_49_and_repeated_resume_is_idempotent(worker_seam,monkeypatch,crash_point):
 rt,args,calls=worker_seam;original=rt.collect_rollout
 def crash(env,model,teachers,phase,ledger,budget_limit):
  ledger.reserve('unmarked-step','D1',1024,kind='learning',limit=budget_limit)
  if crash_point!='reserved':calls['virtual_execution']=True
  if crash_point=='marked':ledger.mark_executed('unmarked-step')
  raise RuntimeError('hard crash seam')
 monkeypatch.setattr(rt,'collect_rollout',crash)
 with pytest.raises(RuntimeError):rt.worker(**args)
 initial=Path(args['output_dir'])/'trial_initial.pt'
 monkeypatch.setattr(rt,'collect_rollout',original)
 recovered=rt.worker(**dict(args,resume=initial,output_dir=Path(args['output_dir']).parent/'recover'))
 assert recovered['retained_completed_updates']==49 and recovered['stop_reason']=='trial_budget_exhausted'
 assert recovered['budget']['uncertain']['learning:D1']==(0 if crash_point=='marked' else 1024)
 assert calls.get('virtual_execution',False)==(crash_point!='reserved')
 assert recovered['budget']['charged']['learning:D1']==49*49152+1024
 again=rt.worker(**dict(args,resume=recovered['checkpoint'],output_dir=Path(args['output_dir']).parent/'repeat'))
 assert again['retained_completed_updates']==49 and calls['updates']==49


def test_final_save_failure_is_not_a_success_receipt(worker_seam,monkeypatch):
 rt,args,calls=worker_seam;save=rt.save_worker
 def fail_after_entry(path,*a,**kw):
  if Path(path).name!='trial_initial.pt':raise OSError('final disk full')
  return save(path,*a,**kw)
 monkeypatch.setattr(rt,'save_worker',fail_after_entry)
 with pytest.raises(OSError,match='final disk full'):rt.worker(**args)
 assert calls['updates']==50 and (Path(args['output_dir'])/'trial_initial.pt').exists()
 assert not (Path(args['output_dir'])/'worker_result.json').exists()
