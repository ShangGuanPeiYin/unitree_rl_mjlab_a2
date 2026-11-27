"""CPU-only explicit handoff: real state/Adam/checkpoints, virtual sample ledger."""
import copy
import json
from pathlib import Path
import pytest
import torch
from src.tasks.velocity.v10 import spec,cases,checkpoint as cp
from src.tasks.velocity.v10.replay import TaskReservoir,TASKS
from src.tasks.velocity.v10.teachers import teacher_identity

OLD='a'*64
NEW='b'*64
PHYS='c'*64

@pytest.fixture
def setup(tmp_path,monkeypatch):
 from src.tasks.velocity.v10 import handoff as h,trial_guard as tg
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 monkeypatch.setattr(cp,'code_fingerprint',lambda:OLD)
 monkeypatch.setattr(h,'code_fingerprint',lambda:NEW)
 monkeypatch.setattr(tg,'CANONICAL_GUARD',tmp_path/'guard.json')
 def ref(p):return dict(path=str(p.resolve()),sha256=cp.sha256(p))
 def write(name,row):p=tmp_path/name;p.write_text(json.dumps(row));return ref(p)
 ledger=tmp_path/'learning.jsonl';account=cp.BudgetLedger(ledger);account.charge('D1',2457600,limit=24576000,kind='learning');account.record_update('trial','D1',2457600)
 ev=tmp_path/'evaluation.jsonl';ev.touch()
 guard=dict(schema='a2-v10-adjusted-trial-v1',trial_id='cpu',guard_path=str(tg.CANONICAL_GUARD),ledger_path=str(ledger),course='D1',num_envs=1024,stop_round=50,charged_cap=2457600,sds_sha256=spec.SDS_SHA256)
 g=write('guard.json',guard)
 model=torch.nn.Linear(45,12);optimizer=torch.optim.Adam(model.parameters(),lr=3e-5)
 loss=model(torch.ones(6,45)).square().mean();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
 b=dict(obs={'proprio':torch.ones(6,45)},actions=torch.zeros(6,12),task=list(TASKS),teacher=['blind']*6,case=['c']*6,episode=list(range(6)),control_step=[1]*6)
 replay=TaskReservoir(capacity=1);replay.add(b)
 trial=dict(identity=guard,retained_completed_updates=50,pre_gate_adjusted_trial=True,teacher_admitted=False,accepted=None,training_accepted=False,gate_reached=False)
 env_state=dict(rng=__import__('random').Random(17).getstate(),role_cursors={t:7 for t in TASKS},episode_ids=list(range(1024)),control_steps=2400)
 origin=tmp_path/'origin.pt';cp.checkpoint_save(origin,dict(model=model.state_dict(),learner=dict(optimizer=optimizer.state_dict()),reservoir=replay.state_dict(),env=env_state,budgets=account.totals(),accounting=account.report(),rounds={'D1':50},course='D1',num_envs=1024,physical_sha256=PHYS,keep=None,extra={'trial':trial}))
 originref=ref(origin)
 result=write('trial_result.json',dict(checkpoint=str(origin),checkpoint_sha256=originref['sha256'],course='D1',round=50,retained_completed_updates=50,updated_samples=2457600,historical_updated_samples=2457600,consumed=2457600,budget=account.report(),stop_reason='trial_limit',physical_sha256=PHYS,**{k:v for k,v in trial.items() if k not in ('identity','retained_completed_updates')}))
 report=write('resource_report.json',dict(passed=True,gpu_total_peak_mib=7000,gpu_task_peak_mib=6500,host_available_min_bytes=5*1024**3,gpu_task_valid_samples=10))
 from src.tasks.velocity.v10.diagnostic_a1 import diagnostic_manifest
 rulings=cp.sha256(h.RULINGS)
 def receipt(manifest,course,suite,replicas,code):
  groups={}
  for c in manifest:
   row=dict(case_sha256=c['case_sha256'],group_id=c['group_id'],passed=True,expected_windows=['walk_0'],windows={'walk_0':dict(rmse=0.,pitch_rms=0.,pitch_p99=0.)})
   groups.setdefault(c['group_id'],[]).append(row)
  return dict(course=course,suite=suite,replicas=replicas,reference='teachers',code_sha256=code,sds_sha256=spec.SDS_SHA256,rulings_sha256=rulings,physical_sha256=PHYS,model_sha256=cases.digest(teacher_identity()),suite_sha256=cases.digest(manifest),complete=True,aborted_for_teacher_failure=False,groups=groups,results=[x for rows in groups.values() for x in rows],tested_case_count=len(manifest),planned_case_count=len(manifest),diagnostic_only=False,resource_admitted=True,resource_report_path=report['path'],resource_report_sha256=report['sha256'])
 diagnostic=receipt(diagnostic_manifest(),'D1','A',4,OLD);diagnostic.update(diagnostic_only=True,diagnostic_protocol='A1-diagnostic-112-v1',passed=True)
 diag=write('diagnostic.json',diagnostic);resource=write('resource.json',dict(passed=True,num_envs=1024))
 compatibility=write('compat.json',{'fixture':True})
 compat=dict(path=compatibility['path'],sha256=compatibility['sha256'],document=dict(physical_sha256=PHYS),from_code_sha256=OLD,to_code_sha256=NEW)
 monkeypatch.setattr(h,'validate_compatibility',lambda p:compat)
 def artifact(path,compatibility,purpose,operation_id=None):
  if purpose=='trial_checkpoint':return ref(Path(path))
  return json.loads(Path(path).read_text())
 monkeypatch.setattr(h,'validate_compatible_artifact',artifact);monkeypatch.setattr(h,'validate_compatible_resource',lambda p,c:json.loads(Path(p).read_text()))
 plan=write('plan.json',dict(schema='a2-v10-formal-handoff-plan-v1',source_compatibility=compatibility,origin_checkpoint=originref,trial_guard=g,trial_result=result,learning_ledger_path=str(ledger),evaluation_ledger_path=str(ev),learning_ledger_prefix=h._prefix(ledger),evaluation_ledger_prefix=h._prefix(ev),num_envs=1024,course='D1',initial_round=50,resource_selection=resource,diagnostic=diag,handoff_path=str(tmp_path/'handoff.json'),sds_sha256=spec.SDS_SHA256,rulings_sha256=rulings,physical_sha256=PHYS))
 teachers={f'teacher_admission_{course}_{suite}':write(f'{course}_{suite}.json',receipt(cases.suite(course,suite,20),course,suite,20,NEW)) for course in ('D1','D2','D3') for suite in ('A','B')}
 monkeypatch.setattr(cp,'code_fingerprint',lambda:NEW)
 return dict(h=h,plan=plan,teachers=teachers,origin=originref,ledger=ledger,ref=ref,write=write,batch=b,optimizer=optimizer,model=model,replay=replay,env_state=env_state,origin_rng=cp.state_fingerprint(torch.load(origin,map_location='cpu',weights_only=False)['rng']))


def test_plan_is_not_handoff_and_missing_teacher_gate_cannot_finalize(setup):
 s=setup;h=s['h'];plan=h.validate_plan(s['plan']['path'])
 assert plan['initial_round']==50 and plan['learning_ledger_path']==str(s['ledger'])
 with pytest.raises(ValueError):h.validate_handoff(s['plan']['path'])
 with pytest.raises(ValueError):h.finalize_handoff(s['plan']['path'],dict(list(s['teachers'].items())[:-1]))
 assert not (Path(s['plan']['path']).parent/'handoff.json').exists()


def test_finalization_is_complete_immutable_and_idempotent(setup):
 s=setup;h=s['h'];done=h.finalize_handoff(s['plan']['path'],s['teachers'])
 assert len(done['teacher_admission'])==6 and all(r['passed'] for r in done['teacher_admission'].values())
 assert h.validate_handoff(done['path'])['sha256']==done['sha256']
 assert h.finalize_handoff(s['plan']['path'],s['teachers'])['sha256']==done['sha256']
 before=cp.sha256(s['origin']['path']);guard=Path(s['plan']['path']).parent/'guard.json';guard_before=cp.sha256(guard)
 Path(done['path']).write_text('{}')
 with pytest.raises(ValueError):h.finalize_handoff(s['plan']['path'],s['teachers'])
 assert cp.sha256(s['origin']['path'])==before and cp.sha256(guard)==guard_before

@pytest.mark.parametrize('mutation',['failed_group','missing_case','wrong_case','wrong_source','no_resource'])
def test_teacher_receipts_require_full_native_identity_and_grade(setup,mutation):
 s=setup;key='teacher_admission_D1_A';ref=s['teachers'][key];row=json.loads(Path(ref['path']).read_text());group=next(iter(row['groups']))
 if mutation=='failed_group':
  for r in row['groups'][group][:3]:r['passed']=False
  row['results']=[r for rows in row['groups'].values() for r in rows]
 elif mutation=='missing_case':row['groups'][group].pop()
 elif mutation=='wrong_case':row['groups'][group][0]['case_sha256']='wrong'
 elif mutation=='wrong_source':row['code_sha256']=OLD
 else:row['resource_admitted']=False
 Path(ref['path']).write_text(json.dumps(row));s['teachers'][key]=s['ref'](Path(ref['path']))
 with pytest.raises(ValueError):s['h'].finalize_handoff(s['plan']['path'],s['teachers'])


def test_old_origin_checkpoint_only_loads_with_exact_explicit_handoff(setup):
 s=setup;h=s['h'];done=h.finalize_handoff(s['plan']['path'],s['teachers'])
 with pytest.raises(ValueError):cp.checkpoint_load(s['origin']['path'],num_envs=1024,physical_sha256=PHYS,restore_random=False)
 state=cp.checkpoint_load(s['origin']['path'],num_envs=1024,physical_sha256=PHYS,restore_random=False,formal_handoff=done)
 assert cp.state_fingerprint(state['model'])==cp.state_fingerprint(s['model'].state_dict())
 extra=h.bind_handoff(state,done,checkpoint_path=s['origin']['path'],course='D1')
 assert extra['trial']==state['extra']['trial'] and extra['formal_handoff']['credential_sha256']==done['sha256']
 with pytest.raises(ValueError):h.bind_handoff(state,done,checkpoint_path=s['origin']['path'],course='D2')
 with pytest.raises(ValueError):cp.checkpoint_load(s['origin']['path'],num_envs=512,physical_sha256=PHYS,restore_random=False,formal_handoff=done)


def test_ledger_prefix_accepts_append_but_rejects_rewrite(setup):
 s=setup;h=s['h'];p=h.validate_plan(s['plan']['path'])
 account=cp.BudgetLedger(s['ledger']);account.charge('D1',49152,limit=24576000,kind='learning')
 assert h.validate_plan(s['plan']['path'])['sha256']==p['sha256']
 data=s['ledger'].read_bytes();s['ledger'].write_bytes(data.replace(b'2457600',b'2457601',1))
 with pytest.raises(ValueError,match='history changed'):h.validate_plan(s['plan']['path'])

@pytest.fixture
def runtime_seam(setup,monkeypatch):
 s=setup;h=s['h'];done=h.finalize_handoff(s['plan']['path'],s['teachers'])
 from src.tasks.velocity.v10 import runtime as rt
 import src.tasks.velocity.v10.env as em
 class Env:
  def __init__(self,num_envs,course,**kw):self.num_envs=num_envs;self.course=course
  def state_dict(self):return s['env_state']
  def load_state_dict(self,state):
   assert state==s['env_state'];calls['env_restored']=True
  def close(self):pass
 class Model(torch.nn.Linear):
  @classmethod
  def from_stairs(cls,device):return cls(45,12).to(device)
  def set_phase(self,course):pass
 expected=dict(model=cp.state_fingerprint(s['model'].state_dict()),optimizer=cp.state_fingerprint(s['optimizer'].state_dict()),replay=s['replay'].integrity_manifest(),rng=s['origin_rng'])
 calls={'updates':0,'collect':0,'interrupt_at':None}
 class Learner:
  def __init__(self,m):self.model=m;self.optimizer=torch.optim.Adam(m.parameters(),lr=3e-5)
  def update(self,b,r,n):
   if calls['updates']==0:
    assert cp.state_fingerprint(self.optimizer.state_dict())==expected['optimizer']
    assert r.integrity_manifest()==expected['replay']
   calls['updates']+=1;self.optimizer.zero_grad();loss=self.model(b['obs']['proprio']).square().mean();loss.backward();self.optimizer.step();r.add(b);return {'loss':loss.item()}
 def collect(env,model,teachers,phase,ledger,budget_limit):
  if calls['updates']==0:
   assert cp.state_fingerprint(model.state_dict())==expected['model']
   assert cp.state_fingerprint(cp.rng_state())==expected['rng']
   assert calls['env_restored']
  calls['collect']+=1
  if calls['collect']==calls['interrupt_at']:raise RuntimeError('interruption after saved checkpoint')
  ledger.charge('D1',49152,limit=budget_limit,kind='learning')
  return s['batch'],dict(rollout_id=f'formal-{calls["collect"]}',episodes=[],per_task={})
 monkeypatch.setattr(em,'V10Env',Env);monkeypatch.setattr(rt,'V10ActorCritic',Model);monkeypatch.setattr(rt,'DAggerLearner',Learner)
 monkeypatch.setattr(rt,'TaskReservoir',lambda:TaskReservoir(capacity=1));monkeypatch.setattr(rt,'frozen_teachers',lambda d:None)
 monkeypatch.setattr(rt,'physical_contract',lambda e:{'sha256':PHYS,'contract':{}});monkeypatch.setattr(rt,'release_host_memory',lambda:None)
 monkeypatch.setattr(rt,'collect_rollout',collect)
 from src.tasks.velocity.v10 import identity
 monkeypatch.setattr(identity,'code_fingerprint',lambda:NEW)
 args=dict(course='D1',num_envs=1024,stop_round=51,output_dir=Path(s['plan']['path']).parent/'formal',resume=s['origin']['path'],ledger_path=s['ledger'],device='cpu',formal_handoff=done['path'],controller_state={'worker_operation_id':'learning_D1_51'})
 return s,rt,args,calls,expected


def test_real_worker_restores_trial_state_once_and_keeps_provenance(runtime_seam):
 s,rt,args,calls,expected=runtime_seam
 result=rt.worker(**args)
 assert calls['updates']==1 and result['consumed']==51*49152
 state=cp.checkpoint_load(result['checkpoint'],num_envs=1024,physical_sha256=PHYS,restore_random=False)
 assert state['extra']['trial']['retained_completed_updates']==50
 assert state['extra']['formal_handoff']['origin_checkpoint']==s['origin']
 assert cp.sha256(s['origin']['path'])==s['origin']['sha256']
 manifest=json.loads((Path(args['output_dir'])/'worker_checkpoint_manifest.json').read_text())
 assert manifest['operation_id']=='learning_D1_51' and manifest['round']==51
 assert manifest['checkpoint']['sha256']==result['checkpoint_sha256']
 # The same credential authorizes its strictly current-source descendants.
 r=TaskReservoir(capacity=1);r.take_state_dict(state.pop('reservoir'))
 expected.update(model=cp.state_fingerprint(state['model']),optimizer=cp.state_fingerprint(state['learner']['optimizer']),replay=r.integrity_manifest(),rng=cp.state_fingerprint(state['rng']))
 calls['updates']=0
 again=rt.worker(**dict(args,stop_round=52,resume=result['checkpoint'],controller_state={'worker_operation_id':'learning_D1_52'}))
 assert calls['updates']==1 and again['consumed']==52*49152
 plain={k:v for k,v in args.items() if k!='formal_handoff'}
 with pytest.raises(ValueError):rt.worker(**dict(plain,resume=again['checkpoint'],stop_round=52))


def test_periodic_checkpoint_manifest_survives_later_interruption(runtime_seam):
 s,rt,args,calls,expected=runtime_seam;calls['interrupt_at']=51
 args.update(stop_round=101,controller_state={'worker_operation_id':'learning_D1_101'})
 with pytest.raises(RuntimeError,match='interruption'):rt.worker(**args)
 manifest=json.loads((Path(args['output_dir'])/'worker_checkpoint_manifest.json').read_text())
 assert manifest['round']==100 and manifest['target_round']==101 and manifest['confirmed_samples']==100*49152
 assert manifest['code_sha256']==NEW and manifest['formal_handoff_sha256']==cp.sha256(args['formal_handoff'])
 state=cp.checkpoint_load(manifest['checkpoint']['path'],num_envs=1024,physical_sha256=PHYS,restore_random=False)
 assert cp.sha256(manifest['checkpoint']['path'])==manifest['checkpoint']['sha256']
 expected.update(model=cp.state_fingerprint(state['model']),optimizer=cp.state_fingerprint(state['learner']['optimizer']),rng=cp.state_fingerprint(state['rng']))
 # Reservoir payload remains the same six synthetic rows, sampler history is restored.
 r=TaskReservoir(capacity=1);r.take_state_dict(state['reservoir']);expected['replay']=r.integrity_manifest()
 calls.update(updates=0,interrupt_at=None)
 result=rt.worker(**dict(args,resume=manifest['checkpoint']['path']))
 assert calls['updates']==1 and result['consumed']==101*49152


def test_checkpoint_loader_rejects_bare_authorization_dict(setup):
 s=setup
 forged=dict(origin_checkpoint=s['origin'],plan_document={'physical_sha256':PHYS},compatibility={'from_code_sha256':OLD})
 with pytest.raises(ValueError):cp.checkpoint_load(s['origin']['path'],num_envs=1024,physical_sha256=PHYS,restore_random=False,formal_handoff=forged)
