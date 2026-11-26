import torch
import pytest
from src.tasks.velocity.v10.checkpoint import BudgetLedger, checkpoint_save, checkpoint_load

def test_external_ledger_is_monotone_on_old_checkpoint(tmp_path):
 p=tmp_path/'ledger.jsonl';b=BudgetLedger(p);b.charge('D1',48,limit=100,kind='learning'); old=b.totals()
 b.charge('D1',48,limit=100,kind='learning'); assert BudgetLedger(p).totals()['learning:D1']==96
 assert old['learning:D1']==48
 with pytest.raises(RuntimeError):b.charge('D1',48,limit=100,kind='learning')

def test_checkpoint_identity_and_rng(tmp_path):
 torch.manual_seed(17);state=dict(model={'a':torch.ones(2)},num_envs=512,course='D1',rounds={'D1':1},physical_sha256='abc',budgets={'learning:D1':24576})
 path=tmp_path/'c.pt';checkpoint_save(path,state);expected=torch.rand(5);got=checkpoint_load(path,num_envs=512,physical_sha256='abc');assert torch.equal(expected,torch.rand(5))
 assert got['schema']=='a2-v10-state-v1'
 with pytest.raises(ValueError):checkpoint_load(path,num_envs=1024,physical_sha256='abc')
 with pytest.raises(ValueError):checkpoint_load(path,num_envs=512,physical_sha256='wrong')

def test_reservation_crash_reconcile_idempotent_and_boundary(tmp_path):
 b=BudgetLedger(tmp_path/'l');assert b.reserve('run:attempt:rollout:0','D1',8,limit=24,kind='learning')
 assert not b.reserve('run:attempt:rollout:0','D1',8,limit=24,kind='learning')
 with pytest.raises(RuntimeError):b.reserve('next','D1',8,limit=24,kind='learning')
 restored=BudgetLedger(tmp_path/'l');restored.reconcile_uncertain();restored.reconcile_uncertain()
 assert restored.report()['uncertain']['learning:D1']==8
 assert not restored.reserve('run:attempt:rollout:0','D1',8,limit=24,kind='learning')
 with pytest.raises(RuntimeError):restored.mark_executed('run:attempt:rollout:0')
 assert restored.reserve('next','D1',8,limit=24,kind='learning');assert restored.mark_executed('next');assert not restored.mark_executed('next')
 assert restored.report()['executed']['learning:D1']==8
 restored.reserve('last','D1',8,limit=24,kind='learning');restored.mark_executed('last')
 with pytest.raises(RuntimeError):restored.reserve('over','D1',8,limit=24,kind='learning')
 assert restored.totals()['learning:D1']==24

def test_resume_student_never_becomes_stair_teacher():
 from src.tasks.velocity.v10.model import V10ActorCritic
 from src.tasks.velocity.v10.runtime import frozen_teachers
 torch.set_num_threads(1)
 student=V10ActorCritic.from_stairs()
 with torch.no_grad():student.actor.mlp[-1].bias.add_(100.)
 teachers=frozen_teachers('cpu');source=V10ActorCritic.from_stairs()
 x=dict(proprio=torch.zeros(2,45),history=torch.zeros(2,5,45),heightmap=torch.zeros(2,3,40,30),route_error=torch.zeros(2,3))
 assert torch.equal(teachers.label(x,[True,True]),source(x))
 assert not torch.equal(teachers.label(x,[True,True]),student(x))

def test_physical_hash_excludes_actual_terrain_mass_but_tracks_robot_change():
 import mujoco
 from types import SimpleNamespace
 from src.tasks.velocity.v10.runtime import physical_contract
 def make(terrain_size,robot_mass=1.):
  xml=f'<mujoco><worldbody><body name="terrain"><geom type="box" size="{terrain_size} 1 .1"/></body><body name="robot"><joint name="joint" type="hinge"/><geom type="sphere" size=".1" mass="{robot_mass}"/></body></worldbody><actuator><motor joint="joint"/></actuator></mujoco>'
  m=mujoco.MjModel.from_xml_string(xml)
  indexing=SimpleNamespace(body_ids=[2],geom_ids=[1],ctrl_ids=[0])
  return SimpleNamespace(sim=SimpleNamespace(mj_model=m),robot=SimpleNamespace(indexing=indexing),joint_ids=[0],terrains=[terrain_size])
 a=physical_contract(make(1));b=physical_contract(make(8));c=physical_contract(make(8,2.))
 assert a['sha256']==b['sha256'];assert a['terrain_sha256']!=b['terrain_sha256'];assert b['sha256']!=c['sha256']

def test_gpu_pid_missing_cannot_pass_resource_admission(monkeypatch):
 from src.tasks.velocity.v10.resources import ResourceMonitor
 monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
 monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
 m=ResourceMonitor();m.rows=[dict(gpu_total_mib=100,gpu_task_mib=0,gpu_task_seen=False,host_available_bytes=8*1024**3)]
 assert not m.summary()['passed']

def test_ledger_incremental_read_catches_other_writer(tmp_path):
 a=BudgetLedger(tmp_path/'l');b=BudgetLedger(tmp_path/'l')
 a.charge('D1',1,limit=100,kind='learning');b.charge('D1',2,limit=100,kind='learning')
 assert a.totals()['learning:D1']==3
 a.charge('D1',3,limit=100,kind='learning');assert b.totals()['learning:D1']==6
 with (tmp_path/'l').open('w') as f:f.write('')
 with pytest.raises(RuntimeError):a.totals()

def test_completed_updates_exclude_discarded_partial_rollout(tmp_path):
 b=BudgetLedger(tmp_path/'l');b.charge('D1',48,limit=100,kind='learning')
 assert b.record_update('update1','D1',48);assert not b.record_update('update1','D1',48)
 b.reserve('partial','D1',4,limit=100,kind='learning');b.mark_executed('partial')
 b.reserve('crashed','D1',4,limit=100,kind='learning');b.reconcile_uncertain()
 report=BudgetLedger(tmp_path/'l').report()
 assert report['charged']['learning:D1']==56 and report['executed']['learning:D1']==52
 assert report['updated']['learning:D1']==48 and report['uncertain']['learning:D1']==4
 with pytest.raises(RuntimeError):b.record_update('illegal','D1',48)

def test_checkpoint_threshold_crossing_with_partial_rollout_offset():
 from src.tasks.velocity.v10.runtime import save_due
 from src.tasks.velocity.v10.spec import SAVE_SAMPLES
 assert save_due(SAVE_SAMPLES-100,SAVE_SAMPLES+100)
 assert not save_due(SAVE_SAMPLES+100,SAVE_SAMPLES+200)
 assert save_due(SAVE_SAMPLES+100,2*SAVE_SAMPLES+100)
