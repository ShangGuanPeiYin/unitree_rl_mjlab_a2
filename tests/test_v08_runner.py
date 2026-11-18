from test_v08_evaluation import failure_raw
from pathlib import Path
import sys
import copy
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

@__import__('functools').lru_cache(maxsize=1)
def logical_fixture():
    """Actual zero-step CPU env snapshot for synthetic learner lifecycle tests."""
    from src.tasks.velocity.v08.env import V08Env
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    env=V08Env(num_envs=8,device='cpu',seed=42,initial_cases=CurriculumManager(num_envs=8).schedule().cases)
    try:return copy.deepcopy(env.state_dict())
    finally:env.close()

def test_real_migrated_checkpoint_roundtrip_and_restore_once(tmp_path):
    from src.tasks.velocity.v08 import runner,recovery,model,ppo,continuation_v08 as migration
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    class Env:
        num_envs=8;device='cpu'
        def __init__(self): self.resets=0;self.schedule_calls=0;self._rng={};self._sampler={}
        def reset(self):self.resets+=1;return model.synthetic_observations(batch=8),{}
        def get_observations(self):return model.synthetic_observations(batch=8)
        def set_schedule(self,schedule,stage_iteration=0):self.schedule_calls+=1
        def state_dict(self):
            state=copy.deepcopy(logical_fixture());state['audit_fixture']=dict(place='mid',request=0.,ramp_index=51);return state
        def load_state_dict(self,state,restore_physics=False):
            assert restore_physics is False;self.saved=copy.deepcopy(state);self.reset()
        def rng_state(self):return self._rng
        def sampler_rng_state(self):return self._sampler
        def load_rng_state(self,x):self._rng=x
        def load_sampler_rng_state(self,x):self._sampler=x
    env=Env();net=model.V08ActorCritic();trainer=ppo.PPOTrainer(net,env)
    state,diff=migration.migrate_stop_6400(migration.load_designated_source(),source_sha256=migration.SOURCE_SHA256)
    r=runner.V08Runner(net,trainer,CurriculumManager(num_envs=8),env,tmp_path,mode='smoke')
    r.initialize(state)
    assert env.resets==1 and env.schedule_calls==1
    path,sha=r.save_verified('loaded')
    snapshot=recovery.load_verified(path,sha)
    assert migration.tree_digest(snapshot['environment'])==migration.tree_digest(env.state_dict())
    assert snapshot['learning']['control_samples']==157286400
    r.load(path,sha)
    assert env.resets==2
    assert r.ledger.updates==0 and r.update==6400
    assert r.ppo.obs is not None
    reordered=copy.deepcopy(snapshot)
    reordered['optimizer']['param_groups'][0]['params'].reverse()
    with pytest.raises(ValueError):recovery.validate_checkpoint(reordered,mode='smoke')
    nonfinite=copy.deepcopy(snapshot)
    next(iter(nonfinite['optimizer']['state'].values()))['exp_avg'].view(-1)[0]=float('nan')
    with pytest.raises(ValueError):recovery.validate_checkpoint(nonfinite,mode='smoke')
    bad=copy.deepcopy(snapshot);bad['ledger']['pending_update']={'update':6401}
    badpath=tmp_path/'partial.pt';badsha=recovery.atomic_save(badpath,bad)
    with pytest.raises(ValueError):r.load(badpath,badsha)

def test_control_invalid_signed_requests_and_latched_stop(tmp_path):
    from src.tasks.velocity.v08.operator_control import ControlProtocol,write_control
    p=tmp_path/'control.json';control=ControlProtocol(p,tmp_path/'evidence')
    with pytest.raises(ValueError):write_control(p,{'action':'request_speed','speed':.7})
    write_control(p,{'action':'request_speed','speed':-.5,'env_ids':[0]})
    assert control.poll()['speed']==-.5
    write_control(p,{'action':'stop'})
    assert control.poll()['action']=='stop'
    write_control(p,{'action':'continue'})
    assert control.poll()['action']=='stop'

def test_real_cpu_env_one_reset_after_rng_restore_and_one_native_reset(tmp_path):
    """CPU-compatible RNG fixture checks reset count; source CUDA RNG is NOT claimed."""
    from src.tasks.velocity.v08 import runner,recovery,model,ppo,continuation_v08 as migration
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    from src.tasks.velocity.v08.env import V08Env
    class ObservedEnv(V08Env):
        def __init__(self,*args,**kwargs):self.reset_batches=[];super().__init__(*args,**kwargs)
        def _reset_ids(self,ids):
            self.reset_batches.append(torch.as_tensor(ids).detach().cpu().tolist())
            return super()._reset_ids(ids)
    manager=CurriculumManager(num_envs=8,seed=42)
    env=ObservedEnv(num_envs=8,device='cpu',seed=42,phase='A',initial_cases=manager.schedule().cases)
    try:
        state,diff=migration.migrate_stop_6400(migration.load_designated_source(),source_sha256=migration.SOURCE_SHA256)
        # The source uses CUDA generator bytes; replace only this explicit test
        # fixture RNG with CPU-compatible bytes. Learning state remains real6400.
        state['rng']=recovery.capture_rng(env)
        net=model.V08ActorCritic();trainer=ppo.PPOTrainer(net,env)
        r=runner.V08Runner(net,trainer,manager,env,tmp_path,mode='smoke')
        before=len(env.reset_batches);r.initialize(state)
        assert env.reset_batches[before:]==[list(range(8))]
        path,sha=r.save_verified('cpu_rng_fixture')
        broken=r.checkpoint_state();del broken['environment']['command']['k']
        with pytest.raises(ValueError):recovery.validate_checkpoint(broken,mode='smoke')
        before=len(env.reset_batches);r.load(path,sha)
        assert env.reset_batches[before:]==[list(range(8))]
    finally:env.close()

def test_control_stop_inside_collect_prevents_any_adam_update(tmp_path):
    """Synthetic observations; proves lifecycle stop, not robot policy behavior."""
    from src.tasks.velocity.v08 import runner,model,ppo,continuation_v08 as migration
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    from src.tasks.velocity.v08.operator_control import write_control
    class Env:
        num_envs=8;device='cpu'
        def __init__(self):self.steps=0;self._rng={};self._sampler={}
        def reset(self):return model.synthetic_observations(batch=8),{}
        def get_observations(self):return model.synthetic_observations(batch=8)
        def set_schedule(self,*args,**kwargs):pass
        def state_dict(self):return copy.deepcopy(logical_fixture())
        def rng_state(self):return self._rng
        def sampler_rng_state(self):return self._sampler
        def load_rng_state(self,x):self._rng=x
        def load_sampler_rng_state(self,x):self._sampler=x
        def step(self,actions):
            self.steps+=1
            write_control(tmp_path/'control.json',{'action':'stop'})
            return self.get_observations(),torch.zeros(8),torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool),{}
    env=Env();net=model.V08ActorCritic();trainer=ppo.PPOTrainer(net,env)
    r=runner.V08Runner(net,trainer,CurriculumManager(num_envs=8),env,tmp_path,mode='smoke')
    state,_=migration.migrate_stop_6400(migration.load_designated_source(),source_sha256=migration.SOURCE_SHA256)
    r.initialize(state);r.step()
    assert env.steps==1
    assert r.update==6400 and r.ppo.learning_state['optimizer_steps']==128000
    assert r.ledger.state['stop_state']=='user_requested_stop'
    assert r.ledger.state['discarded_rollout_control_samples']==8

def test_durable_evaluation_sidecar_is_reused_without_running_cases_again(tmp_path):
    """Failure records are synthetic fixtures, never claimed robot evidence."""
    from src.tasks.velocity.v08 import runner,model,ppo,recovery,continuation_v08 as migration
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    class Env:
        num_envs=8;device='cpu'
        def reset(self):return model.synthetic_observations(batch=8),{}
        def set_schedule(self,*args,**kwargs):pass
        def rng_state(self):return {}
        def sampler_rng_state(self):return {}
        def load_rng_state(self,x):pass
        def load_sampler_rng_state(self,x):pass
    env=Env();net=model.V08ActorCritic();trainer=ppo.PPOTrainer(net,env)
    r=runner.V08Runner(net,trainer,CurriculumManager(num_envs=8),env,tmp_path,mode='smoke')
    state,_=migration.migrate_stop_6400(migration.load_designated_source(),source_sha256=migration.SOURCE_SHA256)
    r.initialize(state)
    class Cases:
        supports_batch=True;control_samples=0
        def __call__(self,batch):
            cost=3*len(batch);self.control_samples+=cost;key=recovery.digest([c['case_id'] for c in batch])
            return [dict(failure_raw(c),reason='physical_failure',
                collector_batch_id=key,collector_batch_control_samples=cost) for c in batch]
    r.case_runner=Cases();first=r._evaluate('h00')
    assert r.ledger.state['evaluation_control_samples']==45
    assert 'payload' not in r.evaluation['h00:6400']
    r.evaluation={};r.ledger.state['evaluation_control_samples']=0
    class MustNotRun:
        def __call__(self,*args):raise AssertionError('already paid cases must not run again')
    r.case_runner=MustNotRun();again=r._evaluate('h00')
    assert again.summary==first.summary
    assert r.ledger.state['evaluation_control_samples']==45

def test_observed_stop_stays_latched_when_evidence_write_fails(tmp_path,monkeypatch):
    from src.tasks.velocity.v08.operator_control import ControlProtocol,write_control
    control=ControlProtocol(tmp_path/'control.json',tmp_path/'evidence')
    write_control(control.path,{'action':'stop'})
    original=control._snapshot
    def failed(raw):raise OSError('injected evidence failure')
    monkeypatch.setattr(control,'_snapshot',failed)
    with pytest.raises(OSError):control.poll()
    write_control(control.path,{'action':'continue'})
    monkeypatch.setattr(control,'_snapshot',original)
    assert control.poll()['action']=='stop'

def boundary_fixture(tmp_path):
    """Native serialization fixture at6600; no real PPO or robot actions."""
    from src.tasks.velocity.v08 import runner,model,ppo,continuation_v08 as migration
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    class Env:
        num_envs=8;device='cpu'
        def __init__(self):self.phase='A';self.resets=0;self.schedule=None;self._rng={};self._sampler={}
        def reset(self):self.resets+=1;return self.get_observations(),{}
        def get_observations(self):return model.synthetic_observations(batch=8)
        def set_schedule(self,schedule,stage_iteration=0):
            changed=self.phase!=schedule.phase;self.phase=schedule.phase;self.schedule=schedule;self.stage_iteration=stage_iteration
            if changed:self.reset()
        def state_dict(self):
            value=copy.deepcopy(logical_fixture());value.update(phase=self.phase,case_templates=list(self.schedule.cases),
                generation=self.schedule.generation,roles=list(self.schedule.roles),stage_iteration=self.stage_iteration)
            return value
        def load_state_dict(self,value,restore_physics=False):
            assert restore_physics is False
            self.phase=value['phase'];self.reset()
        def rng_state(self):return self._rng
        def sampler_rng_state(self):return self._sampler
        def load_rng_state(self,x):self._rng=x
        def load_sampler_rng_state(self,x):self._sampler=x
    env=Env();net=model.V08ActorCritic();trainer=ppo.PPOTrainer(net,env);manager=CurriculumManager(num_envs=8)
    r=runner.V08Runner(net,trainer,manager,env,tmp_path,mode='smoke')
    state,_=migration.migrate_stop_6400(migration.load_designated_source(),source_sha256=migration.SOURCE_SHA256)
    r.initialize(state)
    r.ledger.state.update(engineering_updates=200,control_samples=200*8*48,physics_substeps=200*8*48*4)
    trainer.learning_state.update(completed_ppo_updates=6600,optimizer_steps=132000)
    for moment in trainer.optimizer.state.values():moment['step'].fill_(132000)
    manager.state.spent_by_phase['A']=200;manager.state.last_update=6600;manager.state.generation_updates=200
    manager.state.gate_streak=1;manager.state.last_gate_update=6500
    return r

@pytest.mark.parametrize('pending_gate',[False,True])
def test_native_phase_handoff_finishes_before_one_fresh_reset_and_any_ppo(tmp_path,pending_gate):
    from src.tasks.velocity.v08 import recovery
    r=boundary_fixture(tmp_path)
    r.ledger.state['pending_boundary']=dict(update=6600,phase='A',events=['gate'],index=0 if pending_gate else 1)
    if not pending_gate:r.curriculum.record_gate(dict(gate_pass=True,phase='A'),update=6600)
    original=r.env.state_dict();path,sha=r.save_verified('handoff_fixture')
    original_rng=recovery.load_verified(path,sha)['rng']
    # This is a synthetic passing gate result, not a physics acceptance claim.
    def finish_gate(kind,phase=None):
        assert r._ready is False
        with pytest.raises(ValueError,match='reset'):r.ppo.collect()
        r.curriculum.record_gate(dict(gate_pass=True,phase=phase),update=6600)
    r._evaluate=finish_gate
    before=r.env.resets;r.load(path,sha)
    assert r.curriculum.phase=='B' and r.env.phase=='B'
    assert r.env.resets-before==1
    assert r.saved_task_provenance['phase']=='A'
    # Any PPO entry observes current task/phase, even after a persisted handoff.
    observed=[]
    class BeforePPO(RuntimeError):pass
    def observe():observed.append((r.curriculum.phase,r.env.phase));raise BeforePPO('no synthetic update executed')
    r.ppo.train_step=observe
    with pytest.raises(BeforePPO):r.step()
    assert observed==[('B','B')]
    assert r.curriculum.state.spent_by_phase['B']==0
    if pending_gate:
        snapshots=[recovery.load_verified(p,__import__('json').loads(p.with_suffix('.json').read_text())['sha256'])
            for p in (tmp_path/'checkpoints').glob('boundary_event*.pt')]
        assert snapshots and snapshots[0]['recovery_pending_fresh_reset'] is True
        assert snapshots[0]['environment']['phase']=='A'
        assert __import__('src.tasks.velocity.v08.continuation_v08',fromlist=['tree_digest']).tree_digest(snapshots[0]['environment'])==__import__('src.tasks.velocity.v08.continuation_v08',fromlist=['tree_digest']).tree_digest(original)
        assert __import__('src.tasks.velocity.v08.continuation_v08',fromlist=['tree_digest']).tree_digest(snapshots[0]['rng'])==__import__('src.tasks.velocity.v08.continuation_v08',fromlist=['tree_digest']).tree_digest(original_rng)

def test_partial_evaluation_has_durable_progress_and_cannot_restart_or_reset(tmp_path):
    from src.tasks.velocity.v08 import recovery
    r=boundary_fixture(tmp_path);path,sha=r.save_verified('before_eval')
    class Cases:
        supports_batch=True;control_samples=0;calls=0
        def __call__(self,batch):
            self.calls+=1
            assert list((tmp_path/'evaluation_transactions').glob('*.intent.json'))
            self.control_samples+=3*len(batch)
            raise KeyboardInterrupt('simulated process loss after actual batch samples')
    base=Cases();r.case_runner=base
    with pytest.raises(KeyboardInterrupt):r._evaluate('h00')
    assert base.calls==1
    progress=list((tmp_path/'evaluation_transactions').glob('*.batches/*.commit.json'))
    assert progress and __import__('json').loads(progress[0].read_text())['control_samples']==24
    with pytest.raises(ValueError,match='evaluation'):r._evaluate('h00')
    before=r.env.resets
    with pytest.raises(ValueError,match='evaluation'):r.load(path,sha)
    assert base.calls==1 and r.env.resets==before


def test_completed_evaluation_journal_reuses_sidecar_and_rejects_cost_tamper(tmp_path):
    from src.tasks.velocity.v08 import recovery
    r=boundary_fixture(tmp_path)
    class Cases:
        supports_batch=True;control_samples=0;calls=0
        def __call__(self,batch):
            self.calls+=1;cost=3*len(batch);self.control_samples+=cost
            key=recovery.digest([c['case_id'] for c in batch])
            return [dict(failure_raw(c),reason='physical_failure',
                collector_batch_id=key,collector_batch_control_samples=cost) for c in batch]
    base=Cases();r.case_runner=base;r._evaluate('h00');assert base.calls==2
    transactions=recovery.verify_evaluation_journal(tmp_path)
    assert len(transactions)==1 and next(iter(transactions.values()))['control_samples']==45
    wrong_ledger=r.checkpoint_state();wrong_ledger['ledger']['evaluation_control_samples']=0
    with pytest.raises(ValueError,match='cost'):recovery.validate_checkpoint(wrong_ledger,mode='smoke')
    # Simulate loss after the complete immutable sidecar and batch progress but
    # before the transaction completion receipt. It must be completed by reuse.
    commit=next((tmp_path/'evaluation_transactions').glob('*.commit.json'));commit.unlink()
    r.evaluation={};r.ledger.state['evaluation_control_samples']=0;r._evaluate('h00')
    assert base.calls==2 and r.ledger.state['evaluation_control_samples']==45 and commit.exists()
    data=__import__('json').loads(commit.read_text());data['reference']['control_samples']+=1
    recovery.write_json(commit,data)
    with pytest.raises(ValueError):recovery.verify_evaluation_journal(tmp_path)

def test_recovery_interruption_after_gate_checkpoint_resumes_without_double_reset(tmp_path):
    from src.tasks.velocity.v08 import recovery
    r=boundary_fixture(tmp_path)
    r.ledger.state['pending_boundary']=dict(update=6600,phase='A',events=['gate'],index=0)
    path,sha=r.save_verified('pending_gate_fault_fixture')
    calls=[]
    def gate(kind,phase=None):calls.append(kind);r.curriculum.record_gate(dict(gate_pass=True,phase=phase),update=6600)
    r._evaluate=gate
    original_save=r.save_verified;durable=[]
    class LostAfterSave(RuntimeError):pass
    def fail_after_save(reason):
        result=original_save(reason)
        if reason=='boundary_event':durable.append(result);raise LostAfterSave('simulated loss before fresh reset')
        return result
    r.save_verified=fail_after_save;before=r.env.resets
    with pytest.raises(LostAfterSave):r.load(path,sha)
    assert r.env.resets==before and not r._ready
    state=recovery.load_verified(*durable[0]);assert state['recovery_pending_fresh_reset'] is True
    assert state['curriculum']['state']['phase']=='B' and state['environment']['phase']=='A'
    r.save_verified=original_save;r.load(*durable[0])
    assert r.env.resets-before==1 and r.env.phase==r.curriculum.phase=='B'
    assert calls==['gate'] and r.curriculum.state.spent_by_phase['B']==0

def test_final_review_real_overflow_producer_collector_normalize_runner_blocks_without_adam(tmp_path,monkeypatch):
    from src.tasks.velocity.v08 import runner,evaluation,continuation_v08 as migration
    from src.tasks.velocity.v08.env import V08Env
    r=boundary_fixture(tmp_path)
    before=migration.tree_digest(r.ppo.optimizer.state_dict());learning=copy.deepcopy(r.ppo.learning_state)
    def factory(batch):
        env=V08Env(num_envs=len(batch),device='cpu',seed=42,evaluation=True,initial_cases=batch)
        checks=iter([True,False,False,False,False]);monkeypatch.setattr(env.contacts,'check_overflow',lambda:next(checks))
        return env
    r.case_runner=runner.environment_case_runner(r.model,factory,device='cpu',max_steps=5)
    from src.tasks.velocity.v08 import recovery
    with pytest.raises(recovery.RecoveryBlocked,match='engineering errors'):r._evaluate('h00')
    assert r.case_runner.control_samples==15 and r.ledger.state['evaluation_control_samples']==15
    assert r.curriculum.state.gate_streak==1
    assert migration.tree_digest(r.ppo.optimizer.state_dict())==before and r.ppo.learning_state==learning
    import json
    from src.tasks.velocity.v08.evaluation_records import restore_result
    payload=json.loads((tmp_path/'evaluations/h00_A_006600.json').read_text());result=restore_result(payload,payload['identity'])
    before_cost=r.case_runner.control_samples
    with pytest.raises(recovery.RecoveryBlocked):r._evaluate('h00')
    assert r.case_runner.control_samples==before_cost
    assert result.summary['engineering_error'] and result.summary['error']==15 and result.summary['valid']==0
    assert not result.summary['gate_pass']
    assert all(row['raw']['diagnostics']['reason']=='contact_overflow' for row in result.records)

def test_final_review_real_training_overflow_closes_before_adam_and_keeps_only_known_physics(tmp_path,monkeypatch):
    from src.tasks.velocity.v08 import runner,model,ppo,continuation_v08 as migration
    from src.tasks.velocity.v08.curriculum import CurriculumManager
    from src.tasks.velocity.v08.env import V08Env,ContactOverflow
    env=V08Env(num_envs=8,device='cpu',seed=42)
    try:
        net=model.V08ActorCritic();trainer=ppo.PPOTrainer(net,env)
        r=runner.V08Runner(net,trainer,CurriculumManager(num_envs=8),env,tmp_path,mode='smoke')
        state,_=migration.migrate_stop_6400(migration.load_designated_source(),source_sha256=migration.SOURCE_SHA256)
        # Explicit CPU-compatible RNG fixture only; source CUDA RNG restoration
        # remains unverified here. The inherited real learning/Adam state is intact.
        from src.tasks.velocity.v08 import recovery
        state['rng']=recovery.capture_rng(env);r.initialize(state)
        before=migration.tree_digest(trainer.optimizer.state_dict());learning=copy.deepcopy(trainer.learning_state)
        checks=iter([True,False,False,False,False]);monkeypatch.setattr(env.contacts,'check_overflow',lambda:next(checks))
        with pytest.raises(ContactOverflow):r.step()
        assert migration.tree_digest(trainer.optimizer.state_dict())==before and trainer.learning_state==learning
        assert r.update==6400 and r.ledger.state['engineering_updates']==0 and r.ledger.state['control_samples']==0
        assert r.ledger.state['discarded_rollout_control_samples']==8
        assert r.ledger.state['engineering_failure_diagnostics']['known_physics_substeps']==32
        assert r.ledger.state['stop_state']=='engineering_error' and not r.step()
    finally:env.close()
