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
            return [dict(case_id=c['case_id'],complete=True,raw_complete=True,finite=True,success=False,
                reason='physical_failure',segment='walk',physical_failure=True,
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
