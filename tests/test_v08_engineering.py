from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

def test_case_collector_keeps_first_episodes_and_counts_async_extra_steps():
    from src.tasks.velocity.v08.runner import environment_case_runner
    class Policy:
        def act(self,obs,deterministic=False):assert deterministic;return torch.zeros(2,12)
    class Env:
        def __init__(self):self.steps=0;self.closed=False
        def get_observations(self):return {}
        def step(self,action):
            self.steps+=1
            rows=[dict(env_id=0,case_id='a',success=False,reason='physical',complete=True)]
            if self.steps==3:rows.append(dict(env_id=1,case_id='b',success=False,reason='timeout',complete=True))
            return {},None,None,None,dict(episodes=rows)
        def close(self):self.closed=True
    env=Env();run=environment_case_runner(Policy(),lambda batch:env,max_steps=10)
    rows=run([dict(case_id='a'),dict(case_id='b')])
    assert [r['case_id'] for r in rows]==['a','b']
    assert run.control_samples==6 and env.closed

def test_probe_5_warm_50_measure_and_reload56_require_real_save_receipt():
    from src.tasks.velocity.v08.runner import probe_accounting
    result=probe_accounting([float(i) for i in range(1,57)],save_update=6455,reload_update=6455,completed_update=6456)
    assert result['warmup_updates']==5 and result['measurement_updates']==50
    assert result['recovery_updates']==1 and result['measured_seconds']==sum(range(6,56))
    with pytest.raises(ValueError):probe_accounting([1.]*56,save_update=None,reload_update=6455,completed_update=6456)

def test_optional_connection_uses_true_remote_ack_and_v08_identity(tmp_path):
    from src.tasks.velocity.v08.connection import ReliableConnection
    class Backend:
        enabled=True;entity='test';project='test'
        def __init__(self):self.rows=[]
        def start(self,run_id,config):self.run_id=run_id
        def log(self,record,step=None):self.rows.append(dict(record,_step=len(self.rows)))
        def finish(self,result):pass
    backend=Backend()
    class Reader:
        def __init__(self,*args,**kwargs):pass
        def authenticated(self):return True
        def run_identity(self):return dict(name='test-v08',group='v08',config={})
        def rows(self,lo,hi):return [r for r in backend.rows if lo<=r['_step']<hi]
        def close(self):pass
    telemetry=ReliableConnection(tmp_path,backend=backend,reader_factory=Reader,proxy='http://127.0.0.1:7897')
    started=telemetry.start('test-v08',{})
    assert started['remote_verified'] is True
    event=telemetry.emit('test',{'metric':1.},step=6401)
    assert event['id'] not in telemetry.state['acked']
    assert telemetry.flush() is True
    assert event['id'] in telemetry.state['acked']
    assert telemetry.delivery_timeout==120 and telemetry.max_in_flight==32

def test_wandb_explicit_off_overrides_existing_key(monkeypatch):
    from src.tasks.velocity.v08.runtime import wandb_enabled
    monkeypatch.setenv('WANDB_API_KEY','test-only-placeholder')
    monkeypatch.setenv('A2_V08_WANDB','0');assert wandb_enabled() is False
    monkeypatch.setenv('A2_V08_WANDB','1');assert wandb_enabled() is True
    monkeypatch.delenv('A2_V08_WANDB');assert wandb_enabled() is True

def test_case_adapter_honors_requested_eval_parallelism_and_current_phase(monkeypatch):
    from src.tasks.velocity.v08 import runtime
    sizes=[];phases=[]
    class Policy:
        def act(self,obs,deterministic=False):return torch.zeros(obs['n'],12)
    class Env:
        def __init__(self,cases):self.cases=cases
        def get_observations(self):return {'n':len(self.cases)}
        def step(self,actions):return self.get_observations(),None,None,None,{'episodes':[dict(env_id=i,case_id=c['case_id']) for i,c in enumerate(self.cases)]}
        def close(self):pass
    def factory(contract,**kwargs):
        sizes.append(kwargs['num_envs']);phases.append(kwargs['phase']);return Env(kwargs['initial_cases'])
    monkeypatch.setattr(runtime,'build_environment',factory)
    run=runtime.build_case_runner(Policy(),dict(num_envs=2,device='cpu',max_control_steps=10),phase='A')
    run.phase='B';rows=run([dict(case_id=str(i)) for i in range(5)])
    assert sizes==[2,2,1] and phases==['B','B','B']
    assert len(rows)==5 and run.control_samples==5
