from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

def test_control_boundary_refresh_survives_rollout_observation_assignment():
    from src.tasks.velocity.v08 import model,ppo
    class Env:
        num_envs=8;device='cpu'
        def reset(self):return model.synthetic_observations(batch=8),{}
        def step(self,actions):
            return model.synthetic_observations(batch=8),torch.zeros(8),torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool),{}
    learner=ppo.PPOTrainer(model.V08ActorCritic(),Env())
    def refresh(payload):
        learner.obs={k:v.clone() for k,v in learner.obs.items()}
        learner.obs['proprio'][:,0]=.123
    learner.control_step_callback=refresh
    learner.collect()
    assert torch.all(learner.obs['proprio'][:,0]==.123)

def test_smoke_learning_keeps_real6400_sample_baseline():
    from src.tasks.velocity.v08 import model,ppo,continuation_v08
    class Env:num_envs=8;device='cpu'
    learner=ppo.PPOTrainer(model.V08ActorCritic(),Env())
    source=continuation_v08.load_designated_source()
    learner.load_learning_counters(source['learning'])
    assert learner.learning['control_samples']==157286400
    learner.learning_state['completed_ppo_updates']+=1
    learner.learning_state['optimizer_steps']+=20
    assert learner.learning['control_samples']==157286400+384

def test_signed_request_callback_changes_next_actor_but_keeps_old_action_reward():
    from src.tasks.velocity.v08 import model,ppo
    class Env:
        num_envs=8;device='cpu'
        def __init__(self):self.command=.5;self.steps=0
        def get_observations(self):
            obs=model.synthetic_observations(batch=8);obs['proprio'][:,6]=self.command
            return obs
        def reset(self):return self.get_observations(),{}
        def request_speed(self,ids,target):self.command=target
        def step(self,actions):
            self.steps+=1;old=self.command
            return self.get_observations(),torch.full((8,),old),torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool),{}
    env=Env();net=model.V08ActorCritic();learner=ppo.PPOTrainer(net,env)
    def change(payload):
        if payload['step_index']==1:
            env.request_speed(list(range(8)),-.5);learner.obs=env.get_observations()
    learner.control_step_callback=change
    learner.collect()
    assert torch.all(learner.storage['obs']['proprio'][:8,6]==.5)
    assert torch.all(learner.storage['obs']['proprio'][8:16,6]==-.5)
    assert torch.all(learner.storage['reward'][:8]==.5)
    assert torch.all(learner.storage['reward'][8:16]==-.5)
