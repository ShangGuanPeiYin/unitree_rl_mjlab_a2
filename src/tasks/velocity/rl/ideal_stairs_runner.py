"""Isolated SDS PPO, strict new-episode resume, and validation curriculum gates."""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch

STAGE_LIMITS = (500, 2000, 2000, 2500, 3000, 4000)
ETH_PATH = '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt'
TASK = 'Unitree-A2-IdealHeightMap-Stairs'

@dataclass(frozen=True)
class PPOConfig:
    rollout: int = 48
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    max_lr: float = 3e-4
    desired_kl: float = .01
    clip: float = .2
    gamma: float = .99
    gae_lambda: float = .95
    entropy: float = .005
    value_coef: float = 1.
    max_grad_norm: float = 1.

CONTRACT = dict(version=1, task=TASK, observation={'proprio':[45], 'history':[5,45],
    'heightmap':[3,48,32], 'privileged':[35]}, control_dt=.02, physics_dt=.005,
    action_scale=.25, foot_order=['FL','FR','RL','RR'], map_hz=10,
    stage_limits=list(STAGE_LIMITS), reward_version='SDS-v1.3', ppo=asdict(PPOConfig()),
    architecture_version='a2-ideal-heightmap-v1',
    eth_sha256='8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722',
    joint_order=[leg+'_'+joint for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')],
    default_angles=[-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8],
    kp=[100,100,150],kd=[4,4,6],torque_limits=[120,120,180],armature=.03,
    map_bounds=[-.8,1.6,-.8,.8],map_resolution=.05,map_height_clip=1.5,map_age_scale=.5,
    map_cache='sample_frame_hold',normalization='frozen_ETH_clip5_std_floor1e-5',
    terrain={'N':[0,4,8,12,20,40],'h':[0,.08,.12,.16,.2,.2],'d':.2,'w':.8,'t':.01,'tile':[24,8]},
    curriculum={'ratios':[[1,0,0],[.1,0,.9],[.1,.2,.7]],'ramp':400,'eval_interval':200,
      'save_interval':100,'gate_repeats':2,'gate_successes':18,'gate_episodes':20},
    support={'force':5,'normal_deg':30,'top_tolerance':.005,'samples':2,'radius':.032},
    terminal={'stop_seconds':3,'speed':.05,'roll_pitch_deg':70,'nonfoot_seconds':.1,'nonfoot_force':10})

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

CORE_SOURCES = (
    'src/tasks/velocity/rl/ideal_stairs_runner.py',
    'scripts/train_a2_ideal_stairs.py',
    'scripts/evaluate_a2_ideal_stairs.py',
    'scripts/run_a2_ideal_stairs.py',
    'src/tasks/velocity/rl/ideal_heightmap_actor_critic.py',
    'src/tasks/velocity/mdp/ideal_stairs_env.py',
    'src/tasks/velocity/mdp/ideal_stairs_contacts.py',
    'src/tasks/velocity/mdp/ideal_heightmap.py',
    'src/terrains/ideal_stairs_atlas.py',
    'src/tasks/velocity/config/a2/ideal_stairs_env_cfg.py',
)
_ROOT = Path(__file__).resolve().parents[4]
CONTRACT['behavior_sha256'] = {p: sha256(_ROOT/p) for p in CORE_SOURCES}

def finite(name, value):
    if not torch.isfinite(value).all(): raise FloatingPointError(f'Non-finite {name}')

def atomic_json(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)); os.replace(tmp,path)

class Curriculum:
    def __init__(self):
        self.stage=0; self.stage_iteration=0; self.iteration=0
        self.consecutive_passes=0; self.consumed=[0]*6; self.history=[]
    def updated(self):
        self.iteration+=1; self.stage_iteration+=1; self.consumed[self.stage]+=1
    def validate(self, summary):
        expected=[0.,.2,.4,.6] if self.stage==0 else [.2,.4,.6]
        if sorted(x.get('vx',-1) for x in summary.get('speeds',[]))!=expected or any(x.get('episodes')!=20 for x in summary.get('speeds',[])):
            raise ValueError('Validation must contain every required speed with exactly 20 episodes')
        if self.stage==0:
            passed=all(x['failures']==0 for x in summary['speeds'])
            for x in summary['speeds']:
                passed &= x['mean_speed']<.05 if x['vx']==0 else abs(x['mean_vx']-x['vx'])/x['vx']<=.2
        else:
            passed=all(x['episodes']==20 and x['successes']>=18 for x in summary['speeds']) and len(summary['speeds'])==3
        self.consecutive_passes=self.consecutive_passes+1 if passed else 0
        self.history.append(dict(iteration=self.iteration,stage=self.stage,passed=bool(passed),summary=summary))
        return self.consecutive_passes>=2
    @property
    def exhausted(self): return self.stage_iteration>=STAGE_LIMITS[self.stage]
    def advance(self):
        if self.consecutive_passes<2 or self.stage>=5: raise RuntimeError('Invalid curriculum advance')
        self.stage+=1; self.stage_iteration=0; self.consecutive_passes=0
    def state_dict(self): return dict(vars(self))
    def load_state_dict(self,state):
        if set(state)!=set(vars(self)): raise ValueError('Curriculum fields incompatible')
        if sum(state['consumed'])!=state['iteration']: raise ValueError('Consumed budget mismatch')
        if state['consumed'][state['stage']]!=state['stage_iteration']: raise ValueError('Stage budget mismatch')
        if any(v<0 or v>limit for v,limit in zip(state['consumed'],STAGE_LIMITS,strict=True)): raise ValueError('Invalid budget')
        self.__dict__.update(state)

class PPO:
    def __init__(self,model,env,cfg=PPOConfig()):
        if cfg!=PPOConfig(): raise ValueError('PPO configuration differs from approved SDS')
        self.model=model; self.env=env; self.cfg=cfg
        self.optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
        self.obs=None; self.storage=None
    def reset(self): self.obs=self.env.reset()[0]; self.storage=None
    def collect(self):
        if self.obs is None: self.reset()
        c=self.cfg; n=self.env.num_envs; device=self.env.device
        # One shared raw map, never detached encoded features.
        obs={k:torch.empty((c.rollout,*v.shape),dtype=v.dtype,device=device) for k,v in self.obs.items()}
        fields={k:torch.empty((c.rollout,n),device=device) for k in ('logp','value','reward','terminated','done','next_value')}
        actions=torch.empty((c.rollout,n,12),device=device)
        means=torch.empty_like(actions); stds=torch.empty_like(actions)
        records=[]; reward_sums={}
        self.model.train()
        # Update once before the rollout and freeze throughout PPO: consistent old/new values.
        self.model.update_critic_statistics(self.obs)
        for t in range(c.rollout):
            with torch.no_grad():
                for k,v in self.obs.items(): obs[k][t].copy_(v)
                dist=self.model.action_distribution(self.obs); action=dist.sample()
                finite('actions',action)
                value=self.model.evaluate(self.obs).squeeze(-1)
                next_obs,reward,terminated,truncated,info=self.env.step(action)
                terminal_obs=info.get('terminal_observation',next_obs)
                next_value=self.model.evaluate(terminal_obs).squeeze(-1)
                for key,val in [('logp',dist.log_prob(action).sum(-1)),('value',value),('reward',reward),
                    ('terminated',terminated),('done',terminated|truncated),('next_value',next_value)]: fields[key][t].copy_(val)
                actions[t].copy_(action); means[t].copy_(dist.mean); stds[t].copy_(dist.stddev)
                records.extend(info.get('episodes',[])); self.obs=next_obs
                for key,val in info.get('reward_terms',{}).items():reward_sums[key]=reward_sums.get(key,0.)+val
        advantage=torch.zeros_like(fields['reward']); last=torch.zeros(n,device=device)
        for t in reversed(range(c.rollout)):
            delta=fields['reward'][t]+c.gamma*(1-fields['terminated'][t])*fields['next_value'][t]-fields['value'][t]
            last=delta+c.gamma*c.gae_lambda*(1-fields['done'][t])*last
            advantage[t]=last
        returns=advantage+fields['value']
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        self.storage=dict(obs={k:v.flatten(0,1) for k,v in obs.items()},actions=actions.flatten(0,1),
            means=means.flatten(0,1),stds=stds.flatten(0,1),advantage=advantage.flatten(),returns=returns.flatten(),
            **{k:v.flatten() for k,v in fields.items()})
        self.rollout_metrics={key:val/c.rollout for key,val in reward_sums.items()}
        self.rollout_metrics.update(map_valid_fraction=float(self.obs['heightmap'][:,1].mean()),map_age_mean=float(self.obs['heightmap'][:,2].mean()))
        return records
    def update(self):
        c=self.cfg; s=self.storage
        if s is None: raise RuntimeError('Missing rollout')
        batch=len(s['advantage'])
        if batch%c.minibatches: raise ValueError('Rollout not divisible into minibatches')
        result=[]
        for _ in range(c.epochs):
            for idx in torch.randperm(batch,device=self.env.device).chunk(c.minibatches):
                obs={k:v[idx] for k,v in s['obs'].items()}; dist=self.model.action_distribution(obs)
                value=self.model.evaluate(obs).squeeze(-1)
                with torch.no_grad():
                    kl=(torch.log(dist.stddev/s['stds'][idx])+(s['stds'][idx]**2+(s['means'][idx]-dist.mean)**2)/(2*dist.stddev**2)-.5).sum(-1).mean()
                    finite('KL',kl)
                    lr=self.optimizer.param_groups[0]['lr']
                    if kl>2*c.desired_kl: lr=max(c.min_lr,lr/1.5)
                    elif 0<kl<c.desired_kl/2: lr=min(c.max_lr,lr*1.5)
                    for group in self.optimizer.param_groups: group['lr']=lr
                ratio=torch.exp(dist.log_prob(s['actions'][idx]).sum(-1)-s['logp'][idx])
                policy=torch.maximum(-s['advantage'][idx]*ratio,-s['advantage'][idx]*ratio.clamp(1-c.clip,1+c.clip)).mean()
                clipped=s['value'][idx]+(value-s['value'][idx]).clamp(-c.clip,c.clip)
                value_loss=torch.maximum((value-s['returns'][idx])**2,(clipped-s['returns'][idx])**2).mean()
                entropy=dist.entropy().sum(-1).mean()
                loss=policy+c.value_coef*value_loss-c.entropy*entropy; finite('loss',loss)
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                for name,p in self.model.named_parameters():
                    if p.grad is not None: finite(f'gradient:{name}',p.grad)
                grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),c.max_grad_norm)
                self.optimizer.step()
                self.model.constrain_std()
                result.append([float(loss),float(policy),float(value_loss),float(entropy),float(kl),float(grad)])
        values=np.mean(result,axis=0)
        return dict(zip(('loss','policy_loss','value_loss','entropy','kl','gradient_norm'),values.tolist()),lr=self.optimizer.param_groups[0]['lr'],std_min=float(dist.stddev.min()),std_max=float(dist.stddev.max()),**self.rollout_metrics)
    def save(self,path,curriculum,metadata):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        state=dict(contract=CONTRACT,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            curriculum=curriculum.state_dict(),metadata=metadata,num_envs=self.env.num_envs,
            rng=dict(environment=self.env.rng.get_state() if hasattr(self.env,'rng') else None,python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),resume_mode='new_episode')
        tmp=path.with_suffix('.tmp'); torch.save(state,tmp); os.replace(tmp,path)
        return sha256(path)
    def load(self,path,curriculum,expected_sha):
        if sha256(path)!=expected_sha: raise ValueError('Checkpoint hash mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['contract']!=CONTRACT or state['resume_mode']!='new_episode': raise ValueError('Checkpoint contract incompatible')
        self.model.load_state_dict(state['model'],strict=True); self.optimizer.load_state_dict(state['optimizer'])
        curriculum.load_state_dict(state['curriculum'])
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy']); torch.set_rng_state(state['rng']['torch'])
        if state['rng']['cuda']: torch.cuda.set_rng_state_all(state['rng']['cuda'])
        self.env.set_stage(curriculum.stage,curriculum.stage_iteration)
        if state['rng']['environment'] is not None: self.env.rng.set_state(state['rng']['environment'])
        self.reset()
        return state
