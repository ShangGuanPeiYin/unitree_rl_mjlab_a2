"""On-policy PPO with rollout-normalized GAE and frozen-student retention."""
import torch
from . import spec
from .dagger import adam,observation_to
from .replay import index_batch

def gae(rewards,values,done,last_value,gamma=.99,lam=.95):
    if rewards.shape!=values.shape or values.shape!=done.shape: raise ValueError('GAE shape mismatch')
    advantage=torch.zeros_like(values); tail=torch.zeros_like(last_value)
    for t in range(len(rewards)-1,-1,-1):
        live=(~done[t]).to(values.dtype)
        next_value=last_value if t==len(rewards)-1 else values[t+1]
        delta=rewards[t]+gamma*next_value*live-values[t]
        tail=delta+gamma*lam*live*tail; advantage[t]=tail
    return advantage,advantage+values

def epoch_batches(size,generator=None):
    if size%4: raise ValueError('PPO size must divide exactly into 4')
    for _ in range(5): yield list(torch.randperm(size,generator=generator).chunk(4))

def normalize_advantage(a): return (a-a.mean())/(a.std(unbiased=False)+1e-8)

class PPOLearner:
    def __init__(self,model,seed=42):
        self.model=model
        # Both P0 and P1 groups exist at construction so state survives unfreeze.
        self.optimizer=adam(model.parameters()); self.rng=torch.Generator().manual_seed(seed)
    def update(self,rollout,keep,reservoir,phase):
        """Flat rollout obs/actions/log_prob/values/returns/advantages; raw sampled actions."""
        if phase not in ('P0','P1'): raise ValueError(phase)
        self.model.set_phase(phase); device=next(self.model.parameters()).device
        size=len(rollout['actions']); adv=normalize_advantage(rollout['advantages'])
        batch=dict(rollout,advantages=adv); totals=dict(policy=0.,value=0.,entropy=0.,keep=0.,kl=0.)
        for logical_batches in epoch_batches(size,self.rng):
            for ids in logical_batches:
                n=len(ids); self.optimizer.zero_grad(set_to_none=True)
                logical=index_batch(batch,ids)
                for start in range(0,n,spec.MICROBATCH):
                    b=index_batch(logical,range(start,min(start+spec.MICROBATCH,n)))
                    obs=observation_to(b['obs'],device); values=self.model.evaluate(obs).flatten()
                    target=b['returns'].to(device); old_value=b['values'].to(device)
                    value=torch.maximum((values-target).square(),(old_value+(values-old_value).clamp(-.2,.2)-target).square()).sum()/n
                    loss=value; totals['value']+=float(value.detach())
                    if phase=='P1':
                        dist=self.model.action_distribution(obs)
                        logp=dist.log_prob(b['actions'].to(device)).sum(-1)
                        old_logp=b['log_prob'].to(device); ratio=(logp-old_logp).exp(); a=b['advantages'].to(device)
                        policy=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).sum()/n
                        entropy=dist.entropy().sum(-1).sum()/n
                        loss=loss+policy-.003*entropy
                        totals['policy']+=float(policy.detach()); totals['entropy']+=float(entropy.detach())
                        totals['kl']+=float(((ratio-1)-(logp-old_logp)).detach().sum()/n)
                    loss.backward()
                if phase=='P1':
                    old=reservoir.sample_per_task(256); count=len(old['actions'])
                    for start in range(0,count,spec.MICROBATCH):
                        obs=observation_to(index_batch(old,range(start,min(start+spec.MICROBATCH,count)))['obs'],device)
                        with torch.no_grad(): target=keep(obs)
                        retention=(self.model(obs)-target).square().sum()/(12*count)
                        retention.backward(); totals['keep']+=float(retention.detach())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
                self.optimizer.step(); self.model.constrain_std()
        return {k:v/20 for k,v in totals.items()}
    def state_dict(self): return dict(optimizer=self.optimizer.state_dict(),rng=self.rng.get_state())
    def load_state_dict(self,state): self.optimizer.load_state_dict(state['optimizer']); self.rng.set_state(state['rng'])
