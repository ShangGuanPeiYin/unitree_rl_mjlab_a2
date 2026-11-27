"""Equal-task DAgger optimizer; aggregation happens after all current updates."""
import torch
from .replay import index_batch
from . import spec

def adam(parameters):
    return torch.optim.Adam(parameters,lr=spec.LR,betas=(.9,.999),eps=1e-8,weight_decay=0)

def observation_to(obs,device): return {k:v.to(device) for k,v in obs.items()}

class DAggerLearner:
    def __init__(self,model):
        self.model=model; self.optimizer=adam(p for p in model.parameters() if p.requires_grad)
    def update(self,current,reservoir,num_envs):
        if type(num_envs) is not int or num_envs<=0:raise ValueError('positive integer environment count required')
        device=next(self.model.parameters()).device; total=0.
        for _ in range(10 if num_envs==512 else 20):
            batch=reservoir.sample_logical(current); size=len(batch['actions'])
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0,size,spec.MICROBATCH):
                micro=index_batch(batch,range(start,min(start+spec.MICROBATCH,size)))
                prediction=self.model(observation_to(micro['obs'],device))
                loss=(prediction-micro['actions'].to(device)).square().sum()/(12*size)
                loss.backward(); total+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step()
        reservoir.add(current)
        return {'loss':total/(10 if num_envs==512 else 20),'grad_norm':float(norm)}
