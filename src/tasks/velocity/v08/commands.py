"""Causal action-command latch and interruptible fixed-length interpolation."""
import torch
from . import spec

class CommandState:
    def __init__(self,num_envs,device='cpu'):
        self.device=torch.device(device)
        self.current=torch.zeros(num_envs,device=device)
        self.executed=self.current.clone();self.requested=self.current.clone();self.start=self.current.clone()
        self.k=torch.full((num_envs,),spec.TRANSITION_STEPS,dtype=torch.long,device=device)
        self.executed_transition=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.executed_requested=self.current.clone()
    def reset(self,ids,value):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        value=torch.as_tensor(value,device=self.device,dtype=self.current.dtype)
        for n in ('current','executed','requested','start','executed_requested'):getattr(self,n)[ids]=value
        self.k[ids]=spec.TRANSITION_STEPS;self.executed_transition[ids]=False
    def request(self,ids,target):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long).flatten()
        target=torch.as_tensor(target,device=self.device,dtype=self.current.dtype).expand(ids.shape)
        if not torch.isfinite(target).all() or (target.abs()>spec.SPEED_MAX+spec.COMMAND_TOLERANCE).any():raise ValueError('speed must be finite in [-0.6,+0.6]')
        changed=(target-self.requested[ids]).abs()>spec.COMMAND_TOLERANCE
        ids=ids[changed];target=target[changed]
        # current is the last observation/action boundary command. At an external
        # boundary it has not yet caused an action, so interrupt from executed.
        self.start[ids]=self.executed[ids];self.current[ids]=self.executed[ids]
        self.requested[ids]=target;self.k[ids]=0
    def advance(self):
        active=self.k<spec.TRANSITION_STEPS
        self.k[active]+=1
        self.current=torch.where(active,self.start+(self.requested-self.start)*self.k/spec.TRANSITION_STEPS,self.requested)
        return self.current
    def latch(self):
        self.executed=self.current.clone();self.executed_requested=self.requested.clone()
        # k==100 is still the last transition sample.
        self.executed_transition=(self.k<=spec.TRANSITION_STEPS)&((self.start-self.requested).abs()>spec.COMMAND_TOLERANCE)
        self.start=torch.where(self.k>=spec.TRANSITION_STEPS,self.requested,self.start)
        return self.executed
    @property
    def parking(self):return self.executed_requested.abs()<=spec.COMMAND_TOLERANCE
    def state_dict(self):return {k:v.clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor)}
    def load_state_dict(self,state):
        for k,v in state.items():getattr(self,k).copy_(v.to(self.device))
