"""Signed, independent constant-cruise stall windows; diagnostics only."""
import torch
from . import spec

class StallDiagnostics:
    def __init__(self,num_envs,device='cpu'):
        b=num_envs;self.device=torch.device(device);w=spec.STALL_HISTORY_SAMPLES
        self.x_history=torch.zeros(b,w,device=device);self.m_history=torch.zeros_like(self.x_history)
        for n in ('count','consecutive_count','active_samples','stall_samples'):setattr(self,n,torch.zeros(b,dtype=torch.long,device=device))
        for n in ('dx_1s','highwater_dx_1s','last_command'):setattr(self,n,torch.zeros(b,device=device))
        for n in ('previous_active','window_ready','qualifying','penalty_on'):setattr(self,n,torch.zeros(b,dtype=torch.bool,device=device))
    def reset(self,ids,initial_x=None):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        for v in vars(self).values():
            if isinstance(v,torch.Tensor):v[ids]=0
    def update(self,x,executed,transition,parking,first_step,endpoint):
        active=(executed.abs()>spec.STALL_ACTIVE_COMMAND)&~transition&~parking&~endpoint
        changed=(executed-self.last_command).abs()>spec.COMMAND_TOLERANCE
        new=active&(~self.previous_active|changed)
        signed=executed.sign()*x
        self.x_history=torch.where(new[:,None],signed[:,None],self.x_history)
        self.m_history=torch.where(new[:,None],signed[:,None],self.m_history)
        self.count=torch.where(new,torch.zeros_like(self.count),self.count)
        high=torch.maximum(self.m_history[:,-1],signed)
        self.x_history=torch.cat((self.x_history[:,1:],signed[:,None]),-1);self.m_history=torch.cat((self.m_history[:,1:],high[:,None]),-1)
        self.count=torch.where(active,(self.count+1).clamp_max(spec.STALL_HISTORY_SAMPLES),0)
        self.dx_1s=self.x_history[:,-1]-self.x_history[:,0];self.highwater_dx_1s=self.m_history[:,-1]-self.m_history[:,0]
        self.window_ready=active&(self.count>=spec.STALL_HISTORY_SAMPLES)
        self.qualifying=self.window_ready&first_step&(self.dx_1s<spec.STALL_DX_FRACTION_PER_SECOND*executed.abs()*spec.STALL_DX_THRESHOLD_SCALE)&(self.highwater_dx_1s<spec.STALL_HIGHWATER_DX)
        self.consecutive_count=torch.where(self.qualifying,self.consecutive_count+1,0)
        self.penalty_on=self.qualifying&(self.consecutive_count>=spec.STALL_CONSECUTIVE_SAMPLES)
        self.active_samples+=active.long();self.stall_samples+=self.penalty_on.long();self.previous_active=active;self.last_command=executed.clone()
        return self
    def state_dict(self):return {k:v.clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor)}
    def load_state_dict(self,state):
        for k,v in state.items():getattr(self,k).copy_(v.to(self.device))
