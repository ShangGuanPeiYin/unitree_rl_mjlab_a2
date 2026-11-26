"""Three independent contact signals; physical extraction is reused, not old filtering."""
import torch

class ContactFilter:
    def __init__(self,n,device):
        self.previous_raw=torch.zeros(n,4,dtype=torch.bool,device=device)
        self.previous_top=torch.zeros_like(self.previous_raw)
    def reset(self,ids):
        self.previous_raw[ids]=False;self.previous_top[ids]=False
    def update(self,force,upward,inside):
        raw=force.norm(dim=-1)>5.
        top=upward>5.
        result=dict(contact_raw=raw,reward_contact=raw|self.previous_raw,
                    safe_support=top&self.previous_top&inside)
        self.previous_raw=raw.clone();self.previous_top=top.clone()
        return result
