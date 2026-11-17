"""CPU-testable 50 Hz climb, parking and exact four-class allocation contracts."""
from dataclasses import dataclass
import numpy as np
import torch
from .spec import STAGE_BY_ID, STAGES


def allocate_stage_slots(num_envs,stage,previous_stage=None):
    """Return tile ids, explicit class ids and counts, using ordered remainders."""
    if num_envs<1 or stage not in STAGE_BY_ID: raise ValueError('invalid allocation')
    current=STAGE_BY_ID[stage]
    if stage=='flat': weights=np.array([1.,0.,0.,0.])
    elif stage=='R0': weights=np.array([.4,.1,0.,.5])
    else: weights=np.array([.25,.05,.2,.5])
    if current.N and stage!='R0':
        expected=STAGES[[s.id for s in STAGES].index(stage)-1].id
        if previous_stage is None: previous_stage=expected
        if previous_stage!=expected: raise ValueError('Previous stage must be immediate formal predecessor')
    raw=weights*num_envs;counts=np.floor(raw).astype(np.int64)
    order=sorted(range(4),key=lambda k:(-(raw[k]-counts[k]),k))
    for k in order[:num_envs-int(counts.sum())]: counts[k]+=1
    classes=np.repeat(np.arange(4,dtype=np.int64),counts)
    previous=STAGE_BY_ID[previous_stage].tile_id if previous_stage else 0
    tiles=np.array([0,0,previous,current.tile_id],dtype=np.int64)[classes]
    return tiles,classes,counts


@dataclass(frozen=True)
class ClimbResult:
    success: torch.Tensor
    climbed: torch.Tensor
    current_support: torch.Tensor
    recent_support: torch.Tensor
    consecutive_steps: torch.Tensor


class ClimbState:
    """Debounce each foot on one (tile,surface); retain support for <=50 samples.

    ``candidate`` already means an actual >5 N top-face contact and a safe foot
    projection. A surface identifies its physical coplanar support plane, never
    merely the foot coordinates. ``feet_safe`` is the *current* platform inset.
    """
    def __init__(self,num_envs,device='cpu'):
        self.last_surface=torch.full((num_envs,4),-1,dtype=torch.long,device=device)
        self.last_tile=torch.full_like(self.last_surface,-1)
        self.support_count=torch.zeros_like(self.last_surface)
        self.last_support_step=torch.full_like(self.last_surface,-1000000)
        self.consecutive_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.climbed=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.climb_step=torch.full((num_envs,),-1,dtype=torch.long,device=device)
        self.current_support=torch.zeros_like(self.last_surface,dtype=torch.bool)

    def reset(self,ids):
        for x in (self.last_surface,self.last_tile): x[ids]=-1
        self.support_count[ids]=0;self.last_support_step[ids]=-1000000
        self.consecutive_steps[ids]=0;self.climbed[ids]=False;self.climb_step[ids]=-1
        self.current_support[ids]=False

    def update(self,step,stairs,tile_ids,surface_ids,candidate,feet_safe,body_x,x_top,body_y,failure):
        valid=candidate & feet_safe & stairs[:,None] & (surface_ids>=0)
        same=valid & (surface_ids==self.last_surface) & (tile_ids[:,None]==self.last_tile)
        self.support_count=torch.where(same,self.support_count+1,valid.long())
        self.last_surface=torch.where(valid,surface_ids,-1)
        self.last_tile=torch.where(valid,tile_ids[:,None],-1)
        self.current_support=valid & (self.support_count>=2)
        self.last_support_step=torch.where(self.current_support,step[:,None],self.last_support_step)
        recent=(step[:,None]-self.last_support_step)<=50
        current=stairs & (body_x>=x_top+.8) & (body_y.abs()<=.2) & feet_safe.all(-1) & recent.all(-1) & (self.current_support.sum(-1)>=2) & ~failure
        self.consecutive_steps=torch.where(current,self.consecutive_steps+1,0)
        event=(self.consecutive_steps>=5) & ~self.climbed & ~failure
        self.climbed|=event
        self.climb_step=torch.where(event,step,self.climb_step)
        return ClimbResult(event.clone(),self.climbed.clone(),self.current_support.clone(),recent,self.consecutive_steps.clone())


class ParkingDiagnosticState:
    """Diagnostic only: latch climb, require 150 safe slow cycles, cap at 10 s."""
    def __init__(self,num_envs,device='cpu'):
        self.parking_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.succeeded=torch.zeros(num_envs,dtype=torch.bool,device=device)
    def reset(self,ids):
        self.parking_steps[ids]=0;self.succeeded[ids]=False
    def update(self,step,climb_step,climbed,platform_support,body_y,speed,failure):
        # Counting starts after the arrival sample; no extra x-line restriction.
        entered=climbed & (step>climb_step)
        good=entered & platform_support.all(-1) & (body_y.abs()<=.2) & (speed<.05) & ~failure
        self.parking_steps=torch.where(good,self.parking_steps+1,0)
        self.succeeded|=(self.parking_steps>=150)&~failure
        timeout=climbed & ((step-climb_step)>=500)
        return self.succeeded.clone(),timeout
