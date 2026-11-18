"""Fixed final windows; each sample comes from actual simulated kinematics."""
import torch
from . import spec

class StandWindow:
    def __init__(self,num_envs,device='cpu'):
        self.device=torch.device(device);b=num_envs;w=spec.ENDPOINT_WINDOW_STEPS
        self.count=torch.zeros(b,dtype=torch.long,device=device)
        self.xy=torch.zeros(b,w,2,device=device)
        self.values=torch.zeros(b,w,10,device=device)
        self.last_height=torch.zeros(b,4,device=device)
        self.height_valid=torch.zeros(b,4,dtype=torch.bool,device=device)
    def reset(self,ids):
        for value in (self.count,self.xy,self.values,self.last_height,self.height_valid):value[ids]=0
    def support_height(self,heights,support):
        self.last_height=torch.where(support,heights,self.last_height);self.height_valid|=support
        # torch.median returns lower middle; statistical median needs mean of 2.
        sorted_h=self.last_height.sort(-1).values
        return sorted_h[:,1:3].mean(-1),self.height_valid.all(-1)
    def update(self,xy,speed,roll,pitch,yaw,omega,height,foot_speed,four_support,surface_ok,mid):
        vals=torch.stack((speed,roll.abs(),pitch.abs(),yaw.abs(),omega,height,foot_speed,four_support.float(),surface_ok.float(),torch.ones_like(speed)),-1)
        self.xy=torch.cat((self.xy[:,1:],xy[:,None]),1);self.values=torch.cat((self.values[:,1:],vals[:,None]),1);self.count+=1
        result=torch.zeros_like(mid)
        for ismid,w in ((False,spec.ENDPOINT_WINDOW_STEPS),(True,spec.MID_WINDOW_STEPS)):
            v=self.values[:,-w:];p=self.xy[:,-w:]
            valid=(self.count>=w)&(v[:,:,0].mean(-1)<=spec.STAND_MEAN_SPEED)&(v[:,:,0].amax(-1)<=spec.STAND_MAX_SPEED)
            valid&=(v[:,:,1].amax(-1)<=(spec.MID_ROLL if ismid else spec.LEVEL_ROLL))&(v[:,:,2].amax(-1)<=(spec.MID_PITCH if ismid else spec.LEVEL_PITCH))
            valid&=(v[:,:,4].amax(-1)<=spec.STAND_OMEGA)&(v[:,:,5].amin(-1)>=(spec.MID_HEIGHT if ismid else spec.LEVEL_HEIGHT))
            valid&=(v[:,:,6].mean(-1)<=spec.STAND_FOOT_SPEED)&(v[:,:,7].mean(-1)>=spec.STAND_FOUR_FRACTION)
            valid&=((p[:,-1]-p[:,0]).norm(dim=-1)<=spec.STAND_DRIFT)&v[:,:,8].bool().all(-1)
            if ismid:valid&=v[:,:,3].amax(-1)<=spec.MID_YAW
            result|=valid&(mid==ismid)
        return result
    def state_dict(self):return {k:v.clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor)}
    def load_state_dict(self,state):
        for k,v in state.items():setattr(self,k,v.to(self.device).clone())
