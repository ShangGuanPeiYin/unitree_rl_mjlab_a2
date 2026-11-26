"""SDS 10 reward terms: post-physics quantities, pre-action command/phase."""
import torch
from . import spec

class AirtimeTracker:
    def __init__(self,num_envs,device='cpu'):
        self.air=torch.zeros(num_envs,4,device=device)
        self.established=torch.zeros_like(self.air,dtype=torch.bool)
        self.previous=torch.zeros_like(self.established)
    def reset(self,ids):
        self.air[ids]=0; self.established[ids]=False; self.previous[ids]=False
    def update(self,contact,cruise):
        moving=cruise[:,None]
        landing=contact & ~self.previous & self.established & moving
        result=torch.where(landing,self.air-.10,0).sum(-1)
        self.air=torch.where(moving & ~contact & self.established,self.air+spec.DT,0)
        self.established=torch.where(moving,self.established|contact,False)
        self.previous=contact & moving
        return result
    step=update

def compute_reward(t):
    v=t['body_velocity']; w=t['angular_velocity']; g=t['projected_gravity']; c=t['command']; level=t['g_level']
    q=t['q']; q0=t['q0']; qdot=t['qdot']; C=t['reward_contact']; cruise=t['cruise']
    fspeed=t['foot_velocity'][...,:2].norm(dim=-1)
    height=torch.where(t['base_surface_valid'],t['base_z']-t['base_surface_height']-.4,1.5)
    fh=torch.where(t['foot_surface_valid'],(t['foot_site_z']-.032-t['foot_surface_height']).clamp(-.05,1.5),1.5)
    target=torch.where(t['foot_is_stairs'],.22,.075)
    limits=t['hard_limits']; middle=limits.mean(-1); half=(limits[...,1]-limits[...,0])*.45
    lo=middle-half; hi=middle+half
    terms={
      'velocity':torch.exp(-((v[:,0]-c).square()+v[:,1].square())/.25),
      'yaw_velocity':torch.exp(-w[:,2].square()/.25),
      'vertical_velocity':(v[:,2].abs()-.15).clamp_min(0).square(),
      'roll':g[:,1].square(),'pitch':level*g[:,0].square(),'height':height.square(),
      'action_rate':(t['action']-t['previous_action']).square().sum(-1),
      'posture':(q-q0).abs().sum(-1),
      'joint_acceleration':((qdot-t['previous_qdot'])/spec.DT).square().sum(-1),
      'joint_velocity':qdot.square().sum(-1),'angular_xy':w[:,:2].square().sum(-1),
      'airtime':t['airtime_reward'], 'slip':(C*fspeed.square()).sum(-1),
      'foot_clearance':((~C)*(fh-target).square()*fspeed).sum(-1),
      'tracking':(t['q_applied']-q).square().sum(-1),
      'lateral':((t['local_y']/.2).square()).clamp_max(4),
      'heading':1-torch.cos(t['yaw_error']),
      'soft_limits':((lo-q).clamp_min(0)+(q-hi).clamp_min(0)).sum(-1),
      'overspeed':level*(t['heading_speed']-c-.15).clamp_min(0).square(),
      'stable':(t['stable'] & (c.abs()<=1e-7)).to(v.dtype)}
    weights=dict(velocity=1.5,yaw_velocity=.8,vertical_velocity=-1.,roll=-5.,pitch=-5.,height=-.1,
                 action_rate=-.01,posture=-.05,joint_acceleration=-2.5e-7,joint_velocity=-5e-4,
                 angular_xy=torch.where(cruise,-.05,-.20),airtime=.2*cruise,slip=-.15,
                 foot_clearance=-.5,tracking=-.1,lateral=-.5,heading=-.5,soft_limits=-10.,overspeed=-2.,stable=(~cruise).to(v.dtype))
    weighted={k:terms[k]*weights[k]*spec.DT for k in terms}
    failure=t['failure']; top=t['top'] & ~failure; success=t['success'] & ~failure
    weighted.update(displacement=.4*t['delta_x']*cruise,top=2.*top,success=8.*success,failure=-10.*failure)
    reward=sum(weighted.values())
    if not torch.isfinite(reward).all(): raise FloatingPointError('nonfinite v10 reward')
    return reward,dict(raw=terms,weighted=weighted)
