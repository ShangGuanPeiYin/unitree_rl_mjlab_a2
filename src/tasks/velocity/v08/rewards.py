"""SDS v08 differences composed over the unchanged v07 reward computer."""
from dataclasses import replace
import torch
from ..v07.rewards import RewardFrame,RewardResult
from ..v07 import rewards as inherited
from . import spec

def parking_adjustments(*,dx,executed,transition,parking,level,gravity_x,angular_xy_sq,stand):
    cruise=~transition&~parking&(executed.abs()>spec.COMMAND_TOLERANCE)
    return dict(forward_progress=spec.FORWARD_PROGRESS_PER_M*executed.sign()*dx*cruise,
        parking_pitch=spec.CONTROL_DT*spec.PARK_PITCH_WEIGHT*gravity_x.square()*parking*level,
        parking_angular=spec.CONTROL_DT*(spec.PARK_ANG_WEIGHT-spec.REWARD_WEIGHTS['ang_vel_xy'])*angular_xy_sq*parking,
        standing=spec.CONTROL_DT*spec.STAND_WEIGHT*stand*parking*(executed.abs()<=spec.COMMAND_TOLERANCE))

class RewardComputer:
    def __init__(self):self.base=inherited.RewardComputer();self.was_parking=None
    @property
    def state(self):return self.base.state
    def reset(self,ids=None):
        self.base.reset(ids)
        if self.was_parking is not None:
            if ids is None:self.was_parking.zero_()
            else:self.was_parking[ids]=False
    def __call__(self,frame,*,transition,parking,level,stand,arrival,success,failure):
        b=frame.executed_command.numel();self.base._ensure_state(b,frame.executed_command.device)
        if self.was_parking is None:self.was_parking=torch.zeros_like(parking)
        # Stop accumulation throughout parking and clear once more on restart;
        # inherited settling and differencing are otherwise exactly unchanged.
        clear=parking|self.was_parking
        if clear.any():self.base.state.air_time.air_time[clear]=0.
        zeros=torch.zeros_like(failure)
        result=self.base(replace(frame,success=zeros,failure=zeros,resets=zeros))
        terms=result.weighted_terms;raw=result.raw_terms
        adjustment=parking_adjustments(dx=raw['forward_progress'],executed=frame.executed_command,transition=transition,parking=parking,level=level,gravity_x=frame.gravity_b[:,0],angular_xy_sq=frame.body_ang_vel_b[:,:2].square().sum(-1),stand=stand)
        terms.update(adjustment);terms['feet_air_time']=torch.where(parking,torch.zeros_like(terms['feet_air_time']),terms['feet_air_time'])
        terms['arrival']=spec.ARRIVAL_REWARD*(arrival&~failure).float();terms['success']=spec.SUCCESS_REWARD*(success&~failure).float();terms['failure']=spec.FAILURE_REWARD*failure.float()
        raw.update(arrival=(arrival&~failure).float(),success=(success&~failure).float(),failure=failure.float())
        self.was_parking=parking.clone()
        return RewardResult(raw,terms,torch.stack(list(terms.values())).sum(0))
    def state_dict(self):return dict(base=None if self.base.state is None else self.base.state.state_dict(),was_parking=None if self.was_parking is None else self.was_parking.clone())
    def load_state_dict(self,value,num_envs,device):
        if value['base'] is not None:self.base._ensure_state(num_envs,torch.device(device));self.base.state.load_state_dict(value['base'])
        self.was_parking=None if value['was_parking'] is None else value['was_parking'].to(device).clone()
