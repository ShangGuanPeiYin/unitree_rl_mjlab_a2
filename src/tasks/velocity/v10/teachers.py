"""Deterministic frozen teacher adapters consuming the student's same snapshot."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .spec import BLIND_PATH
from src.tasks.velocity.rl.ideal_heightmap_actor_critic import FrozenETHNormalizer,mlp
BLIND_SHA='8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'

# Planner-approved, empirical A1 contract; not a claim of teacher training support.
BLIND_YAW_PROTOCOL='a2-v10-blind-yaw-A1-v1'
BLIND_ROUTE_Y_SCALE=.20
BLIND_YAW_LATERAL_GAIN=.20
BLIND_YAW_MIN_SPEED=.10
BLIND_YAW_HEADING_LIMIT=.15
BLIND_YAW_HEADING_GAIN=.50
BLIND_YAW_RATE_LIMIT=.15
BLIND_YAW_RAMP_SPEED=.30


def blind_adapter_configuration():
    """Canonical immutable-by-value recipe for cache/checkpoint teacher identity."""
    return dict(protocol=BLIND_YAW_PROTOCOL,route_y_scale=BLIND_ROUTE_Y_SCALE,
        lateral_gain=BLIND_YAW_LATERAL_GAIN,min_speed=BLIND_YAW_MIN_SPEED,
        desired_heading_limit=BLIND_YAW_HEADING_LIMIT,heading_gain=BLIND_YAW_HEADING_GAIN,
        yaw_rate_limit=BLIND_YAW_RATE_LIMIT,speed_gate_scale=BLIND_YAW_RAMP_SPEED,
        request_vx_column=6,private_command_columns=[6,7,8],route_columns=[0,1,2],
        yaw='atan2(route1,route2)',desired_heading='clip(-atan2(lateral_gain*y,max(vx,min_speed)),+-desired_heading_limit)',
        wrap='atan2(sin(error),cos(error))',yaw_rate='clip(heading_gain*wrap(desired_heading-yaw),+-yaw_rate_limit)*clip(vx/speed_gate_scale,0,1)',
        positive_vx='preserve_request',nonpositive_vx='all_zero',vy='zero',state='none',
        scope='all_blind_calls',student_snapshot='unchanged',stairs='unchanged')


def teacher_identity():
    from . import spec
    encoded=json.dumps(blind_adapter_configuration(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(blind=BLIND_SHA,stairs=spec.STAIRS_SHA256,blind_adapter=hashlib.sha256(encoded).hexdigest())


def blind_yaw_command(proprio,route_error):
    """Same observable request/route -> private teacher command; no state or gates."""
    vx=proprio[:,6]
    y=BLIND_ROUTE_Y_SCALE*route_error[:,0]
    psi=torch.atan2(route_error[:,1],route_error[:,2])
    psi_desired=(-torch.atan2(BLIND_YAW_LATERAL_GAIN*y,vx.clamp_min(BLIND_YAW_MIN_SPEED))).clamp(-BLIND_YAW_HEADING_LIMIT,BLIND_YAW_HEADING_LIMIT)
    error=psi_desired-psi
    wrapped_error=torch.atan2(torch.sin(error),torch.cos(error))
    rate=(BLIND_YAW_HEADING_GAIN*wrapped_error).clamp(-BLIND_YAW_RATE_LIMIT,BLIND_YAW_RATE_LIMIT)
    rate=rate*(vx/BLIND_YAW_RAMP_SPEED).clamp(0,1)
    command=torch.stack((vx,torch.zeros_like(vx),rate),-1)
    return torch.where((vx>0)[:,None],command,torch.zeros_like(command))

class BlindTeacher(nn.Module):
    def __init__(self,path=BLIND_PATH,expected_sha=BLIND_SHA):
        super().__init__()
        if expected_sha!=BLIND_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha: raise ValueError('blind identity mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
        self.normalizer=FrozenETHNormalizer(); self.mlp=mlp([45,512,256,128,12])
        self.normalizer.load_state_dict({k.removeprefix('obs_normalizer.'):v for k,v in state.items() if k.startswith('obs_normalizer.')})
        self.mlp.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')})
        self.eval(); self.requires_grad_(False)
    @torch.no_grad()
    def forward(self,obs):
        proprio=obs['proprio'].clone()
        proprio[:,6:9]=blind_yaw_command(obs['proprio'],obs['route_error'])
        return self.mlp(self.normalizer(proprio))

class DualTeacher(nn.Module):
    def __init__(self,stairs,blind_path=BLIND_PATH,blind_sha=BLIND_SHA):
        super().__init__(); self.stairs=copy.deepcopy(stairs); self.blind=BlindTeacher(blind_path,blind_sha)
        self.eval(); self.requires_grad_(False)
    @torch.no_grad()
    def label(self,obs,stairs_mask):
        mask=torch.as_tensor(stairs_mask,device=obs['proprio'].device,dtype=torch.bool)
        result=torch.empty((len(mask),12),device=mask.device)
        for use_stairs,model in ((False,self.blind),(True,self.stairs)):
            ids=(mask==use_stairs).nonzero().flatten()
            if len(ids): result[ids]=model({k:v[ids] for k,v in obs.items()})
        return result
