"""Deterministic frozen teacher adapters consuming the student's same snapshot."""
import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
from src.tasks.velocity.rl.ideal_heightmap_actor_critic import FrozenETHNormalizer,mlp
BLIND_SHA='8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'

class BlindTeacher(nn.Module):
    def __init__(self,path='/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt',expected_sha=BLIND_SHA):
        super().__init__()
        if expected_sha!=BLIND_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha: raise ValueError('blind identity mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
        self.normalizer=FrozenETHNormalizer(); self.mlp=mlp([45,512,256,128,12])
        self.normalizer.load_state_dict({k.removeprefix('obs_normalizer.'):v for k,v in state.items() if k.startswith('obs_normalizer.')})
        self.mlp.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')})
        self.eval(); self.requires_grad_(False)
    @torch.no_grad()
    def forward(self,obs): return self.mlp(self.normalizer(obs['proprio']))

class DualTeacher(nn.Module):
    def __init__(self,stairs,blind_path='/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt',blind_sha=BLIND_SHA):
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
