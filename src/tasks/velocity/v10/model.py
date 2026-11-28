"""v1.0 model: exact S2 Actor; independent seeded PPO Critic initialization."""
import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from src.tasks.velocity.v09.model import V09ActorCritic
from .spec import INITIAL_PATH,INITIAL_SHA256

STAIRS_SHA=INITIAL_SHA256  # Legacy name: student initialization remains S2.

class V10ActorCritic(V09ActorCritic):
    architecture_version='a2-v10-native40x30'
    def __init__(self):
        super().__init__(); self.set_phase('D1')
    @classmethod
    def from_stairs(cls,path=INITIAL_PATH,expected_sha=STAIRS_SHA,device='cpu'):
        if expected_sha!=STAIRS_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError('S2 teacher identity mismatch')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=cls(); model.load_state_dict(checkpoint['model'],strict=True)
        return model.to(device)
    def set_phase(self,phase):
        if phase not in ('D1','D2','D3','P0','P1'): raise ValueError(phase)
        for name,p in self.named_parameters():
            p.requires_grad_((name.startswith('actor.') and phase!='P0') or
                             (name.startswith('critic') and phase in ('P0','P1')) or
                             (name=='std' and phase=='P1'))
        self.phase=phase; self.actor.normalizer.eval(); self.critic_normalizer.eval()
    def initialize_ppo(self,seed=42):
        keep=copy.deepcopy(self).eval()
        for p in keep.parameters(): p.requires_grad_(False)
        # fork_rng restores caller RNG, and only CPU RNG is used for initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            fresh=V10ActorCritic()
            for name,module in fresh.named_modules():
                if name.startswith('critic') and isinstance(module,(nn.Linear,nn.Conv2d)):
                    module.reset_parameters()
            nn.init.zeros_(fresh.critic[-1].weight); nn.init.zeros_(fresh.critic[-1].bias)
        state=self.state_dict()
        for name,t in fresh.state_dict().items():
            if name.startswith(('critic.','critic_map_encoder.','critic_history_encoder.')): state[name]=t
        self.load_state_dict(state)
        with torch.no_grad(): self.std.fill_(.10)
        self.set_phase('P0')
        return keep
    def action_distribution(self,obs):
        return Normal(self(obs),self.std.clamp(.05,.30))
    @torch.no_grad()
    def constrain_std(self): self.std.clamp_(.05,.30)
