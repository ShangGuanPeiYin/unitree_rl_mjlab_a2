"""Strict, inference-only adapter for the supplied ETH A2 checkpoint."""
import hashlib
from pathlib import Path
import torch
from torch import nn

ETH_SHA256 = '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        for key, value in {'_mean': torch.zeros(1,45), '_var': torch.ones(1,45), '_std': torch.ones(1,45), 'count': torch.tensor(0.)}.items():
            self.register_buffer(key, value)

class EthActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.ones(12), requires_grad=False)
        self.obs_normalizer = Normalizer()
        self.mlp = nn.Sequential(nn.Linear(45,512), nn.ELU(), nn.Linear(512,256), nn.ELU(), nn.Linear(256,128), nn.ELU(), nn.Linear(128,12))

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = obs['actor']
        if obs.shape[-1] != 45 or not torch.isfinite(obs).all():
            raise ValueError('ETH observations must be finite and 45-dimensional')
        n = self.obs_normalizer
        action = self.mlp(((obs-n._mean)/n._std.clamp_min(1e-5)).clamp(-5,5))
        if not torch.isfinite(action).all():
            raise ValueError('Non-finite ETH action')
        return action

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != ETH_SHA256:
            raise ValueError('ETH checkpoint SHA256 mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        actor = cls()
        state = checkpoint['actor_state_dict']
        for key, value in state.items():
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite checkpoint parameter: {key}')
        actor.load_state_dict(state, strict=True)
        if (actor.obs_normalizer._std <= 0).any() or (actor.obs_normalizer._var < 0).any():
            raise ValueError('Invalid normalization statistics')
        actor.requires_grad_(False)
        return actor.to(device).eval()
