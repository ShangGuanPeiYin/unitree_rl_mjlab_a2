"""Independent PPO model implementing SDS v1.3's frozen ETH input contract.

Statistics updates are explicit, never a side effect of forward/evaluate. Rollout
and evaluation therefore cannot silently modify either normalization contract.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal

ETH_SHA256 = '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'


def mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def map_encoder():
    return nn.Sequential(nn.Conv2d(3,16,3,2,1), nn.ELU(),
                         nn.Conv2d(16,32,3,2,1), nn.ELU(),
                         nn.Conv2d(32,32,3,2,1), nn.ELU(), nn.Flatten(),
                         nn.Linear(768,64), nn.ELU())


def history_encoder():
    return nn.Sequential(nn.Flatten(), mlp([225,128,64]), nn.ELU())


class FrozenETHNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        for key, value in {'_mean':torch.zeros(1,45), '_var':torch.ones(1,45),
                           '_std':torch.ones(1,45), 'count':torch.tensor(0.)}.items():
            self.register_buffer(key, value)

    def forward(self, value):
        return ((value-self._mean)/self._std.clamp_min(1e-5)).clamp(-5,5)


class CriticNormalizer(nn.Module):
    """Population moments for 45 proprio +35 privileged; binary flags stay raw."""
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(80))
        self.register_buffer('var', torch.ones(80))
        self.register_buffer('count', torch.tensor(0., dtype=torch.float64))
        mask = torch.ones(80, dtype=torch.bool)
        mask[48:52] = False
        self.register_buffer('continuous', mask)

    @torch.no_grad()
    def update(self, value):
        if not self.training or value.shape[0] == 0:
            return
        n = value.shape[0]
        batch_mean, batch_var = value.mean(0), value.var(0, unbiased=False)
        total = self.count + n
        delta = batch_mean-self.mean
        mean = self.mean + delta * (n/total).to(value.dtype)
        var = (self.var*self.count + batch_var*n + delta.square()*self.count*n/total)/total
        self.mean.copy_(torch.where(self.continuous, mean, 0))
        self.var.copy_(torch.where(self.continuous, var, 1))
        self.count.copy_(total)

    def forward(self, value):
        normalized = ((value-self.mean)/self.var.sqrt().clamp_min(1e-5)).clamp(-5,5)
        return torch.where(self.continuous, normalized, value)


class FusedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = FrozenETHNormalizer()
        self.map_encoder = map_encoder()
        self.history_encoder = history_encoder()
        self.mlp = mlp([45,512,256,128,12])
        self.fusion = nn.Linear(128,512)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, obs):
        features = torch.cat((self.map_encoder(obs['heightmap']),
                              self.history_encoder(self.normalizer(obs['history']))), -1)
        hidden = self.mlp[1](self.mlp[0](self.normalizer(obs['proprio'])) + self.fusion(features))
        return self.mlp[2:](hidden)


class IdealHeightmapActorCritic(nn.Module):
    """forward=deterministic mean; evaluate=value; distribution is unsquashed Normal.

    Inputs are raw SDS observations. No action scaling/clipping is performed here;
    the control path owns its single final limit and 0.25 joint-offset scale.
    """
    architecture_version = 'a2-ideal-heightmap-v1'

    def __init__(self):
        super().__init__()
        self.actor = FusedActor()
        self.critic_map_encoder = map_encoder()
        self.critic_history_encoder = history_encoder()
        self.critic_normalizer = CriticNormalizer()
        self.critic = mlp([208,512,256,128,1])
        self.std = nn.Parameter(torch.full((12,), .30))
        self.transfer_metadata = {}

    @staticmethod
    def _validate(obs, critic=False):
        shapes = {'proprio':(45,), 'history':(5,45), 'heightmap':(3,48,32)}
        if critic:
            shapes['privileged'] = (35,)
        batch = None
        for key, shape in shapes.items():
            value = obs[key]
            if value.ndim != len(shape)+1 or tuple(value.shape[1:]) != shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid finite observation shape: {key}, expected [B,{shape}]')
            if batch is not None and batch != value.shape[0]:
                raise ValueError('Observation batch sizes differ')
            batch = value.shape[0]

    def forward(self, obs):
        self._validate(obs)
        return self.actor(obs)

    def evaluate(self, obs):
        self._validate(obs, critic=True)
        raw = self.critic_normalizer(torch.cat((obs['proprio'], obs['privileged']), -1))
        # History uses critic-owned current-proprio population moments, never ETH stats.
        history = ((obs['history']-self.critic_normalizer.mean[:45]) /
                   self.critic_normalizer.var[:45].sqrt().clamp_min(1e-5)).clamp(-5,5)
        return self.critic(torch.cat((raw, self.critic_map_encoder(obs['heightmap']),
                                      self.critic_history_encoder(history)), -1))

    def update_critic_statistics(self, obs):
        self._validate(obs, critic=True)
        self.critic_normalizer.update(torch.cat((obs['proprio'],obs['privileged']), -1).detach())

    @torch.no_grad()
    def constrain_std(self):
        """Project the trainable std after each optimizer step (distribution also clamps)."""
        self.std.clamp_(.05, .30)

    def action_distribution(self, obs):
        return Normal(self(obs), self.std.clamp(.05,.30))

    def act(self, obs, deterministic=False):
        return self(obs) if deterministic else self.action_distribution(obs).sample()

    @classmethod
    def from_eth(cls, path, device='cpu'):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != ETH_SHA256:
            raise ValueError('ETH checkpoint SHA256 mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['actor_state_dict']
        expected = {'std'} | {'obs_normalizer.'+k for k in FrozenETHNormalizer().state_dict()} | {'mlp.'+k for k in mlp([45,512,256,128,12]).state_dict()}
        if set(state) != expected:
            raise ValueError('ETH actor keys do not exactly match migration contract')
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Non-finite ETH parameter')
        if (state['obs_normalizer._std'] <= 0).any() or (state['obs_normalizer._var'] < 0).any() or (state['std'] <= 0).any():
            raise ValueError('Invalid ETH normalization or std')
        model = cls()
        model.actor.mlp.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('mlp.')}, strict=True)
        model.actor.normalizer.load_state_dict({k[len('obs_normalizer.'):]:v for k,v in state.items() if k.startswith('obs_normalizer.')}, strict=True)
        with torch.no_grad():
            model.std.copy_(state['std'].clamp(.05,.30))
        model.transfer_metadata = {'sha256':ETH_SHA256, 'source_iteration':int(checkpoint.get('iter',2000)),
                                   'std_before':state['std'].tolist(), 'std_after':model.std.detach().tolist()}
        return model.to(device)
