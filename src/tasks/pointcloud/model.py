"""Actor-only sensory interface; privileged observations have a separate critic path."""
import torch
from torch import nn
from torch.nn import functional as F
from src.tasks.velocity.rl.eth_actor import EthActor

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(32, 96)
        self.out = nn.Linear(32, 32)
        self.norm = nn.LayerNorm(32)
    def forward(self, x, mask):
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k = F.elu(q) + 1, (F.elu(k) + 1) * mask[..., None]
        v = v * mask[..., None]
        kv = k.transpose(-1, -2) @ v
        denominator = (q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-6)
        return self.norm(x + self.out((q @ kv) / denominator)) * mask[..., None]

class PointCloudActor(nn.Module):
    def __init__(self, eth=None, use_points=True):
        super().__init__()
        self.eth = eth if eth is not None else EthActor()
        self.use_points = use_points
        if not use_points:
            return
        self.query = nn.Sequential(nn.Linear(5*46+3,512), nn.LayerNorm(512), nn.SiLU(), nn.Linear(512,512), nn.LayerNorm(512), nn.SiLU())
        self.embed = nn.Linear(4,32)
        self.attention = nn.ModuleList([LinearAttention(),LinearAttention()])
        self.q = nn.Linear(512,512)
        self.k = nn.Linear(32,512)
        self.v = nn.Linear(32,512)
        self.fusion = nn.Sequential(nn.Linear(1024,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.delta = nn.Linear(128,12)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        self.velocity = nn.Linear(128,3)
        self.clearance = nn.Linear(128,4)
    def features(self, obs_history, points, mask, ages):
        n=self.eth.obs_normalizer
        normalized=((obs_history-n._mean)/n._std.clamp_min(1e-5)).clamp(-5,5)
        s=self.query(torch.cat((torch.cat((normalized,torch.tensor([.08,.06,.04,.02,0.],device=normalized.device,dtype=normalized.dtype)[None,:,None].expand(normalized.shape[0],-1,-1)),-1).flatten(1),obs_history[:,-1,6:9]),-1))
        valid=mask.flatten(1).bool()
        p=torch.where(mask[...,None],points,torch.zeros_like(points))
        x=self.embed(torch.cat((p,ages[:,:,None,None].expand(-1,-1,70,1)),-1).flatten(1,2))
        for layer in self.attention: x=layer(x,valid)
        b=x.shape[0]
        q=self.q(s).reshape(b,16,1,32)
        k=self.k(x).reshape(b,-1,16,32).transpose(1,2)
        v=self.v(x).reshape(b,-1,16,32).transpose(1,2)
        logits=torch.matmul(q,k.transpose(-1,-2)).squeeze(-2)/32**0.5
        weights=torch.softmax(logits.masked_fill(~valid[:,None,:],-1e9),-1)*valid[:,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        z=torch.matmul(weights.unsqueeze(-2),v).squeeze(-2).reshape(b,512)
        if not self.use_points: z=torch.zeros_like(z)
        return self.fusion(torch.cat((s,z),-1))
    def forward(self, obs_history, points, mask, ages):
        if not self.use_points:
            return self.eth(obs_history[:,-1])
        h=self.features(obs_history,points,mask,ages)
        return self.eth(obs_history[:,-1])+self.delta(h)
    def auxiliary(self, obs_history, points, mask, ages):
        if not self.use_points:
            return None, None
        h=self.features(obs_history,points,mask,ages)
        return self.velocity(h),self.clearance(h)

class PointCloudActorCritic(nn.Module):
    def __init__(self, privileged_dim, eth=None, use_points=True):
        super().__init__()
        self.actor=PointCloudActor(eth,use_points)
        self.critic=nn.Sequential(nn.Linear(privileged_dim,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.full((12,),float(torch.log(torch.tensor(.2)))))
    def distribution(self, inputs):
        return torch.distributions.Normal(self.actor(**inputs),self.log_std.clamp(-6,2).exp())
    def freeze_eth(self, frozen):
        self.actor.eth.mlp.requires_grad_(not frozen)
