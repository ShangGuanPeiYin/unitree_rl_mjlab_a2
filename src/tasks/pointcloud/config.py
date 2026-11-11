"""Explicit, single-GPU SDS settings shared by server entry points."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib

ETH_SHA256 = '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'

@dataclass
class TrainSettings:
    eth_checkpoint: str
    output_dir: str
    eth_sha256: str = ETH_SHA256
    physical_gpu: int = 0
    num_envs: int = 8096
    max_iterations: int = 4000
    rollout: int = 24
    epochs: int = 5
    mini_batches: int = 64
    seed: int = 42
    save_interval: int = 25
    validation_interval: int = 100
    vx_min: float = .2
    vx_max: float = .6
    scan_seed: int = 42
    variant: str = 'full'
    resume: str | None = None

    def validate(self, *, resource_test=False):
        fixed = dict(max_iterations=4000, rollout=24, epochs=5, mini_batches=64,
                     save_interval=25, validation_interval=100, vx_min=.2, vx_max=.6)
        for key, value in fixed.items():
            if getattr(self,key) != value:
                raise ValueError(f'SDS requires {key}={value}; changing recipe requires explicit revision')
        if self.num_envs != 8096 and not resource_test:
            raise ValueError('Formal training requires num_envs=8096; use resource test for capacity measurements')
        if self.num_envs*self.rollout % self.mini_batches:
            raise ValueError('Rollout must divide exactly into minibatches')
        if self.physical_gpu < 0 or self.variant not in ('full','blind'):
            raise ValueError('Invalid GPU or experiment variant')
        if self.eth_sha256 != ETH_SHA256:
            raise ValueError('SDS ETH checksum differs')
        path=Path(self.eth_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.eth_sha256:
            raise ValueError('ETH file checksum mismatch')
        return asdict(self)


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--eth-checkpoint',required=True)
    p.add_argument('--output-dir',required=True)
    for name,default,type_ in [('eth-sha256',ETH_SHA256,str),('physical-gpu',0,int),
        ('num-envs',8096,int),('max-iterations',4000,int),('rollout',24,int),
        ('epochs',5,int),('mini-batches',64,int),('seed',42,int),
        ('save-interval',25,int),('validation-interval',100,int),
        ('vx-min',.2,float),('vx-max',.6,float),('scan-seed',42,int)]:
        p.add_argument('--'+name,default=default,type=type_)
    p.add_argument('--variant',choices=['full','blind'],default='full')
    p.add_argument('--resume',default=None)
    return p
