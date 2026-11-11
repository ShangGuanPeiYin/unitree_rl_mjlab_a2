"""Read-only ETH initialization preflight on synthetic inputs (no simulation)."""
import argparse
import json
import torch
from src.tasks.velocity.rl.eth_actor import EthActor, ETH_SHA256
from src.tasks.pointcloud.model import PointCloudActorCritic

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eth-checkpoint',required=True)
    p.add_argument('--eth-sha256',default=ETH_SHA256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--variant',choices=['full','blind'],default='full')
    a=p.parse_args()
    if a.eth_sha256!=ETH_SHA256: p.error('ETH hash must match reviewed SDS')
    torch.manual_seed(a.seed); torch.set_num_threads(1)
    eth=EthActor.load(a.eth_checkpoint,a.device)
    model=PointCloudActorCritic(10,eth,use_points=a.variant=='full').to(a.device)
    errors=[]
    with torch.no_grad():
        for scale in [0,1,1e6]:
            o=torch.randn(a.batch_size,5,45,device=a.device)*scale
            args=dict(obs_history=o,points=torch.zeros(a.batch_size,5,70,3,device=a.device),mask=torch.zeros(a.batch_size,5,70,dtype=torch.bool,device=a.device),ages=torch.zeros(a.batch_size,5,device=a.device))
            errors.append(float((model.actor(**args)-eth(o[:,-1])).abs().max()))
    if max(errors)>1e-6: raise ValueError(errors)
    print(json.dumps(dict(eth_sha256=ETH_SHA256,max_errors=errors,training_started=False,variant=a.variant),indent=2))
if __name__=='__main__': main()
