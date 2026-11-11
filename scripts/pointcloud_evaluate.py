"""Deterministic held-out inference only; never invokes PPO updates."""
import argparse
import json
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--eth-checkpoint',type=Path)
    p.add_argument('--method',choices=['full','blind','eth','occlusion'],default='full')
    p.add_argument('--condition',choices=['standard','degraded'],default='standard')
    p.add_argument('--heldout-seed',type=int,required=True)
    p.add_argument('--physical-gpu',type=int,default=0)
    p.add_argument('--episodes-per-speed',type=int,default=200)
    p.add_argument('--speeds',type=float,nargs='+',default=[.2,.3,.4,.5,.6])
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.heldout_seed in [42,43,44]: p.error('Heldout seed must differ from training seeds')
    if a.episodes_per_speed!=200 or a.speeds!=[.2,.3,.4,.5,.6]: p.error('Final acceptance requires 200 episodes at each of five SDS speeds')
    if int(os.environ.get('WORLD_SIZE','1'))!=1: p.error('DDP prohibited')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.physical_gpu); os.environ['MUJOCO_GL']='egl'
    import torch
    from src.tasks.pointcloud.model import PointCloudActorCritic
    from src.tasks.velocity.rl.eth_actor import EthActor
    from src.tasks.pointcloud.validation import validate_stage
    torch.manual_seed(a.heldout_seed)
    if a.method=='eth':
        if a.eth_checkpoint is None: p.error('--eth-checkpoint required for ETH inference')
        model=PointCloudActorCritic(10,EthActor.load(a.eth_checkpoint,'cuda:0'),use_points=False).to('cuda:0')
    else:
        if a.checkpoint is None: p.error('--checkpoint required')
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        full=state['config'].get('variant','full')=='full'
        if (a.method=='blind')==full: p.error('Checkpoint architecture does not match requested method')
        model=PointCloudActorCritic(state['model']['critic.0.weight'].shape[1],use_points=full).to('cuda:0')
        model.load_state_dict(state['model'],strict=True)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    stage=('P4',40,.2,1600,tuple(a.speeds),200)
    passed,report=validate_stage(model,stage,0,device='cuda:0',seed=a.heldout_seed,output_dir=a.output_dir,episodes_per_speed=a.episodes_per_speed,success_threshold=.90 if a.condition=='degraded' else .95,degraded=a.condition=='degraded',occluded=a.method=='occlusion')
    (a.output_dir/'acceptance.json').write_text(json.dumps(dict(passed=passed,method=a.method,condition=a.condition,heldout_seed=a.heldout_seed,speeds=report),indent=2))
if __name__=='__main__': main()
