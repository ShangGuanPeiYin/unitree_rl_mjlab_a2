"""Bounded deterministic inference diagnostics; no optimizer or learning updates."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eth-checkpoint',required=True)
    p.add_argument('--physical-gpu',type=int,default=0)
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--max-steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scan-seed',type=int,default=42)
    p.add_argument('--stage',choices=['P1','P2a','P2b','P2c','P3a','P3b','P3c','P4','P5'],default='P1')
    p.add_argument('--vx',type=float,default=.2)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.num_envs<=4 or not 1<=a.max_steps<=5000:
        p.error('Diagnostics hard limit: 1..4 environments, 1..5000 steps')
    if not .2<=a.vx<=.6 and a.vx!=0: p.error('vx must be zero or within 0.2..0.6')
    if a.stage!='P1' and a.vx==0: p.error('stairs do not use zero command')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.physical_gpu)
    os.environ['MUJOCO_GL']='egl'
    import numpy as np
    import torch
    from src.tasks.pointcloud.environment import PointCloudEnv
    from src.tasks.velocity.rl.eth_actor import EthActor
    torch.manual_seed(a.seed)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    eth=EthActor.load(a.eth_checkpoint,'cuda:0')
    env=PointCloudEnv(num_envs=a.num_envs,device='cuda:0',seed=a.seed,stage=a.stage,
                      evaluation=True,command_speed=a.vx,scan_seed=a.scan_seed)
    rows=[]
    try:
        obs=env.reset()
        with torch.no_grad():
            for step in range(a.max_steps):
                action=eth(obs['actor']['obs_history'][:,-1])
                obs,reward,failure,success,timeout,info=env.step(action)
                terminal=info['terminal_observation']['actor']
                state=info['terminal_state']
                row={k:v.detach().cpu().numpy() for k,v in terminal.items()}
                row.update({k:state[k].detach().cpu().numpy() for k in ['body_world','body_terrain','feet']})
                row.update(failure=failure.cpu().numpy(),success=success.cpu().numpy(),timeout=timeout.cpu().numpy())
                rows.append(row)
                # Never concatenate a reset episode into the same diagnostic track.
                if (failure|success|timeout).any(): break
        arrays={k:np.stack([row[k] for row in rows]) for k in rows[0]}
        np.savez_compressed(a.output_dir/'trajectory.npz',**arrays)
        clouds=arrays['points'][-1,0].reshape(-1,3)
        valid=arrays['mask'][-1,0].reshape(-1)
        circles=[]
        for x,y,z in clouds[valid]:
            circles.append(f'<circle cx="{(x+1.2)*250:.2f}" cy="{(0.5-z)*300:.2f}" r="2" fill="#245fa0"/>')
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="500" viewBox="0 0 640 500"><rect width="640" height="500" fill="white"/><text x="10" y="480">Body-frame x/z projection; acquisition-frame history, no registration</text>'+''.join(circles)+'</svg>'
        (a.output_dir/'pointcloud_xz.svg').write_text(svg)
        (a.output_dir/'summary.json').write_text(json.dumps(dict(steps=len(rows),num_envs=a.num_envs,stage=a.stage,vx=a.vx,training_started=False,policy='original ETH deterministic',stopped_on_first_end=bool((failure|success|timeout).any()),coordinates='points: acquisition body frame; feet: terrain frame; body_world: world frame'),indent=2))
    finally: env.close()

if __name__=='__main__': main()
