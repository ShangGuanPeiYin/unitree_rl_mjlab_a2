"""Single-environment deterministic ETH flat evaluation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from src.tasks.velocity.config.a2.eth_env_cfg import unitree_a2_eth_flat_env_cfg
from src.tasks.velocity.rl.eth_actor import EthActor,ETH_SHA256

class RecordedEnv(ManagerBasedRlEnv):
    terminal_snapshot = None
    terminal_reasons = None
    def snapshot(self):
        d=self.scene['robot'].data
        return {k:getattr(d,k).detach().cpu().numpy().copy() for k in ('root_link_pos_w','root_link_quat_w','root_link_lin_vel_b','joint_pos','joint_vel','actuator_force','root_link_ang_vel_b','root_link_lin_vel_w')}
    def _reset_idx(self,env_ids=None):
        if getattr(self,'common_step_counter',0)>0:
            self.terminal_snapshot=self.snapshot()
            self.terminal_reasons={name:bool(self.termination_manager.get_term(name).any()) for name in self.termination_manager.active_terms}
        return super()._reset_idx(env_ids)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',action='store_true')
    p.add_argument('--checkpoint',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-envs',type=int,choices=[1],default=1)
    p.add_argument('--duration',type=float,default=30.)
    p.add_argument('--task',choices=['Unitree-A2-ETH-Flat'],default='Unitree-A2-ETH-Flat')
    p.add_argument('--speeds',nargs=3,type=float,default=[0.,.3,.6])
    p.add_argument('--vy',type=float,default=0.)
    p.add_argument('--wz',type=float,default=0.)
    args=p.parse_args()
    if args.duration <= 0: p.error('--duration must be positive')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
    actor=EthActor.load(args.checkpoint,args.device)
    results=[]
    for name,speeds in [('stand',[args.speeds[0]]),('walk_03',[args.speeds[1]]),('walk_06',[args.speeds[2]]),('stop_go',[args.speeds[0],args.speeds[1],args.speeds[0]])]:
        cfg=unitree_a2_eth_flat_env_cfg(); cfg.seed=args.seed
        cfg.commands['twist'].ranges.lin_vel_y=(args.vy,args.vy)
        cfg.commands['twist'].ranges.ang_vel_z=(args.wz,args.wz)
        cfg.viewer.width=640; cfg.viewer.height=480
        env=RecordedEnv(cfg,device=args.device,render_mode="rgb_array" if args.video else None)
        frames=[]
        expected=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
        assert list(env.scene['robot'].joint_names)==expected,env.scene['robot'].joint_names
        assert env.action_manager.get_term('joint_pos')._target_names == expected
        obs,_=env.reset(seed=args.seed)
        initial=env.snapshot()
        rows=[]; terminated=timeout=False; finite=True
        term=env.command_manager.get_term('twist')
        for step in range(round(args.duration/env.step_dt)):
            vx=speeds[min(int(step*env.step_dt/(args.duration/len(speeds))),len(speeds)-1)]
            term.cfg.ranges.lin_vel_x=(vx,vx)
            term.command[:]=torch.tensor([vx,args.vy,args.wz],device=args.device)
            obs=env.observation_manager.compute(update_history=False)
            with torch.inference_mode(): action=actor(obs)
            obs,_,done,trunc,_=env.step(action)
            state=env.terminal_snapshot or env.snapshot()
            state['target_angles']=env.action_manager.get_term('joint_pos')._processed_actions.detach().cpu().numpy().copy()
            state['action']=action.cpu().numpy(); state['command']=np.array([vx,args.vy,args.wz]);state['time']=np.array((step+1)*env.step_dt)
            rows.append(state)
            if args.video and step % 5 == 0: frames.append(env.render())
            finite=all(np.isfinite(v).all() for v in state.values())
            terminated=bool(done.any());timeout=bool(trunc.any())
            if terminated or timeout or not finite: break
        path=out/name;path.mkdir()
        if frames:
            import mediapy
            mediapy.write_video(str(path/'preview.mp4'),frames,fps=10)
        np.savez_compressed(path/'trajectory.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        result={'case':name,'seconds':len(rows)*env.step_dt,'terminated':terminated,'termination_reasons':env.terminal_reasons or {},'falls':int(bool((env.terminal_reasons or {}).get('fell_over',False))),'timeout':timeout,'automatic_resets':int(terminated or timeout),'finite':finite,'initializations':1,'displacement_world':(rows[-1]['root_link_pos_w'][0]-initial['root_link_pos_w'][0]).tolist(),'mean_velocity_body':np.mean([r['root_link_lin_vel_b'][0] for r in rows],axis=0).tolist()}
        (path/'result.json').write_text(json.dumps(result,indent=2));results.append(result);env.close()
    (out/'summary.json').write_text(json.dumps({'sha256':ETH_SHA256,'arguments':vars(args),'results':results},indent=2))
if __name__=='__main__':main()
