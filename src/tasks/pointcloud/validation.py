"""Deterministic first-episode validation, without optimizer calls."""
from pathlib import Path
import json
import torch
from .evaluation import EpisodeSpeed, stage_gate, wilson


def episode(model, env, command, flat, *, occluded=False, video_path=None):
    obs=env.reset()
    metric=EpisodeSpeed(command,flat)
    trajectory=[]
    writer=None
    if video_path is not None:
        import imageio.v2 as imageio
        writer=imageio.get_writer(video_path,fps=50)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(1500 if flat else 4500):
                inputs=obs['actor']
                if occluded:
                    inputs={**inputs,'mask':torch.zeros_like(inputs['mask'])}
                action=model.actor(**inputs)
                obs,reward,failure,success,timeout,info=env.step(action)
                state=info['terminal_state']
                row={k:(v[0].detach().cpu().tolist() if isinstance(v,torch.Tensor) else v) for k,v in state.items()}
                metric.update(row['elapsed'],row['body_terrain'][0],row['stair_end'],
                    row['world_velocity'],row['terrain_yaw'],row['body_world'][:2],
                    bool(failure[0]),bool(success[0]),bool(timeout[0]),row)
                trajectory.append(row)
                if writer is not None:
                    frame=info['terminal_frame']
                    if frame is not None: writer.append_data(frame)
                if metric.finished: break
    finally:
        if writer is not None: writer.close()
    result=metric.result()
    result['trajectory']=trajectory
    return result


def validate_stage(model,stage,iteration,*,device,seed,output_dir,
                   episodes_per_speed=None,success_threshold=.90,degraded=False,occluded=False):
    from .environment import PointCloudEnv
    name,steps,height,budget,speeds,count=stage
    count=count if episodes_per_speed is None else episodes_per_speed
    records={}
    folder=Path(output_dir)/f'evaluation_{iteration:06d}_{name}'
    folder.mkdir(parents=True,exist_ok=False)
    for speed in speeds:
        env=PointCloudEnv(num_envs=1,device=device,seed=seed+int(speed*1000),stage=name,
            evaluation=True,command_speed=speed,terrain_kind='flat' if name=='P1' else 'stairs',
            perception_degraded=degraded,record_video=True)
        try:
            records[speed]=[]
            for index in range(count):
                video=folder/f'vx_{speed:.1f}_episode_{index:04d}.mp4'
                result=episode(model,env,speed,name=='P1',occluded=occluded,video_path=video)
                (folder/f'vx_{speed:.1f}_episode_{index:04d}.json').write_text(json.dumps(result))
                records[speed].append({k:v for k,v in result.items() if k!='trajectory'})
        finally: env.close()
    passed,report=stage_gate(records,flat=name=='P1',success_threshold=success_threshold)
    for speed,rows in records.items():
        n=sum(bool(r['termination'] and r['termination']['success']) for r in rows)
        report[speed]['success_interval_95']=wilson(n,len(rows))
        report[speed]['episodes']=len(rows)
    (folder/'summary.json').write_text(json.dumps({'passed':passed,'speeds':report},indent=2))
    return passed,report
