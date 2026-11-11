"""SERVER ONLY: full PPO capacity measurement; executing this performs learning updates."""
import argparse
import json
import os
from pathlib import Path

def main():
    from src.tasks.pointcloud.config import TrainSettings, ETH_SHA256
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eth-checkpoint',required=True)
    p.add_argument('--eth-sha256',default=ETH_SHA256)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-envs',type=int,nargs='+',default=[1024,2048,4096,8096])
    p.add_argument('--warmup-iterations',type=int,default=10)
    p.add_argument('--measure-iterations',type=int,default=200)
    p.add_argument('--memory-limit-gib',type=float,default=21)
    p.add_argument('--rollout',type=int,default=24)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--mini-batches',type=int,default=64)
    a=p.parse_args()
    if a.num_envs != [1024,2048,4096,8096] or a.measure_iterations != 200 or a.memory_limit_gib != 21: p.error('SDS resource ladder, 200 measured updates and 21 GiB limit are fixed')
    if a.warmup_iterations<1: p.error('Warmup must be positive')
    if int(os.environ.get('WORLD_SIZE','1'))!=1: p.error('DDP prohibited')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.physical_gpu); os.environ['MUJOCO_GL']='egl'
    import torch
    from src.tasks.pointcloud.environment import PointCloudEnv
    from src.tasks.pointcloud.runner import PointCloudRunner
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for count in a.num_envs:
        settings=TrainSettings(eth_checkpoint=a.eth_checkpoint,eth_sha256=a.eth_sha256,output_dir=str(a.output_dir),physical_gpu=a.physical_gpu,num_envs=count,seed=a.seed,rollout=a.rollout,epochs=a.epochs,mini_batches=a.mini_batches)
        settings.validate(resource_test=True)
        env=None
        try:
            torch.manual_seed(a.seed)
            env=PointCloudEnv(num_envs=count,device='cuda:0',seed=a.seed,stage='P5',scan_seed=a.seed)
            runner=PointCloudRunner(env,settings,resource_test=True)
            # P5 geometry/delays/randomization and all parameters unfrozen.
            from src.tasks.pointcloud.curriculum import Curriculum
            runner.curriculum=Curriculum(index=8)
            runner.iteration=25
            for _ in range(a.warmup_iterations): runner.update()
            torch.cuda.reset_peak_memory_stats()
            rows=[]
            for _ in range(a.measure_iterations):
                metrics=runner.update()
                rows.append(metrics)
                with (a.output_dir/f'envs_{count}.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
            peak=torch.cuda.max_memory_allocated()
            report=dict(num_envs=count,peak_memory_bytes=peak,warmup_iterations=a.warmup_iterations,measure_iterations=a.measure_iterations,memory_pass=peak<=a.memory_limit_gib*1024**3,contact_capacity_hits=sum(r['contact_capacity_hits'] for r in rows),constraint_capacity_hits=sum(r['constraint_capacity_hits'] for r in rows))
            (a.output_dir/f'envs_{count}_summary.json').write_text(json.dumps(report,indent=2))
            if report['contact_capacity_hits'] or report['constraint_capacity_hits']: raise RuntimeError('Contact/constraint capacity reached; resource gate failed')
            if not report['memory_pass']: raise RuntimeError('21 GiB capacity gate failed; no automatic fallback')
        except Exception as error:
            (a.output_dir/'failure.json').write_text(json.dumps(dict(num_envs=count,error=repr(error),automatic_fallback=False),indent=2))
            raise
        finally:
            if env is not None: env.close()
        del runner,env
        torch.cuda.empty_cache()
if __name__=='__main__': main()
