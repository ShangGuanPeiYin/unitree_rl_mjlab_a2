"""Server-only explicit single-GPU training entry point."""
import os
from src.tasks.pointcloud.config import TrainSettings, parser

def main():
    args=parser('SDS single RTX 4090 training; starts PPO only when explicitly invoked').parse_args()
    settings=TrainSettings(**vars(args))
    settings.validate()
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise RuntimeError('DDP is prohibited for this task')
    os.environ['CUDA_VISIBLE_DEVICES']=str(settings.physical_gpu)
    os.environ['MUJOCO_GL']='egl'
    from src.tasks.pointcloud.environment import PointCloudEnv
    from src.tasks.pointcloud.runner import PointCloudRunner
    from src.tasks.pointcloud.validation import validate_stage
    import torch
    import numpy as np
    import random
    torch.manual_seed(settings.seed); np.random.seed(settings.seed); random.seed(settings.seed)
    if not torch.cuda.is_available(): raise RuntimeError('Training requires one CUDA GPU')
    env=PointCloudEnv(num_envs=settings.num_envs,device='cuda:0',seed=settings.seed,
                      stage='P1',scan_seed=settings.scan_seed)
    try:
        runner=PointCloudRunner(env,settings)
        runner.learn(lambda model,stage,iteration: validate_stage(model,stage,iteration,
                     device='cuda:0',seed=settings.seed+10000,output_dir=settings.output_dir))
    finally:
        env.close()

if __name__=='__main__': main()
