"""Discoverable task registration with an explicit specialized runner boundary."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.config.a2.rl_cfg import unitree_a2_ppo_runner_cfg
from .env_cfg import pointcloud_env_cfg

class DedicatedEntryRequired:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('PointCloud requires its terminal-observation and sensor lifecycle; use scripts/pointcloud_train.py with explicit ETH path. Generic scripts/train.py is not compatible.')

cfg=unitree_a2_ppo_runner_cfg()
cfg.max_iterations=4000
cfg.num_steps_per_env=24
cfg.save_interval=25
register_mjlab_task(task_id='Unitree-A2-PointCloud-OpenStairs',
    env_cfg=pointcloud_env_cfg(num_envs=8096),play_env_cfg=pointcloud_env_cfg(num_envs=1),
    rl_cfg=cfg,runner_cls=DedicatedEntryRequired)
