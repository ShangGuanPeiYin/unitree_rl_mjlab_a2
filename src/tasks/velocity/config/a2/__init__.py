from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_a2_flat_env_cfg,
  unitree_a2_rough_env_cfg,
)
from .rl_cfg import unitree_a2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-A2-Rough",
  env_cfg=unitree_a2_rough_env_cfg(),
  play_env_cfg=unitree_a2_rough_env_cfg(play=True),
  rl_cfg=unitree_a2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-A2-Flat",
  env_cfg=unitree_a2_flat_env_cfg(),
  play_env_cfg=unitree_a2_flat_env_cfg(play=True),
  rl_cfg=unitree_a2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from .eth_env_cfg import unitree_a2_eth_flat_env_cfg

register_mjlab_task(
  task_id="Unitree-A2-ETH-Flat",
  env_cfg=unitree_a2_eth_flat_env_cfg(),
  play_env_cfg=unitree_a2_eth_flat_env_cfg(),
  rl_cfg=unitree_a2_ppo_runner_cfg(),
)

def unitree_a2_ideal_stairs_rl_cfg():
  cfg = unitree_a2_ppo_runner_cfg()
  cfg.experiment_name = "a2_ideal_stairs"
  cfg.num_steps_per_env = 48
  cfg.max_iterations = 14000
  cfg.save_interval = 100
  cfg.algorithm.learning_rate = 1e-4
  cfg.algorithm.entropy_coef = .005
  # Dedicated runner additionally enforces LR bounds, strict checkpoints and gates.
  return cfg

# This task uses its dedicated step/runner through scripts/train.py and play.py.
from .ideal_stairs_env_cfg import unitree_a2_ideal_stairs_env_cfg
register_mjlab_task(
  task_id="Unitree-A2-IdealHeightMap-Stairs",
  env_cfg=unitree_a2_ideal_stairs_env_cfg(),
  play_env_cfg=unitree_a2_ideal_stairs_env_cfg(num_envs=1),
  rl_cfg=unitree_a2_ideal_stairs_rl_cfg(),
)

# V02 remains a separate environment and dedicated execution route.
from src.tasks.velocity.v02.env_cfg import v02_env_cfg
def unitree_a2_ideal_stairs_v02_rl_cfg():
  cfg=unitree_a2_ideal_stairs_rl_cfg()
  cfg.experiment_name='a2_ideal_stairs_v02'
  cfg.max_iterations=28000
  return cfg

register_mjlab_task(
  task_id='Unitree-A2-IdealHeightMap-Stairs-V02',
  env_cfg=v02_env_cfg(),play_env_cfg=v02_env_cfg(num_envs=1),
  rl_cfg=unitree_a2_ideal_stairs_v02_rl_cfg(),
)
