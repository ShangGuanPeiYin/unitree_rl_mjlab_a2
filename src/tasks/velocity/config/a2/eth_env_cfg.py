"""ETH inference configuration; existing A2 configurations remain unchanged."""
from copy import deepcopy
from .env_cfgs import unitree_a2_flat_env_cfg

def unitree_a2_eth_flat_env_cfg():
    cfg = deepcopy(unitree_a2_flat_env_cfg(play=True))
    cfg.observations['actor'].terms.pop('phase')
    cfg.observations['actor'].enable_corruption = False
    cfg.events = {k:v for k,v in cfg.events.items() if k in ('reset_base','reset_robot_joints')}
    cfg.events['reset_base'].params['pose_range'] = {k:(0.,0.) for k in ('x','y','z','yaw')}
    cmd = cfg.commands['twist']
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.init_velocity_prob = 0.
    cmd.rel_standing_envs = 0.
    cmd.ranges.lin_vel_x = (0.,0.)
    cmd.ranges.lin_vel_y = (0.,0.)
    cmd.ranges.ang_vel_z = (0.,0.)
    cfg.scene.num_envs = 1
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    return cfg
