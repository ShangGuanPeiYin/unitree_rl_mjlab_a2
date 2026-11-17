"""Configuration isolated from ETH and all existing task factories."""
from .eth_env_cfg import unitree_a2_eth_flat_env_cfg
from src.terrains.ideal_stairs_atlas import atlas_generator_cfg


def unitree_a2_ideal_stairs_env_cfg(num_envs=1024, seed=42):
    cfg = unitree_a2_eth_flat_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.terrain_type = 'generator'
    cfg.scene.terrain.terrain_generator = atlas_generator_cfg(seed)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.events = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.observations = {'actor': cfg.observations['actor']}
    cfg.scale_rewards_by_dt = True
    cfg.episode_length_s = 160.
    cfg.sim.nconmax = 64
    cfg.sim.njmax = 600
    return cfg
