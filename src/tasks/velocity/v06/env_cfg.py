"""v0.6 scene configuration: terrain, contacts, sensors and observation terms.

The terrain generator, contact sensors and observation keys come from the v0.6
production line (:mod:`src.tasks.velocity.v06.terrain` and
:mod:`src.tasks.velocity.v06.spec`).  No v0.2 atlas or stage table is reachable
from this configuration; the returned ``ManagerBasedRlEnvCfg`` is handed to
:class:`src.tasks.velocity.v06.env.V06Env` only.
"""
from __future__ import annotations

from ..config.a2.eth_env_cfg import unitree_a2_eth_flat_env_cfg
from . import spec
from .terrain import terrain_generator_cfg

#: v0.6 episode cap for the scene cfg; the real truncation is the per-tile
#: SDS 6.2 timeout computed by ``v06.events.timeout_steps``.
EPISODE_LENGTH_S = 160.0


def env_cfg(num_envs: int = 512, seed: int = 42):
    """Build the v0.6 ``ManagerBasedRlEnvCfg`` on the real 22-tile terrain."""
    cfg = unitree_a2_eth_flat_env_cfg()
    cfg.seed = int(seed)
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.terrain.terrain_type = 'generator'
    cfg.scene.terrain.terrain_generator = terrain_generator_cfg(seed)
    cfg.scene.terrain.max_init_terrain_level = 0
    # The v0.6 environment composes rewards/terminations/events itself from spec.
    cfg.events = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.observations = {'actor': cfg.observations['actor']}
    cfg.scale_rewards_by_dt = True  # inert: no scene reward terms exist
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 64
    cfg.sim.njmax = 600
    _validate(cfg)
    return cfg


#: W0-compatible alias (``env_cfg`` is the frozen name).
v06_env_cfg = env_cfg


def _validate(cfg):
    if cfg.scene.terrain.terrain_type != 'generator':
        raise RuntimeError('v06 scene must use the generated v0.6 terrain')
    generator = cfg.scene.terrain.terrain_generator
    if generator is None or 'v06_atlas' not in tuple(generator.sub_terrains):
        raise RuntimeError('v06 scene is not wired to the v06 atlas sub-terrain')
    if int(cfg.scene.num_envs) < 1:
        raise RuntimeError('v06 scene needs a positive environment count')
    if cfg.decimation != int(spec.DECIMATION):
        raise RuntimeError('v06 scene decimation drifted from spec')
    if abs(float(cfg.sim.mujoco.timestep) - float(spec.PHYSICS_DT)) > 1e-12:
        raise RuntimeError('v06 scene physics dt drifted from spec')
    sensors = {str(sensor.name) for sensor in cfg.scene.sensors}
    for required in ('feet_ground_contact', 'nonfoot_ground_touch'):
        if required not in sensors:
            raise RuntimeError(f'v06 scene lost the {required} contact sensor')
    if tuple(cfg.observations) != ('actor',):
        raise RuntimeError('v06 scene must expose only the actor observation group')


def observation_shapes():
    """The five v0.6 observation keys and their per-environment shapes."""
    return {'proprio': (int(spec.PROPRIO_DIM),),
            'history': (int(spec.HISTORY_LEN), int(spec.PROPRIO_DIM)),
            'heightmap': tuple(int(v) for v in spec.HEIGHTMAP_SHAPE),
            'privileged': (int(spec.PRIVILEGED_DIM),),
            'route_error': (int(spec.ROUTE_DIM),)}
