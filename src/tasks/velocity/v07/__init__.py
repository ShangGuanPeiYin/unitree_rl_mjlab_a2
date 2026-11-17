"""Independent A2 ideal-heightmap stairs 0.6 protocol (schema 8,
``v06-route-state-v1``).

The package exports the frozen v0.6 identity, the 22-tile terrain table, the
phase/substage budget table, the reward-term table, the curriculum constants and
the contract/manifest helpers only.  It never imports the legacy ``src.tasks``
root initializer, never scans a task directory and never registers a task or a
gym environment, so importing this package cannot change the behaviour hash of
another version.

``V07ActorCritic`` is exported lazily on first attribute access so importing
``v07.spec`` (for example from ``speed.py``) never drags in ``torch``.
"""
from .spec import (ACTION_SCALE, ARCHITECTURE, CODE_VERSION, CONTROL_DT,
                   CRITIC_WIDTH, CURRICULUM, DECIMATION, FINAL_TILE_ID,
                   FLAT_TILE_ID, GATE_CASES, GATE_INTERVAL, GATE_PASS_MIN,
                   GATE_SPEEDS_F0_H_D_L, GATE_SPEEDS_V_R, HEIGHTMAP_SHAPE,
                   HISTORY_LEN, INIT_PATH, INIT_SHA256, LOG_ROOT, PHASE_BUDGETS,
                   PHASE_BY_ID, PHASE_ORDER, PHASES, PHYSICS_DT, PPO_CONFIG,
                   PROTOCOL_ID, RECIPE_VERSION, RETENTION_INTERVAL, REWARD_TERMS,
                   REWARD_WEIGHTS, ROLE_COUNTS, ROLE_NAMES, ROOT, ROUTE_DIM,
                   ROUTE_ERROR_KEY, ROUTE_LATERAL_CLIP, ROUTE_LATERAL_SCALE,
                   ROUTE_STATE, SCHEMA_VERSION, SDS_REVISION, TASK,
                   TERRAIN_BY_LESSON, TERRAIN_BY_TILE, TERRAIN_TABLE,
                   TOTAL_BUDGET, T_TILE_COUNT, behavior_hashes, canonical_json,
                   clearance_target, contract, contract_sha256, identity_block,
                   phase_budget, phase_spec, product_entries, recipe_config,
                   reward_config, reward_config_sha256, sha256, terrain_spec,
                   tile_id_for, total_budget)

__all__ = [
    'CODE_VERSION', 'RECIPE_VERSION', 'SDS_REVISION', 'SCHEMA_VERSION', 'ARCHITECTURE',
    'TASK', 'PROTOCOL_ID', 'LOG_ROOT', 'ROOT', 'INIT_PATH', 'INIT_SHA256',
    'CONTROL_DT', 'PHYSICS_DT', 'DECIMATION', 'ACTION_SCALE', 'HISTORY_LEN',
    'HEIGHTMAP_SHAPE', 'ROUTE_DIM', 'CRITIC_WIDTH', 'ROUTE_ERROR_KEY',
    'ROUTE_LATERAL_SCALE', 'ROUTE_LATERAL_CLIP', 'ROUTE_STATE', 'FLAT_TILE_ID',
    'FINAL_TILE_ID', 'T_TILE_COUNT', 'TERRAIN_TABLE', 'TERRAIN_BY_TILE',
    'TERRAIN_BY_LESSON', 'terrain_spec', 'tile_id_for', 'PHASES', 'PHASE_BY_ID',
    'PHASE_ORDER', 'PHASE_BUDGETS', 'TOTAL_BUDGET', 'phase_spec', 'phase_budget',
    'total_budget', 'REWARD_TERMS', 'REWARD_WEIGHTS', 'clearance_target',
    'PPO_CONFIG', 'CURRICULUM', 'GATE_SPEEDS_F0_H_D_L', 'GATE_SPEEDS_V_R',
    'GATE_INTERVAL', 'GATE_CASES', 'GATE_PASS_MIN', 'RETENTION_INTERVAL',
    'ROLE_COUNTS', 'ROLE_NAMES', 'reward_config', 'reward_config_sha256',
    'recipe_config', 'behavior_hashes', 'product_entries', 'contract',
    'contract_sha256', 'identity_block', 'sha256', 'canonical_json',
    'V07ActorCritic',
]

_LAZY_MODEL_EXPORTS = {'V07ActorCritic': 'V07ActorCritic'}


def __getattr__(name):
    """Lazily expose the v0.6 network without importing torch for spec-only users."""
    target = _LAZY_MODEL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from . import model

    value = getattr(model, target)
    globals()[name] = value
    return value
