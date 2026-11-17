"""Independent A2 ideal-heightmap stairs 0.5-r2 protocol (schema 7,
``v05-route-state-v1``).

The package exports the frozen r2 identity, stage table, mode budget, five-key
observation contract and the route-state contract only.  It never imports the
legacy ``src/tasks`` root initializer, never scans a task directory and never
registers a task or a gym environment, so importing this package cannot change
the behavior hash of another version (SDS 1.3 items 7 and 8).

``V05R2ActorCritic`` is exported lazily on first attribute access so importing
``v05_r2.spec`` (for example from ``speed.py``) never drags in ``torch``.
"""
from .spec import (ACTOR_OBSERVATION_KEYS, ARCHITECTURE, CODE_VERSION, CONTROL_DT,
                   CRITIC_INPUT_DIMS, CRITIC_OBSERVATION_KEYS, FLAT_PROTOCOL_ID,
                   INIT_PATH, INIT_SHA256, LOG_ROOT, MODE_BUDGETS, OBSERVATION,
                   OBSERVATION_KEYS, PHYSICS_DT, PROTOCOL_ID, RECIPE_VERSION, ROOT,
                   ROUTE_ERROR_DIM, ROUTE_ERROR_KEY, ROUTE_LATERAL_CLIP,
                   ROUTE_LATERAL_SCALE, ROUTE_STATE, ROUTE_YAW_AXIS, S5_RAMP_UPDATES,
                   SCHEMA_VERSION, SDS_REVISION, SLOT_CLASSES, SPAWN_FLAT, STAGES,
                   STAGE_BY_ID, STAGE_BY_TILE,
                   STAGE_ORDER, TASK, TOTAL_BUDGET, behavior_hashes, contract,
                   contract_sha256, identity_block, mode_budget, observation_contract,
                   product_entries, recipe_config, reward_config, reward_config_sha256,
                   route_state, slot_counts, slot_layout, slot_tiles, stage_budget)

__all__ = [
    'CODE_VERSION', 'RECIPE_VERSION', 'SDS_REVISION', 'SCHEMA_VERSION', 'ARCHITECTURE',
    'TASK', 'PROTOCOL_ID', 'FLAT_PROTOCOL_ID', 'LOG_ROOT', 'ROOT', 'INIT_PATH',
    'INIT_SHA256', 'CONTROL_DT', 'PHYSICS_DT', 'STAGES', 'STAGE_BY_ID', 'STAGE_BY_TILE',
    'STAGE_ORDER', 'TOTAL_BUDGET', 'SLOT_CLASSES', 'slot_counts', 'slot_layout',
    'slot_tiles', 'reward_config', 'reward_config_sha256', 'recipe_config',
    'behavior_hashes', 'product_entries', 'contract', 'contract_sha256',
    'MODE_BUDGETS', 'mode_budget', 'stage_budget', 'identity_block',
    'OBSERVATION', 'OBSERVATION_KEYS', 'ACTOR_OBSERVATION_KEYS', 'CRITIC_OBSERVATION_KEYS',
    'observation_contract', 'ROUTE_ERROR_KEY', 'ROUTE_ERROR_DIM', 'ROUTE_LATERAL_SCALE',
    'ROUTE_LATERAL_CLIP', 'ROUTE_YAW_AXIS', 'ROUTE_STATE', 'route_state',
    'CRITIC_INPUT_DIMS', 'SPAWN_FLAT', 'S5_RAMP_UPDATES', 'V05R2ActorCritic',
]

_LAZY_MODEL_EXPORTS = {'V05R2ActorCritic': 'V05ActorCritic'}


def __getattr__(name):
    """Lazily expose the r2 network without importing torch for spec-only users."""
    target = _LAZY_MODEL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from . import model

    value = getattr(model, target)
    globals()[name] = value
    return value
