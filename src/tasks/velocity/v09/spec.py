"""Identity and S1 constants frozen by SDS v1.1.

Later-stage configuration is enabled only after the required behavior evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SDS_PATH = PROJECT_ROOT / 'doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md'
SDS_SHA256 = '500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976'
SOURCE_PATH = PROJECT_ROOT / 'logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt'
SOURCE_SHA256 = '30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2'
CODE_VERSION = '0.9.0'
RECIPE = 'v09-native40x30-sequential-r2'
CHECKPOINT_SCHEMA = 11
MAP_SHAPE = (3, 40, 30)
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
ROLLOUT_STEPS = 48
EPOCHS = 5
MINIBATCHES = 4
MICROBATCH_MAX = 512
LEARNING_RATE = 3e-5
MAX_GRAD_NORM = 1.0
SEED = 42
SPEEDS = (0.3, 0.5, 0.6)
ENV_CANDIDATES = (2048, 1536, 1024, 512)
S1_CAP = 1_000_000_000
S1_LEARNING_CAP = 850_000_000
S1_ENGINEERING_CAP = 150_000_000
S1_BASELINE_CAP = 50_000_000
S1_ADAPT_CAP = 20_000_000
S1_RECOVERY_CAP = 630_000_000
S1_FINAL_RETRY_CAP = 100_000_000
CHECKPOINT_INTERVAL = 5_000_000


def validate_sources():
    """Fail before running when the specification or source checkpoint differs."""
    result = {}
    for label, path, expected in (
        ('sds', SDS_PATH, SDS_SHA256),
        ('source', SOURCE_PATH, SOURCE_SHA256),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'{label} SHA256 mismatch: {actual}')
        result[label] = {'path': str(path), 'sha256': actual}
    return result


def effective_cap(cap, num_envs):
    """Only complete N x 48 rollouts fit within a learning allocation."""
    if num_envs not in ENV_CANDIDATES or cap < 0:
        raise ValueError('unsupported environment count or negative cap')
    samples_per_iteration = num_envs * ROLLOUT_STEPS
    return cap // samples_per_iteration * samples_per_iteration

HEIGHTMAP_SHAPE = MAP_SHAPE
SOURCE_CHECKPOINT = SOURCE_PATH
ARCHITECTURE = 'a2-v09-native40x30-route-history'
RECIPE_VERSION = RECIPE


def code_snapshot():
    """Bind execution evidence to project sources, including inherited physics."""
    roots = (PROJECT_ROOT / 'src', PROJECT_ROOT / 'scripts')
    suffixes = {'.py', '.xml', '.urdf', '.usd', '.toml', '.yaml', '.yml'}
    paths = sorted(path for root in roots for path in root.rglob('*')
                   if path.is_file() and path.suffix in suffixes
                   and '__pycache__' not in path.parts)
    paths.extend((SDS_PATH, RULINGS_PATH))
    paths.sort()
    files = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    import json
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'code_sha256': hashlib.sha256(encoded).hexdigest(), 'files': files}

# Planning Q002 fixes DECEL/HOLD indexing while preserving the frozen timeouts.
FLAT_RAMP_TIMING_DECISION = {
    'first_zero_actor_index': 499,
    'hold_actor_index': 500,
    'completion_steps': 1500,
    'hard_timeout_steps': 1501,
    'standing_window_actor_index': 1000,
}
RULINGS_PATH = PROJECT_ROOT / 'doc/sds/reviews/v09_execution_rulings.md'
