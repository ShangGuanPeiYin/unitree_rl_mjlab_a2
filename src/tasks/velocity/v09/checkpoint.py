"""Atomic schema-11 snapshots, with identity validation before recovery."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from . import spec

IDENTITY = dict(schema=spec.CHECKPOINT_SCHEMA, code_version=spec.CODE_VERSION,
                sds_sha256=spec.SDS_SHA256, recipe=spec.RECIPE,
                source_sha256=spec.SOURCE_SHA256, map_shape=spec.MAP_SHAPE,
                architecture=spec.ARCHITECTURE)
REQUIRED_METADATA = ('stage', 'course', 'value_scale', 'manifest_sha256',
                     'ledger_run_id', 'learning_samples', 'gate_state',
                     'role_state', 'random_state', 'code_sha256')


def _validate_metadata(metadata):
    missing = set(REQUIRED_METADATA) - metadata.keys()
    if missing:
        raise ValueError(f'missing checkpoint metadata: {sorted(missing)}')
    if metadata['stage'] != 'S1' or metadata['course'] not in ('S1.0', 'S1.1', 'S1.2'):
        raise ValueError('only the implemented S1 checkpoint contract is enabled')
    scale, samples = metadata['value_scale'], metadata['learning_samples']
    if (isinstance(scale, bool) or not isinstance(scale, (float, int))
            or not math.isfinite(scale) or scale < 1):
        raise ValueError('invalid finite calibration value scale')
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError('learning_samples must be a nonnegative integer')
    for key in ('manifest_sha256', 'code_sha256'):
        digest = metadata[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'invalid {key}')
    if not isinstance(metadata['ledger_run_id'], str) or not metadata['ledger_run_id']:
        raise ValueError('ledger_run_id is required')
    required = {
        'gate_state': ('next_suite', 'attempts', 'final_suites_used', 'recoveries',
                       'next_gate_samples', 'passes'),
        'role_state': ('num_envs', 'episode_counters', 'assignments'),
        'random_state': ('python', 'numpy'),
    }
    for key, fields in required.items():
        state = metadata[key]
        if not isinstance(state, dict) or not set(fields) <= state.keys():
            raise ValueError(f'incomplete {key}')
    if metadata['role_state']['num_envs'] not in spec.ENV_CANDIDATES:
        raise ValueError('unsupported resource admission environment count')


def save_checkpoint(path, model, optimizer, metadata):
    """Replace the target only after a complete snapshot reaches durable storage.

    The external budget ledger is intentionally never restored from this file.
    """
    _validate_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(IDENTITY, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   metadata=metadata, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, *, device='cpu'):
    """Validate identity before the caller applies any model/optimizer state."""
    payload = torch.load(path, map_location=device, weights_only=False)
    for key, expected in IDENTITY.items():
        if payload.get(key) != expected:
            raise ValueError(f'checkpoint {key} mismatch')
    _validate_metadata(payload['metadata'])
    if 'model' not in payload or 'optimizer' not in payload:
        raise ValueError('checkpoint lacks model or optimizer state')
    return payload
