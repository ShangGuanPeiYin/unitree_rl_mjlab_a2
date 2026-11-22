"""Schema-11 S2 checkpoint metadata contract (does not modify S1 checkpoint.py)."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile

from . import s2_spec
from . import spec as s1_spec

IDENTITY = dict(schema=s2_spec.CHECKPOINT_SCHEMA, stage='S2',
                code_version=s2_spec.CODE_VERSION, sds_sha256=s2_spec.SDS_SHA256,
                recipe=s2_spec.RECIPE, map_shape=list(s1_spec.MAP_SHAPE),
                source_sha256=s2_spec.S1_ENTRY_WEIGHT_SHA256,
                architecture='a2-v09-native40x30-route-history')
REQUIRED_METADATA = (
    'stage', 'course', 'curriculum', 'strength', 'model_state', 'optimizer_state',
    'policy_std', 'normalizer_state', 'teacher_source', 'beta_flat',
    'learning_samples', 'evaluation_samples', 'engineering_samples',
    'rollback_samples', 'gates_passed', 'case_manifest_sha256',
    'candidate_sha256', 'gate_ab_sequence', 'gate_uses', 'final_uses',
    'engineering_rerun_reasons', 'random_state', 'role_assignment',
    'randomization', 'command_flow', 'delay_config', 'entry_weight',
    'ledger_run_id', 'code_sha256', 'manifest_sha256')
COURSES = ('S2.1', 'S2.2', 'S2.3')


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in '0123456789abcdef' for character in value):
        raise ValueError(f'{name} must be a 64-char SHA256')
    return value


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be a mapping')
    missing = set(REQUIRED_METADATA) - set(metadata)
    if missing:
        raise ValueError(f'missing S2 checkpoint metadata: {sorted(missing)}')
    if metadata['stage'] != 'S2' or metadata['course'] not in COURSES:
        raise ValueError('S2 checkpoint requires stage S2 and course S2.1/2/3')
    for key in ('model_state', 'optimizer_state'):
        if metadata[key] is None or (hasattr(metadata[key], '__len__')
                                     and len(metadata[key]) == 0):
            raise ValueError(f'{key} must be nonempty')
    teacher = metadata['teacher_source']
    entry = metadata['entry_weight']
    if not isinstance(teacher, dict) or not isinstance(teacher.get('sha256'), str):
        raise ValueError('teacher_source must include a SHA256')
    _digest(teacher['sha256'], 'teacher_source.sha256')
    if not isinstance(entry, dict):
        raise ValueError('entry_weight must be a mapping')
    _digest(entry.get('sha256'), 'entry_weight.sha256')
    for key in ('case_manifest_sha256', 'candidate_sha256', 'code_sha256',
                'manifest_sha256'):
        _digest(metadata[key], key)
    if not isinstance(metadata['gates_passed'], list):
        raise ValueError('gates_passed must be a list')
    for entry in metadata['gates_passed']:
        for key in ('suite', 'case_sha256', 'candidate_sha256', 'ab_sequence',
                    'gate_uses'):
            if key not in entry:
                raise ValueError(f'gate entry missing {key}')
        if entry['ab_sequence'] not in (['gate_A', 'gate_B'],
                                        ['gate_B', 'gate_A'], 'gate_A', 'gate_B'):
            raise ValueError('gate ab_sequence must be a valid A/B sequence')
    for key in ('role_assignment', 'randomization', 'command_flow', 'delay_config',
                'random_state', 'normalizer_state'):
        if not isinstance(metadata[key], dict):
            raise ValueError(f'{key} must be a mapping')
    if not 0 <= float(metadata['beta_flat']) <= 1:
        raise ValueError('beta_flat must be in [0,1]')
    for key in ('learning_samples', 'evaluation_samples', 'engineering_samples',
                'rollback_samples'):
        if isinstance(metadata[key], bool) or not isinstance(metadata[key], int) \
                or metadata[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    return True


def load_checkpoint(path, *, device='cpu'):
    """Load schema-11 S2 state; reject foreign/legacy identities."""
    import torch
    payload = torch.load(path, map_location=device, weights_only=False)
    for key, expected in IDENTITY.items():
        if payload.get(key) != expected:
            raise ValueError(f'S2 checkpoint {key} mismatch')
    if payload.get('schema') != s2_spec.CHECKPOINT_SCHEMA:
        raise ValueError('S2 checkpoint schema mismatch')
    if payload.get('sds_sha256') != s2_spec.SDS_SHA256:
        raise ValueError('S2 checkpoint SDS mismatch')
    validate_metadata(payload.get('metadata', {}))
    if not isinstance(payload.get('model'), dict) or not payload['model']:
        raise ValueError('S2 checkpoint model state missing')
    if not isinstance(payload.get('optimizer'), dict) or not payload['optimizer']:
        raise ValueError('S2 checkpoint optimizer state missing')
    return payload


def save_checkpoint(path, model_state, optimizer_state, metadata):
    """Atomic schema-11 S2 snapshot; the external ledger is never restored."""
    validate_metadata(metadata)
    if not model_state or not optimizer_state:
        raise ValueError('model/optimizer state must be nonempty')
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(IDENTITY, model=model_state, optimizer=optimizer_state,
                   metadata=metadata)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return hashlib.sha256(path.read_bytes()).hexdigest()
