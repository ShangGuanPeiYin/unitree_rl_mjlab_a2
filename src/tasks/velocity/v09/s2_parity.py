"""T9 observation/action parity comparison helpers (CPU; no simulator)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', 'route_error')
ACTION_KEYS = ('a_request', 'a_applied', 'pd_target', 'c_exec')
CONTACT_FRICTION = (0.6, 0.005, 0.0001)
V07_CONTACT_FRICTION = (0.6, 0.005, 0.0001)


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def compare_observation(left, right):
    result = {}
    for key in OBSERVATION_KEYS:
        if key not in left or key not in right:
            result[key] = dict(present=False, shape_ok=False, max_abs=None)
            continue
        a, b = _array(left[key]), _array(right[key])
        shape_ok = a.shape == b.shape
        result[key] = dict(present=True, shape_ok=shape_ok,
                           shape_left=list(a.shape), shape_right=list(b.shape),
                           max_abs=(float(np.max(np.abs(a - b))) if shape_ok else None))
    return result


def compare_trace_tables(left_rows, right_rows, *, fields=OBSERVATION_KEYS + ACTION_KEYS):
    if len(left_rows) != len(right_rows):
        return dict(frame_count_match=False, left=len(left_rows),
                    right=len(right_rows), fields={})
    result = {}
    for field in fields:
        values = []
        for left, right in zip(left_rows, right_rows):
            a, b = _array(left[field]), _array(right[field])
            values.append(float(np.max(np.abs(a - b))) if a.shape == b.shape else None)
        finite = [v for v in values if v is not None]
        result[field] = dict(frames=len(values),
                             max_abs=max(finite) if finite else None,
                             missing=sum(v is None for v in values))
    return dict(frame_count_match=True, left=len(left_rows),
                right=len(right_rows), fields=result)


def load_trace(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def friction_contract():
    return dict(s2=list(CONTACT_FRICTION), v07=list(V07_CONTACT_FRICTION),
                equal=tuple(CONTACT_FRICTION) == tuple(V07_CONTACT_FRICTION),
                source='mjlab geom friction in S2AtlasCfg / V07AtlasCfg')
