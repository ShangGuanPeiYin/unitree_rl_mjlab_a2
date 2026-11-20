"""Certificates for actual compiled original terrain and contact-map indexing."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import inspect
import numpy as np
import torch

from ..v07 import spec as original
from ..v07 import terrain
from . import spec


def geometry_certificate(env, *, code_sha256):
    """Re-run the original compiled-geometry validator and compare cached indices."""
    if terrain._PLAYBACK_PLATFORM is not None or terrain._AIR_PLATFORM is not None:
        raise ValueError('S1 requires original training terrain, playback overrides are active')
    table = [asdict(row) for row in original.TERRAIN_TABLE]
    if [asdict(row) for row in env.specs] != table:
        raise ValueError('environment terrain differs from the original 23-tile table')
    index = terrain.index_geometry(env.sim.mj_model, env.specs)
    for name, actual, expected in (
        ('tile', env.geom_tile, index.tile), ('step', env.geom_step, index.step),
        ('top', env.geom_top, index.top), ('support_bounds', env.geom_step_bounds, index.support_bounds),
    ):
        expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        if not torch.equal(actual, expected):
            raise ValueError(f'cached contact index differs from compiled geometry: {name}')
    model = env.sim.mj_model
    payload = dict(sds_sha256=spec.SDS_SHA256, code_sha256=code_sha256,
        validator='v07.terrain.index_geometry', validated=True,
        tolerance=dict(position_absolute=1e-6, size_atol=1e-9,
                       size_rtol=float(inspect.signature(np.allclose).parameters['rtol'].default),
                       top_absolute=1e-9),
        original_table=table, terrain_geom_count=len(index.geom_ids),
        compiled_geoms=[dict(id=int(identity), name=model.geom(int(identity)).name,
            position=model.geom_pos[identity].tolist(), size=model.geom_size[identity].tolist())
            for identity in index.geom_ids],
        index=dict(tile=index.tile.tolist(), step=index.step.tolist(), top=index.top.tolist(),
                   support_bounds=index.support_bounds.tolist()))
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    payload['certificate_sha256'] = hashlib.sha256(encoded).hexdigest()
    return payload
