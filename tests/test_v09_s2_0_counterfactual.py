"""Q031 offline same-frame counterfactual and input capture (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_counterfactual, s2_physics
from src.tasks.velocity.v09 import s2_cases


def rows(n=8):
    out = []
    for step in range(n):
        height = np.zeros((3, 40, 30), dtype=np.float32)
        height[1] = 1.0
        out.append(dict(
            step=step,
            proprio=np.full(45, 0.01 * step, dtype=np.float32),
            history=np.full((5, 45), 0.01 * step, dtype=np.float32),
            heightmap=height,
            route_error=np.zeros(3, dtype=np.float32),
            c_exec=np.float32(0.3),
            c_request=np.float32(0.3),
            recorded_action=np.zeros(12, dtype=np.float32),
            a_request=np.zeros(12, dtype=np.float32),
            refresh_index=step // 2,
            sample_sim_time_s=0.02 * step,
            produced_at_actor_step=step,
            base_sample_z=0.0,
            plane_world_z=1.0,
            refreshed=(step % 2 == 1),
            front_foot_top_support=(step >= 3),
        ))
    return out


def policy(obs):
    height = obs['heightmap'][:, 0, 0, 0]
    proprio = obs['proprio'][:, 0]
    return torch.stack([(proprio + height)] * 12, dim=1)


def test_same_frame_plane_derivation_and_stages():
    report = s2_counterfactual.analyze(
        rows(), policy=policy, task='stairs_walk',
        case_sha256='a' * 64, weight_sha256='b' * 64)
    assert report['rule']['trigger_frame'] == 3
    assert report['rule']['activation_frame'] == 5
    assert report['rule']['before_exact_equal'] is True
    assert report['rule']['after_valid_age_equal'] is True
    assert report['rule']['after_invalid_height_equal'] is True
    assert report['rule']['after_replacement_value_checked'] is True
    assert report['stages']['pre_trigger']['n'] == 5
    assert report['stages']['activation_refresh']['n'] == 1
    assert report['stages']['post_activation']['n'] == 2
    assert report['stages']['pre_trigger']['action_max_abs'] == 0.0
    assert report['stages']['post_activation']['action_max_abs'] > 0.0
    assert report['stages']['post_activation']['a_a_noise_max_abs'] == 0.0
    assert report['stages']['post_activation']['b_b_noise_max_abs'] == 0.0
    assert report['stages']['post_activation']['difference_significant'] is True
    assert report['evidence_grade'] == 'pending'
    assert report['physical_trajectory'] == 'not_proven'
    assert report['clip_position'] == 'not_clipped'


def test_capture_rejects_wrong_shape():
    bad = rows()
    bad[2]['heightmap'] = np.zeros((3, 39, 30), dtype=np.float32)
    with pytest.raises(s2_counterfactual.InputContractError):
        s2_counterfactual.derive_plane_rows(bad, task='stairs_walk')


def test_input_npz_roundtrip(tmp_path):
    payload = rows()
    npz = tmp_path / 'inputs.npz'
    np.savez_compressed(
        npz,
        proprio=np.asarray([r['proprio'] for r in payload]),
        history=np.asarray([r['history'] for r in payload]),
        heightmap=np.asarray([r['heightmap'] for r in payload]),
        route_error=np.asarray([r['route_error'] for r in payload]),
        c_exec=np.asarray([r['c_exec'] for r in payload]),
        c_request=np.asarray([r['c_request'] for r in payload]),
        recorded_action=np.asarray([r['recorded_action'] for r in payload]),
        a_request=np.asarray([r['a_request'] for r in payload]),
        step=np.asarray([r['step'] for r in payload]),
        refresh_index=np.asarray([r['refresh_index'] for r in payload]),
        sample_sim_time_s=np.asarray([r['sample_sim_time_s'] for r in payload]),
        produced_at_actor_step=np.asarray([r['produced_at_actor_step'] for r in payload]),
        base_sample_z=np.asarray([r['base_sample_z'] for r in payload]),
        plane_world_z=np.asarray([r['plane_world_z'] for r in payload]),
        refreshed=np.asarray([r['refreshed'] for r in payload]),
        front_foot_top_support=np.asarray([r['front_foot_top_support'] for r in payload]),
    )
    meta = tmp_path / 'inputs_metadata.json'
    meta.write_text(json.dumps({'task': 'stairs_walk', 'case_sha256': 'a' * 64,
                                'weight_sha256': 'b' * 64}))
    loaded, metadata = s2_counterfactual.load_saved_inputs(npz, meta)
    assert len(loaded) == len(payload)
    assert metadata['task'] == 'stairs_walk'
    assert loaded[0]['proprio'].shape == (45,)


def test_s2_physics_captures_full_input_when_requested():
    case = next(row for row in s2_cases.build_s2_0_manifest()['cases']
                if row['task'] == 'flat_walk')
    env = s2_physics.S2PhysicsEnv(
        case, backend_factory=s2_physics.S2FakeBackend, device='cpu',
        capture_inputs=True)
    env.reset()
    env.step(np.zeros(12, dtype=np.float32))
    assert len(env.input_rows) == 1
    row = env.input_rows[0]
    assert row['proprio'].shape == (45,)
    assert row['history'].shape == (5, 45)
    assert row['heightmap'].shape == (3, 40, 30)
    assert row['route_error'].shape == (3,)
    assert len(row['recorded_action']) == 12


def test_actor_identity_binds_normalizer_and_inference():
    class Norm:
        def state_dict(self):
            return {'mean': torch.zeros(45), 'std': torch.ones(45),
                    'count': torch.tensor(0)}
    class Model:
        def __init__(self):
            self.actor = type('Actor', (), {'normalizer': Norm()})()
    identity = s2_counterfactual.actor_identity(
        {'model': Model(), 'state_sha256': 'a' * 64})
    assert identity['weight_sha256'] == 'a' * 64
    assert identity['normalizer_version'] == 'frozen_ETH_actor_normalizer'
    assert len(identity['normalizer_state_sha256']) == 64
    assert identity['inference_mode'] == 'eval/no_grad/deterministic_mean'
    assert identity['action_scale'] == 0.25
    assert identity['clip_position'] == 'not_clipped'
