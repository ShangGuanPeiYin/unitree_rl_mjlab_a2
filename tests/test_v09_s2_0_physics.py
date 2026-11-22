"""CPU-only s2_physics interface tests with the fake backend (no MuJoCo)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_env, s2_physics


def case(task):
    return next(row for row in s2_cases.build_s2_0_manifest()['cases']
                if row['task'] == task)


def make_env(task='stairs_walk', branch='real', steps=40, h_legacy=False):
    return s2_physics.S2PhysicsEnv(
        case(task), backend_factory=s2_physics.S2FakeBackend,
        device='cpu', seed=42, map_branch=branch, h_legacy=h_legacy)


def run(env, steps=40):
    obs = env.reset()
    done = False
    for _ in range(steps):
        obs, done, info = env.step(np.zeros(12, dtype=np.float32))
        if done[0]:
            break
    return obs, done, info


def test_reset_step_observe_contract_shapes():
    env = make_env('stairs_walk')
    obs = env.reset()
    assert obs['proprio'].shape == (45,)
    assert obs['history'].shape == (5, 45)
    assert obs['heightmap'].shape == (3, 40, 30)
    assert obs['route_error'].shape == (3,)
    assert 'c_request' in obs and 'c_exec' in obs
    obs, done, info = env.step(np.zeros(12, dtype=np.float32))
    assert obs['proprio'].shape == (45,)
    assert env.diagnostics()['feet_xy'] is not None
    assert env.diagnostics()['contact_pairs']
    assert env.evidence(0)['map_evidence'] is not None


def test_plane_branch_trigger_and_s_minus_t():
    env = make_env('stairs_walk', branch='plane_diagnostic', steps=40)
    run(env, 40)
    report = env.evidence(0)['map_evidence']
    assert report['branch'] == 'plane_diagnostic'
    assert report['trigger_frame'] is not None
    assert report['first_consumed_replacement_frame'] is not None
    assert report['s_minus_t'] == (
        report['first_consumed_replacement_frame'] - report['trigger_frame'])
    assert report['valid_fraction'] is not None
    assert report['saturation'] is not None


def test_branch_prefix_is_identical_before_intervention():
    real = make_env('stairs_walk', branch='real', steps=40)
    plane = make_env('stairs_walk', branch='plane_diagnostic', steps=40)
    obs_r = real.reset()
    obs_p = plane.reset()
    actions = np.zeros(12, dtype=np.float32)
    for step in range(30):
        _, done_r, _ = real.step(actions)
        _, done_p, _ = plane.step(actions)
        if done_r[0] or done_p[0]:
            break
    real_trace = real.recorder.trace()
    plane_trace = plane.recorder.trace()
    prefix = plane.ablation.first_consumed_replacement_frame
    assert prefix is not None and prefix > 0
    digest = s2_env.assert_branches_identical_until(
        real_trace, plane_trace, end_index=prefix)
    assert len(digest) == 64


def test_h_legacy_uses_v07_success_and_original_timeout():
    record = s2_cases.build_h_series_case0_manifest(ROOT)['records'][0]['case']
    env = s2_physics.S2PhysicsEnv(
        record, backend_factory=s2_physics.S2FakeBackend, device='cpu',
        map_branch='real', h_legacy=True)
    obs = env.reset()
    for _ in range(20):
        obs, done, _ = env.step(np.zeros(12, dtype=np.float32))
        if done[0]:
            break
    assert env.machine.success is True
    assert env.machine.config.timeout_steps == 1301
    assert env.task_adapter.config.legacy_h is True


def test_build_s2_env_defaults_to_the_real_mjlab_backend_factory():
    # Static factory identity only; no scene/GPU/CPU physics is constructed here.
    assert s2_physics.MjlabS2Backend is not s2_physics.S2FakeBackend
    assert callable(s2_physics.build_s2_env)


def test_assembly_contract_matches_v07_v09_constants():
    checks = s2_physics.assembly_contract()
    assert checks['control_dt'][0] == 0.02
    assert checks['map_shape'][0] == (3, 40, 30)
    assert checks['map_refresh_steps'][0] == 5
    for name, values in checks.items():
        if name in ('control_dt', 'physics_dt', 'decimation', 'action_scale',
                    'joints', 'map_shape', 'map_grid_step', 'map_x', 'map_y',
                    'map_refresh_steps'):
            assert values[0] == values[1], name


def test_s1_actor_loads_deterministic_mean_cpu():
    import torch
    actor = s2_physics.load_s1_actor(device='cpu')
    assert actor['frozen'] is True and actor['requires_grad'] is False
    obs = dict(
        proprio=torch.zeros(2, 45), history=torch.zeros(2, 5, 45),
        heightmap=torch.zeros(2, 3, 40, 30),
        route_error=torch.zeros(2, 3))
    out = s2_physics.actor_mean(actor, obs)
    assert out.shape == (2, 12)
    assert out.requires_grad is False
    assert actor['normalizer_updates'] == 0


def test_s2_patch_context_restores_v07_functions():
    from src.tasks.velocity.v07 import terrain
    before = (terrain.build_terrain_specs, terrain.tile_boxes,
              terrain.query_surface)
    with s2_physics._s2_patch_context(s2_physics.s2_terrain.terrain_specs()):
        assert terrain.build_terrain_specs is not before[0]
    after = (terrain.build_terrain_specs, terrain.tile_boxes,
             terrain.query_surface)
    assert after == before


def test_h_legacy_backend_is_separate_and_has_original_timeout_contract():
    assert issubclass(s2_physics.MjlabHLegacyBackend,
                      s2_physics.MjlabS2Backend)
    record = s2_cases.build_h_series_case0_manifest(ROOT)['records'][0]['case']
    adapter = s2_env.S2TaskAdapter.from_case(record)
    assert adapter.config.legacy_h is True
    assert adapter.config.timeout_steps == 1301


def test_b_v7a_tile_mapping_is_legal_for_every_s2_case():
    manifest = s2_cases.build_s2_0_manifest()
    table = s2_physics.s2_terrain.terrain_specs()
    for case in manifest['cases']:
        spec = s2_physics.s2_terrain.spec_from_geometry(case['geometry'])
        assert 0 <= spec.tile_id < len(table)
        assert s2_physics.s2_terrain.schedule_tile_id(case['geometry']) == spec.tile_id
    keys = s2_physics.s2_terrain.schedule_tile_table()
    assert len(keys) == len(set(keys))
    assert len(set(keys.values())) == len(keys)


def test_b_v7a_first_step_precedes_top_and_native_sample_is_not_after_consumption():
    env = make_env('stairs_walk', branch='plane_diagnostic', steps=40)
    run(env, 30)
    names = [event['name'] for event in env.machine.events]
    assert 'enter_climb' in names
    assert names.index('enter_climb') < names.index('top_arrived')
    report = env.evidence(0)['map_evidence']
    sample = report['native_sample_id']
    assert sample is not None
    assert sample['produced_at_actor_step'] <= report['first_replacement_actor_step']
    assert report['s_minus_t'] >= 0
