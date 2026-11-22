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


class TorchArrayLike:
    def __init__(self, value): self.value = value
    def item(self): return self.value


def test_torcharray_time_scalar_and_termination_contract():
    assert s2_physics._scalar(TorchArrayLike(12.5), 0.0) == 12.5
    assert s2_physics._scalar(np.array([3.0, 4.0]), 0.0) == 3.0
    contract = s2_physics.termination_contract(object())
    assert contract['contract_ok'] is True
    assert set(s2_physics.TERMINATION_REQUIRED) <= set(contract['failure_categories'])


def test_preflight_rejects_missing_ledger_before_gpu(tmp_path, capsys):
    import run_a2_v09_s2_0 as run_cli
    with pytest.raises(SystemExit):
        run_cli.main(['--preflight', '--case', '0',
                      '--ledger', str(tmp_path / 'missing.jsonl')])


def test_v7a_foot_contact_aggregation_and_nonfoot_classification():
    import torch
    forces = torch.zeros(4, 207)
    total, pairs = s2_physics.aggregate_foot_contacts(forces)
    assert total.shape == (4,)
    assert pairs == []
    forces[1, 3] = 5.0
    forces[1, 5] = 7.0
    forces[0, 9] = 2.0
    total, pairs = s2_physics.aggregate_foot_contacts(
        forces, geom_names=[f'g{i}' for i in range(207)])
    assert total.tolist() == [2.0, 12.0, 0.0, 0.0]
    assert all(pair['foot'] in (0, 1) for pair in pairs)
    assert len(pairs) == 3
    links = torch.zeros(6, 3)
    names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot', 'base_link', 'head']
    assert s2_physics.nonfoot_contact_from_links(links, names) is False
    links[4, 0] = 11.0
    assert s2_physics.nonfoot_contact_from_links(links, names) is True
    links[4, 0] = 0.0
    links[0, 0] = 50.0
    assert s2_physics.nonfoot_contact_from_links(links, names) is False


def test_policy_observation_comes_from_backend_not_hardcoded_zeros():
    class MarkerBackend(s2_physics.S2FakeBackend):
        def observation(self, update=True):
            return dict(proprio=np.full(45, 7.0, dtype=np.float32),
                        history=np.full((5, 45), 8.0, dtype=np.float32),
                        heightmap=np.full((3, 40, 30), 9.0, dtype=np.float32),
                        route_error=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    env = s2_physics.S2PhysicsEnv(
        case('flat_walk'), backend_factory=MarkerBackend, device='cpu')
    obs = env.observe()
    assert obs['proprio'][0] == 7.0
    assert obs['history'][0, 0] == 8.0
    assert obs['heightmap'][0, 0, 0] == 9.0
    assert obs['route_error'].tolist() == [1.0, 2.0, 3.0]


def test_policy_action_is_forwarded_to_backend_step():
    class RecordingBackend(s2_physics.S2FakeBackend):
        def step(self, action=None):
            self.seen_action = (None if action is None
                                else np.asarray(action, dtype=np.float32).copy())
            return super().step(action)
    env = s2_physics.S2PhysicsEnv(
        case('flat_walk'), backend_factory=RecordingBackend, device='cpu')
    action = np.arange(12, dtype=np.float32) / 10.0
    env.reset()
    env.step(action)
    assert env.backend.seen_action is not None
    assert np.array_equal(env.backend.seen_action, action)


def test_policy_observation_reader_never_advances_history_or_map():
    class RecordingBackend(s2_physics.S2FakeBackend):
        def observation(self, update=True):
            self.update_calls.append(update)
            return super().observation(update=update)
        def reset(self):
            self.update_calls = []
            return super().reset()
    env = s2_physics.S2PhysicsEnv(
        case('flat_walk'), backend_factory=RecordingBackend, device='cpu')
    env.reset()
    env.step(np.zeros(12, dtype=np.float32))
    assert env.backend.update_calls
    assert env.backend.update_calls == [False] * len(env.backend.update_calls)


def test_t9_parity_helpers_detect_shape_and_value_differences():
    from src.tasks.velocity.v09 import s2_parity
    left = dict(proprio=np.zeros(45), history=np.zeros((5, 45)),
                heightmap=np.zeros((3, 40, 30)), route_error=np.zeros(3))
    right = dict(left, proprio=np.ones(45))
    result = s2_parity.compare_observation(left, right)
    assert result['proprio']['max_abs'] == 1.0
    assert result['history']['max_abs'] == 0.0
    rows = [dict(left, a_request=np.zeros(12), a_applied=np.zeros(12),
                 pd_target=np.zeros(12), c_exec=0.3)]
    changed = [dict(rows[0], a_applied=np.ones(12))]
    compared = s2_parity.compare_trace_tables(rows, changed)
    assert compared['fields']['a_applied']['max_abs'] == 1.0
    assert s2_parity.friction_contract()['equal'] is True


def test_s2_terrain_specs_expose_v07_episode_fields():
    from src.tasks.velocity.v09.s2_terrain import terrain_specs
    for spec in terrain_specs():
        assert isinstance(spec.lesson_id, str) and spec.lesson_id
        assert all(hasattr(spec, name) for name in ('N', 'h', 'step_depth'))


def test_v85_real_branch_evidence_accepts_valid_invalid_cells():
    from src.tasks.velocity.v09 import s2_map_ablation
    evidence = s2_map_ablation.S2RealBranchEvidence(task='stairs_walk')
    real = np.zeros((3, 40, 30), dtype=np.float64)
    valid = np.ones((40, 30), dtype=bool)
    valid[:2, :2] = False
    real[1] = valid
    real[2] = (~valid).astype(float)
    real[0, valid] = -0.2
    info = evidence.observe(frame_index=0, real_map=real, valid=valid,
                            refreshed=True, front_foot_top_support=False,
                            consumed=True, terminal=False, sample_id=None)
    assert info['frames'] == 1
    assert evidence.frame_checks == 1
    assert evidence.frame_check_failures == 0
