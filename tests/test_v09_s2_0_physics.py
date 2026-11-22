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
