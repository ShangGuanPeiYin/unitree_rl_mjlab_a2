"""T12 S2.1 training stack CPU tests (fake env; no physics/ledger writes)."""
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
from src.tasks.velocity.v09 import (s2_budget, s2_cases, s2_checkpoint,
                                    s2_ledger, s2_loss, s2_runner, s2_spec,
                                    s2_train, s2_train_env)


def test_role_allocation_largest_remainder_fixed_order():
    assert s2_train_env.largest_remainder(1024) == {
        'flat_walk': 205, 'flat_stop': 205, 'stairs_walk': 410,
        'platform_walk': 102, 'l40_milestone': 102}
    assert sum(s2_train_env.largest_remainder(512).values()) == 512


def test_s2_1_curriculum_balance():
    plan = s2_train_env.build_role_plan(1024)
    assert s2_train_env.validate_role_plan(plan) is True
    stairs = [row for row in plan['envs'] if row['role'] == 'stairs_walk']
    assert len(stairs) == 410
    assert {row['stair_count'] for row in stairs} == {10, 20}
    assert {row['platform_width'] for row in stairs} == {0.8, 4.0}
    assert {row['speed'] for row in stairs} == {0.3, 0.5, 0.6}
    for levels in (10, 20):
        assert sum(row['stair_count'] == levels for row in stairs) in (204, 205)
    for width in (0.8, 4.0):
        assert abs(sum(row['platform_width'] == width for row in stairs) - 205) <= 1
    assert {row['post_distance'] for row in stairs} == {5.0}


def test_rollout_and_checkpoint_cadence():
    plan = s2_train.rollout_plan(1024)
    assert plan['samples_per_rollout'] == 49_152
    assert plan['max_rollouts'] == 16_276
    assert plan['checkpoint_every_rollouts'] == 102
    assert s2_spec.complete_rollouts(800_000_000, 1024) == 16_276


def test_kl_lr_std_and_loss_contracts():
    assert s2_runner.adjust_learning_rate(3e-5, 0.03) == pytest.approx(2e-5)
    assert s2_runner.adjust_learning_rate(3e-5, 0.001) == pytest.approx(4.5e-5)
    assert s2_runner.project_std(0.001) == 0.05
    assert s2_runner.project_std(9.0) == 1.0
    g = [np.ones(3), np.ones(3)]
    assert s2_runner.weighted_accumulate([(g[0], 1), (g[1], 3)]).tolist() == [1, 1, 1]
    acc = s2_loss.AnchorAccumulator()
    clean = np.array([True, False, True])
    acc.add(np.zeros((3, 12)), np.ones((3, 12)), clean)
    assert acc.n_clean == 2
    assert acc.value() == pytest.approx(1.0)


def test_gate_matrix_has_36_groups_and_ab_validation():
    plan = s2_train.gate_plan()
    assert plan['groups'] == 36
    assert plan['total_cases'] == 36 * 20 * 2
    groups = s2_train.gate_groups()
    results = [dict(group=group['id'], suite=suite, successes=18, total=20)
               for group in groups for suite in ('gate_A', 'gate_B')]
    assert s2_train.evaluate_gate_results(results)['passed'] is True
    results[0]['successes'] = 17
    assert s2_train.evaluate_gate_results(results)['passed'] is False


def _ledger_with_learning(tmp_path):
    cases = ROOT / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', ROOT)
    h = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', ROOT)
    plan = s2_budget.build_reservation_plan(matrix, h)
    path = tmp_path / 'ledger.jsonl'
    ledger = s2_ledger.S2Ledger(path, run_id='r')
    ledger.reserve_stage(plan)
    return ledger


def test_learning_budget_reserve_commit(tmp_path):
    ledger = _ledger_with_learning(tmp_path)
    budget = s2_train.S2LearningBudget(ledger)
    budget.begin('t12')
    budget.finish('t12', s2_train.S2_1_SAMPLES_PER_ROLLOUT)
    state = ledger.state()
    assert state['learning_obligations']['learning_S2_1']['spent'] == 49_152
    assert state['learning_obligations']['learning_S2_1']['reserved'] == 0


def test_checkpoint_roundtrip_and_resume_keeps_lr_std(tmp_path):
    from src.tasks.velocity.v09.model import V09ActorCritic
    env = s2_train_env.S2FakeTrainEnv(num_envs=2,
                                      role_plan=s2_train_env.build_role_plan(2))
    model = V09ActorCritic()
    trainer = s2_train.S2Trainer(model=model, env=env, device='cpu',
                                 learning_rate=3e-5, attempt_id='test')
    trainer.model.std.data.fill_(0.31)
    trainer.learning_rate = 4.4e-5
    for group in trainer.optimizer.param_groups:
        group['lr'] = 4.4e-5
    path = tmp_path / 'cp.pt'
    trainer.save(path)
    fresh_env = s2_train_env.S2FakeTrainEnv(num_envs=2,
                                            role_plan=s2_train_env.build_role_plan(2))
    fresh = s2_train.S2Trainer(model=V09ActorCritic(), env=fresh_env, device='cpu',
                               learning_rate=3e-5, attempt_id='test')
    fresh.resume(path, device='cpu')
    assert fresh.learning_rate == pytest.approx(4.4e-5)
    assert float(fresh.model.std.mean().item()) == pytest.approx(0.31)
    assert s2_checkpoint.load_checkpoint(path, device='cpu')['metadata']['course'] == 'S2.1'


def test_probe_plan_and_512_only_after_1024_failure():
    from src.tasks.velocity.v09 import s2_resources
    plan = s2_resources.probe_plan(1024)
    assert plan['warmups'] == 5 and plan['measurements'] == 50
    assert plan['restore_checks'] == 1
    failed = s2_resources.ProbeResult(
        num_envs=1024, peak_mib=8000, host_free_mib=8192, oom=False,
        nonfinite=False, contact_overflow=False, warmups=5, measurements=50,
        restore_checks=1, independent_copy=True, evaluation_envs=100)
    fallback = s2_resources.ProbeResult(
        num_envs=512, peak_mib=6000, host_free_mib=8192, oom=False,
        nonfinite=False, contact_overflow=False, warmups=5, measurements=50,
        restore_checks=1, independent_copy=True, evaluation_envs=100)
    assert failed.admitted() is False
    chosen = s2_resources.choose_resource([failed, fallback])
    assert chosen['selected'] == 512


def test_one_fake_rollout_updates_ppo_and_std():
    from src.tasks.velocity.v09.model import V09ActorCritic
    env = s2_train_env.S2FakeTrainEnv(num_envs=2,
                                      role_plan=s2_train_env.build_role_plan(2))
    trainer = s2_train.S2Trainer(model=V09ActorCritic(), env=env, device='cpu',
                                 learning_rate=3e-5, attempt_id='test')
    result = trainer.train_rollout()
    assert trainer.learning_samples == 96
    assert result['rollouts'] == 1
    assert 0.05 <= min(result['std']) and max(result['std']) <= 1.0
    assert 1e-5 <= result['learning_rate'] <= 1e-4


def test_real_env_plan_maps_roles_to_s2_terrain():
    from src.tasks.velocity.v09 import s2_train_real
    plan = s2_train_real.build_env_plan(1024)
    assert len(plan['rows']) == 1024
    assert set(plan['role_counts']) == set(s2_train_env.ROLE_ORDER)
    for row in plan['rows']:
        assert row['geometry']['tile_id'] == row['tile_id']
        assert row['geometry']['kind'] in ('flat', 's2_stairs')
        assert row['spawn'][3] in s2_train_env.SPEEDS
        if row['role'] in ('stairs_walk', 'platform_walk'):
            assert row['geometry']['platform_length'] == 20.0
            assert row['geometry']['platform_width'] in (0.8, 4.0)
    assert plan['observation']['heightmap'] == (3, 40, 30)
    assert plan['observation']['privileged'] == (35,)
    assert plan['observation']['critic'] == 211
    assert plan['map_refresh_steps'] == 5


def test_train_execute_refuses_without_probe_admission(tmp_path):
    import train_a2_v09_s2 as train_cli
    with pytest.raises(SystemExit):
        train_cli.main(['--execute', '--num-envs', '1024',
                        '--fake-backend', '--ledger', str(tmp_path / 'missing.jsonl'),
                        '--probe-report', str(tmp_path / 'missing_probe.json'),
                        '--admission-evidence', str(tmp_path / 'missing_admit.json')])


def test_probe_plan_cli_is_cpu_only(tmp_path):
    import probe_a2_v09_s2_resources as probe_cli
    assert probe_cli.main(['--plan-only', '--num-envs', '1024',
                           '--output-dir', str(tmp_path)]) == 0
    doc = json.loads((tmp_path / 'probe_plan.json').read_text())
    assert doc['warmups'] == 5
    assert doc['measurements'] == 50
    assert doc['restore_checks'] == 1
    assert doc['control_samples'] == 3_709_184
