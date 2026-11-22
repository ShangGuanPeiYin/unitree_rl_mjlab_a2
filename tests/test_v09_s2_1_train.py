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
    plan512 = s2_train.rollout_plan(512)
    assert plan512['samples_per_rollout'] == 24_576
    assert plan512['max_rollouts'] == 32_552
    assert plan512['checkpoint_every_rollouts'] == 204
    assert s2_spec.complete_rollouts(800_000_000, 512) == 32_552
    assert 32_552 * 24_576 <= 800_000_000
    assert 16_276 * 49_152 <= 800_000_000


def test_512_learning_budget_uses_512_rollout_size():
    class Ledger:
        def __init__(self):
            self.calls = []

        def reserve(self, obligation_id, samples, *, operation_id, reason):
            self.calls.append(('reserve', samples, operation_id))

        def commit(self, obligation_id, samples, *, operation_id, reason):
            self.calls.append(('commit', samples, operation_id))

        def release(self, obligation_id, samples, *, operation_id, reason):
            self.calls.append(('release', samples, operation_id))

    ledger = Ledger()
    budget = s2_train.S2LearningBudget(
        ledger, samples_per_rollout=s2_train.rollout_plan(512)['samples_per_rollout'])
    budget.begin('s2.1-r1')
    budget.finish('s2.1-r1', 512 * 48)
    assert ledger.calls[0] == ('reserve', 24_576, 's2.1-r1:rollout0:reserve')
    assert ledger.calls[1] == ('commit', 24_576, 's2.1-r1:rollout0:commit')
    assert not [row for row in ledger.calls if row[0] == 'release']


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


def test_learning_budget_releases_on_rollout_exception():
    class Ledger:
        def __init__(self):
            self.calls = []

        def reserve(self, obligation_id, samples, *, operation_id, reason):
            self.calls.append(('reserve', samples, operation_id))

        def commit(self, obligation_id, samples, *, operation_id, reason):
            self.calls.append(('commit', samples, operation_id))

        def release(self, obligation_id, samples, *, operation_id, reason):
            self.calls.append(('release', samples, operation_id))

    class CrashingTrainer(s2_train.S2Trainer):
        def __init__(self):
            self.attempt_id = 'crash'
            self.rollouts = 0

        def collect_rollout(self):
            raise RuntimeError('injected rollout failure')

    ledger = Ledger()
    budget = s2_train.S2LearningBudget(ledger, samples_per_rollout=24_576)
    with pytest.raises(RuntimeError, match='injected'):
        CrashingTrainer().train_rollout(budget=budget)
    assert ledger.calls == [
        ('reserve', 24_576, 'crash:rollout0:reserve'),
        ('release', 24_576, 'crash:rollout0:release'),
    ]


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
        if row['role'] == 'platform_walk':
            assert row['spawn_z'] == row['geometry']['platform_height'] + 0.4
    assert plan['observation']['heightmap'] == (3, 40, 30)
    assert plan['observation']['privileged'] == (35,)
    assert plan['observation']['critic'] == 211
    assert plan['map_refresh_steps'] == 5


def test_512_dry_run_uses_512_rollout_plan(tmp_path):
    import train_a2_v09_s2 as train_cli
    evidence = tmp_path / 'fail1024.json'
    evidence.write_text(json.dumps({'num_envs': 1024, 'admitted': False,
                                    'failure_reasons': ['oom']}))
    out = tmp_path / 'dry'
    assert train_cli.main(['--dry-run', '--num-envs', '512',
                           '--fallback-evidence', str(evidence),
                           '--output-dir', str(out)]) == 0
    doc = json.loads((out / 'train_dry_run_summary.json').read_text())
    assert doc['plan']['training_plan']['samples_per_rollout'] == 24_576
    assert doc['plan']['training_plan']['max_rollouts'] == 32_552
    assert doc['plan']['training_plan']['checkpoint_every_rollouts'] == 204


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


def _admitted_probe_report(*, num_envs=512, **overrides):
    report = dict(
        num_envs=num_envs, admitted=True, peak_mib=6000, host_free_mib=8192,
        oom=False, nonfinite=False, contact_overflow=False,
        warmups=5, measurements=50, timings_s=[5.0] * 50,
        restore_checks=1, restore_ok=True, independent_copy=True,
        independent_copy_requested=True,
        entry_source_unchanged=True,
        entry_source_sha256_before='a' * 64,
        entry_source_sha256_after='a' * 64,
        entry_probe_copy_sha256='a' * 64,
        entry_actor_state_sha256_before='b' * 64,
        entry_actor_state_sha256_after_reload='b' * 64,
        process=dict(pid=123, ppid=1, independent_process=True),
        evaluation_envs=100, evaluation_steps=9472,
        video_envs=1, video_steps=200, failure_reasons=[])
    report.update(overrides)
    return report


def test_admission_binds_probe_sha_and_reviews_all_requirements(tmp_path):
    from src.tasks.velocity.v09 import s2_resources
    probe = tmp_path / 'probe.json'
    report = _admitted_probe_report(num_envs=1024)
    probe.write_text(json.dumps(report))
    admission = s2_resources.write_admission(
        tmp_path / 'admission.json', probe_report=report, probe_path=probe,
        num_envs=1024, independent_review_sha256='a' * 64)
    assert s2_resources.validate_admission(
        admission, probe_path=probe, num_envs=1024) is True
    probe.write_text(probe.read_text() + '\n')
    with pytest.raises(ValueError, match='probe SHA'):
        s2_resources.validate_admission(admission, probe_path=probe,
                                        num_envs=1024)


def test_admission_rejects_independent_copy_false(tmp_path):
    from src.tasks.velocity.v09 import s2_resources
    probe = tmp_path / 'probe.json'
    report = _admitted_probe_report(independent_copy=False,
                                    failure_reasons=[])
    probe.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='independent_copy'):
        s2_resources.write_admission(
            tmp_path / 'admission.json', probe_report=report, probe_path=probe,
            num_envs=512, independent_review_sha256='a' * 64)


def test_admission_rejects_tampered_entry_or_probe_requirements(tmp_path):
    from src.tasks.velocity.v09 import s2_resources
    probe = tmp_path / 'probe.json'
    report = _admitted_probe_report(entry_source_sha256_after='c' * 64)
    probe.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='entry_source_sha256'):
        s2_resources.write_admission(
            tmp_path / 'admission.json', probe_report=report, probe_path=probe,
            num_envs=512, independent_review_sha256='a' * 64)
    report = _admitted_probe_report(evaluation_steps=9471)
    probe.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='evaluation_steps'):
        s2_resources.write_admission(
            tmp_path / 'admission2.json', probe_report=report, probe_path=probe,
            num_envs=512, independent_review_sha256='a' * 64)


def test_spawn_z_is_applied_on_both_resets_and_training_plan():
    from types import SimpleNamespace
    from src.tasks.velocity.v09 import s2_physics, s2_train_real
    data = SimpleNamespace(default_root_state=torch.zeros(2, 13))
    env = SimpleNamespace(robot=SimpleNamespace(data=data))
    s2_physics.apply_default_spawn_z(env, 8.4)
    assert torch.all(data.default_root_state[:, 2] == 8.4)
    s2_physics.apply_default_spawn_z(env, 8.4)
    assert torch.all(data.default_root_state[:, 2] == 8.4)
    plan = s2_train_real.build_env_plan(1024)
    for row in plan['rows']:
        if row['role'] == 'platform_walk':
            assert row['spawn_z'] == row['geometry']['platform_height'] + 0.4
        else:
            assert row['spawn_z'] == 0.4
    train_data = SimpleNamespace(default_root_state=torch.zeros(1024, 13))
    train_env = SimpleNamespace(robot=SimpleNamespace(data=train_data))
    s2_train_real.apply_train_spawn_z(train_env, plan)
    assert torch.all(train_data.default_root_state[:, 2] ==
                     torch.tensor([row['spawn_z'] for row in plan['rows']]))


def test_course_rollout_plans_and_512_budget():
    for course, cap in (('S2.1', 800_000_000), ('S2.2', 1_200_000_000),
                        ('S2.3', 1_200_000_000)):
        plan = s2_train.rollout_plan(512, course=course)
        assert plan['learning_cap'] == cap
        assert plan['samples_per_rollout'] == 24_576
        assert plan['max_rollouts'] * 24_576 <= cap
        assert plan['checkpoint_every_rollouts'] == 204
    assert s2_train.rollout_plan(512, course='S2.2')['max_rollouts'] == 48_828
    assert s2_train.rollout_plan(512, course='S2.3')['max_rollouts'] == 48_828


def test_course_role_plans_switch_main_and_review():
    s21 = s2_train_env.build_role_plan(512, course='S2.1')
    s22 = s2_train_env.build_role_plan(512, course='S2.2')
    s23 = s2_train_env.build_role_plan(512, course='S2.3')
    assert {row['stair_count'] for row in s21['envs']
            if row['role'] == 'stairs_walk'} == {10, 20}
    assert {row['stair_count'] for row in s22['envs']
            if row['role'] == 'stairs_walk'} == {40}
    assert {row['task'] for row in s23['envs']
            if row['role'] == 'stairs_walk'} == {'stairs_stop'}
    assert {row['task'] for row in s23['envs']
            if row['role'] == 'platform_walk'} == {'platform_stop'}
    assert {row['stair_count'] for row in s22['envs']
            if row['role'] == 'l40_milestone'} == {10, 20}
    assert {row['stair_count'] for row in s23['envs']
            if row['role'] == 'l40_milestone'} == {10, 20}
    assert {row['stop_mode'] for row in s23['envs']
            if row['task'] == 'stairs_stop'} == {'instant', 'ramp_2s'}
    assert all(5.0 <= row['post_distance'] <= 10.0 for row in s22['envs']
               if row['role'] == 'stairs_walk')
    assert all(3.0 <= row['post_distance'] <= 10.0 for row in s23['envs']
               if row['role'] == 'stairs_walk')


def test_course_dry_run_switches_cap(tmp_path):
    import train_a2_v09_s2 as train_cli
    evidence = tmp_path / 'fail1024.json'
    evidence.write_text(json.dumps({'num_envs': 1024, 'admitted': False,
                                    'failure_reasons': ['oom']}))
    out = tmp_path / 'dry22'
    assert train_cli.main(['--dry-run', '--num-envs', '512',
                           '--course', 'S2.2',
                           '--fallback-evidence', str(evidence),
                           '--output-dir', str(out)]) == 0
    doc = json.loads((out / 'train_dry_run_summary.json').read_text())
    assert doc['plan']['course'] == 'S2.2'
    assert doc['plan']['training_plan']['max_rollouts'] == 48_828
    assert doc['plan']['training_plan']['learning_cap'] == 1_200_000_000


def test_course_env_plan_geometry_without_simulator():
    from src.tasks.velocity.v09 import s2_train_real
    p21 = s2_train_real.build_env_plan(512, course='S2.1')
    p22 = s2_train_real.build_env_plan(512, course='S2.2')
    p23 = s2_train_real.build_env_plan(512, course='S2.3')
    assert all(row['geometry']['platform_length'] == 20.0 for row in p21['rows']
               if row['role'] == 'l40_milestone')
    assert all(row['geometry']['platform_length'] == 20.0 for row in p22['rows']
               if row['task'] == 'stairs_walk' and row['stair_count'] == 40)
    assert any(row['task'] == 'platform_stop' for row in p23['rows'])
    assert any(row['task'] == 'stairs_stop' for row in p23['rows'])


def test_course_checkpoint_metadata_preserves_course():
    from src.tasks.velocity.v09.model import V09ActorCritic
    env = s2_train_env.S2FakeTrainEnv(
        num_envs=2, role_plan=s2_train_env.build_role_plan(2, course='S2.3'))
    trainer = s2_train.S2Trainer(model=V09ActorCritic(), env=env, device='cpu',
                                 course='S2.3')
    metadata = trainer.checkpoint_metadata()
    assert metadata['course'] == 'S2.3'
    assert metadata['curriculum']['role'] == 'S2.3'


def test_old_log_flatten_is_time_major():
    import torch
    transitions = [
        dict(log_probs=torch.tensor([0.0, 10.0, 20.0])),
        dict(log_probs=torch.tensor([1.0, 11.0, 21.0])),
    ]
    flat = s2_train.flatten_time_major_old_logs(transitions)
    assert flat.tolist() == [0.0, 10.0, 20.0, 1.0, 11.0, 21.0]
    env_major = torch.stack([row['log_probs'] for row in transitions], 1).reshape(-1)
    assert env_major.tolist() != flat.tolist()


def test_flat_anchor_is_called_during_update():
    from src.tasks.velocity.v09.model import V09ActorCritic
    import torch
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, obs):
            self.calls += 1
            return torch.zeros(obs['proprio'].shape[0], 12)
    env = s2_train_env.S2FakeTrainEnv(
        num_envs=2, role_plan=s2_train_env.build_role_plan(2))
    trainer = s2_train.S2Trainer(model=V09ActorCritic(), env=env,
                                 teacher=Teacher(), device='cpu')
    trainer.train_rollout()
    assert trainer.teacher.calls > 0
