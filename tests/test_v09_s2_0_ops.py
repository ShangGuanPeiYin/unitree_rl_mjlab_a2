"""CPU-only resources/checkpoint/ledger/acceptance/train-CLI tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import (s2_acceptance, s2_checkpoint, s2_ledger,
                                    s2_resources, s2_runner)


def test_probe_plan_and_1024_then_512_selection():
    assert s2_resources.probe_plan(1024)['control_samples'] > 0
    good = s2_resources.ProbeResult(1024, 7000, 5000, False, False, False,
                                    5, 50, 1, True, 100, 'a' * 64)
    assert good.admitted() is True
    bad = s2_resources.ProbeResult(1024, 7500, 5000, False, False, False,
                                   5, 50, 1, True, 100, 'b' * 64)
    assert bad.admitted() is False
    chosen = s2_resources.choose_resource([bad, good])
    assert chosen['selected'] == 1024
    fallback = s2_resources.ProbeResult(512, 6000, 5000, False, False, False,
                                        5, 50, 1, True, 100, 'c' * 64)
    chosen = s2_resources.choose_resource([bad, fallback])
    assert chosen['selected'] == 512
    assert chosen['fallback_evidence'][0]['failures'] == ['gpu_peak_mib']


def _metadata(**overrides):
    base = dict(
        stage='S2', course='S2.1', curriculum={'name': 'short'}, strength='S2.1',
        model_state='model', optimizer_state='adam', policy_std={'value': 0.3},
        normalizer_state={'frozen': True}, teacher_source={'source': '6400', 'sha256': '1' * 64},
        beta_flat=0.1, learning_samples=1, evaluation_samples=2,
        engineering_samples=3, rollback_samples=0,
        gates_passed=[dict(suite='gate_A', case_sha256='a' * 64,
                           candidate_sha256='b' * 64,
                           ab_sequence=['gate_A', 'gate_B'], gate_uses=1)],
        case_manifest_sha256='c' * 64, candidate_sha256='d' * 64,
        gate_ab_sequence=['gate_A'], gate_uses=1, final_uses=0,
        engineering_rerun_reasons=[], random_state={'torch': 1},
        role_assignment={'flat_walk': 1}, randomization={'enabled': False},
        command_flow={'ramp_steps': 100}, delay_config={'steps': 0},
        entry_weight={'sha256': 'e' * 64}, ledger_run_id='run',
        code_sha256='f' * 64, manifest_sha256='0' * 64)
    base.update(overrides)
    return base


def test_s2_checkpoint_metadata_contract():
    assert s2_checkpoint.validate_metadata(_metadata()) is True
    broken = _metadata()
    broken.pop('policy_std')
    with pytest.raises(ValueError):
        s2_checkpoint.validate_metadata(broken)
    with pytest.raises(ValueError):
        s2_checkpoint.validate_metadata(_metadata(course='S1.2'))
    with pytest.raises(ValueError):
        s2_checkpoint.validate_metadata(_metadata(beta_flat=2.0))


def test_s2_ledger_hash_chain_and_namespace(tmp_path):
    from src.tasks.velocity.v09 import s2_budget, s2_cases
    root = ROOT
    cases = root / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', root)
    h_series = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', root)
    plan = s2_budget.build_reservation_plan(matrix, h_series)
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='run')
    receipt = ledger.reserve_stage(plan)
    assert receipt['sequence'] == 1
    ledger.reserve('resource_probe_1024', 100, operation_id='op-reserve', reason='test')
    ledger.commit('resource_probe_1024', 40, operation_id='op-commit', reason='test')
    state = ledger.state()
    assert state['envelope'] == plan['total_control_samples']
    assert state['obligations']['resource_probe_1024']['reserved'] == 60
    assert state['obligations']['resource_probe_1024']['spent'] == 40
    with pytest.raises(ValueError):
        ledger.reserve_stage(plan)
    with pytest.raises(ValueError):
        s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='other')


def test_s2_acceptance_quality_and_matrices():
    metrics = dict(pitch_rms_deg=3.0, pitch_p99_deg=8.0,
                   pitch_over15_max_frames=0, speed_rmse=0.05,
                   nonfoot_contact=False, cruise_stall_latched=False,
                   valid_cruise_window=True)
    assert s2_acceptance.quality_reasons(metrics) == []
    assert s2_acceptance.quality_reasons(
        dict(metrics, pitch_rms_deg=6.0)) == ['pitch_rms']
    stop_metrics = dict(metrics, recovery_seconds=1.0, stand_mean_speed=0.01,
                        stand_peak_speed=0.05, stand_max_displacement=0.02,
                        stand_support_fraction=0.95,
                        stand_roll_abs_max_deg=2.0,
                        stand_pitch_abs_max_deg=2.0, stand_height_min=0.35)
    assert s2_acceptance.quality_reasons(stop_metrics, stop=True) == []
    matrix = s2_acceptance.gate_matrix()
    assert matrix['S2.1']['groups'] == 36
    assert matrix['S2.2']['groups'] == 33
    assert matrix['S2.3']['groups'] == 51


def test_final_acceptance_requires_2700_and_95_percent():
    records = []
    refs = {}
    for group in range(27):
        flat = group < 9
        task = 'flat_walk' if flat else 'stairs_walk'
        if flat:
            refs[group] = dict(speed_rmse=0.05, pitch_rms_deg=1.0, slip=0.1,
                               action_change=0.1, drift=0.02, stride=0.2,
                               frequency=2.0)
        for case in range(100):
            records.append(dict(
                group=group, task=task, success=True, stop=False,
                metrics=dict(valid_cruise_window=True, pitch_rms_deg=1.0,
                             pitch_p99_deg=2.0, pitch_over15_max_frames=0,
                             speed_rmse=0.05, slip=0.1, action_change=0.1,
                             drift=0.0, stride=0.2, frequency=2.0)))
    result = s2_acceptance.final_acceptance(records, original_6400=refs)
    assert result['group_count'] == 27 and result['passed'] is True
    for index in range(6):
        records[index]['success'] = False
    result = s2_acceptance.final_acceptance(records, original_6400=refs)
    assert result['groups'][0]['successes'] == 94
    assert result['passed'] is False
    # A 50/50 split cannot masquerade as 27×100.
    short = records[:2700 - 1]
    with pytest.raises(ValueError):
        s2_acceptance.final_acceptance(short, original_6400=refs)
    # Missing 6400 reference must fail, not silently pass.
    result = s2_acceptance.final_acceptance(records, original_6400=None)
    assert result['passed'] is False
    assert 'missing_6400_reference' in result['flat_retention_reasons']


def test_train_cli_dry_run(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import train_a2_v09_s2 as train_cli
    code = train_cli.main(['--dry-run', '--num-envs', '1024',
                           '--output-dir', str(tmp_path)])
    assert code == 0
    doc = json.loads((tmp_path / 'train_dry_run_summary.json').read_text())
    assert doc['dry_run'] is True and doc['training_started'] is False
    assert doc['plan']['num_envs'] == 1024
    with pytest.raises(SystemExit):
        train_cli.main(['--num-envs', '1024'])
    with pytest.raises(SystemExit):
        train_cli.main(['--dry-run', '--num-envs', '2048'])


def test_b_v6_3_512_requires_1024_failure_and_probe_has_video():
    good512 = s2_resources.ProbeResult(512, 6000, 5000, False, False, False,
                                       5, 50, 1, True, 100, 'c' * 64)
    with pytest.raises(ValueError):
        s2_resources.choose_resource([good512])
    bad1024 = s2_resources.ProbeResult(1024, 7500, 5000, False, False, False,
                                       5, 50, 1, True, 100, 'b' * 64)
    assert s2_resources.choose_resource([bad1024, good512])['selected'] == 512
    plan = s2_resources.probe_plan(1024)
    assert plan['evaluation_timeout_steps'] == 9472
    assert plan['evaluation_samples'] == 947200
    assert plan['video_samples'] == 9472
    assert plan['control_samples'] == 3_709_184


def test_b_v6_3_train_cli_512_requires_fallback_evidence(tmp_path):
    import train_a2_v09_s2 as train_cli
    with pytest.raises(SystemExit):
        train_cli.main(['--dry-run', '--num-envs', '512',
                        '--output-dir', str(tmp_path / 'a')])
    evidence = tmp_path / 'fail1024.json'
    evidence.write_text(json.dumps({'num_envs': 1024, 'admitted': False}))
    code = train_cli.main(['--dry-run', '--num-envs', '512',
                           '--fallback-evidence', str(evidence),
                           '--output-dir', str(tmp_path / 'b')])
    assert code == 0


def test_b_v6_4_duplicate_operation_is_rejected(tmp_path):
    from src.tasks.velocity.v09 import s2_budget, s2_cases
    cases = ROOT / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', ROOT)
    h_series = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', ROOT)
    plan = s2_budget.build_reservation_plan(matrix, h_series)
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    ledger.reserve_stage(plan)
    ledger.reserve('resource_probe_1024', 10, operation_id='o1', reason='test')
    with pytest.raises(ValueError):
        ledger.reserve('resource_probe_1024', 10, operation_id='o1', reason='dup')
    ledger.commit('resource_probe_1024', 10, operation_id='o1-commit', reason='ok')


def test_b_v6_5_checkpoint_loader_and_cross_version_guard(tmp_path):
    import torch
    meta = _metadata()
    meta['model_state'] = {'w': 1}
    meta['optimizer_state'] = {'step': 0}
    path = tmp_path / 's2.pt'
    s2_checkpoint.save_checkpoint(path, {'w': 1}, {'step': 0}, meta)
    payload = s2_checkpoint.load_checkpoint(path)
    assert payload['metadata']['course'] == 'S2.1'
    bad = tmp_path / 'bad.pt'
    torch.save({'schema': 10, 'sds_sha256': 'x'}, bad)
    with pytest.raises(ValueError):
        s2_checkpoint.load_checkpoint(bad)
    broken = _metadata()
    broken['gates_passed'][0].pop('ab_sequence')
    with pytest.raises(ValueError):
        s2_checkpoint.validate_metadata(broken)
    with pytest.raises(ValueError):
        s2_checkpoint.save_checkpoint(tmp_path / 'empty.pt', {}, {}, meta)


def test_b_v6_6_ppo_logical_minibatch_lr_and_course_transition():
    assert s2_runner.gaussian_entropy(0.5) > 0
    assert s2_runner.approximate_kl([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.0)
    state = s2_runner.PPOTrainerState(learning_rate=3e-5, std=0.3, course='S2.0')
    before = state.std
    result = state.logical_minibatch_update(
        old_log_prob=[0.0, 0.0], new_log_prob=[0.01, -0.01],
        microbatch_gradients=[(np.array([1.0]), 2), (np.array([3.0]), 6)])
    assert result['optimizer_steps'] == 1
    assert result['adam_updates'] == 1
    assert result['gradient'].tolist() == [2.5]
    assert result['learning_rate'] == pytest.approx(4.5e-5)  # KL < 0.005 → ×1.5
    switch = state.switch_course('S2.1')
    assert switch['learning_rate'] == pytest.approx(4.5e-5)
    assert state.std == pytest.approx(before)
    assert state.adam_id == 'adam_s2.1'
    with pytest.raises(ValueError):
        s2_runner.assert_frozen(teacher_eval=False, teacher_grad_enabled=False,
                                normalizer_updates=0)
    with pytest.raises(ValueError):
        s2_runner.assert_frozen(teacher_eval=True, teacher_grad_enabled=False,
                                normalizer_updates=1)
