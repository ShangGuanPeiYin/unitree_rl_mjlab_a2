"""CPU-only resources/checkpoint/ledger/acceptance/train-CLI tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

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
        normalizer_state={'frozen': True}, teacher_source='6400',
        beta_flat=0.1, learning_samples=1, evaluation_samples=2,
        engineering_samples=3, rollback_samples=0,
        gates_passed=[dict(suite='gate_A', case_sha256='a' * 64,
                           candidate_sha256='b' * 64)],
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
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.append('stage_reserve', samples=1000)
    ledger.append('reserve', category='engineering', samples=100)
    ledger.append('commit', category='engineering', samples=40)
    state = ledger.state()
    assert state['envelope'] == 1000
    assert state['reserved']['engineering'] == 60
    assert state['spent']['engineering'] == 40
    lines = (tmp_path / 'ledger.jsonl').read_text().splitlines()
    assert len(lines) == 4
    with pytest.raises(ValueError):
        s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='other')


def test_s2_acceptance_quality_and_matrices():
    metrics = dict(pitch_rms_deg=3.0, pitch_p99_deg=8.0,
                   pitch_over15_max_frames=0, speed_rmse=0.05,
                   nonfoot_contact=False, cruise_stall_latched=False)
    assert s2_acceptance.quality_reasons(metrics) == []
    assert s2_acceptance.quality_reasons(
        dict(metrics, pitch_rms_deg=6.0)) == ['pitch_rms']
    stop_metrics = dict(metrics, recovery_seconds=1.0, stand_mean_speed=0.01,
                        stand_peak_speed=0.05, stand_max_displacement=0.02,
                        stand_support_fraction=0.95)
    assert s2_acceptance.quality_reasons(stop_metrics, stop=True) == []
    matrix = s2_acceptance.gate_matrix()
    assert matrix['S2.1']['groups'] == 36
    assert matrix['S2.2']['groups'] == 33
    assert matrix['S2.3']['groups'] == 51


def test_final_acceptance_requires_2700_and_95_percent():
    records = []
    for group in range(27):
        for case in range(100):
            records.append(dict(group=group, success=True,
                                metrics=dict(pitch_rms_deg=1.0,
                                             pitch_p99_deg=2.0,
                                             pitch_over15_max_frames=0,
                                             speed_rmse=0.05),
                                stop=False))
    result = s2_acceptance.final_acceptance(records)
    assert result['group_count'] == 27 and result['passed'] is True
    for index in range(6):
        records[index]['success'] = False
    result = s2_acceptance.final_acceptance(records)
    assert result['groups'][0]['successes'] == 94
    assert result['passed'] is False


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
