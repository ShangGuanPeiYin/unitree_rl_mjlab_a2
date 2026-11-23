"""CPU regression tests for the V13 S2.1 evidence pack (plan v13 §6.1).

These tests exercise the same checks as ``scripts/verify_a2_v09_s2_v13.py`` so a
regression in the production reward/PPO/GAE/ledger/R1/anchor code fails fast.
Expected reward/anchor values always come from the frozen hand-calc JSON, never
from production reward code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap  # noqa: E402

a2_v07_bootstrap.activate(ROOT)

import verify_a2_v09_s2_v13 as evidence  # noqa: E402
from src.tasks.velocity.v09 import s2_loss, s2_r1, s2_runner  # noqa: E402

HANDCALC = ROOT / 'logs/a2_v09/s2_0/verification/numerical_acceptance_handcalc_v13.json'


# ---------------------------------------------------------------------------
# hand-calc loader + 13 frozen cases
# ---------------------------------------------------------------------------
def test_handcalc_loader_and_13_cases_match_production():
    result = evidence.check_handcalc(HANDCALC)
    assert result['case_count'] == 13
    assert result['input_sha256_matches_plan'] is True
    assert result['passed'] is True, [
        case['case'] for case in result['cases'] if not case['passed']]
    for case in result['cases']:
        assert case['term_count_ok'] is True
        assert len(case['terms']) == 20
        assert set(case['terms']) == set(evidence.s2_rewards.WEIGHTS)
        assert case['passed'] is True, case['case']
        assert all(term['passed'] for term in case['terms'].values())
        assert case['scalars']['total']['abs_error'] <= 1e-12
        assert all(flag['passed'] for flag in case['discrete'].values())
    # max per-term error stays well inside the float64 tolerance row
    assert max(result['max_abs_error_per_term'].values()) <= 1e-12
    assert result['max_abs_error']['total'] <= 1e-12
    # v07 is contrast only, never a judge: the five documented cases must be
    # present (the frozen JSON actually carries expected_v07 for all 13).
    assert len(result['v07_difference_report']) >= 5
    reported = {row['case'] for row in result['v07_difference_report']}
    assert {'flat_cruise_full20', 'decel_partial_command',
            'timeout_pays_failure', 'success_only', 'top_only'} <= reported
    assert 'NOT a judge' in result['v07_note']


def test_handcalc_json_expected_values_are_self_consistent():
    """The loader maps every key and the JSON expectation obeys plan §2.4."""
    data = json.loads(HANDCALC.read_text(encoding='utf-8'))
    for name, case in data['cases'].items():
        assert set(case['input']) == set(evidence.HANDCALC_INPUT_MAP), name
        expected = case['expected_s2']
        column = expected['column']
        continuous = 0.02 * sum(
            data['weights'][term][column] * expected['terms'][term]
            for term in data['weights'])
        assert continuous == pytest.approx(expected['continuous'], abs=1e-12)
        total = (expected['continuous'] + expected['forward']
                 + 2.0 * expected['top'] + 8.0 * expected['success']
                 - 10.0 * expected['failure'])
        assert total == pytest.approx(expected['total'], abs=1e-12)


# ---------------------------------------------------------------------------
# g_level ramp
# ---------------------------------------------------------------------------
def test_g_level_sequence_matches_frozen_json():
    result = evidence.check_g_level(HANDCALC)
    assert result['passed'] is True
    assert result['ramp_steps'] == 25
    produced = [row['produced']['g_level'] for row in result['rows']]
    assert produced == pytest.approx([0.0, 0.0, 0.04, 0.96, 1.0, 0.0])
    assert [row['passed'] for row in result['rows']] == [True] * 6
    # production lifecycle adapter reproduces the same ramp
    assert [row['produced_g_level'] for row in result['lifecycle_ramp']] == \
        pytest.approx(produced)
    assert all(row['passed'] for row in result['lifecycle_ramp'])


# ---------------------------------------------------------------------------
# PPO storage / logical minibatches / Adam count
# ---------------------------------------------------------------------------
def test_ppo_storage_unique_labels_and_adam_count():
    result = evidence.check_ppo_storage(num_envs=3, time_steps=6)
    assert result['passed'] is True
    assert result['samples_total'] == 18
    assert result['alignment']['all_aligned'] is True
    assert result['measured_adam_updates'] == result['expected_adam_updates']
    assert result['expected_adam_updates'] == (
        s2_runner.EPOCHS * s2_runner.MINIBATCHES)
    assert result['coverage_per_sample']['min'] == s2_runner.EPOCHS
    assert result['coverage_per_sample']['max'] == s2_runner.EPOCHS
    # env-major flatten differs from the time-major contract
    assert result['env_major_counterexample']['mismatches'] > 0
    for sample in result['alignment']['per_sample']:
        assert sample['aligned'] is True
        assert sample['expected_label'] == 1000 * sample['env_id'] + sample['step']
        assert sample['obs'] == sample['actions'] == sample['old_logp'] \
            == sample['advantage'] == sample['returns'] == sample['expected_label']


# ---------------------------------------------------------------------------
# GAE boundary
# ---------------------------------------------------------------------------
def test_gae_boundary_matches_independent_recompute():
    result = evidence.check_gae_boundary()
    assert result['passed'] is True
    assert len(result['rows']) == 2
    for row in result['rows']:
        assert row['passed'] is True
        assert row['max_abs_error']['advantages'] <= 1e-12
        assert row['max_abs_error']['returns'] <= 1e-12
        assert row['produced']['advantages'] == pytest.approx(
            row['independent']['advantages'], abs=1e-12)
        assert row['produced']['returns'] == pytest.approx(
            row['independent']['returns'], abs=1e-12)
    probes = result['boundary_probes']
    assert probes['last_step_not_done_bootstraps_last_value']['passed'] is True
    assert probes['last_step_not_done_bootstraps_last_value'][
        'measured_advantage_delta_at_last_step'] == pytest.approx(
            s2_runner.GAMMA, abs=1e-12)
    assert probes['middle_done_blocks_cross_reset']['passed'] is True
    assert probes['middle_done_blocks_cross_reset'][
        'measured_max_advantage_change'] <= 1e-12
    assert probes['last_step_done_ignores_last_value']['passed'] is True
    contract = result['collect_rollout_contract']
    assert contract['static_ok'] is True
    assert all(value is not None for value in contract['static_assertions'].values())


# ---------------------------------------------------------------------------
# §5.4 flat anchor
# ---------------------------------------------------------------------------
def test_flat_anchor_examples_match_frozen_json():
    result = evidence.check_flat_anchor(HANDCALC)
    assert result['examples_passed'] is True
    by_name = {row['name']: row for row in result['examples']}
    assert by_name['one_clean_sample']['accumulated_mean'] == pytest.approx(0.01)
    assert by_name['one_clean_sample']['loss'] == pytest.approx(0.001)
    assert by_name['two_clean_samples']['accumulated_mean'] == pytest.approx(0.025)
    assert by_name['two_clean_samples']['loss'] == pytest.approx(0.0025)
    assert by_name['no_clean_sample']['accumulated_mean'] == 0.0
    assert by_name['no_clean_sample']['loss'] == 0.0


def test_flat_anchor_trainer_contract_is_documented_behaviour():
    """Pass only when the trainer accumulates the numerator per micro-batch.

    If production regresses to mode ``b`` (divide per micro by the global
    rollout n_clean) the evidence must be an honest FAIL, and this test asserts
    that documented behaviour instead of skipping.
    """
    result = evidence.check_flat_anchor(HANDCALC)
    aggregation = result['trainer_aggregation']
    assert aggregation['mode'] in ('a', 'b', 'unknown')
    if aggregation['mode'] == 'a':
        assert result['passed'] is True
        assert aggregation['defect'] is None
    elif aggregation['mode'] == 'b':
        assert result['passed'] is False
        assert aggregation['defect']
    else:
        assert result['passed'] is False


def test_anchor_accumulator_handles_microbatches():
    """Two micro-batches must average to the same logical mean."""
    student = np.array([[0.1] * 12, [0.2] * 12], dtype=np.float64)
    teacher = np.zeros((2, 12), dtype=np.float64)
    accumulator = s2_loss.AnchorAccumulator()
    accumulator.add(student[:1], teacher[:1], np.array([True]))
    value = accumulator.add(student[1:], teacher[1:], np.array([True]))
    assert accumulator.n_clean == 2
    assert value == pytest.approx(0.025)
    assert accumulator.total(0.0) == pytest.approx(s2_loss.BETA_FLAT * 0.025)


# ---------------------------------------------------------------------------
# R1 rejection chain
# ---------------------------------------------------------------------------
def test_r1_rejection_chain(tmp_path):
    result = evidence.check_r1_rejection(tmp_path / 'r1')
    assert result['passed'] is True
    assert len(result['invalidated_sha256']) == 4
    assert result['sha256_matches_handcalc_json'] is True
    assert result['prefixes_cover_all_four'] is True
    assert all(row['rejected'] for row in result['dummy_file_rejection'])
    present = [row for row in result['real_file_rejection'] if row['present']]
    for row in present:
        assert row['sha256_matches'] is True
        assert row['rejected'] is True
    cli = result['training_cli']
    assert cli['default_is_not_r1'] is True
    assert cli['attempt_id_default'] == 's2.1-r2'
    assert cli['reject_invalid_attempt_rejects_r1'] is True
    assert cli['reject_invalid_attempt_accepts_default'] is True
    assert result['wiring']['ok'] is True
    assert result['wiring']['gate_checkpoint_calls']
    assert result['wiring']['cli_resume_calls']
    assert result['wiring']['trainer_resume_calls']


def test_r1_reject_invalid_attempt_and_defaults():
    assert s2_r1.reject_invalid_attempt('s2.1-r2') is True
    with pytest.raises(ValueError, match='s2.1-r1'):
        s2_r1.reject_invalid_attempt('s2.1-r1')


# ---------------------------------------------------------------------------
# exception settlement
# ---------------------------------------------------------------------------
def test_exception_settlement_amounts(tmp_path):
    ledger_path = ROOT / 'logs/a2_v09/s2/ledger.jsonl'
    before = (hashlib.sha256(ledger_path.read_bytes()).hexdigest()
              if ledger_path.is_file() else None)
    result = evidence.check_exception_settlement(tmp_path / 'settlement')
    after = (hashlib.sha256(ledger_path.read_bytes()).hexdigest()
             if ledger_path.is_file() else None)
    assert before == after, 'production ledger must not be modified'

    assert result['passed'] is True
    assert result['rollout_counter_static']['rollout_steps_zeroed_before_reset'] is True
    by_name = {row['scenario']: row for row in result['scenarios']}
    assert set(by_name) == {'reset_failure', 'step_failure', 'update_failure',
                            'checkpoint_failure'}

    reset = by_name['reset_failure']
    assert reset['reserve'] == 96 and reset['commit'] == 0 and reset['release'] == 96
    assert reset['physically_executed_samples'] == 0
    assert reset['ledger_spent'] == 0 and reset['ledger_reserved'] == 0

    step = by_name['step_failure']
    assert step['reserve'] == 96 and step['commit'] == 6 and step['release'] == 90
    assert step['physically_executed_samples'] == 6
    assert step['ledger_spent'] == 6 and step['ledger_reserved'] == 0

    update = by_name['update_failure']
    assert update['commit'] == 96 and update['release'] == 0
    assert update['physically_executed_samples'] == 96
    assert update['ledger_spent'] == 96 and update['ledger_reserved'] == 0

    checkpoint = by_name['checkpoint_failure']
    assert checkpoint['checkpoint_error']
    assert checkpoint['commit'] == 96 and checkpoint['release'] == 0
    assert checkpoint['ledger_spent'] == 96 and checkpoint['ledger_reserved'] == 0
    assert checkpoint['learning_samples'] == 96
    assert result['known_gap']['present'] is True
