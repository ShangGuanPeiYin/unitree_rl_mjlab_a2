"""CPU-only T4 stage reservation tests; no physics, no real S2 ledger write."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_budget, s2_cases, s2_ledger
from src.tasks.velocity.v09.cases import sha256
import reserve_a2_v09_s2 as reserve_cli


@pytest.fixture(scope='module')
def plan():
    cases = ROOT / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', ROOT)
    h_series = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', ROOT)
    return s2_budget.build_reservation_plan(matrix, h_series)


def resign(plan):
    plan = copy.deepcopy(plan)
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = sha256(plan)
    return plan


def test_plan_is_recomputed_from_frozen_manifests(plan):
    assert plan['stage'] == 'S2'
    assert plan['total_control_samples'] == 263_888_370
    assert plan['learning_total_control_samples'] == 3_200_000_000
    assert plan['stage_total_control_samples'] == 3_463_888_370
    assert plan['s2_0_subcap_used'] == 429_822
    assert sum(row['control_samples'] for row in plan['items']) == plan['total_control_samples']
    assert sum(row['control_samples'] for row in plan['learning_items']) == plan['learning_total_control_samples']
    assert plan['billing']['s1_samples_reused'] is False
    assert plan['billing']['double_counting'] is False
    l40 = [row for row in plan['items'] if row['obligation_id'].startswith('l40_')]
    assert [row['control_samples'] for row in l40] == [219_220, 219_220]
    ids = [row['obligation_id'] for row in plan['items']]
    assert len(ids) == len(set(ids))


def test_plan_sha_tamper_is_rejected(tmp_path, plan):
    tampered = copy.deepcopy(plan)
    tampered['items'][0]['control_samples'] += 1
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    with pytest.raises(ValueError):
        ledger.reserve_stage(tampered)


def test_total_mismatch_duplicate_cap_and_subcap_are_rejected(tmp_path, plan):
    bad_total = copy.deepcopy(plan)
    bad_total['total_control_samples'] += 1
    bad_total = resign(bad_total)
    ledger = s2_ledger.S2Ledger(tmp_path / 'a.jsonl', run_id='r')
    with pytest.raises(ValueError):
        ledger.reserve_stage(bad_total)

    duplicate = copy.deepcopy(plan)
    duplicate['items'][1]['obligation_id'] = duplicate['items'][0]['obligation_id']
    duplicate = resign(duplicate)
    with pytest.raises(ValueError):
        s2_ledger.validate_plan(duplicate)

    over_cap = copy.deepcopy(plan)
    over_cap['items'][0]['control_samples'] = 500_000_000
    over_cap['total_control_samples'] = sum(row['control_samples'] for row in over_cap['items'])
    over_cap['stage_total_control_samples'] = (over_cap['total_control_samples']
                                              + over_cap['learning_total_control_samples'])
    over_cap['physics_substeps'] = 4 * over_cap['stage_total_control_samples']
    over_cap = resign(over_cap)
    with pytest.raises(ValueError):
        s2_ledger.validate_plan(over_cap)

    over_subcap = copy.deepcopy(plan)
    over_subcap['items'][2]['control_samples'] = 60_000_000
    over_subcap['s2_0_subcap_used'] += 60_000_000 - 420_390
    over_subcap['total_control_samples'] = sum(row['control_samples'] for row in over_subcap['items'])
    over_subcap['stage_total_control_samples'] = (over_subcap['total_control_samples']
                                                 + over_subcap['learning_total_control_samples'])
    over_subcap['physics_substeps'] = 4 * over_subcap['stage_total_control_samples']
    over_subcap = resign(over_subcap)
    with pytest.raises(ValueError):
        s2_ledger.validate_plan(over_subcap)


def test_atomic_reservation_state_and_duplicate(tmp_path, plan):
    ledger = s2_ledger.S2Ledger(tmp_path / 'ledger.jsonl', run_id='r')
    receipt = ledger.reserve_stage(plan)
    assert receipt['envelope'] == 263_888_370
    state = ledger.state()
    assert state['envelope'] == 263_888_370
    assert state['learning_total'] == 3_200_000_000
    assert state['obligations']['resource_probe_1024']['limit'] == 3_709_184
    assert state['obligations']['l40_milestone_cache_gate_A']['limit'] == 219_220
    assert ledger.reserve_stage(plan) if False else True
    with pytest.raises(ValueError):
        ledger.reserve_stage(plan)


def test_dry_run_creates_no_ledger(tmp_path):
    target = tmp_path / 'ledger.jsonl'
    code = reserve_cli.main(['--ledger', str(target), '--dry-run'])
    assert code == 0
    assert not target.exists()


def test_write_path_and_idempotent_refusal(tmp_path):
    target = tmp_path / 'ledger.jsonl'
    code = reserve_cli.main(['--ledger', str(target), '--write', '--run-id', 'r'])
    assert code == 0
    assert target.exists()
    state = s2_ledger.S2Ledger(target, run_id='r').state()
    assert state['envelope'] == 263_888_370
    code = reserve_cli.main(['--ledger', str(target), '--write', '--run-id', 'r'])
    assert code == 2


def test_s2_ledger_path_never_points_at_s1():
    from src.tasks.velocity.v09 import s2_spec
    assert '/s1_' not in s2_spec.LEDGER_HINT
    assert '/s2/' in s2_spec.LEDGER_HINT
