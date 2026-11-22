"""CPU-only T11 matrix runner selection/dry-run and post_reset tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
import run_a2_v09_s2_matrix as matrix
from src.tasks.velocity.v09 import s2_env


def test_matrix_entries_cover_150_plus_h_and_budget():
    entries = matrix.load_entries()
    assert len(entries) == 156
    s2 = [entry for entry in entries if entry['kind'] == 's2']
    h = [entry for entry in entries if entry['kind'] == 'h']
    assert len(s2) == 150 and len(h) == 6
    assert sum(entry['steps'] for entry in s2) == 419_402
    assert sum(entry['steps'] for entry in entries if entry['kind'] == 'h') == 7_874
    assert all(entry['obligation'] == 's2_0_matrix_two_branches' for entry in s2)
    assert all(entry['obligation'] == 'h_series_baseline_supplement' for entry in h)


def test_dry_run_limit_two_stairs_is_real_plus_plane(tmp_path):
    code = matrix.main(['--dry-run', '--only-task', 'stairs_walk', '--limit', '2',
                        '--ledger', str(tmp_path / 'missing.jsonl')])
    assert code == 0


def test_post_reset_is_boolean_trace_field():
    assert 'post_reset' in s2_env.TRACE_FIELDS
    assert 'post_reset' in s2_env.BOOLEAN_FIELDS
    assert 'sim_state_sha256' in s2_env.TRACE_FIELDS


def test_orphan_release_is_idempotent(tmp_path):
    import json
    from src.tasks.velocity.v09 import s2_budget, s2_cases, s2_ledger
    cases = ROOT / 'logs/a2_v09/s2_0/cases'
    matrix_manifest = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', ROOT)
    h_manifest = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', ROOT)
    plan = s2_budget.build_reservation_plan(matrix_manifest, h_manifest)
    path = tmp_path / 'ledger.jsonl'
    ledger = s2_ledger.S2Ledger(path, run_id='r')
    ledger.reserve_stage(plan)
    base = 'smoke:S2_DIAG:0:test-attempt'
    ledger.reserve('s2_0_matrix_two_branches', 10,
                   operation_id=f'{base}:reserve', reason='child reserve')
    receipt = matrix.settle_orphan(path, base, entry={'case_sha256': 'a' * 64})
    assert receipt['samples'] == 10
    state = s2_ledger.S2Ledger(path, run_id='r').state()
    assert state['obligations']['s2_0_matrix_two_branches']['reserved'] == 0
    assert matrix.settle_orphan(path, base, entry={'case_sha256': 'a' * 64}) is None
