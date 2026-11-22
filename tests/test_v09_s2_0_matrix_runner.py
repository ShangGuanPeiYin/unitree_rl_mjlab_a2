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
