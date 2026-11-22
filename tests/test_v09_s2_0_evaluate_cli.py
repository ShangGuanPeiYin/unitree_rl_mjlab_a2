"""CPU-only S2.0 evaluator dry-run tests; no physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
import evaluate_a2_v09_s2_0 as cli


def test_dry_run_both_branches_and_summary(tmp_path):
    summary, output = cli.dry_run(map_branch='both', max_envs=20,
                                  output_dir=tmp_path)
    assert summary['dry_run'] is True
    assert summary['physics_started'] is False
    assert summary['frozen_cases'] == 75
    assert summary['h_series_cases'] == 6
    assert summary['selected_branch_trajectories'] == 150
    assert summary['entries_by_task'] == {'flat_walk': 30, 'stairs_walk': 60,
                                          'platform_walk': 60}
    assert summary['env_batches'] == 8
    assert summary['run_plan_batches'] == 30
    assert len(summary['batch_table']) == 30
    assert summary['evidence_contract_sha256']
    assert output.exists() and output.name == 'dry_run_summary.json'


def test_dry_run_single_branch_real(tmp_path):
    summary, _ = cli.dry_run(map_branch='real', max_envs=10,
                             output_dir=tmp_path / 'real')
    assert summary['selected_branch'] == ['real']
    assert summary['selected_branch_trajectories'] == 75
    assert summary['env_batches'] == 8
    assert summary['run_plan_batches'] == 15
    assert len(summary['batch_table']) == 15


def test_dry_run_main_writes_no_physics(tmp_path):
    code = cli.main(['--dry-run', '--map-branch', 'plane_ablation',
                     '--max-envs', '5', '--output-dir', str(tmp_path)])
    assert code == 0
    doc = json.loads((tmp_path / 'dry_run_summary.json').read_text())
    assert doc['dry_run'] is True
    assert doc['selected_branch'] == ['plane_diagnostic']
    assert doc['selected_branch_trajectories'] == 75


def test_non_dry_run_is_gated():
    with pytest.raises(SystemExit):
        cli.main(['--map-branch', 'both'])


def test_max_envs_above_frozen_batch_bound_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(['--dry-run', '--max-envs', '200', '--output-dir', str(tmp_path)])
    with pytest.raises(ValueError):
        cli.dry_run(max_envs=21, output_dir=tmp_path / 'x')
