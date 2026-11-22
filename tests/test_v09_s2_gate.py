"""CPU-only tests for the S2.1 gate CLI (no physics, no ledger write)."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_gate_cases


def test_gate_cli_dry_run_reads_frozen_manifest(tmp_path):
    import run_a2_v09_s2_gate as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    manifest_path = tmp_path / 'S2_GATE_A.json'
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / 'fake.pt'
    checkpoint.write_bytes(b'not-loaded-in-dry-run')
    out = tmp_path / 'plan'
    assert cli.main([
        '--course', 'S2.1', '--suite', 'gate_A', '--dry-run',
        '--manifest', str(manifest_path), '--checkpoint', str(checkpoint),
        '--output-dir', str(out)]) == 0
    plan = json.loads((out / 'S2.1_gate_plan.json').read_text())
    assert plan['groups'] == 36
    assert plan['cases'] == 720
    assert plan['control_samples'] == 1_620_000
    assert plan['l40_cache_samples'] == 219_220
    assert plan['manifest_sha256'] == manifest['manifest_sha256']
    assert plan['physics_started'] is False
    assert plan['ledger_written'] is False


def test_gate_cli_refuses_without_execute_or_dry_run(tmp_path):
    import run_a2_v09_s2_gate as cli
    with pytest.raises(SystemExit):
        cli.main(['--suite', 'gate_A', '--checkpoint', str(tmp_path / 'x')])


def test_gate_cli_refuses_missing_checkpoint(tmp_path):
    import run_a2_v09_s2_gate as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    manifest_path = tmp_path / 'S2_GATE_A.json'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match='checkpoint missing'):
        cli.main(['--suite', 'gate_A', '--dry-run',
                  '--manifest', str(manifest_path),
                  '--checkpoint', str(tmp_path / 'missing.pt')])


def test_reference_collector_dry_run_budgets(tmp_path):
    import collect_a2_v09_s2_gate_reference as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    manifest_path = tmp_path / 'S2_GATE_A.json'
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / 'flat'
    assert cli.main([
        '--reference', '6400_flat', '--suite', 'gate_A', '--dry-run',
        '--manifest', str(manifest_path), '--output-dir', str(out)]) == 0
    plan = json.loads((out / '6400_flat_gate_A_plan.json').read_text())
    assert plan['control_samples'] == 234_180
    out2 = tmp_path / 'l40'
    assert cli.main([
        '--reference', 's1_l40', '--suite', 'gate_A', '--dry-run',
        '--manifest', str(manifest_path), '--output-dir', str(out2)]) == 0
    plan2 = json.loads((out2 / 's1_l40_gate_A_plan.json').read_text())
    assert plan2['control_samples'] == 219_220
    assert plan2['obligation_id'] == 'l40_milestone_cache_gate_A'


def test_t14_cache_builder_dry_run_and_source_model(tmp_path):
    import build_a2_v09_s2_gate_caches as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    manifest_path = tmp_path / 'S2_GATE_A.json'
    manifest_path.write_text(json.dumps(manifest))
    out_l40 = tmp_path / 'l40'
    assert cli.main([
        '--suite', 'gate_A', '--cache-kind', 'l40', '--dry-run',
        '--manifest', str(manifest_path), '--output-dir', str(out_l40)]) == 0
    plan = json.loads((out_l40 / 'l40_gate_A_plan.json').read_text())
    assert plan['control_samples'] == 219_220
    assert plan['source_sha256'] == 'efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24'
    assert plan['reference'] == 's1_l40'
    out_6400 = tmp_path / '6400'
    assert cli.main([
        '--suite', 'gate_A', '--cache-kind', '6400', '--dry-run',
        '--manifest', str(manifest_path), '--output-dir', str(out_6400)]) == 0
    plan = json.loads((out_6400 / '6400_gate_A_plan.json').read_text())
    assert plan['control_samples'] == 234_180
    assert plan['source_sha256'] == (
        '30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2')
    assert plan['reference'] == '6400_flat'


def test_t14_cache_builder_refuses_unknown_mode(tmp_path):
    import build_a2_v09_s2_gate_caches as cli
    with pytest.raises(SystemExit):
        cli.main(['--suite', 'gate_A', '--cache-kind', 'l40',
                  '--output-dir', str(tmp_path)])
