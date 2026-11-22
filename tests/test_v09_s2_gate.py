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
    out2 = tmp_path / 'l40'
    assert cli.main([
        '--reference', 's1_l40', '--suite', 'gate_A', '--dry-run',
        '--manifest', str(manifest_path), '--output-dir', str(out2)]) == 0
    plan2 = json.loads((out2 / 's1_l40_gate_A_plan.json').read_text())
    assert plan2['control_samples'] == 219_220
    assert plan2['obligation_id'] == 'l40_milestone_cache_gate_A'
    with pytest.raises(SystemExit, match='DSH-Q034'):
        cli.main([
            '--reference', '6400_flat', '--suite', 'gate_A', '--dry-run',
            '--manifest', str(manifest_path), '--output-dir', str(tmp_path / 'mig')])


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
    assert plan['envs_per_batch'] == 1
    assert plan['estimated_seconds'] == 4.7 * 3600
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
    assert plan['reference'] == 'original_6400_flat'


def test_t14_cache_builder_refuses_unknown_mode(tmp_path):
    import build_a2_v09_s2_gate_caches as cli
    with pytest.raises(SystemExit):
        cli.main(['--suite', 'gate_A', '--cache-kind', 'l40',
                  '--output-dir', str(tmp_path)])


def test_gate_rejects_migrated_or_wrong_source_6400_cache(tmp_path):
    import run_a2_v09_s2_gate as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    groups = {row['group_id']: {'metrics': {}} for row in manifest['groups']
              if row['group_id'].startswith('flat/')}
    base = dict(schema='a2_v09_s2_gate_6400_flat_v1',
                suite='S2_GATE_A', manifest_sha256=manifest['manifest_sha256'],
                groups=groups, source_sha256='30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2',
                model_type='original_6400', observation_shape=[3, 48, 32],
                migrated=False)
    path = tmp_path / 'cache.json'
    path.write_text(json.dumps(base))
    assert cli._load_flat_cache(path, manifest) == groups
    for field, value in (('source_sha256', '0' * 64),
                         ('model_type', 'migrated_6400'),
                         ('observation_shape', [3, 40, 30]),
                         ('migrated', True)):
        bad = dict(base, **{field: value})
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            cli._load_flat_cache(path, manifest)


def test_q034a_original_6400_dry_run_and_old_map_contract(tmp_path):
    import collect_a2_v09_s2_gate_original_6400 as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    manifest_path = tmp_path / 'S2_GATE_A.json'
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / '6400'
    assert cli.main([
        '--suite', 'gate_A', '--dry-run',
        '--manifest', str(manifest_path), '--output-dir', str(out)]) == 0
    plan = json.loads((out / '6400_gate_plan.json').read_text())
    assert plan['groups'] == 9
    assert plan['cases'] == 180
    assert plan['control_samples'] == 234_180
    assert plan['model_type'] == 'original_6400'
    assert plan['observation_shape'] == [3, 48, 32]
    assert plan['source_sha256'] == (
        '30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2')
    assert plan['envs_per_batch'] == 20
    assert plan['physics_started'] is False


def test_q034a_teacher_old_map_shape_guard_cpu():
    import torch
    import collect_a2_v09_s2_gate_original_6400 as cli
    good = {'heightmap': torch.zeros(20, 3, 48, 32)}
    assert cli.validate_teacher_observation(good, batch=20) == (20, 3, 48, 32)
    with pytest.raises(ValueError, match='3,48,32'):
        cli.validate_teacher_observation(
            {'heightmap': torch.zeros(20, 3, 40, 30)}, batch=20)


def test_gate_validates_l40_cache_source_fields_and_percentiles(tmp_path):
    import run_a2_v09_s2_gate as cli
    manifest = s2_gate_cases.build_gate_manifest('gate_A')
    names = [row['group_id'] for row in manifest['groups']
             if row['task'] == 'legacy_l40']
    case_files = []
    for name in names:
        for index in range(20):
            sha = f'{name}-{index}'.replace('/', '_')
            case = dict(case_sha256=sha, group=name, start={'x': -1.0},
                        last_valid_terminal={'x': 1.0}, net_distance=2.0,
                        success=True, failure_class=None, timeout=False,
                        engineering_failure=False)
            path = tmp_path / f'{sha}.json'
            path.write_text(json.dumps(case))
            case_files.append(dict(case_sha256=sha, path=str(path), sha256=None))
    groups = {name: dict(p10=0.0, p50=1.0, p90=2.0) for name in names}
    doc = dict(schema='a2_v09_s2_gate_l40_cache_v1', suite='S2_GATE_A',
               manifest_sha256=manifest['manifest_sha256'],
               source_sha256='efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24',
               groups=groups, case_files=case_files)
    path = tmp_path / 'l40.json'
    path.write_text(json.dumps(doc))
    assert cli._load_l40_cache(path, manifest) == groups
    bad = json.loads(json.dumps(doc))
    bad['source_sha256'] = '0' * 64
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='s1_least'):
        cli._load_l40_cache(path, manifest)
    bad = json.loads(json.dumps(doc))
    del bad['groups'][names[0]]['p50']
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='p50'):
        cli._load_l40_cache(path, manifest)


def test_l40_serial_helpers_post_reset_settlement_and_group_stats():
    import collect_a2_v09_s2_gate_reference as cli
    trace = dict(route_x=[0.0, 1.0, 2.0, 99.0],
                 post_reset=[False, False, False, True])
    last = cli.last_valid_terminal(trace, start_x=0.0)
    assert last == {'x': 2.0, 'index': 2, 'post_reset': False}
    rows = [dict(group='l40/0.3', steps=100, net_distance=1.0, success=True,
                 failure=False, timeout=False, engineering_failure=False,
                 termination_reason=None) for _ in range(20)]
    settlement = cli.suite_settlement(rows)
    assert settlement['actual'] == 2000
    assert settlement['actual'] <= 219_220
    assert settlement['unused'] == 217_220
    groups = [dict(group_id='l40/0.3')]
    stats = cli.group_statistics(groups, rows)
    assert stats['l40/0.3']['cases'] == 20
    assert stats['l40/0.3']['successes'] == 20
    assert stats['l40/0.3']['p10'] == 1.0
    assert stats['l40/0.3']['p50'] == 1.0
    assert stats['l40/0.3']['p90'] == 1.0
