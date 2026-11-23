"""CPU-only contract tests for the S2 real-env smoke CLI."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_concurrency

SPEC = importlib.util.spec_from_file_location(
    'smoke_a2_v09_s2_train', ROOT / 'scripts/smoke_a2_v09_s2_train.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _args(**overrides):
    base = dict(course='S2.1', num_envs=4, device='cuda:0', steps=200,
                role_filter=None, output_dir=None, ledger=None,
                attempt_id='s2.1-v13-smoke', dry_run=True, execute=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dry_run_plan_is_real_env_plan_without_physics(tmp_path):
    out = tmp_path / 'plan'
    code = smoke.main(['--dry-run', '--output-dir', str(out),
                       '--num-envs', '4', '--steps', '200'])
    assert code == 0
    plan = json.loads((out / 'smoke_plan.json').read_text())
    assert plan['schema'] == 'a2_v09_s2_smoke_plan_v1'
    assert plan['physics_started'] is False
    assert plan['training_started'] is False
    assert plan['selected_envs'] == [0, 1, 2, 3]
    assert sum(plan['role_counts'].values()) == 4
    assert len(plan['selected_envs']) == 4


def test_role_filter_selects_only_requested_rows():
    plan = smoke._plan(_args(role_filter='stairs'))
    assert plan['selected_envs'] == [2, 3]
    flat = smoke._plan(_args(role_filter='flat'))
    assert flat['selected_envs'] == [0, 1]
    platform = smoke._plan(_args(num_envs=5, role_filter='platform'))
    assert platform['selected_envs'] == [4]


def test_role_filter_without_matching_env_is_rejected():
    with pytest.raises(SystemExit, match='no env for role filter'):
        smoke._plan(_args(num_envs=4, role_filter='platform'))


def test_execute_rejects_when_cache_job_is_running(tmp_path, monkeypatch):
    monkeypatch.setattr(s2_concurrency, 'detect_training_processes',
                        lambda: [dict(pid=2002833, argv=[
                            'python', 'scripts/collect_a2_v09_s2_gate_original_6400.py',
                            '--execute'])])
    monkeypatch.setattr(s2_concurrency, 'gpu_memory_mib',
                        lambda device='cuda:0': dict(free_mib=6000, total_mib=8192))
    out = tmp_path / 'run'
    with pytest.raises(SystemExit, match='S2.1 training is running'):
        smoke.main(['--execute', '--output-dir', str(out),
                    '--num-envs', '4', '--device', 'cuda:0'])
    assert not out.exists()


def test_execute_rejects_when_free_memory_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(s2_concurrency, 'detect_training_processes', lambda: [])
    monkeypatch.setattr(s2_concurrency, 'gpu_memory_mib',
                        lambda device='cuda:0': dict(free_mib=1000, total_mib=8192))
    with pytest.raises(SystemExit, match='insufficient free GPU memory'):
        smoke.main(['--execute', '--output-dir', str(tmp_path / 'run'),
                    '--device', 'cuda:0'])
