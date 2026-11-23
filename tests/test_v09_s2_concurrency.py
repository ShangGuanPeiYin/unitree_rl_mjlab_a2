"""CPU-only concurrency guard tests for gate-cache jobs."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_concurrency


def test_guard_refuses_running_training_even_if_memory_looks_free():
    training = [dict(pid=123, argv=['python', 'scripts/train_a2_v09_s2.py',
                                     '--execute'])]
    with pytest.raises(s2_concurrency.ConcurrencyError, match='training is running'):
        s2_concurrency.require_safe_window(
            processes=training, memory=dict(free_mib=5000, total_mib=8192))


def test_guard_allows_explicit_concurrent_only_with_enough_free_memory():
    training = [dict(pid=123, argv=['python', 'scripts/train_a2_v09_s2.py',
                                     '--execute'])]
    evidence = s2_concurrency.require_safe_window(
        processes=training, allow_concurrent=True,
        memory=dict(free_mib=4096, total_mib=8192))
    assert evidence['concurrent_with_training'] is True
    assert evidence['allow_concurrent'] is True
    assert evidence['required_free_mib'] == 3072
    with pytest.raises(s2_concurrency.ConcurrencyError, match='insufficient free'):
        s2_concurrency.require_safe_window(
            processes=training, allow_concurrent=True,
            memory=dict(free_mib=1385, total_mib=8192))


def test_guard_refuses_even_without_training_when_free_is_insufficient():
    with pytest.raises(s2_concurrency.ConcurrencyError, match='insufficient free'):
        s2_concurrency.require_safe_window(
            processes=[], memory=dict(free_mib=1385, total_mib=8192))


@pytest.mark.parametrize('script', s2_concurrency.CACHE_SCRIPTS)
def test_detector_sees_running_cache_jobs(tmp_path, script):
    proc = tmp_path / 'proc'
    pid = proc / '4321'
    pid.mkdir(parents=True)
    (pid / 'cmdline').write_bytes(
        ('python\0' + 'scripts/' + script + '\0--execute\0').encode())
    found = s2_concurrency.detect_training_processes(
        proc_root=proc, current_pid=9999)
    assert [row['pid'] for row in found] == [4321]


def test_detector_ignores_cache_dry_run(tmp_path):
    proc = tmp_path / 'proc'
    pid = proc / '4321'
    pid.mkdir(parents=True)
    (pid / 'cmdline').write_bytes(
        b'python\0scripts/collect_a2_v09_s2_gate_reference.py\0--dry-run\0')
    assert s2_concurrency.detect_training_processes(
        proc_root=proc, current_pid=9999) == []
