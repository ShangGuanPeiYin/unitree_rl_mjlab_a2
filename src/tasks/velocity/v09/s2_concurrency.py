"""Concurrency guard for gate-cache GPU jobs.

Gate caches must not accidentally overlap the long S2.1 training run.  The
guard detects a live training launcher from /proc and checks free GPU memory
before any ledger reserve or environment allocation.
"""
from __future__ import annotations

import os
from pathlib import Path

ESTIMATE_20_ENV_MIB = 2048
SAFETY_MARGIN_MIB = 1024
TRAIN_SCRIPT = 'train_a2_v09_s2.py'


class ConcurrencyError(RuntimeError):
    pass


def detect_training_processes(*, proc_root='/proc', current_pid=None):
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    found = []
    root = Path(proc_root)
    if not root.exists():
        return found
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            raw = (entry / 'cmdline').read_bytes()
        except Exception:
            continue
        argv = [part.decode(errors='replace') for part in raw.split(b'\0') if part]
        if not argv:
            continue
        joined = ' '.join(argv)
        if TRAIN_SCRIPT in joined and '--execute' in joined:
            found.append(dict(pid=pid, argv=argv))
    return found


def gpu_memory_mib(device='cuda:0'):
    import torch
    if not torch.cuda.is_available():
        raise ConcurrencyError('CUDA unavailable for gate cache')
    index = torch.device(device).index or 0
    free, total = torch.cuda.mem_get_info(index)
    return dict(free_mib=free / 2 ** 20, total_mib=total / 2 ** 20)


def require_safe_window(*, device='cuda:0', allow_concurrent=False,
                        estimate_mib=ESTIMATE_20_ENV_MIB,
                        safety_mib=SAFETY_MARGIN_MIB,
                        processes=None, memory=None):
    """Return evidence or raise; no allocation/ledger write occurs here."""
    training = detect_training_processes() if processes is None else list(processes)
    mem = gpu_memory_mib(device) if memory is None else dict(memory)
    required = float(estimate_mib) + float(safety_mib)
    evidence = dict(
        training_processes=training,
        concurrent_with_training=bool(training),
        allow_concurrent=bool(allow_concurrent),
        gpu_free_mib=float(mem['free_mib']),
        gpu_total_mib=float(mem['total_mib']),
        estimate_mib=float(estimate_mib),
        safety_margin_mib=float(safety_mib),
        required_free_mib=required)
    if training and not allow_concurrent:
        raise ConcurrencyError(
            'S2.1 training is running (PIDs '
            + ','.join(str(row['pid']) for row in training)
            + '); wait for training window or pass --allow-concurrent '
              'only if the measured free memory is sufficient')
    if float(mem['free_mib']) < required:
        raise ConcurrencyError(
            f'insufficient free GPU memory: {mem["free_mib"]:.1f} MiB < '
            f'{required:.1f} MiB (20 env estimate + safety); wait for '
            'training window')
    return evidence
