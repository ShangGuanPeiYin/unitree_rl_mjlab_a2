"""S2.0 execution orchestration: reservation gate, batches, evidence, JSONL.

The physics environment is injected as an ``env_factory`` (the existing
v09/mjlab assembly is expected to provide it).  This module owns the parts
that must be independently testable on CPU: frozen batch order, S2 ledger
reserve→step→commit sequencing, terminal latching, evidence writing and the
GPU/device discipline.  It never starts training or updates a weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import secrets
import time

import numpy as np

from . import s2_env, s2_spec
from .cases import canonical_json, sha256

GPU_PEAK_MIB = 7168

def new_attempt_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(3)
SMOKE_DEFAULT_STEPS = 64


class S2PhysicsAssemblyError(RuntimeError):
    """Raised when no real v09/mjlab S2 env assembly has been injected."""


class ReservationError(RuntimeError):
    pass


def batch_table(run_plan, branches=('real', 'plane_diagnostic')):
    rows = []
    for branch in branches:
        branch_rows = [row for row in run_plan['entries']
                       if row['map_branch'] == branch]
        groups = {}
        for row in branch_rows:
            groups.setdefault(row['group_role_sha256'], []).append(row)
        for group_sha, entries in sorted(groups.items()):
            entries = sorted(entries, key=lambda row: row['case_id'])
            max_timeout = max(row['hard_timeout_steps'] for row in entries)
            rows.append(dict(
                branch=branch, group_role_sha256=group_sha,
                task=entries[0]['task'], speed=entries[0]['speed'],
                platform_width=entries[0]['platform_width'],
                case_ids=[row['case_id'] for row in entries],
                case_sha256s=[row['case_sha256'] for row in entries],
                batch_size=len(entries), max_hard_timeout_steps=max_timeout,
                control_samples=len(entries) * max_timeout,
                operation_ids=[row['operation_id'] for row in entries]))
    return rows


def device_report(device):
    import torch
    report = dict(device=str(device), cuda=torch.cuda.is_available())
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested but unavailable')
        index = torch.device(device).index or 0
        free, total = torch.cuda.mem_get_info(index)
        report.update(device_name=torch.cuda.get_device_name(index),
                      total_mib=total / 2 ** 20, free_mib=free / 2 ** 20,
                      peak_limit_mib=GPU_PEAK_MIB)
    return report


class ReservationGate:
    """One stage reservation; each batch reserves then commits actual steps."""

    def __init__(self, ledger, *, obligation_id='s2_0_matrix_two_branches',
                 run_id=None):
        self.ledger = ledger
        self.obligation_id = obligation_id
        self.run_id = run_id

    def require_stage(self):
        state = self.ledger.state()
        if not state['envelope'] or self.obligation_id not in state['obligations']:
            raise ReservationError('S2 stage reservation is missing')
        duty = state['obligations'][self.obligation_id]
        available = duty['limit'] - duty['spent'] - duty['reserved']
        return dict(state=state, obligation=duty, available=available)

    def begin_batch(self, batch, *, operation_id):
        info = self.require_stage()
        cost = int(batch['control_samples'])
        if info['available'] < cost:
            raise ReservationError('insufficient obligation balance for batch')
        receipt = self.ledger.reserve(
            self.obligation_id, cost,
            operation_id=f'{operation_id}:reserve',
            reason=f"S2.0 batch {batch['branch']} {batch['group_role_sha256'][:12]}")
        return dict(receipt=receipt, reserved=cost)

    def finish_batch(self, batch, *, operation_id, actual_samples):
        actual = int(actual_samples)
        reserved = int(batch['control_samples'])
        if actual < 0 or actual > reserved:
            raise ReservationError('actual samples outside the batch upper bound')
        # §2.5: bill actual advanced samples; release the unused reservation.
        if actual:
            self.ledger.commit(
                self.obligation_id, actual,
                operation_id=f'{operation_id}:commit',
                reason=f"S2.0 batch actual {batch['branch']}")
        unused = reserved - actual
        if unused:
            self.ledger.release(
                self.obligation_id, unused,
                operation_id=f'{operation_id}:release',
                reason=f"S2.0 batch unused reservation {batch['branch']}")
        return dict(committed=actual, released=unused, actual_samples=actual,
                    waiting_padding=0)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite evidence {path}')
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(canonical_json(row) + '\n')
    return sha256([row for row in rows])


class S2ExecutionDriver:
    """Generic batch driver; the env factory provides real physics."""

    def __init__(self, *, env_factory, policy, ledger, output_dir,
                 device='cuda:0', reservation_gate=None, max_steps=None,
                 attempt_id=None):
        if not callable(env_factory) or not callable(policy):
            raise ValueError('env_factory and policy must be callable')
        self.env_factory = env_factory
        self.policy = policy
        self.ledger = ledger
        self.output_dir = Path(output_dir)
        self.device = device
        self.gate = reservation_gate or ReservationGate(ledger)
        self.max_steps = max_steps
        self.attempt_id = attempt_id or new_attempt_id()
        if not self.attempt_id or any(ch in self.attempt_id for ch in ': /'):
            raise ValueError('attempt_id must be nonempty without : or /')

    def run_batch(self, batch, cases, *, branch, operation_id):
        self.gate.require_stage()
        started = time.time()
        reserved = self.gate.begin_batch(batch, operation_id=operation_id)
        completed = 0
        steps = 0
        records, traces, terminals = [], {}, {}
        try:
            env = self.env_factory(cases, device=self.device, branch=branch)
            steps = 0
            while not terminals and (self.max_steps is None or steps < self.max_steps):
                obs = env.observation()
                actions = self.policy(obs)
                obs, done, info = env.step(actions)
                steps += 1
                for index, finished in enumerate(done):
                    if finished and not terminals.get(index, False):
                        terminals[index] = True
                        trace = env.evidence(index)
                        adapter = env.adapter(index)
                        metrics = s2_env.case_record(
                            case=cases[index], adapter=adapter,
                            recorder=env.recorder(index),
                            metrics=getattr(env, 'metrics', lambda index: None)(index),
                            map_evidence=trace.pop('map_evidence', None),
                            state_summary=trace.pop('state_summary', None),
                            failure_context=trace.pop('failure_context', None),
                            g_level_report=trace.pop('g_level_report', None))
                        metrics['trace_rows'] = len(trace.get('step', []))
                        metrics['attempt_id'] = self.attempt_id
                        records.append(metrics)
                        traces[index] = trace
                if all(terminals.get(i, False) for i in range(len(cases))):
                    break
            if not all(terminals.get(i, False) for i in range(len(cases))):
                raise RuntimeError('batch did not terminate every case before max steps')
            completed = len(cases) * steps
            settled = self.gate.finish_batch(batch, operation_id=operation_id,
                                             actual_samples=completed)
            return dict(batch=batch, records=records, traces=traces,
                        control_steps=steps, actual_samples=completed,
                        reserved=reserved, settled=settled,
                        elapsed_s=time.time() - started)
        except Exception:
            # Bill only the actually advanced controls and release the rest,
            # even when the batch fails mid-way.
            advanced = len(cases) * steps
            try:
                self.gate.finish_batch(batch, operation_id=operation_id,
                                       actual_samples=advanced)
            except Exception:
                pass
            raise

    def run(self, batches, case_index):
        rows = []
        for sequence, batch in enumerate(batches):
            cases = [case_index[sha] for sha in batch['case_sha256s']]
            operation_id = (f"{batch['branch']}:{batch['group_role_sha256']}:"
                            f"{sequence:02d}:{self.attempt_id}")
            rows.append(self.run_batch(batch, cases, branch=batch['branch'],
                                       operation_id=operation_id))
        return rows


def reservation_preflight(ledger, *, obligation_id='s2_0_matrix_two_branches'):
    state = ledger.state()
    if not state['envelope']:
        raise ReservationError('no S2 stage reservation')
    duty = state['obligations'].get(obligation_id)
    if duty is None:
        raise ReservationError('obligation missing')
    return dict(envelope=state['envelope'], limit=duty['limit'],
                spent=duty['spent'], reserved=duty['reserved'],
                available=duty['limit'] - duty['spent'] - duty['reserved'])
