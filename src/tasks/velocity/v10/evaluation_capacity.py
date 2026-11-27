"""Exclusive stage capacity, separate from nonrefundable per-step evaluation costs."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
from .checkpoint import BudgetLedger, atomic_json


@contextlib.contextmanager
def evaluation_capacity(ledger_path, stage_id, required, limit):
    """Hold one parent-owned stage reservation; children debit only the Q1 ledger.

    A hard-crash stale lease is intentionally fail-closed. It requires verification
    that the old worker is gone before explicit recovery, rather than silently
    releasing capacity while an orphan may still be executing.
    """
    if not stage_id or not isinstance(required,int) or required<=0:
        raise ValueError('positive stage capacity and stable identity required')
    path=Path(ledger_path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    lock_path=Path(str(path)+'.capacity.lock');lease_path=Path(str(path)+'.capacity.json')
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('evaluation capacity busy') from None
        prior=json.loads(lease_path.read_text()) if lease_path.exists() else {}
        if prior.get('status')=='active':
            raise RuntimeError('evaluation capacity stale active lease requires verified recovery')
        ledger=BudgetLedger(path);before=ledger.report()
        if before['pending']:raise RuntimeError('evaluation pending step requires verified recovery')
        charged=before['charged'].get('evaluation:ALL',0);remaining=limit-charged
        if remaining<required:
            raise RuntimeError(f'evaluation capacity insufficient required={required} remaining={remaining} shortfall={required-remaining}')
        lease=dict(schema='a2-v10-evaluation-capacity-v1',stage_id=str(stage_id),owner_pid=os.getpid(),
                   required=required,limit=limit,charged_before=charged,status='active',charges_are_separate=True)
        atomic_json(lease_path,lease)
        try:yield dict(lease)
        finally:
            after=ledger.report()['charged'].get('evaluation:ALL',0)
            atomic_json(lease_path,dict(lease,status='released',charged_after=after,
                used_capacity=after-charged,released_unused_capacity=max(0,required-(after-charged))))
