"""Explicit one-off D1 adjusted trial identity; Q1 ledger remains the cost authority."""
import json
import os
from pathlib import Path
from . import spec
from .checkpoint import BudgetLedger

CANONICAL_GUARD=spec.ROOT/'logs/a2_v10/pre_gate_adjusted_trial_guard.json'
TRIAL_CAP=2457600
TRIAL_UPDATES=50

class TrialGuard:
    def __init__(self,identity):
        self.identity=identity;self.cap=identity['charged_cap'];self.retained=0

    @classmethod
    def open(cls,*,course,num_envs,stop_round,cap,trial_id,guard_path,ledger_path,resume):
        if (course,num_envs,stop_round,cap)!=('D1',1024,TRIAL_UPDATES,TRIAL_CAP):
            raise ValueError('trial requires D1 / 1024 environments / stop-round 50 / charged-cap 2457600')
        if not isinstance(trial_id,str) or not trial_id.strip() or not guard_path or not ledger_path:
            raise ValueError('trial requires explicit stable trial-id, canonical guard and ledger-path')
        path=Path(guard_path).resolve()
        if path!=Path(CANONICAL_GUARD).resolve():raise ValueError('trial guard must be canonical')
        ledger=BudgetLedger(ledger_path)
        if not resume and sum(ledger.totals().values()):raise ValueError('spent trial ledger requires explicit resume checkpoint')
        identity=dict(schema='a2-v10-adjusted-trial-v1',trial_id=trial_id,guard_path=str(path),ledger_path=str(Path(ledger_path).resolve()),
            course=course,num_envs=num_envs,stop_round=stop_round,charged_cap=cap,sds_sha256=spec.SDS_SHA256)
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:
            if json.loads(path.read_text())!=identity:raise ValueError('trial identity/ledger already bound; fresh quota prohibited')
        else:
            with os.fdopen(fd,'w') as f:
                json.dump(identity,f,sort_keys=True);f.flush();os.fsync(f.fileno())
            directory=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
        return cls(identity)

    def restore(self,extra):
        state=extra.get('trial')
        if not isinstance(state,dict) or state.get('identity')!=self.identity:raise ValueError('matching trial checkpoint required')
        count=state.get('retained_completed_updates')
        if type(count) is not int or not 0<=count<=TRIAL_UPDATES:raise ValueError('invalid retained trial updates')
        self.retained=count

    def can_collect(self,charged):
        return self.retained<TRIAL_UPDATES and charged+1024*spec.ROLLOUT<=self.cap

    @property
    def stop_reason(self):return 'trial_limit' if self.retained==TRIAL_UPDATES else 'trial_budget_exhausted'

    def status(self):
        return dict(pre_gate_adjusted_trial=True,teacher_admitted=False,accepted=None,training_accepted=False,
            retained_completed_updates=self.retained,gate_reached=False)

    def checkpoint_state(self):return dict(identity=self.identity,**self.status())
