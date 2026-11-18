"""Frozen independent v0.8 cases and honest complete-denominator evaluation.

Successful claims are checked against actual stop windows and movement evidence.
A missing sample is engineering data loss, never a fabricated policy fall.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import numpy as np
import torch
from . import spec
from .curriculum import spawn_bounds, role_case, SCRIPTS

class EvaluationError(ValueError): pass

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def _digest(value): return hashlib.sha256(_canonical(value).encode()).hexdigest()

def _seed(seed, stream, key):
    return int.from_bytes(hashlib.sha256(f'v08:{int(seed)}:{stream}:{key}'.encode()).digest()[:8],'big')&((1<<63)-1)

@dataclass(frozen=True)
class CaseTable:
    kind: str
    phase: str
    speeds: tuple
    cases: tuple
    seed: int = 42
    substage: object = None
    schema: str = 'v08-frozen-cases-v1'
    @property
    def sha256(self): return case_table_sha(self)
    def to_json(self):
        return dict(schema=self.schema,kind=self.kind,phase=self.phase,substage=self.substage,
                    seed=self.seed,speeds=list(self.speeds),cases=deepcopy(list(self.cases)))
    def as_dict(self): return self.to_json()

def case_table_sha(table): return _digest(table.to_json() if isinstance(table,CaseTable) else table)

def _roles_for(phase,kind):
    if kind=='baseline': return ('flat','up_L40','down_D20','down_L40')
    if kind=='h00': return ('down_H00',) if phase in ('C','D') else ('up_H00',)
    if kind=='final' or phase=='E': return ('flat','up_L40','down_L40','E1','E2','E3','E4')
    return ('flat','up_L40')+(('down_D20',) if phase=='C' else ('down_L40',) if phase=='D' else ())

def make_cases(phase='A', substage=None, kind='gate', seed=42, **kwargs):
    phase=str(phase).upper();kind=str(kind).lower()
    if phase not in spec.PHASE_ORDER or kind not in ('baseline','gate','h00','final'):
        raise EvaluationError('unknown case phase/kind')
    if kwargs: raise EvaluationError('unexpected case options: '+','.join(kwargs))
    if substage is not None: raise EvaluationError('v08 has no substages')
    if kind=='final': phase='E'
    stream=f'{kind}:{phase}'
    count=100 if kind=='final' else 5 if kind in ('baseline','h00') else 20
    bounds=spawn_bounds('B' if kind=='final' else phase,evaluation=True)
    cases=[]
    for role in _roles_for(phase,kind):
        for speed in spec.EVAL_SPEEDS:
            for i in range(count):
                row=role_case(role,i)
                key=f'{role}:{speed}:{i}'
                case_seed=_seed(seed,stream,key)
                rng=np.random.default_rng(case_seed)
                x=row['spawn_x'] if row['spawn_mode']=='top' else float(rng.uniform(*bounds['x']))
                y=float(rng.uniform(*bounds['y'])); yaw=float(rng.uniform(*bounds['yaw']))
                trigger=None
                if row['task'] in spec.TRIGGER_RANGES:
                    lo,hi=spec.TRIGGER_RANGES[row['task']];trigger=(lo,(lo+hi)/2,hi)[i%3]
                row.update(case_id=f'{kind}-{phase}-{role}-{speed:.1f}-{i:03d}', seed=case_seed,stream=stream,
                           speed=float(speed),vx=float(speed),cruise_speed=float(speed),
                           command=0. if row['spawn_mode']=='top' else float(speed),
                           x=x,y=y,yaw=yaw,spawn_x=x,spawn_y=y,spawn_yaw=yaw,
                           spawn_x_bounds=(x,x) if row['spawn_mode']=='top' else bounds['x'],
                           spawn_y_bounds=bounds['y'],spawn_yaw_bounds=bounds['yaw'],trigger_p=trigger,
                           deterministic=True,training=False)
                cases.append(row)
    return CaseTable(kind,phase,tuple(spec.EVAL_SPEEDS),tuple(cases),int(seed))

def table_from_json(payload, expected_sha=None):
    try:
        table=CaseTable(payload['kind'],payload['phase'],tuple(payload['speeds']),tuple(deepcopy(payload['cases'])),payload['seed'],payload['substage'],payload['schema'])
        canonical=make_cases(table.phase,kind=table.kind,seed=table.seed)
        if table.to_json()!=json.loads(_canonical(canonical.to_json())) and _canonical(table.to_json())!=_canonical(canonical.to_json()):
            raise EvaluationError('case table differs from frozen recipe')
        if expected_sha is not None and table.sha256!=expected_sha: raise EvaluationError('case SHA mismatch')
        return table
    except (KeyError,TypeError) as error: raise EvaluationError('invalid case table') from error

def _finite(value):
    if isinstance(value,dict): return all(_finite(x) for x in value.values())
    if isinstance(value,(list,tuple)): return all(_finite(x) for x in value)
    if isinstance(value,(float,int,np.number)) and not isinstance(value,bool): return bool(np.isfinite(value))
    return True

def _stop_window(window):
    kind=window['kind'];mid=kind=='mid'
    needed=spec.MID_WINDOW_STEPS if mid else spec.ENDPOINT_WINDOW_STEPS
    hold=spec.MID_HOLD_STEPS if mid else spec.ENDPOINT_HOLD_STEPS
    xy=np.asarray(window['xy'],dtype=float);values=np.asarray(window['values'],dtype=float)
    if kind not in ('flat','top','bottom','mid') or xy.shape!=(needed,2) or values.shape!=(needed,10):
        raise EvaluationError('raw window has wrong shape or location')
    if type(window['count']) is not int or window['count']<0: raise EvaluationError('invalid zero-hold control count')
    if int(window['count'])<hold: return False,'incomplete_hold'
    if not np.isfinite(xy).all() or not np.isfinite(values).all(): raise EvaluationError('nonfinite window')
    if (values[:,[0,4,6]]<0).any() or not np.isin(values[:,7:10],(0.,1.)).all():
        raise EvaluationError('invalid raw norm or support domain')
    checks=(('mean_speed',values[:,0].mean()<=spec.STAND_MEAN_SPEED),
            ('max_speed',values[:,0].max()<=spec.STAND_MAX_SPEED),
            ('roll',np.abs(values[:,1]).max()<=(spec.MID_ROLL if mid else spec.LEVEL_ROLL)),
            ('pitch',np.abs(values[:,2]).max()<=(spec.MID_PITCH if mid else spec.LEVEL_PITCH)),
            ('omega',values[:,4].max()<=spec.STAND_OMEGA),
            ('height',values[:,5].min()>=(spec.MID_HEIGHT if mid else spec.LEVEL_HEIGHT)),
            ('foot_speed',values[:,6].mean()<=spec.STAND_FOOT_SPEED),
            ('support',values[:,7].mean()>=spec.STAND_FOUR_FRACTION),
            ('drift',np.linalg.norm(xy[-1]-xy[0])<=spec.STAND_DRIFT),
            ('surface',np.all(values[:,8]==1)),
            ('valid',np.all(values[:,9]==1)))
    if mid: checks+= (('yaw',np.abs(values[:,3]).max()<=spec.MID_YAW),)
    for reason,passed in checks:
        if not passed: return False,('mid_stop_' if mid else 'endpoint_stop_')+reason
    return True,'success'

def _required_windows(case):
    task=case['task']
    if task=='flat_stop': return ['flat']
    if task=='up_stop': return ['top']
    if task in ('down_stop','E1'): return ['bottom']
    if task=='E2': return ['top']
    if task=='E4': return ['top','bottom']
    return ['mid','top' if case['resume_direction']>0 else 'bottom']

def needs_direction_check(case):
    return case['task'] in ('E1','E2','E4') or (case['task']=='E3' and case['resume_direction']!=case['initial_direction'])

def normalize_record(raw,case):
    row=dict(case_id=case['case_id'],task=case['task'],script=case['script'],speed=case['speed'],
             status='invalid',success=False,reason='missing_raw',segment='unknown')
    if not isinstance(raw,dict): return row
    # Store complete raw data for audit; nonfinite payloads become JSON-safe errors.
    if not _finite(raw): row['reason']='nonfinite_raw';return row
    row['raw']=deepcopy(raw);row['segment']=str(raw.get('segment','unknown'))
    if raw.get('exception') is not None or raw.get('status')=='error':
        row.update(status='error',reason='engineering_exception');return row
    if raw.get('case_id')!=case['case_id']: row['reason']='case_id_mismatch';return row
    if raw.get('complete') is not True: row['reason']='incomplete_episode';return row
    if raw.get('raw_complete') is not True: row['reason']='missing_raw';return row
    if raw.get('finite') is not True: row['reason']='nonfinite_raw';return row
    if not isinstance(raw.get('success'),bool) or not raw.get('reason') or not raw.get('segment'):
        row['reason']='missing_terminal_fields';return row
    # A true physical failure always wins, including an erroneous simultaneous success.
    if raw.get('physical_failure') is True or raw.get('failure_kind')=='physical':
        row.update(status='failure',reason=raw['reason'],physical_failure=True);return row
    if not raw['success']:
        row.update(status='failure',reason=raw['reason'],physical_failure=False);return row
    try:
        windows=raw['stop_windows']
        if not isinstance(windows,list) or [x['kind'] for x in windows]!=_required_windows(case):
            raise EvaluationError('missing or reordered required stop windows')
        for window in windows:
            passed,reason=_stop_window(window)
            if not passed: row.update(status='failure',reason=reason);return row
        if case['task']=='flat_stop':
            p=raw['flat_progress']
            if int(p['walk_steps'])!=spec.FLAT_WALK_STEPS: raise EvaluationError('flat walk evidence incomplete')
            if p['net_dx']<spec.FLAT_PROGRESS_FACTOR*case['speed']*spec.FLAT_WALK_SECONDS:
                row.update(status='failure',reason='insufficient_flat_progress');return row
            if p['max_abs_y']>.20 or p['max_abs_yaw']>math.radians(10):
                row.update(status='failure',reason='flat_walk_route');return row
        if case['task'] in ('down_stop','E1','E2','E3','E4'):
            if raw['max_stair_abs_yaw']>spec.REVERSE_YAW:
                row.update(status='failure',reason='stair_yaw');return row
        if needs_direction_check(case):
            checks=raw['direction_checks']
            if not isinstance(checks,list) or len(checks)!=1: raise EvaluationError('direction evidence incomplete')
            d=checks[0];direction= -1 if case['task'] in ('E1','E4') else 1 if case['task']=='E2' else case['resume_direction']
            if d['direction']!=direction or d['net_progress']<spec.REVERSE_PROGRESS:
                row.update(status='failure',reason='wrong_direction');return row
            clocks=[d[k] for k in ('request_step','cruise_start_step','completion_step','elapsed_steps')]
            if any(type(v) is not int or v<0 for v in clocks): raise EvaluationError('direction clocks must be integer control boundaries')
            if d['cruise_start_step']!=d['request_step']+spec.TRANSITION_STEPS+1 or d['completion_step']<d['cruise_start_step'] or d['elapsed_steps']!=d['completion_step']-d['request_step']:
                raise EvaluationError('direction evidence is not first cruise after this request')
            if not 0<=d['elapsed_steps']<=spec.REVERSE_PROGRESS_STEPS:
                row.update(status='failure',reason='direction_timeout');return row
        row.update(status='success',success=True,reason='success')
    except (KeyError,TypeError,ValueError,IndexError) as error:
        row.update(status='invalid',success=False,reason='invalid_success_evidence',detail=str(error))
    return row

def _group(rows,expected,threshold):
    counts=Counter(x['status'] for x in rows)
    success=counts['success'];valid=counts['success']+counts['failure']
    return dict(expected=expected,received=len(rows),valid=valid,invalid=counts['invalid']+max(expected-len(rows),0),
                error=counts['error'],success=success,failure=counts['failure'],threshold=threshold,
                passed=len(rows)==expected and valid==expected and success>=threshold,
                reasons=dict(Counter(x['reason'] for x in rows)),
                segments=dict(Counter(x.get('segment','unknown') for x in rows)))

def summarize(records,table):
    expected={c['case_id']:c for c in table.cases};by_id={};duplicates=[];unknown=[]
    for row in records:
        if not isinstance(row,dict) or row.get('status') not in ('success','failure','invalid','error'):
            raise EvaluationError('unknown normalized record status')
        if row.get('success') is not (row['status']=='success'):
            raise EvaluationError('status and success flag disagree')
        cid=row.get('case_id')
        if cid not in expected: unknown.append(cid);continue
        if cid in by_id: duplicates.append(cid);continue
        by_id[cid]=row
    groups={};scripts={}
    for task in dict.fromkeys(c['task'] for c in table.cases):
        for speed in table.speeds:
            cases=[c for c in table.cases if c['task']==task and c['speed']==speed]
            if not cases: continue
            rows=[by_id[c['case_id']] for c in cases if c['case_id'] in by_id]
            threshold=95 if table.kind=='final' else 18 if table.kind=='gate' else len(cases)
            groups[f'{task}@{speed:.1f}']=_group(rows,len(cases),threshold)
    for script in SCRIPTS:
        for speed in table.speeds:
            cases=[c for c in table.cases if c['task']=='E3' and c['script']==script and c['speed']==speed]
            if cases:
                rows=[by_id[c['case_id']] for c in cases if c['case_id'] in by_id]
                scripts[f'{script}@{speed:.1f}']=_group(rows,len(cases),23 if table.kind=='final' else 0)
    allrows=list(by_id.values());summary=_group(allrows,len(expected),0)
    engineering=bool(duplicates or unknown or summary['invalid'] or summary['error'])
    passed=all(g['passed'] for g in groups.values()) and not engineering
    if table.kind=='final':passed=passed and all(g['passed'] for g in scripts.values())
    summary.update(kind=table.kind,phase=table.phase,case_table_sha256=table.sha256,groups=groups,e3_subgates=scripts,
                   physical_failure=sum(r.get('physical_failure',False) for r in allrows),
                   gate_pass=passed if table.kind in ('gate','final') else False,
                   report_only=table.kind=='h00',engineering_error=engineering,
                   duplicate_ids=duplicates,unknown_ids=unknown,full_denominator=len(expected))
    return summary

def gate_result(summary):
    return deepcopy(summary.summary if hasattr(summary,'summary') else summary)

@dataclass
class EvalResult:
    table: CaseTable
    summary: dict
    records: list
    isolation: dict
    evidence_sha256: dict = field(default_factory=dict)

def _state_digest(value):
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();digest.update(str((t.dtype,tuple(t.shape))).encode());digest.update(t.numpy().tobytes())
        elif isinstance(x,np.ndarray):digest.update(str((x.dtype,x.shape)).encode());digest.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):digest.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for y in x:visit(y)
        else:digest.update(repr(x).encode())
    visit(value);return digest.hexdigest()

def evaluate_cases(model,table,case_runner,*,optimizer=None,counters=None,device='cpu'):
    # Table validation happens before any external runner can observe it.
    table_from_json(table.to_json(),expected_sha=table.sha256)
    rng=(random.getstate(),deepcopy(np.random.get_state()),torch.get_rng_state().clone(),
         torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    snapshot=dict(model=deepcopy(model.state_dict()) if model is not None else None,
                  optimizer=deepcopy(optimizer.state_dict()) if optimizer is not None else None,
                  counters=deepcopy(counters))
    before=_state_digest(snapshot);mode=model.training if model is not None else None
    records=[];changed=[]
    try:
        if model is not None:model.eval()
        with torch.no_grad():
            if getattr(case_runner,'supports_batch',False):
                for i in range(0,len(table.cases),8):
                    batch=deepcopy(list(table.cases[i:i+8]))
                    try:
                        raw=case_runner(batch)
                        if not isinstance(raw,(list,tuple)) or len(raw)!=len(batch):raise EvaluationError('batch runner omitted cases')
                        records.extend(normalize_record(r,c) for r,c in zip(raw,batch))
                    except Exception as error:
                        records.extend(normalize_record(dict(exception=f'{type(error).__name__}: {error}'),c) for c in batch)
            else:
                for case in table.cases:
                    try:raw=case_runner(deepcopy(case))
                    except Exception as error:raw=dict(exception=f'{type(error).__name__}: {error}')
                    records.append(normalize_record(raw,case))
    finally:
        current=dict(model=model.state_dict() if model is not None else None,
                     optimizer=optimizer.state_dict() if optimizer is not None else None,counters=counters)
        for key in snapshot:
            if _state_digest(snapshot[key])!=_state_digest(current[key]):changed.append(key)
        if model is not None:
            if 'model' in changed:model.load_state_dict(snapshot['model'])
            model.train(mode)
        if optimizer is not None and 'optimizer' in changed:optimizer.load_state_dict(snapshot['optimizer'])
        if isinstance(counters,dict) and 'counters' in changed:counters.clear();counters.update(snapshot['counters'])
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3] is not None:torch.cuda.set_rng_state_all(rng[3])
    if changed:raise EvaluationError('evaluation mutated training state: '+','.join(changed))
    isolation=dict(statistics_unchanged=True,statistics_changed=[],snapshot_sha=before)
    summary=summarize(records,table);summary.update(isolation)
    evidence=dict(case_table_sha256=table.sha256,records_sha256=_digest(records),summary_sha256=_digest(summary))
    return EvalResult(table,summary,records,isolation,evidence)
