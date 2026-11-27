"""Explicit state-preserving trial-to-formal handoff; no acceptance is fabricated."""
import json
import math
import os
import tempfile
from pathlib import Path
from . import cases,spec,trial_guard
from .checkpoint import sha256,BudgetLedger,SCHEMA
from .identity import code_fingerprint
from .teachers import teacher_identity
from .source_compatibility import validate_compatibility,validate_compatible_artifact,validate_compatible_resource

RULINGS=spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'
PLAN_SCHEMA='a2-v10-formal-handoff-plan-v1'
HANDOFF_SCHEMA='a2-v10-formal-handoff-v1'
TEACHER_KEYS=tuple(f'teacher_admission_{c}_{s}' for c in ('D1','D2','D3') for s in ('A','B'))


def _read(path):return json.loads(Path(path).read_text())

def _ref(value):
    if isinstance(value,dict):return dict(value)
    path=Path(value).resolve();return dict(path=str(path),sha256=sha256(path))

def _bound(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or sha256(path)!=ref['sha256']:raise ValueError('handoff bound reference changed')
    return path.resolve()

def _prefix(path,size=None):
    import hashlib
    path=Path(path);size=path.stat().st_size if size is None else size
    if type(size) is not int or size<0 or path.stat().st_size<size:raise ValueError('handoff ledger prefix truncated')
    digest=hashlib.sha256();left=size
    with path.open('rb') as f:
        while left:
            block=f.read(min(left,8*1024*1024))
            if not block:raise ValueError('handoff ledger prefix truncated')
            digest.update(block);left-=len(block)
    return dict(bytes=size,sha256=digest.hexdigest())

def _immutable_json(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(row,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        try:os.link(temp,path)
        except FileExistsError:
            if _read(path)!=row:raise ValueError('immutable handoff output already differs')
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.unlink(temp)


def build_plan(path,*,source_compatibility,origin_checkpoint,trial_guard,trial_result,learning_ledger_path,evaluation_ledger_path,resource_selection,diagnostic,handoff_path,physical_sha256):
    """Write an immutable pending plan; this does not admit teachers or training."""
    learning=str(Path(learning_ledger_path).resolve());evaluation=str(Path(evaluation_ledger_path).resolve())
    document=dict(schema=PLAN_SCHEMA,source_compatibility=_ref(source_compatibility),origin_checkpoint=_ref(origin_checkpoint),
        trial_guard=_ref(trial_guard),trial_result=_ref(trial_result),learning_ledger_path=learning,evaluation_ledger_path=evaluation,
        learning_ledger_prefix=_prefix(learning),evaluation_ledger_prefix=_prefix(evaluation),num_envs=1024,course='D1',initial_round=50,
        resource_selection=_ref(resource_selection),diagnostic=_ref(diagnostic),handoff_path=str(Path(handoff_path).resolve()),
        sds_sha256=spec.SDS_SHA256,rulings_sha256=sha256(RULINGS),physical_sha256=physical_sha256)
    _immutable_json(path,document)
    return validate_plan(path)


def _resource_report(row):
    if row.get('resource_admitted') is not True:raise ValueError('handoff gate lacks resource admission')
    report=_read(_bound(dict(path=row['resource_report_path'],sha256=row['resource_report_sha256'])))
    fields=('gpu_total_peak_mib','gpu_task_peak_mib','host_available_min_bytes','gpu_task_valid_samples')
    if (report.get('passed') is not True or any(not math.isfinite(float(report.get(k,float('nan')))) for k in fields)
        or report['gpu_total_peak_mib']>7680 or not 0<report['gpu_task_peak_mib']<=7168
        or report['host_available_min_bytes']<4*1024**3 or report['gpu_task_valid_samples']<=0):raise ValueError('handoff gate resource report invalid')


def _evaluation(row,manifest,*,course,suite,replicas,source,physical):
    expected=dict(course=course,suite=suite,replicas=replicas,reference='teachers',code_sha256=source,sds_sha256=spec.SDS_SHA256,
        rulings_sha256=sha256(RULINGS),physical_sha256=physical,model_sha256=cases.digest(teacher_identity()),suite_sha256=cases.digest(manifest))
    if any(row.get(k)!=v for k,v in expected.items()) or row.get('complete') is not True or row.get('aborted_for_teacher_failure') is not False:
        raise ValueError('handoff gate identity/completeness invalid')
    if row.get('tested_case_count')!=len(manifest) or row.get('planned_case_count')!=len(manifest):raise ValueError('handoff gate case denominator invalid')
    expected_groups={}
    for c in manifest:expected_groups.setdefault(c['group_id'],set()).add(c['case_sha256'])
    groups=row.get('groups',{})
    if set(groups)!=set(expected_groups):raise ValueError('handoff gate groups incomplete')
    flattened={}
    for key,ids in expected_groups.items():
        rows=groups[key]
        if len(rows)!=replicas or len(ids)!=replicas or {r['case_sha256'] for r in rows}!=ids:raise ValueError('handoff gate case identity invalid')
        for result in rows:
            if result.get('group_id')!=key or result['case_sha256'] in flattened:raise ValueError('handoff gate duplicate case')
            if result.get('passed') is True and (not result.get('expected_windows') or not set(result['expected_windows'])<=set(result.get('windows',{}))):raise ValueError('handoff passed case lacks quality windows')
            for metrics in result.get('windows',{}).values():
                if any(not math.isfinite(float(v)) for v in metrics.values()):raise ValueError('nonfinite handoff gate quality')
            flattened[result['case_sha256']]=result
    if len(row.get('results',[]))!=len(manifest) or {r['case_sha256']:r for r in row['results']}!=flattened:raise ValueError('handoff flat/group results disagree')
    _resource_report(row)
    return groups


def validate_plan(plan_path,*,ledger_path=None,num_envs=None,physical_sha256=None):
    path=Path(plan_path).resolve();d=_read(path)
    if (d.get('schema')!=PLAN_SCHEMA or d.get('course')!='D1' or d.get('num_envs')!=1024 or d.get('initial_round')!=50
        or d.get('sds_sha256')!=spec.SDS_SHA256 or d.get('rulings_sha256')!=sha256(RULINGS)
        or len(d.get('physical_sha256',''))!=64):raise ValueError('invalid formal handoff plan')
    if num_envs is not None and num_envs!=1024:raise ValueError('handoff requires N1024')
    if physical_sha256 is not None and physical_sha256!=d['physical_sha256']:raise ValueError('handoff physical contract differs')
    compatibility=validate_compatibility(_bound(d['source_compatibility']))
    if compatibility['sha256']!=d['source_compatibility']['sha256'] or compatibility['to_code_sha256']!=code_fingerprint() or compatibility['document']['physical_sha256']!=d['physical_sha256']:
        raise ValueError('handoff source compatibility differs')
    for name,purpose in (('origin_checkpoint','trial_checkpoint'),('trial_guard','trial_guard'),('trial_result','trial_result'),('diagnostic','diagnostic')):
        validate_compatible_artifact(_bound(d[name]),compatibility,purpose,operation_id='formal_origin_D1' if name not in ('diagnostic',) else 'diagnostic_112')
    learning=Path(d['learning_ledger_path']);evaluation=Path(d['evaluation_ledger_path'])
    for name,p in (('learning',learning),('evaluation',evaluation)):
        if not p.is_absolute() or not p.is_file():raise ValueError('handoff requires existing ledger')
        prefix=d[name+'_ledger_prefix']
        if _prefix(p,prefix['bytes'])!=prefix:raise ValueError('handoff ledger history changed')
    if ledger_path is not None and Path(ledger_path).resolve()!=learning.resolve():raise ValueError('handoff learning ledger differs')
    guard=_read(_bound(d['trial_guard']))
    if (Path(d['trial_guard']['path']).resolve()!=Path(trial_guard.CANONICAL_GUARD).resolve() or guard.get('guard_path')!=str(Path(trial_guard.CANONICAL_GUARD).resolve())
        or guard.get('schema')!='a2-v10-adjusted-trial-v1' or guard.get('ledger_path')!=str(learning.resolve()) or guard.get('course')!='D1'
        or guard.get('num_envs')!=1024 or guard.get('stop_round')!=50 or guard.get('charged_cap')!=2457600 or not guard.get('trial_id')
        or guard.get('sds_sha256')!=spec.SDS_SHA256):raise ValueError('handoff trial guard differs')
    origin=dict(path=str(_bound(d['origin_checkpoint'])),sha256=d['origin_checkpoint']['sha256'])
    trial=_read(_bound(d['trial_result']))
    expected=dict(checkpoint=origin['path'],checkpoint_sha256=origin['sha256'],course='D1',round=50,retained_completed_updates=50,
        consumed=2457600,updated_samples=2457600,stop_reason='trial_limit',physical_sha256=d['physical_sha256'],
        pre_gate_adjusted_trial=True,teacher_admitted=False,accepted=None,training_accepted=False,gate_reached=False)
    if any(k not in trial or trial[k]!=v for k,v in expected.items()):raise ValueError('handoff requires completed original 50-update trial')
    key='learning:D1'
    if any(trial.get('budget',{}).get(k,{}).get(key,0)!=v for k,v in (('charged',2457600),('executed',2457600),('uncertain',0),('updated',2457600))):raise ValueError('handoff trial accounting invalid')
    current=BudgetLedger(learning).report()
    if any(current[k].get(key,0)<2457600 for k in ('charged','executed','updated')):raise ValueError('handoff learning ledger behind trial')
    resource=validate_compatible_resource(_bound(d['resource_selection']),compatibility)
    if resource.get('passed') is not True or resource.get('num_envs')!=1024:raise ValueError('handoff resources not passed')
    from .diagnostic_a1 import diagnostic_manifest,PROTOCOL
    diagnostic=_read(_bound(d['diagnostic']))
    groups=_evaluation(diagnostic,diagnostic_manifest(),course='D1',suite='A',replicas=4,source=compatibility['from_code_sha256'],physical=d['physical_sha256'])
    if diagnostic.get('diagnostic_only') is not True or diagnostic.get('diagnostic_protocol')!=PROTOCOL or diagnostic.get('passed') is not True or not all(r['passed'] is True for rows in groups.values() for r in rows):raise ValueError('handoff 112 diagnostic not all 4/4')
    if not Path(d['handoff_path']).is_absolute():raise ValueError('handoff output must be absolute')
    return dict(path=str(path),sha256=sha256(path),document=d,compatibility=compatibility,origin_checkpoint=origin,
        learning_ledger_path=str(learning.resolve()),evaluation_ledger_path=str(evaluation.resolve()),initial_round=50,teacher_admission={})


def _teachers(receipts,plan):
    if set(receipts)!=set(TEACHER_KEYS):raise ValueError('all six teacher A/B receipts required')
    from .evaluation import grade_group
    grades={}
    for key in TEACHER_KEYS:
        course,suite=key.removeprefix('teacher_admission_').split('_');row=_read(_bound(receipts[key]))
        groups=_evaluation(row,cases.suite(course,suite,20),course=course,suite=suite,replicas=20,
            source=plan['compatibility']['to_code_sha256'],physical=plan['document']['physical_sha256'])
        if row.get('diagnostic_only') is not False:raise ValueError('diagnostic cannot admit formal teachers')
        grouped={k:grade_group(rows,rows,final=False) for k,rows in groups.items()}
        if not all(g['passed'] for g in grouped.values()):raise ValueError('formal teacher A/B capability gap')
        grades[key]=dict(passed=True,groups=grouped,candidate_model_sha256=row['model_sha256'],reference_model_sha256=row['model_sha256'])
    return grades


def finalize_handoff(plan_path,teacher_receipts):
    plan=validate_plan(plan_path);receipts={k:_ref(v) for k,v in teacher_receipts.items()};grades=_teachers(receipts,plan)
    document=dict(schema=HANDOFF_SCHEMA,plan=dict(path=plan['path'],sha256=plan['sha256']),teacher_receipts=receipts,
        teacher_admission=grades,code_sha256=code_fingerprint(),student_accepted=False)
    _immutable_json(plan['document']['handoff_path'],document)
    return validate_handoff(plan['document']['handoff_path'])


def validate_handoff(path,*,ledger_path=None,num_envs=None,physical_sha256=None):
    path=Path(path).resolve();d=_read(path)
    if d.get('schema')!=HANDOFF_SCHEMA or d.get('code_sha256')!=code_fingerprint() or d.get('student_accepted') is not False:raise ValueError('explicit completed formal handoff required')
    plan=validate_plan(_bound(d['plan']),ledger_path=ledger_path,num_envs=num_envs,physical_sha256=physical_sha256)
    if path!=Path(plan['document']['handoff_path']).resolve():raise ValueError('handoff output path differs')
    grades=_teachers(d['teacher_receipts'],plan)
    if d.get('teacher_admission')!=grades:raise ValueError('handoff teacher grades changed')
    return dict(plan,path=str(path),sha256=sha256(path),document=d,plan_document=plan['document'],teacher_admission=grades)


def authorize_origin_checkpoint(state,checked,*,checkpoint_path,num_envs,physical_sha256):
    """The only old-source exception: exact approved original trial file."""
    origin=checked['origin_checkpoint'];plan=checked['plan_document']
    if (Path(checkpoint_path).resolve()!=Path(origin['path']) or sha256(checkpoint_path)!=origin['sha256'] or num_envs!=1024
        or state.get('schema')!=SCHEMA or state.get('sds_sha256')!=spec.SDS_SHA256 or state.get('recipe')!=spec.RECIPE
        or state.get('code_sha256')!=checked['compatibility']['from_code_sha256'] or state.get('teachers')!=teacher_identity()
        or state.get('num_envs')!=num_envs or state.get('physical_sha256')!=physical_sha256 or physical_sha256!=plan['physical_sha256']):raise ValueError('exact approved origin checkpoint required')


def bind_handoff(state,checked,*,checkpoint_path,course):
    extra=state.get('extra',{});prior=extra.get('formal_handoff')
    provenance=dict(credential_path=checked['path'],credential_sha256=checked['sha256'],origin_checkpoint=checked['origin_checkpoint'],
        learning_ledger_path=checked['learning_ledger_path'],from_code_sha256=checked['compatibility']['from_code_sha256'],to_code_sha256=checked['compatibility']['to_code_sha256'])
    if prior is not None:
        if prior!=provenance:raise ValueError('formal handoff lineage differs')
    else:
        if course!='D1' or state.get('course')!='D1':raise ValueError('handoff origin must enter D1')
        authorize_origin_checkpoint(state,checked,checkpoint_path=checkpoint_path,num_envs=1024,physical_sha256=checked['plan_document']['physical_sha256'])
        trial=extra.get('trial',{})
        if (trial.get('identity')!=_read(checked['plan_document']['trial_guard']['path']) or trial.get('retained_completed_updates')!=50
            or trial.get('pre_gate_adjusted_trial') is not True or trial.get('teacher_admitted') is not False or trial.get('accepted') is not None
            or trial.get('training_accepted') is not False or state.get('rounds',{}).get('D1')!=50 or state.get('budgets',{}).get('learning:D1')!=2457600):raise ValueError('origin trial state is not retained 50-update state')
        required=('model','learner','reservoir','rng','env','accounting')
        if any(k not in state for k in required) or 'optimizer' not in state['learner']:raise ValueError('complete origin learning state required')
    return dict(extra,formal_handoff=provenance)
