"""One fixed, explicitly non-admission A1 diagnostic; no gain search or repeats."""
from pathlib import Path
from . import cases, spec

PROTOCOL='A1-diagnostic-112-v1'
MAX_SAMPLES=251288


def diagnostic_manifest():
    selected=[]
    for course,width in (('D1',1.2),('D2',.8)):
        for row in cases.suite(course,'A',4):
            if (course=='D1' and row['task'].startswith('F_')) or (
                row['task'].startswith('P_') and row['height']==2. and row['width']==width and row['encoding']=='G'):
                selected.append(row)
    groups={}
    for row in selected:groups.setdefault(row['group_id'],[]).append(row)
    if len(selected)!=112 or len(groups)!=28 or sum(c['max_steps'] for c in selected)!=MAX_SAMPLES:
        raise RuntimeError('A1 diagnostic manifest/budget differs from approved 112/251288')
    if any(len(v)!=4 or len({r['max_steps'] for r in v})!=1 for v in groups.values()):
        raise RuntimeError('A1 diagnostic batch bounds differ')
    return selected


def diagnostic_plan():
    manifest=diagnostic_manifest()
    return dict(protocol=PROTOCOL,diagnostic_only=True,manifest_sha256=cases.digest(manifest),
        cases=manifest,case_count=112,groups=28,max_samples=MAX_SAMPLES,
        group_threshold='4/4',formal_group_threshold='18/20',sds_sha256=spec.SDS_SHA256,
        historical_evaluation_charged=304100,evaluation_global_cap=1685390840,
        revised_no_cache_maximum=1685946228,potential_no_cache_shortfall=555388)


def validate_diagnostic_resource(path):
    """A diagnostic cannot run ahead of the exact-code linked two-stack gate."""
    import json
    from .identity import code_fingerprint
    from .resource_a1 import verify_linkage_selection
    try:
        selection=json.loads(Path(path).read_text())
        if (selection.get('passed') is not True or selection.get('num_envs')!=1024
            or selection.get('sds_sha256')!=spec.SDS_SHA256 or selection.get('code_sha256')!=code_fingerprint()):
            raise ValueError('resource selection identity invalid')
        verify_linkage_selection(selection)
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise ValueError('A1 diagnostic resource gate invalid') from error
    return selection


def claim_diagnostic_once(ledger_path,output_dir,plan):
    """A durable O_EXCL claim prevents a new directory from buying an extra run."""
    import json
    import os
    from .checkpoint import sha256
    target=Path(str(Path(ledger_path).resolve())+'.A1-diagnostic-112-v1.claim.json')
    payload=dict(protocol=PROTOCOL,manifest_sha256=plan['manifest_sha256'],
                 output_dir=str(Path(output_dir).resolve()),status='claimed_no_automatic_repeat',
                 rulings_sha256=sha256(spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'))
    try:fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:raise RuntimeError('A1 diagnostic already claimed; explicit new ruling required') from None
    with os.fdopen(fd,'w') as f:
        json.dump(payload,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    parent_fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    return payload


def run_diagnostic(*,output_dir,ledger_path,resource_selection,eval_limit=1685390840,device='cuda:0'):
    from .checkpoint import atomic_json,sha256
    from .identity import code_fingerprint
    from .eval_worker import run_suite
    from .resources import ResourceMonitor
    from .gpu_lock import gpu_execution_lock
    from .evaluation_capacity import evaluation_capacity
    from .teachers import teacher_identity
    import torch
    validate_diagnostic_resource(resource_selection)
    if eval_limit!=1685390840:raise ValueError('A1 evaluation cap must remain 1685390840')
    out=Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('diagnostic output must be new; no automatic repeat')
    out.mkdir(parents=True,exist_ok=True)
    plan=diagnostic_plan();plan.update(code_sha256=code_fingerprint(),teachers=teacher_identity(),resource_selection_path=str(Path(resource_selection).resolve()),resource_selection_sha256=sha256(resource_selection))
    atomic_json(out/'fixed_execution_plan.json',plan)
    with evaluation_capacity(ledger_path,PROTOCOL,MAX_SAMPLES,eval_limit):
        claim_diagnostic_once(ledger_path,out,plan)
        with gpu_execution_lock():
            with ResourceMonitor() as monitor:
                torch.cuda.reset_peak_memory_stats()
                result=run_suite(course='D1',suite_name='A',replicas=4,model_path=None,reference='teachers',
                    output_dir=out,ledger_path=ledger_path,eval_limit=eval_limit,device=device,
                    stop_on_failed_group=True,diagnostic_manifest=plan['cases'])
                torch.cuda.synchronize()
            summary=monitor.summary();report=out/'resource_report.json';atomic_json(report,summary)
            result.pop('_new_caches',None)
            result.update(diagnostic_only=True,diagnostic_protocol=PROTOCOL,
                resource_report_path=str(report),resource_report_sha256=sha256(report),resource_admitted=summary['passed'],
                passed=(summary['passed'] and result['complete'] and result['tested_case_count']==112
                        and all(r['passed'] for rows in result['groups'].values() for r in rows)))
            atomic_json(out/'result.json',result)
            return result
