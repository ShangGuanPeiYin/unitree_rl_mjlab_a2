"""Explicit, reviewed source/artifact bridge; never changes an old receipt's identity."""
import hashlib
import json
from pathlib import Path
from . import spec
from .identity import code_fingerprint

ORIGIN_SOURCE='d54265108c298d5920ae8ea9316fd3d48be4b75ce989ac0507bdbf9a2b91498e'
SCHEMA='a2-v10-formal-source-compatibility-v1'
RULINGS=spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'
PURPOSES={'resource_selection','diagnostic','trial_checkpoint','trial_result','trial_guard','controller_state','frozen_execution_plan'}
SOURCE_PURPOSES={'handoff_validation','recovery_entry','entry_dispatch','cpu_test'}
ALLOWED_SOURCES={f'src/tasks/velocity/v10/{name}.py' for name in ('source_compatibility','handoff','orchestration','checkpoint','runtime','resource_a1')}|{'scripts/train_a2_v10.py','scripts/run_a2_v10.py','scripts/run_a2_v10_formal.py'}


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _read(path):return json.loads(Path(path).read_text())


def _files(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():_sha(p) for p in set((root/'src').rglob('*.py'))|set((root/'scripts').glob('*a2_v10*.py'))|set((root/'tests').glob('test_v10*.py'))}


def _bound(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or _sha(path)!=ref['sha256']:raise ValueError('compatibility reference changed')
    return path


def source_changes(baseline_root):
    old=_files(baseline_root);new=_files(spec.ROOT)
    return [dict(path=p,before_sha256=old.get(p),after_sha256=new.get(p)) for p in sorted(set(old)|set(new)) if old.get(p)!=new.get(p)]


def build_compatibility_template(path,baseline_root,artifacts,approval_path,physical_sha256):
    root=Path(baseline_root).resolve();manifest=root/'manifest.json'
    rows=source_changes(root)
    for row in rows:row['purpose']='cpu_test' if row['path'].startswith('tests/') else 'entry_dispatch' if row['path'].startswith('scripts/') else 'handoff_validation'
    document=dict(schema=SCHEMA,approved=False,from_code_sha256=code_fingerprint(root),to_code_sha256=code_fingerprint(),
        sds_sha256=spec.SDS_SHA256,rulings_sha256=_sha(RULINGS),physical_sha256=physical_sha256,num_envs=1024,
        baseline=dict(root=str(root),manifest_path=str(manifest),manifest_sha256=_sha(manifest)),source_changes=rows,
        approval=dict(path=str(Path(approval_path).resolve()),sha256=_sha(approval_path)),
        independent_review=dict(passed=False,report_path=None,report_sha256=None),artifacts=artifacts)
    from .checkpoint import atomic_json
    atomic_json(path,document);return document


def validate_compatibility(path):
    path=Path(path).resolve();document=_read(path)
    if (document.get('schema')!=SCHEMA or document.get('approved') is not True or document.get('num_envs')!=1024
        or document.get('sds_sha256')!=spec.SDS_SHA256 or document.get('rulings_sha256')!=_sha(RULINGS)
        or document.get('from_code_sha256')!=ORIGIN_SOURCE or document.get('to_code_sha256')!=code_fingerprint() or len(document.get('physical_sha256',''))!=64):
        raise ValueError('explicit current source compatibility approval required')
    baseline=document['baseline'];root=Path(baseline['root'])
    if not root.is_absolute() or Path(baseline['manifest_path']).resolve()!=root/'manifest.json' or _sha(root/'manifest.json')!=baseline['manifest_sha256']:
        raise ValueError('compatibility baseline manifest changed')
    if _read(root/'manifest.json')!=_files(root) or code_fingerprint(root)!=document.get('from_code_sha256'):
        raise ValueError('compatibility frozen baseline bytes changed')
    actual=source_changes(root);claimed=document.get('source_changes',[])
    if [dict(path=r.get('path'),before_sha256=r.get('before_sha256'),after_sha256=r.get('after_sha256')) for r in claimed]!=actual:
        raise ValueError('compatibility source difference list incomplete')
    for row in claimed:
        p=row['path'];test=p.startswith('tests/test_v10_') and p.endswith('.py')
        if row.get('purpose') not in SOURCE_PURPOSES or (p not in ALLOWED_SOURCES and not test):raise ValueError('compatibility change outside approved handoff scope')
        if test and row['purpose']!='cpu_test':raise ValueError('compatibility test purpose invalid')
    _bound(document['approval'])
    review=document.get('independent_review',{})
    if review.get('passed') is not True:raise ValueError('independent compatibility review required')
    _bound(dict(path=review.get('report_path'),sha256=review.get('report_sha256')))
    seen=set()
    for artifact in document.get('artifacts',[]):
        _bound(artifact);identity=(artifact['path'],artifact.get('purpose'))
        if identity in seen or artifact.get('purpose') not in PURPOSES or not artifact.get('operation_ids') or len(set(artifact['operation_ids']))!=len(artifact['operation_ids']):
            raise ValueError('exact unique artifact purposes/operations required')
        seen.add(identity)
    return dict(path=str(path),sha256=_sha(path),document=document,from_code_sha256=document['from_code_sha256'],to_code_sha256=document['to_code_sha256'])


def validate_compatible_artifact(path,compatibility,purpose,operation_id=None):
    # Re-read the credential each time, not a caller-supplied mutable dictionary.
    checked=validate_compatibility(compatibility['path'] if isinstance(compatibility,dict) else compatibility)
    if isinstance(compatibility,dict) and checked['sha256']!=compatibility['sha256']:raise ValueError('compatibility credential changed')
    document=checked['document'];path=Path(path).resolve()
    matches=[r for r in document['artifacts'] if r['path']==str(path) and r['purpose']==purpose and (operation_id is None or operation_id in r['operation_ids'])]
    if len(matches)!=1:raise ValueError('artifact/operation not explicitly approved for source inheritance')
    _bound(matches[0])
    if purpose=='trial_checkpoint':return dict(path=str(path),sha256=matches[0]['sha256'])
    row=_read(path)
    if purpose in ('resource_selection','diagnostic'):
        expected=dict(code_sha256=document['from_code_sha256'],sds_sha256=document['sds_sha256'],rulings_sha256=document['rulings_sha256'],physical_sha256=document['physical_sha256'])
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('inherited artifact original identity mismatch')
    return row


def validate_compatible_resource(path,compatibility):
    selection=validate_compatible_artifact(path,compatibility,'resource_selection',operation_id='prepare_resource')
    from .resource_a1 import verify_linkage_selection
    verify_linkage_selection(selection,source_compatibility=compatibility)
    return selection
