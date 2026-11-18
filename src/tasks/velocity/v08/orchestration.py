"""Independent acceptance → zero-learning baseline → unique formal admission."""
import json
from pathlib import Path
from . import recovery
from .continuation_v08 import SOURCE_SHA256,SDS_SHA256
class AdmissionError(ValueError):pass

def verify_acceptance(path,frozen,case_tables):
    path=Path(path)
    value=json.loads(path.read_text())
    expected=dict(accepted=True,blocking=0,reviewer_role='independent',
        freeze_sha256=frozen['sha256'],source_checkpoint_sha256=SOURCE_SHA256,
        sds_sha256=SDS_SHA256,case_tables_sha256=recovery.digest(case_tables))
    if type(value.get('blocking')) is not int or any(value.get(k)!=v for k,v in expected.items()):
        raise AdmissionError('independent acceptance is not blocking0 for actual source/config/cases')
    report=Path(value.get('report_path',''))
    if not report.is_file() or recovery.sha256(report)!=value.get('report_sha256'):
        raise AdmissionError('independent acceptance report SHA differs')
    return dict(path=str(path.resolve()),sha256=recovery.sha256(path),verdict=value)

def verify_baseline(path,frozen,case_tables,acceptance):
    from .evaluation_records import restore_result,verify_sidecar
    path=Path(path);value=json.loads(path.read_text())
    expected=dict(mode='baseline',complete=True,source_checkpoint_sha256=SOURCE_SHA256,
        freeze_sha256=frozen['sha256'],case_tables_sha256=recovery.digest(case_tables),
        acceptance_sha256=acceptance['sha256'])
    if any(value.get(k)!=v for k,v in expected.items()):raise AdmissionError('baseline is absent/incomplete or foreign')
    evidence=value['evaluation']
    from . import spec
    identity=dict(update=6400,phase='A',kind='baseline',case_table_sha256=case_tables['baseline:A']['sha256'],
        source_sha256=SOURCE_SHA256,recipe_sha256=recovery.digest(recovery.jsonable(spec.contract())),freeze_sha256=frozen['sha256'])
    result=restore_result(evidence['payload'],identity)
    verify_sidecar(evidence['binding'],evidence['payload'])
    reference=value.get('evaluation_transaction')
    if not isinstance(reference,dict) or reference.get('binding')!=evidence['binding'] or reference.get('identity')!=identity:
        raise AdmissionError('baseline durable evaluation transaction missing or foreign')
    recovery.load_evaluation_reference(reference)
    summary=result.summary
    if result.table.sha256!=case_tables['baseline:A']['sha256'] or summary['expected']!=60 or summary['valid']!=60 or summary['invalid'] or summary['error']:
        raise AdmissionError('baseline full valid60 denominator required (policy failures permitted)')
    return dict(path=str(path.resolve()),sha256=recovery.sha256(path))

def admit(*,acceptance_file,baseline_report,frozen,case_tables,formal=False):
    if not acceptance_file:raise AdmissionError('independent acceptance file is required')
    acceptance=verify_acceptance(acceptance_file,frozen,case_tables)
    baseline=None
    if formal:
        if not baseline_report:raise AdmissionError('completed source-bound baseline required before formal learning')
        baseline=verify_baseline(baseline_report,frozen,case_tables,acceptance)
    return dict(acceptance=acceptance,baseline=baseline)
