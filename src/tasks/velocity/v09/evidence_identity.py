"""Q010: exact, independently approved old S1.0 bytes; never a code bypass.

This context grants reads, not physics, resources, learning or stage promotion.
Final raw reads additionally require the live current-candidate qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
from types import MappingProxyType

from . import cases, spec
from .budget import AppendOnlyLedger

OLD_CODE_SHA256 = '44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f'
OLD_RUN_ID = 'a2-v09-s1-seed42-r3'
APPROVAL_TYPE = 'q010_s1_0_evidence_identity'
RUNTIME_KINDS = frozenset(('baseline_index', 'calibration', 'teacher_summary', 'teacher_episodes'))
SUITES = frozenset(cases.GATE_SUITES + cases.FINAL_SUITES + ('train_calibration', 'S1_DIAG'))


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('a complete lowercase SHA256 is required')
    return value


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_path(run_dir, kind, suite):
    if kind == 'baseline_index' and suite is None:
        return run_dir / 'baseline_admission.json'
    if kind == 'calibration' and suite == 'train_calibration':
        return run_dir / 'calibration/calibration/calibration.json'
    if kind in ('teacher_summary', 'teacher_episodes') and suite in cases.GATE_SUITES + cases.FINAL_SUITES:
        return run_dir / 'evidence' / ('teacher_' + suite) / (
            'summary.json' if kind == 'teacher_summary' else 'episodes.jsonl')
    raise ValueError('this evidence kind/suite is outside Q010 S1.0 reuse')


def _check_prefix(path, prefix, run_id, *, validate_chain):
    if not path.is_file():
        raise ValueError('the live external ledger must already exist')
    size, sequence = prefix.get('byte_length'), prefix.get('sequence')
    if (isinstance(size, bool) or not isinstance(size, int) or size <= 0
            or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0):
        raise ValueError('explicit complete ledger-prefix length/sequence required')
    expected = _sha(prefix.get('sha256'))
    event_sha = _sha(prefix.get('event_sha256'))
    # Capture a coherent image under the ledger writers' inode lock. Both the
    # approved prefix and current state below are derived from this same image.
    with path.open('rb') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        try:
            image = stream.read()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    raw = image[:size]
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('the approved external ledger prefix changed or was rolled back')
    if not raw.endswith(b'\n') or raw.count(b'\n') != sequence + 1:
        raise ValueError('the approved prefix must contain complete original ledger events')
    last = json.loads(raw.rsplit(b'\n', 2)[-2])
    if last.get('sequence') != sequence or last.get('event_sha256') != event_sha:
        raise ValueError('the original ledger prefix event identity differs')
    if validate_chain:
        # Reuse the existing ledger parser/state rules without its constructor
        # or snapshot(), which would reopen a possibly different file version.
        reader = object.__new__(AppendOnlyLedger)
        reader.path, reader.run_id = path, run_id
        current = reader._state(reader._read(io.StringIO(image.decode('utf-8'))))
        if current['sequence'] < sequence or current['run_id'] != run_id:
            raise ValueError('current ledger is older or belongs to another run')


@dataclass(frozen=True)
class EvidenceRef:
    category: str
    kind: str
    suite: str | None
    path: str
    file_sha256: str
    manifest_sha256: str | None
    producer_code_sha256: str
    consumer_code_sha256: str
    run_id: str
    run_dir: str
    approval_path: str
    approval_sha256: str


class EvidenceIdentityContext:
    """Immutable approved catalog. Construct only through from_approval."""
    __slots__ = ('_approval_path', '_approval_sha256', '_run_dir', '_run_id',
                 '_ledger_path', '_consumer_code_sha256', '_references', '_prefix')

    def __init__(self, *args, **kwargs):
        raise TypeError('use EvidenceIdentityContext.from_approval')

    def __setattr__(self, name, value):
        raise AttributeError('approved evidence identity is immutable')

    @property
    def approval_path(self):
        return self._approval_path

    @property
    def approval_sha256(self):
        return self._approval_sha256

    @property
    def run_dir(self):
        return self._run_dir

    @property
    def run_id(self):
        return self._run_id

    @property
    def ledger_path(self):
        return self._ledger_path

    @property
    def consumer_code_sha256(self):
        return self._consumer_code_sha256

    @classmethod
    def from_approval(cls, approval_path, approval_sha256, *, consumer_code_sha256,
                      run_id, run_dir, ledger_path, sds_sha256, source_sha256):
        approval_sha256 = _sha(approval_sha256)
        consumer = _sha(consumer_code_sha256)
        if consumer == OLD_CODE_SHA256 or run_id != OLD_RUN_ID:
            raise ValueError('Q010 permits only this direct old producer to a new consumer/run')
        if sds_sha256 != spec.SDS_SHA256 or source_sha256 != spec.SOURCE_SHA256:
            raise ValueError('Q010 source/SDS identity differs')
        approval_path = Path(approval_path).resolve()
        raw = approval_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != approval_sha256:
            raise ValueError('independent approval file SHA differs')
        body = json.loads(raw)
        run_dir = Path(run_dir).resolve()
        ledger_path = Path(ledger_path).resolve()
        if (type(body.get('schema')) is not int or body.get('schema') != 1
                or body.get('approval_type') != APPROVAL_TYPE
                or body.get('scope') != ['S1.0_evidence_reuse']
                or body.get('approved') is not True
                or body.get('producer_code_sha256') != OLD_CODE_SHA256
                or body.get('consumer_code_sha256') != consumer
                or body.get('run_id') != run_id
                or body.get('run_dir') != str(run_dir)
                or body.get('sds_sha256') != sds_sha256
                or body.get('source_sha256') != source_sha256):
            raise ValueError('independent direct-reuse approval identity/scope differs')
        if ledger_path != run_dir / 'budget.jsonl':
            raise ValueError('Q010 must use this run live external ledger')
        prefix = body.get('ledger_prefix')
        if not isinstance(prefix, dict) or prefix.get('path') != str(ledger_path):
            raise ValueError('approval must bind the exact live ledger and frozen prefix')
        _check_prefix(ledger_path, prefix, run_id, validate_chain=True)
        review = body.get('review')
        if (not isinstance(review, dict) or review.get('approved') is not True
                or review.get('old_producer_compatible') is not True):
            raise ValueError('independent old/new compatibility review is not approved')
        if (_file_sha(Path(review['report_path']).resolve())
                != _sha(review.get('report_sha256'))):
            raise ValueError('independent compatibility review content SHA differs')
        refs = {}
        for category in ('runtime_evidence', 'frozen_inputs', 'audit_material'):
            entries = body.get(category)
            if not isinstance(entries, list):
                raise ValueError('explicit independent catalog lists are required')
            for row in entries:
                if not isinstance(row, dict) or not isinstance(row.get('path'), str):
                    raise ValueError('explicit artifact path is required')
                path = Path(row['path']).resolve()
                if not path.is_relative_to(run_dir) or str(path) in refs:
                    raise ValueError('duplicate or outside-run evidence path')
                kind, suite = row.get('kind'), row.get('suite')
                manifest = row.get('manifest_sha256')
                if manifest is not None:
                    _sha(manifest)
                if category == 'runtime_evidence':
                    if kind not in RUNTIME_KINDS or path != _runtime_path(run_dir, kind, suite):
                        raise ValueError('old resources/approvals/checkpoints/DIAG are not reusable')
                    if kind != 'baseline_index' and manifest is None:
                        raise ValueError('runtime evidence must bind its frozen manifest')
                elif category == 'frozen_inputs':
                    if (kind != 'frozen_manifest' or suite not in SUITES or manifest is None
                            or path != run_dir / 'cases' / (suite + '.json')):
                        raise ValueError('frozen input must name its exact suite/manifest')
                elif kind != 'audit_material':
                    raise ValueError('audit-only material does not grant runtime reads')
                if row.get('producer_code_sha256', OLD_CODE_SHA256) != OLD_CODE_SHA256:
                    raise ValueError('transitive or different producer reuse is forbidden')
                refs[str(path)] = EvidenceRef(category, kind, suite, str(path),
                    _sha(row.get('file_sha256')), manifest, OLD_CODE_SHA256, consumer,
                    run_id, str(run_dir), str(approval_path), approval_sha256)
        if not refs:
            raise ValueError('an explicit nonempty artifact allowlist is required')
        obj = object.__new__(cls)
        for name, value in dict(_approval_path=approval_path, _approval_sha256=approval_sha256,
                _run_dir=run_dir, _run_id=run_id, _ledger_path=ledger_path,
                _consumer_code_sha256=consumer, _references=MappingProxyType(refs),
                _prefix=MappingProxyType(dict(prefix))).items():
            object.__setattr__(obj, name, value)
        return obj

    def expected_ref(self, path, *, kind, suite):
        ref = self._references.get(str(Path(path).resolve()))
        if ref is None or ref.kind != kind or ref.suite != suite:
            raise ValueError('path/kind/suite is not explicitly approved')
        return ref

    def _reference(self, path, kind, suite, manifest_sha256):
        ref = self.expected_ref(path, kind=kind, suite=suite)
        if manifest_sha256 is not None and ref.manifest_sha256 != manifest_sha256:
            raise ValueError('the trusted frozen manifest SHA differs')
        _check_prefix(self.ledger_path, self._prefix, self.run_id, validate_chain=True)
        return ref

    def verify_ref(self, path, *, kind, suite, manifest_sha256=None):
        ref = self._reference(path, kind, suite, manifest_sha256)
        if _file_sha(ref.path) != ref.file_sha256:
            raise ValueError('approved evidence content SHA differs')
        return ref

    def _final_authority(self, ref, candidate_sha256):
        if ref.suite not in cases.FINAL_SUITES:
            return
        try:
            _sha(candidate_sha256)
        except ValueError as error:
            raise PermissionError('final evidence remains sealed without the real candidate') from error
        from .s1_runner import PlanningRequired, validate_final_unseal
        try:
            validate_final_unseal(run_dir=self.run_dir,
                ledger=AppendOnlyLedger(self.ledger_path, run_id=self.run_id),
                suite=ref.suite, candidate_sha256=candidate_sha256)
        except PlanningRequired as error:
            raise PermissionError('final evidence remains sealed without live qualification') from error

    def _checked_bytes(self, ref):
        raw = Path(ref.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref.file_sha256:
            raise ValueError('approved evidence content SHA differs')
        return raw

    def _runtime_bytes(self, path, kind, suite, manifest_sha256, candidate_sha256):
        ref = self._reference(path, kind, suite, manifest_sha256)
        if ref.category != 'runtime_evidence':
            raise PermissionError('audit data is hash-only; frozen inputs require read_manifest')
        self._final_authority(ref, candidate_sha256)
        if kind == 'teacher_episodes':
            # JSONL has no producer-code field: the approved related summary
            # proves that producer, together with both exact content hashes.
            self.read_legacy_json(Path(ref.path).with_name('summary.json'),
                kind='teacher_summary', suite=suite, manifest_sha256=ref.manifest_sha256,
                candidate_sha256=candidate_sha256)
        return self._checked_bytes(ref), ref

    @staticmethod
    def _original_json(data, ref):
        if not isinstance(data, dict) or data.get('code_sha256') != OLD_CODE_SHA256:
            raise ValueError('original evidence producer code differs')
        if ref.kind in ('baseline_index', 'calibration'):
            if data.get('sds_sha256') != spec.SDS_SHA256 or data.get('source_sha256') != spec.SOURCE_SHA256:
                raise ValueError('original source/SDS identity differs')
        elif ref.kind == 'teacher_summary' and data.get('checkpoint_sha256') != spec.SOURCE_SHA256:
            raise ValueError('the original teacher checkpoint differs')
        if ref.manifest_sha256 is not None and data.get('manifest_sha256') != ref.manifest_sha256:
            raise ValueError('the original evidence manifest differs')

    def read_legacy_bytes(self, path, *, kind, suite, manifest_sha256=None, candidate_sha256=None):
        raw, ref = self._runtime_bytes(path, kind, suite, manifest_sha256, candidate_sha256)
        if kind != 'teacher_episodes':
            self._original_json(json.loads(raw), ref)
        return raw, ref

    def read_legacy_json(self, path, *, kind, suite, manifest_sha256=None, candidate_sha256=None):
        raw, ref = self._runtime_bytes(path, kind, suite, manifest_sha256, candidate_sha256)
        if kind == 'teacher_episodes':
            raise ValueError('JSONL episodes require the verified bytes reader')
        data = json.loads(raw)
        self._original_json(data, ref)
        return data, ref

    def read_manifest(self, path, *, purpose, suite, candidate_sha256=None):
        ref = self._reference(path, 'frozen_manifest', suite, None)
        if ref.category != 'frozen_inputs':
            raise ValueError('manifest is not an approved frozen input')
        if purpose not in ('training', 'gate', 'acceptance', 'freeze_audit', 'diagnostic', 'evidence'):
            raise ValueError('explicit manifest purpose is required')
        if suite in cases.FINAL_SUITES and purpose not in ('acceptance', 'freeze_audit'):
            raise PermissionError('final input is sealed from training/model selection')
        if purpose in ('diagnostic', 'evidence') and suite != 'S1_DIAG':
            raise PermissionError('diagnostic/video may only read S1_DIAG')
        if suite == 'S1_DIAG' and purpose not in ('diagnostic', 'evidence', 'freeze_audit'):
            raise PermissionError('S1_DIAG is an independent diagnostic stream')
        if purpose == 'training' and suite != 'train_calibration':
            raise PermissionError('training cannot read gate/final input')
        self._final_authority(ref, candidate_sha256)
        data = json.loads(self._checked_bytes(ref))
        cases.validate_manifest(data)
        if data['suite'] != suite or data['manifest_sha256'] != ref.manifest_sha256:
            raise ValueError('frozen input suite/manifest identity differs')
        return data, ref
