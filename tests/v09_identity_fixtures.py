"""Small CPU fixtures for identity consumers, never actual run approvals."""
import hashlib
import json
from pathlib import Path

from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.budget import AppendOnlyLedger

OLD_CODE = '44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f'
RUN_ID = 'a2-v09-s1-seed42-r3'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IdentityFixture:
    def __init__(self, directory, consumer_code_sha256='c' * 64):
        self.run_dir = Path(directory) / 'run'
        self.run_dir.mkdir(parents=True)
        self.run_id = RUN_ID
        self.consumer_code_sha256 = consumer_code_sha256
        self.ledger_path = self.run_dir / 'budget.jsonl'
        self.ledger = AppendOnlyLedger(self.ledger_path, run_id=RUN_ID)
        prefix = self.ledger_path.read_bytes()
        state = self.ledger.snapshot()
        review = self.run_dir / 'independent-unit-review.md'
        review.write_text('Independent fixture only; not an actual execution approval.\n')
        self.approval = dict(schema=1, approval_type='q010_s1_0_evidence_identity',
            scope=['S1.0_evidence_reuse'], approved=True,
            producer_code_sha256=OLD_CODE, consumer_code_sha256=consumer_code_sha256,
            run_id=RUN_ID, run_dir=str(self.run_dir.resolve()),
            sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
            ledger_prefix=dict(path=str(self.ledger_path.resolve()), byte_length=len(prefix),
                sha256=hashlib.sha256(prefix).hexdigest(), sequence=state['sequence'],
                event_sha256=state['ledger_sha256']),
            frozen_inputs=[], runtime_evidence=[], audit_material=[],
            review=dict(approved=True, old_producer_compatible=True,
                report_path=str(review.resolve()), report_sha256=file_sha(review)))
        self.approval_path = self.run_dir / 'identity-approval.json'
        self.calibration_path = self.run_dir / 'calibration/calibration/calibration.json'
        self.calibration = dict(code_sha256=OLD_CODE, sds_sha256=spec.SDS_SHA256,
            source_sha256=spec.SOURCE_SHA256, manifest_sha256='a' * 64,
            value_scale=2., value_samples=600, physical_control_samples=720,
            value_mean=1., value_m2=2400., method='population_std_max_1', case_count=600)
        self.put_json(self.calibration_path, self.calibration)
        self.bind_file(self.calibration_path, kind='calibration', suite='train_calibration',
                       manifest_sha256='a' * 64)
        self.seal()

    def put_json(self, path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
        return path

    def bind_file(self, path, *, kind, suite, manifest_sha256=None, category='runtime_evidence'):
        path = Path(path).resolve()
        row = dict(path=str(path), kind=kind, suite=suite,
                   file_sha256=file_sha(path), manifest_sha256=manifest_sha256)
        self.approval[category] = [r for r in self.approval[category] if r['path'] != str(path)] + [row]
        return row

    def seal(self):
        self.put_json(self.approval_path, self.approval)
        self.approval_sha256 = file_sha(self.approval_path)
        return self.approval_path, self.approval_sha256

    def context_kwargs(self):
        return dict(consumer_code_sha256=self.consumer_code_sha256,
            run_id=self.run_id, run_dir=self.run_dir, ledger_path=self.ledger_path,
            sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256)


def make_identity_fixture(directory, consumer_code_sha256='c' * 64):
    return IdentityFixture(directory, consumer_code_sha256)
