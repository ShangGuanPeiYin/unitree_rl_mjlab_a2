"""Compare initial/adapted models on the same frozen physical diagnostic data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import load_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--before-dir', type=Path, required=True)
    p.add_argument('--after-dir', type=Path, required=True)
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--initial-checkpoint-sha256', required=True)
    p.add_argument('--adapted-checkpoint', type=Path, required=True)
    p.add_argument('--adapted-checkpoint-sha256', required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--code-sha256', required=True)
    p.add_argument('--evidence-identity', type=Path)
    p.add_argument('--evidence-identity-sha256')
    p.add_argument('--device', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec.validate_sources()
    if spec.code_snapshot()['code_sha256'] != a.code_sha256:
        raise ValueError('source snapshot mismatch')
    from src.tasks.velocity.v09.s1_runner import load_current_checkpoint
    payloads = {}
    for label, path, expected in (
        ('initial', a.initial_checkpoint, a.initial_checkpoint_sha256),
        ('adapted', a.adapted_checkpoint, a.adapted_checkpoint_sha256)):
        payloads[label] = load_current_checkpoint(path, device='cpu',
            code_sha256=a.code_sha256, checkpoint_sha256=expected)
    from src.tasks.velocity.v09.model import V09ActorCritic
    from src.tasks.velocity.v09.checkpoint import load_checkpoint
    from src.tasks.velocity.v09.diagnostics import loss_report
    from src.tasks.velocity.v09.s1_runner import create_identity_context, identity_binding
    from src.tasks.velocity.v09.resources import load_calibration
    from dataclasses import asdict
    identity_context = create_identity_context(approval_path=a.evidence_identity,
        approval_sha256=a.evidence_identity_sha256, code_sha256=a.code_sha256,
        run_id=a.run_id, run_dir=a.run_dir)
    manifest, manifest_ref = identity_context.read_manifest(
        a.run_dir / 'cases/S1_DIAG.json', purpose='diagnostic', suite='S1_DIAG')
    calibration = load_calibration(a.calibration, code_sha256=a.code_sha256,
                                   identity_context=identity_context)
    if calibration['sds_sha256'] != spec.SDS_SHA256 or calibration['source_sha256'] != spec.SOURCE_SHA256:
        raise ValueError('calibration identity mismatch')
    report = dict(sds_sha256=spec.SDS_SHA256, code_sha256=a.code_sha256,
                  manifest_sha256=manifest['manifest_sha256'], losses={}, behavior_pairs=[],
                  evidence_identity=identity_binding(identity_context),
                  calibration_evidence_ref=calibration['evidence_ref'],
                  frozen_manifest_evidence_ref=asdict(manifest_ref))
    for label, path, expected in (
        ('initial', a.initial_checkpoint, a.initial_checkpoint_sha256),
        ('adapted', a.adapted_checkpoint, a.adapted_checkpoint_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('model checkpoint SHA mismatch')
        payload = payloads[label]
        model = V09ActorCritic().to(a.device)
        model.load_state_dict(payload['model'], strict=True)
        report['losses'][label] = {data: loss_report(model, folder, manifest,
            value_scale=calibration['value_scale'], device=a.device, code_sha256=a.code_sha256,
            dataset_checkpoint_sha256=(a.initial_checkpoint_sha256 if data == 'before_physical_states' else a.adapted_checkpoint_sha256))
            for data, folder in (('before_physical_states', a.before_dir), ('after_physical_states', a.after_dir))}
        del model
    migration_reference = V09ActorCritic.migrate_6400_checkpoint().state_dict()
    initial_state = payloads['initial']['model']
    import torch
    migration_valid = all(torch.equal(initial_state[key].cpu(), tensor.cpu())
                          for key, tensor in migration_reference.items())
    if not migration_valid:
        raise ValueError('initial diagnostic model differs from exact approved 6400 migration')
    rows = []
    for folder in (a.before_dir, a.after_dir):
        if (folder / 'engineering_error.json').exists():
            raise ValueError('diagnostic suite has an engineering error')
        current = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        by_id = {row['case_sha256']: row for row in current}
        if len(current) != len(manifest['cases']) or set(by_id) != {case['case_sha256'] for case in manifest['cases']}:
            raise ValueError('diagnostic behavior denominator incomplete')
        rows.append(by_id)
    for case in manifest['cases']:
        before, after = (entries[case['case_sha256']] for entries in rows)
        report['behavior_pairs'].append(dict(case_sha256=case['case_sha256'],
            before={key: before[key] for key in ('success', 'failure', 'timeout', 'reason', 'forward_distance', 'metrics')},
            after={key: after[key] for key in ('success', 'failure', 'timeout', 'reason', 'forward_distance', 'metrics')}))
    overflow = any(row.get('failure_masks', {}).get('contact_overflow')
                   or row.get('failure_masks', {}).get('nonfinite')
                   for entries in rows for row in entries.values())
    report.update(engineering_valid=not overflow, finite=True,
                  migration_valid=migration_valid, contacts_valid=not overflow,
                  contact_validation='validated compiled geometry certificates matching source SHA and contact indices, plus no runtime overflow/nonfinite masks')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(a.output), sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
