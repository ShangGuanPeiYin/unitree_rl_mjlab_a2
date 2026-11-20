"""Freeze s_V from independent training calibration trajectories, before S1.1."""
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
from src.tasks.velocity.v09.cases import load_manifest, canonical_json
from src.tasks.velocity.v09.budget import AppendOnlyLedger, batch_worst_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--segment-id', required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec.validate_sources()
    snapshot = spec.code_snapshot()
    if snapshot['code_sha256'] != args.code_sha256:
        raise ValueError('code differs from the independently approved source snapshot')
    manifest = load_manifest(args.run_dir / 'cases/train_calibration.json', purpose='training')
    plan = batch_worst_cost(manifest['cases'], batch_size=args.batch_size)
    if plan['control_samples'] > 1_000_000:
        raise ValueError('calibration worst physical cost exceeds one million')
    if not args.execute:
        print(json.dumps(dict(cases=plan['case_count'], batches=len(plan['batches']),
            worst_control_samples=plan['control_samples'], device=args.device), indent=2))
        return
    import torch
    from src.tasks.velocity.v09.env import V09S1Env
    from src.tasks.velocity.v09.model import Frozen6400Teacher
    from src.tasks.velocity.v09.runtime import ValueMoments, calibrate_batch
    from src.tasks.velocity.v09.geometry import geometry_certificate
    ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
    if not ledger.snapshot()['stages']['S1']['envelope']:
        raise RuntimeError('stage budget admission is required')
    output = args.run_dir / 'calibration' / args.segment_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source_snapshot.json').write_text(canonical_json(snapshot) + '\n')
    teacher = Frozen6400Teacher.from_6400_checkpoint(device=args.device)
    moments = ValueMoments()
    by_sha = {case['case_sha256']: case for case in manifest['cases']}
    physical_samples = 0
    records = []
    for index, batch in enumerate(plan['batches']):
        operation = ledger.reserve(segment_id=args.segment_id, operation_id=f'batch_{index:04d}',
            stage='S1', course='S1.0', category='engineering',
            num_envs=args.batch_size, max_control_steps=batch['control_steps'],
            checkpoint_sha256=spec.SOURCE_SHA256, reason='independent_training_value_calibration',
            case_sha256=manifest['manifest_sha256'], obligation_id='calibration',
            case_sha256s=batch['case_sha256s'])
        env, result, settled = None, None, False
        try:
            cases = [by_sha[identity] for identity in batch['case_sha256s']]
            env = V09S1Env(cases, device=args.device, seed=42)
            geometry = geometry_certificate(env, code_sha256=snapshot['code_sha256'])
            (output / f'batch_{index:04d}_geometry.json').write_text(canonical_json(geometry) + '\n')
            result = calibrate_batch(env, teacher, max_steps=batch['control_steps'], moments=moments)
            for record, case in zip(result['records'], cases):
                record.update(operation_course='S1.0', case_id=case['case_id'], case_sha256=case['case_sha256'])
            torch.save(dict(values=result['calibration_values'],
                            case_slots=result['value_case_slots'],
                            case_sha256s=batch['case_sha256s'],
                            checkpoint_sha256=spec.SOURCE_SHA256),
                       output / f'batch_{index:04d}_values.pt')
            with (output / 'episodes.jsonl').open('a') as stream:
                for record in result['records']:
                    stream.write(canonical_json(record) + '\n')
            ledger.commit(operation, actual_control_steps=result['control_steps'])
            settled = True
            physical_samples += result['physical_control_samples']
            records.extend(result['records'])
            print(json.dumps(dict(batch=index + 1, total_batches=len(plan['batches']),
                physical_control_samples=physical_samples, value_samples=moments.count)), flush=True)
        except BaseException as error:
            if not settled:
                if result is None:
                    ledger.commit(operation, status='unknown')
                else:
                    ledger.commit(operation, actual_control_steps=result['control_steps'], status='interrupted')
            ledger.invalidate_obligation('S1', 'calibration', reason=type(error).__name__)
            (output / 'engineering_error.json').write_text(json.dumps(dict(
                error_type=type(error).__name__, error=str(error))) + '\n')
            raise
        finally:
            if env is not None:
                try:
                    env.close()
                except BaseException as error:
                    ledger.invalidate_obligation('S1', 'calibration', reason='environment_close')
                    (output / 'engineering_error.json').write_text(json.dumps(dict(
                        error_type=type(error).__name__, error=str(error))) + '\n')
                    raise
    if len(records) != plan['case_count']:
        raise RuntimeError('calibration case denominator incomplete')
    calibration = dict(sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
        manifest_sha256=manifest['manifest_sha256'], code_sha256=snapshot['code_sha256'],
        value_scale=moments.value_scale, value_samples=moments.count,
        value_mean=moments.mean, value_m2=moments.m2, physical_control_samples=physical_samples,
        method='population_std_max_1', case_count=len(records), ledger=ledger.checkpoint_metadata())
    target = output / 'calibration.json'
    with target.open('x') as stream:
        json.dump(calibration, stream, indent=2)
        stream.write('\n')
    ledger.close_obligation('S1', 'calibration', reason='frozen_training_calibration_saved')
    print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         value_scale=moments.value_scale)), flush=True)


if __name__ == '__main__':
    main()
