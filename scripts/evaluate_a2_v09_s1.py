"""Run one frozen S1 suite with physical-slot accounting and first outcomes."""
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
    parser.add_argument('--suite', choices=('gate_A', 'gate_B', 'final_1', 'final_2', 'final_3', 'S1_DIAG'), required=True)
    parser.add_argument('--policy', choices=('teacher', 'student'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--obligation-id', required=True)
    parser.add_argument('--segment-id', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec.validate_sources()
    snapshot = spec.code_snapshot()
    if snapshot['code_sha256'] != args.code_sha256:
        raise ValueError('source snapshot differs from independently approved code SHA')
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        raise ValueError('checkpoint SHA mismatch')
    final_unseal_proof = None
    if args.policy == 'student' and args.suite.startswith('final_'):
        from src.tasks.velocity.v09.s1_runner import validate_final_unseal
        ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
        final_unseal_proof = validate_final_unseal(run_dir=args.run_dir, ledger=ledger,
            suite=args.suite, candidate_sha256=digest)
    manifest = load_manifest(args.run_dir / 'cases' / f'{args.suite}.json', purpose=(
        'diagnostic' if args.suite == 'S1_DIAG' else 'freeze_audit' if args.policy == 'teacher' else 'acceptance' if args.suite.startswith('final_') else 'gate'))
    plan = batch_worst_cost(manifest['cases'], batch_size=args.batch_size)
    if not args.execute:
        print(json.dumps(dict(suite=args.suite, cases=plan['case_count'],
            batches=len(plan['batches']), worst_control_samples=plan['control_samples'],
            checkpoint_sha256=digest, device=args.device), indent=2))
        return
    import torch
    from src.tasks.velocity.v09.env import V09S1Env
    from src.tasks.velocity.v09.model import Frozen6400Teacher, V09ActorCritic
    from src.tasks.velocity.v09.checkpoint import load_checkpoint
    from src.tasks.velocity.v09.runtime import evaluate_batch
    from src.tasks.velocity.v09.geometry import geometry_certificate
    ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
    if not ledger.snapshot()['stages']['S1']['envelope']:
        raise RuntimeError('the stage must be budget-admitted before evaluation')
    output = args.run_dir / 'evidence' / args.segment_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source_snapshot.json').write_text(canonical_json(snapshot) + '\n')
    if args.policy == 'teacher':
        if digest != spec.SOURCE_SHA256:
            raise ValueError('baseline teacher must be the unique source 6400')
        policy = Frozen6400Teacher.from_6400_checkpoint(args.checkpoint, device=args.device)
    else:
        payload = load_checkpoint(args.checkpoint, device=args.device)
        policy = V09ActorCritic().to(args.device)
        policy.load_state_dict(payload['model'], strict=True)
    policy.eval()
    audit_teacher = Frozen6400Teacher.from_6400_checkpoint(device=args.device) if args.suite == 'S1_DIAG' else None
    operation_course = 'S1_EVIDENCE' if args.suite == 'S1_DIAG' else 'S1.0' if args.policy == 'teacher' else 'S1.2'
    by_sha = {case['case_sha256']: case for case in manifest['cases']}
    all_records = []
    try:
        for index, batch in enumerate(plan['batches']):
            cases = [by_sha[identity] for identity in batch['case_sha256s']]
            operation = ledger.reserve(segment_id=args.segment_id, operation_id=f'batch_{index:04d}',
                stage='S1', course=operation_course,
                category='engineering' if args.suite == 'S1_DIAG' else 'evaluation',
                num_envs=args.batch_size, max_control_steps=batch['control_steps'],
                checkpoint_sha256=digest, reason='frozen_first_terminal_suite',
                case_sha256=manifest['manifest_sha256'], obligation_id=args.obligation_id,
                case_sha256s=batch['case_sha256s'])
            env = None
            result = None
            settled = False
            try:
                env = V09S1Env(cases, device=args.device, seed=42)
                geometry = geometry_certificate(env, code_sha256=snapshot['code_sha256'])
                (output / f'batch_{index:04d}_geometry.json').write_text(canonical_json(geometry) + '\n')
                teacher_obs, student_obs = env.observation_pair()
                with torch.no_grad():
                    obs = teacher_obs if args.policy == 'teacher' else student_obs
                    audit = dict(observation={key: value.detach().cpu() for key, value in obs.items()},
                                 mean=policy(obs).detach().cpu(), value=policy.evaluate(obs).detach().cpu())
                result = evaluate_batch(env, policy, policy_map=args.policy, max_steps=batch['control_steps'],
                                        audit_teacher=audit_teacher, audit_every=50)
                for record, case in zip(result['records'], cases):
                    record.update(operation_course=operation_course, case_id=case['case_id'], case_sha256=case['case_sha256'],
                                  checkpoint_sha256=digest, manifest_sha256=manifest['manifest_sha256'])
                torch.save(dict(identity=dict(sds_sha256=spec.SDS_SHA256,
                                code_sha256=snapshot['code_sha256'], manifest_sha256=manifest['manifest_sha256'],
                                policy_checkpoint_sha256=digest, teacher_source_sha256=spec.SOURCE_SHA256, suite=args.suite),
                                geometry=geometry, evidence=result['evidence'], fixed_observation_audit=audit,
                                diagnostic_samples=result['audit_samples'], case_sha256s=batch['case_sha256s']),
                           output / f'batch_{index:04d}.pt')
                with (output / 'episodes.jsonl').open('a') as stream:
                    for record in result['records']:
                        stream.write(canonical_json(record) + '\n')
                ledger.commit(operation, actual_control_steps=result['control_steps'])
                settled = True
                all_records.extend(result['records'])
                print(json.dumps(dict(batch=index + 1, total_batches=len(plan['batches']),
                    cases_completed=len(all_records), control_steps=result['control_steps'],
                    physical_control_samples=result['physical_control_samples'])), flush=True)
            except BaseException as error:
                # Some substeps may have executed before the exception. Charge
                # the reserved upper bound and invalidate the entire suite.
                if not settled:
                    if result is not None:
                        ledger.commit(operation, actual_control_steps=result['control_steps'], status='interrupted')
                    else:
                        ledger.commit(operation, status='unknown')
                ledger.invalidate_obligation('S1', args.obligation_id, reason=type(error).__name__)
                (output / 'engineering_error.json').write_text(json.dumps(dict(
                    batch=index, error_type=type(error).__name__, error=str(error))) + '\n')
                raise
            finally:
                if env is not None:
                    try:
                        env.close()
                    except BaseException as error:
                        ledger.invalidate_obligation('S1', args.obligation_id,
                                                     reason='environment_close_' + type(error).__name__)
                        (output / 'engineering_error.json').write_text(json.dumps(dict(
                            batch=index, error_type=type(error).__name__, error=str(error))) + '\n')
                        raise
        if len(all_records) != plan['case_count']:
            raise RuntimeError('incomplete fixed denominator')
        summary = dict(final_unseal_proof=final_unseal_proof, suite=args.suite, policy=args.policy, total=len(all_records),
            success=sum(bool(row['success']) for row in all_records),
            failure=sum(bool(row['failure']) for row in all_records),
            timeout=sum(bool(row['timeout']) for row in all_records),
            checkpoint_sha256=digest, manifest_sha256=manifest['manifest_sha256'],
            code_sha256=snapshot['code_sha256'])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        ledger.close_obligation('S1', args.obligation_id, reason='all_frozen_cases_completed_and_saved')
    finally:
        del policy


if __name__ == '__main__':
    main()
