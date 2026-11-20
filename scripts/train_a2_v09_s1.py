"""S1 pure distillation; physics requires complete independently approved evidence."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.budget import AppendOnlyLedger
from src.tasks.velocity.v09.s1_runner import (PlanningRequired, RealS1Trainer,
    ScheduleJournal, SubprocessEvidenceHooks, file_sha256, validate_admissions, validate_learning_approval, load_current_checkpoint, create_identity_context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--segment-id', required=True)
    parser.add_argument('--resource-admission', type=Path, required=True)
    parser.add_argument('--baseline-admission', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--evidence-identity', type=Path)
    parser.add_argument('--evidence-identity-sha256')
    parser.add_argument('--num-envs', type=int, choices=spec.ENV_CANDIDATES, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout-steps', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--minibatches', type=int, default=4)
    parser.add_argument('--microbatch', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--checkpoint-every-samples', type=int, default=5_000_000)
    parser.add_argument('--approval-report', type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-checkpoint-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    fixed = (args.seed, args.rollout_steps, args.epochs, args.minibatches, args.microbatch,
             args.learning_rate, args.checkpoint_every_samples)
    if fixed != (42, 48, 5, 4, 512, 3e-5, 5_000_000):
        raise ValueError('S1 recipe parameters are fixed by SDS, not CLI tuning options')
    spec.validate_sources()
    if spec.code_snapshot()['code_sha256'] != args.code_sha256:
        raise ValueError('current code snapshot differs from independently approved code SHA')
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint_sha256:
            raise ValueError('explicit resume checkpoint SHA required')
        load_current_checkpoint(args.resume_checkpoint, device='cpu', code_sha256=args.code_sha256,
                                checkpoint_sha256=args.resume_checkpoint_sha256)
    identity_context = create_identity_context(approval_path=args.evidence_identity,
        approval_sha256=args.evidence_identity_sha256, code_sha256=args.code_sha256,
        run_id=args.run_id, run_dir=args.run_dir)
    if not (args.run_dir / 'budget.jsonl').is_file():
        raise PlanningRequired('run prepare --reserve and all nonlearning admissions before S1 learning')
    ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
    admissions = validate_admissions(run_dir=args.run_dir, resource_path=args.resource_admission,
        baseline_path=args.baseline_admission, calibration_path=args.calibration,
        code_sha256=args.code_sha256, ledger=ledger, identity_context=identity_context)
    if args.num_envs != admissions['num_envs']:
        raise ValueError('explicit N differs from the largest stable resource admission')
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint_sha256 or file_sha256(args.resume_checkpoint) != args.resume_checkpoint_sha256:
            raise ValueError('explicit resume checkpoint SHA required and must match')
    if not args.execute:
        print(json.dumps(dict(physics_started=False, num_envs=args.num_envs,
            samples_per_iteration=args.num_envs * 48, adam_updates_per_iteration=20,
            adaptation_iterations=20_000_000 // (args.num_envs * 48),
            main_iterations=630_000_000 // (args.num_envs * 48),
            retry_iterations_each=100_000_000 // (args.num_envs * 48),
            maximum_gate_attempts=11, maximum_final_suites=3, value_scale=admissions['value_scale']), indent=2))
        return
    if args.approval_report is None:
        raise PlanningRequired('independent S1 implementation approval and Q008 evidence driver required')
    approval = json.loads(args.approval_report.read_text())
    validate_learning_approval(approval, run_dir=args.run_dir, run_id=args.run_id,
                               code_sha256=args.code_sha256, identity_context=identity_context)
    hooks = SubprocessEvidenceHooks(python=sys.executable, calibration_path=args.calibration,
        run_dir=args.run_dir, run_id=args.run_id, segment_id=args.segment_id,
        device=args.device, code_sha256=args.code_sha256, identity_context=identity_context)
    with (args.run_dir / 'formal_training.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PlanningRequired('another formal learner owns this external ledger/episode stream') from error
        trainer = RealS1Trainer(run_dir=args.run_dir, run_id=args.run_id,
            segment_id=args.segment_id, device=args.device, code_sha256=args.code_sha256,
            admissions=admissions, ledger=ledger,
            journal=ScheduleJournal(args.run_dir / 's1_schedule.jsonl'),
            evidence_hooks=hooks, resume=args.resume_checkpoint)
        result = trainer.run()
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
