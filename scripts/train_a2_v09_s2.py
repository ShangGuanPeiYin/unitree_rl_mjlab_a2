"""S2.1 training launcher with explicit admission gates and resume.

CPU/fake execution is available for stack tests.  Real 1024-env execution is
still gated on the probe report and independent admission evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import (s2_checkpoint, s2_ledger, s2_physics,
                                    s2_resources, s2_runner, s2_spec,
                                    s2_train, s2_train_env, s2_train_real)
from src.tasks.velocity.v09.cases import canonical_json


def _load_json(path, name):
    if path is None or not Path(path).exists():
        raise SystemExit(f'{name} is required')
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=1024,
                        choices=s2_spec.S2_ENV_CANDIDATES)
    parser.add_argument('--course', default='S2.1',
                        choices=s2_spec.S2_COURSE_NAMES)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fallback-evidence', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--probe-report', type=Path)
    parser.add_argument('--admission-evidence', type=Path)
    parser.add_argument('--fake-backend', action='store_true')
    parser.add_argument('--max-rollouts', type=int)
    parser.add_argument('--checkpoint-every-rollouts', type=int)
    parser.add_argument('--attempt-id', default='s2.1-r1')
    args = parser.parse_args(argv)
    identity = s2_spec.validate_identity()
    course_key = s2_spec.course_key(args.course)
    training_plan = s2_train.rollout_plan(args.num_envs, course=course_key)
    effective_cap = training_plan['max_rollouts'] * training_plan['samples_per_rollout']
    if effective_cap > s2_spec.S2_COURSE_LEARNING_CAPS[course_key]:
        raise SystemExit(f'training rollout plan exceeds {args.course} learning cap')
    if args.num_envs == 512:
        evidence = _load_json(args.fallback_evidence, '--fallback-evidence')
        if evidence.get('num_envs') != 1024 or evidence.get('admitted') is not False:
            raise SystemExit('fallback evidence must show 1024 admitted=false')
        if not evidence.get('failure_reasons'):
            raise SystemExit('fallback evidence must record at least one 1024 failure')
    plan = s2_runner.dry_run_plan(num_envs=args.num_envs)
    plan['training_plan'] = training_plan
    plan['course'] = args.course
    if not (args.dry_run or args.execute):
        raise SystemExit('use --dry-run or --execute')
    if args.dry_run:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = args.output_dir or (ROOT / 'logs/a2_v09/s2/train_dry_run' / stamp)
        directory.mkdir(parents=True, exist_ok=True)
        summary = dict(identity=identity, plan=plan, device=args.device,
                       dry_run=True, physics_started=False, training_started=False)
        output = directory / 'train_dry_run_summary.json'
        if output.exists():
            raise FileExistsError(f'refusing to overwrite {output}')
        output.write_text(canonical_json(summary) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not args.ledger.exists():
        raise SystemExit('S2 ledger missing')
    if args.probe_report is None or not args.probe_report.exists():
        raise SystemExit('--probe-report is required before GPU allocation')
    if args.admission_evidence is None or not args.admission_evidence.exists():
        raise SystemExit('--admission-evidence is required before GPU allocation')
    probe = _load_json(args.probe_report, '--probe-report')
    admission = _load_json(args.admission_evidence, '--admission-evidence')
    if probe.get('num_envs') != args.num_envs or probe.get('admitted') is not True:
        raise SystemExit('probe report does not admit this num_envs')
    try:
        s2_resources.validate_admission(
            admission, probe_path=args.probe_report, num_envs=args.num_envs)
    except Exception as error:
        raise SystemExit(f'admission invalid: {error}')
    if not args.fake_backend and args.device == 'cpu':
        raise SystemExit('real S2.1 training requires a CUDA device or --fake-backend')
    run_id = s2_ledger.read_run_id(args.ledger)
    ledger = s2_ledger.S2Ledger(args.ledger, run_id=run_id)
    ledger_state = ledger.state()
    obligation_id = f'learning_{course_key}'
    learning = ledger_state.get('learning_obligations', {}).get(obligation_id)
    if learning is None:
        raise SystemExit(f'{obligation_id} obligation missing from S2 ledger')
    if int(learning.get('limit', 0)) < effective_cap:
        raise SystemExit(f'{obligation_id} obligation is smaller than the course cap')
    if int(learning.get('reserved', 0)) != 0:
        raise SystemExit(f'{obligation_id} has a pending reservation; release it before '
                         'a new/resumed run so the crashed rollout cannot double-charge')
    matrix_manifest = __import__('src.tasks.velocity.v09.s2_cases', fromlist=['load_frozen_manifest']).load_frozen_manifest(
        ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG.json', ROOT)
    budget = s2_train.S2LearningBudget(
        ledger, obligation_id=obligation_id,
        samples_per_rollout=training_plan['samples_per_rollout'])
    role_plan = s2_train_env.build_role_plan(args.num_envs, course=args.course)
    if args.fake_backend:
        env = s2_train_env.S2FakeTrainEnv(num_envs=args.num_envs,
                                          role_plan=role_plan)
    else:
        env = s2_train_real.S2TrainEnv(num_envs=args.num_envs,
                                       device=args.device, course=args.course)
    actor_bundle = s2_physics.load_s1_actor(device=args.device)
    model = actor_bundle['model']
    teacher = None
    if not args.fake_backend:
        from src.tasks.velocity.v09.model import V09ActorCritic
        teacher = V09ActorCritic.migrate_6400_checkpoint(device=args.device)
        teacher.eval()
    trainer = s2_train.S2Trainer(
        model=model, env=env, ledger=ledger, output_dir=args.output_dir,
        device=args.device, attempt_id=args.attempt_id,
        ledger_run_id=run_id, course=args.course, teacher=teacher)
    if args.resume is not None:
        trainer.resume(args.resume, device=args.device)
        budget.rollouts = trainer.rollouts
    max_rollouts = min(args.max_rollouts or training_plan['max_rollouts'],
                       training_plan['max_rollouts'])
    checkpoint_every = (args.checkpoint_every_rollouts
                        or training_plan['checkpoint_every_rollouts'])
    if max_rollouts <= 0 or checkpoint_every <= 0:
        raise SystemExit('max_rollouts and checkpoint_every_rollouts must be positive')
    output = args.output_dir or (ROOT / 'logs/a2_v09/s2/train' / args.attempt_id)
    output.mkdir(parents=True, exist_ok=True)
    for rollout in range(max_rollouts):
        if trainer.learning_samples >= effective_cap:
            break
        result = trainer.train_rollout(budget=budget)
        if (trainer.rollouts % checkpoint_every == 0
                or rollout == max_rollouts - 1
                or trainer.learning_samples >= effective_cap):
            path = output / f's2_1_{trainer.learning_samples:012d}.pt'
            sha = trainer.save(path, case_manifest_sha256=matrix_manifest['manifest_sha256'],
                               manifest_sha256=matrix_manifest['manifest_sha256'])
            result['checkpoint'] = str(path)
            result['checkpoint_sha256'] = sha
        print(json.dumps(dict(rollout=trainer.rollouts, **result),
                         ensure_ascii=False), flush=True)
        if trainer.learning_samples >= effective_cap:
            break
    print(json.dumps(dict(status='completed', rollouts=trainer.rollouts,
                          learning_samples=trainer.learning_samples,
                          course=args.course, effective_cap=effective_cap,
                          samples_per_rollout=training_plan['samples_per_rollout'],
                          learning_rate=trainer.learning_rate,
                          std=trainer.model.std.detach().cpu().tolist()),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
