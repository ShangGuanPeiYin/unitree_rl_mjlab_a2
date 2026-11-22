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
                                    s2_train, s2_train_env)
from src.tasks.velocity.v09.cases import canonical_json


def _load_json(path, name):
    if path is None or not Path(path).exists():
        raise SystemExit(f'{name} is required')
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=1024,
                        choices=s2_spec.S2_ENV_CANDIDATES)
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
    if args.num_envs == 512:
        evidence = _load_json(args.fallback_evidence, '--fallback-evidence')
        if evidence.get('num_envs') != 1024 or evidence.get('admitted') is not False:
            raise SystemExit('fallback evidence must show 1024 admitted=false')
    plan = s2_runner.dry_run_plan(num_envs=args.num_envs)
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
    if not args.probe_report.exists():
        raise SystemExit('--probe-report is required before GPU allocation')
    if not args.admission_evidence.exists():
        raise SystemExit('--admission-evidence is required before GPU allocation')
    probe = _load_json(args.probe_report, '--probe-report')
    admission = _load_json(args.admission_evidence, '--admission-evidence')
    if probe.get('num_envs') != args.num_envs or probe.get('admitted') is not True:
        raise SystemExit('probe report does not admit this num_envs')
    if admission.get('admitted') is not True:
        raise SystemExit('independent admission is not true')
    if not args.fake_backend:
        raise SystemExit('real 1024-env S2.1 assembly is not installed; '
                         'use --fake-backend for CPU stack tests')
    run_id = s2_ledger.read_run_id(args.ledger)
    ledger = s2_ledger.S2Ledger(args.ledger, run_id=run_id)
    matrix_manifest = __import__('src.tasks.velocity.v09.s2_cases', fromlist=['load_frozen_manifest']).load_frozen_manifest(
        ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG.json', ROOT)
    budget = s2_train.S2LearningBudget(ledger)
    role_plan = s2_train_env.build_role_plan(args.num_envs)
    env = s2_train_env.S2FakeTrainEnv(num_envs=args.num_envs, role_plan=role_plan)
    actor_bundle = s2_physics.load_s1_actor(device=args.device)
    model = actor_bundle['model']
    trainer = s2_train.S2Trainer(
        model=model, env=env, ledger=ledger, output_dir=args.output_dir,
        device=args.device, attempt_id=args.attempt_id,
        ledger_run_id=run_id)
    if args.resume is not None:
        trainer.resume(args.resume, device=args.device)
        budget.rollouts = trainer.rollouts
    max_rollouts = args.max_rollouts or s2_train.S2_1_MAX_ROLLOUTS
    checkpoint_every = args.checkpoint_every_rollouts or \
        s2_train.S2_1_CHECKPOINT_ROLLOUTS
    output = args.output_dir or (ROOT / 'logs/a2_v09/s2/train' / args.attempt_id)
    output.mkdir(parents=True, exist_ok=True)
    for rollout in range(max_rollouts):
        result = trainer.train_rollout(budget=budget)
        if trainer.rollouts % checkpoint_every == 0 or rollout == max_rollouts - 1:
            path = output / f's2_1_{trainer.learning_samples:012d}.pt'
            sha = trainer.save(path, case_manifest_sha256=matrix_manifest['manifest_sha256'],
                               manifest_sha256=matrix_manifest['manifest_sha256'])
            result['checkpoint'] = str(path)
            result['checkpoint_sha256'] = sha
        print(json.dumps(dict(rollout=trainer.rollouts, **result),
                         ensure_ascii=False), flush=True)
        if trainer.learning_samples >= s2_train.S2_1_SAMPLES_PER_ROLLOUT * \
                s2_train.S2_1_MAX_ROLLOUTS:
            break
    print(json.dumps(dict(status='completed', rollouts=trainer.rollouts,
                          learning_samples=trainer.learning_samples,
                          learning_rate=trainer.learning_rate,
                          std=trainer.model.std.detach().cpu().tolist()),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
