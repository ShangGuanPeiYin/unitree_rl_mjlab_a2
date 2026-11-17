"""v0.6 frozen-test evaluation entry point (three speeds x 100 cases, >=95 each).

The case table comes from ``src/tasks/velocity/v06/evaluation.py`` (kind
``test``, the fixed ``400342`` table seed).  The policy always runs the
deterministic action mean, and the training state is snapshotted before/after so
the report's ``statistics_unchanged`` flag is computed, never hard-coded.

This script never updates the model, the normalizer, the optimizer or a counter
and never starts a training loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(str(ROOT))


def _rooted(path):
    """Run-local paths resolve against the project root, never the CWD."""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--phase', default='R')
    p.add_argument('--substage', default=None)
    p.add_argument('--kind', default='test', choices=('test', 'validation', 'diagnostic',
                                                      'flat_retention', 'stair_retention'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--output-dir', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from src.tasks.velocity.v06 import evaluation
    from src.tasks.velocity.v06 import model as model_module
    from src.tasks.velocity.v06 import ppo as ppo_module
    from src.tasks.velocity.v06 import recovery as recovery_module
    from src.tasks.velocity.v06 import runner as runner_module
    from train_a2_ideal_stairs_v06 import build_case_runner

    output = _rooted(args.output_dir)
    if output.exists():
        raise SystemExit(f'--output-dir already exists (refused): {output}')
    checkpoint = _rooted(args.checkpoint)
    state = recovery_module.preload_checkpoint(checkpoint, args.checkpoint_sha256)
    model = model_module.V06ActorCritic().to(args.device)
    recovery_module.load_model_state(model, state['model'])
    contract = dict(device=args.device, seed=int(args.seed),
                    eval_num_envs=int(args.num_envs))
    case_runner = build_case_runner(model, contract, args.phase, args.substage)
    result = evaluation.evaluate_checkpoint(
        model, args.phase, args.substage, args.kind, int(args.seed), case_runner,
        round=None, device=args.device)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(checkpoint=str(checkpoint),
                   checkpoint_sha256=args.checkpoint_sha256,
                   gate=evaluation.gate_result(result.summary),
                   summary=result.summary, evidence=result.evidence_sha256,
                   training_state=state['learning'])
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8')
    (output / 'records.json').write_text(
        json.dumps(dict(records=result.records), indent=2, sort_keys=True,
                   allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary['gate'], indent=2, sort_keys=True))
    return 0 if summary['gate']['gate_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
