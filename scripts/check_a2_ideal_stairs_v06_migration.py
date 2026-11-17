"""v0.6 ETH migration audit entry point (CPU, seed42).

Verifies the SDS 9.2 migration contract and writes a report:

* the source SHA256 is the approved ETH checkpoint;
* only the Actor MLP and the frozen 45-dimension normalizer are migrated;
* the heightmap/history fusion and the route bypass are exactly zero;
* the initial action std is exactly ``0.30`` and the Critic/Adam are fresh;
* the deterministic action mean on synthetic inputs matches ETH within ``1e-6``
  (random-sampling agreement is explicitly **not** accepted as evidence).

Exit code 0 only when every check passes.
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


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init-checkpoint', default=None)
    p.add_argument('--init-sha256', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default=None,
                   help='optional new JSON file receiving the report')
    return p


def audit(init_checkpoint=None, init_sha256=None, seed=42, device='cpu'):
    from src.tasks.velocity.v06 import model as model_module

    checkpoint = init_checkpoint or model_module.INIT_CHECKPOINT
    sha = init_sha256 or model_module.INIT_SHA256
    model = model_module.V06ActorCritic.migrate_eth_checkpoint(
        checkpoint, sha, seed=int(seed), device=device)
    source, state, forbidden = model_module.load_validated_eth(checkpoint, sha)
    report = model_module.audit_migration(model, state, seed=int(seed))
    report['source_checkpoint_sha256'] = model_module.sha256_file(checkpoint)
    report['forbidden_source_blocks'] = sorted(forbidden)
    report['action_mean']['method_note'] = (
        'deterministic mean on synthetic inputs; random action sampling equality is not '
        'accepted as migration evidence')
    checks = dict(
        sha=report['source_checkpoint_sha256'] == model_module.INIT_SHA256,
        mean_error=report['action_mean']['max_abs_error'] <= 1e-6,
        bypass_zero=bool(report['bypass_all_zero']),
        std_exact=bool(report['initial_std_exact']),
        std_value=all(abs(value - 0.30) <= 1e-7 for value in report['initial_std']),
        critic=report['critic_input_dims'] == 211,
        normalizer_frozen=bool(report['normalizer_frozen']),
        optimizer_fresh=report['optimizer'] == 'fresh_adam_not_built_by_model',
    )
    report['checks'] = checks
    report['passed'] = all(checks.values())
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    report = audit(args.init_checkpoint, args.init_sha256, args.seed, args.device)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f'--output must be a new file: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
