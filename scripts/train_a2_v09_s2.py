"""S2.1 training launcher: CPU dry-run only until Q021 reservation/GPU window.

No optimizer, model, environment or physics is constructed here.  The dry-run
validates the S2 identity, entry weight, PPO recipe and resource choice and
prints the plan; non-dry-run is explicitly gated.
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
from src.tasks.velocity.v09 import s2_runner, s2_spec
from src.tasks.velocity.v09.cases import canonical_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=1024,
                        choices=s2_spec.S2_ENV_CANDIDATES)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    identity = s2_spec.validate_identity()
    plan = s2_runner.dry_run_plan(num_envs=args.num_envs)
    if not args.dry_run:
        raise SystemExit('S2.1 training is gated: Q021 reservation and a GPU '
                         'execution window are required; use --dry-run')
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


if __name__ == '__main__':
    raise SystemExit(main())
