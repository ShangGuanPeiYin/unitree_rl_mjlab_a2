"""S2 resource probe (1024 first, documented 512 fallback).

`--plan-only` is CPU-only.  `--execute` requires the real 1024-env training
assembly and is intentionally refused until that adapter is installed; it must
not silently measure a fake backend as if it were GPU admission.
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
from src.tasks.velocity.v09 import s2_resources, s2_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=1024,
                        choices=s2_spec.S2_ENV_CANDIDATES)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--evaluation-timeout-steps', type=int,
                        default=s2_resources.PROBE_EVAL_TIMEOUT_STEPS)
    args = parser.parse_args(argv)
    if not (args.plan_only or args.execute):
        raise SystemExit('use --plan-only or --execute')
    plan = s2_resources.probe_plan(
        args.num_envs, evaluation_timeout_steps=args.evaluation_timeout_steps)
    if args.plan_only:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = args.output_dir or (ROOT / 'logs/a2_v09/s2/probe_plan' / stamp)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'probe_plan.json'
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(
        'real 1024-env probe requires the multi-env training assembly; '
        'use --plan-only until the GPU window is approved')
