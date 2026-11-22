"""Freeze the S2.0 diagnostic cases and the six original H case0 records.

Pure CPU: validates the v1.2 SDS and the approved S1 entry weight, writes
create-once manifests under ``logs/a2_v09/s2_0/cases``, and prints the
receipt.  No simulator, physics step, rollout or gradient update occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_spec
from src.tasks.velocity.v09 import s2_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cases-per-group', type=int, default=5,
                        help='Q020 fixes five S2.0 cases per task/width/speed group')
    parser.add_argument('--identity-only', action='store_true',
                        help='Validate SDS/entry-weight identity without freezing')
    args = parser.parse_args()
    identity = s2_spec.validate_identity()
    if args.identity_only:
        print(json.dumps(dict(identity=identity, frozen=False), sort_keys=True, indent=2))
        return 0
    receipt = s2_cases.freeze_s2_0(args.root, cases_per_group=args.cases_per_group)
    print(json.dumps(dict(identity=identity, receipt=receipt), sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
