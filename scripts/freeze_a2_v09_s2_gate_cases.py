"""Freeze the S2.1 A/B gate case manifests (append-only, no physics)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_gate_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'logs/a2_v09/s2/gate_cases/v1')
    parser.add_argument('--version', default=s2_gate_cases.GATE_VERSION)
    parser.add_argument('--course', default='S2.1', choices=('S2.1','S2.2','S2.3'))
    args = parser.parse_args(argv)
    if args.course == 'S2.1':
        receipt = s2_gate_cases.freeze_gate_cases(
            args.output_dir, version=args.version)
    else:
        receipt = s2_gate_cases.freeze_course_cases(args.output_dir, args.course)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
