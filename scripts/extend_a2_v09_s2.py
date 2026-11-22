"""Append an engineering debug obligation to an existing S2 ledger (append-only).

The extension cannot modify any historical row or the original ten obligation
limits.  It raises the nonlearning envelope by exactly the new obligation limit
and enforces the 4e8 nonlearning / 5e7 S2.0 sub-cap guards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tasks.velocity.v09 import s2_ledger, s2_spec


def build_item(args):
    return dict(
        obligation_id=args.obligation_id,
        name=args.name,
        category='engineering',
        subcap=args.subcap,
        control_samples=int(args.samples),
        basis=args.basis)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path,
                        default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--run-id')
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--obligation-id', default='s2_0_pipeline_debug')
    parser.add_argument('--name', default='S2.0 pipeline/debug smoke, parity and retries')
    parser.add_argument('--subcap', choices=('S2_0', 'none'), default='S2_0')
    parser.add_argument('--basis',
                        default='SDS §4.3 / DSH-V8-2: pipeline debug separated from the 150-branch matrix')
    parser.add_argument('--reason',
                        default='append S2.0 pipeline/debug obligation without touching matrix history')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    if not args.ledger.exists():
        raise SystemExit(f'S2 ledger does not exist: {args.ledger}')
    run_id = s2_ledger.read_run_id(args.ledger)
    if args.run_id is not None and args.run_id != run_id:
        raise SystemExit(f'run_id mismatch: {args.run_id!r} != {run_id!r}')
    ledger = s2_ledger.S2Ledger(args.ledger, run_id=run_id)
    state = ledger.state()
    item = build_item(args)
    if args.subcap == 'none':
        item.pop('subcap')
    s2_ledger.S2Ledger.validate_extension_item(item, state)
    output = dict(dry_run=not args.write, run_id=run_id, item=item,
                  before=dict(envelope=state['envelope'],
                              learning_total=state['learning_total']),
                  after=dict(envelope=state['envelope'] + item['control_samples'],
                             learning_total=state['learning_total']),
                  original_limits={key: value['limit'] for key, value in
                                   state['obligations'].items()})
    if args.write:
        receipt = ledger.extend_stage(
            item, operation_id=f"stage_extend:{item['obligation_id']}",
            reason=args.reason)
        output['receipt'] = receipt
        output['state'] = ledger.state()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
