"""Recompute and (only with --write) atomically reserve the S2 envelope.

The plan is never hard-coded: every number is recomputed from the frozen
S2_DIAG/H manifests and ``s2_budget``.  Without ``--write`` no ledger file is
created or touched.
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
from src.tasks.velocity.v09 import s2_budget, s2_cases, s2_ledger, s2_spec


def load_plan(root=ROOT):
    root = Path(root)
    cases = root / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases / 'S2_DIAG.json', root)
    h_series = s2_cases.load_frozen_manifest(cases / 'h_series_case0.json', root)
    plan = s2_budget.build_reservation_plan(matrix, h_series)
    s2_ledger.validate_plan(plan)
    return plan


def table(plan):
    rows = []
    for item in plan['items'] + plan['learning_items']:
        rows.append(dict(obligation_id=item['obligation_id'],
                         category=item['category'],
                         control_samples=item['control_samples'],
                         basis=item['basis'], name=item['name']))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--run-id', default='a2-v09-s2-q021-r1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    if args.write and args.dry_run:
        raise SystemExit('choose --dry-run or --write, not both')
    plan = load_plan(args.root)
    summary = dict(stage=plan['stage'], sds_sha256=plan['sds_sha256'],
                   plan_sha256=plan['plan_sha256'],
                   nonlearning_total=plan['total_control_samples'],
                   learning_total=plan['learning_total_control_samples'],
                   stage_total=plan['stage_total_control_samples'],
                   s2_0_subcap_used=plan['s2_0_subcap_used'],
                   physics_substeps=plan['physics_substeps'],
                   billing=plan['billing'], items=table(plan),
                   dry_run=not args.write, ledger=str(args.ledger))
    if not args.write:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    ledger = s2_ledger.S2Ledger(args.ledger, run_id=args.run_id)
    state = ledger.state()
    if state['envelope'] or state['obligations']:
        print(json.dumps(dict(summary, written=False,
                              refused='S2 stage already reserved'),
                         ensure_ascii=False, indent=2))
        return 2
    receipt = ledger.reserve_stage(plan)
    post = ledger.state()
    summary.update(written=True, receipt=receipt,
                   envelope=post['envelope'],
                   obligations={key: dict(limit=row['limit'], spent=row['spent'],
                                          reserved=row['reserved'])
                                for key, row in post['obligations'].items()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
