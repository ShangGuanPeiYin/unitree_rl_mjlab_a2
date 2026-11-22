"""S2 final acceptance planner (27 groups x100, sealed final sets)."""
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
from src.tasks.velocity.v09 import s2_acceptance, s2_budget, s2_spec


def final_groups():
    groups = []
    for speed in s2_spec.SPEEDS:
        groups.append(dict(id=f'flat_walk/{speed}', task='flat_walk', speed=speed,
                           stop_mode='none', post_distance=None))
    for speed in s2_spec.SPEEDS:
        for mode in ('instant', 'ramp_2s'):
            groups.append(dict(id=f'flat_stop/{speed}/{mode}', task='flat_stop',
                               speed=speed, stop_mode=mode, post_distance=None))
    for speed in s2_spec.SPEEDS:
        for mode in ('instant', 'ramp_2s'):
            for width in s2_spec.PLATFORM_WIDTHS:
                groups.append(dict(id=f'l40_stop/{speed}/{width}/{mode}',
                                   task='stairs_stop', speed=speed,
                                   stop_mode=mode, width=width,
                                   post_distance=10.0))
    for speed in s2_spec.SPEEDS:
        for width in s2_spec.PLATFORM_WIDTHS:
            groups.append(dict(id=f'l40_walk/{speed}/{width}',
                               task='stairs_walk', speed=speed,
                               stop_mode='none', width=width,
                               post_distance=10.0))
    if len(groups) != 27:
        raise AssertionError('final acceptance must have 27 groups')
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sets', type=int, default=s2_budget.FINAL_SETS)
    args = parser.parse_args(argv)
    if not (args.dry_run or args.execute):
        raise SystemExit('use --dry-run or --execute')
    groups = final_groups()
    cost = s2_budget.final_acceptance_cost()
    plan = dict(schema='a2_v09_s2_final_plan_v1', stage='S2',
                groups=len(groups), cases_per_group=s2_budget.FINAL_CASES,
                total_cases=len(groups) * s2_budget.FINAL_CASES,
                threshold=s2_acceptance.FINAL_THRESHOLD, sets=args.sets,
                per_set_samples=cost['per_set_samples'],
                three_sets_samples=cost['three_sets_samples'],
                original_6400_flat_samples=cost['original_6400_flat_samples'],
                group_list=groups, checkpoint=str(args.checkpoint) if args.checkpoint else None,
                sealed=True, physics_started=False, ledger_written=False,
                acceptance=dict(status='pending_final_evidence', accepted=False,
                                flag='S2_WALK_STOP_ACCEPTED'),
                h_series='reuse report_a2_v09_s2_0 H logic after final evidence')
    if args.dry_run:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = args.output_dir or (ROOT / 'logs/a2_v09/s2/final_dry_run' / stamp)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'S2_final_plan.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False,
                         indent=2))
        return 0
    # Real final acceptance requires the sealed frozen final manifests, the
    # official model, and the 6400 flat cache generated at unsealing time.
    raise SystemExit('S2 final execute is gated until final case manifests are '
                     'frozen and unsealed; use --dry-run now')


if __name__ == '__main__':
    raise SystemExit(main())
