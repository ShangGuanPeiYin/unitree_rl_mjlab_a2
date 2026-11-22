"""T14: build S2.1 gate reference caches (L40 s1_least, 6400 flat).

Thin, explicit wrapper around the frozen-case reference runner.  The L40
cache uses the frozen S1 entry (s1_least.pt); the 6400 flat cache uses the
unique original 6400 weights through the exact 40x30 migration.  It never
edits the frozen cases, never overwrites an existing cache, and never writes
the ledger in --dry-run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
import collect_a2_v09_s2_gate_reference as collector
from src.tasks.velocity.v09 import s2_gate_cases, s2_spec, spec as s1_spec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=('gate_A', 'gate_B'))
    parser.add_argument('--cache-kind', required=True, choices=('l40', '6400'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--obligation-id')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not (args.dry_run or args.execute):
        raise SystemExit('use --dry-run or --execute')
    manifest_path = args.manifest or (
        ROOT / 'logs/a2_v09/s2/gate_cases/v1' /
        ('S2_GATE_A.json' if args.suite == 'gate_A' else 'S2_GATE_B.json'))
    manifest = s2_gate_cases.load_gate_manifest(
        manifest_path, expected_suite=args.suite, course='S2.1')
    if args.cache_kind == 'l40':
        reference = 's1_l40'
        groups = [row for row in manifest['groups'] if row['task'] == 'legacy_l40']
        cases = [row for row in manifest['cases'] if row['task'] == 'legacy_l40']
        obligation = args.obligation_id or (
            'l40_milestone_cache_gate_A' if args.suite == 'gate_A'
            else 'l40_milestone_cache_gate_B')
        source_path = s2_spec.S1_ENTRY_WEIGHT
        source_sha = s2_spec.S1_ENTRY_WEIGHT_SHA256
    else:
        reference = '6400_flat'
        groups = [row for row in manifest['groups']
                  if row['group_id'].startswith('flat/')]
        cases = [row for row in manifest['cases'] if row['task'].startswith('flat_')]
        obligation = args.obligation_id or 'gate_admissions_up_to_40'
        source_path = s1_spec.SOURCE_PATH
        source_sha = s1_spec.SOURCE_SHA256
    cost = sum(int(row['hard_timeout_steps']) * int(row['cases'])
               for row in groups)
    if args.cache_kind == 'l40' and cost != 219_220:
        raise SystemExit(f'L40 cache cost {cost} != 219,220')
    if args.cache_kind == '6400' and cost != 234_180:
        raise SystemExit(f'6400 cache cost {cost} != 234,180')
    plan = dict(
        schema='a2_v09_s2_gate_cache_plan_v1', course='S2.1',
        suite=manifest['suite'], cache_kind=args.cache_kind,
        reference=reference, manifest_path=str(manifest_path),
        manifest_sha256=manifest['manifest_sha256'],
        manifest_file_sha256=_sha(manifest_path),
        source_path=str(source_path), source_sha256=source_sha,
        groups=len(groups), cases=len(cases), control_samples=cost,
        envs_per_batch=20, batches=len(groups), max_batch_samples=45_000,
        peak_limit_mib=7168, obligation_id=obligation,
        physics_started=False, ledger_written=False)
    if args.dry_run:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'{args.cache_kind}_{args.suite}_plan.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False,
                         indent=2))
        return 0

    # Real reference collection is delegated to the frozen-case runner.  It
    # writes no cache on a failed run and refuses overwrite.
    rc = collector.main([
        '--reference', reference, '--suite', args.suite, '--execute',
        '--manifest', str(manifest_path), '--ledger', str(args.ledger),
        '--device', args.device, '--output-dir', str(args.output_dir),
        '--obligation-id', obligation])
    if rc != 0:
        return rc
    matches = sorted(args.output_dir.glob(f'{reference}_{manifest["suite"]}_cache.json'))
    if not matches:
        raise SystemExit('cache file was not produced')
    cache = matches[-1]
    receipt = dict(
        schema='a2_v09_s2_gate_cache_receipt_v1', plan=plan,
        cache_path=str(cache), cache_sha256=_sha(cache),
        created_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        note='L40 uses s1_least; 6400 uses the exact 40x30 migration of the '
             'unique 6400 source; 20 cases per group, single process.')
    receipt_path = args.output_dir / 'cache_receipt.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
