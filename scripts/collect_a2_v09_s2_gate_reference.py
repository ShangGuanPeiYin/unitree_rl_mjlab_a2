"""Collect one frozen S2.1 gate reference cache (6400 flat or old L40).

This is an evaluation-only tool.  It reads the frozen S2_GATE_A/B manifest,
runs the unique original 6400 teacher on the same cases, reserves the matching
evaluation obligation, and writes a versioned cache consumed by
``run_a2_v09_s2_gate.py``.  ``--dry-run`` does no physics and no ledger write.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_env, s2_gate_cases, s2_spec, spec as s1_spec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True,
                        choices=('6400_flat', 's1_l40'))
    parser.add_argument('--suite', required=True, choices=('gate_A', 'gate_B'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--obligation-id')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not (args.dry_run or args.execute):
        raise SystemExit('use --dry-run or --execute')
    if args.reference == '6400_flat':
        raise SystemExit(
            'DSH-Q034: migrated 40x30 6400 is not the original 6400 reference; '
            'this bypass is disabled (use the Q034-A original old-map path only)')
    manifest_path = args.manifest or (
        ROOT / 'logs/a2_v09/s2/gate_cases/v1' /
        ('S2_GATE_A.json' if args.suite == 'gate_A' else 'S2_GATE_B.json'))
    manifest = s2_gate_cases.load_gate_manifest(
        manifest_path, expected_suite=args.suite, course='S2.1')
    if args.reference == '6400_flat':
        groups = [row for row in manifest['groups']
                  if row['group_id'].startswith('flat/')]
        cases = [row for row in manifest['cases'] if row['task'].startswith('flat_')]
        obligation = args.obligation_id or 'gate_admissions_up_to_40'
        cache_schema = 'a2_v09_s2_gate_6400_flat_v1'
    else:
        groups = [row for row in manifest['groups']
                  if row['task'] == 'legacy_l40']
        cases = [row for row in manifest['cases'] if row['task'] == 'legacy_l40']
        suffix = 'A' if manifest['suite'] == 'S2_GATE_A' else 'B'
        obligation = args.obligation_id or f'l40_milestone_cache_gate_{suffix}'
        cache_schema = 'a2_v09_s2_gate_l40_cache_v1'
    cost = sum(int(row['hard_timeout_steps']) * int(row['cases']) for row in groups)
    if args.reference == '6400_flat' and cost != 234_180:
        raise SystemExit(f'6400 flat cache cost {cost} != 234,180')
    if args.reference == 's1_l40' and cost != 219_220:
        raise SystemExit(f'L40 cache cost {cost} != 219,220')
    plan = dict(reference=args.reference, suite=manifest['suite'],
                manifest_sha256=manifest['manifest_sha256'],
                manifest_file_sha256=_sha(manifest_path),
                groups=len(groups), cases=len(cases), control_samples=cost,
                obligation_id=obligation,
                output_schema=cache_schema, physics_started=False,
                ledger_written=False)
    if args.dry_run:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = args.output_dir or (ROOT / 'logs/a2_v09/s2/gate_reference_dry_run' / stamp)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'{args.reference}_{args.suite}_plan.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False,
                         indent=2))
        return 0

    if args.reference == 's1_l40':
        checkpoint = args.checkpoint or s2_spec.S1_ENTRY_WEIGHT
        if _sha(checkpoint) != s2_spec.S1_ENTRY_WEIGHT_SHA256:
            raise SystemExit('L40 reference checkpoint is not s1_least.pt')
    else:
        checkpoint = args.checkpoint or s1_spec.SOURCE_PATH
        if _sha(checkpoint) != s1_spec.SOURCE_SHA256:
            raise SystemExit('reference checkpoint is not the unique 6400 source')
    if not args.ledger.is_file():
        raise SystemExit('ledger missing')
    from src.tasks.velocity.v09 import s2_ledger
    from src.tasks.velocity.v09 import s2_physics
    ledger = s2_ledger.S2Ledger(
        args.ledger, run_id=s2_ledger.read_run_id(args.ledger))
    duty = ledger.state()['obligations'].get(obligation)
    if duty is None:
        raise SystemExit(f'missing obligation {obligation}')
    if duty['limit'] - duty['spent'] - duty['reserved'] < cost:
        raise SystemExit(f'insufficient {obligation} balance')
    if args.reference != 's1_l40':
        raise SystemExit('DSH-Q034: only the s1_least L40 reference path is implemented here')
    teacher = s2_physics.load_s1_actor(device=args.device)['model']
    source_sha = s2_spec.S1_ENTRY_WEIGHT_SHA256

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.output_dir or (ROOT / 'logs/a2_v09/s2/gate_reference' /
                              f'{args.reference}-{args.suite}-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        h_legacy = case['task'] == 'legacy_l40'
        env, _ = s2_physics.build_s2_env(
            case, device=args.device, seed=s2_spec.SEED, map_branch='real',
            h_legacy=h_legacy)
        reserved = int(case['hard_timeout_steps'])
        op = f'ref:{args.reference}:{case["case_sha256"][:12]}'
        ledger.reserve(obligation, reserved,
                       operation_id=f'{op}:reserve', reason='gate reference cache')
        steps = 0
        try:
            import torch
            env.reset()
            done = False
            while not done and steps < reserved:
                obs = env.observe()
                tensor_obs = {
                    key: (value[None] if isinstance(value, torch.Tensor) else value)
                    for key, value in obs.items()
                    if key in ('proprio', 'history', 'heightmap', 'route_error')}
                with torch.no_grad():
                    action = teacher(tensor_obs)
                _, done_list, _ = env.step(action[0].detach().cpu().numpy())
                steps += 1
                done = bool(done_list[0])
            trace = s2_env.jsonable(env.evidence(0))
            route = np.asarray(trace['route_x'], dtype=float)
            post_reset = np.asarray(trace.get('post_reset', [False] * len(route)),
                                    dtype=bool)
            valid = np.flatnonzero(~post_reset)
            last_index = int(valid[-1]) if valid.size else 0
            start_x = float(case['spawn']['local_position_xy'][0])
            terminal_x = float(route[last_index]) if route.size else start_x
            results.append(dict(
                group=case['group_id'], case_sha256=case['case_sha256'],
                success=bool(env.machine.success),
                failure=bool(env.machine.failure),
                timeout=bool(env.machine.timeout),
                physical_failure=bool(env.machine.failure),
                engineering_failure=False,
                steps=steps,
                start_x=start_x, terminal_x=terminal_x,
                last_valid_index=last_index,
                terminal_is_post_reset=bool(post_reset[last_index]) if post_reset.size else None,
                route_max=float(route.max()) if route.size else start_x,
                net_distance=terminal_x - start_x,
                termination_reason=getattr(env.last, 'termination_reason', None),
                trace=trace))
        except Exception:
            if steps:
                ledger.commit(obligation, steps, operation_id=f'{op}:commit',
                              reason='gate reference partial')
            unused = reserved - steps
            if unused:
                ledger.release(obligation, unused, operation_id=f'{op}:release',
                               reason='gate reference failed')
            raise
        ledger.commit(obligation, steps, operation_id=f'{op}:commit',
                      reason='gate reference actual')
        unused = reserved - steps
        if unused:
            ledger.release(obligation, unused, operation_id=f'{op}:release',
                           reason='gate reference unused')
    grouped = {}
    for row in results:
        grouped.setdefault(row['group'], []).append(row)
    groups_out = {}
    for group in groups:
        rows = grouped.get(group['group_id'], [])
        groups_out[group['group_id']] = dict(
            cases=len(rows), successes=sum(bool(row['success']) for row in rows),
            rows=rows)
    for group in groups:
        rows = grouped.get(group['group_id'], [])
        values = np.asarray([row['net_distance'] for row in rows], dtype=float)
        groups_out[group['group_id']].update(
            p10=float(np.percentile(values, 10)) if values.size else None,
            p50=float(np.percentile(values, 50)) if values.size else None,
            p90=float(np.percentile(values, 90)) if values.size else None,
            failure_classes={
                str(name): sum(1 for row in rows
                               if row.get('termination_reason') == name)
                for name in {row.get('termination_reason') for row in rows
                             if row.get('failure')}},
            timeouts=sum(bool(row.get('timeout')) for row in rows),
            engineering_failures=sum(bool(row.get('engineering_failure')) for row in rows))
    if args.reference == 's1_l40':
        cases_dir = out / 'cases'
        cases_dir.mkdir(parents=True, exist_ok=True)
        case_files = []
        for row in results:
            path = cases_dir / f"{row['case_sha256']}.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
            case_files.append(dict(case_sha256=row['case_sha256'],
                                   path=str(path), sha256=_sha(path)))
        cache = dict(schema=cache_schema, suite=manifest['suite'],
                     manifest_sha256=manifest['manifest_sha256'],
                     source_sha256=source_sha, model_sha256=source_sha,
                     checkpoint=str(checkpoint), groups=groups_out,
                     case_files=case_files,
                     generated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
        path = out / 'reference_cache.json'
    else:
        cache = dict(schema=cache_schema, suite=manifest['suite'],
                     manifest_sha256=manifest['manifest_sha256'],
                     source_sha256=source_sha,
                     checkpoint=str(checkpoint), groups=groups_out,
                     generated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
        path = out / f'{args.reference}_{args.suite}_cache.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), groups=len(groups_out),
                          cases=len(results), cost=cost), ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
