"""Q034-A: original 6400 flat reference on the old 3x48x32 teacher map.

The frozen S2_GATE_A/B flat cases are executed through the v09 S1 adapter
(V07 physics + native 3x48x32 student/teacher map), preserving the S2 case's
spawn/streams/command/stop timing/hard timeout.  The policy is the untouched
original 6400 actor; no migrated 40x30 model is accepted.
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
from src.tasks.velocity.v09 import s2_env, s2_gate_cases, spec as s1_spec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _s1_case(case):
    """Same S2 case identity/spawn/command, executed by the S1 old-map adapter."""
    item = json.loads(json.dumps(case))
    item['stage'] = 'S1'
    item['termination_contract'] = 'q034_original_6400_old_map'
    return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=('gate_A', 'gate_B'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--ledger', type=Path, default=ROOT / 'logs/a2_v09/s2/ledger.jsonl')
    parser.add_argument('--obligation-id', default='gate_admissions_up_to_40')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
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
    groups = [row for row in manifest['groups'] if row['group_id'].startswith('flat/')]
    cases = [row for row in manifest['cases'] if row['task'].startswith('flat_')]
    cost = sum(int(row['hard_timeout_steps']) * int(row['cases']) for row in groups)
    if cost != 234_180:
        raise SystemExit(f'6400 flat cache cost {cost} != 234,180')
    plan = dict(
        schema='a2_v09_s2_gate_6400_flat_plan_v1', suite=manifest['suite'],
        manifest_sha256=manifest['manifest_sha256'],
        manifest_file_sha256=_sha(manifest_path),
        model_sha256=s1_spec.SOURCE_SHA256,
        source_sha256=s1_spec.SOURCE_SHA256, model_type='original_6400',
        observation_shape=[3, 48, 32], map='native_v07_48x32',
        groups=len(groups), cases=len(cases), control_samples=cost,
        envs_per_batch=20, batches=len(groups), peak_limit_mib=7168,
        action_consistency_check='same old-map input against S1 teacher path',
        physics_started=False, ledger_written=False)
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / '6400_gate_plan.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False,
                         indent=2))
        return 0

    if not args.ledger.is_file():
        raise SystemExit('ledger missing')
    from src.tasks.velocity.v09 import s2_ledger
    from src.tasks.velocity.v09.env import V09S1Env
    from src.tasks.velocity.v09.model import Frozen6400Teacher
    ledger = s2_ledger.S2Ledger(args.ledger,
                                run_id=s2_ledger.read_run_id(args.ledger))
    duty = ledger.state()['obligations'].get(args.obligation_id)
    if duty is None:
        raise SystemExit(f'missing obligation {args.obligation_id}')
    if duty['limit'] - duty['spent'] - duty['reserved'] < cost:
        raise SystemExit('insufficient gate obligation balance')
    teacher = Frozen6400Teacher.from_6400_checkpoint(
        s1_spec.SOURCE_PATH, s1_spec.SOURCE_SHA256, device=args.device)
    teacher.eval()
    # Action consistency: same native 3x48x32 old-map input, same original
    # teacher path, compared against a fresh reload of the identical source.
    teacher2 = Frozen6400Teacher.from_6400_checkpoint(
        s1_spec.SOURCE_PATH, s1_spec.SOURCE_SHA256, device=args.device)
    import torch
    probe = dict(
        proprio=torch.zeros(2, 45, device=args.device),
        history=torch.zeros(2, 5, 45, device=args.device),
        heightmap=torch.zeros(2, 3, 48, 32, device=args.device),
        route_error=torch.zeros(2, 3, device=args.device))
    with torch.no_grad():
        a1 = teacher(probe)
        a2 = teacher2(probe)
    delta = float((a1 - a2).abs().max().item())
    consistency = dict(check='same_old_map_action_original_teacher_reload',
                       passed=bool(torch.equal(a1, a2)), max_abs_delta=delta,
                       note='3x48x32 input, no migrated model')
    if not consistency['passed']:
        raise SystemExit('original 6400 action consistency self-check failed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cases_dir = out / 'cases'
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in groups:
        batch = [case for case in cases if case['group_id'] == group['group_id']]
        if len(batch) != 20:
            raise SystemExit('gate group must have 20 cases')
        env = V09S1Env([_s1_case(case) for case in batch], device=args.device,
                       seed=42, course='S1.2')
        reserved = int(group['hard_timeout_steps']) * 20
        op = f'q034a:{manifest["suite"]}:{group["group_id"]}'
        ledger.reserve(args.obligation_id, reserved,
                       operation_id=f'{op}:reserve',
                       reason='Q034-A original 6400 flat cache')
        steps = 0
        try:
            done = False
            while not done and steps < int(group['hard_timeout_steps']):
                obs = env._observations(update=True)
                tensor_obs = {key: value for key, value in obs.items()
                              if key in ('proprio', 'history', 'heightmap',
                                         'route_error')}
                with torch.no_grad():
                    action = teacher(tensor_obs)
                _, _, terminated, _, info = env.step(action)
                steps += 1
                done = bool(terminated.all())
            evidence = info.get('v09_evidence', {})
            for index, case in enumerate(batch):
                trace = s2_env.jsonable(evidence.get(index, {}))
                route = np.asarray(trace.get('position', []), dtype=float)
                if route.size:
                    terminal_x = float(route[-1, 0])
                else:
                    terminal_x = float(env.v09_cases[index]['spawn']['local_position_xy'][0])
                start_x = float(case['spawn']['local_position_xy'][0])
                metrics = s2_env.jsonable(env._completed_metrics.get(index, {}))
                outcomes = info.get('v09_flat_outcomes', {})
                rows.append(dict(
                    group=case['group_id'], case_sha256=case['case_sha256'],
                    start_x=start_x, terminal_x=terminal_x,
                    net_distance=terminal_x - start_x,
                    success=bool(metrics.get('quality_pass', False)),
                    failure=not bool(metrics.get('quality_pass', False)),
                    timeout=bool(outcomes.get(index) == 'task_timeout'),
                    engineering_failure=False,
                    failure_class=outcomes.get(index),
                    steps=steps, trace=trace, metrics=metrics))
                path = cases_dir / f"{case['case_sha256']}.json"
                if path.exists():
                    raise FileExistsError(path)
                path.write_text(json.dumps(rows[-1], ensure_ascii=False,
                                           indent=2) + '\n')
            env.close()
        except Exception:
            if steps:
                ledger.commit(args.obligation_id, steps * 20,
                              operation_id=f'{op}:commit',
                              reason='Q034-A original 6400 partial')
            unused = reserved - steps * 20
            if unused > 0:
                ledger.release(args.obligation_id, unused,
                               operation_id=f'{op}:release',
                               reason='Q034-A original 6400 failed')
            raise
        actual = steps * 20
        ledger.commit(args.obligation_id, actual,
                      operation_id=f'{op}:commit',
                      reason='Q034-A original 6400 actual')
        unused = reserved - actual
        if unused:
            ledger.release(args.obligation_id, unused,
                           operation_id=f'{op}:release',
                           reason='Q034-A original 6400 unused')
    grouped = {}
    for row in rows:
        grouped.setdefault(row['group'], []).append(row)
    groups_out = {}
    for group in groups:
        rows_g = grouped.get(group['group_id'], [])
        values = np.asarray([row['net_distance'] for row in rows_g], dtype=float)
        metric_keys = ('speed_rmse', 'pitch_rms_deg', 'slip', 'action_change',
                       'drift', 'stride', 'frequency')
        metrics = {}
        for key in metric_keys:
            vals = [float(row.get('metrics', {}).get(key, 0.0)) for row in rows_g]
            metrics[key] = (sum(vals) / len(vals)) if vals else 0.0
        groups_out[group['group_id']] = dict(
            cases=len(rows_g),
            successes=sum(bool(row['success']) for row in rows_g),
            p10=float(np.percentile(values, 10)), p50=float(np.percentile(values, 50)),
            p90=float(np.percentile(values, 90)), metrics=metrics,
            failure_classes={str(name): sum(1 for row in rows_g
                                             if row.get('failure_class') == name)
                             for name in {row.get('failure_class') for row in rows_g}})
    cache = dict(schema='a2_v09_s2_gate_6400_flat_v1', suite=manifest['suite'],
                 manifest_sha256=manifest['manifest_sha256'],
                 source_sha256=s1_spec.SOURCE_SHA256,
                 model_sha256=s1_spec.SOURCE_SHA256,
                 model_type='original_6400', migrated=False,
                 observation_shape=[3, 48, 32], map='native_v07_48x32',
                 action_consistency=consistency,
                 groups=groups_out,
                 case_files=[dict(case_sha256=row['case_sha256'],
                                  path=str(cases_dir / f"{row['case_sha256']}.json"))
                             for row in rows],
                 generated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    path = out / 'reference_cache.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), groups=len(groups_out),
                          cases=len(rows), cost=cost,
                          action_consistency=consistency), ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
