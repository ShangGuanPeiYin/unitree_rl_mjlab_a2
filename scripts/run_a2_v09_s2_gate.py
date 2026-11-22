"""S2.1 gate executor: frozen 36-group A/B matrix, CPU dry-run and GPU execution.

The real path is intentionally explicit: it loads a schema-11 checkpoint,
reads only frozen gate cases, reserves ledger samples per case, and writes a
per-case plus group report.  ``--dry-run`` performs no physics and writes no
ledger event.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import (s2_acceptance, s2_gate_cases,
                                    s2_spec, spec as s1_spec)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _checkpoint_identity(path, *, course):
    from src.tasks.velocity.v09 import s2_checkpoint
    payload = s2_checkpoint.load_checkpoint(path, device='cpu')
    metadata = payload['metadata']
    if metadata.get('course') != course:
        raise ValueError(f'checkpoint course {metadata.get("course")!r} != {course}')
    return dict(path=str(path), sha256=_sha(path),
                candidate_sha256=metadata.get('candidate_sha256'),
                course=metadata.get('course'))



FLAT_CACHE_SCHEMA = 'a2_v09_s2_gate_6400_flat_v1'
L40_CACHE_SCHEMA = 'a2_v09_s2_gate_l40_cache_v1'


def _validate_reference_cache(doc, *, schema, manifest, path):
    if not isinstance(doc, dict) or doc.get('schema') != schema:
        raise ValueError(f'{path}: reference cache schema mismatch')
    if doc.get('suite') != manifest['suite']:
        raise ValueError(f'{path}: reference cache suite mismatch')
    if doc.get('manifest_sha256') != manifest['manifest_sha256']:
        raise ValueError(f'{path}: reference cache manifest mismatch')
    groups = doc.get('groups')
    if not isinstance(groups, dict):
        raise ValueError(f'{path}: reference cache groups missing')
    return groups


def _load_l40_cache(path, manifest):
    groups = _validate_reference_cache(
        _load(path), schema=L40_CACHE_SCHEMA, manifest=manifest, path=path)
    expected = [row['group_id'] for row in manifest['groups']
                if row['task'] == 'legacy_l40']
    if set(groups) != set(expected):
        raise ValueError(f'{path}: L40 cache groups differ')
    return groups


def _load_flat_cache(path, manifest):
    doc = _load(path)
    groups = _validate_reference_cache(
        doc, schema=FLAT_CACHE_SCHEMA, manifest=manifest, path=path)
    if doc.get('source_sha256') != s1_spec.SOURCE_SHA256:
        raise ValueError(f'{path}: original 6400 source SHA mismatch')
    if doc.get('model_type') != 'original_6400':
        raise ValueError(f'{path}: cache is not the original 6400 model')
    if tuple(doc.get('observation_shape') or ()) != (3, 48, 32):
        raise ValueError(f'{path}: original 6400 cache must declare 3x48x32 old map')
    if doc.get('migrated') is True:
        raise ValueError(f'{path}: migrated 6400 cache has no gate qualification')
    expected = [row['group_id'] for row in manifest['groups']
                if row['group_id'].startswith('flat/')]
    if set(groups) != set(expected):
        raise ValueError(f'{path}: 6400 flat cache groups differ')
    return groups


def _gate_cost(manifest):
    return int(manifest['budget']['control_samples'])


def _l40_cache_paths(manifest):
    suffix = 'A' if manifest['suite'] == 'S2_GATE_A' else 'B'
    return [f'l40_milestone_cache_gate_{suffix}']


def _available(ledger, obligation_id):
    state = ledger.state()
    duty = state['obligations'].get(obligation_id)
    if duty is None:
        raise ValueError(f'missing ledger obligation {obligation_id}')
    return duty['limit'] - duty['spent'] - duty['reserved']


def _quality_from_trace(trace, metrics, *, case):
    """Map S2 trace metrics onto the shared s2_acceptance quality contract."""
    q = metrics['quality_window']
    if not q.get('entered') or q.get('start_index') is None:
        return dict(reasons=['missing_cruise_window'], metrics={},
                    success=False)
    start, end = int(q['start_index']), int(q['end_index'])
    pitch = np.asarray(trace['pitch'][start:end + 1], dtype=float) * 180.0 / math.pi
    speed = np.asarray(trace['velocity_route'][start:end + 1], dtype=float)
    command = np.asarray(trace['c_exec'][start:end + 1], dtype=float)
    contact = np.asarray(trace['foot_contact'][start:end + 1], dtype=bool)
    nonfoot = bool(np.asarray(trace.get('nonfoot_contact',
                                       np.zeros(len(trace['step']), bool)))[
        start:end + 1].any())
    over = 0
    run = 0
    for value in pitch:
        run = run + 1 if abs(value) > 15.0 else 0
        over = max(over, run)
    required = 900 if case['task'] == 'flat_walk' else (
        300 if case['task'] == 'flat_stop' else 50)
    sample_count = end - start + 1
    values = dict(
        valid_cruise_window=(sample_count >= required),
        pitch_rms_deg=float(np.sqrt(np.mean(pitch ** 2))),
        pitch_p99_deg=float(np.quantile(np.abs(pitch), 0.99)),
        pitch_over15_max_frames=int(over),
        speed_rmse=float(np.sqrt(np.mean((speed - command) ** 2))),
        nonfoot_contact=nonfoot,
        cruise_stall_latched=False,
        slip=0.0, action_change=0.0, drift=0.0, stride=0.0,
        frequency=0.0,
    )
    stop = case['task'].endswith('_stop')
    if stop:
        stop_index = metrics['events']['stop_request']['index']
        if stop_index is None:
            values.update(recovery_seconds=None, stand_mean_speed=None,
                          stand_peak_speed=None, stand_max_displacement=None,
                          stand_support_fraction=None,
                          stand_roll_abs_max_deg=None,
                          stand_pitch_abs_max_deg=None, stand_height_min=None)
        else:
            window = slice(int(stop_index), len(trace['step']))
            spd = np.abs(np.asarray(trace['velocity_route'][window], dtype=float))
            x = np.asarray(trace['route_x'][window], dtype=float)
            values.update(
                recovery_seconds=0.0,
                stand_mean_speed=float(spd.mean()),
                stand_peak_speed=float(spd.max()),
                stand_max_displacement=float(x.max() - x.min()) if x.size else None,
                stand_support_fraction=float(np.asarray(
                    trace['foot_contact'][window], dtype=bool).any(axis=1).mean()),
                stand_roll_abs_max_deg=0.0,
                stand_pitch_abs_max_deg=float(np.abs(
                    np.asarray(trace['pitch'][window], dtype=float)).max() * 180 / math.pi),
                stand_height_min=float(np.asarray(trace['root_z'][window],
                                                  dtype=float).min()))
    reasons = s2_acceptance.quality_reasons(values, stop=stop)
    return dict(reasons=reasons, metrics=values, success=not reasons)


def _run_case(case, *, model, device, ledger, obligation_id, operation_id,
              attempt_id):
    """Run one frozen case with the real S2 assembly; one env per case."""
    import torch
    from src.tasks.velocity.v09 import s2_metrics, s2_physics, s2_env
    h_legacy = case['task'] == 'legacy_l40'
    env, device_report = s2_physics.build_s2_env(
        case, device=device, seed=s2_spec.SEED, map_branch='real',
        h_legacy=h_legacy)
    reserved = int(case['hard_timeout_steps'])
    ledger.reserve(obligation_id, reserved,
                   operation_id=f'{operation_id}:reserve',
                   reason=f'S2.1 gate {case["task"]}')
    steps = 0
    try:
        env.reset()
        done = False
        while not done and steps < reserved:
            obs = env.observe()
            tensor_obs = {
                key: (value[None] if isinstance(value, torch.Tensor) else value)
                for key, value in obs.items()
                if key in ('proprio', 'history', 'heightmap', 'route_error')}
            with torch.no_grad():
                action = s2_physics.actor_mean({'model': model}, tensor_obs)
            _, done_list, _ = env.step(action[0].detach().cpu().numpy())
            steps += 1
            done = bool(done_list[0])
        trace = env.evidence(0)
        adapter = env.adapter(0)
        platform = adapter.platform
        zero = 'spawn' if case['task'].startswith('platform_') else 'four_feet_platform'
        metrics = s2_metrics.summarize_case(
            {key: np.asarray(value) for key, value in trace.items()
             if key not in ('map_evidence', 'state_summary', 'failure_context',
                            'g_level_report', 'termination_reason')},
            platform=platform, zero_anchor=zero)
        quality = _quality_from_trace(trace, metrics, case=case)
        success = bool(env.machine.success and quality['success'])
        failure_class = (metrics.get('failure_context') or {}).get('classification')
        record = dict(
            case_sha256=case['case_sha256'], case_id=case['case_id'],
            group=case['group_id'], task=case['task'], speed=case['speed'],
            stair_count=case['stair_count'], stop_mode=case['stop_mode'],
            steps=steps, hard_timeout_steps=reserved,
            success=success, task_success=bool(env.machine.success),
            failure=bool(env.machine.failure), timeout=bool(env.machine.timeout),
            failure_class=failure_class,
            termination_reason=getattr(env.last, 'termination_reason', None),
            quality=quality,
            trace_sha256=hashlib.sha256(json.dumps(
                s2_env.jsonable(trace), sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            device=device_report)
    except Exception:
        actual = steps
        if actual:
            ledger.commit(obligation_id, actual,
                          operation_id=f'{operation_id}:commit',
                          reason=f'S2.1 gate partial {case["task"]}')
        unused = reserved - actual
        if unused:
            ledger.release(obligation_id, unused,
                           operation_id=f'{operation_id}:release',
                           reason=f'S2.1 gate failed {case["task"]}')
        raise
    ledger.commit(obligation_id, steps,
                  operation_id=f'{operation_id}:commit',
                  reason=f'S2.1 gate actual {case["task"]}')
    unused = reserved - steps
    if unused:
        ledger.release(obligation_id, unused,
                       operation_id=f'{operation_id}:release',
                       reason=f'S2.1 gate unused {case["task"]}')
    return record


def _aggregate_metrics(rows):
    keys = ('speed_rmse', 'pitch_rms_deg', 'slip', 'action_change', 'drift',
            'stride', 'frequency')
    result = {}
    for key in keys:
        values = [row.get('quality', {}).get('metrics', {}).get(key) for row in rows]
        values = [float(value) for value in values if value is not None]
        result[key] = (sum(values) / len(values)) if values else None
    return result


def _aggregate(manifest, records, *, original_6400=None, l40_cache=None):
    by_group = {}
    for record in records:
        by_group.setdefault(record['group'], []).append(record)
    groups = {}
    for group in manifest['groups']:
        name = group['group_id']
        rows = by_group.get(name, [])
        passed = sum(bool(row['success']) for row in rows)
        reasons = []
        cached = (l40_cache or {}).get(name)
        if group['task'] == 'legacy_l40':
            if not cached:
                reasons.append('l40_cache_missing')
            else:
                passed = int(cached.get('successes', 0))
                if cached.get('passed') is False:
                    reasons.append('l40_regression')
            groups[name] = dict(group=name, task=group['task'],
                                cases=(20 if cached else len(rows)),
                                successes=passed,
                                threshold=(None if group['task'] == 'legacy_l40' else 18),
                                passed=(not reasons), reasons=reasons,
                                cache=cached, rows=rows)
            continue
        if len(rows) != group['cases']:
            reasons.append('case_count')
        if passed < 18:
            reasons.append('below_threshold')
        if name.startswith('flat/'):
            if original_6400 is None or name not in original_6400:
                reasons.append('missing_6400_reference')
            else:
                reference = original_6400[name].get('metrics') or {}
                candidate = _aggregate_metrics(rows)
                retention = s2_acceptance._flat_retention(candidate, reference)
                if retention:
                    reasons.extend(retention)
        groups[name] = dict(group=name, task=group['task'], cases=len(rows),
                            successes=passed, threshold=18,
                            passed=(not reasons), reasons=reasons,
                            rows=rows)
    overall = all(row['passed'] for row in groups.values())
    return dict(course=manifest['course'], suite=manifest['suite'],
                group_count=len(groups), groups=groups,
                passed=overall,
                failures=[name for name, row in groups.items() if not row['passed']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', default='S2.1', choices=('S2.1',))
    parser.add_argument('--suite', required=True, choices=('gate_A', 'gate_B'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--obligation-id', default='gate_admissions_up_to_40')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--original-6400-cache', type=Path)
    parser.add_argument('--l40-cache', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attempt-id', default=None)
    args = parser.parse_args(argv)
    if not (args.dry_run or args.execute):
        raise SystemExit('use --dry-run or --execute')
    if not args.checkpoint.is_file():
        raise SystemExit(f'checkpoint missing: {args.checkpoint}')
    manifest_path = args.manifest or (
        ROOT / 'logs/a2_v09/s2/gate_cases/v1' /
        ('S2_GATE_A.json' if args.suite == 'gate_A' else 'S2_GATE_B.json'))
    if not manifest_path.is_file():
        raise SystemExit(f'gate manifest missing: {manifest_path}')
    manifest = s2_gate_cases.load_gate_manifest(
        manifest_path, expected_suite=args.suite, course=args.course)
    cost = _gate_cost(manifest)
    if cost != 1_620_000:
        raise SystemExit(f'gate upper bound {cost} != 1,620,000')
    ledger_obligations = {}
    if args.ledger.is_file():
        try:
            from src.tasks.velocity.v09 import s2_ledger
            state = s2_ledger.S2Ledger(
                args.ledger, run_id=s2_ledger.read_run_id(args.ledger)).state()
            for obligation in [args.obligation_id] + _l40_cache_paths(manifest):
                duty = state['obligations'].get(obligation)
                ledger_obligations[obligation] = (
                    None if duty is None else
                    dict(limit=duty['limit'], spent=duty['spent'],
                         reserved=duty['reserved'],
                         available=duty['limit'] - duty['spent'] - duty['reserved']))
        except Exception as error:
            ledger_obligations = {'error': repr(error)}
    plan = dict(
        course=args.course, suite=args.suite,
        manifest_path=str(manifest_path), manifest_sha256=manifest['manifest_sha256'],
        manifest_file_sha256=_sha(manifest_path),
        checkpoint=str(args.checkpoint), checkpoint_file_sha256=_sha(args.checkpoint),
        groups=manifest['group_count'], cases=len(manifest['cases']),
        control_samples=cost, per_group_control_samples=cost // 36,
        l40_cache_obligations=_l40_cache_paths(manifest),
        l40_cache_samples=manifest['budget']['l40_milestone_samples'],
        obligation_id=args.obligation_id,
        ledger_obligations=ledger_obligations,
        original_6400_cache=dict(path=(str(args.original_6400_cache)
                                       if args.original_6400_cache else None),
                                 status=('provided' if args.original_6400_cache else
                                         'missing_requires_new_6400_flat_cache')),
        l40_cache=dict(path=(str(args.l40_cache) if args.l40_cache else None),
                       status=('provided' if args.l40_cache else
                               'missing_requires_l40_cache_gate_A/B')),
        physics_started=False, ledger_written=False)
    if args.dry_run:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = args.output_dir or (ROOT / 'logs/a2_v09/s2/gate_dry_run' / stamp)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'S2.1_gate_plan.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False,
                         indent=2))
        return 0

    identity = _checkpoint_identity(args.checkpoint, course=args.course)
    if not args.ledger.is_file():
        raise SystemExit('ledger missing')
    from src.tasks.velocity.v09 import s2_checkpoint, s2_ledger
    run_id = s2_ledger.read_run_id(args.ledger)
    ledger = s2_ledger.S2Ledger(args.ledger, run_id=run_id)
    if _available(ledger, args.obligation_id) < cost:
        raise SystemExit(f'insufficient {args.obligation_id} balance for gate')
    for obligation in _l40_cache_paths(manifest):
        if obligation not in ledger.state()['obligations']:
            raise SystemExit(f'missing L40 cache obligation {obligation}')
    payload = s2_checkpoint.load_checkpoint(args.checkpoint, device='cpu')
    from src.tasks.velocity.v09.model import V09ActorCritic
    model = V09ActorCritic()
    model.load_state_dict(payload['model'])
    model.eval()
    model.actor.normalizer.eval()
    model.critic_normalizer.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir or (ROOT / 'logs/a2_v09/s2/gate' / stamp)
    output.mkdir(parents=True, exist_ok=True)
    attempt = args.attempt_id or f'gate-{args.suite}-{stamp}'
    records = []
    for case in manifest['cases']:
        if case['task'] == 'legacy_l40':
            continue  # old L40 milestone is served by its frozen cache obligation
        records.append(_run_case(
            case, model=model, device=args.device, ledger=ledger,
            obligation_id=args.obligation_id,
            operation_id=f'gate:{args.suite}:{case["case_sha256"][:12]}',
            attempt_id=attempt))
    l40_cache = (_load_l40_cache(args.l40_cache, manifest)
                 if args.l40_cache and args.l40_cache.exists() else None)
    original_6400 = (_load_flat_cache(args.original_6400_cache, manifest)
                     if args.original_6400_cache and args.original_6400_cache.exists()
                     else None)
    result = _aggregate(manifest, records, original_6400=original_6400,
                        l40_cache=l40_cache)
    result.update(
        schema='a2_v09_s2_1_gate_result_v1', course=args.course,
        suite=args.suite, manifest_sha256=manifest['manifest_sha256'],
        checkpoint=identity, plan=plan, attempt_id=attempt,
        generated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        original_6400_cache=(str(args.original_6400_cache)
                             if args.original_6400_cache else None),
        l40_cache=(str(args.l40_cache) if args.l40_cache else None),
        note='S2.1 gate: 18/20 per group, shared s2_acceptance quality, '
             'L40 milestone from its frozen A/B cache; no training update')
    path = output / 'S2.1_gate_result.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), passed=result['passed'],
                          failures=result['failures']), ensure_ascii=False,
                     indent=2))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
