"""Execute S2.0 diagnostic batches against a reserved S2 ledger (GPU gated).

The physics assembly is injected by ``s2_physics.build_s2_env``.  If that
module/adapter is absent the script stops before allocating a GPU; this keeps
the reservation and batch logic independently auditable on CPU.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import (s2_cases, s2_env, s2_execute, s2_ledger, s2_physics, s2_spec)
import numpy as np
import torch


def load_inputs(root=ROOT):
    root = Path(root)
    directory = root / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(directory / 'S2_DIAG.json', root)
    h_series = s2_cases.load_frozen_manifest(directory / 'h_series_case0.json', root)
    run_plan = json.loads((directory / 'S2_DIAG_run_plan.json').read_text())
    s2_cases.validate_s2_diag_run_plan(run_plan, matrix)
    return matrix, h_series, run_plan


def case_metadata(case, *, suite, source):
    geometry = case.get('geometry') or {}
    return dict(
        suite=suite, manifest_suite=case.get('suite'), source=source,
        case_id=case['case_id'], case_sha256=case['case_sha256'],
        task=case['task'], stair_count=case.get('stair_count'),
        speed=float(case['speed']),
        hard_timeout_steps=int(case['hard_timeout_steps']),
        geometry=geometry,
        platform_width=geometry.get('platform_width', case.get('platform_width')),
        platform_front_local_x=geometry.get('platform_front_local_x'),
        spawn_x=float(case['spawn']['local_position_xy'][0]),
        spawn_y=float(case['spawn']['local_position_xy'][1]))


def _candidate(row):
    return (f"case_id={row['case_id']} sha={row['case_sha256'][:12]} "
            f"task={row['task']} stairs={row.get('stair_count')} "
            f"speed={row['speed']} width={(row.get('geometry') or {}).get('platform_width')} "
            f"timeout={row['hard_timeout_steps']}")


def select_case(matrix, h_series, *, suite='S2_DIAG', case_id=None,
                case_sha256=None, task=None, speed=None,
                platform_width=None):
    """Select one frozen case by identity, never by list position.

    ``case_id`` repeats across S2_DIAG groups (for example the flat and stair
    groups both contain case_id 0), so an ambiguous filter is an error rather
    than silently taking the first row.
    """
    if suite == 'H_SERIES':
        rows = [record['case'] for record in h_series['records']]
        source = 'h_series_case0.json'
    elif suite == 'S2_DIAG':
        rows = list(matrix['cases'])
        source = 'S2_DIAG.json'
    else:
        raise ValueError(f'unknown suite {suite!r}')
    if case_id is not None and not case_sha256 and (
            task is None or speed is None or platform_width is None):
        raise ValueError(
            '--case-id is not globally unique; also pass --task, --speed and '
            '--platform-width, or use --case-sha')
    if case_sha256:
        rows = [row for row in rows if row['case_sha256'].startswith(case_sha256)]
    if case_id is not None:
        rows = [row for row in rows if int(row['case_id']) == int(case_id)]
    if task is not None:
        rows = [row for row in rows if row['task'] == task]
    if speed is not None:
        rows = [row for row in rows if abs(float(row['speed']) - float(speed)) < 1e-9]
    if platform_width is not None:
        rows = [row for row in rows
                if (row.get('geometry') or {}).get('platform_width') is not None
                and abs(float((row.get('geometry') or {})['platform_width'])
                        - float(platform_width)) < 1e-9]
    if not rows:
        raise ValueError(
            f'no {suite} case matches case_id={case_id} sha={case_sha256} '
            f'task={task} speed={speed} platform_width={platform_width}')
    if len(rows) > 1:
        listing = '\n  '.join(_candidate(row) for row in rows)
        raise ValueError(
            f'ambiguous {suite} case selection ({len(rows)} matches); add '
            f'--case-sha256 or --task/--speed/--platform-width:\n  {listing}')
    return rows[0], case_metadata(rows[0], suite=suite, source=source)


def build_factory(module_name='src.tasks.velocity.v09.s2_physics'):
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise s2_execute.S2PhysicsAssemblyError(
            'real S2 v09/mjlab physics assembly is not installed; provide '
            'src.tasks.velocity.v09.s2_physics.build_s2_env before GPU smoke') from error
    factory = getattr(module, 'build_s2_env', None)
    if not callable(factory):
        raise s2_execute.S2PhysicsAssemblyError('s2_physics.build_s2_env missing')
    return factory


def _sha256_file(path):
    path = Path(path)
    return (hashlib.sha256(path.read_bytes()).hexdigest()
            if path.exists() and path.is_file() else None)


def _pitch_stats(rows):
    if not rows:
        return None
    values = np.asarray([float(row['pitch']) for row in rows]) * 180.0 / np.pi
    return dict(frames=int(values.size),
                mean_deg=float(values.mean()),
                min_deg=float(values.min()),
                max_deg=float(values.max()),
                positive_forward_lean_fraction=float(np.mean(values > 0.0)))


def _first_event(events, name):
    return next((event for event in events if event.get('name') == name), None)


def _frame_stats(rows):
    if not rows:
        return None
    route = np.asarray([float(row['route_x']) for row in rows])
    speed = np.asarray([float(row['velocity_route']) for row in rows])
    pitch = np.asarray([float(row['pitch']) for row in rows]) * 180.0 / np.pi
    feet_contact = np.asarray([row['foot_contact'] for row in rows], dtype=bool)
    platform_support = np.asarray(
        [sum(bool(v) for v in row['foot_on_platform']) for row in rows])
    return dict(
        frames=len(rows),
        route_x_start=float(route[0]), route_x_end=float(route[-1]),
        route_x_delta=float(route[-1] - route[0]), route_x_max=float(route.max()),
        speed_mean=float(speed.mean()),
        pitch_min_deg=float(pitch.min()), pitch_max_deg=float(pitch.max()),
        c_exec_mean=float(np.mean([float(row['c_exec']) for row in rows])),
        contact_feet_mean=float(feet_contact.sum(axis=1).mean()),
        platform_support_feet_mean=float(platform_support.mean()),
        platform_support_feet_last=int(platform_support[-1]),
        contact_pairs_mean=float(np.mean(
            [len(row['contact_pairs']) for row in rows])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dump-observation', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--suite', choices=('S2_DIAG', 'H_SERIES'),
                        help='frozen case suite; defaults to S2_DIAG (H with --h-legacy)')
    parser.add_argument('--case-id', type=int,
                        help='exact case_id field; repeated IDs require more filters')
    parser.add_argument('--case', type=int, help='deprecated alias of --case-id')
    parser.add_argument('--case-sha', '--case-sha256', dest='case_sha256',
                        help='full or unique prefix of the frozen case SHA256')
    parser.add_argument('--task', help='optional case filter, e.g. stairs_walk')
    parser.add_argument('--speed', type=float, help='optional exact speed filter')
    parser.add_argument('--platform-width', type=float,
                        help='optional exact platform width filter')
    parser.add_argument('--steps', type=int, default=None,
                        help=f'control steps (default {s2_execute.SMOKE_DEFAULT_STEPS}; '
                             'clamped to hard_timeout_steps)')
    parser.add_argument('--branch', choices=('real', 'plane_diagnostic'),
                        default='real')
    parser.add_argument('--h-legacy', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--obligation-id', default='s2_0_pipeline_debug',
                        help='ledger obligation for smoke/debug; matrix uses the frozen matrix ID')
    parser.add_argument('--run-id', help='optional; must equal the ledger header run_id')
    parser.add_argument('--attempt-id',
                        help='unique attempt id; defaults to UTC timestamp+random')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    matrix, h_series, run_plan = load_inputs()
    if args.case is not None and args.case_id is not None \
            and int(args.case) != int(args.case_id):
        raise SystemExit('--case and --case-id disagree')
    case_id = args.case_id if args.case_id is not None else args.case
    suite = args.suite or ('H_SERIES' if args.h_legacy else 'S2_DIAG')
    h_legacy = bool(args.h_legacy or suite == 'H_SERIES')
    requested_steps = (s2_execute.SMOKE_DEFAULT_STEPS if args.steps is None
                       else int(args.steps))
    if requested_steps <= 0:
        raise SystemExit('--steps must be positive')
    attempt_id = args.attempt_id or s2_execute.new_attempt_id()
    if not attempt_id or ':' in attempt_id or '/' in attempt_id:
        raise SystemExit('attempt_id must be nonempty without : or /')
    batches = s2_execute.batch_table(run_plan)
    if args.plan_only:
        print(json.dumps(dict(batches=len(batches), batch_table=batches,
                              physics_started=False), ensure_ascii=False, indent=2))
        return 0
    if args.preflight:
        if not args.ledger.exists():
            raise SystemExit('S2 ledger does not exist: run reserve_a2_v09_s2.py --write first')
        header_run_id = s2_ledger.read_run_id(args.ledger)
        if args.run_id is not None and args.run_id != header_run_id:
            raise SystemExit(f'run_id mismatch: argument {args.run_id!r} != ledger header {header_run_id!r}')
        ledger = s2_ledger.S2Ledger(args.ledger, run_id=header_run_id)
        preflight = s2_execute.reservation_preflight(ledger, obligation_id=args.obligation_id)
        try:
            case, meta = select_case(
                matrix, h_series, suite=suite, case_id=case_id,
                case_sha256=args.case_sha256, task=args.task, speed=args.speed,
                platform_width=args.platform_width)
        except ValueError as error:
            raise SystemExit(str(error))
        steps = min(requested_steps, int(case['hard_timeout_steps']))
        env, report = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch=args.branch,
            h_legacy=h_legacy)
        backend_env = getattr(env.backend, 'env', None)
        obs = env.observe()
        peak = float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0
        out = dict(preflight=True, physics_started=False, stepped=False,
                   device=report, preflight_reservation=preflight,
                   case=case['case_id'], case_meta=meta,
                   requested_steps=requested_steps, steps=steps,
                   steps_clamped=(steps != requested_steps),
                   attempt_id=attempt_id, termination=env.termination,
                   terrain_geom_count=len(getattr(backend_env, 'geom_ids', ())) if backend_env else None,
                   robot_joint_count=len(getattr(backend_env, 'joint_ids', ())) if backend_env else None,
                   observation_shapes={key: list(value.shape) for key, value in obs.items() if hasattr(value, 'shape')},
                   peak_mib=peak)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if args.dump_observation:
        if not args.ledger.exists():
            raise SystemExit('S2 ledger does not exist: run reserve_a2_v09_s2.py --write first')
        header_run_id = s2_ledger.read_run_id(args.ledger)
        if args.run_id is not None and args.run_id != header_run_id:
            raise SystemExit(f'run_id mismatch: argument {args.run_id!r} != ledger header {header_run_id!r}')
        ledger = s2_ledger.S2Ledger(args.ledger, run_id=header_run_id)
        s2_execute.reservation_preflight(ledger, obligation_id=args.obligation_id)
        try:
            case, meta = select_case(
                matrix, h_series, suite=suite, case_id=case_id,
                case_sha256=args.case_sha256, task=args.task, speed=args.speed,
                platform_width=args.platform_width)
        except ValueError as error:
            raise SystemExit(str(error))
        env, report = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch=args.branch,
            h_legacy=h_legacy)
        output = args.output_dir or (ROOT / 'logs/a2_v09/s2_0/evidence' / attempt_id)
        output.mkdir(parents=True, exist_ok=True)
        try:
            obs = env.reset()
            diagnostics = s2_env.observation_diagnostics(obs)
            status = 'ok'
        except Exception as error:
            diagnostics = {'error': repr(error)}
            status = 'failed'
        path = output / 'observation_dump.json'
        path.write_text(json.dumps(dict(status=status, diagnostics=diagnostics,
                                        device=report, case=case['case_id'],
                                        case_meta=meta, attempt_id=attempt_id,
                                        stepped=False),
                                   ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(status=status, path=str(path),
                              diagnostics=diagnostics), ensure_ascii=False,
                         indent=2))
        return 0 if status == 'ok' else 1
    if not args.execute:
        raise SystemExit('use --execute (with a reserved S2 ledger), --preflight, or --plan-only')
    if not args.ledger.exists():
        raise SystemExit('S2 ledger does not exist: run reserve_a2_v09_s2.py --write first')
    header_run_id = s2_ledger.read_run_id(args.ledger)
    if args.run_id is not None and args.run_id != header_run_id:
        raise SystemExit(f'run_id mismatch: argument {args.run_id!r} != ledger header {header_run_id!r}')
    ledger = s2_ledger.S2Ledger(args.ledger, run_id=header_run_id)
    try:
        preflight = s2_execute.reservation_preflight(ledger, obligation_id=args.obligation_id)
    except s2_execute.ReservationError as error:
        raise SystemExit(str(error))
    # Load and freeze the Actor on CPU before any device allocation.
    actor = s2_physics.load_s1_actor(device='cpu')
    factory = build_factory()
    if args.smoke:
        try:
            case, meta = select_case(
                matrix, h_series, suite=suite, case_id=case_id,
                case_sha256=args.case_sha256, task=args.task, speed=args.speed,
                platform_width=args.platform_width)
        except ValueError as error:
            raise SystemExit(str(error))
        steps = min(requested_steps, int(case['hard_timeout_steps']))
        device_report = s2_execute.device_report(args.device)
        # Actor on the execution device; no optimizer/training state is loaded.
        actor = s2_physics.load_s1_actor(device=args.device)
        env, runtime = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch=args.branch,
            h_legacy=h_legacy)
        batch = dict(branch=args.branch,
                     group_role_sha256='smoke',
                     control_samples=int(steps),
                     case_sha256s=[case['case_sha256']],
                     case_ids=[case['case_id']], batch_size=1,
                     max_hard_timeout_steps=int(case['hard_timeout_steps']))
        gate = s2_execute.ReservationGate(ledger, obligation_id=args.obligation_id)
        operation_id = f"smoke:{suite}:{case['case_id']}:{attempt_id}"
        started = gate.begin_batch(batch, operation_id=operation_id)
        summary = dict(device=device_report, preflight=preflight,
                       actor_sha256=actor['state_sha256'],
                       actor_frozen=actor['frozen'], case=case['case_id'],
                       case_meta=meta, **meta,
                       steps=steps, requested_steps=requested_steps,
                       steps_clamped=(steps != requested_steps),
                       branch=args.branch, h_legacy=h_legacy,
                       attempt_id=attempt_id,
                       obligation_id=args.obligation_id,
                       estimated_control_steps=steps,
                       physics_started=True)
        output = args.output_dir or (ROOT / 'logs/a2_v09/s2_0/evidence' /
                                     attempt_id)
        output.mkdir(parents=True, exist_ok=True)
        trace_path = env.recorder.attach_trace(output / 'trace.jsonl')
        summary.update(evidence_dir=str(output), trace_path=str(trace_path))
        rows = []
        policy_calls = 0
        first_obs_sha256 = None
        first_action = None
        action_absmax = []
        action_std = []
        try:
            env.reset()
            for step in range(int(steps)):
                obs = env.observe()
                if first_obs_sha256 is None:
                    digest = hashlib.sha256()
                    for key in ('proprio', 'history', 'heightmap', 'route_error'):
                        digest.update(key.encode())
                        digest.update(np.ascontiguousarray(
                            s2_env.to_numpy(obs[key]), dtype=np.float32).tobytes())
                    first_obs_sha256 = digest.hexdigest()
                tensor_obs = {key: torch.as_tensor(value, device=args.device,
                                                   dtype=torch.float32)[None]
                              for key, value in obs.items()
                              if key in ('proprio', 'history', 'heightmap',
                                         'route_error')}
                action = s2_physics.actor_mean(actor, tensor_obs)
                policy_calls += 1
                if first_action is None:
                    first_action = s2_env.jsonable(action[0])
                raw_action = action[0].detach().cpu().numpy()
                action_absmax.append(float(np.max(np.abs(raw_action))))
                action_std.append(float(np.std(raw_action)))
                _, done, info = env.step(raw_action)
                rows.append(dict(step=step, done=bool(done[0]),
                                 g_level=float(info.get('g_level', 0.0)),
                                 state=info.get('state', {}).get('state')))
                if done[0]:
                    break
            summary.update(control_steps=len(rows), status='completed',
                           native_sample_id=env.evidence(0)['map_evidence'].get('native_sample_id'))
        except Exception as error:
            summary.update(status='failed', error=repr(error),
                           control_steps=len(rows), evidence_partial=True)
        finally:
            evidence_rows = list(getattr(env.recorder, 'rows', []))
            summary['evidence_frames'] = len(evidence_rows)
            summary['evidence_partial'] = summary.get('evidence_partial', False) or (
                len(evidence_rows) != summary.get('control_steps', -1))
            events = s2_env.jsonable(getattr(env.machine, 'events', []))
            summary['termination_reason'] = getattr(env.last, 'termination_reason', None)
            try:
                summary['map_evidence'] = env.evidence(0)['map_evidence']
            except Exception as evidence_error:
                summary['map_evidence'] = {'error': repr(evidence_error)}
            for event in events:
                event['time_s'] = float(event['step']) * s2_spec.CONTROL_DT
            summary['events'] = events
            summary['key_events'] = {
                name: _first_event(events, name)
                for name in ('enter_climb', 'top_arrived', 'four_feet_platform',
                             'distance_reached', 'success', 'failure')}
            if evidence_rows:
                actions = [row['a_request'] for row in evidence_rows]
                nonzero = [frame for frame in actions
                           if float(np.max(np.abs(np.asarray(frame)))) > 1e-8]
                summary['action_nonzero_frames'] = len(nonzero)
                summary['action_absmax_mean'] = (
                    sum(float(np.max(np.abs(np.asarray(frame)))) for frame in actions)
                    / len(actions))
                summary['action_std_mean'] = (
                    sum(float(np.std(np.asarray(frame))) for frame in actions)
                    / len(actions))
                if not nonzero:
                    summary['status'] = 'failed'
                    summary['error'] = 'policy_not_applied'
                summary['first_64_summary'] = _frame_stats(evidence_rows[:64])
                summary['run_summary'] = _frame_stats(evidence_rows)
                route = np.asarray([float(row['route_x']) for row in evidence_rows])
                feet_x = np.asarray([row['feet_route_x'] for row in evidence_rows],
                                    dtype=np.float64)
                feet_z = np.asarray([row['feet_z'] for row in evidence_rows],
                                    dtype=np.float64)
                foot_contact = np.asarray([row['foot_contact'] for row in evidence_rows],
                                          dtype=bool)
                geometry = case.get('geometry') or {}
                front = geometry.get('platform_front_local_x')
                step_depth = float(geometry.get('step_depth') or 0.2)
                stair_count = int(case.get('stair_count') or 0)
                route_max = float(route.max())
                summary['progress'] = dict(
                    crossed_zero=bool(np.any(route > 0.0)),
                    reached_platform_front=(None if front is None else
                                            bool(route_max >= float(front))),
                    route_x_end=float(route[-1]), route_x_max=route_max,
                    platform_front_local_x=front,
                    stairs_estimated_body_level=int(np.clip(
                        np.floor(max(route_max, 0.0) / step_depth),
                        0, stair_count)) if stair_count else 0,
                    stairs_estimated_feet_level=int(np.clip(
                        np.floor(max(float(feet_x.max()), 0.0) / step_depth),
                        0, stair_count)) if stair_count else 0,
                    feet_route_x_start=feet_x[0].tolist(),
                    feet_route_x_end=feet_x[-1].tolist(),
                    feet_route_x_max=feet_x.max(axis=0).tolist(),
                    feet_z_start=feet_z[0].tolist(),
                    feet_z_end=feet_z[-1].tolist(),
                    feet_z_max=feet_z.max(axis=0).tolist(),
                    feet_vertical_excursion=(feet_z.max(axis=0)
                                             - feet_z.min(axis=0)).tolist(),
                    per_foot_contact_fraction=foot_contact.mean(axis=0).tolist(),
                    four_feet_platform_frames=int(np.sum(
                        np.all([row['foot_on_platform'] for row in evidence_rows],
                               axis=1))))
                if front is not None:
                    crossed = np.where(route >= float(front))[0]
                    summary['pitch_after_platform'] = (
                        _pitch_stats(evidence_rows[int(crossed[0]):])
                        if crossed.size else None)
                top = _first_event(events, 'top_arrived')
                summary['pitch_after_top'] = (
                    _pitch_stats(evidence_rows[int(top['step']):])
                    if top is not None else None)
                state = env.machine.summary()
                state.pop('command_trace', None)
                state.pop('events', None)
                summary['state_summary'] = state
                if bool(evidence_rows[-1].get('terminal')):
                    last = evidence_rows[-1]
                    summary['pre_terminal_sample'] = dict(
                        row_index=len(evidence_rows) - 2,
                        route_x=(float(evidence_rows[-2]['route_x'])
                                 if len(evidence_rows) > 1 else None),
                        pitch_deg=(float(evidence_rows[-2]['pitch']) * 180.0 / np.pi
                                   if len(evidence_rows) > 1 else None),
                        terminal_row_route_x=float(last['route_x']),
                        note='v07 auto-reset means the terminal row is the new episode start')
                summary['top_arrived'] = bool(env.machine.top_arrived)
                summary['failure'] = bool(env.machine.failure)
                summary['timeout'] = bool(env.machine.timeout)
            summary['policy_calls'] = policy_calls
            summary['first_obs_sha256'] = first_obs_sha256
            summary['first_action'] = first_action
            if summary.get('status') == 'completed' and (
                    policy_calls != summary.get('control_steps')
                    or summary.get('action_nonzero_frames', 0) == 0):
                summary['status'] = 'failed'
                summary['error'] = 'policy_not_applied'
            settled = gate.finish_batch(batch, operation_id=operation_id,
                                        actual_samples=len(rows))
            summary['settled'] = settled
            summary['peak_mib'] = (float(torch.cuda.max_memory_allocated() / 2 ** 20)
                                   if torch.cuda.is_available() else 0.0)
            summary['trace_sha256'] = _sha256_file(trace_path)
            written = None
            try:
                if env.recorder.rows:
                    written = env.recorder.write(output, attempt_id=attempt_id)
            except Exception as evidence_error:
                summary['evidence_error'] = repr(evidence_error)
                summary['evidence_partial'] = True
            if written:
                summary['npz_sha256'] = written['npz_sha256']
                summary['trace_json_sha256'] = written['trace_sha256']
            files = []
            for kind, path in (
                    ('trace_jsonl', trace_path),
                    ('npz', written['npz'] if written else None),
                    ('case_metadata', written['metadata'] if written else None),
                    ('case_trace_jsonl', written['trace_json'] if written else None)):
                if path is None or not Path(path).exists():
                    continue
                files.append(dict(kind=kind, path=str(path),
                                  sha256=_sha256_file(path)))
            summary['evidence_files'] = files
            (output / 'smoke_summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary['status'] == 'completed' else 1
    raise SystemExit('full 150-branch execution is gated until the V6 smoke passes')


if __name__ == '__main__':
    raise SystemExit(main())
