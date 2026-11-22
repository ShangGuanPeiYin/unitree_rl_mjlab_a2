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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dump-observation', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--case', type=int)
    parser.add_argument('--steps', type=int, default=s2_execute.SMOKE_DEFAULT_STEPS)
    parser.add_argument('--branch', choices=('real', 'plane_diagnostic'),
                        default='real')
    parser.add_argument('--h-legacy', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--run-id', help='optional; must equal the ledger header run_id')
    parser.add_argument('--attempt-id',
                        help='unique attempt id; defaults to UTC timestamp+random')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    matrix, h_series, run_plan = load_inputs()
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
        preflight = s2_execute.reservation_preflight(ledger)
        case = next(row for row in matrix['cases'] if row['case_id'] == args.case) if args.case is not None else matrix['cases'][0]
        env, report = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch=args.branch,
            h_legacy=args.h_legacy)
        backend_env = getattr(env.backend, 'env', None)
        obs = env.observe()
        peak = float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0
        out = dict(preflight=True, physics_started=False, stepped=False,
                   device=report, preflight_reservation=preflight,
                   case=case['case_id'], steps=args.steps,
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
        s2_execute.reservation_preflight(ledger)
        case = next(row for row in matrix['cases'] if row['case_id'] == args.case) if args.case is not None else matrix['cases'][0]
        env, report = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch=args.branch,
            h_legacy=args.h_legacy)
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
                                        attempt_id=attempt_id, stepped=False),
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
        preflight = s2_execute.reservation_preflight(ledger)
    except s2_execute.ReservationError as error:
        raise SystemExit(str(error))
    # Load and freeze the Actor on CPU before any device allocation.
    actor = s2_physics.load_s1_actor(device='cpu')
    factory = build_factory()
    cases = {row['case_sha256']: row for row in matrix['cases']}
    if args.smoke:
        if args.case is None:
            raise SystemExit('--smoke requires --case')
        pool = matrix['cases']
        selected = [row for row in pool if row['case_id'] == args.case
                    and row['task'] in ('flat_walk', 'stairs_walk', 'platform_walk')]
        if args.h_legacy:
            selected = [row for row in h_series['records']
                        if row['case']['case_id'] == args.case]
            selected = [row['case'] for row in selected]
        if not selected:
            raise SystemExit('smoke case not found')
        case = selected[0]
        device_report = s2_execute.device_report(args.device)
        # Actor on the execution device; no optimizer/training state is loaded.
        actor = s2_physics.load_s1_actor(device=args.device)
        env, runtime = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch=args.branch,
            h_legacy=args.h_legacy)
        batch = dict(branch=args.branch,
                     group_role_sha256='smoke',
                     control_samples=int(args.steps),
                     case_sha256s=[case['case_sha256']],
                     case_ids=[case['case_id']], batch_size=1,
                     max_hard_timeout_steps=int(case['hard_timeout_steps']))
        gate = s2_execute.ReservationGate(ledger)
        operation_id = f"smoke:{args.case}:{attempt_id}"
        started = gate.begin_batch(batch, operation_id=operation_id)
        summary = dict(device=device_report, preflight=preflight,
                       actor_sha256=actor['state_sha256'],
                       actor_frozen=actor['frozen'], case=args.case,
                       steps=args.steps, branch=args.branch,
                       h_legacy=args.h_legacy,
                       attempt_id=attempt_id,
                       estimated_control_steps=args.steps,
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
            for step in range(int(args.steps)):
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
            if evidence_rows:
                pitches = [float(row['pitch']) for row in evidence_rows]
                speeds = [float(row['velocity_route']) for row in evidence_rows]
                c_exec = [float(row['c_exec']) for row in evidence_rows]
                platform_support = [int(sum(bool(v) for v in row['foot_on_platform']))
                                    for row in evidence_rows]
                contact_feet = [int(sum(bool(v) for v in row['foot_contact']))
                                for row in evidence_rows]
                route = [float(row['route_x']) for row in evidence_rows]
                pair_counts = [len(row['contact_pairs']) for row in evidence_rows]
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
                summary['first_64_summary'] = dict(
                    frames=len(evidence_rows),
                    pitch_min_deg=min(pitches) * 180 / 3.141592653589793,
                    pitch_max_deg=max(pitches) * 180 / 3.141592653589793,
                    speed_mean=sum(speeds) / len(speeds),
                    velocity_route_mean=sum(speeds) / len(speeds),
                    route_x_start=route[0], route_x_end=route[-1],
                    route_x_delta=route[-1] - route[0],
                    c_exec_mean=sum(c_exec) / len(c_exec),
                    platform_support_feet_mean=sum(platform_support) / len(platform_support),
                    platform_support_feet_last=platform_support[-1],
                    contact_feet_mean=sum(contact_feet) / len(contact_feet),
                    foot_on_platform_mean=sum(platform_support) / len(platform_support),
                    contact_pairs_mean=sum(pair_counts) / len(pair_counts))
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
            try:
                if env.recorder.rows:
                    env.recorder.write(output)
            except Exception as evidence_error:
                summary['evidence_error'] = repr(evidence_error)
                summary['evidence_partial'] = True
            (output / 'smoke_summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary['status'] == 'completed' else 1
    raise SystemExit('full 150-branch execution is gated until the V6 smoke passes')


if __name__ == '__main__':
    raise SystemExit(main())
