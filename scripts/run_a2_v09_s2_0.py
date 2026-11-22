"""Execute S2.0 diagnostic batches against a reserved S2 ledger (GPU gated).

The physics assembly is injected by ``s2_physics.build_s2_env``.  If that
module/adapter is absent the script stops before allocating a GPU; this keeps
the reservation and batch logic independently auditable on CPU.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import (s2_cases, s2_execute, s2_ledger, s2_physics, s2_spec)
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
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--case', type=int)
    parser.add_argument('--steps', type=int, default=s2_execute.SMOKE_DEFAULT_STEPS)
    parser.add_argument('--branch', choices=('real', 'plane_diagnostic'),
                        default='real')
    parser.add_argument('--h-legacy', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    matrix, h_series, run_plan = load_inputs()
    batches = s2_execute.batch_table(run_plan)
    if args.plan_only:
        print(json.dumps(dict(batches=len(batches), batch_table=batches,
                              physics_started=False), ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        raise SystemExit('use --execute (with a reserved S2 ledger) or --plan-only')
    if not args.ledger.exists():
        raise SystemExit('S2 ledger does not exist: run reserve_a2_v09_s2.py --write first')
    ledger = s2_ledger.S2Ledger(args.ledger, run_id='s2-0-execute')
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
        started = gate.begin_batch(batch, operation_id=f'smoke:{args.case}')
        summary = dict(device=device_report, preflight=preflight,
                       actor_sha256=actor['state_sha256'],
                       actor_frozen=actor['frozen'], case=args.case,
                       steps=args.steps, branch=args.branch,
                       h_legacy=args.h_legacy,
                       estimated_control_steps=args.steps,
                       physics_started=True)
        rows = []
        try:
            env.reset()
            for step in range(int(args.steps)):
                obs = env.observe()
                tensor_obs = {key: torch.as_tensor(value, device=args.device,
                                                   dtype=torch.float32)[None]
                              for key, value in obs.items()
                              if key in ('proprio', 'history', 'heightmap',
                                         'route_error')}
                action = s2_physics.actor_mean(actor, tensor_obs)
                _, done, info = env.step(action[0].detach().cpu().numpy())
                rows.append(dict(step=step, done=bool(done[0]),
                                 g_level=float(info.get('g_level', 0.0)),
                                 state=info.get('state', {}).get('state')))
                if done[0]:
                    break
            summary.update(control_steps=len(rows), status='completed',
                           native_sample_id=env.evidence(0)['map_evidence'].get('native_sample_id'))
        except Exception as error:
            summary.update(status='failed', error=repr(error),
                           control_steps=len(rows))
        finally:
            settled = gate.finish_batch(batch, operation_id=f'smoke:{args.case}',
                                        actual_samples=len(rows) or int(args.steps))
            summary['settled'] = settled
            output = args.output_dir or (ROOT / 'logs/a2_v09/s2/smoke' /
                                         f"case{args.case}")
            output.mkdir(parents=True, exist_ok=True)
            (output / 'smoke_summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            try:
                if env.recorder.rows:
                    env.recorder.write(output)
            except Exception as evidence_error:
                summary['evidence_error'] = repr(evidence_error)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary['status'] == 'completed' else 1
    raise SystemExit('full 150-branch execution is gated until the V6 smoke passes')


if __name__ == '__main__':
    raise SystemExit(main())
