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
from src.tasks.velocity.v09 import (s2_cases, s2_execute, s2_ledger, s2_spec)


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
    factory = build_factory()
    cases = {row['case_sha256']: row for row in matrix['cases']}
    if args.smoke:
        if args.case is None:
            raise SystemExit('--smoke requires --case')
        selected = [row for row in matrix['cases'] if row['case_id'] == args.case
                    and row['task'] in ('flat_walk', 'stairs_walk', 'platform_walk')]
        smoke_cases = [selected[0]] if selected else []
        if not smoke_cases:
            raise SystemExit('smoke case not found')
        rows = [row for row in batches if row['case_ids'] == [args.case]
                and row['branch'] == args.branch]
        if not rows:
            raise SystemExit('smoke batch not found')
        driver = s2_execute.S2ExecutionDriver(
            env_factory=factory, policy=None, ledger=ledger,
            device=args.device, max_steps=args.steps)
        print(json.dumps(dict(device=s2_execute.device_report(args.device),
                              preflight=preflight, case=args.case,
                              steps=args.steps, branch=args.branch),
                         ensure_ascii=False, indent=2))
        # The policy must be supplied by the real assembly; refuse otherwise.
        raise SystemExit('s2_physics.build_s2_env must also provide the Actor '
                         'mean policy through the smoke factory')
    raise SystemExit('full 150-branch execution is gated until the V6 smoke passes')


if __name__ == '__main__':
    raise SystemExit(main())
