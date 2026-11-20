"""Freeze approved S1 inputs and reserve budget before any simulator step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import (
    GATE_SUITES, FINAL_SUITES, CALIBRATION_SUITE, build_s1_manifest,
    canonical_json, freeze_manifest, load_manifest,
)
from src.tasks.velocity.v09.budget import AppendOnlyLedger, s1_reservation_plan, batch_worst_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--calibration-cases-per-group', type=int, required=True)
    parser.add_argument('--reserve', action='store_true',
                        help='Write immutable cases and admit S1 nonlearning envelope')
    args = parser.parse_args()
    sources = spec.validate_sources()
    manifests = {suite: build_s1_manifest(suite) for suite in GATE_SUITES + FINAL_SUITES}
    diagnostic = build_s1_manifest('S1_DIAG')
    calibration = build_s1_manifest(CALIBRATION_SUITE,
        calibration_cases_per_group=args.calibration_cases_per_group)
    calibration_cost = batch_worst_cost(calibration['cases'],
        batch_size=args.calibration_cases_per_group)['control_samples']
    if calibration_cost > 1_000_000:
        raise ValueError('frozen calibration worst cost exceeds one million controls')
    # Resource-admission evaluation uses the longest original L40 hard timeout.
    longest = max(case['hard_timeout_steps'] for case in manifests['gate_A']['cases'])
    extras = {count: [dict(num_envs=count, control_steps=48,
                          reason='save_restore_full_rollout_20_adam'),
                      dict(num_envs=100, control_steps=longest,
                           reason='independent_process_evaluation_original_L40_vx0.3'),
                      dict(num_envs=1, control_steps=longest,
                           reason='independent_process_video_original_L40_vx0.3'),
                      dict(num_envs=count, control_steps=48,
                           reason='restore_after_external_processes_full_rollout_20_adam')]
              for count in spec.ENV_CANDIDATES}
    plan = s1_reservation_plan(manifests, calibration_manifest=calibration, diagnostic_manifest=diagnostic, probe_extra_costs=extras,
                              engineering_extra_samples=0, calibration_samples=1_000_000)
    if not plan['admitted']:
        raise RuntimeError('S1 worst-case budget exceeds its permitted envelope')
    report = dict(sources=sources, plan=plan, calibration_worst_control_samples=calibration_cost, calibration_manifest_sha256=calibration['manifest_sha256'],
                  calibration_cases_per_group=args.calibration_cases_per_group,
                  formal_physics_started=False, behavior_verified=False)
    if args.reserve:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for suite, manifest in {**manifests, CALIBRATION_SUITE: calibration, 'S1_DIAG': diagnostic}.items():
            path = args.output_dir / 'cases' / f'{suite}.json'
            if path.exists():
                previous = load_manifest(path, purpose='freeze_audit')
                if canonical_json(previous) != canonical_json(manifest):
                    raise ValueError(f'existing immutable manifest differs: {suite}')
            else:
                freeze_manifest(path, manifest)
        path = args.output_dir / 'preflight.json'
        if path.exists() and json.loads(path.read_text()) != report:
            raise ValueError('existing preflight identity differs')
        if not path.exists():
            with path.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        ledger = AppendOnlyLedger(args.output_dir / 'budget.jsonl', run_id=args.run_id)
        if not ledger.snapshot()['stages']['S1']['envelope']:
            ledger.reserve_stage(plan)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
