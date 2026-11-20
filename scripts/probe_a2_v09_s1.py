"""Probe disposable full S1 workloads after stage budget and frozen calibration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.budget import AppendOnlyLedger
from src.tasks.velocity.v09.cases import load_manifest, canonical_json
from src.tasks.velocity.v09.resources import (
    EVAL_ENVS, EVAL_STEPS, RealProbeCandidate, WholeCardMonitor,
    load_calibration, run_resource_probe, resource_assessment_cases,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--segment-id', required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--evidence-identity', type=Path)
    parser.add_argument('--evidence-identity-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--physical-gpu-index', type=int, required=True)
    parser.add_argument('--monitor-interval', type=float, default=.02)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec.validate_sources()
    snapshot = spec.code_snapshot()
    if snapshot['code_sha256'] != args.code_sha256:
        raise ValueError('code differs from the independently approved source snapshot')
    if (args.evidence_identity is None) != (args.evidence_identity_sha256 is None):
        raise ValueError('both evidence identity path and complete SHA are required')
    identity_context = None
    input_evidence = {}
    if args.evidence_identity is not None:
        from src.tasks.velocity.v09.evidence_identity import EvidenceIdentityContext
        identity_context = EvidenceIdentityContext.from_approval(args.evidence_identity, args.evidence_identity_sha256,
            consumer_code_sha256=args.code_sha256, run_id=args.run_id, run_dir=args.run_dir,
            ledger_path=args.run_dir / 'budget.jsonl', sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256)
    calibration = load_calibration(args.calibration, code_sha256=snapshot['code_sha256'],
                                   identity_context=identity_context)
    if identity_context is None:
        train_manifest = load_manifest(args.run_dir / 'cases/train_calibration.json', purpose='training')
        gate = load_manifest(args.run_dir / 'cases/gate_A.json', purpose='gate')
    else:
        train_manifest, train_ref = identity_context.read_manifest(args.run_dir / 'cases/train_calibration.json',
                                                                   purpose='training', suite='train_calibration')
        gate, gate_ref = identity_context.read_manifest(args.run_dir / 'cases/gate_A.json', purpose='gate', suite='gate_A')
        input_evidence = {'train_calibration': asdict(train_ref), 'gate_A': asdict(gate_ref)}
    if calibration['manifest_sha256'] != train_manifest['manifest_sha256']:
        raise ValueError('calibration must bind this run frozen training calibration manifest')
    assessment = resource_assessment_cases(gate['cases'], num_envs=EVAL_ENVS)
    if not args.execute:
        print(json.dumps(dict(device=args.device, physical_gpu_index=args.physical_gpu_index,
            candidates=list(spec.ENV_CANDIDATES), warmups=5, measurements=50,
            rollout_steps=48, adam_updates_per_iteration=20, recovery_extra_steps=48,
            evaluation_num_envs=100, evaluation_control_steps=5005,
            video_num_envs=1, video_control_steps=5005, post_evidence_restore_steps=48,
            source_case_count=len({case['case_sha256'] for case in assessment}),
            input_evidence=input_evidence,
            calibration=calibration, code_sha256=snapshot['code_sha256'],
            formal_physics_started=False), indent=2))
        return 0
    ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
    if not ledger.snapshot()['stages']['S1']['envelope']:
        raise RuntimeError('S1 full stage reservation is required before any resource probe')
    factory = lambda n, directory: RealProbeCandidate(n, device=args.device,
        calibration=calibration, output_dir=directory, gate_cases=gate['cases'],
        code_sha256=snapshot['code_sha256'], ledger_run_id=args.run_id, run_dir=args.run_dir,
        ledger_path=args.run_dir / 'budget.jsonl', gate_manifest_path=args.run_dir / 'cases/gate_A.json',
        identity_context=identity_context)
    monitors = lambda n: WholeCardMonitor(device=args.device,
        physical_gpu_index=args.physical_gpu_index, interval=args.monitor_interval)
    report = run_resource_probe(ledger=ledger, segment_id=args.segment_id,
        calibration=calibration, output_dir=args.output_dir, code_sha256=snapshot['code_sha256'],
        candidate_factory=factory, monitor_factory=monitors,
        progress=lambda event: print(json.dumps(event), flush=True), source_snapshot=snapshot,
        input_evidence=input_evidence)
    print(json.dumps(dict(admitted=report['admitted'], selected_num_envs=report['selected_num_envs'],
                         report_sha256=report['report_sha256'], behavior_verified=False), indent=2))
    return 130 if report.get('cancelled') else 0 if report['admitted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
