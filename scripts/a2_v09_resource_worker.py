"""Q009 external resource child; only the parent reserves/settles control steps."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import load_manifest, sha256
from src.tasks.velocity.v09.resources import (
    validate_child_reservation, resource_assessment_cases, collect_padded_assessment,
    collect_padded_video, _write_new_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('evaluation', 'video'), required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--operation-key', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--gate-manifest', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--control-steps', type=int, required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec.validate_sources()
    if spec.code_snapshot()['code_sha256'] != args.code_sha256:
        raise ValueError('external worker source snapshot mismatch')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise ValueError('external worker checkpoint SHA mismatch')
    expected_n = 100 if args.mode == 'evaluation' else 1
    if (args.num_envs != expected_n or args.control_steps != 5005
            or (args.width, args.height, args.fps) != (640, 480, 25)):
        raise ValueError('external worker must use the exact Q009 workload')
    if not args.ledger.is_file():
        raise ValueError('child cannot create a ledger or execute without a parent reservation')
    from src.tasks.velocity.v09.budget import AppendOnlyLedger
    ledger = AppendOnlyLedger(args.ledger, run_id=args.run_id)
    snapshot = ledger.snapshot()
    validate_child_reservation(snapshot, args.operation_key, mode=args.mode,
                               checkpoint_sha256=args.checkpoint_sha256)
    manifest = load_manifest(args.gate_manifest, purpose='gate')
    if manifest['suite'] != 'gate_A':
        raise ValueError('resource child uses gate_A only, never sealed final cases')
    cases = resource_assessment_cases(manifest['cases'], num_envs=args.num_envs)
    if not args.execute:
        print(json.dumps(dict(mode=args.mode, num_envs=args.num_envs, control_steps=5005,
                              reservation_verified=True, physical_control_steps=0)))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from src.tasks.velocity.v09.env import V09S1Env
    from src.tasks.velocity.v09.model import V09ActorCritic
    from src.tasks.velocity.v09.checkpoint import load_checkpoint
    env = renderer = writer = None
    error = result = None
    try:
        payload = load_checkpoint(args.checkpoint, device='cpu')
        if payload['metadata']['course'] != 'S1.2' or payload['metadata']['code_sha256'] != args.code_sha256:
            raise ValueError('external worker requires the frozen full S1.2 parent')
        policy = V09ActorCritic().to(args.device)
        policy.load_state_dict(payload['model'], strict=True)
        policy.eval()
        env = V09S1Env(cases, device=args.device, seed=42)
        if args.mode == 'evaluation':
            result = collect_padded_assessment(env, policy, control_steps=5005)
        else:
            from src.tasks.velocity.v09.video import MujocoSnapshotRenderer, FFmpegWriter
            renderer = MujocoSnapshotRenderer(env, width=640, height=480)
            writer = FFmpegWriter(args.output_dir / 'resource_video.mp4', width=640, height=480, fps=25)
            result = collect_padded_video(env, policy, cases[0], args.output_dir / 'resource_video.mp4',
                                          renderer=renderer, writer=writer, control_steps=5005)
    except BaseException as failure:
        error = dict(type=type(failure).__name__, message=str(failure))
    finally:
        cleanups = ([writer.abort] if writer is not None and not writer.finished else [])
        cleanups += [resource.close for resource in (renderer, env) if resource is not None]
        for cleanup in cleanups:
            try:
                cleanup()
            except BaseException as failure:
                error = dict(type=type(failure).__name__, message='cleanup: ' + str(failure))
    if error is not None:
        _write_new_json(args.output_dir / 'worker_error.json', dict(error,
            sds_sha256=spec.SDS_SHA256, operation_key=args.operation_key,
            parent_settlement_required=True, child_budget_mutations=0))
        return 2
    result.update(schema_version=11, mode=args.mode, sds_sha256=spec.SDS_SHA256,
        source_sha256=spec.SOURCE_SHA256, code_sha256=args.code_sha256,
        checkpoint_sha256=args.checkpoint_sha256, operation_key=args.operation_key,
        reservation_verified=True, reservation_ledger_sha256=snapshot['ledger_sha256'],
        manifest_sha256=manifest['manifest_sha256'], child_budget_mutations=0)
    result['report_sha256'] = sha256(result)
    _write_new_json(args.output_dir / 'worker_result.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
