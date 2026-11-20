"""Record all75 frozen DIAG case0 videos, including failures and pre-reset poses."""
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
from src.tasks.velocity.v09.budget import AppendOnlyLedger, batch_worst_cost
from src.tasks.velocity.v09.cases import load_manifest, canonical_json
from src.tasks.velocity.v09.video import VIDEO_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--segment-id', required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--policy', choices=('teacher', 'student'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--obligation-id', choices=tuple('video_' + item for item in VIDEO_MODELS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec.validate_sources()
    snapshot = spec.code_snapshot()
    if snapshot['code_sha256'] != args.code_sha256:
        raise ValueError('video code differs from independently approved source SHA')
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        raise ValueError('video checkpoint SHA mismatch')
    manifest = load_manifest(args.run_dir / 'cases/S1_DIAG.json', purpose='evidence')
    selected = [case for case in manifest['cases'] if case['case_id'] == 0]
    plan = batch_worst_cost(selected, batch_size=1)
    if (args.width, args.height, args.fps) != (640, 480, 25):
        raise ValueError('fixed video contract is640x480 at25fps')
    if not args.execute:
        print(json.dumps(dict(cases=75, worst_control_samples=plan['control_samples'],
            manifest_sha256=manifest['manifest_sha256'], checkpoint_sha256=digest,
            width=args.width, height=args.height, fps=args.fps, device=args.device,
            formal_physics_started=False), indent=2))
        return 0
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from src.tasks.velocity.v09.env import V09S1Env
    from src.tasks.velocity.v09.model import Frozen6400Teacher, V09ActorCritic
    from src.tasks.velocity.v09.checkpoint import load_checkpoint
    from src.tasks.velocity.v09.video import (
        FFmpegWriter, MujocoSnapshotRenderer, record_video_suite,
    )
    if args.policy == 'teacher':
        if args.obligation_id != 'video_6400' or digest != spec.SOURCE_SHA256:
            raise ValueError('teacher video must bind the unique6400 obligation/source')
        policy = Frozen6400Teacher.from_6400_checkpoint(args.checkpoint, device=args.device)
    else:
        if args.obligation_id == 'video_6400':
            raise ValueError('6400 obligation cannot use a student')
        payload = load_checkpoint(args.checkpoint, device=args.device)
        policy = V09ActorCritic().to(args.device)
        policy.load_state_dict(payload['model'], strict=True)
    policy.eval()
    ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
    if not ledger.snapshot()['stages']['S1']['envelope']:
        raise RuntimeError('full stage budget must be admitted before any video physics')
    report = record_video_suite(ledger=ledger, manifest=manifest, policy=policy,
        policy_map=args.policy, checkpoint_sha256=digest, code_sha256=snapshot['code_sha256'],
        obligation_id=args.obligation_id, segment_id=args.segment_id, output_dir=args.output_dir,
        env_factory=lambda case: V09S1Env([case], device=args.device, seed=42),
        renderer_factory=MujocoSnapshotRenderer, writer_factory=FFmpegWriter,
        width=args.width, height=args.height, fps=args.fps, source_snapshot=snapshot)
    print(json.dumps(dict(engineering_valid=report['engineering_valid'], full75=report['full75'],
                         report_sha256=report['report_sha256'], report=str(args.output_dir / 'video_summary.json')), indent=2))
    return 0 if report['engineering_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
