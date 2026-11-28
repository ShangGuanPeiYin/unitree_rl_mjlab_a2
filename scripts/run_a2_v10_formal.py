#!/usr/bin/env python3
"""Manual full SDS execution: initialize from S2 at round zero, then resume this run."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'all', 'teachers'), default='all')
    parser.add_argument('--device', choices=('cuda:0',), default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--rollout', type=int, choices=(48,), default=48)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--resume-reason', default='User manual start or resume of fresh SDS run')
    args = parser.parse_args(argv)
    from src.tasks.velocity.v10.dynamic_execution import run_dynamic_execution
    if args.num_envs < 10:
        parser.error('--num-envs must be at least 10 to cover all six tasks')
    base = ROOT / 'logs/a2_v10'
    run = base / 'server_fresh'
    print(json.dumps(dict(mode=args.mode, num_envs=args.num_envs, rollout=args.rollout, seed=args.seed, device=args.device, save_every_rounds=50, resource_status='unverified_for_requested_N'), ensure_ascii=False), flush=True)
    result = run_dynamic_execution(
        run_dir=run, profile_path=run/f'execution_profile_N{args.num_envs}.json',
        num_envs=args.num_envs, device=args.device,
        teachers_only=args.mode == 'teachers', check_only=args.mode == 'check',
        resume_reason=args.resume_reason, fresh=True)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
