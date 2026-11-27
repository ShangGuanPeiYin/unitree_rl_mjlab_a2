#!/usr/bin/env python3
"""Manual full SDS execution using the existing run, immutable evidence and budgets."""
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
    parser.add_argument('--num-envs', type=int, choices=(1024,), default=1024)
    parser.add_argument('--rollout', type=int, choices=(48,), default=48)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--resume-reason', default='User manual restart after requested Ctrl+C')
    args = parser.parse_args(argv)
    from src.tasks.velocity.v10.orchestration import Orchestrator, preflight_formal_handoff
    base = ROOT / 'logs/a2_v10'
    run = base / 'formal_A1_20261127_approved'
    engineering = base / 'engineering_admission_formal_handoff_20261127.json'
    compatibility = base / 'formal_handoff_compatibility_20261127.json'
    plan = base / 'formal_handoff_plan_20261127.json'
    selection = run / 'resource_probe_v2_sds_resume_20261127_183400/selection.json'
    checked = preflight_formal_handoff(plan, engineering, compatibility)
    print(json.dumps(dict(mode=args.mode, run=str(run), num_envs=args.num_envs,
        rollout=args.rollout, seed=args.seed, device=args.device,
        save_every_rounds=50, dagger_budget_rounds=dict(D1=500,D2=1000,D3=1500),
        ppo_budget_rounds=1000, retained_trial_rounds=50,
        operation_logs=str(run/'operations'), preflight_passed=checked['passed']), ensure_ascii=False, default=str), flush=True)
    if args.mode == 'check':
        return 0
    selected = json.loads(selection.read_text())
    ctl = Orchestrator(run, engineering, selection, device=args.device,
        resume_reason=args.resume_reason,
        global_probe_ledger=base/'global_probe_costs.jsonl',
        probe_budget_authorization=base/'probe_authorization_A1_v2_inherited_20261127.json',
        probe_sequence_id=selected['probe_sequence_id'],
        evaluation_ledger=base/'formal_A_20261127_approved/evaluation_costs.jsonl',
        formal_handoff_plan=plan, source_compatibility=compatibility,
        learning_ledger=run/'learning_costs.jsonl')
    result = ctl.run(teachers_only=args.mode == 'teachers')
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
