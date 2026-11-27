#!/usr/bin/env python3
"""Run the entire approved v1.0 protocol, stopping rather than bypassing a gate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,help='new run directory, or the same durable controller directory on recovery')
    parser.add_argument('--engineering-admission',required=True,help='independently reviewed engineering approval JSON tied to exact code/SDS')
    parser.add_argument('--evaluation-ledger',help='existing shared evaluation ledger; never copied or reset across new source runs')
    parser.add_argument('--resource-selection',help='optional existing exact-code two-stack selection; otherwise isolated 1024/512 probes run first')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--global-probe-ledger',default=str(ROOT/'logs/a2_v10/global_probe_costs.jsonl'))
    parser.add_argument('--probe-budget-authorization',help='explicit user-approved cumulative probe-only budget; mandatory before any probe child')
    parser.add_argument('--probe-sequence-id',help='durable unique sequence identity; changing run directory never resets the global ledger')
    parser.add_argument('--resume-reason',help='explicit engineering reason for retrying a failed non-final operation; never refunds budget')
    parser.add_argument('--plan-only',action='store_true',help='print immutable schedule/cost upper bound without simulation or training')
    a=parser.parse_args(argv)
    from src.tasks.velocity.v10.orchestration import Orchestrator,execution_plan
    if a.plan_only:
        from src.tasks.velocity.v10.cases import evaluation_plan
        result=dict(plan_1024=execution_plan(1024),plan_512=execution_plan(512),evaluation=evaluation_plan(),executed=False)
    else:
        result=Orchestrator(a.run_dir,a.engineering_admission,a.resource_selection,a.device,a.resume_reason,
            global_probe_ledger=a.global_probe_ledger,probe_budget_authorization=a.probe_budget_authorization,probe_sequence_id=a.probe_sequence_id,evaluation_ledger=a.evaluation_ledger).run()
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result.get('passed',True) else 1


if __name__=='__main__':raise SystemExit(main())
