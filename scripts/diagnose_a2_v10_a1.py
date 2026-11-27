#!/usr/bin/env python3
"""Run only the approved, bounded A1 112-case diagnostic; never formal admission."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--ledger',required=True)
    p.add_argument('--resource-selection',required=True)
    p.add_argument('--eval-limit',required=True,type=int)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    from src.tasks.velocity.v10.diagnostic_a1 import diagnostic_plan,run_diagnostic
    result=diagnostic_plan() if a.plan_only else run_diagnostic(output_dir=a.output_dir,ledger_path=a.ledger,resource_selection=a.resource_selection,eval_limit=a.eval_limit,device=a.device)
    if a.plan_only:result={k:v for k,v in result.items() if k!='cases'}
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result.get('passed',True) else 1

if __name__=='__main__':raise SystemExit(main())
