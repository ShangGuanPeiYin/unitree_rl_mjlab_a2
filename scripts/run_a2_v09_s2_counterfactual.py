"""Q031 offline same-frame real/plane input counterfactual (CPU)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_counterfactual, s2_physics, s2_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--task')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    rows, metadata = s2_counterfactual.load_saved_inputs(
        args.inputs, args.metadata)
    task = args.task or metadata.get('task')
    if task is None:
        raise SystemExit('--task is required when inputs metadata has no task')
    actor = s2_physics.load_s1_actor(device=args.device)
    policy = lambda obs: s2_physics.actor_mean(actor, obs)
    report = s2_counterfactual.analyze(
        rows, policy=policy, task=task,
        case_sha256=metadata.get('case_sha256'),
        weight_sha256=metadata.get('weight_sha256'))
    output = args.output or (args.inputs.parent / 'counterfactual_report.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
