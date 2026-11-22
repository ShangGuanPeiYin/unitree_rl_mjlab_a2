"""Compare two recorded S2/S1 trace.jsonl files for observation/action parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_env, s2_parity, s2_physics
from src.tasks.velocity.v09.cases import build_s1_manifest
from src.tasks.velocity.v07.events import euler_heading
import numpy as np


def run_s1_reference(args):
    from src.tasks.velocity.v09.env import V09S1Env
    manifest = build_s1_manifest('S1_DIAG')
    cases = [row for row in manifest['cases']
             if row['task'] == 'flat_walk' and row['speed'] == 0.3]
    case = cases[args.case]
    env = V09S1Env([case], device=args.device, seed=42)
    actor = s2_physics.load_s1_actor(device=args.device)
    output = args.output or (ROOT / 'logs/a2_v09/s2_0/parity/s1_flat_case0/trace.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f'refusing to overwrite {output}')
    env.reset()
    rows = []
    for step in range(args.steps):
        teacher, student = env.observation_pair()
        tensor_obs = {key: student[key] for key in
                      ('proprio', 'history', 'heightmap', 'route_error')}
        action = s2_physics.actor_mean(actor, tensor_obs)
        _, _, terminated, truncated, _ = env.step(action)
        data = env.robot.data
        origin = env.origins[env.tile_ids[0]]
        local = data.root_link_pos_w[0] - origin
        feet = data.site_pos_w[0, env.foot_ids]
        pair_count = int((env.contacts.top_forces[0] > 0).sum())
        row = dict(step=step, route_x=float(local[0]),
                   base_xy=[float(local[0]), float(local[1])],
                   feet_xy=feet[:, :2].detach().cpu().numpy().tolist(),
                   feet_route_x=(feet[:, 0] - origin[0]).detach().cpu().numpy().tolist(),
                   pitch=float(euler_heading(data.root_link_quat_w[0:1])[1][0]),
                   velocity_route=float(data.root_link_lin_vel_w[0, 0]),
                   c_exec=float(env.sample_command[0, 0]),
                   c_request=float(env.command[0, 0]),
                   a_request=env.action_manager.action[0].detach().cpu().numpy().tolist(),
                   a_applied=env.action_term._processed_actions[0].detach().cpu().numpy().tolist(),
                   pd_target=env.action_term._processed_actions[0].detach().cpu().numpy().tolist(),
                   foot_contact=env.contacts.raw[0].detach().cpu().numpy().tolist(),
                   foot_on_platform=env.current_platform[0].detach().cpu().numpy().tolist(),
                   foot_normal_force=env.contacts.top_forces[0].sum(-1).detach().cpu().numpy().tolist(),
                   map_valid=env.paired_maps.student.map[0, 1].detach().cpu().numpy().astype(bool).tolist(),
                   contact_pairs=pair_count,
                   top_arrived=False, failure=bool(terminated[0]),
                   stop_request=False, terminal=bool(terminated[0] or truncated[0]))
        rows.append(row)
        with output.open('a', encoding='utf-8') as handle:
            handle.write(s2_env.canonical_json(s2_env._jsonable(row)) + '\n')
            handle.flush()
            import os
            os.fsync(handle.fileno())
        if terminated[0] or truncated[0]:
            break
    print(json.dumps(dict(mode='s1_reference', trace=str(output),
                          frames=len(rows), case_sha256=case['case_sha256'],
                          reservation=dict(obligation='s2_0_matrix_two_branches',
                          samples=len(rows), note='main conversation handles S2 ledger')),
                     ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path)
    parser.add_argument('--right', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-s1-reference', action='store_true')
    parser.add_argument('--case', type=int, default=0)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.run_s1_reference:
        return run_s1_reference(args)
    left, right = s2_parity.load_trace(args.left), s2_parity.load_trace(args.right)
    report = dict(left=str(args.left), right=str(args.right),
                  frames=len(left), comparison=s2_parity.compare_trace_tables(left, right),
                  friction=s2_parity.friction_contract())
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
