"""T11 same-process real/plane prefix replay with qpos/qvel hashes.

This is a diagnostic: no ledger write.  It runs the real branch first, then the
plane branch in the same Python process with the exact same action sequence,
and compares each frame's `sim_state_sha256` (qpos+qvel) until the plane map is
first consumed.  Exact equality is the required Q016 prefix proof; if the GPU
solver is not deterministic, the report says so instead of hiding it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_physics


def _select(matrix, sha_prefix):
    rows = [row for row in matrix['cases']
            if row['case_sha256'].startswith(sha_prefix)]
    if len(rows) != 1:
        raise SystemExit(f'case sha prefix selects {len(rows)} cases')
    return rows[0]


def _run(env, actor, actions=None, *, steps, output):
    from src.tasks.velocity.v09 import s2_env
    env.recorder.attach_trace(output / 'trace.jsonl')
    env.reset()
    used = []
    for step in range(steps):
        obs = env.observe()
        if actions is None:
            tensor_obs = {key: torch.as_tensor(s2_env.to_numpy(obs[key]),
                                               device=env.device,
                                               dtype=torch.float32)[None]
                          for key in ('proprio', 'history', 'heightmap', 'route_error')}
            action = s2_physics.actor_mean(actor, tensor_obs)[0].detach().cpu().numpy()
        else:
            action = np.asarray(actions[step], dtype=np.float32)
        used.append(action.copy())
        _, done, _ = env.step(action)
        if bool(done[0]):
            break
    return used


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-sha', required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--attempt-id')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    matrix = s2_cases.load_frozen_manifest(
        ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG.json', ROOT)
    case = _select(matrix, args.case_sha)
    attempt = args.attempt_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                  + '-' + case['case_sha256'][:8])
    out = args.output or (ROOT / 'logs/a2_v09/s2_0/pair' /
                          case['case_sha256'][:12] / attempt)
    out.mkdir(parents=True, exist_ok=True)
    actor = s2_physics.load_s1_actor(device=args.device)

    real, _ = s2_physics.build_s2_env(
        case, device=args.device, seed=42, map_branch='real', h_legacy=False)
    real_actions = _run(real, actor, steps=args.steps, output=out / 'real')
    real_map = real.evidence(0)['map_evidence']
    real_rows = list(real.recorder.rows)
    real.close = getattr(real.backend, 'close', lambda: None)
    real.close()
    del real
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    plane, _ = s2_physics.build_s2_env(
        case, device=args.device, seed=42, map_branch='plane_diagnostic', h_legacy=False)
    plane_actions = _run(plane, actor, actions=real_actions,
                         steps=len(real_actions), output=out / 'plane')
    plane_map = plane.evidence(0)['map_evidence']
    plane_rows = list(plane.recorder.rows)
    plane.close = getattr(plane.backend, 'close', lambda: None)
    plane.close()

    prefix = plane_map.get('first_consumed_replacement_frame')
    if prefix is None:
        prefix = min(len(real_rows), len(plane_rows))
    else:
        prefix = int(prefix)
    prefix = min(prefix, len(real_rows), len(plane_rows))
    diffs = []
    for index in range(prefix):
        a = real_rows[index].get('sim_state_sha256')
        b = plane_rows[index].get('sim_state_sha256')
        if a != b:
            diffs.append(dict(step=index, real=a, plane=b))
            break
    first = diffs[0] if diffs else None
    import hashlib
    real_chain = hashlib.sha256(''.join(
        real_rows[i]['sim_state_sha256'] for i in range(prefix)).encode()).hexdigest()
    plane_chain = hashlib.sha256(''.join(
        plane_rows[i]['sim_state_sha256'] for i in range(prefix)).encode()).hexdigest()
    report = dict(
        attempt_id=attempt, case_sha256=case['case_sha256'],
        task=case['task'], speed=case['speed'],
        sim_state='sha256(qpos,qvel float64 + shapes)',
        real_steps=len(real_rows), plane_steps=len(plane_rows),
        prefix_end_frame=prefix, prefix_exact=(first is None),
        branch_prefix_sha256=dict(real=real_chain, plane=plane_chain),
        post_reset_rule='terminal final row is post_reset; excluded from prefix comparison',
        first_state_hash_divergence=first,
        real_actions_sha256=hashlib.sha256(
            np.ascontiguousarray(np.asarray(real_actions, dtype=np.float32)).tobytes()
        ).hexdigest(),
        plane_actions_sha256=hashlib.sha256(
            np.ascontiguousarray(np.asarray(plane_actions, dtype=np.float32)).tobytes()
        ).hexdigest(),
        real_map_evidence=real_map, plane_map_evidence=plane_map,
        evidence=dict(real_trace=str(out / 'real' / 'trace.jsonl'),
                      plane_trace=str(out / 'plane' / 'trace.jsonl')),
        note=('Sequential same-process replay with identical actions. Exact qpos/qvel '
              'hashes at every prefix frame are the Q016 proof; any divergence is '
              'reported as engine nondeterminism, not tolerated as "close".'))
    (out / 'pair_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['prefix_exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
