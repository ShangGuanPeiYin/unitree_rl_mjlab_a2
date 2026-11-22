"""T10 same-case S1/S2 observation parity with a flat-terrain S1 proxy.

The S1 reference is built from the S2 case's spawn/yaw/streams/command but the
S1 env is the frozen flat_walk task on flat tile 0, as required by DSH-V8-3.
Both envs receive the same S1-actor actions, so the report measures the
observation/state contract rather than policy drift.  Scope: exact on flat
cases; for a staircase S2 case the proxy is useful only before stair contact.
"""
from __future__ import annotations

import argparse
import copy
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
from src.tasks.velocity.v09 import s2_cases, s2_env, s2_execute, s2_ledger, s2_physics, s2_spec
from src.tasks.velocity.v09.cases import build_s1_manifest, sha256

OBS_KEYS = ('proprio', 'history', 'heightmap', 'route_error')
TOL = 1e-6
STRICT_TOL = 1e-6


def _numpy(value):
    array = np.asarray(s2_env.to_numpy(value), dtype=np.float32)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def _obs_digest(obs):
    digest = hashlib.sha256()
    for key in OBS_KEYS:
        value = np.ascontiguousarray(_numpy(obs[key]), dtype=np.float32)
        digest.update(key.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _obs_stat(obs):
    return {key: dict(shape=list(_numpy(obs[key]).shape),
                      min=float(np.min(_numpy(obs[key]))),
                      max=float(np.max(_numpy(obs[key]))),
                      sha256=_obs_digest({key: obs[key]}))
            for key in OBS_KEYS}


def _select_case(matrix, sha_prefix):
    rows = [row for row in matrix['cases']
            if row['case_sha256'].startswith(sha_prefix)]
    if len(rows) != 1:
        raise SystemExit(f'case-sha prefix must select exactly one S2 case, got {len(rows)}')
    return rows[0]


def _s1_proxy(case):
    base = next(row for row in build_s1_manifest('S1_DIAG')['cases']
                if row['task'] == 'flat_walk' and row['speed'] == 0.3)
    proxy = copy.deepcopy(base)
    proxy['spawn'] = copy.deepcopy(case['spawn'])
    proxy['speed'] = float(case['speed'])
    proxy['streams'] = copy.deepcopy(case['streams'])
    proxy['suite'] = 'PARITY_S1_PROXY'
    proxy['parity_source_s2_case_sha256'] = case['case_sha256']
    proxy['hard_timeout_steps'] = int(base['hard_timeout_steps'])
    proxy.pop('case_sha256', None)
    proxy['case_sha256'] = sha256(proxy)
    return proxy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-sha', required=True,
                        help='S2 case SHA or unique prefix')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--obligation-id', default='s2_0_pipeline_debug')
    parser.add_argument('--attempt-id')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-ledger', action='store_true',
                        help='only for CPU/dry diagnostics; not evidence')
    args = parser.parse_args(argv)
    if args.steps <= 0:
        raise SystemExit('--steps must be positive')
    matrix = s2_cases.load_frozen_manifest(
        ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG.json', ROOT)
    case = _select_case(matrix, args.case_sha)
    proxy = _s1_proxy(case)
    attempt_id = args.attempt_id or (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' +
        hashlib.sha256(f'{case["case_sha256"]}'.encode()).hexdigest()[:6])
    output = args.output or (ROOT / 'logs/a2_v09/s2_0/parity/t10' /
                             case['case_sha256'][:12] / attempt_id)
    output.mkdir(parents=True, exist_ok=True)
    ledger = None
    gate = None
    batch = None
    operation_id = f"parity:{case['case_sha256'][:12]}:{attempt_id}"
    receipt = None
    if not args.skip_ledger:
        ledger = s2_ledger.S2Ledger(args.ledger,
                                    run_id=s2_ledger.read_run_id(args.ledger))
        gate = s2_execute.ReservationGate(ledger, obligation_id=args.obligation_id)
        batch = dict(branch='real', group_role_sha256='parity',
                     control_samples=2 * int(args.steps),
                     case_sha256s=[case['case_sha256']], case_ids=[case['case_id']],
                     batch_size=1,
                     max_hard_timeout_steps=int(case['hard_timeout_steps']))
        receipt = gate.begin_batch(batch, operation_id=operation_id)

    actor = s2_physics.load_s1_actor(device=args.device)
    s1_rows = []
    s1_obs = []
    actions = []
    s1_steps = 0
    s2_steps = 0
    try:
        from src.tasks.velocity.v09.env import V09S1Env
        s1 = V09S1Env([proxy], device=args.device, seed=42)
        s1.reset()
        for step in range(int(args.steps)):
            teacher, student = s1.observation_pair()
            obs = {key: _numpy(student[key]) for key in OBS_KEYS}
            tensor_obs = {key: torch.as_tensor(obs[key], device=args.device,
                                               dtype=torch.float32)[None]
                          for key in OBS_KEYS}
            action = s2_physics.actor_mean(actor, tensor_obs)
            actions.append(action[0].detach().cpu().numpy().copy())
            data = s1.robot.data
            origin = s1.origins[s1.tile_ids[0]]
            local = data.root_link_pos_w[0] - origin
            s1_rows.append(dict(step=step, obs_sha256=_obs_digest(obs),
                                obs_stats={key: dict(min=float(obs[key].min()),
                                                     max=float(obs[key].max()))
                                           for key in OBS_KEYS},
                                route_x=float(local[0]),
                                sim_state_sha256=s2_physics.sim_state_digest(
                                    s1.sim.data.qpos[0], s1.sim.data.qvel[0]),
                                qpos=s1.sim.data.qpos[0].detach().cpu().numpy().tolist(),
                                qvel=s1.sim.data.qvel[0].detach().cpu().numpy().tolist(),
                                action=action[0].detach().cpu().numpy().tolist()))
            s1_obs.append(obs)
            _obs, _reward, terminated, truncated, _info = s1.step(action)
            s1_steps += 1
            if bool(terminated[0] or truncated[0]):
                break
        del s1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        env, _runtime = s2_physics.build_s2_env(
            case, device=args.device, seed=42, map_branch='real', h_legacy=False)
        env.reset()
        s2_rows = []
        diffs = {key: [] for key in OBS_KEYS}
        dim_max = {key: None for key in OBS_KEYS}
        first_divergence = None
        first_strict_divergence = None
        first_sim_state_divergence = None
        state_dim_max = {'qpos': None, 'qvel': None}
        first_state_dim = None
        obs_dim_by_step = []
        for step, action in enumerate(actions):
            obs2 = env.observe()
            row = dict(step=step, obs_sha256=_obs_digest(obs2),
                       obs_stats={key: dict(min=float(_numpy(obs2[key]).min()),
                                            max=float(_numpy(obs2[key]).max()))
                                  for key in OBS_KEYS},
                       route_x=float(env.last.route_x),
                       sim_state_sha256=env.last.sim_state_sha256,
                       qpos=env.backend.env.sim.data.qpos[0].detach().cpu().numpy().tolist(),
                       qvel=env.backend.env.sim.data.qvel[0].detach().cpu().numpy().tolist(),
                       action=action.tolist())
            s2_rows.append(row)
            for key in OBS_KEYS:
                a = _numpy(s1_obs[step][key]).astype(np.float64)
                b = _numpy(obs2[key]).astype(np.float64)
                diff = (float(np.max(np.abs(a - b))) if a.shape == b.shape
                        else float('inf'))
                diffs[key].append(diff)
                flat = np.abs(a - b).reshape(-1)
                if step < 4:
                    obs_dim_by_step.append(dict(
                        step=step, key=key,
                        dims=[dict(index=int(i), max_abs=float(v))
                              for i, v in enumerate(flat) if v > 1e-7]))
                if dim_max[key] is None:
                    dim_max[key] = flat.copy()
                else:
                    dim_max[key] = np.maximum(dim_max[key], flat)
                if diff > TOL and first_divergence is None:
                    first_divergence = dict(step=step, key=key, max_abs=diff)
                if diff > STRICT_TOL and first_strict_divergence is None:
                    first_strict_divergence = dict(step=step, key=key, max_abs=diff)
            for state_key in ('qpos', 'qvel'):
                a_state = np.asarray(s1_rows[step][state_key], dtype=np.float64)
                b_state = np.asarray(row[state_key], dtype=np.float64)
                d_state = np.abs(a_state - b_state)
                state_dim_max[state_key] = (d_state.copy() if state_dim_max[state_key] is None
                                            else np.maximum(state_dim_max[state_key], d_state))
                if first_state_dim is None and float(np.max(d_state)) > 1e-7:
                    index = int(np.argmax(d_state))
                    first_state_dim = dict(step=step, key=state_key, index=index,
                                           max_abs=float(d_state[index]))
            if s1_rows[step]['sim_state_sha256'] != row['sim_state_sha256'] and first_sim_state_divergence is None:
                first_sim_state_divergence = dict(
                    step=step, s1=s1_rows[step]['sim_state_sha256'],
                    s2=row['sim_state_sha256'])
            _, done, _info = env.step(action)
            s2_steps += 1
            if bool(done[0]):
                break
    except Exception:
        if gate is not None and batch is not None:
            try:
                gate.finish_batch(batch, operation_id=operation_id,
                                  actual_samples=s1_steps + s2_steps)
            except Exception:
                pass
        raise
    settled = None
    if gate is not None and batch is not None:
        settled = gate.finish_batch(batch, operation_id=operation_id,
                                    actual_samples=s1_steps + s2_steps)

    s1_trace = output / 's1_reference.jsonl'
    s2_trace = output / 's2_reference.jsonl'
    for path, rows in ((s1_trace, s1_rows), (s2_trace, s2_rows)):
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                for row in rows), encoding='utf-8')
    summary = dict(
        attempt_id=attempt_id, case_sha256=case['case_sha256'],
        s1_proxy_case_sha256=proxy['case_sha256'],
        s1_proxy_suite=proxy['suite'], s2_task=case['task'],
        s2_stair_count=case['stair_count'], speed=case['speed'],
        spawn=case['spawn'], streams=case['streams'],
        command=case.get('command'), steps_requested=int(args.steps),
        s1_steps=s1_steps, s2_steps=s2_steps,
        obs_sha256_s1=[row['obs_sha256'] for row in s1_rows],
        obs_sha256_s2=[row['obs_sha256'] for row in s2_rows],
        action_sha256=hashlib.sha256(
            np.ascontiguousarray(np.asarray(actions, dtype=np.float32)).tobytes()
        ).hexdigest() if actions else None,
        obs_key_max_abs={key: (max(values) if values else None)
                         for key, values in diffs.items()},
        obs_dim_max_abs={key: [dict(index=int(i), max_abs=float(v))
                               for i, v in enumerate(values) if v > TOL]
                         for key, values in dim_max.items() if values is not None},
        first_sim_state_divergence=first_sim_state_divergence,
        sim_state_prefix_equal=(first_sim_state_divergence is None),
        first_state_dim=first_state_dim,
        obs_dim_by_step=obs_dim_by_step,
        state_dim_max_abs={key: (float(np.max(value)) if value is not None else None)
                           for key, value in state_dim_max.items()},
        first_divergence=first_divergence,
        first_strict_divergence=first_strict_divergence,
        tolerance=TOL, strict_tolerance=STRICT_TOL,
        parity_ok=(first_divergence is None),
        strict_parity_ok=(first_strict_divergence is None),
        attribution=('no reproducible observation-pipeline root cause; '
                     'different runs show inconsistent magnitudes (e.g. 4.65e-6 vs 1.75e-4); '
                     'strict consistency is not established; no 1e-3 practical pass'),
        scope=('S1 flat-terrain proxy from this S2 case; exact for flat, and '
               'staircase only before stair contact'),
        ledger=dict(obligation_id=args.obligation_id,
                    receipt=receipt, settled=settled),
        evidence_files={
            's1_reference': dict(path=str(s1_trace),
                                 sha256=hashlib.sha256(s1_trace.read_bytes()).hexdigest()),
            's2_reference': dict(path=str(s2_trace),
                                 sha256=hashlib.sha256(s2_trace.read_bytes()).hexdigest())},
    )
    path = output / 'parity_summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['parity_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
