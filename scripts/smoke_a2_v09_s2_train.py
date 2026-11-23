"""Real S2TrainEnv smoke trace for V13 evidence (GPU, small batch)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_concurrency, s2_train_real


def _role_indices(plan, role_filter):
    if role_filter is None:
        return list(range(plan['num_envs']))
    if role_filter == 'stairs':
        names = ('stairs_walk',)
    elif role_filter == 'platform':
        names = ('platform_walk',)
    else:
        names = ('flat_walk', 'flat_stop')
    rows = plan.get('rows') or []
    return [i for i, row in enumerate(rows) if row.get('role') in names]


def _plan(args):
    plan = s2_train_real.build_env_plan(args.num_envs, course=args.course)
    selected = _role_indices(plan, args.role_filter)
    if not selected:
        raise SystemExit(f'no env for role filter {args.role_filter!r}')
    return dict(schema='a2_v09_s2_smoke_plan_v1', course=args.course,
                num_envs=args.num_envs, steps=args.steps,
                role_filter=args.role_filter, selected_envs=selected,
                role_counts=plan['role_counts'], device=args.device,
                attempt_id=args.attempt_id, ledger=str(args.ledger) if args.ledger else None,
                output_dir=str(args.output_dir) if args.output_dir else None,
                physics_started=False, training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', default='S2.1', choices=('S2.1', 'S2.3'))
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--role-filter', choices=('stairs', 'platform', 'flat'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--attempt-id', default='s2.1-v13-smoke')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not (args.dry_run or args.execute):
        raise SystemExit('use --dry-run or --execute')
    if args.num_envs < 1:
        raise SystemExit('num_envs must be positive')
    if args.steps < 1:
        raise SystemExit('steps must be positive')
    plan = _plan(args)
    if args.dry_run:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = args.output_dir or (ROOT / 'logs/a2_v09/s2/smoke' / stamp)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'smoke_plan.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(plan=plan, path=str(path)), ensure_ascii=False,
                         indent=2))
        return 0
    try:
        window = s2_concurrency.require_safe_window(device=args.device)
    except s2_concurrency.ConcurrencyError as error:
        raise SystemExit(str(error)) from error
    if args.ledger is not None and not args.ledger.is_file():
        raise SystemExit('ledger missing')

    from src.tasks.velocity.v09 import s2_env, s2_physics, s2_train
    out = args.output_dir or (ROOT / 'logs/a2_v09/s2/smoke' /
                              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / 'smoke_trace.jsonl'
    if trace_path.exists():
        raise FileExistsError(trace_path)
    env = s2_train_real.S2TrainEnv(num_envs=args.num_envs, device=args.device,
                                   course=args.course)
    actor = s2_physics.load_s1_actor(device=args.device)
    trainer = s2_train.S2Trainer(model=actor['model'], env=env,
                                 device=args.device, attempt_id=args.attempt_id,
                                 course=args.course)
    obs = env.reset()
    selected = plan['selected_envs']
    rows = []
    with trace_path.open('x', encoding='utf-8') as handle:
        for step in range(args.steps):
            dist = trainer.model.action_distribution(obs)
            action = dist.sample()
            obs, reward, done, info = env.step(action.detach())
            phase_values = info.get('phase') or [None] * args.num_envs
            for index in selected:
                row = dict(
                    step=step, env_id=index,
                    phase=phase_values[index],
                    local_x=float(info['local_x'][index]),
                    c_request=float(info['c_request'][index]),
                    c_exec=float(info['c_exec'][index]),
                    a_request=s2_env.jsonable(info['a_request'][index]),
                    a_applied=s2_env.jsonable(info['a_applied'][index]),
                    success=bool(info['success'][index]),
                    failure=bool(info['failure'][index]),
                    timeout=bool(info['timeout'][index]),
                    top_arrived=bool(info['top_arrived'][index]),
                    done=bool(done[index]))
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    env.close()
    summary = dict(plan=plan, concurrency=window, rows=len(rows),
                   trace=str(trace_path), completed=True)
    (out / 'smoke_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
