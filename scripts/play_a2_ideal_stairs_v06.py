"""v0.6 playback entry point (deterministic policy, no training).

The checkpoint SHA is mandatory: a v0.6 policy is only played from an explicitly
identified schema-8 file.  The script refuses R1/R2, smoke or probe containers and
requires the v0.6 final stair (tile ``L40``).

Without ``--execute`` it prints the validated playback plan only (no MuJoCo
context, no GPU, no video).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(str(ROOT))


def _rooted(path):
    """Run-local paths resolve against the project root, never the CWD."""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--phase', default='R')
    p.add_argument('--substage', default=None)
    p.add_argument('--vx', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--execute', action='store_true',
                   help='actually open the simulator (default: plan only)')
    p.add_argument('--record-video', action='store_true')
    return p


def plan(args):
    from src.tasks.velocity.v06 import model as model_module
    from src.tasks.velocity.v06 import recovery as recovery_module
    from src.tasks.velocity.v06 import speed

    checkpoint = _rooted(args.checkpoint)
    state = recovery_module.preload_checkpoint(checkpoint, args.checkpoint_sha256)
    speeds = speed.gate_speeds(args.phase)
    if not (min(speeds) - 1e-9 <= args.vx <= max(speeds) + 1e-9):
        raise SystemExit(f'--vx {args.vx} is outside the {args.phase} gate speeds {speeds}')
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=args.checkpoint_sha256,
                phase=args.phase, substage=args.substage, vx=float(args.vx),
                seed=int(args.seed), device=args.device, num_envs=int(args.num_envs),
                max_steps=int(args.max_steps), output_dir=str(args.output_dir),
                gate_speeds=list(speeds), deterministic=True,
                training_state=state['learning'],
                architecture=model_module.ARCHITECTURE,
                recipe=model_module.RECIPE_VERSION)


def main(argv=None):
    args = parser().parse_args(argv)
    record = plan(args)
    output = _rooted(args.output_dir)
    if args.execute:
        if output.exists():
            raise SystemExit(f'--output-dir already exists (refused): {output}')
        output.mkdir(parents=True, exist_ok=False)
    if args.execute:
        (output / 'plan.json').write_text(
            json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2, sort_keys=True))
    if not args.execute:
        print('playback plan only; add --execute to open the simulator')
        return 0
    from src.tasks.velocity.v06 import model as model_module
    from src.tasks.velocity.v06 import recovery as recovery_module
    from train_a2_ideal_stairs_v06 import build_environment

    checkpoint = _rooted(args.checkpoint)
    state = recovery_module.preload_checkpoint(checkpoint, args.checkpoint_sha256)
    model = model_module.V06ActorCritic().to(args.device)
    recovery_module.load_model_state(model, state['model'])
    model.eval()
    contract = dict(device=args.device, seed=int(args.seed))
    env = build_environment(contract, phase=args.phase, substage=args.substage,
                            evaluation=True, num_envs=int(args.num_envs),
                            fixed_vx=float(args.vx))
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    frames = 0
    with __import__('torch').no_grad():
        for _step in range(int(args.max_steps)):
            obs, _reward, terminated, truncated, _info = env.step(
                model.act(obs, deterministic=True))
            frames += 1
            if bool(terminated.any()) or bool(truncated.any()):
                break
    summary = dict(record, frames=int(frames))
    (output / 'playback.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
