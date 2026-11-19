"""Live v07 L40 playback with an explicitly extended landing, without resets."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import time

from a2_v07_bootstrap import activate
activate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--platform-length', type=float, default=20.0)
    parser.add_argument('--vx', type=float, default=0.6)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--wz', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--camera-azimuth', type=float, default=90.0)
    parser.add_argument('--camera-elevation', type=float, default=-28.0)
    parser.add_argument('--camera-distance', type=float, default=3.8)
    parser.add_argument('--headless-steps', type=int, default=0,
                        help='Positive value runs a finite smoke check without a window')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.vx, args.vy, args.wz)) or args.vx <= 0:
        parser.error('L40 playback requires finite commands and vx > 0')
    if args.headless_steps < 0:
        parser.error('--headless-steps must be >= 0')
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        parser.error(f'Checkpoint SHA256 mismatch: {digest}')

    import torch
    import mujoco
    from src.tasks.velocity.v07 import terrain, spec
    from src.tasks.velocity.v07.env import V07Env
    from src.tasks.velocity.v07.model import V07ActorCritic
    terrain.configure_playback_platform(tile_id=21, length=args.platform_length)
    torch.manual_seed(args.seed)
    model = V07ActorCritic().to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    env = V07Env(num_envs=1, device=args.device, seed=args.seed, phase='L',
                 substage='L40', evaluation=True, fixed_vx=args.vx, frontier_tile_id=21,
                 initial_cases=[dict(x=-1., y=0., yaw=0., vx=args.vx)])
    try:
        env.reset()
        # User requested continuous observation including success/failure aftermath.
        # Keep event calculations but suppress teleport/reset for this viewer only.
        env._reset_ids = lambda ids, **kwargs: None
        command = torch.tensor([args.vx, args.vy, args.wz], device=args.device)
        print(f'L40: 40 steps, h=d=0.2m, landing x=[8,{8 + args.platform_length:g}]m; '
              f'constant command={command.tolist()}; no automatic reset', flush=True)

        @torch.inference_mode()
        def advance():
            env.command[:] = command
            obs = env._observations(update=False)
            env.step(model.act(obs, deterministic=True))

        if args.headless_steps:
            for _ in range(args.headless_steps):
                advance()
            assert torch.isfinite(env.sim.data.qpos).all()
            print(f'PASS: {args.headless_steps} policy steps, finite qpos; '
                  f'position={env.robot.data.root_link_pos_w[0].tolist()}', flush=True)
            return
        import mujoco.viewer
        mj_model = env.sim.mj_model
        mj_data = mujoco.MjData(mj_model)
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
            viewer.cam.distance = args.camera_distance
            while viewer.is_running():
                start = time.perf_counter()
                advance()
                with viewer.lock():
                    mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                    mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                    mujoco.mj_forward(mj_model, mj_data)
                    viewer.cam.lookat[:] = env.robot.data.root_link_pos_w[0].cpu().numpy()
                viewer.sync()
                time.sleep(max(0.0, spec.CONTROL_DT - (time.perf_counter() - start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
