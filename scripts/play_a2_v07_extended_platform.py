"""Live v07 L40 playback with an explicitly extended landing, without resets."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import time

from a2_v07_bootstrap import activate
activate()


def draw_heightmap(scene, env):
    """Render cached policy scan only; no simulator geoms or observations are edited."""
    import numpy as np
    import mujoco
    from src.tasks.velocity.v07.terrain import query_surface
    from src.tasks.velocity.v07.map import GRID_STEP

    cache = env.map_cache
    xy = cache.sample_points_world[0]
    # Query unclipped physical surface at the SAME cached world sample positions.
    # Inverting the clipped policy channel would falsely put deep ground near the body.
    heights, _ = query_surface(xy, cache.sample_tile[0])
    points = xy.detach().cpu().numpy()
    heights = heights.detach().cpu().numpy()
    channels = cache.map[0].detach().cpu().numpy()
    origin = cache.sample_position[0].detach().cpu().numpy()
    yaw = float(cache.sample_yaw[0])
    plane_z = float(origin[2]) + .35
    colors = np.empty((*heights.shape, 4), dtype=np.float32)
    colors[:] = (0., .9, 1., 1.)  # valid, unsaturated
    colors[np.abs(channels[0]) >= 1.] = (1., .45, .05, 1.)
    colors[channels[1] < .5] = (1., .1, .25, 1.)
    required = heights.size + colors[::4, ::4].shape[0] * colors[::4, ::4].shape[1] + 5
    if scene.maxgeom < required:
        raise RuntimeError(f'Heightmap overlay needs {required} visual geoms')
    scene.ngeom = 0
    identity = np.eye(3).ravel()

    def point(pos, color, radius):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, radius), np.asarray(pos), identity, color)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    for i in range(heights.shape[0]):
        for j in range(heights.shape[1]):
            z = heights[i, j] + .015 if channels[1, i, j] >= .5 else plane_z
            point((*points[i, j], z), colors[i, j], .012)
    # Sparse overhead projection keeps deep samples visible from a close side view.
    for i in range(0, heights.shape[0], 4):
        for j in range(0, heights.shape[1], 4):
            point((*points[i, j], plane_z), colors[i, j], .012)

    grid = cache.grid.detach().cpu().numpy()
    x0, x1 = grid[0, 0, 0] - GRID_STEP / 2, grid[-1, 0, 0] + GRID_STEP / 2
    y0, y1 = grid[0, 0, 1] - GRID_STEP / 2, grid[0, -1, 1] + GRID_STEP / 2

    def world(x, y):
        return np.array((origin[0] + math.cos(yaw)*x - math.sin(yaw)*y,
                         origin[1] + math.sin(yaw)*x + math.cos(yaw)*y, plane_z))

    def line(a, b, color, arrow=False):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.zeros(3), np.zeros(3), identity, np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE,
                            .015 if arrow else 3., a, b)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    corners = [world(x1, y0), world(x1, y1), world(x0, y1), world(x0, y0)]
    for i in range(4):
        line(corners[i], corners[(i+1) % 4], (1., 1., 0., 1.) if i == 0 else (.7, 1., 1., 1.))
    line(world(0, 0), world(.4, 0), (1., 1., 0., 1.), arrow=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--terrain', choices=('stairs', 'air-platform'), default='stairs')
    parser.add_argument('--platform-height', type=float, default=2.0)
    parser.add_argument('--stair-steps', type=int, default=40,
                        help='Playback-only stair count; frozen training terrain is unchanged')
    parser.add_argument('--platform-length', type=float, default=20.0)
    parser.add_argument('--platform-width', type=float, default=4.0)
    parser.add_argument('--vx', type=float, default=0.6)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--wz', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--camera-azimuth', type=float, default=90.0)
    parser.add_argument('--camera-elevation', type=float, default=-28.0)
    parser.add_argument('--camera-distance', type=float, default=3.8)
    parser.add_argument('--show-heightmap', action='store_true',
                        help='Show actual cached scan points and overhead scan footprint')
    parser.add_argument('--headless-steps', type=int, default=0,
                        help='Positive value runs a finite smoke check without a window')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.vx, args.vy, args.wz)) or args.vx <= 0:
        parser.error('Playback requires finite commands and vx > 0')
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
    air = args.terrain == 'air-platform'
    if air:
        terrain.configure_air_platform(length=args.platform_length, width=args.platform_width,
                                       height=args.platform_height)
    else:
        terrain.configure_playback_platform(tile_id=21, length=args.platform_length,
                                            width=args.platform_width, steps=args.stair_steps)
    torch.manual_seed(args.seed)
    model = V07ActorCritic().to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    env = V07Env(num_envs=1, device=args.device, seed=args.seed, phase='F0' if air else 'L',
                 substage=None if air else 'L40', evaluation=True, fixed_vx=args.vx, frontier_tile_id=0 if air else 21,
                 initial_cases=[dict(x=1.0 if air else -1., y=0., yaw=0., vx=args.vx)])
    try:
        env.reset()
        # User requested continuous observation including success/failure aftermath.
        # Keep event calculations but suppress teleport/reset for this viewer only.
        env._reset_ids = lambda ids, **kwargs: None
        command = torch.tensor([args.vx, args.vy, args.wz], device=args.device)
        description = (f'Air platform: {args.platform_length:g} x {args.platform_width:g}m, '
                       f'height={args.platform_height:g}m, spawn x=1m' if air else
                       f'Stairs: {args.stair_steps}, h=d=0.2m, '
                       f'landing {args.platform_length:g} x {args.platform_width:g}m')
        print(f'{description}; command={command.tolist()}; no automatic reset', flush=True)

        @torch.inference_mode()
        def advance():
            env.command[:] = command
            obs = env._observations(update=False)
            env.step(model.act(obs, deterministic=True))

        if args.show_heightmap:
            print('Scan: front 1.6m / rear 0.8m / left,right 0.8m; 48x32 at 5cm, 10Hz. '
                  'Cyan=valid; orange=height clipped; red=invalid; yellow=forward. '
                  'Overhead dots project the same samples; ground dots show physical height.', flush=True)
        if args.headless_steps:
            overlay = mujoco.MjvScene(env.sim.mj_model, maxgeom=4096) if args.show_heightmap else None
            for _ in range(args.headless_steps):
                advance()
                if overlay is not None:
                    draw_heightmap(overlay, env)
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
                    if args.show_heightmap:
                        draw_heightmap(viewer.user_scn, env)
                viewer.sync()
                time.sleep(max(0.0, spec.CONTROL_DT - (time.perf_counter() - start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
