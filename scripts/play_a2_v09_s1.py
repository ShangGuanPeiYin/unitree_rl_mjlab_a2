"""Native v09 S1 playback with the policy's actual cached height scan."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def select_case(manifest, lesson, vx, case_id):
    matches = [c for c in manifest['cases'] if c['task'] == 'stairs'
               and c['lesson'] == lesson and c['speed'] == vx and c['case_id'] == case_id]
    if len(matches) != 1:
        raise ValueError(f'没有唯一匹配的 S1 楼梯病例: {lesson}, vx={vx}, case_id={case_id}')
    return matches[0]


def configure_terrain(case, *, steps=None, width=None, length=None):
    """Optional process-local geometry; frozen cases/specs remain unchanged."""
    from src.tasks.velocity.v07 import terrain, spec
    tile = spec.terrain_spec(int(case['geometry']['tile_id']))
    if steps is not None or width is not None or length is not None:
        terrain.configure_playback_platform(
            tile_id=tile.tile_id, length=tile.platform_length if length is None else length,
            width=tile.width if width is None else width, steps=steps)
    played = terrain.build_terrain_specs()[tile.tile_id]
    return played, terrain.platform_width(played)


def continuation_state(state, reached):
    """Suppress only successful termination and post-top timeout, preserving failure."""
    from dataclasses import replace
    import torch
    reached = reached | state.success
    continuing = reached & ~state.failure
    reason = torch.where(continuing & ((state.reason == 1) | (state.reason == 2)),
                         torch.zeros_like(state.reason), state.reason)
    return replace(state, success=torch.zeros_like(state.success),
                   timeout=state.timeout & ~continuing, reason=reason), reached


def enable_continue_after_success(env):
    """Playback-only stair lifecycle; no bounded S1 acceptance trace is produced."""
    from types import MethodType
    import torch
    from src.tasks.velocity.v07.env import V07Env

    if any(case['task'] != 'stairs' for case in env.v09_cases):
        raise ValueError('Continuation is only supported for stair playback')
    env.playback_reached_top = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    original_reset = env._reset_ids

    def task_state(self):
        state = V07Env._task_state(self)
        self._flat_outcome = {}
        self._physical_failure = state.failure.clone()
        state, reached = continuation_state(state, self.playback_reached_top)
        if bool((reached & ~self.playback_reached_top).any()):
            print('TOP REACHED: continuing forward on the landing; no success reset/timeout.', flush=True)
        self.playback_reached_top = reached
        return state

    def reset_ids(self, ids, **kwargs):
        original_reset(ids, **kwargs)
        self.playback_reached_top[ids] = False

    env._task_state = MethodType(task_state, env)
    env._reset_ids = MethodType(reset_ids, env)
    # This mode exceeds the frozen S1 evidence horizon; use legacy diagnostic
    # records instead of accumulating/claiming S1 acceptance trajectories.
    env._episode_records = MethodType(V07Env._episode_records, env)
    env._outcomes = MethodType(V07Env._outcomes, env)


def playback_course(payload, *, allow_s2):
    meta = payload.get('metadata', {})
    if payload.get('schema') != 11:
        raise ValueError('Playback requires schema11 v09 checkpoints')
    if meta.get('stage') == 'S1' and meta.get('course') in ('S1.1', 'S1.2'):
        return meta['course']
    if meta.get('stage') == 'S2' and meta.get('course') in ('S2.1', 'S2.2', 'S2.3'):
        if not allow_s2:
            raise ValueError('S2 weights require --allow-s2-diagnostic for S1-scene playback')
        return 'S1.2'
    raise ValueError('Unsupported checkpoint stage/course')


def scan_bounds(grid, spacing):
    return (float(grid[..., 0].min()) - spacing / 2,
            float(grid[..., 0].max()) + spacing / 2,
            float(grid[..., 1].min()) - spacing / 2,
            float(grid[..., 1].max()) + spacing / 2)


def draw_scan(scene, env):
    """Decorations only; use the same cached frame and physical query as the scan."""
    import numpy as np
    import mujoco
    from src.tasks.velocity.v09.map import GRID_STEP, query_surface

    cache = env.map_cache
    z, _ = query_surface(cache.sample_points_world[0], cache.sample_tile[0])
    xy = cache.sample_points_world[0].detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    channels = cache.map[0].detach().cpu().numpy()
    origin = cache.sample_position[0].detach().cpu().numpy()
    yaw = float(cache.sample_yaw[0])
    plane_z = float(origin[2]) + 0.35
    scene.ngeom = 0
    if scene.maxgeom < z.size + 5:
        raise RuntimeError('Insufficient scan decoration capacity')

    for i, j in np.ndindex(z.shape):
        valid = channels[1, i, j] >= 0.5
        color = ((0., .9, 1., 1.) if abs(channels[0, i, j]) < 1.
                 else (1., .45, .05, 1.)) if valid else (1., .1, .25, 1.)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, .012), np.array([*xy[i, j], z[i, j] + .015 if valid else plane_z]),
                           np.eye(3).ravel(), np.array(color, dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    def world(x, y):
        return np.array([origin[0] + math.cos(yaw)*x - math.sin(yaw)*y,
                         origin[1] + math.sin(yaw)*x + math.cos(yaw)*y, plane_z])

    x0, x1, y0, y1 = scan_bounds(cache.grid.detach().cpu().numpy(), GRID_STEP)
    corners = [world(x1, y0), world(x1, y1), world(x0, y1), world(x0, y0)]
    segments = [(corners[i], corners[(i+1) % 4], False) for i in range(4)]
    segments.append((world(0, 0), world(.4, 0), True))
    for a, b, arrow in segments:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), np.array([1., 1., 0., 1.], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE,
                            .015 if arrow else 3., a, b)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '--checkpoint-file', dest='checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--allow-s2-diagnostic', action='store_true', help='Play S2 weights in the S1 diagnostic scene; not S2 evaluation or resume')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'logs/a2_v09/s1_seed42_r3/cases/S1_DIAG.json')
    parser.add_argument('--lesson', default='L40')
    parser.add_argument('--case-id', type=int, default=0)
    parser.add_argument('--stair-steps', type=int, default=None, help='Temporary playback stair count')
    parser.add_argument('--platform-width', type=float, default=None, help='Temporary landing width in metres; stair width unchanged')
    parser.add_argument('--platform-length', type=float, default=None, help='Temporary landing length in metres (>=2)')
    parser.add_argument('--continue-after-success', action='store_true', help='Keep walking after top success; retain failure resets, ignore post-top timeout')
    parser.add_argument('--vx', type=float, default=0.5)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--wz', type=float, default=0.0)
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--num-envs', type=int, choices=[1], default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--viewer', choices=['native'], default='native')
    parser.add_argument('--duration', type=float, default=None, help='Playback seconds; otherwise until window closes')
    parser.add_argument('--headless-steps', type=int, default=0, help='Finite physics/scan smoke without a window')
    args = parser.parse_args(argv)
    if args.platform_length is not None and (not math.isfinite(args.platform_length) or args.platform_length < 2.0):
        parser.error('platform-length must be finite and >= 2 metres')
    if args.stair_steps is not None and args.stair_steps <= 0:
        parser.error('stair-steps must be positive')
    if args.platform_width is not None and (not math.isfinite(args.platform_width) or not 0.8 <= args.platform_width <= 4.0):
        parser.error('platform-width must be finite and within [0.8, 4] metres')
    if args.vy != 0 or args.wz != 0:
        parser.error('Frozen S1 stairs require --vy 0 --wz 0')
    if args.headless_steps < 0 or (args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0)):
        parser.error('headless-steps must be nonnegative and duration must be finite and positive')
    if args.duration is not None and args.headless_steps:
        parser.error('duration and headless-steps are mutually exclusive')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        parser.error('Checkpoint SHA256 mismatch')
    case = select_case(json.loads(args.manifest.read_text()), args.lesson, args.vx, args.case_id)
    if not args.headless_steps and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Native viewer requires a desktop DISPLAY; use --headless-steps for a smoke check')

    from a2_v07_bootstrap import activate
    activate(ROOT)
    import torch
    import mujoco
    from src.tasks.velocity.v09.model import V09ActorCritic
    from src.tasks.velocity.v09.env import V09S1Env
    from src.tasks.velocity.v09.spec import CONTROL_DT
    from src.tasks.velocity.v09.map import GRID_X, GRID_Y, GRID_STEP, MAP_HZ, MAP_SHAPE

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = payload.get('metadata', {})
    try:
        course = playback_course(payload, allow_s2=args.allow_s2_diagnostic)
    except ValueError as error:
        parser.error(str(error))
    if meta.get('stage') == 'S2':
        print('DIAGNOSTIC ONLY: S2 weights in the S1 scene; not S2 evaluation/resume.', flush=True)
    if payload.get('architecture') != 'a2-v09-native40x30-route-history':
        parser.error('Incompatible model architecture')
    played_tile, landing_width = configure_terrain(case, steps=args.stair_steps, width=args.platform_width, length=args.platform_length)
    print(f'Playback geometry: stairs={played_tile.N}, h={played_tile.h:g}m, depth={played_tile.step_depth:g}m, '
          f'stair width={played_tile.width:g}m, landing length={played_tile.platform_length:g}m, '
          f'landing width={landing_width:g}m; demo only, not frozen-case evaluation', flush=True)
    torch.manual_seed(args.seed)
    model = V09ActorCritic().to(args.device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    scan_text = (f'Scan {GRID_X[1]-GRID_X[0]:g}x{GRID_Y[1]-GRID_Y[0]:g}m | '
                 f'front {GRID_X[1]:g} rear {-GRID_X[0]:g} left {GRID_Y[1]:g} right {-GRID_Y[0]:g}m | '
                 f'{MAP_SHAPE[1]}x{MAP_SHAPE[2]} {GRID_STEP:g}m {MAP_HZ}Hz')
    print(f'{args.lesson}, vx={args.vx}, vy={args.vy}, wz={args.wz}, seed={args.seed}, '
          f'envs=1, deterministic, continue-after-success={args.continue_after_success}\n{scan_text}\n'
          'Cyan=valid, orange=clipped, red=invalid; yellow=scan boundary/forward.', flush=True)
    env = V09S1Env([case], device=args.device, seed=args.seed, course=course)
    try:
        if args.continue_after_success:
            enable_continue_after_success(env)
        obs, _ = env.reset()
        with torch.inference_mode():
            if args.headless_steps:
                scene = mujoco.MjvScene(env.sim.mj_model, maxgeom=4096)
                for _ in range(args.headless_steps):
                    obs, _, _, _, _ = env.step(model.act(obs, deterministic=True))
                    draw_scan(scene, env)
                assert torch.isfinite(env.sim.data.qpos).all()
                print(f'PASS: {args.headless_steps} policy steps, finite qpos, {scene.ngeom} scan decorations', flush=True)
                return
            import mujoco.viewer
            data = mujoco.MjData(env.sim.mj_model)
            with mujoco.viewer.launch_passive(env.sim.mj_model, data) as viewer:
                viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 120., -20., 3.8
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  scan_text.replace(' | ', '\n'),
                                  'Cyan: valid / orange: clipped / red: invalid\nYellow: boundary + forward'))
                limit = None if args.duration is None else max(1, round(args.duration / CONTROL_DT))
                steps = 0
                while viewer.is_running() and (limit is None or steps < limit):
                    start = time.perf_counter()
                    obs, _, _, _, _ = env.step(model.act(obs, deterministic=True))
                    with viewer.lock():
                        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                        mujoco.mj_forward(env.sim.mj_model, data)
                        viewer.cam.lookat[:] = env.robot.data.root_link_pos_w[0].cpu().numpy()
                        draw_scan(viewer.user_scn, env)
                    viewer.sync()
                    steps += 1
                    time.sleep(max(0., CONTROL_DT - (time.perf_counter() - start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
