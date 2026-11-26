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


def select_case(manifest, lesson, vx, case_id, *, scene='stairs'):
    matches = [c for c in manifest['cases'] if c['task'] == ('flat_walk' if scene == 'flat' else 'stairs')
               and c['lesson'] == lesson and c['speed'] == vx and c['case_id'] == case_id]
    if len(matches) != 1:
        raise ValueError(f'没有唯一匹配的 S1 病例: {lesson}, vx={vx}, case_id={case_id}')
    return matches[0]


def resample_spawn(case, seed, *, x_bounds, y_bounds, yaw_bounds):
    """Copy a case and override only its pose for reproducible playback."""
    from copy import deepcopy
    import random
    result = deepcopy(case)
    rng = random.Random(seed)
    x, y, yaw = rng.uniform(*x_bounds), rng.uniform(*y_bounds), rng.uniform(*yaw_bounds)
    pose = result['spawn']
    origin = pose['frame_origin_world']
    z = pose['local_position'][2]
    pose.update(local_position_xy=[x, y], local_yaw_radians=yaw,
                local_position=[x, y, z], world_position_xy=[origin[0]+x, origin[1]+y],
                world_position=[origin[0]+x, origin[1]+y, origin[2]+z],
                quaternion_wxyz=[math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
    result['playback_source_case_sha256'] = result.pop('case_sha256')
    result['playback_spawn_seed'] = seed
    result['case_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def platform_start_case(case, tile, seed):
    from copy import deepcopy
    from src.tasks.velocity.v07 import spec
    copied = deepcopy(case)
    copied['spawn']['local_position'][2] += tile.N * tile.h
    x = tile.N * tile.step_depth + 1.0
    return resample_spawn(copied, seed, x_bounds=(x, x), y_bounds=spec.SPAWN_Y,
                          yaw_bounds=spec.SPAWN_YAW_RADIANS)


def configure_air_case(case, *, length, width, height, seed):
    from copy import deepcopy
    from dataclasses import asdict
    from src.tasks.velocity.v07 import terrain, spec
    terrain.configure_air_platform(length=length, width=width, height=height)
    tile = terrain.build_terrain_specs()[0]
    copied = deepcopy(case)
    copied['geometry'] = asdict(tile)
    copied['spawn']['frame_origin_world'] = list(tile.frame_origin_world)
    return resample_spawn(copied, seed, x_bounds=(1., 1.), y_bounds=spec.SPAWN_Y,
                          yaw_bounds=spec.SPAWN_YAW_RADIANS)


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


class KeyboardControl:
    """Viewer thread enqueues keys; simulation thread alone changes the command."""
    def __init__(self, cruise):
        from queue import SimpleQueue
        self.cruise = cruise
        self.walking = False
        self.events = SimpleQueue()

    def on_key(self, keycode):
        if keycode in (ord('W'), ord('w'), ord('S'), ord('s')):
            self.events.put(keycode)

    def read(self):
        from queue import Empty
        while True:
            try:
                key = self.events.get_nowait()
            except Empty:
                break
            self.walking = key in (ord('W'), ord('w'))
        return ('WALK', self.cruise) if self.walking else ('STOP', 0.0)


def flat_sequence(step, cruise, walk_seconds, stop_seconds, dt=0.02):
    walk_steps = max(1, round(walk_seconds / dt))
    stop_steps = max(1, round(stop_seconds / dt))
    if step < walk_steps:
        return 'WALK', cruise
    if step < walk_steps + stop_steps:
        return 'STOP', 0.0
    return 'WALK_AGAIN', cruise


def enable_flat_sequence(env):
    """Playback-only continuous flat sequence; keep physical failure resets."""
    from dataclasses import replace
    from types import MethodType
    import torch
    from src.tasks.velocity.v07.env import V07Env

    if any(case['task'] != 'flat_walk' for case in env.v09_cases):
        raise ValueError('Flat sequence requires flat_walk cases')

    def task_state(self):
        state = V07Env._task_state(self)
        self._flat_outcome = {}
        self._physical_failure = state.failure.clone()
        return replace(state, success=torch.zeros_like(state.success),
                       timeout=torch.zeros_like(state.timeout),
                       reason=torch.where(state.failure, state.reason, torch.zeros_like(state.reason)))

    env._task_state = MethodType(task_state, env)
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
    parser.add_argument('--scene', choices=['stairs', 'flat'], default='stairs')
    parser.add_argument('--keyboard', action='store_true', help='Flat manual control: W walk at --vx, S stop; initially stopped')
    parser.add_argument('--air-platform', action='store_true', help='Elevate the flat tile as an isolated platform')
    parser.add_argument('--platform-height', type=float, default=2.0)
    parser.add_argument('--continuous-walk', action='store_true', help='Flat constant forward command, no timed stop')
    parser.add_argument('--walk-seconds', type=float, default=8.0)
    parser.add_argument('--stop-seconds', type=float, default=5.0)
    parser.add_argument('--case-id', type=int, default=0)
    parser.add_argument('--start-on-platform', action='store_true', help='Start 1m inside the stair landing at its actual height')
    parser.add_argument('--stair-steps', type=int, default=None, help='Temporary playback stair count')
    parser.add_argument('--platform-width', type=float, default=None, help='Temporary landing width in metres; stair width unchanged')
    parser.add_argument('--platform-length', type=float, default=None, help='Temporary landing length in metres (>=2)')
    parser.add_argument('--continue-after-success', action='store_true', help='Keep walking after top success; retain failure resets, ignore post-top timeout')
    parser.add_argument('--vx', type=float, default=0.5)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--wz', type=float, default=0.0)
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--spawn-seed', type=int, default=None, help='Playback-only seed for a new spawn pose within original bounds')
    parser.add_argument('--num-envs', type=int, choices=[1], default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--viewer', choices=['native'], default='native')
    parser.add_argument('--duration', type=float, default=None, help='Playback seconds; otherwise until window closes')
    parser.add_argument('--headless-steps', type=int, default=0, help='Finite physics/scan smoke without a window')
    args = parser.parse_args(argv)
    if args.start_on_platform and args.scene != 'stairs':
        parser.error('start-on-platform requires --scene stairs')
    if (args.air_platform or args.continuous_walk) and args.scene != 'flat':
        parser.error('air-platform and continuous-walk require --scene flat')
    if args.keyboard and args.continuous_walk:
        parser.error('keyboard and continuous-walk are mutually exclusive')
    if args.air_platform and (not math.isfinite(args.platform_height) or args.platform_height <= 0 or args.platform_length is None or args.platform_width is None or args.platform_length < 3):
        parser.error('air-platform requires explicit length >=3, width and positive height')
    if args.keyboard and args.scene != 'flat':
        parser.error('--keyboard requires --scene flat')
    if not all(math.isfinite(t) and t > 0 for t in (args.walk_seconds, args.stop_seconds)):
        parser.error('walk-seconds and stop-seconds must be finite and positive')
    if args.scene == 'flat' and (args.continue_after_success or args.stair_steps is not None or (not args.air_platform and any(v is not None for v in (args.platform_width, args.platform_length)))):
        parser.error('Stair geometry and continue-after-success options do not apply to flat playback')
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
    case = select_case(json.loads(args.manifest.read_text()), 'FLAT' if args.scene == 'flat' else args.lesson,
                       args.vx, args.case_id, scene=args.scene)
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

    if args.spawn_seed is not None and not args.air_platform:
        from src.tasks.velocity.v07 import spec as original_spec
        case = resample_spawn(case, args.spawn_seed, x_bounds=original_spec.SPAWN_X,
                              y_bounds=original_spec.SPAWN_Y, yaw_bounds=original_spec.SPAWN_YAW_RADIANS)
        print(f'Playback spawn seed={args.spawn_seed}: {case["spawn"]["local_position_xy"]}, '
              f'yaw={case["spawn"]["local_yaw_radians"]}; diagnostic pose override only', flush=True)
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
    if args.air_platform:
        case = configure_air_case(case, length=args.platform_length, width=args.platform_width,
                                  height=args.platform_height, seed=args.spawn_seed if args.spawn_seed is not None else args.seed)
        print(f'Air platform: width={args.platform_width:g}m length={args.platform_length:g}m '
              f'height={args.platform_height:g}m; spawn x=1m, diagnostic only', flush=True)
    if args.scene == 'stairs':
        played_tile, landing_width = configure_terrain(case, steps=args.stair_steps, width=args.platform_width, length=args.platform_length)
        if args.start_on_platform:
            case = platform_start_case(case, played_tile, args.spawn_seed if args.spawn_seed is not None else args.seed)
            print(f'Playback start on landing: local position={case["spawn"]["local_position"]}', flush=True)
        print(f'Playback geometry: stairs={played_tile.N}, h={played_tile.h:g}m, depth={played_tile.step_depth:g}m, '
              f'stair width={played_tile.width:g}m, landing length={played_tile.platform_length:g}m, '
              f'landing width={landing_width:g}m; demo only, not frozen-case evaluation', flush=True)
    elif args.continuous_walk:
        print(f'Continuous flat WALK: vx={args.vx:g}m/s; physical failures still reset', flush=True)
    elif args.keyboard:
        print(f'Flat keyboard control: initially STOP; W=walk at {args.vx:g}m/s, S=stop; click viewer to focus. '
              'Automatic walk/stop timing is disabled.', flush=True)
    else:
        print(f'Flat diagnostic sequence: WALK {args.walk_seconds:g}s at {args.vx:g}m/s -> '
              f'STOP {args.stop_seconds:g}s at 0m/s -> WALK_AGAIN at {args.vx:g}m/s until closed; '
              'no scheduled resets, physical failures still reset', flush=True)
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
        if args.start_on_platform:
            env.robot.data.default_root_state[:, 2] = case['spawn']['local_position'][2]
        if args.scene == 'flat':
            enable_flat_sequence(env)
        elif args.continue_after_success:
            enable_continue_after_success(env)
        obs, _ = env.reset()
        phase_text = ''
        keyboard = KeyboardControl(args.vx) if args.keyboard else None
        def advance():
            nonlocal obs, phase_text
            if args.scene == 'flat':
                if keyboard is not None:
                    phase, command = keyboard.read()
                elif args.continuous_walk:
                    phase, command = 'WALK', args.vx
                else:
                    phase, command = flat_sequence(int(env.episode_length_buf[0]), args.vx,
                                                   args.walk_seconds, args.stop_seconds, CONTROL_DT)
                if phase != phase_text:
                    print(f'FLAT PHASE {phase}: vx={command:g}m/s, episode_step={int(env.episode_length_buf[0])}', flush=True)
                phase_text = phase
                env.command[:, 0] = command
                env.command[:, 1:] = 0
                # Refresh current proprio command without adding history/map samples.
                obs = env._observations(update=False)
            obs, _, _, _, _ = env.step(model.act(obs, deterministic=True))

        with torch.inference_mode():
            if args.headless_steps:
                scene = mujoco.MjvScene(env.sim.mj_model, maxgeom=4096)
                for _ in range(args.headless_steps):
                    advance()
                    draw_scan(scene, env)
                assert torch.isfinite(env.sim.data.qpos).all()
                print(f'PASS: {args.headless_steps} policy steps, finite qpos, {scene.ngeom} scan decorations', flush=True)
                return
            import mujoco.viewer
            data = mujoco.MjData(env.sim.mj_model)
            with mujoco.viewer.launch_passive(env.sim.mj_model, data,
                                              key_callback=keyboard.on_key if keyboard is not None else None) as viewer:
                viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 120., -20., 3.8
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  scan_text.replace(' | ', '\n'),
                                  'Cyan: valid / orange: clipped / red: invalid\nYellow: boundary + forward'))
                limit = None if args.duration is None else max(1, round(args.duration / CONTROL_DT))
                steps = 0
                while viewer.is_running() and (limit is None or steps < limit):
                    start = time.perf_counter()
                    advance()
                    with viewer.lock():
                        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                        mujoco.mj_forward(env.sim.mj_model, data)
                        viewer.cam.lookat[:] = env.robot.data.root_link_pos_w[0].cpu().numpy()
                        draw_scan(viewer.user_scn, env)
                    if args.scene == 'flat':
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                          scan_text.replace(' | ', '\n'),
                                          f'{phase_text} | vx={float(env.command[0, 0]):g} m/s'
                                          + (' | W: walk / S: stop' if keyboard is not None else '')))
                    viewer.sync()
                    steps += 1
                    time.sleep(max(0., CONTROL_DT - (time.perf_counter() - start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
