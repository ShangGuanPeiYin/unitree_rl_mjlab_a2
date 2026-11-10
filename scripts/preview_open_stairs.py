"""Inspect the shared open-stair geometry. This does not run a policy or training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from src.terrains import OpenStairsTerrainCfg, build_open_stairs_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step-count', type=int, default=40)
    parser.add_argument('--step-height', type=float, default=0.20)
    parser.add_argument('--step-depth', type=float, default=0.20)
    parser.add_argument('--stair-width', type=float, default=0.80)
    parser.add_argument('--tread-thickness', type=float, default=0.01)
    parser.add_argument('--approach-length', type=float, default=2.0)
    parser.add_argument('--landing-length', type=float, default=2.0)
    parser.add_argument('--ground-width', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--robot', choices=('a2', 'none'), default='a2')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--render', action='store_true', help='Save an offscreen overview PNG.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    cfg = OpenStairsTerrainCfg(**{key: getattr(args, key) for key in (
        'step_count', 'step_height', 'step_depth', 'stair_width', 'tread_thickness',
        'approach_length', 'landing_length', 'ground_width')})
    spec, spawn = build_open_stairs_spec(cfg, seed=args.seed)
    if args.robot == 'a2':
        from src.assets.robots.unitree_a2.a2_constants import get_spec
        robot = get_spec()
        robot.body('base_link').pos = (*spawn[:2], 0.4)
        robot.body('base_link').quat = (1, 0, 0, 0)
        frame = spec.worldbody.add_frame()
        spec.attach(robot, prefix='robot/', frame=frame)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    model = spec.compile()
    data = mujoco.MjData(model)
    if args.robot == 'a2':
        for index in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name.endswith('_joint') and model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE:
                angle = 0.9 if 'thigh' in name else -1.8 if 'calf' in name else (
                    -0.1 if ('FL_' in name or 'RL_' in name) else 0.1)
                data.qpos[model.jnt_qposadr[index]] = angle
    mujoco.mj_forward(model, data)
    summary = {'terrain': asdict(cfg), 'seed': args.seed,
               'spawn_ground_xyz': spawn.tolist(), 'robot': args.robot,
               'robot_base_height': 0.4 if args.robot == 'a2' else None,
               'terrain_geometries': cfg.step_count + 2,
               'top_height': cfg.step_count * cfg.step_height,
               'policy_executed': False, 'physics_stepped': False,
               'preview': 'static initial pose; not a locomotion validation'}
    print(json.dumps(summary, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / 'scene.json').write_text(json.dumps(summary, indent=2) + '\n')
        from mjlab.utils.xml import fix_spec_xml, strip_buffer_textures
        export = spec.copy()
        strip_buffer_textures(export)
        (args.output_dir / 'scene.xml').write_text(fix_spec_xml(export.to_xml(), meshdir='assets'))
        for name, contents in export.assets.items():
            dest = args.output_dir / 'assets' / name.removeprefix('assets/')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(contents)
    if args.render and args.output_dir is None:
        parser.error('--render requires --output-dir')
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (cfg.step_count * cfg.step_depth / 2, 0, cfg.step_count * cfg.step_height / 2)
    camera.distance = 1.5 * max(cfg.size[0], cfg.step_count * cfg.step_height)
    camera.azimuth = 125
    camera.elevation = -20
    if args.render:
        import imageio.v3 as iio
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            renderer.update_scene(data, camera=camera)
            iio.imwrite(args.output_dir / 'overview.png', renderer.render())
    if not args.headless:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == '__main__':
    main()
