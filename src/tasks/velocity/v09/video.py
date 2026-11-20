"""Fixed-case evidence videos captured after physics and before automatic reset.

Rendering uses a separate MuJoCo MjData: it cannot advance or mutate the live
training state. Only the existing instance reward method is wrapped, and its
return value/side effects remain untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
import shutil
from pathlib import Path
import subprocess

import torch

from . import spec
from .cases import canonical_json, sha256, validate_manifest

VIDEO_MODELS = ('6400', 'init', 'adapt', 'final1', 'final2', 'final3')


def _write_json(path, payload):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(payload) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def task_phase(case, actor_index, *, local_x=None, success=False):
    if case['task'] == 'stairs':
        return 'TOP_ARRIVED' if success else 'APPROACH' if local_x is not None and local_x < 0 else 'CLIMB'
    if case['task'] == 'flat_walk' or actor_index < 400:
        return 'CRUISE'
    if case['stop_mode'] == 'ramp_2s' and actor_index < 500:
        return 'DECEL'
    return 'HOLD'


def tensor_digest(tensor):
    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def ffmpeg_executable():
    installed = shutil.which('ffmpeg')
    if installed:
        return installed
    import imageio_ffmpeg
    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(bundled).is_file() or not os.access(bundled, os.X_OK):
        raise RuntimeError('no existing local ffmpeg binary is available')
    return bundled


def verify_video(path, *, width, height, fps, frames):
    probe_executable = shutil.which('ffprobe')
    if probe_executable:
        probe = subprocess.run([
            probe_executable, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration',
            '-show_entries', 'format=duration', '-of', 'json', str(path)],
            check=True, capture_output=True, text=True, timeout=60)
        data = dict(json.loads(probe.stdout), backend='ffprobe')
    else:
        import imageio_ffmpeg
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        try:
            metadata = next(reader)
            decoded = 0
            for frame in reader:
                if len(frame) != width * height * 3:
                    raise RuntimeError('decoded video frame dimensions mismatch')
                decoded += 1
        finally:
            reader.close()
        actual_width, actual_height = metadata['size']
        rate = str(Fraction(float(metadata['fps'])).limit_denominator(100000))
        if '/' not in rate:
            rate += '/1'
        data = dict(backend='bundled_ffmpeg_decode', decoder_binary=ffmpeg_executable(),
                    decoder_metadata=metadata, streams=[dict(width=actual_width, height=actual_height,
                        r_frame_rate=rate, nb_read_frames=str(decoded))])
    streams = data.get('streams', [])
    if len(streams) != 1:
        raise RuntimeError('video verification did not find one stream')
    stream = streams[0]
    if (stream.get('width') != width or stream.get('height') != height
            or stream.get('r_frame_rate') != f'{fps}/1'
            or int(stream.get('nb_read_frames', -1)) != frames):
        raise RuntimeError('video verification dimensions/fps/frame count mismatch')
    return data


class MujocoSnapshotRenderer:
    """qpos/qvel copy -> mj_forward on independent MjData -> followed camera."""
    def __init__(self, env, *, width=640, height=480):
        import mujoco
        self.mujoco, self.env = mujoco, env
        self.width, self.height = width, height
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 2.5
        self.camera.azimuth = 135
        self.camera.elevation = -20

    def capture(self, overlay):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        qpos = self.env.sim.data.qpos[0].detach().cpu().numpy().copy()
        qvel = self.env.sim.data.qvel[0].detach().cpu().numpy().copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = overlay['simulation_time']
        self.mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.env.robot.data.root_link_pos_w[0].detach().cpu().numpy()
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render()).convert('RGB')
        draw = ImageDraw.Draw(frame)
        font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
        draw.rectangle((0, 0, self.width, 76), fill=(12, 15, 19))
        lines = [f"c_request=[{overlay['c_request']:.3f},0,0] m/s   c_exec=[{overlay['c_exec']:.3f},0,0] m/s",
                 f"phase={overlay['phase']}   pitch={overlay['pitch_degrees']:.2f} deg",
                 f"t={overlay['simulation_time']:.2f}s  step={overlay['completed_steps']}  terminal={overlay['terminal']}"]
        for line, text in enumerate(lines):
            draw.text((8, 4 + line * 23), text, fill=(245, 247, 250), font=font)
        return np.asarray(frame).copy()

    def close(self):
        self.renderer.close()


class FFmpegWriter:
    def __init__(self, path, *, width=640, height=480, fps=25):
        self.path, self.width, self.height, self.fps = Path(path), width, height, fps
        self.frames = 0
        self.stderr = self.path.with_suffix('.ffmpeg.log').open('xb')
        self.process = subprocess.Popen([
            ffmpeg_executable(), '-n', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-r', str(fps), '-movflags', '+faststart', str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr)
        self.finished = False

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3) or str(frame.dtype) != 'uint8':
            raise ValueError('renderer must produce exact uint8 RGB frame dimensions')
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def finish(self):
        if not self.finished:
            self.process.stdin.close()
            code = self.process.wait(timeout=60)
            self.stderr.close()
            self.finished = True
            if code:
                raise RuntimeError(f'ffmpeg failed with status{code}')
        return verify_video(self.path, width=self.width, height=self.height, fps=self.fps, frames=self.frames)

    def abort(self):
        if not self.finished:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.stderr.close()
            self.finished = True


def record_case(env, policy, case, *, policy_map, path, renderer, writer, fps=25):
    """First-terminal video, including the real same-frame terminal pose."""
    if env.num_envs != 1 or policy_map not in ('teacher', 'student') or fps != 25:
        raise ValueError('fixed evidence requires one environment and explicit grid at25fps')
    original_reward = env._reward
    audit = []
    terminal_record = None

    def capture(state=None):
        from src.tasks.velocity.v07.events import euler_heading
        completed = int(env.episode_length_buf[0])
        actor_index = max(0, completed - 1)
        terminal = state is not None and bool((state.success | state.failure | state.timeout)[0])
        if completed and completed % 2 and not terminal:
            return
        pitch = (float(state.pitch[0]) if state is not None
                 else float(euler_heading(env.robot.data.root_link_quat_w)[1][0]))
        local_x = (float(state.local[0, 0]) if state is not None else
                   float(env.robot.data.root_link_pos_w[0, 0] - env.origins[env.tile_ids[0], 0]))
        phase = task_phase(case, actor_index, local_x=local_x,
                           success=state is not None and bool(state.success[0]))
        executed = float(env.sample_command[0, 0]) if state is not None else float(env.command[0, 0])
        overlay = dict(completed_steps=completed, actor_index=actor_index,
                       c_request=case['speed'] if phase not in ('DECEL', 'HOLD') else 0.,
                       c_exec=executed, phase=phase, pitch_degrees=pitch * 180 / 3.141592653589793,
                       simulation_time=completed * .02, terminal=terminal,
                       qpos_sha256=tensor_digest(env.sim.data.qpos[0]),
                       qvel_sha256=tensor_digest(env.sim.data.qvel[0]),
                       frame_source='post_physics_pre_reset' if state is not None else 'initial_reset_state')
        frame = renderer.capture(overlay)
        writer.write(frame)
        audit.append(overlay)

    def capture_reward(state, previous_action):
        value = original_reward(state, previous_action)
        capture(state)
        return value

    env._reward = capture_reward
    try:
        capture()
        with torch.no_grad():
            for step in range(1, case['hard_timeout_steps'] + 1):
                teacher, student = env.observation_pair()
                actions = policy(teacher if policy_map == 'teacher' else student)
                _, _, _, _, info = env.step(actions)
                if info['episodes']:
                    terminal_record = dict(info['episodes'][0])
                    terminal_record['operation_course'] = 'S1_EVIDENCE'
                    break
        if terminal_record is None:
            raise RuntimeError('video case has no first-terminal at its full hard timeout')
        if not audit or not audit[-1]['terminal'] or audit[-1]['completed_steps'] != step:
            raise RuntimeError('missing true pre-reset terminal frame')
        if any(terminal_record.get('failure_masks', {}).get(name) for name in ('nonfinite', 'contact_overflow')):
            raise RuntimeError('engineering failure in evidence trajectory')
        probe = writer.finish()
        with Path(path).open('rb') as video_file:
            os.fsync(video_file.fileno())
        return dict(control_steps=step, physical_control_samples=step,
                    episode=terminal_record, frame_audit=audit, verification=probe,
                    ffprobe=probe if probe.get('backend') == 'ffprobe' else None,
                    video_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    terminal_qpos_sha256=audit[-1]['qpos_sha256'],
                    terminal_frame_pre_reset=True, frame_count=len(audit),
                    video_duration_note='25fps encoding includes initial and exact terminal snapshots')
    except BaseException:
        writer.abort()
        raise
    finally:
        env._reward = original_reward


def record_video_suite(*, ledger, manifest, policy, policy_map, checkpoint_sha256,
                       code_sha256, obligation_id, segment_id, output_dir,
                       env_factory, renderer_factory, writer_factory, width=640, height=480, fps=25,
                       source_snapshot=None):
    validate_manifest(manifest)
    if (manifest['suite'] != 'S1_DIAG' or obligation_id not in tuple('video_' + item for item in VIDEO_MODELS)
            or (width, height, fps) != (640, 480, 25)):
        raise ValueError('video evidence must use fixed DIAG75, known model ID,640x480/25fps')
    if obligation_id == 'video_6400' and (policy_map != 'teacher' or checkpoint_sha256 != spec.SOURCE_SHA256):
        raise ValueError('6400 fixed videos require the original teacher checkpoint/grid')
    if obligation_id != 'video_6400' and policy_map != 'student':
        raise ValueError('candidate fixed videos require the native student grid')
    selected = [case for case in manifest['cases'] if case['case_id'] == 0]
    if len(selected) != 75 or len({case['case_sha256'] for case in selected}) != 75:
        raise ValueError('exactly75 unique fixed case0 entries required')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if source_snapshot is not None:
        if source_snapshot.get('code_sha256') != code_sha256:
            raise ValueError('video source snapshot SHA mismatch')
        _write_json(output_dir / 'source_snapshot.json', source_snapshot)
    rows, error, charged = [], None, 0
    for index, case in enumerate(selected):
        operation = ledger.reserve(segment_id=segment_id, operation_id=f'video_{index:03d}',
            stage='S1', course='S1_EVIDENCE', category='engineering', num_envs=1,
            max_control_steps=case['hard_timeout_steps'], checkpoint_sha256=checkpoint_sha256,
            obligation_id=obligation_id, case_sha256=manifest['manifest_sha256'],
            case_sha256s=[case['case_sha256']], reason='fixed_DIAG_case0_first_terminal_video')
        env = renderer = writer = None
        settled = False
        path = output_dir / f'{index:03d}_{case["lesson"]}_{case["speed"]}_{case["stop_mode"]}.mp4'
        try:
            env = env_factory(case)
            renderer = renderer_factory(env, width=width, height=height)
            writer = writer_factory(path, width=width, height=height, fps=fps)
            result = record_case(env, policy, case, policy_map=policy_map, path=path,
                                 renderer=renderer, writer=writer, fps=fps)
            metadata = dict(result, case_sha256=case['case_sha256'], case_id=0,
                            manifest_sha256=manifest['manifest_sha256'], code_sha256=code_sha256,
                            checkpoint_sha256=checkpoint_sha256, sds_sha256=spec.SDS_SHA256,
                            source_sha256=spec.SOURCE_SHA256, video_path=str(path.resolve()),
                            width=width, height=height, fps=fps)
            _write_json(path.with_suffix('.json'), metadata)
            _write_json(path.with_suffix('.verification.json'), result['verification'])
            if result['ffprobe'] is not None:
                _write_json(path.with_suffix('.ffprobe.json'), result['ffprobe'])
            ledger.commit(operation, actual_control_steps=result['control_steps'])
            charged += result['control_steps']
            settled = True
            rows.append(metadata)
        except BaseException as failure:
            if not settled:
                ledger.commit(operation, status='unknown')
                charged += case['hard_timeout_steps']
            ledger.invalidate_obligation('S1', obligation_id, reason=f'video_recording_{type(failure).__name__}')
            error = dict(type=type(failure).__name__, message=str(failure), case_sha256=case['case_sha256'])
        finally:
            cleanup_errors = []
            cleanups = ([writer.abort] if writer is not None and not getattr(writer, 'finished', False) else [])
            cleanups += [resource.close for resource in (renderer, env) if resource is not None]
            for cleanup in cleanups:
                try:
                    cleanup()
                except BaseException as failure:
                    cleanup_errors.append(dict(type=type(failure).__name__, message=str(failure)))
            if cleanup_errors:
                ledger.invalidate_obligation('S1', obligation_id, reason='video_cleanup_failure')
                error = dict(type='VideoCleanupFailure', errors=cleanup_errors, case_sha256=case['case_sha256'])
        if error:
            break
    engineering_valid = error is None and len(rows) == 75
    report = dict(schema_version=11, sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
                  code_sha256=code_sha256, checkpoint_sha256=checkpoint_sha256,
                  candidate_sha256=checkpoint_sha256, manifest_sha256=manifest['manifest_sha256'],
                  obligation_id=obligation_id, expected_cases=75, actual_cases=len(rows), full75=len(rows) == 75,
                  engineering_valid=engineering_valid, error=error, width=width, height=height, fps=fps,
                  video_paths=[row['video_path'] for row in rows], cases=rows,
                  physical_control_samples=sum(row['physical_control_samples'] for row in rows),
                  charged_control_samples=charged,
                  ledger=ledger.checkpoint_metadata(), behavior_gate=False)
    report['report_sha256'] = sha256(report)
    # R-VID01: durable artifacts must exist before declaring the obligation
    # complete. Summary errors invalidate coverage but never reverse costs.
    try:
        _write_json(output_dir / 'video_summary.json', report)
    except BaseException:
        ledger.invalidate_obligation('S1', obligation_id, reason='video_summary_write_failure')
        raise
    if engineering_valid:
        ledger.close_obligation('S1', obligation_id, reason='all75_fixed_case0_videos_with_pre_reset_terminal_frames')
    return report
