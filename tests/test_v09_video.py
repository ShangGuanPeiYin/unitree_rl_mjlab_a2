"""Synthetic pre-reset video evidence tests; no renderer or physics is created."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import video as v
from src.tasks.velocity.v09.cases import build_s1_manifest, sha256


class Env:
    num_envs = 1
    def __init__(self, case, *, physical_failure=True):
        self.case, self.physical_failure = case, physical_failure
        self.episode_length_buf = torch.tensor([0])
        self.command = self.sample_command = torch.tensor([[case['speed'], 0., 0.]])
        self.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(1, 19), qvel=torch.zeros(1, 18)))
        self.origins = torch.tensor([case['geometry']['frame_origin_world']])
        self.tile_ids = torch.tensor([0])
        root = self.origins + torch.tensor([[*case['spawn']['local_position_xy'], .4]])
        self.robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                                                          root_link_pos_w=root))
        self.reward_calls, self.closed = 0, False
    def _reward(self, state, previous):
        self.reward_calls += 1
        return torch.tensor([42.])
    def observation_pair(self):
        obs = {'proprio': torch.zeros(1, 45)}
        return obs, obs
    def step(self, action):
        self.episode_length_buf += 1
        step = int(self.episode_length_buf[0])
        self.sim.data.qpos[:] = step
        self.sim.data.qvel[:] = step / 10
        self.robot.data.root_link_pos_w[0, 0] = self.origins[0, 0] + step
        done = step == 2
        state = SimpleNamespace(success=torch.tensor([done and not self.physical_failure]),
            failure=torch.tensor([done and self.physical_failure]), timeout=torch.tensor([False]),
            pitch=torch.tensor([.1]), local=torch.tensor([[float(step), 0., .4]]))
        reward = self._reward(state, torch.zeros(1, 12))
        assert reward.item() == 42
        records = [dict(env_id=0, success=not self.physical_failure, failure=self.physical_failure,
                        failure_masks={'body': self.physical_failure}, terminal_step=step)] if done else []
        if done:
            self.sim.data.qpos[:] = 999.  # proves a post-reset terminal image is wrong
            self.episode_length_buf[:] = 0
        return {}, reward, state.failure, state.timeout, {'episodes': records}
    def close(self):
        self.closed = True


class Renderer:
    def __init__(self, env, *, width=640, height=480):
        self.env, self.width, self.height = env, width, height
        self.captured = []
    def capture(self, overlay):
        qpos = float(self.env.sim.data.qpos[0, 0])
        self.captured.append((qpos, dict(overlay)))
        return np.full((self.height, self.width, 3), int(qpos) % 255, dtype=np.uint8)
    def close(self):
        pass


class Writer:
    def __init__(self, path, *, width=640, height=480, fps=25, fail=False):
        self.path, self.width, self.height, self.fps = Path(path), width, height, fps
        self.frames, self.finished, self.fail = [], False, fail
    def write(self, frame):
        if self.fail:
            raise RuntimeError('synthetic encoder failure')
        self.frames.append(int(frame[0, 0, 0]))
    def finish(self):
        self.path.write_bytes(bytes(self.frames))
        self.finished = True
        return {'streams': [{'width': self.width, 'height': self.height,
                              'r_frame_rate': f'{self.fps}/1', 'nb_read_frames': str(len(self.frames))}]}
    def abort(self):
        self.finished = True


class Ledger:
    def __init__(self):
        self.operations = {}
        self.invalidated = self.closed = False
        self.charged = 0
    def reserve(self, **kwargs):
        assert kwargs['course'] == 'S1_EVIDENCE' and kwargs['category'] == 'engineering'
        assert kwargs['num_envs'] == 1 and len(kwargs['case_sha256s']) == 1
        key = kwargs['operation_id']
        self.operations[key] = kwargs
        return key
    def commit(self, key, *, actual_control_steps=None, status='complete'):
        self.operations[key]['status'] = status
        self.charged += self.operations[key]['max_control_steps'] if status == 'unknown' else actual_control_steps
    def invalidate_obligation(self, *args, **kwargs):
        self.invalidated = True
    def close_obligation(self, *args, **kwargs):
        self.closed = True
    def checkpoint_metadata(self):
        return {'charged': self.charged}


def policy(obs):
    return torch.zeros(1, 12)


def test_terminal_frame_is_real_pre_reset_pose_and_reward_not_changed(tmp_path):
    case = build_s1_manifest('S1_DIAG')['cases'][0]
    env = Env(case)
    renderer, writer = Renderer(env), Writer(tmp_path / 'case.mp4')
    original = env._reward
    result = v.record_case(env, policy, case, policy_map='student', path=writer.path,
                           renderer=renderer, writer=writer, fps=25)
    assert env.sim.data.qpos[0, 0] == 999  # live env is already reset
    assert renderer.captured[-1][0] == 2  # video captured before that reset
    assert renderer.captured[-1][1]['frame_source'] == 'post_physics_pre_reset'
    assert renderer.captured[-1][1]['terminal']
    assert result['terminal_frame_pre_reset'] and result['control_steps'] == 2
    assert writer.frames == [0, 2]
    assert env._reward == original and env.reward_calls == 2
    assert result['episode']['failure']  # a failure is still a valid kept video


def test_q002_video_phase_index_and_overlay_real_actor_command():
    case = {'task': 'flat_stop', 'stop_mode': 'ramp_2s'}
    assert v.task_phase(case, 399) == 'CRUISE'
    assert v.task_phase(case, 499) == 'DECEL'
    assert v.task_phase(case, 500) == 'HOLD'
    assert v.task_phase({'task': 'stairs'}, 10, local_x=-.1) == 'APPROACH'
    assert v.task_phase({'task': 'stairs'}, 10, local_x=.1, success=True) == 'TOP_ARRIVED'


def test_initial_stair_overlay_uses_actual_negative_tile_local_position(tmp_path):
    case = next(case for case in build_s1_manifest('S1_DIAG')['cases'] if case['task'] == 'stairs')
    env = Env(case)
    renderer, writer = Renderer(env), Writer(tmp_path / 'initial_stairs.mp4')
    assert env.robot.data.root_link_pos_w[0, 0] - env.origins[0, 0] < 0
    v.record_case(env, policy, case, policy_map='student', path=writer.path,
                  renderer=renderer, writer=writer, fps=25)
    assert renderer.captured[0][1]['completed_steps'] == 0
    assert renderer.captured[0][1]['phase'] == 'APPROACH'


def test_all75_fixed_case0_failures_are_kept_and_charged(tmp_path):
    manifest = build_s1_manifest('S1_DIAG')
    ledger = Ledger()
    report = v.record_video_suite(ledger=ledger, manifest=manifest, policy=policy,
        policy_map='student', checkpoint_sha256='c' * 64, code_sha256='d' * 64,
        obligation_id='video_init', segment_id='video_init', output_dir=tmp_path / 'suite',
        env_factory=Env, renderer_factory=Renderer, writer_factory=Writer)
    assert report['engineering_valid'] and report['full75']
    assert len(report['video_paths']) == len(ledger.operations) == 75
    assert all(row['episode']['failure'] for row in report['cases'])
    assert all(Path(path).exists() for path in report['video_paths'])
    assert report['physical_control_samples'] == report['charged_control_samples'] == 150
    assert ledger.charged == 150 and ledger.closed and not ledger.invalidated
    payload = {key: value for key, value in report.items() if key != 'report_sha256'}
    assert report['report_sha256'] == sha256(payload)


def test_recording_error_invalidates_whole_suite_and_bills_reserved_upper(tmp_path):
    manifest, ledger = build_s1_manifest('S1_DIAG'), Ledger()
    envs = []
    def make_env(case):
        env = Env(case)
        envs.append(env)
        return env
    report = v.record_video_suite(ledger=ledger, manifest=manifest, policy=policy,
        policy_map='student', checkpoint_sha256='c' * 64, code_sha256='d' * 64,
        obligation_id='video_init', segment_id='broken', output_dir=tmp_path / 'broken',
        env_factory=make_env, renderer_factory=Renderer,
        writer_factory=lambda path, **kwargs: Writer(path, fail=True, **kwargs))
    assert not report['engineering_valid'] and not report['full75']
    assert len(ledger.operations) == 1 and ledger.invalidated and not ledger.closed
    assert ledger.charged == manifest['cases'][0]['hard_timeout_steps']
    assert report['charged_control_samples'] == ledger.charged
    assert envs[0].closed


def test_fixed_video_teacher_cannot_use_candidate_or_wrong_checkpoint(tmp_path):
    manifest = build_s1_manifest('S1_DIAG')
    with pytest.raises(ValueError):
        v.record_video_suite(ledger=Ledger(), manifest=manifest, policy=policy,
            policy_map='student', checkpoint_sha256='c' * 64, code_sha256='d' * 64,
            obligation_id='video_6400', segment_id='invalid', output_dir=tmp_path / 'invalid',
            env_factory=Env, renderer_factory=Renderer, writer_factory=Writer)


def test_summary_write_failure_keeps_obligation_open_and_costs_once(tmp_path, monkeypatch):
    manifest, ledger = build_s1_manifest('S1_DIAG'), Ledger()
    original = v._write_json
    def fail_summary(path, payload):
        if Path(path).name == 'video_summary.json':
            raise OSError('summary filesystem failure')
        return original(path, payload)
    monkeypatch.setattr(v, '_write_json', fail_summary)
    with pytest.raises(OSError):
        v.record_video_suite(ledger=ledger, manifest=manifest, policy=policy,
            policy_map='student', checkpoint_sha256='c' * 64, code_sha256='d' * 64,
            obligation_id='video_init', segment_id='summary_error', output_dir=tmp_path / 'summary_error',
            env_factory=Env, renderer_factory=Renderer, writer_factory=Writer)
    assert ledger.charged == 150 and len(ledger.operations) == 75
    assert ledger.invalidated and not ledger.closed


@pytest.mark.parametrize('component', ['renderer', 'env', 'writer'])
def test_cleanup_errors_invalidate_without_skipping_other_cleanup(tmp_path, component):
    manifest, ledger, envs = build_s1_manifest('S1_DIAG'), Ledger(), []
    class BadEnv(Env):
        def close(self):
            super().close()
            if component == 'env':
                raise RuntimeError('env close failed')
    class BadRenderer(Renderer):
        def close(self):
            if component == 'renderer':
                raise RuntimeError('renderer close failed')
    class BadWriter(Writer):
        def write(self, frame):
            if component == 'writer':
                raise RuntimeError('encoder write failed')
            return super().write(frame)
        def abort(self):
            if component == 'writer':
                raise RuntimeError('encoder abort failed')
            super().abort()
    def make_env(case):
        env = BadEnv(case)
        envs.append(env)
        return env
    report = v.record_video_suite(ledger=ledger, manifest=manifest, policy=policy,
        policy_map='student', checkpoint_sha256='c' * 64, code_sha256='d' * 64,
        obligation_id='video_init', segment_id='close_error', output_dir=tmp_path / 'close_error',
        env_factory=make_env, renderer_factory=BadRenderer, writer_factory=BadWriter)
    assert not report['engineering_valid'] and ledger.invalidated and not ledger.closed
    assert envs[0].closed
    assert len(ledger.operations) == 1
    assert ledger.charged == (manifest['cases'][0]['hard_timeout_steps'] if component == 'writer' else 2)


def test_bundled_ffmpeg_encodes_and_fully_decodes_when_PATH_tools_are_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(v.shutil, 'which', lambda command: None)
    writer = v.FFmpegWriter(tmp_path / 'bundled.mp4', width=640, height=480, fps=25)
    try:
        for value in (0, 50, 150, 255):
            writer.write(np.full((480, 640, 3), value, dtype=np.uint8))
        verification = writer.finish()
    finally:
        writer.abort()
    assert verification['backend'] == 'bundled_ffmpeg_decode'
    assert verification['streams'][0]['width'] == 640
    assert verification['streams'][0]['height'] == 480
    assert verification['streams'][0]['r_frame_rate'] == '25/1'
    assert int(verification['streams'][0]['nb_read_frames']) == 4
