"""Budgeted, independent S1 resource admission; no physics runs on import.

One measured iteration is 48 real control steps plus twenty actual Adam updates.
Q009 unloads training before independent 100-slot assessment and one-slot video
subprocesses. Each advances all5005 intervals. Parent restoration then executes
another full iteration. This is engineering evidence, not a behavior gate.
"""
from __future__ import annotations

import ctypes
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from uuid import UUID

import torch

from . import spec
from .cases import canonical_json, sha256, s1_groups, _case

WARMUPS = 5
MEASUREMENTS = 50
GPU_LIMIT_BYTES = 7168 * 1024**2
HOST_MIN_BYTES = 4 * 1024**3
EVAL_ENVS = 100
EVAL_STEPS = 5005
VIDEO_ENVS = 1
VIDEO_STEPS = 5005


def load_calibration(path, *, code_sha256):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    required = {'sds_sha256', 'source_sha256', 'manifest_sha256', 'code_sha256',
                'value_scale', 'value_samples', 'physical_control_samples', 'method'}
    if not required <= data.keys():
        raise ValueError('complete explicit frozen calibration JSON is required')
    if (data['sds_sha256'] != spec.SDS_SHA256 or data['source_sha256'] != spec.SOURCE_SHA256
            or data['code_sha256'] != code_sha256 or data['method'] != 'population_std_max_1'):
        raise ValueError('calibration identity/method differs from current execution')
    if (isinstance(data['value_scale'], bool) or not isinstance(data['value_scale'], (int, float))
            or not math.isfinite(data['value_scale']) or data['value_scale'] < 1):
        raise ValueError('explicit finite frozen value_scale >=1 is required')
    if (not isinstance(data['value_samples'], int) or isinstance(data['value_samples'], bool)
            or not isinstance(data['physical_control_samples'], int) or isinstance(data['physical_control_samples'], bool)
            or not 0 < data['value_samples'] <= data['physical_control_samples'] <= 1_000_000):
        raise ValueError('calibration must contain real samples within its one-million cap')
    digest = data['manifest_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('calibration manifest SHA is invalid')
    return dict(data, calibration_file_sha256=hashlib.sha256(raw).hexdigest(),
                calibration_path=str(path.resolve()))


def host_available_bytes():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('Linux MemAvailable is unavailable')


def canonical_gpu_uuid(value):
    """Compare the full UUID identity across CUDA objects and NVML GPU strings."""
    if value is None:
        raise ValueError('GPU UUID is missing')
    text = value.decode('ascii') if isinstance(value, bytes) else str(value)
    if text[:4].upper() == 'GPU-':
        text = text[4:]
    return str(UUID(text))


class WholeCardMonitor:
    """NVML + CUDA whole-card polling plus conservative allocator peak bound."""
    def __init__(self, *, device='cuda:0', physical_gpu_index=0, interval=.02):
        if not torch.cuda.is_available() or not str(device).startswith('cuda:'):
            raise RuntimeError('CUDA resource admission requires an explicit CUDA device')
        self.device = torch.device(device)
        self.index = self.device.index
        self.interval = float(interval)
        if not 0 < self.interval <= .1:
            raise ValueError('monitor interval must lie in (0,.1] seconds')
        self.nvml = ctypes.CDLL('libnvidia-ml.so.1')
        if self.nvml.nvmlInit_v2() != 0:
            raise RuntimeError('NVML initialization failed')
        self.handle = ctypes.c_void_p()
        if self.nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(physical_gpu_index), ctypes.byref(self.handle)):
            raise RuntimeError('NVML physical device lookup failed')
        uuid = ctypes.create_string_buffer(96)
        if self.nvml.nvmlDeviceGetUUID(self.handle, uuid, ctypes.c_uint(len(uuid))):
            raise RuntimeError('NVML UUID query failed')
        self.gpu_uuid = uuid.value.decode()
        cuda_uuid = getattr(torch.cuda.get_device_properties(self.device), 'uuid', None)
        try:
            same_gpu = canonical_gpu_uuid(cuda_uuid) == canonical_gpu_uuid(self.gpu_uuid)
        except (TypeError, ValueError) as error:
            raise RuntimeError('Invalid CUDA/NVML GPU UUID identity') from error
        if not same_gpu:
            raise RuntimeError('CUDA/NVML GPU UUID mismatch; supply matching physical_gpu_index')
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.samples = 0
        self.peak_card = 0
        self.min_host = 2**63 - 1
        self.error = None
        self.current_phase = 'training'
        self.phase_peak_card = self.phase_non_torch_peak = 0
        self.phases = []

    def observe(self):
        class Memory(ctypes.Structure):
            _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                        ('used', ctypes.c_ulonglong)]
        memory = Memory()
        with self.lock:
            # Requery under the same lock as begin_phase/reset so a delayed
            # polling sample cannot be assigned to a different layout phase.
            if self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)):
                raise RuntimeError('NVML whole-card memory query failed')
            free, total = torch.cuda.mem_get_info(self.device)
            used = max(int(memory.used), total - free)
            reserved = torch.cuda.memory_reserved(self.device)
            host = host_available_bytes()
            self.peak_card = max(self.peak_card, used)
            self.phase_peak_card = max(self.phase_peak_card, used)
            self.phase_non_torch_peak = max(self.phase_non_torch_peak, max(0, used - reserved))
            self.min_host = min(self.min_host, host)
            self.samples += 1

    def _finish_phase(self):
        allocator = torch.cuda.max_memory_reserved(self.device)
        self.phases.append(dict(name=self.current_phase,
            observed_whole_card_peak_bytes=self.phase_peak_card,
            parent_allocator_peak_reserved_bytes=allocator,
            non_parent_allocator_peak_bytes=self.phase_non_torch_peak,
            conservative_whole_card_peak_bytes=max(self.phase_peak_card, self.phase_non_torch_peak + allocator)))

    def begin_phase(self, name):
        if not isinstance(name, str) or not name or name == self.current_phase:
            raise ValueError('a distinct named resource layout phase is required')
        torch.cuda.synchronize(self.device)
        with self.lock:
            self.observe()
            self._finish_phase()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.current_phase = name
            self.phase_peak_card = self.phase_non_torch_peak = 0
            self.observe()

    def _poll(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.observe()
            except BaseException as error:
                self.error = str(error)
                self.stop_event.set()

    def start(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.observe()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.observe()
        with self.lock:
            self._finish_phase()
        peak_reserved = max(row['parent_allocator_peak_reserved_bytes'] for row in self.phases)
        # Q009 mutually exclusive layouts are never added together. NVML still
        # observes the entire card, including parent/child CUDA contexts.
        bound = max(self.peak_card, max(row['conservative_whole_card_peak_bytes'] for row in self.phases))
        return dict(gpu_uuid=self.gpu_uuid, interval_seconds=self.interval, samples=self.samples,
                    observed_whole_card_peak_bytes=self.peak_card,
                    torch_allocator_peak_reserved_bytes=peak_reserved,
                    conservative_whole_card_peak_bytes=bound,
                    host_min_available_bytes=self.min_host,
                    monitor_error=self.error, phases=self.phases,
                    monitoring_valid=self.error is None and not self.thread.is_alive())


def qualifies(result):
    memory = result.get('memory', {})
    phases = {row['name'] for row in memory.get('phases', [])}
    return (result.get('warmups_completed') == WARMUPS
            and result.get('measurements_completed') == MEASUREMENTS
            and result.get('restored_iteration_completed') is True
            and result.get('full_s1_2_workload_verified') is True
            and result.get('evaluation_control_steps') == EVAL_STEPS
            and result.get('evaluation_num_envs') == EVAL_ENVS
            and result.get('video_control_steps') == VIDEO_STEPS
            and result.get('post_evidence_iteration_completed') is True
            and {'training', 'full_s1_2', 'unloaded_child_evaluation', 'unloaded_child_video', 'resume_training'} <= phases
            and result.get('error') is None and memory.get('monitoring_valid') is True
            and memory.get('conservative_whole_card_peak_bytes', 2**63) <= GPU_LIMIT_BYTES
            and memory.get('host_min_available_bytes', 0) >= HOST_MIN_BYTES)


def role_allocation(num_envs):
    roles = ('flat_walk', 'flat_stop', 'H00_H15', 'D25_D22_D20_L12_L20_L40')
    weights = (.2, .2, .3, .3)
    counts = [int(num_envs * weight) for weight in weights]
    order = sorted(range(4), key=lambda i: (-(num_envs * weights[i] - counts[i]), i))
    for i in order[:num_envs - sum(counts)]:
        counts[i] += 1
    return [role for role, count in zip(roles, counts) for _ in range(count)]


def training_provider(namespace):
    groups = s1_groups()
    families = {
        'flat_walk': [g for g in groups if g['task'] == 'flat_walk'],
        'flat_stop': [g for g in groups if g['task'] == 'flat_stop'],
        'H00_H15': [g for g in groups if g['lesson'].startswith('H')],
        'D25_D22_D20_L12_L20_L40': [g for g in groups if g['task'] == 'stairs' and not g['lesson'].startswith('H')],
    }
    def provide(role, episode_id):
        available = families[role]
        group = available[episode_id % len(available)]
        return _case(group, namespace, episode_id)
    return provide


def validate_child_reservation(snapshot, operation_key, *, mode, checkpoint_sha256):
    """Read-only capability check: a child never reserves or commits budget."""
    if mode not in ('evaluation', 'video'):
        raise ValueError('unknown external resource worker mode')
    operation = snapshot.get('operations', {}).get(operation_key)
    expected_n = EVAL_ENVS if mode == 'evaluation' else VIDEO_ENVS
    expected_item = 'extra_002' if mode == 'evaluation' else 'extra_003'
    if (not operation or operation.get('settled') or operation.get('stage') != 'S1'
            or operation.get('course') != 'S1_RESOURCE' or operation.get('category') != 'engineering'
            or operation.get('num_envs') != expected_n or operation.get('max_control_steps') != 5005
            or operation.get('work_item_id') != expected_item
            or operation.get('checkpoint_sha256') != checkpoint_sha256
            or not operation.get('obligation_id', '').startswith('probe_')):
        raise ValueError('external worker lacks the exact unsettled parent reservation')
    return operation


def resource_assessment_cases(gate_cases, *, num_envs):
    chosen = [case for case in gate_cases if case['lesson'] == 'L40' and case['speed'] == .3]
    if len(chosen) != 20 or any(case['hard_timeout_steps'] != 5005 for case in chosen):
        raise ValueError('resource cases must come from20 gate_A original L40/.3 cases')
    if num_envs == 100:
        return [deepcopy(case) for _ in range(5) for case in chosen]
    if num_envs == 1:
        return [deepcopy(chosen[0])]
    raise ValueError('external resource assessment supports exactly100 or1 environments')


def collect_padded_assessment(env, policy, *, control_steps=5005):
    first = [None] * env.num_envs
    with torch.no_grad():
        for _ in range(control_steps):
            _, obs = env.observation_pair()
            _, _, _, _, info = env.step(policy(obs))
            for record in info['episodes']:
                if any(record.get('failure_masks', {}).get(key) for key in ('nonfinite', 'contact_overflow')):
                    raise RuntimeError('engineering anomaly in external resource assessment')
                if first[record['env_id']] is None:
                    first[record['env_id']] = deepcopy(record)
                    first[record['env_id']]['operation_course'] = 'S1_RESOURCE'
    if any(record is None for record in first):
        raise RuntimeError('external resource assessment lacks first-terminal evidence')
    return dict(control_steps=control_steps, num_envs=env.num_envs,
                physical_control_samples=env.num_envs * control_steps,
                first_terminal_records=first, padding_charged=True, behavior_gate=False)


def collect_padded_video(env, policy, case, path, *, renderer, writer, control_steps=5005):
    """Engineering video stress: record through resets while charging all padding."""
    from .video import tensor_digest, task_phase
    from src.tasks.velocity.v07.events import euler_heading
    if env.num_envs != 1:
        raise ValueError('resource video requires one environment')
    original_reward = env._reward
    global_step, frames, terminal_frames = [0], [], []

    def capture(state=None):
        episode_steps = int(env.episode_length_buf[0])
        terminal = state is not None and bool((state.success | state.failure | state.timeout)[0])
        if global_step[0] % 2 and not terminal and global_step[0] != control_steps:
            return
        local_x = (float(state.local[0, 0]) if state is not None else
                   float(env.robot.data.root_link_pos_w[0, 0] - env.origins[env.tile_ids[0], 0]))
        pitch = (float(state.pitch[0]) if state is not None else
                 float(euler_heading(env.robot.data.root_link_quat_w)[1][0]))
        overlay = dict(completed_steps=episode_steps, global_control_step=global_step[0],
            actor_index=max(0, episode_steps - 1), c_request=case['speed'],
            c_exec=float(env.sample_command[0, 0] if state is not None else env.command[0, 0]),
            phase=task_phase(case, max(0, episode_steps - 1), local_x=local_x,
                             success=state is not None and bool(state.success[0])),
            pitch_degrees=pitch * 180 / math.pi, simulation_time=global_step[0] * .02,
            terminal=terminal, qpos_sha256=tensor_digest(env.sim.data.qpos[0]),
            qvel_sha256=tensor_digest(env.sim.data.qvel[0]), frame_source='post_physics_pre_reset' if state is not None else 'initial_reset_state')
        writer.write(renderer.capture(overlay))
        frames.append(overlay)
        if terminal:
            terminal_frames.append(overlay)

    def reward_with_capture(state, previous):
        reward = original_reward(state, previous)
        capture(state)
        return reward
    env._reward = reward_with_capture
    try:
        capture()
        first = None
        with torch.no_grad():
            for step in range(1, control_steps + 1):
                global_step[0] = step
                _, obs = env.observation_pair()
                _, _, _, _, info = env.step(policy(obs))
                for record in info['episodes']:
                    if any(record.get('failure_masks', {}).get(key) for key in ('nonfinite', 'contact_overflow')):
                        raise RuntimeError('engineering anomaly in resource video')
                    if first is None:
                        first = deepcopy(record)
                        first['operation_course'] = 'S1_RESOURCE'
        if first is None or not terminal_frames:
            raise RuntimeError('resource video lacks pre-reset first-terminal evidence')
        verification = writer.finish()
        with Path(path).open('rb') as video_file:
            os.fsync(video_file.fileno())
        return dict(control_steps=control_steps, num_envs=1, physical_control_samples=control_steps,
            first_terminal_record=first, frame_audit=frames, video_verified=True,
            terminal_frames_pre_reset=True, verification=verification,
            video_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            padding_charged=True, behavior_gate=False)
    except BaseException:
        writer.abort()
        raise
    finally:
        env._reward = original_reward


class RealProbeCandidate:
    """Owns only disposable models/optimizer/environments, never a formal run."""
    def __init__(self, num_envs, *, device, calibration, output_dir, gate_cases, code_sha256,
                 ledger_run_id, ledger_path, gate_manifest_path):
        from .env import EpisodeIds
        from .model import Frozen6400Teacher, V09ActorCritic
        self.num_envs, self.device = num_envs, device
        self.calibration, self.output_dir = calibration, Path(output_dir)
        self.course = 'S1.1'
        self.code_sha256 = code_sha256
        self.ledger_run_id = ledger_run_id
        self.ledger_path, self.gate_manifest_path = Path(ledger_path), Path(gate_manifest_path)
        self.gate_cases = gate_cases
        self.roles = role_allocation(num_envs)
        self.ids = EpisodeIds()
        self.provider = training_provider(f'resource_{num_envs}')
        self.teacher = Frozen6400Teacher.from_6400_checkpoint(device=device)
        self.student = V09ActorCritic.migrate_6400_checkpoint(device=device)
        self.optimizer = torch.optim.Adam(self.student.optimizer_parameters(), lr=3e-5,
                                         betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.generator = torch.Generator(device=device).manual_seed(42)
        self.buffers = [None, None]


        self.env = self._new_env()
        self.eval_env = None
        self.iterations = 0
        self.frozen = self._frozen_digest()
        self.identity = deepcopy(self.student.transfer_metadata)

    def _new_env(self):
        from .env import V09S1Env
        cases = [self.provider(role, self.ids.next_by_role.get(role, 0)) for role in self.roles]
        return V09S1Env(cases, device=self.device, seed=42, roles=self.roles,
                        case_provider=self.provider, episode_ids=self.ids, course=self.course)

    def _saturation(self, rollout):
        flattened = {key: value.flatten(0, 1) for key, value in rollout['observations'].items()}
        count = rollout['teacher_mean'].shape[0] * self.num_envs
        totals, valid_points, saturated_valid_sum = {}, 0, 0.
        for start in range(0, count, 512):
            stop = min(start + 512, count)
            stats = self.student.normalizer_saturation({key: value[start:stop] for key, value in flattened.items()})
            for key, value in stats.items():
                if key == 'heightmap_valid_saturation':
                    points = int(flattened['heightmap'][start:stop, 1].bool().sum())
                    valid_points += points
                    if value is not None:
                        saturated_valid_sum += value * points
                else:
                    totals[key] = totals.get(key, 0.) + value * (stop - start) / count
        totals['heightmap_valid_saturation'] = saturated_valid_sum / valid_points if valid_points else None
        return dict(scope='complete_student_rollout', samples=count, fractions=totals)

    def _frozen_digest(self):
        from .model import state_sha256
        return {name: state_sha256({key: value for key, value in model.state_dict().items()
                                    if 'normalizer' in key or key == 'std'})
                for name, model in (('teacher', self.teacher), ('student', self.student))}

    def iteration(self, index):
        from .runtime import collect_rollout
        from .distillation import train_iteration
        slot = index % 2
        self.buffers[slot] = None
        rollout = collect_rollout(self.env, self.teacher, self.student)
        self.buffers[slot] = rollout
        for episode in rollout['episodes']:
            episode['operation_course'] = 'S1_RESOURCE'
            if any(episode.get('failure_masks', {}).get(key) for key in ('nonfinite', 'contact_overflow')):
                raise RuntimeError('engineering anomaly in resource rollout')
        result = train_iteration(self.student, self.optimizer, rollout['observations'],
                                 rollout['teacher_mean'], rollout['teacher_value'],
                                 value_scale=self.calibration['value_scale'],
                                 generator=self.generator, microbatch=512)
        if (result['adam_steps'] != 20 or result['samples'] != self.num_envs * 48
                or any(not math.isfinite(result[key]) for key in ('loss', 'action_loss', 'value_loss', 'gradient_norm'))):
            raise RuntimeError('resource iteration did not complete finite 48-step/20-Adam workload')
        if self._frozen_digest() != self.frozen:
            raise RuntimeError('resource workload changed frozen normalizer/std')
        trainable = {name: parameter for name, parameter in self.student.named_parameters() if parameter.requires_grad}
        allocated = {name for name, parameter in trainable.items()
                     if parameter.grad is not None and 'exp_avg' in self.optimizer.state.get(parameter, {})
                     and 'exp_avg_sq' in self.optimizer.state.get(parameter, {})}
        if allocated != set(trainable) or self.student.std in self.optimizer.state:
            raise RuntimeError('resource probe did not allocate complete trainable Adam/gradient state')
        self.iterations += 1
        return dict(result, control_steps=rollout['control_steps'],
                    resource_course=self.course, normalizer_saturation=self._saturation(rollout),
                    trainable_parameter_count=len(trainable), adam_allocated_parameter_count=len(allocated),
                    full_parameter_adam_names=sorted(allocated) if self.course == 'S1.2' else None,
                    metrics_storage_bytes=self.env.trace.storage_bytes)

    def save_and_restore(self):
        from .checkpoint import save_checkpoint, load_checkpoint
        from .model import V09ActorCritic, state_sha256
        from src.tasks.velocity.v07.continuation_v07 import tree_digest
        import numpy as np
        path = self.output_dir / 'restore_probe.pt'
        before = state_sha256(self.student.state_dict())
        optimizer_before = tree_digest(self.optimizer.state_dict())
        metadata = dict(stage='S1', course='S1.1', value_scale=self.calibration['value_scale'],
            manifest_sha256=self.calibration['manifest_sha256'], ledger_run_id=self.ledger_run_id,
            learning_samples=0, code_sha256=self.code_sha256,
            gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=0,
                            recoveries=0, next_gate_samples=0, passes=[]),
            role_state=dict(num_envs=self.num_envs, episode_counters=self.ids.state_dict(), assignments=self.roles),
            random_state=dict(python=random.getstate(), numpy=np.random.get_state()),
            probe=True, completed_resource_iterations=self.iterations,
            shuffle_generator=self.generator.get_state())
        checkpoint_sha = save_checkpoint(path, self.student, self.optimizer, metadata)
        # The new physical rollout starts a fresh episode; counters remain
        # monotonic and are not reconstructed from slot or update number.
        self.env.close()
        self.env = None
        # Keep the opposite rollout alive through recovery/evaluation so the
        # assessment covers real double-buffer memory, not an empty cache.
        del self.optimizer, self.student
        gc.collect()
        torch.cuda.empty_cache()
        payload = load_checkpoint(path, device='cpu')
        self.student = V09ActorCritic().to(self.device)
        self.student.load_state_dict(payload['model'], strict=True)
        self.student.set_s1_course('S1.1')
        self.optimizer = torch.optim.Adam(self.student.optimizer_parameters(), lr=3e-5,
                                         betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.generator.set_state(payload['metadata']['shuffle_generator'])
        # R-RES01: the prescribed recovery iteration covers S1.2's full
        # mean/critic gradients and new Adam state, on the same loaded Adam.
        self.student.set_s1_course('S1.2')
        self.course = 'S1.2'
        self.env = self._new_env()
        torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        random.setstate(payload['metadata']['random_state']['python'])
        np.random.set_state(payload['metadata']['random_state']['numpy'])
        if (state_sha256(self.student.state_dict()) != before or self._frozen_digest() != self.frozen
                or tree_digest(self.optimizer.state_dict()) != optimizer_before):
            raise RuntimeError('resource checkpoint restore changed model/frozen state')
        return dict(checkpoint_sha256=checkpoint_sha, source_model_state_sha256=before,
                    optimizer_sha256=optimizer_before, resume_mode='new_episode', optimizer_restored=True,
                    restored_course='S1.1', resource_course_after='S1.2', teacher_episode_probability=.1)

    def _parameter_identity(self):
        return dict(teacher=[id(p) for p in self.teacher.parameters()],
                    student=[id(p) for p in self.student.parameters()],
                    optimizer=id(self.optimizer),
                    optimizer_parameters=[[id(p) for p in group['params']] for group in self.optimizer.param_groups])

    def offload_for_evidence(self):
        """Q009: commit complete training first, then save and move parent to CPU."""
        from .checkpoint import save_checkpoint
        from .model import state_sha256
        from src.tasks.velocity.v07.continuation_v07 import tree_digest
        import numpy as np
        if self.course != 'S1.2':
            raise RuntimeError('full S1.2 workload must precede external evidence')
        self.offload_identity = self._parameter_identity()
        self.offload_fingerprints = dict(model=state_sha256(self.student.state_dict()),
            teacher=state_sha256(self.teacher.state_dict()), optimizer=tree_digest(self.optimizer.state_dict()))
        metadata = dict(stage='S1', course='S1.2', value_scale=self.calibration['value_scale'],
            manifest_sha256=self.calibration['manifest_sha256'], ledger_run_id=self.ledger_run_id,
            learning_samples=0, code_sha256=self.code_sha256,
            gate_state=dict(next_suite='gate_A', attempts=0, final_suites_used=0,
                            recoveries=0, next_gate_samples=0, passes=[]),
            role_state=dict(num_envs=self.num_envs, episode_counters=self.ids.state_dict(), assignments=self.roles),
            random_state=dict(python=random.getstate(), numpy=np.random.get_state()), probe=True,
            completed_resource_iterations=self.iterations, shuffle_generator=self.generator.get_state())
        path = self.output_dir / 'external_evidence_parent.pt'
        checkpoint_sha = save_checkpoint(path, self.student, self.optimizer, metadata)
        self.external_checkpoint = dict(path=str(path.resolve()), checkpoint_sha256=checkpoint_sha,
                                       fingerprints=self.offload_fingerprints)
        self.optimizer_devices = {p: {key: value.device for key, value in state.items() if torch.is_tensor(value)}
                                  for p, state in self.optimizer.state.items()}
        self.env.close()
        self.env = None
        self.buffers = [None, None]
        self.teacher.to('cpu')
        self.student.to('cpu')
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to('cpu')
        if self._parameter_identity() != self.offload_identity:
            raise RuntimeError('CPU offload changed Parameter/optimizer identity')
        gc.collect()
        torch.cuda.empty_cache()
        return self.external_checkpoint

    def external_assessment(self, mode, operation_key):
        """Parent is the sole payer; the subprocess only verifies its reservation."""
        if mode not in ('evaluation', 'video') or self.env is not None or any(self.buffers):
            raise RuntimeError('external evidence requires the unloaded parent layout')
        count = EVAL_ENVS if mode == 'evaluation' else VIDEO_ENVS
        directory = self.output_dir / ('external_' + mode)
        command = [sys.executable, str(spec.PROJECT_ROOT / 'scripts/a2_v09_resource_worker.py'),
            '--mode', mode, '--ledger', str(self.ledger_path), '--run-id', self.ledger_run_id,
            '--operation-key', operation_key, '--checkpoint', self.external_checkpoint['path'],
            '--checkpoint-sha256', self.external_checkpoint['checkpoint_sha256'],
            '--code-sha256', self.code_sha256, '--gate-manifest', str(self.gate_manifest_path),
            '--device', str(self.device), '--num-envs', str(count), '--control-steps', '5005',
            '--width', '640', '--height', '480', '--fps', '25', '--output-dir', str(directory), '--execute']
        process = subprocess.run(command, capture_output=True, text=True)
        self.output_dir.joinpath(mode + '_stdout.log').write_text(process.stdout)
        self.output_dir.joinpath(mode + '_stderr.log').write_text(process.stderr)
        if process.returncode:
            error_path = directory / 'worker_error.json'
            error = json.loads(error_path.read_text()) if error_path.exists() else dict(
                type='ExternalWorkerError', message=process.stderr[-4000:])
            if error.get('type') == 'OutOfMemoryError':
                raise torch.cuda.OutOfMemoryError(error['message'])
            if error.get('type') == 'MemoryError':
                raise MemoryError(error['message'])
            raise RuntimeError('external resource worker: ' + error.get('message', str(error)))
        result = json.loads((directory / 'worker_result.json').read_text())
        payload = {key: value for key, value in result.items() if key != 'report_sha256'}
        if (result.get('report_sha256') != sha256(payload)
                or result.get('checkpoint_sha256') != self.external_checkpoint['checkpoint_sha256']
                or result.get('operation_key') != operation_key
                or result.get('sds_sha256') != spec.SDS_SHA256 or result.get('code_sha256') != self.code_sha256
                or result.get('reservation_verified') is not True
                or result.get('num_envs') != count or result.get('control_steps') != 5005):
            raise RuntimeError('external resource worker result identity/workload mismatch')
        return result

    def restore_after_evidence(self):
        from .checkpoint import load_checkpoint
        from .model import state_sha256
        from src.tasks.velocity.v07.continuation_v07 import tree_digest
        import numpy as np
        info = self.external_checkpoint
        path = Path(info['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['checkpoint_sha256']:
            raise RuntimeError('offloaded parent checkpoint SHA changed')
        payload = load_checkpoint(path, device='cpu')
        actual = dict(model=state_sha256(self.student.state_dict()),
            teacher=state_sha256(self.teacher.state_dict()), optimizer=tree_digest(self.optimizer.state_dict()))
        if (actual != info['fingerprints'] or state_sha256(payload['model']) != actual['model']
                or tree_digest(payload['optimizer']) != actual['optimizer']
                or payload['metadata']['role_state']['episode_counters'] != self.ids.state_dict()
                or self._parameter_identity() != self.offload_identity):
            raise RuntimeError('offloaded parent model/Adam/role identity changed during evidence')
        self.teacher.to(self.device)
        self.student.to(self.device)
        for parameter, devices in self.optimizer_devices.items():
            for key, device in devices.items():
                self.optimizer.state[parameter][key] = self.optimizer.state[parameter][key].to(device)
        if self._parameter_identity() != self.offload_identity:
            raise RuntimeError('GPU restore changed Parameter/optimizer identity')
        self.generator.set_state(payload['metadata']['shuffle_generator'])
        self.env = self._new_env()
        torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        random.setstate(payload['metadata']['random_state']['python'])
        np.random.set_state(payload['metadata']['random_state']['numpy'])
        if (self._frozen_digest() != self.frozen
                or tree_digest(dict(torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if payload['cuda_rng'] else [],
                    shuffle=self.generator.get_state(), python=random.getstate(), numpy=np.random.get_state()))
                != tree_digest(dict(torch=payload['torch_rng'], cuda=payload['cuda_rng'],
                    shuffle=payload['metadata']['shuffle_generator'],
                    python=payload['metadata']['random_state']['python'],
                    numpy=payload['metadata']['random_state']['numpy']))
                or state_sha256(self.student.state_dict()) != actual['model']
                or tree_digest(self.optimizer.state_dict()) != actual['optimizer']
                or not torch.equal(self.generator.get_state(), payload['metadata']['shuffle_generator'])):
            raise RuntimeError('post-evidence restore failed state/RNG verification')
        return dict(checkpoint_sha256=info['checkpoint_sha256'], fingerprints_verified=True,
                    parameter_optimizer_identity_preserved=True, optimizer_devices_restored=True,
                    rng_restored=True, resume_mode='new_episode', episode_counters=self.ids.state_dict())

    def close(self):
        for name in ('env', 'eval_env'):
            env = getattr(self, name, None)
            if env is not None:
                env.close()
                setattr(self, name, None)
        self.buffers = [None, None]


def _write_new_json(path, result):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(result) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def run_resource_probe(*, ledger, segment_id, calibration, output_dir, code_sha256,
                       candidate_factory, monitor_factory, progress=None, source_snapshot=None):
    """Strict descending candidates; exceptions retain evidence and billed cost.

    Mock factories are supported for nonphysical CPU tests. A real caller must
    pass RealProbeCandidate/WholeCardMonitor factories with explicit inputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if source_snapshot is not None:
        if source_snapshot.get('code_sha256') != code_sha256:
            raise ValueError('resource source snapshot SHA mismatch')
        _write_new_json(output_dir / 'source_snapshot.json', source_snapshot)
    results, selected, cancelled, engineering_halt = [], None, False, False
    selected_evidence_sha = None
    for n in spec.ENV_CANDIDATES:
        directory = output_dir / f'n{n}'
        directory.mkdir()
        result = dict(num_envs=n, warmups_completed=0, measurements_completed=0,
                      restored_iteration_completed=False, evaluation_control_steps=0,
                      evaluation_num_envs=0, video_control_steps=0, post_evidence_iteration_completed=False,
                      iterations=[], error=None, workload_courses=['S1.1', 'S1.2'], ledger_course='S1_RESOURCE')
        worker = monitor = None
        operation = None
        attempted = False
        try:
            monitor = monitor_factory(n).start()
            # Creation allocates the complete physical scene and cache, but
            # does not advance a control interval. Reserve the first rollout
            # before construction, conservatively charging an uncertain failure.
            operation = ledger.reserve(segment_id=segment_id, operation_id=f'n{n}_rollout_001',
                stage='S1', course='S1_RESOURCE', category='engineering', num_envs=n,
                max_control_steps=48, checkpoint_sha256=spec.SOURCE_SHA256,
                obligation_id=f'probe_{n}', work_item_id='rollout_001', reason='resource_full_iteration')
            attempted = True
            worker = candidate_factory(n, directory)
            result['model_identity'] = deepcopy(getattr(worker, 'identity', {}))
            for i in range(WARMUPS + MEASUREMENTS):
                if i:
                    operation = ledger.reserve(segment_id=segment_id, operation_id=f'n{n}_rollout_{i + 1:03d}',
                        stage='S1', course='S1_RESOURCE', category='engineering', num_envs=n,
                        max_control_steps=48, checkpoint_sha256=spec.SOURCE_SHA256,
                        obligation_id=f'probe_{n}', work_item_id=f'rollout_{i + 1:03d}',
                        reason='resource_full_iteration')
                start = time.perf_counter()
                stats = worker.iteration(i)
                if stats.get('control_steps') != 48 or stats.get('adam_steps') != 20:
                    raise RuntimeError('incomplete physical rollout/Adam workload')
                ledger.commit(operation, actual_control_steps=48)
                operation = None
                monitor.observe()
                result['iterations'].append(dict(stats, seconds=time.perf_counter() - start,
                                                  kind='warmup' if i < WARMUPS else 'measurement'))
                result['warmups_completed' if i < WARMUPS else 'measurements_completed'] += 1
                if progress is not None:
                    progress(dict(num_envs=n, iteration=i + 1, full_iterations_required=55,
                                  kind='warmup' if i < WARMUPS else 'measurement',
                                  warmups_completed=result['warmups_completed'],
                                  measurements_completed=result['measurements_completed'],
                                  statistics=result['iterations'][-1]))
            monitor.begin_phase('full_s1_2')
            operation = ledger.reserve(segment_id=segment_id, operation_id=f'n{n}_extra_001',
                stage='S1', course='S1_RESOURCE', category='engineering', num_envs=n,
                max_control_steps=48, checkpoint_sha256=spec.SOURCE_SHA256,
                obligation_id=f'probe_{n}', work_item_id='extra_001', reason='save_restore_full_iteration')
            result['restore'] = worker.save_and_restore()
            stats = worker.iteration(WARMUPS + MEASUREMENTS)
            if stats.get('control_steps') != 48 or stats.get('adam_steps') != 20:
                raise RuntimeError('restored full iteration incomplete')
            if (stats.get('resource_course') != 'S1.2'
                    or not isinstance(stats.get('trainable_parameter_count'), int)
                    or stats['trainable_parameter_count'] <= 18
                    or stats.get('adam_allocated_parameter_count') != stats['trainable_parameter_count']
                    or len(stats.get('full_parameter_adam_names') or []) != stats['trainable_parameter_count']):
                raise RuntimeError('restored resource iteration lacks full S1.2 gradient/Adam state evidence')
            ledger.commit(operation, actual_control_steps=48)
            operation = None
            result['restored_iteration_completed'] = True
            result['restored_iteration_statistics'] = stats
            result['full_s1_2_workload_verified'] = stats.get('resource_course') == 'S1.2'
            monitor.observe()
            result['offloaded_parent'] = worker.offload_for_evidence()
            external_sha = result['offloaded_parent']['checkpoint_sha256']
            monitor.begin_phase('unloaded_child_evaluation')
            operation = ledger.reserve(segment_id=segment_id, operation_id=f'n{n}_extra_002',
                stage='S1', course='S1_RESOURCE', category='engineering', num_envs=EVAL_ENVS,
                max_control_steps=EVAL_STEPS, checkpoint_sha256=external_sha,
                obligation_id=f'probe_{n}', work_item_id='extra_002', reason='full_resource_evaluation_padding')
            result['evaluation'] = worker.external_assessment('evaluation', operation)
            if (result['evaluation'].get('control_steps') != EVAL_STEPS
                    or result['evaluation'].get('num_envs') != EVAL_ENVS):
                raise RuntimeError('resource assessment did not advance all5005 intervals')
            ledger.commit(operation, actual_control_steps=EVAL_STEPS)
            operation = None
            result['evaluation_control_steps'] = EVAL_STEPS
            result['evaluation_num_envs'] = EVAL_ENVS
            monitor.observe()
            monitor.begin_phase('unloaded_child_video')
            operation = ledger.reserve(segment_id=segment_id, operation_id=f'n{n}_extra_003',
                stage='S1', course='S1_RESOURCE', category='engineering', num_envs=VIDEO_ENVS,
                max_control_steps=VIDEO_STEPS, checkpoint_sha256=external_sha,
                obligation_id=f'probe_{n}', work_item_id='extra_003', reason='full_resource_video_padding')
            result['video'] = worker.external_assessment('video', operation)
            if (result['video'].get('control_steps') != VIDEO_STEPS
                    or result['video'].get('num_envs') != VIDEO_ENVS
                    or result['video'].get('video_verified') is not True
                    or result['video'].get('terminal_frames_pre_reset') is not True):
                raise RuntimeError('resource video workload/verification incomplete')
            ledger.commit(operation, actual_control_steps=VIDEO_STEPS)
            operation = None
            result['video_control_steps'] = VIDEO_STEPS
            monitor.observe()
            monitor.begin_phase('resume_training')
            operation = ledger.reserve(segment_id=segment_id, operation_id=f'n{n}_extra_004',
                stage='S1', course='S1_RESOURCE', category='engineering', num_envs=n,
                max_control_steps=48, checkpoint_sha256=external_sha,
                obligation_id=f'probe_{n}', work_item_id='extra_004', reason='post_external_evidence_restore_full_iteration')
            result['post_evidence_restore'] = worker.restore_after_evidence()
            stats = worker.iteration(WARMUPS + MEASUREMENTS + 1)
            if (stats.get('control_steps') != 48 or stats.get('adam_steps') != 20
                    or stats.get('resource_course') != 'S1.2'
                    or result['post_evidence_restore'].get('fingerprints_verified') is not True
                    or result['post_evidence_restore'].get('rng_restored') is not True):
                raise RuntimeError('post-evidence full restore iteration failed')
            ledger.commit(operation, actual_control_steps=48)
            operation = None
            result['post_evidence_iteration_completed'] = True
            result['post_evidence_iteration_statistics'] = stats
            monitor.observe()
        except BaseException as error:
            if operation is not None:
                ledger.commit(operation, status='unknown')
                operation = None
            result['error'] = dict(type=type(error).__name__, message=str(error))
            cancelled = isinstance(error, (KeyboardInterrupt, SystemExit))
        finally:
            if monitor is not None:
                try:
                    result['memory'] = monitor.finish()
                except BaseException as error:
                    result['memory'] = dict(monitoring_valid=False, monitor_error=str(error))
                    if result['error'] is None:
                        result['error'] = dict(type=type(error).__name__, message=str(error))
            if worker is not None:
                try:
                    worker.close()
                except BaseException as error:
                    result['error'] = dict(type=type(error).__name__, message=f'cleanup: {error}')
            del worker
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result['admitted'] = qualifies(result)
        error = result['error'] or {}
        memory = result.get('memory', {})
        oom = (error.get('type') in ('OutOfMemoryError', 'MemoryError')
               or 'out of memory' in error.get('message', '').lower())
        limit_exceeded = (not error and memory.get('monitoring_valid') is True and (
            memory.get('conservative_whole_card_peak_bytes', 0) > GPU_LIMIT_BYTES
            or memory.get('host_min_available_bytes', HOST_MIN_BYTES) < HOST_MIN_BYTES))
        if not result['admitted']:
            result['failure_kind'] = 'resource_limit' if oom or limit_exceeded else 'engineering_error'
            engineering_halt = result['failure_kind'] == 'engineering_error'
        result['ledger'] = ledger.checkpoint_metadata()
        _write_new_json(directory / 'result.json', result)
        evidence_sha = hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()
        results.append(result)
        if result['admitted']:
            selected = n
            selected_evidence_sha = evidence_sha
            break
        if attempted:
            ledger.fail_probe_obligation('S1', f'probe_{n}', reason='resource_candidate_not_admitted',
                                          evidence_sha256=evidence_sha)
        else:
            # Construction may have failed after reservation. The authoritative
            # ledger decides whether this candidate was actually claimed.
            duty = ledger.snapshot()['stages']['S1']['obligations'][f'probe_{n}']
            if duty['spent']:
                ledger.fail_probe_obligation('S1', f'probe_{n}', reason='resource_candidate_construction_failed',
                                              evidence_sha256=evidence_sha)
            else:
                ledger.skip_unattempted_probe('S1', f'probe_{n}', reason='monitor_unavailable_no_physics',
                                              evidence_sha256=evidence_sha)
        if cancelled or engineering_halt:
            break
    report = dict(schema_version=11, sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
                  recipe=spec.RECIPE, code_sha256=code_sha256,
                  calibration=calibration, selected_num_envs=selected, candidates=results,
                  admitted=selected is not None, cancelled=cancelled,
                  halted_for_engineering=engineering_halt, behavior_verified=False,
                  ledger=ledger.checkpoint_metadata())
    report['report_sha256'] = sha256(report)
    # A failed write leaves all paid work in the external ledger, while the
    # selected obligation remains open for explicit artifact reconciliation.
    # Never rerun those physical operations merely because a report is missing.
    _write_new_json(output_dir / 'resource_admission.json', report)
    if selected is not None:
        ledger.close_obligation('S1', f'probe_{selected}', reason='all55_iterations_restore_and5005_eval_artifacts_saved')
        for lower in spec.ENV_CANDIDATES[spec.ENV_CANDIDATES.index(selected) + 1:]:
            ledger.skip_unattempted_probe('S1', f'probe_{lower}',
                reason=f'larger_stable_candidate_{selected}', evidence_sha256=selected_evidence_sha)
        receipt = dict(schema_version=11, sds_sha256=spec.SDS_SHA256,
                       resource_report_sha256=report['report_sha256'], selected_num_envs=selected,
                       ledger=ledger.checkpoint_metadata())
        receipt['receipt_sha256'] = sha256(receipt)
        _write_new_json(output_dir / 'resource_settlement.json', receipt)
    return report
