"""Frozen S1 inputs, independent named streams, and no behavior-result access.

The freeze tool may materialize all suites. Training callers must request the
training view; final contents are exposed only to an explicit acceptance role.
The hash-based generator is versioned and independent of Python's random module.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from ..v07 import spec as original
from . import spec

GENERATOR = 'v09-s1-sha256-u53-v1'
GATE_SUITES = ('gate_A', 'gate_B')
FINAL_SUITES = ('final_1', 'final_2', 'final_3')
CALIBRATION_SUITE = 'train_calibration'
DIAGNOSTIC_SUITE = 'S1_DIAG'
STREAM_FIELDS = ('spawn_x', 'spawn_y', 'spawn_yaw', 'dynamics', 'noise', 'delay', 'force')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False)


def sha256(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def stream_id(*, role, episode_id, channel, stage='S1', suite='training'):
    """Derive a stream without slot number/N; global episode IDs never rewind."""
    _integer(episode_id, 'episode_id')
    if not role or not channel or not stage or not suite:
        raise ValueError('stream identity fields cannot be empty')
    return sha256([GENERATOR, spec.SDS_SHA256, spec.SEED, stage, suite, role,
                   episode_id, channel])


def _uniform(identity, low, high):
    # Take the most significant 53 bits: an exact, portable [0,1) double.
    fraction = (int(identity[:16], 16) >> 11) / float(1 << 53)
    return low + (high - low) * fraction


def training_rules():
    return dict(schema_version=spec.CHECKPOINT_SCHEMA, generator=GENERATOR,
                sds_sha256=spec.SDS_SHA256, recipe=spec.RECIPE,
                seed=spec.SEED, namespace='S1/training',
                roles={'flat_walk': .20, 'flat_stop': .20, 'H00_H15': .30,
                       'D25_D22_D20_L12_L20_L40': .30},
                uniform_within_family=True, speeds=list(spec.SPEEDS),
                stop_modes=['instant', 'ramp_2s'], stop_modes_equal=True,
                teacher_episode_probability={'S1.1': .50, 'S1.2': .10},
                stream_formula='sha256(generator,SDS,42,stage,suite,role,episode_id,channel)',
                episode_id='external monotonic per role; independent of slot and N')


def s1_groups():
    """75 immutable-in-manifest groups, using original v07 physical geometry."""
    result = []
    definitions = [('flat_walk', 'FLAT', 'none'),
                   ('flat_stop', 'FLAT', 'instant'),
                   ('flat_stop', 'FLAT', 'ramp_2s')]
    definitions += [('stairs', entry.lesson_id, 'none')
                    for entry in original.TERRAIN_TABLE if entry.N > 0]
    for task, lesson, stop_mode in definitions:
        terrain = original.TERRAIN_BY_LESSON[lesson]
        geometry = asdict(terrain)
        for speed in spec.SPEEDS:
            if task == 'flat_walk':
                timeout = 1001
            elif task == 'flat_stop':
                timeout = 1401 if stop_mode == 'instant' else 1501
            else:
                timeout = original.episode_step_limit(terrain.tile_id, speed)
            result.append(dict(stage='S1', task=task, lesson=lesson,
                stair_count=terrain.N, geometry=geometry, speed=speed,
                stop_mode=stop_mode, force_family='none', hard_timeout_steps=timeout,
                termination_contract=('v07_original_stairs' if terrain.N else
                                      'v09_flat_full_window_plus_one_guard_step')))
    assert len(result) == 75
    return result


def _case(group, suite, case_id):
    streams = {field: stream_id(role=sha256(group), episode_id=case_id,
                               channel=field, suite=suite) for field in STREAM_FIELDS}
    x = _uniform(streams['spawn_x'], *original.SPAWN_X)
    y = _uniform(streams['spawn_y'], *original.SPAWN_Y)
    yaw = _uniform(streams['spawn_yaw'], *original.SPAWN_YAW_RADIANS)
    origin = group['geometry']['frame_origin_world']
    # q0 and zero initial velocity retain the existing reset implementation.
    spawn = dict(local_position_xy=[x, y], local_yaw_radians=yaw,
                 frame_origin_world=origin,
                 world_position_xy=[origin[0] + x, origin[1] + y],
                 local_position=[x, y, .4],
                 world_position=[origin[0] + x, origin[1] + y, origin[2] + .4],
                 quaternion_wxyz=[math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
                 joint_position=list(original.A2_DEFAULT_JOINT_POS),
                 joint_velocity=[0.] * 12, base_linear_velocity=[0.] * 3,
                 base_angular_velocity=[0.] * 3,
                 height_contract='a2_constants.INIT_STATE.pos[2]=0.4; v07._reset_ids')
    item = dict(group, suite=suite, case_id=case_id, streams=streams, spawn=spawn,
                command=dict(vx=group['speed'], vy=0., wz=0.,
                    walk_seconds=(20. if group['task'] == 'flat_walk' else
                                  8. if group['task'] == 'flat_stop' else None),
                    ramp_steps=100 if group['stop_mode'] == 'ramp_2s' else 0,
                    hold_steps=1000 if group['task'] == 'flat_stop' else 0),
                dynamics='nominal_original_6400', observation_noise='none',
                action_delay_steps=0, proprio_delay_steps=0, map_delay_steps=0,
                force_events=[])
    item['case_key'] = [item[key] for key in ('stage', 'suite', 'task', 'stair_count',
                       'geometry', 'speed', 'stop_mode', 'force_family', 'case_id')]
    item['case_sha256'] = sha256(item)
    return item


def build_s1_manifest(suite, *, calibration_cases_per_group=None):
    """Freeze complete inputs, never teacher outputs or final behavior results.

    Calibration size is an execution decision and has no invented default; its
    actual physical sampling is separately capped at one million controls.
    """
    if suite in GATE_SUITES or suite == DIAGNOSTIC_SUITE:
        count = 20
    elif suite in FINAL_SUITES:
        count = 100
    elif suite == CALIBRATION_SUITE:
        count = _integer(calibration_cases_per_group, 'calibration_cases_per_group', 1)
    else:
        raise ValueError('unknown S1 suite')
    if suite != CALIBRATION_SUITE and calibration_cases_per_group is not None:
        raise ValueError('calibration size only applies to training calibration')
    result = dict(schema_version=spec.CHECKPOINT_SCHEMA, generator=GENERATOR,
                  sds_sha256=spec.SDS_SHA256, source_sha256=spec.SOURCE_SHA256,
                  recipe=spec.RECIPE, stage='S1', suite=suite,
                  access='acceptance_only' if suite in FINAL_SUITES else 'training_or_gate',
                  sealing=dict(final_input_list='visible_to_freeze_and_acceptance_tools',
                      teacher_behavior_cache='sealed_from_training_and_model_selection',
                      training_reader='train_calibration_only'),
                  cases_per_group=count, training_rules=training_rules(),
                  cases=[_case(group, suite, index) for group in s1_groups()
                         for index in range(count)])
    result['manifest_sha256'] = sha256(result)
    return result


def validate_manifest(manifest):
    expected = manifest.get('manifest_sha256')
    payload = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if expected != sha256(payload):
        raise ValueError('manifest SHA256 mismatch')
    regenerated = build_s1_manifest(manifest['suite'], calibration_cases_per_group=(
        manifest['cases_per_group'] if manifest['suite'] == CALIBRATION_SUITE else None))
    if manifest != json.loads(canonical_json(regenerated)):
        # Tuple/list representations from in-memory geometry are JSON-equivalent.
        if canonical_json(manifest) != canonical_json(regenerated):
            raise ValueError('manifest differs from frozen generation algorithm')
    return True


def load_manifest(path, *, purpose):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    if purpose not in ('training', 'gate', 'acceptance', 'freeze_audit', 'diagnostic', 'evidence'):
        raise ValueError('explicit manifest reader purpose is required')
    suite = manifest['suite']
    if suite in FINAL_SUITES and purpose not in ('acceptance', 'freeze_audit'):
        raise PermissionError('final suite sealed from training/model selection')
    if purpose in ('diagnostic', 'evidence') and suite != DIAGNOSTIC_SUITE:
        raise PermissionError('diagnostic/video readers may only open S1_DIAG')
    if suite == DIAGNOSTIC_SUITE and purpose not in ('diagnostic', 'evidence', 'freeze_audit'):
        raise PermissionError('S1_DIAG is an independent non-gate diagnostic stream')
    if purpose == 'training' and suite != CALIBRATION_SUITE:
        raise PermissionError('training cannot read gate or final case contents')
    return manifest


def freeze_manifest(path, manifest):
    """Create once; refuse replacement, including a differently generated list."""
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(canonical_json(manifest) + '\n')
    return manifest['manifest_sha256']

TRAINING_ROLES = ('flat_walk', 'flat_stop', 'H00_H15', 'D25_D22_D20_L12_L20_L40')


def allocate_training_roles(num_envs):
    """Largest remainder with stable declared role order for integer ties."""
    _integer(num_envs, 'num_envs', 1)
    weights = (2, 2, 3, 3)
    counts = [num_envs * weight // 10 for weight in weights]
    remainders = [num_envs * weight % 10 for weight in weights]
    order = sorted(range(4), key=lambda i: (-remainders[i], i))
    for i in order[:num_envs - sum(counts)]:
        counts[i] += 1
    return dict(zip(TRAINING_ROLES, counts))


def training_case(role, episode_id):
    """Complete training input from a monotonic role episode, independent of N.

    Role assignment is an integer environment allocation; per-episode lesson,
    speed and stop mode use independent named uniform training streams.
    """
    if role not in TRAINING_ROLES:
        raise ValueError('unknown S1 training role')
    _integer(episode_id, 'episode_id')

    def pick(channel, values):
        identity = stream_id(role=role, episode_id=episode_id, channel=channel)
        return values[int(_uniform(identity, 0., float(len(values))))]

    speed = pick('command_speed', list(spec.SPEEDS))
    stop_mode = pick('stop_mode', ['instant', 'ramp_2s']) if role == 'flat_stop' else 'none'
    if role in ('flat_walk', 'flat_stop'):
        lesson, task = 'FLAT', role
    else:
        lessons = ([f'H{index:02d}' for index in range(16)] if role == 'H00_H15' else
                   ['D25', 'D22', 'D20', 'L12', 'L20', 'L40'])
        lesson, task = pick('lesson', lessons), 'stairs'
    group = next(row for row in s1_groups() if row['task'] == task
                 and row['lesson'] == lesson and row['speed'] == speed
                 and row['stop_mode'] == stop_mode)
    case = _case(group, 'training', episode_id)
    case.pop('case_sha256')
    case['training_role'] = role
    case['selection_streams'] = {channel: stream_id(role=role, episode_id=episode_id,
        channel=channel) for channel in ('command_speed', 'stop_mode', 'lesson')}
    case['case_sha256'] = sha256(case)
    return case
