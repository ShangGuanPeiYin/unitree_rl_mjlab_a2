"""Frozen S2.1 gate cases: A/B suites, 36 groups x 20 cases.

The two suites are independent S2 streams.  They are *not* the S2.0 diagnostic
matrix and are not read by training.  A/B share only the group definition; the
spawn/dynamics streams use different suite salts and therefore different cases.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from ..v07 import spec as original
from . import s2_spec, s2_terrain
from .cases import canonical_json, sha256

GATE_VERSION = 'v1'
GENERATOR = 'v09-s2-gate-sha256-u53-v1'
SUITES = ('S2_GATE_A', 'S2_GATE_B', 'S2_GATE_A_V2', 'S2_GATE_B_V2',
          'S2_GATE_A_V3', 'S2_GATE_B_V3')
COURSE_GROUP_COUNTS = {'S2.1': 36, 'S2.2': 33, 'S2.3': 51}
SUITE_ALIASES = {'gate_A': 'S2_GATE_A', 'gate_B': 'S2_GATE_B',
                 'S2_GATE_A': 'S2_GATE_A', 'S2_GATE_B': 'S2_GATE_B'}
CASES_PER_GROUP = 20
GROUP_COUNT = 36
COURSE = 'S2.1'
STREAM_FIELDS = ('spawn_x', 'spawn_y', 'spawn_yaw')


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _uniform(identity, low, high):
    fraction = (int(identity[:16], 16) >> 11) / float(1 << 53)
    return low + (high - low) * fraction


def stream_id(*, role, episode_id, channel, suite):
    if suite not in SUITES:
        raise ValueError('unknown S2 gate suite')
    if not role or not channel:
        raise ValueError('stream identity fields cannot be empty')
    return sha256([GENERATOR, s2_spec.SDS_SHA256, s2_spec.SEED,
                   s2_spec.STAGE, suite, role, int(episode_id), channel])


def flat_geometry():
    spec = s2_terrain.terrain_specs()[0]
    result = asdict(spec)
    result.update(kind='flat', platform_present=False,
                  platform_height=0.0, platform_front_local_x=None,
                  platform_width=8.0, tile_id=0)
    return result


def stairs_geometry(stair_count, platform_width):
    spec = next(row for row in s2_terrain.terrain_specs()
                if row.kind == 's2_stairs'
                and row.N == int(stair_count)
                and abs(row.platform_width - float(platform_width)) < 1e-12)
    result = asdict(spec)
    result.update(kind='s2_stairs', platform_present=True,
                  platform_height=float(spec.N) * float(spec.h),
                  platform_front_local_x=float(spec.N) * float(spec.step_depth))
    return result


def legacy_l40_geometry():
    spec = original.TERRAIN_BY_LESSON['L40']
    result = asdict(spec)
    result.update(kind='s2_stairs', platform_present=True,
                  platform_width=float(spec.width),
                  platform_height=float(spec.N) * float(spec.h),
                  platform_front_local_x=float(spec.N) * float(spec.step_depth),
                  terrain_key='legacy_L40')
    return result


def _timeout(*, task, speed, stair_count=0, geometry=None, post=None, stop_mode='none', ramp_seconds=0.0):
    if task == 'flat_walk':
        return s2_spec.FLAT_WALK_HARD_TIMEOUT_STEPS
    if task == 'flat_stop':
        return s2_spec.flat_stop_schedule(stop_mode)['hard_timeout_steps']
    if task == 'legacy_l40':
        return int(original.episode_step_limit(
            original.TERRAIN_BY_LESSON['L40'].tile_id, speed))
    if task == 'stairs_walk':
        return s2_spec.timeout_steps(
            task='stairs_walk', speed=speed, stair_count=stair_count,
            step_height=float(geometry['h']), step_depth=float(geometry['step_depth']),
            approach_length=s2_spec.T_APPROACH_MAX, post_distance=post)
    if task == 'platform_walk':
        return s2_spec.timeout_steps(task='platform_walk', speed=speed,
                                     post_distance=post)
    if task == 'stairs_stop':
        return s2_spec.timeout_steps(
            task='stairs_stop', speed=speed, stair_count=stair_count,
            step_height=float(geometry['h']), step_depth=float(geometry['step_depth']),
            approach_length=s2_spec.T_APPROACH_MAX, post_distance=post,
            ramp_seconds=float(ramp_seconds), hold_seconds=20.0)
    if task == 'platform_stop':
        return s2_spec.timeout_steps(task='platform_stop', speed=speed,
                                     post_distance=post, ramp_seconds=float(ramp_seconds),
                                     hold_seconds=20.0)
    raise ValueError(f'unsupported gate task {task}')


def gate_groups(course=COURSE):
    if course != COURSE:
        raise ValueError('this gate manifest version freezes S2.1 only')
    groups = []
    for speed in s2_spec.SPEEDS:
        for mode in ('walk', 'stop_instant', 'stop_ramp_2s'):
            task = 'flat_walk' if mode == 'walk' else 'flat_stop'
            stop_mode = 'none' if mode == 'walk' else (
                'instant' if mode == 'stop_instant' else 'ramp_2s')
            geometry = flat_geometry()
            groups.append(dict(
                group_id=f'flat/{mode}/{speed}', task=task,
                stair_count=0, geometry=geometry, speed=float(speed),
                stop_mode=stop_mode, force_family='none', post_distance=None,
                walk_seconds=(s2_spec.FLAT_WALK_SECONDS if task == 'flat_walk'
                              else 8.0),
                hard_timeout_steps=_timeout(task=task, speed=speed,
                                            stop_mode=stop_mode)))
    for levels in (10, 20):
        for width in s2_spec.PLATFORM_WIDTHS:
            for speed in s2_spec.SPEEDS:
                geometry = stairs_geometry(levels, width)
                groups.append(dict(
                    group_id=f'stairs/{levels}/{width}/{speed}',
                    task='stairs_walk', stair_count=int(levels),
                    geometry=geometry, speed=float(speed), stop_mode='none',
                    force_family='none', post_distance=5.0, walk_seconds=None,
                    hard_timeout_steps=_timeout(
                        task='stairs_walk', speed=speed, stair_count=levels,
                        geometry=geometry, post=5.0)))
    for height in (2.0, 4.0):
        for width in s2_spec.PLATFORM_WIDTHS:
            for speed in s2_spec.SPEEDS:
                geometry = stairs_geometry(int(round(height / 0.2)), width)
                groups.append(dict(
                    group_id=f'platform/{height}/{width}/{speed}',
                    task='platform_walk', stair_count=int(round(height / 0.2)),
                    geometry=geometry, speed=float(speed), stop_mode='none',
                    force_family='none', post_distance=5.0, walk_seconds=None,
                    hard_timeout_steps=_timeout(
                        task='platform_walk', speed=speed, post=5.0)))
    for speed in s2_spec.SPEEDS:
        geometry = legacy_l40_geometry()
        groups.append(dict(
            group_id=f'l40/{speed}', task='legacy_l40', stair_count=40,
            geometry=geometry, speed=float(speed), stop_mode='none',
            force_family='none', post_distance=None, walk_seconds=None,
            hard_timeout_steps=_timeout(task='legacy_l40', speed=speed)))
    if len(groups) != GROUP_COUNT:
        raise AssertionError(f'S2.1 gate must have {GROUP_COUNT} groups')
    return groups




def course_gate_groups(course):
    if course not in ('S2.2', 'S2.3'):
        raise ValueError('course gate groups are S2.2/S2.3')
    groups = []
    for speed in s2_spec.SPEEDS:
        for mode in ('walk', 'stop_instant', 'stop_ramp_2s'):
            task = 'flat_walk' if mode == 'walk' else 'flat_stop'
            stop_mode = 'none' if mode == 'walk' else (
                'instant' if mode == 'stop_instant' else 'ramp_2s')
            groups.append(dict(
                group_id=f'flat/{mode}/{speed}', task=task, stair_count=0,
                geometry=flat_geometry(), speed=float(speed),
                stop_mode=stop_mode, force_family='none', post_distance=None,
                walk_seconds=(s2_spec.FLAT_WALK_SECONDS if task == 'flat_walk' else 8.0),
                hard_timeout_steps=_timeout(task=task, speed=speed, stop_mode=stop_mode)))
    for width in s2_spec.PLATFORM_WIDTHS:
        for speed in s2_spec.SPEEDS:
            geometry = stairs_geometry(40, width)
            groups.append(dict(
                group_id=f'l40_walk/{width}/{speed}', task='stairs_walk',
                stair_count=40, geometry=geometry, speed=float(speed),
                stop_mode='none', force_family='none', post_distance=10.0,
                walk_seconds=None,
                hard_timeout_steps=_timeout(task='stairs_walk', speed=speed,
                                            stair_count=40, geometry=geometry,
                                            post=10.0)))
    for levels in (10, 20):
        for width in s2_spec.PLATFORM_WIDTHS:
            for speed in s2_spec.SPEEDS:
                geometry = stairs_geometry(levels, width)
                groups.append(dict(
                    group_id=f'review/{levels}/{width}/{speed}',
                    task='stairs_walk', stair_count=levels, geometry=geometry,
                    speed=float(speed), stop_mode='none', force_family='none',
                    post_distance=5.0, walk_seconds=None,
                    hard_timeout_steps=_timeout(task='stairs_walk', speed=speed,
                                                stair_count=levels,
                                                geometry=geometry, post=5.0)))
    if course == 'S2.2':
      for width in s2_spec.PLATFORM_WIDTHS:
        for speed in s2_spec.SPEEDS:
            geometry = stairs_geometry(40, width)
            groups.append(dict(
                group_id=f'platform/{width}/{speed}', task='platform_walk',
                stair_count=40, geometry=geometry, speed=float(speed),
                stop_mode='none', force_family='none', post_distance=10.0,
                walk_seconds=None,
                hard_timeout_steps=_timeout(task='platform_walk', speed=speed,
                                            post=10.0)))
    if course == 'S2.3':
        for width in s2_spec.PLATFORM_WIDTHS:
            for speed in s2_spec.SPEEDS:
                for mode in ('instant', 'ramp_2s'):
                    geometry = stairs_geometry(40, width)
                    groups.append(dict(
                        group_id=f'l40_stop/{width}/{speed}/{mode}',
                        task='stairs_stop', stair_count=40, geometry=geometry,
                        speed=float(speed), stop_mode=mode, force_family='none',
                        post_distance=10.0, walk_seconds=None,
                        hard_timeout_steps=_timeout(
                            task='stairs_stop', speed=speed, stair_count=40,
                            geometry=geometry, post=10.0,
                            ramp_seconds=(2.0 if mode == 'ramp_2s' else 0.0))))
        for width in s2_spec.PLATFORM_WIDTHS:
            for speed in s2_spec.SPEEDS:
                for mode in ('instant', 'ramp_2s'):
                    geometry = stairs_geometry(40, width)
                    groups.append(dict(
                        group_id=f'platform_stop/{width}/{speed}/{mode}',
                        task='platform_stop', stair_count=40,
                        geometry=geometry, speed=float(speed), stop_mode=mode,
                        force_family='none', post_distance=10.0,
                        walk_seconds=None,
                        hard_timeout_steps=_timeout(
                            task='platform_stop', speed=speed, post=10.0,
                            ramp_seconds=(2.0 if mode == 'ramp_2s' else 0.0))))
    expected = COURSE_GROUP_COUNTS[course]
    if len(groups) != expected:
        raise AssertionError(f'{course} gate has {len(groups)} groups, expected {expected}')
    return groups


def _group_role(group):
    return sha256({key: group[key] for key in
                   ('task', 'stair_count', 'geometry', 'speed', 'stop_mode',
                    'force_family')})


def _spawn(group, suite, case_id):
    streams = {field: stream_id(role=_group_role(group), episode_id=case_id,
                                channel=field, suite=suite)
               for field in STREAM_FIELDS}
    geometry = group['geometry']
    x = _uniform(streams['spawn_x'], *original.SPAWN_X)
    y = _uniform(streams['spawn_y'], *original.SPAWN_Y)
    yaw = _uniform(streams['spawn_yaw'], *original.SPAWN_YAW_RADIANS)
    if group['task'].startswith('platform_'):
        x = float(geometry['platform_front_local_x']) + s2_spec.PLATFORM_START_OFFSET
        z = float(geometry['platform_height']) + 0.4
        surface = 'top_platform'
    else:
        z = 0.4
        surface = 'ground'
    origin = geometry['frame_origin_world']
    return streams, dict(
        local_position_xy=[x, y], local_yaw_radians=yaw,
        frame_origin_world=list(origin),
        world_position_xy=[origin[0] + x, origin[1] + y],
        local_position=[x, y, z],
        world_position=[origin[0] + x, origin[1] + y, origin[2] + z],
        quaternion_wxyz=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        joint_position=list(original.A2_DEFAULT_JOINT_POS),
        joint_velocity=[0.0] * 12, base_linear_velocity=[0.0] * 3,
        base_angular_velocity=[0.0] * 3, spawn_surface=surface,
        height_contract=('top_surface_z + 0.4' if surface == 'top_platform'
                         else 'a2_constants.INIT_STATE.pos[2]=0.4; v07._reset_ids'))


def _case(group, suite, case_id):
    streams, spawn = _spawn(group, suite, case_id)
    command = dict(vx=group['speed'], vy=0.0, wz=0.0,
                   walk_seconds=group['walk_seconds'],
                   post_distance_m=group['post_distance'],
                   ramp_steps=(100 if group['stop_mode'] == 'ramp_2s' else 0),
                   hold_steps=(1000 if group['task'] == 'flat_stop' else 0))
    item = deepcopy(group)
    item.update(
        stage=s2_spec.STAGE, suite=suite, case_id=int(case_id),
        streams=streams, spawn=spawn, command=command,
        dynamics='nominal_original_6400', observation_noise='none',
        action_delay_steps=0, proprio_delay_steps=0, map_delay_steps=0,
        force_events=[], hard_timeout_steps=int(group['hard_timeout_steps']),
        termination_contract=('legacy_v07_l40' if group['task'] == 'legacy_l40'
                              else 's2_gate_fixed_task'),
        post_distance_m=group['post_distance'],
        case_key=[s2_spec.STAGE, suite, group['task'], group['stair_count'],
                  group['geometry'], group['speed'], group['stop_mode'],
                  group['force_family'], int(case_id)])
    item['case_sha256'] = sha256(item)
    return item


def build_gate_manifest(suite, *, course=COURSE):
    suite = SUITE_ALIASES.get(suite)
    if suite is None:
        raise ValueError('suite must be gate_A/gate_B or S2_GATE_A/S2_GATE_B')
    groups = gate_groups(course)
    cases = []
    group_rows = []
    for group in groups:
        members = [_case(group, suite, case_id)
                   for case_id in range(CASES_PER_GROUP)]
        cases.extend(members)
        group_rows.append(dict(
            group_id=group['group_id'], role_sha256=_group_role(group),
            task=group['task'], stair_count=group['stair_count'],
            geometry_key=group['geometry'].get('terrain_key'),
            platform_width=group['geometry'].get('platform_width'),
            speed=group['speed'], stop_mode=group['stop_mode'],
            post_distance_m=group['post_distance'],
            cases=len(members),
            hard_timeout_steps=int(group['hard_timeout_steps']),
            control_samples=len(members) * int(group['hard_timeout_steps'])))
    result = dict(
        schema_version=s2_spec.CHECKPOINT_SCHEMA, manifest_version=1,
        gate_version=GATE_VERSION, status='frozen', frozen=True,
        generator=GENERATOR, sds_sha256=s2_spec.SDS_SHA256, sds_version='v1.2',
        stage=s2_spec.STAGE, suite=suite, course=course,
        access='gate_acceptance_only',
        case_key_fields=list(s2_spec.CASE_KEY_FIELDS),
        cases_per_group=CASES_PER_GROUP, group_count=len(groups),
        groups=group_rows, cases=cases,
        budget=dict(
            control_samples=sum(row['control_samples'] for row in group_rows),
            per_group=sum(row['hard_timeout_steps'] * CASES_PER_GROUP
                          for row in group_rows),
            l40_milestone_groups=sum(row['task'] == 'legacy_l40' for row in group_rows),
            l40_milestone_samples=sum(row['control_samples'] for row in group_rows
                                      if row['task'] == 'legacy_l40')),
        note='Frozen S2.1 A/B gate inputs; training, S2.0 diagnostics and final '
             'acceptance use independent namespaces.')
    result['manifest_sha256'] = sha256(result)
    return result


def validate_gate_manifest(manifest, *, expected_suite=None, course=COURSE):
    expected = manifest.get('manifest_sha256')
    payload = {key: value for key, value in manifest.items()
               if key != 'manifest_sha256'}
    if expected != sha256(payload):
        raise ValueError('S2 gate manifest SHA mismatch')
    if manifest.get('status') != 'frozen' or manifest.get('frozen') is not True:
        raise ValueError('S2 gate manifest is not frozen')
    if manifest.get('course') != course:
        raise ValueError('S2 gate course differs')
    if expected_suite is not None and manifest.get('suite') != SUITE_ALIASES.get(
            expected_suite, expected_suite):
        raise ValueError('S2 gate suite differs')
    expected_groups = COURSE_GROUP_COUNTS.get(course, GROUP_COUNT)
    if len(manifest.get('groups', [])) != expected_groups:
        raise ValueError(f'S2 gate must have {expected_groups} groups')
    if len(manifest.get('cases', [])) != expected_groups * CASES_PER_GROUP:
        raise ValueError(f'S2 gate must have {expected_groups * CASES_PER_GROUP} cases')
    if manifest.get('cases_per_group') != CASES_PER_GROUP:
        raise ValueError('S2 gate cases_per_group must be 20')
    seen = set()
    for group in manifest['groups']:
        if group.get('cases') != CASES_PER_GROUP:
            raise ValueError('S2 gate group has wrong case count')
        if group.get('control_samples') != CASES_PER_GROUP * group['hard_timeout_steps']:
            raise ValueError('S2 gate group budget mismatch')
    for case in manifest['cases']:
        if case['case_sha256'] in seen:
            raise ValueError('duplicate S2 gate case SHA')
        seen.add(case['case_sha256'])
        if case.get('case_key') != [s2_spec.STAGE, manifest['suite'],
                                    case['task'], case['stair_count'],
                                    case['geometry'], case['speed'],
                                    case['stop_mode'], case['force_family'],
                                    case['case_id']]:
            raise ValueError('S2 gate case_key drift')
        if sha256({key: value for key, value in case.items()
                   if key != 'case_sha256'}) != case['case_sha256']:
            raise ValueError('S2 gate case SHA mismatch')
    if manifest['budget']['control_samples'] != sum(
            row['control_samples'] for row in manifest['groups']):
        raise ValueError('S2 gate total budget mismatch')
    return True


def freeze_gate_cases(output_dir, *, version=GATE_VERSION, course=COURSE):
    """Write both A/B manifests; refuse all overwrites (append-only)."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite gate cases in {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for alias in ('gate_A', 'gate_B'):
        manifest = build_gate_manifest(alias, course=course)
        path = output_dir / f'{manifest["suite"]}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        written[alias] = dict(path=str(path), suite=manifest['suite'],
                              manifest_sha256=manifest['manifest_sha256'],
                              cases=len(manifest['cases']),
                              control_samples=manifest['budget']['control_samples'])
    receipt = dict(schema_version=s2_spec.CHECKPOINT_SCHEMA,
                   gate_version=version, course=course,
                   a=written['gate_A'], b=written['gate_B'])
    receipt_path = output_dir / 'freeze_receipt.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


def load_gate_manifest(path, *, expected_suite=None, course=COURSE):
    manifest = json.loads(Path(path).read_text())
    validate_gate_manifest(manifest, expected_suite=expected_suite, course=course)
    return manifest


def batch_groups(cases, *, envs_per_batch=20):
    """Group frozen cases by their gate group; each batch must fit the limit."""
    grouped = {}
    for case in cases:
        grouped.setdefault(case['group_id'], []).append(case)
    batches = []
    for group_id in sorted(grouped):
        rows = grouped[group_id]
        if len(rows) != envs_per_batch:
            raise ValueError(f'group {group_id} has {len(rows)} cases, '
                             f'not {envs_per_batch}')
        batches.append(dict(group_id=group_id, cases=rows,
                            envs_per_batch=envs_per_batch))
    return batches


def build_course_manifest(course, suite):
    if course not in ('S2.2', 'S2.3'):
        raise ValueError('course manifest is S2.2/S2.3')
    version = 'v2' if course == 'S2.2' else 'v3'
    expected_suite = ('S2_GATE_A_' + version.upper() if suite in ('gate_A', 'S2_GATE_A_' + version.upper())
                      else 'S2_GATE_B_' + version.upper())
    groups = course_gate_groups(course)
    cases = []
    group_rows = []
    for group in groups:
        members = [_case(group, expected_suite, case_id)
                   for case_id in range(CASES_PER_GROUP)]
        cases.extend(members)
        group_rows.append(dict(
            group_id=group['group_id'], role_sha256=_group_role(group),
            task=group['task'], stair_count=group['stair_count'],
            geometry_key=group['geometry'].get('terrain_key'),
            platform_width=group['geometry'].get('platform_width'),
            speed=group['speed'], stop_mode=group['stop_mode'],
            post_distance_m=group['post_distance'],
            cases=len(members),
            hard_timeout_steps=int(group['hard_timeout_steps']),
            control_samples=len(members) * int(group['hard_timeout_steps'])))
    result = dict(
        schema_version=s2_spec.CHECKPOINT_SCHEMA, manifest_version=2,
        gate_version=version, status='frozen', frozen=True,
        generator=GENERATOR, sds_sha256=s2_spec.SDS_SHA256, sds_version='v1.2',
        stage=s2_spec.STAGE, suite=expected_suite, course=course,
        access='gate_acceptance_only',
        case_key_fields=list(s2_spec.CASE_KEY_FIELDS),
        cases_per_group=CASES_PER_GROUP, group_count=len(groups),
        groups=group_rows, cases=cases,
        budget=dict(control_samples=sum(row['control_samples'] for row in group_rows),
                    per_group=sum(row['hard_timeout_steps'] * CASES_PER_GROUP
                                  for row in group_rows)),
        note=f'Frozen {course} A/B gate inputs; independent suite salt.')
    result['manifest_sha256'] = sha256(result)
    return result


def validate_course_manifest(manifest, *, expected_suite=None):
    course = manifest.get('course')
    if course not in COURSE_GROUP_COUNTS:
        raise ValueError('unknown course manifest')
    validate_gate_manifest(manifest, expected_suite=None, course=course)
    version = 'v2' if course == 'S2.2' else 'v3'
    if expected_suite is not None:
        expected_name = ('S2_GATE_A_' + version.upper() if expected_suite in
                         ('gate_A', 'S2_GATE_A_' + version.upper()) else
                         'S2_GATE_B_' + version.upper())
        if manifest.get('suite') != expected_name:
            raise ValueError('course gate suite differs')
    if len(manifest['groups']) != COURSE_GROUP_COUNTS[course]:
        raise ValueError(f'{course} gate group count differs')
    if len(manifest['cases']) != COURSE_GROUP_COUNTS[course] * CASES_PER_GROUP:
        raise ValueError(f'{course} gate case count differs')
    return True


def freeze_course_cases(output_dir, course):
    if course not in ('S2.2', 'S2.3'):
        raise ValueError('freeze course is S2.2/S2.3')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    version = 'v2' if course == 'S2.2' else 'v3'
    for alias in ('gate_A', 'gate_B'):
        manifest = build_course_manifest(course, alias)
        path = output_dir / f'{manifest["suite"]}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        written[alias] = dict(path=str(path), suite=manifest['suite'],
                              manifest_sha256=manifest['manifest_sha256'],
                              cases=len(manifest['cases']),
                              control_samples=manifest['budget']['control_samples'])
    receipt = dict(schema_version=s2_spec.CHECKPOINT_SCHEMA,
                   gate_version=version, course=course,
                   a=written['gate_A'], b=written['gate_B'])
    path = output_dir / 'freeze_receipt.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt
