"""Frozen S2.0 diagnostic cases and the six original H failure-case wrappers.

This module only generates and hashes CPU case descriptions.  It never creates
a simulator, advances physics or touches an S1 file.  The S2.0 matrix is a
diagnostic comparison, not a gate or final-acceptance suite; its namespace
(`S2/S2_0`) and generator id are independent of every S1 stream.

Planning rulings DSH-Q015..Q021 are implemented here: S2 streams hash the v1.2
SDS while every S1 field stays on v1.1; the matrix is 15 groups × 5 fixed cases
with two shared map branches; the six H records are an outer wrapper whose
inner case dicts retain every original field and original `case_sha256`.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import glob
import hashlib
import json
import math
from pathlib import Path

from ..v07 import spec as original
from . import s2_spec
from .cases import canonical_json, sha256

MANIFEST_VERSION = 1
STREAM_FIELDS = ('spawn_x', 'spawn_y', 'spawn_yaw')
H_SERIES_LESSONS = ('H00', 'H01', 'H03', 'H06', 'H07', 'H10')
H_SERIES_DIR = 'logs/a2_v09/s1_seed42_r3/user_H_series_20261122'
H_SERIES_REQUIRED_CASE_FIELDS = (
    'stage', 'suite', 'task', 'stair_count', 'lesson', 'geometry', 'speed',
    'stop_mode', 'force_family', 'case_id', 'case_key', 'case_sha256',
    'hard_timeout_steps', 'termination_contract', 'spawn', 'command', 'streams',
    'dynamics', 'observation_noise', 'action_delay_steps', 'proprio_delay_steps',
    'map_delay_steps', 'force_events')
RULINGS = (
    dict(id='Q015', status='ruled', subject='dual SDS identity / S1 stream salt',
         implementation='S1 fields and salts stay v1.1; S2 streams hash v1.2 under an explicit S2 context.'),
    dict(id='Q016', status='ruled', subject='diagnostic plane height-channel ablation',
         implementation='valid-cell height only; valid/age/coords/10 Hz unchanged; interface in s2_map_ablation.'),
    dict(id='Q017', status='ruled', subject='H-series S2_DIAG outer wrapper',
         implementation='outer source S1/S1_DIAG + original path/case_sha + S2 operation_id; inner case unchanged.'),
    dict(id='Q018', status='ruled', subject='H-series hard timeout',
         implementation='original v07 per-case timeouts (1301/1302/1306/1315/1318/1332); reserve 9432.'),
    dict(id='Q019', status='ruled', subject='zero points and platform start',
         implementation='D_post zero at latched top_arrived; first metre zero at four-foot entry; platform spawn +1 m.'),
    dict(id='Q020', status='ruled', subject='S2.0 diagnostic depth',
         implementation='15 groups × case_id 0..4; 20 s flat, L40+5 m, 8 m platform+5 m; two shared map branches = 150 trajectories.'),
    dict(id='Q021', status='ruled', subject='S2 atomic reservation',
         implementation='full-stage itemized upper bound before first physics; no ledger write in CPU code.'),
)
ASSUMPTIONS = (
    dict(id='A1', text='Platform start is local_x = N*d + 1.0 on the true top surface with standard y/yaw ranges (Q019).'),
    dict(id='A2', text='First-metre zero is the base route_x at the first step when all four foot XY projections are on the platform (Q019).'),
    dict(id='A3', text='D_post zero for a complete climb is the base route_x on the latched top_arrived frame (Q019).'),
    dict(id='A4', text='`foot_on_platform` is supplied by the caller from the §5.3 top-support contract; s2_metrics never invents a contact threshold.'),
)
MAP_CHANNEL_MODES = s2_spec.S2_0_MAP_BRANCHES


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _uniform(identity, low, high):
    fraction = (int(identity[:16], 16) >> 11) / float(1 << 53)
    return low + (high - low) * fraction


def stream_id(*, role, episode_id, channel, suite=s2_spec.S2_0_SUITE, stage=s2_spec.STAGE):
    """Independent named S2 stream; never reuses an S1 generator or SHA."""
    _integer(episode_id, 'episode_id')
    if not role or not channel or not stage or not suite:
        raise ValueError('S2 stream identity fields cannot be empty')
    return sha256([s2_spec.S2_0_GENERATOR, s2_spec.S2_STREAM_SALT, s2_spec.SEED,
                   stage, suite, role, episode_id, channel])


# ------------------------------------------------------------- S2 terrain
def s2_terrain_rows():
    """Required S2 collision rows; compilation is still an implementation gap.

    The world frame is explicit so a case's world spawn is reproducible.  Flat
    reuses the frozen v07 flat tile; both stair rows use the S2 geometry and
    only differ in platform width.  A future S2 terrain builder must place each
    row at exactly this origin.
    """
    flat = dict(kind='flat', terrain_key='s2_flat_v07_tile0', tile_id=0,
                source='v07_flat_tile_unchanged')
    flat.update(asdict(original.TERRAIN_BY_LESSON['FLAT']))
    flat.update(kind='flat', terrain_key='s2_flat_v07_tile0', platform_present=False,
                platform_width=8.0, platform_height=0.0,
                platform_front_local_x=None,
                frame_origin_world=list(flat['frame_origin_world']),
                valid_bounds_local=list(flat['valid_bounds_local']))
    rows = [flat]
    for index, width in enumerate(s2_spec.PLATFORM_WIDTHS):
        rows.append(dict(
            kind='s2_stairs', terrain_key=f's2_stairs_h0.20_d0.20_w0.80_pw{width:g}',
            platform_present=True,
            tile_id=None, N=s2_spec.FULL_STAIR_COUNT, h=s2_spec.STAIR_HEIGHT,
            step_depth=s2_spec.STAIR_DEPTH, width=s2_spec.STAIR_WIDTH,
            tread_thickness=original.TREAD_THICKNESS,
            platform_length=s2_spec.PLATFORM_LENGTH,
            platform_width=float(width),
            platform_height=s2_spec.FULL_STAIR_COUNT * s2_spec.STAIR_HEIGHT,
            platform_front_local_x=s2_spec.FULL_STAIR_COUNT * s2_spec.STAIR_DEPTH,
            frame_origin_world=[0.0, 12.0 + 16.0 * index, 0.0],
            valid_bounds_local=[-2.0,
                s2_spec.FULL_STAIR_COUNT * s2_spec.STAIR_DEPTH + s2_spec.PLATFORM_LENGTH,
                -2.0, 2.0],
        ))
    return rows


def _geometry_for(width=None):
    rows = s2_terrain_rows()
    if width is None:
        return deepcopy(rows[0])
    return deepcopy(next(row for row in rows[1:] if row['platform_width'] == float(width)))


# ---------------------------------------------------------------- groups
def s2_0_groups():
    """15 diagnostic groups: 3 flat, 6 full climb, 6 platform start."""
    groups = []
    flat_geometry = _geometry_for()
    for speed in s2_spec.SPEEDS:
        groups.append(dict(stage=s2_spec.STAGE, task='flat_walk', stair_count=0,
                           geometry=flat_geometry, speed=float(speed),
                           stop_mode='none', force_family='none',
                           post_distance_m=None, walk_seconds=s2_spec.FLAT_WALK_SECONDS,
                           timeout_contract='flat_full_window_plus_one_guard_step'))
    for width in s2_spec.PLATFORM_WIDTHS:
        geometry = _geometry_for(width)
        for speed in s2_spec.SPEEDS:
            groups.append(dict(stage=s2_spec.STAGE, task='stairs_walk',
                               stair_count=s2_spec.FULL_STAIR_COUNT, geometry=geometry,
                               speed=float(speed), stop_mode='none',
                               force_family='none',
                               post_distance_m=s2_spec.POST_TOP_DIAGNOSTIC_DISTANCE,
                               walk_seconds=None,
                               timeout_contract='s2_complete_climb_t_max'))
            groups.append(dict(stage=s2_spec.STAGE, task='platform_walk',
                               stair_count=s2_spec.FULL_STAIR_COUNT, geometry=geometry,
                               speed=float(speed), stop_mode='none',
                               force_family='none',
                               post_distance_m=s2_spec.POST_TOP_DIAGNOSTIC_DISTANCE,
                               walk_seconds=None,
                               timeout_contract='s2_platform_start_t_max'))
    assert len(groups) == 15
    return groups


def _group_role(group):
    identity = {key: group[key] for key in ('stage', 'task', 'stair_count', 'geometry',
                                            'speed', 'stop_mode', 'force_family')}
    return sha256(identity)


def _spawn(group, suite, case_id):
    geometry = group['geometry']
    streams = {field: stream_id(role=_group_role(group), episode_id=case_id,
                                channel=field, suite=suite)
               for field in STREAM_FIELDS}
    y = _uniform(streams['spawn_y'], *original.SPAWN_Y)
    yaw = _uniform(streams['spawn_yaw'], *original.SPAWN_YAW_RADIANS)
    if group['task'].startswith('platform_'):
        # A3: fixed one metre onto the true top platform, real surface height.
        x = float(geometry['platform_front_local_x']) + s2_spec.PLATFORM_START_OFFSET
        ground_z = float(geometry['platform_height'])
        spawn_surface = 'top_platform'
    else:
        x = _uniform(streams['spawn_x'], *original.SPAWN_X)
        ground_z = 0.0
        spawn_surface = 'ground'
    origin = geometry['frame_origin_world']
    z = ground_z + 0.4
    return streams, dict(
        local_position_xy=[x, y], local_yaw_radians=yaw,
        frame_origin_world=list(origin),
        world_position_xy=[origin[0] + x, origin[1] + y],
        local_position=[x, y, z],
        world_position=[origin[0] + x, origin[1] + y, origin[2] + z],
        quaternion_wxyz=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        joint_position=list(original.A2_DEFAULT_JOINT_POS),
        joint_velocity=[0.0] * 12, base_linear_velocity=[0.0] * 3,
        base_angular_velocity=[0.0] * 3,
        spawn_surface=spawn_surface,
        height_contract=('top_surface_z + 0.4; a2_constants.INIT_STATE.pos[2] baseline'
                         if spawn_surface == 'top_platform' else
                         'a2_constants.INIT_STATE.pos[2]=0.4; v07._reset_ids'),
    )


def _hard_timeout(group, spawn):
    if group['task'] == 'flat_walk':
        return s2_spec.FLAT_WALK_HARD_TIMEOUT_STEPS
    x, _ = spawn['local_position_xy']
    if group['task'] == 'stairs_walk':
        return s2_spec.timeout_steps(
            task='stairs_walk', speed=group['speed'],
            stair_count=group['stair_count'], step_height=group['geometry']['h'],
            step_depth=group['geometry']['step_depth'], approach_length=-x,
            post_distance=group['post_distance_m'])
    if group['task'] == 'platform_walk':
        return s2_spec.timeout_steps(task='platform_walk', speed=group['speed'],
                                     post_distance=group['post_distance_m'])
    raise ValueError('S2.0 only freezes walk tasks')


def _case(group, suite, case_id):
    streams, spawn = _spawn(group, suite, case_id)
    item = deepcopy(group)
    item.update(suite=suite, case_id=case_id, streams=streams, spawn=spawn,
                hard_timeout_steps=_hard_timeout(group, spawn),
                command=dict(vx=group['speed'], vy=0.0, wz=0.0,
                             walk_seconds=group['walk_seconds'],
                             post_distance_m=group['post_distance_m'],
                             ramp_steps=0, hold_steps=0),
                dynamics='nominal_original_6400', observation_noise='none',
                action_delay_steps=0, proprio_delay_steps=0, map_delay_steps=0,
                force_events=[], diagnostic_map_channels='shared_real_and_plane_pair',
                map_branches=list(s2_spec.S2_0_MAP_BRANCHES))
    item['case_key'] = [item[key] for key in s2_spec.CASE_KEY_FIELDS]
    item['case_sha256'] = sha256(item)
    return item


def build_s2_0_manifest(*, cases_per_group=s2_spec.S2_0_CASES_PER_GROUP):
    _integer(cases_per_group, 'cases_per_group', 1)
    if cases_per_group != s2_spec.S2_0_CASES_PER_GROUP:
        raise ValueError('Q020 fixes exactly five S2.0 cases per group')
    groups = s2_0_groups()
    cases = [_case(group, s2_spec.S2_0_SUITE, case_id)
             for group in groups for case_id in range(cases_per_group)]
    group_rows = []
    for group in groups:
        role = _group_role(group)
        members = [row for row in cases if row['case_key'][:-1] == [
            group['stage'], s2_spec.S2_0_SUITE, group['task'], group['stair_count'],
            group['geometry'], group['speed'], group['stop_mode'], group['force_family']]]
        group_rows.append(dict(
            role_sha256=role, task=group['task'],
            stair_count=group['stair_count'], speed=group['speed'],
            platform_width=group['geometry'].get('platform_width'),
            geometry_key=group['geometry'].get('terrain_key'),
            post_distance_m=group['post_distance_m'],
            cases=len(members),
            max_hard_timeout_steps=max(row['hard_timeout_steps'] for row in members)))
    result = dict(schema_version=s2_spec.CHECKPOINT_SCHEMA,
                  manifest_version=MANIFEST_VERSION,
                  status='frozen', frozen=True,
                  generator=s2_spec.S2_0_GENERATOR,
                  sds_sha256=s2_spec.SDS_SHA256, sds_version='v1.2',
                  sds_identity_status=s2_spec.IDENTITY_STATUS,
                  stream_salt_status='S2_v1.2_independent_stage_suite',
                  stage=s2_spec.STAGE, suite=s2_spec.S2_0_SUITE,
                  course=s2_spec.S2_0_COURSE,
                  supersedes=dict(s2_spec.S2_0_SUPERSEDES),
                  access='diagnostic_only', recipe=s2_spec.RECIPE,
                  entry_weight=s2_spec.S1_ENTRY_IDENTITY,
                  case_key_fields=list(s2_spec.CASE_KEY_FIELDS),
                  cases_per_group=cases_per_group,
                  diagnostic_matrix=dict(flat=3, stairs_walk=6, platform_walk=6,
                                         platform_widths=list(s2_spec.PLATFORM_WIDTHS),
                                         speeds=list(s2_spec.SPEEDS),
                                         stop_tasks='excluded by Q020',
                                         map_branches=list(s2_spec.S2_0_MAP_BRANCHES),
                                         case_ids=list(range(cases_per_group)),
                                         native_trajectories=s2_spec.S2_0_NATIVE_TRAJECTORIES),
                  rulings=list(RULINGS),
                  assumptions=list(ASSUMPTIONS),
                  map_channel_contract=dict(
                      status='ruled_Q016', implemented_in='s2_map_ablation.py',
                      channels_replaced=('height',),
                      channels_unchanged=('valid', 'age'),
                      invalid_cells='0/0/1', refresh_hz=10,
                      height_formula='clip((z_plane_world - z_base_sample)/1.5, -1, 1)',
                      activation='full climb: first normal refresh after first real front-foot top support; '
                                 'platform/flat: first frame; latched to terminal',
                      scope='15 S2.0 diagnostic groups only; H/training/gate/final use the real map'),
                  terrain_rows=s2_terrain_rows(),
                  groups=group_rows, cases=cases)
    result['manifest_sha256'] = sha256(result)
    return result


def validate_s2_0_manifest(manifest):
    expected = manifest.get('manifest_sha256')
    payload = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if expected != sha256(payload):
        raise ValueError('S2.0 manifest SHA256 mismatch')
    if manifest.get('frozen') is not True or manifest.get('status') != 'frozen':
        raise ValueError('S2.0 manifest must be the frozen Q020 matrix')
    regenerated = build_s2_0_manifest(cases_per_group=manifest['cases_per_group'])
    if canonical_json(manifest) != canonical_json(regenerated):
        raise ValueError('S2.0 manifest differs from the frozen generation rules')
    return True


# ------------------------------------------------------- H series case0
def h_series_paths(root):
    root = Path(root)
    directory = root / H_SERIES_DIR
    result = []
    for lesson in H_SERIES_LESSONS:
        matches = sorted(directory.glob(f'*_{lesson}_vx0.5_none.json'))
        if len(matches) != 1:
            raise ValueError(f'expected exactly one case0 JSON for {lesson}, got {len(matches)}')
        result.append(matches[0])
    return result


def _initial_pose_difference(episode_pose, spawn):
    local = spawn['local_position_xy'] + [spawn['local_yaw_radians']]
    return max(abs(float(a) - float(b)) for a, b in zip(episode_pose, local))


def _h_record(path, root):
    payload = path.read_bytes()
    document = json.loads(payload)
    case = deepcopy(document['case'])
    episode = document.get('episode', {})
    missing_case_fields = [key for key in H_SERIES_REQUIRED_CASE_FIELDS if key not in case]
    normalized = {key: value for key, value in case.items() if key != 'case_sha256'}
    recomputed = sha256(normalized)
    if 'case_sha256' in case and recomputed != case['case_sha256']:
        raise ValueError(f'{path.name}: normalized case SHA differs from stored case_sha256')
    diagnostic_missing = []
    spawn = case.get('spawn', {})
    if 'qpos' not in spawn:
        diagnostic_missing.append('spawn.qpos_raw (only frame_audit.qpos_sha256 is retained)')
    if 'qvel' not in spawn:
        diagnostic_missing.append('spawn.qvel_raw (only frame_audit.qvel_sha256 is retained)')
    if 'rng_state' not in episode:
        diagnostic_missing.append('episode.rng_state_raw (only rng_state_sha256_before_sample is retained)')
    audit = document.get('frame_audit') or []
    control_steps = int(episode.get('control_steps', episode.get('terminal_step', 0)))
    if len(audit) < control_steps:
        diagnostic_missing.append(
            f'full_rate_frame_audit ({len(audit)}/{control_steps} frames retained)')
    initial = episode.get('initial_pose')
    pose_difference = _initial_pose_difference(initial, spawn) if initial else None
    geometry = case['geometry']
    platform_front = float(geometry['N']) * float(geometry['step_depth'])
    terminal_x = episode.get('forward_distance_end')
    failure_position = None
    if terminal_x is not None:
        terminal_x = float(terminal_x)
        failure_position = dict(
            terminal_route_x=terminal_x,
            source_field='episode.forward_distance_end',
            signed_distance_at_terminal_episode_field=episode.get('forward_distance_terminal'),
            platform_front_local_x=platform_front,
            signed_from_platform_front_m=terminal_x - platform_front,
            onto_platform_m=max(0.0, terminal_x - platform_front),
            before_platform_m=max(0.0, platform_front - terminal_x),
            surface=('platform' if terminal_x >= platform_front
                     else 'stairs' if terminal_x >= 0.0 else 'approach'),
        )
    timeout_calibers = dict(
        selected='original_v07_Q018',
        original_v07=int(case['hard_timeout_steps']),
        reserve_batch_padding=s2_spec.H_SERIES_RESERVED_SAMPLES,
    )
    operation_id = sha256(['S2_DIAG', 'targeted_regression_h_series_case0',
                           case.get('case_sha256')])
    return dict(
        lesson=geometry['lesson_id'],
        path=str(path.relative_to(root)),
        json_sha256=hashlib.sha256(payload).hexdigest(),
        source=dict(stage='S1', suite='S1_DIAG', original_path=str(path.relative_to(root)),
                    original_case_sha256=case.get('case_sha256'),
                    role='targeted_regression_not_new_random_case'),
        operation_id=operation_id,
        case_sha256_recomputed=recomputed,
        original_case_sha256=case.get('case_sha256'),
        case_sha256_matches=(case.get('case_sha256') == recomputed),
        missing_case_fields=missing_case_fields,
        missing_diagnostic_fields=diagnostic_missing,
        hard_timeout_steps=timeout_calibers,
        original_failure_position=failure_position,
        inner_case_unchanged=True,
        case=case,
        replay=dict(
            spawn=case.get('spawn'), streams=case.get('streams'),
            command=case.get('command'), geometry=case.get('geometry'),
            hard_timeout_steps_original=case.get('hard_timeout_steps'),
            termination_contract=case.get('termination_contract'),
            episode_id=episode.get('episode_id'),
            initial_pose_from_episode=initial,
            initial_pose_max_abs_difference=pose_difference,
            rng_state_sha256_before_sample=episode.get('rng_state_sha256_before_sample'),
            controller_stream=episode.get('controller_stream'),
            terminal_step=episode.get('terminal_step'),
            duration=episode.get('duration'),
            reason=episode.get('reason'),
            failure_masks=episode.get('failure_masks'),
            terminal_platform_support=episode.get('platform_support'),
            forward_distance_end=episode.get('forward_distance_end'),
            route_final=episode.get('route_final'),
            frame_audit_frames=len(audit),
            control_steps=control_steps,
        ),
    )


def build_h_series_case0_manifest(root=None):
    root = Path(root or s2_spec.PROJECT_ROOT)
    records = [_h_record(path, root) for path in h_series_paths(root)]
    result = dict(schema_version=s2_spec.CHECKPOINT_SCHEMA, manifest_version=MANIFEST_VERSION,
                  status='frozen', frozen=True,
                  sds_sha256=s2_spec.SDS_SHA256, sds_version='v1.2',
                  sds_identity_status=s2_spec.IDENTITY_STATUS,
                  stage=s2_spec.STAGE, suite='S2_DIAG',
                  purpose='original_six_failure_pairing', namespace_status='ruled_Q017',
                  source_directory=H_SERIES_DIR, lessons=list(H_SERIES_LESSONS),
                  provenance=dict(
                      command='vx=0.5 m/s, vy=0, wz=0',
                      stair_count=6, step_depth=0.30, stair_width=0.80,
                      platform='original JSON top platform (platform_length as stored)',
                      termination='v07_original_stairs (original JSON)',
                      terminal_position='episode.forward_distance_end; surface judged per case',
                      timeout='original v07 per case; batch reserve 9432 (Q018)',
                      map='real only; plane branch is never used for H (Q016)'),
                  rulings=list(RULINGS),
                  records=records)
    result['manifest_sha256'] = sha256(result)
    return result


def validate_h_series_manifest(manifest, root=None):
    expected = manifest.get('manifest_sha256')
    payload = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if expected != sha256(payload):
        raise ValueError('H-series manifest SHA256 mismatch')
    if manifest.get('frozen') is not True or manifest.get('status') != 'frozen':
        raise ValueError('H-series wrapper must be frozen under Q017')
    regenerated = build_h_series_case0_manifest(root)
    if canonical_json(manifest) != canonical_json(regenerated):
        raise ValueError('H-series wrapper differs from the retained JSON evidence')
    return True


# --------------------------------------------------- branch run plan (Q022)
def build_s2_diag_run_plan(manifest):
    """150 branch trajectories: every frozen case under both map branches."""
    if manifest.get('suite') != s2_spec.S2_0_SUITE:
        raise ValueError('run plan requires the S2_DIAG matrix')
    entries = []
    sequence = 0
    for case in manifest['cases']:
        for branch in s2_spec.S2_0_MAP_BRANCHES:
            entries.append(dict(
                sequence=sequence,
                operation_id=sha256(['S2_DIAG_BRANCH', manifest['manifest_sha256'],
                                     case['case_sha256'], branch]),
                case_sha256=case['case_sha256'], case_id=case['case_id'],
                group_role_sha256=_group_role(dict(
                    stage=case['stage'], task=case['task'],
                    stair_count=case['stair_count'], geometry=case['geometry'],
                    speed=case['speed'], stop_mode=case['stop_mode'],
                    force_family=case['force_family'])),
                task=case['task'], stair_count=case['stair_count'],
                speed=case['speed'],
                platform_width=case['geometry'].get('platform_width'),
                map_branch=branch, hard_timeout_steps=case['hard_timeout_steps'],
                sample_streams=dict(case['streams']),
                command=dict(case['command'])))
            sequence += 1
    if len(entries) != s2_spec.S2_0_NATIVE_TRAJECTORIES:
        raise ValueError('run plan must contain exactly 150 branch trajectories')
    result = dict(schema_version=s2_spec.CHECKPOINT_SCHEMA,
                  stage=s2_spec.STAGE, suite=s2_spec.S2_0_SUITE,
                  course=s2_spec.S2_0_COURSE,
                  manifest_sha256=manifest['manifest_sha256'],
                  cases=len(manifest['cases']),
                  branches=list(s2_spec.S2_0_MAP_BRANCHES),
                  per_branch=len(manifest['cases']),
                  entries=entries, native_trajectories=len(entries),
                  batches=2 * len(manifest['groups']),
                  batch_size=manifest['cases_per_group'],
                  map_contract='branches share one case and one stream set; '
                               'map_branch is an operation id, not a sampling key')
    result['plan_sha256'] = sha256(result)
    return result


def validate_s2_diag_run_plan(plan, manifest):
    expected = plan.get('plan_sha256')
    payload = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if expected != sha256(payload):
        raise ValueError('S2_DIAG run plan SHA256 mismatch')
    if canonical_json(plan) != canonical_json(build_s2_diag_run_plan(manifest)):
        raise ValueError('S2_DIAG run plan differs from the frozen matrix')
    return True


# ----------------------------------------------------------- reservations
def s2_0_matrix_cost(manifest):
    """Q020 two shared map branches; five cases per group, real padding."""
    from . import budget
    batch_size = manifest['cases_per_group']
    cost = budget.batch_worst_cost(manifest['cases'], batch_size=batch_size)
    return dict(real_branch_control_samples=cost['control_samples'],
                plane_branch_control_samples=cost['control_samples'],
                total_control_samples=2 * cost['control_samples'],
                batches=2 * len(cost['batches']), batch_size=batch_size,
                native_trajectories=manifest['diagnostic_matrix']['native_trajectories'])


def h_series_reservation(h_manifest):
    """Q018: original v07 timeouts, same-batch padding, explicit 9432 reserve."""
    timeouts = [int(row['hard_timeout_steps']['original_v07'])
                for row in h_manifest['records']]
    if len(timeouts) != 6 or not all(a == b for a, b in zip(
            timeouts, s2_spec.H_SERIES_TIMEOUTS)):
        raise ValueError('H-series timeouts differ from Q018')
    reserved = max(s2_spec.H_SERIES_BATCH_UPPER_BOUND,
                   s2_spec.H_SERIES_RESERVED_SAMPLES)
    return dict(actual_timeout_sum=sum(timeouts),
                same_batch_upper_bound=s2_spec.H_SERIES_BATCH_UPPER_BOUND,
                actual_batch_control_samples=s2_spec.H_SERIES_BATCH_UPPER_BOUND,
                reserved_control_samples=reserved,
                batches=1)


def s2_0_reservation_draft(manifest, h_manifest):
    """Q021 S2.0 sub-cap line: matrix plus the six H baseline reserve."""
    matrix = s2_0_matrix_cost(manifest)
    h_series = h_series_reservation(h_manifest)
    total = matrix['total_control_samples'] + h_series['reserved_control_samples']
    if total > s2_spec.S2_0_NONLEARNING_SUBCAP:
        raise ValueError('S2.0 worst-case reservation exceeds its 0.5e8 sub-cap')
    return dict(
        stage=s2_spec.STAGE, suite=s2_spec.S2_0_SUITE,
        sds_sha256=s2_spec.SDS_SHA256,
        manifest_sha256=manifest['manifest_sha256'],
        h_manifest_sha256=h_manifest['manifest_sha256'],
        **matrix, h_series=h_series,
        total_control_samples_with_h=total,
        physics_substeps=4 * total,
        nonlearning_subcap=s2_spec.S2_0_NONLEARNING_SUBCAP,
        remaining_nonlearning=s2_spec.S2_0_NONLEARNING_SUBCAP - total,
        admitted=total <= s2_spec.S2_0_NONLEARNING_SUBCAP,
    )


# --------------------------------------------------------------- freezing
def _write_versioned(path, payload):
    """Append-only writer: never overwrite a different frozen target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload) + '\n'
    candidate = path
    version = 1
    while candidate.exists():
        if candidate.read_text(encoding='utf-8') == text:
            return candidate, hashlib.sha256(candidate.read_bytes()).hexdigest()
        version += 1
        candidate = path.with_name(f'{path.stem}_v{version}{path.suffix}')
    with candidate.open('x', encoding='utf-8') as handle:
        handle.write(text)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    return candidate, hashlib.sha256(candidate.read_bytes()).hexdigest()


def freeze_s2_0(root=None, *, cases_per_group=s2_spec.S2_0_CASES_PER_GROUP):
    root = Path(root or s2_spec.PROJECT_ROOT)
    s2_spec.validate_sds_context('S2')
    manifest = build_s2_0_manifest(cases_per_group=cases_per_group)
    h_manifest = build_h_series_case0_manifest(root)
    validate_s2_0_manifest(manifest)
    validate_h_series_manifest(h_manifest, root)
    run_plan = build_s2_diag_run_plan(manifest)
    validate_s2_diag_run_plan(run_plan, manifest)
    draft = s2_0_reservation_draft(manifest, h_manifest)
    directory = root / 'logs/a2_v09/s2_0/cases'
    files = dict(
        s2_diag=dict(path=str((directory / 'S2_DIAG.json').relative_to(root)),
                     manifest_sha256=manifest['manifest_sha256']),
        run_plan=dict(path=str((directory / 'S2_DIAG_run_plan.json').relative_to(root)),
                      plan_sha256=run_plan['plan_sha256']),
        h_series=dict(path=str((directory / 'h_series_case0.json').relative_to(root)),
                      manifest_sha256=h_manifest['manifest_sha256']),
    )
    written, digest = _write_versioned(directory / 'S2_DIAG.json', manifest)
    files['s2_diag']['path'] = str(written.relative_to(root))
    files['s2_diag']['file_sha256'] = digest
    written, digest = _write_versioned(directory / 'S2_DIAG_run_plan.json', run_plan)
    files['run_plan']['path'] = str(written.relative_to(root))
    files['run_plan']['file_sha256'] = digest
    files['h_series']['file_sha256'] = hashlib.sha256(
        (directory / 'h_series_case0.json').read_bytes()).hexdigest()
    receipt = dict(schema_version=s2_spec.CHECKPOINT_SCHEMA, stage=s2_spec.STAGE,
                   suite=s2_spec.S2_0_SUITE, course=s2_spec.S2_0_COURSE,
                   sds_sha256=s2_spec.SDS_SHA256,
                   entry_weight=s2_spec.S1_ENTRY_IDENTITY, files=files,
                   supersedes=dict(s2_spec.S2_0_SUPERSEDES),
                   reservation_draft=draft)
    receipt['receipt_sha256'] = sha256(receipt)
    receipt_path, _ = _write_versioned(directory / 'freeze_receipt_S2_DIAG.json', receipt)
    register_supersede(
        root, old_suite='S2_0',
        old_manifest_sha256=s2_spec.S2_0_SUPERSEDES['manifest_sha256'],
        old_file_sha256=s2_spec.S2_0_SUPERSEDES['file_sha256'],
        new_suite=s2_spec.S2_0_SUITE,
        new_manifest_sha256=manifest['manifest_sha256'],
        new_file_sha256=files['s2_diag']['file_sha256'],
        new_receipt_path=str(receipt_path.relative_to(root)),
        new_receipt_content_sha256=receipt['receipt_sha256'],
        new_receipt_file_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        reason='Q022 更名、旧清单零物理')
    return receipt


def load_frozen_manifest(path, root=None, *, historical=False):
    """Read a frozen artifact; historical superseded files skip regeneration."""
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if historical:
        expected = data.get('manifest_sha256')
        payload = {key: value for key, value in data.items() if key != 'manifest_sha256'}
        if expected != sha256(payload):
            raise ValueError('historical manifest self-hash mismatch')
        return data
    if 'diagnostic_matrix' in data:
        validate_s2_0_manifest(data)
    elif 'records' in data:
        validate_h_series_manifest(data, root)
    else:
        raise ValueError('unrecognized frozen artifact')
    return data


# ------------------------------------------------- supersede register (Q026)
SUPERSEDE_REGISTER = 'logs/a2_v09/s2_0/cases/supersede_register.jsonl'


def _utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def register_supersede(root, *, old_suite, old_manifest_sha256, old_file_sha256,
                       new_suite, new_manifest_sha256, new_file_sha256,
                       new_receipt_path, new_receipt_content_sha256,
                       new_receipt_file_sha256, reason,
                       ruling='DSH-Q026', written_at=None):
    """Append-only, idempotent old→new registration; never rewrites history.

    Both receipt identities are mandatory: content SHA (receipt_sha256 over the
    canonical payload) and file SHA (bytes on disk), per Q026.
    """
    import fcntl
    import os
    root = Path(root)
    if old_suite != 'S2_0' or new_suite != 'S2_DIAG':
        raise ValueError('Q026 registers only S2_0 → S2_DIAG')
    for value, name in ((old_manifest_sha256, 'old_manifest_sha256'),
                        (old_file_sha256, 'old_file_sha256'),
                        (new_manifest_sha256, 'new_manifest_sha256'),
                        (new_file_sha256, 'new_file_sha256'),
                        (new_receipt_content_sha256, 'new_receipt_content_sha256'),
                        (new_receipt_file_sha256, 'new_receipt_file_sha256')):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f'{name} must be a 64-char SHA256')
    path = root / SUPERSEDE_REGISTER
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip()] if path.exists() else []
    supersedes_line = None
    for number, row in enumerate(existing, 1):
        if (row.get('old_manifest_sha256') == old_manifest_sha256
                and row.get('new_manifest_sha256') == new_manifest_sha256):
            supersedes_line = number
    record = dict(old_suite=old_suite, old_manifest_sha256=old_manifest_sha256,
                  old_file_sha256=old_file_sha256, new_suite=new_suite,
                  new_manifest_sha256=new_manifest_sha256,
                  new_file_sha256=new_file_sha256,
                  new_receipt_path=new_receipt_path,
                  new_receipt_content_sha256=new_receipt_content_sha256,
                  new_receipt_file_sha256=new_receipt_file_sha256,
                  reason=reason, ruling=ruling,
                  schema_version=2, supersedes_line=supersedes_line,
                  legacy_pre_dual_sha=False,
                  written_at=written_at or _utc_now())
    with path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get('old_manifest_sha256') == old_manifest_sha256
                        and row.get('new_manifest_sha256') == new_manifest_sha256
                        and row.get('new_receipt_content_sha256') is not None
                        and row.get('new_receipt_file_sha256') is not None):
                    return dict(row, already_registered=True)
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(record) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            return dict(record, already_registered=False)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def lookup_supersede(root, old_sha256):
    """Return the newest registration matching either old manifest/file SHA."""
    path = Path(root) / SUPERSEDE_REGISTER
    if not path.exists():
        return None
    found = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if old_sha256 in (row.get('old_manifest_sha256'), row.get('old_file_sha256')):
            found = row
    return found


HISTORY_PROJECTION = 'logs/a2_v09/s2_0/cases/history/history_index_v2.json'


def history_projection(root):
    """Read-only projection derived solely from the append-only register."""
    root = Path(root)
    register = root / SUPERSEDE_REGISTER
    if not register.exists():
        raise FileNotFoundError('supersede register is the single source')
    rows = [json.loads(line) for line in register.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    entries = []
    for row in rows:
        if not row.get('new_receipt_content_sha256') or not row.get('new_receipt_file_sha256'):
            continue  # legacy pre-dual-SHA Q026 row; the upgraded row for the pair follows
        entries.append(dict(
            old_suite=row['old_suite'],
            old_manifest_sha256=row['old_manifest_sha256'],
            old_file_sha256=row['old_file_sha256'],
            new_suite=row['new_suite'],
            new_manifest_sha256=row['new_manifest_sha256'],
            new_file_sha256=row['new_file_sha256'],
            new_receipt_path=row['new_receipt_path'],
            new_receipt_content_sha256=row['new_receipt_content_sha256'],
            new_receipt_file_sha256=row['new_receipt_file_sha256'],
            reason=row['reason'], ruling=row['ruling'],
            written_at=row['written_at'],
            schema_version=row.get('schema_version', 2),
            supersedes_line=row.get('supersedes_line', 1),
            legacy_pre_dual_sha=bool(row.get('legacy_pre_dual_sha', False))))
    projection = dict(
        schema_version=s2_spec.CHECKPOINT_SCHEMA, stage=s2_spec.STAGE,
        suite=s2_spec.S2_0_SUITE, course=s2_spec.S2_0_COURSE,
        read_only_projection=True,
        source=SUPERSEDE_REGISTER,
        register_sha256=hashlib.sha256(register.read_bytes()).hexdigest(),
        entries=entries, entry_count=len(entries),
        warning='DERIVED FILE: rebuild with scripts/rebuild_a2_v09_s2_0_history_projection.py; never edit by hand',
    )
    projection['projection_sha256'] = sha256(projection)
    return projection


def write_history_projection(root, projection, *, path=HISTORY_PROJECTION):
    """Only a byte-for-byte derivation from the register may be written."""
    required = {'read_only_projection', 'source', 'register_sha256', 'entries',
                'projection_sha256', 'warning'}
    if not isinstance(projection, dict) or not required <= set(projection):
        raise ValueError('refusing to write a non-derived history index')
    if projection.get('read_only_projection') is not True:
        raise ValueError('projection must be explicitly read-only')
    payload = {key: value for key, value in projection.items()
               if key != 'projection_sha256'}
    if projection['projection_sha256'] != sha256(payload):
        raise ValueError('projection SHA mismatch')
    # B-V41-1: a caller can recompute its own SHA, so re-derive now and require
    # canonical byte equality with the register projection.
    derived = history_projection(root)
    if canonical_json(projection) != canonical_json(derived):
        raise ValueError('projection does not equal the current register derivation')
    target = Path(root) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.json.tmp')
    temp.write_text(canonical_json(projection) + '\n', encoding='utf-8')
    import os
    os.replace(temp, target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def rebuild_history_projection(root, *, path=HISTORY_PROJECTION):
    return write_history_projection(root, history_projection(root), path=path)
