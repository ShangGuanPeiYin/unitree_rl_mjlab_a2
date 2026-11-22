"""CPU-only S2.0 frozen case/wrapper tests; no simulator, physics or result."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec as original
from src.tasks.velocity.v09 import spec as s1_spec
from src.tasks.velocity.v09 import s2_spec, s2_cases


@pytest.fixture(scope='module')
def manifest():
    return s2_cases.build_s2_0_manifest()


@pytest.fixture(scope='module')
def h_manifest():
    return s2_cases.build_h_series_case0_manifest(ROOT)


def test_identity_contexts_never_accept_both_versions():
    identity = s2_spec.validate_identity()
    assert s2_spec.SDS_SHA256 == '09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe'
    assert s2_spec.S1_SDS_SHA256_V11 == s1_spec.SDS_SHA256
    assert s2_spec.SDS_SHA256 != s1_spec.SDS_SHA256
    assert s2_spec.validate_sds_context('S2')['matches'] is True
    with pytest.raises(ValueError):
        s2_spec.validate_sds_context('S1')
    with pytest.raises(ValueError):
        s2_spec.validate_sds_context('S1+S2')
    assert identity['entry']['sha256'] == s2_spec.S1_ENTRY_WEIGHT_SHA256
    assert identity['entry_original']['sha256'] == identity['entry']['sha256']
    assert s2_spec.S1_ENTRY_LEARNING_SAMPLES == 100_024_320
    assert 'not_passed' in s2_spec.S1_ENTRY_IDENTITY['s1_gate_a']


def test_q022_formal_suite_is_s2_diag_with_supersedes(manifest):
    s2_cases.validate_s2_0_manifest(manifest)
    assert manifest['status'] == 'frozen' and manifest['frozen'] is True
    assert manifest['stage'] == 'S2'
    assert manifest['suite'] == 'S2_DIAG'
    assert manifest['course'] == 'S2.0'
    assert manifest['generator'] == 'v09-s2-sha256-u53-v1'
    assert manifest['sds_sha256'] == s2_spec.SDS_SHA256
    assert manifest['supersedes']['manifest_sha256'] == (
        '5efda5abc35ad714dcf163b1648e2ac4ceac24edc12852889ee6cd3d3739acdf')
    assert manifest['supersedes']['reason'].startswith('Q022')
    assert len(manifest['cases']) == 75
    assert {row['case_id'] for row in manifest['cases']} == set(range(5))
    assert len(manifest['groups']) == 15


def test_old_s2_0_freeze_is_retained_not_overwritten():
    old = ROOT / 'logs/a2_v09/s2_0/cases/S2_0.json'
    history = ROOT / 'logs/a2_v09/s2_0/cases/history/S2_0_v1_20261122.json'
    assert old.exists() and history.exists()
    assert old.read_bytes() == history.read_bytes()
    loaded = json.loads(old.read_text())
    assert loaded['suite'] == 'S2_0'
    assert loaded['manifest_sha256'] == (
        '5efda5abc35ad714dcf163b1648e2ac4ceac24edc12852889ee6cd3d3739acdf')
    new = ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG.json'
    assert new.exists()
    assert json.loads(new.read_text())['supersedes']['manifest_sha256'] == loaded['manifest_sha256']


def test_run_plan_has_150_branch_trajectories(manifest):
    plan = s2_cases.build_s2_diag_run_plan(manifest)
    s2_cases.validate_s2_diag_run_plan(plan, manifest)
    assert plan['suite'] == 'S2_DIAG'
    assert plan['native_trajectories'] == 150
    assert plan['cases'] == 75
    assert plan['per_branch'] == 75
    assert plan['branches'] == ['real', 'plane_diagnostic']
    assert plan['batches'] == 30
    assert len(plan['entries']) == 150
    for case in manifest['cases']:
        entries = [row for row in plan['entries']
                   if row['case_sha256'] == case['case_sha256']]
        assert {row['map_branch'] for row in entries} == {'real', 'plane_diagnostic'}
        assert {tuple(sorted(row['sample_streams'].items())) for row in entries} == {
            tuple(sorted(case['streams'].items()))}
        assert {row['operation_id'] for row in entries}.__len__() == 2
        assert {row['hard_timeout_steps'] for row in entries} == {
            case['hard_timeout_steps']}


def test_s2_streams_use_v12_and_independent_namespace(manifest):
    from src.tasks.velocity.v09 import cases as s1_cases
    s1_stream = s1_cases.stream_id(role='flat_walk', episode_id=0, channel='spawn_x')
    for row in manifest['cases']:
        assert all(len(value) == 64 for value in row['streams'].values())
        assert row['streams']['spawn_x'] != s1_stream
    assert len({row['streams']['spawn_x'] for row in manifest['cases']}) == 75


def test_generation_is_deterministic(manifest):
    assert json.dumps(s2_cases.build_s2_0_manifest(), sort_keys=True) == json.dumps(
        manifest, sort_keys=True)
    assert s2_cases.build_s2_0_manifest()['manifest_sha256'] == manifest['manifest_sha256']
    with pytest.raises(ValueError):
        s2_cases.build_s2_0_manifest(cases_per_group=20)


def test_case_key_and_sha_match_frozen_fields(manifest):
    from src.tasks.velocity.v09.cases import sha256
    for row in manifest['cases']:
        assert row['case_key'] == [row[key] for key in s2_spec.CASE_KEY_FIELDS]
        payload = {key: value for key, value in row.items() if key != 'case_sha256'}
        assert row['case_sha256'] == sha256(payload)
        assert row['stage'] == 'S2'
        assert row['suite'] == 'S2_DIAG'
        assert row['stop_mode'] == 'none'
        assert row['force_family'] == 'none'
        assert row['map_branches'] == ['real', 'plane_diagnostic']


def test_platform_start_uses_true_top_surface_and_q025_window(manifest):
    platforms = [row for row in manifest['cases'] if row['task'] == 'platform_walk']
    assert len(platforms) == 30
    for row in platforms:
        geometry = row['geometry']
        assert geometry['N'] == 40
        assert geometry['h'] == 0.2
        assert geometry['platform_height'] == 8.0
        assert geometry['platform_front_local_x'] == 8.0
        x, _ = row['spawn']['local_position_xy']
        assert x == pytest.approx(9.0)
        assert row['spawn']['local_position'][2] == pytest.approx(8.4)
        assert row['spawn']['spawn_surface'] == 'top_platform'
        assert -0.05 <= row['spawn']['local_position_xy'][1] <= 0.05
        assert abs(row['spawn']['local_yaw_radians']) <= 5 * 3.141592653589793 / 180
        # Q025: the 5 m platform walk includes the first metre; no extra timeout.
        assert row['hard_timeout_steps'] == s2_spec.timeout_steps(
            task='platform_walk', speed=row['speed'], post_distance=5.0)


def test_full_climb_uses_l40_and_actual_approach(manifest):
    climbs = [row for row in manifest['cases'] if row['task'] == 'stairs_walk']
    assert len(climbs) == 30
    for row in climbs:
        x, _ = row['spawn']['local_position_xy']
        assert -1.1 - 1e-12 <= x <= -0.9 + 1e-12
        expected = s2_spec.timeout_steps(task='stairs_walk', speed=row['speed'],
                                         stair_count=40, step_height=0.2,
                                         step_depth=0.2, approach_length=-x,
                                         post_distance=5.0)
        assert row['hard_timeout_steps'] == expected


def test_timeout_formula_and_flat_guards():
    assert s2_spec.timeout_steps(task='flat_walk', speed=0.5) == 1001
    assert s2_spec.flat_stop_schedule('instant')['hard_timeout_steps'] == 1401
    assert s2_spec.flat_stop_schedule('ramp_2s')['hard_timeout_steps'] == 1501
    assert s2_spec.timeout_steps(task='platform_walk', speed=0.3, post_distance=5.0) == 2167
    assert s2_spec.timeout_steps(task='platform_walk', speed=0.5, post_distance=5.0) == 1500
    assert s2_spec.timeout_steps(task='platform_walk', speed=0.6, post_distance=5.0) == 1334
    assert s2_spec.timeout_steps(task='stairs_walk', speed=0.5, stair_count=40,
                                 step_height=0.2, step_depth=0.2,
                                 approach_length=1.1, post_distance=5.0) == 4223
    assert s2_spec.timeout_steps(task='stairs_walk', speed=0.5, stair_count=40,
                                 step_height=0.2, step_depth=0.2,
                                 approach_length=1.1, post_distance=5.0,
                                 ramp_seconds=2.0, hold_seconds=20.0) == 5323
    with pytest.raises(ValueError):
        s2_spec.timeout_steps(task='flat_walk', speed=0.5, post_distance=5.0)


def test_q025_quality_window_constants_and_timeout_unchanged():
    assert s2_spec.PLATFORM_QUALITY_WINDOW_START_M == 1.0
    assert s2_spec.PLATFORM_QUALITY_WINDOW_END_M == 5.0
    assert s2_spec.PLATFORM_NOMINAL_QUALITY_WINDOW_M == 4.0
    assert s2_spec.CLIMB_QUALITY_WINDOW_OFFSET_M == 1.0
    # Sensitivity only; Q025 explicitly does not add these to the §4.2 timeout.
    assert [s2_spec.extra_first_meter_steps(v) for v in s2_spec.SPEEDS] == [334, 200, 167]
    assert s2_spec.timeout_steps(task='platform_walk', speed=0.3,
                                 post_distance=5.0) == 2167


def test_h_wrapper_retains_inner_source_and_timeouts(h_manifest):
    s2_cases.validate_h_series_manifest(h_manifest, ROOT)
    assert h_manifest['suite'] == 'S2_DIAG'
    assert h_manifest['status'] == 'frozen'
    assert h_manifest['lessons'] == ['H00', 'H01', 'H03', 'H06', 'H07', 'H10']
    assert len(h_manifest['records']) == 6
    expected = {
        'H00': dict(height=0.02, timeout=1301, x=2.2785239219665527, surface='platform'),
        'H01': dict(height=0.030833333333333334, timeout=1302, x=2.571112871170044, surface='platform'),
        'H03': dict(height=0.052500000000000005, timeout=1306, x=2.6570470333099365, surface='platform'),
        'H06': dict(height=0.085, timeout=1315, x=2.5360591411590576, surface='platform'),
        'H07': dict(height=0.09583333333333334, timeout=1318, x=1.0333220958709717, surface='stairs'),
        'H10': dict(height=0.12833333333333333, timeout=1332, x=1.203683853149414, surface='stairs'),
    }
    for record in h_manifest['records']:
        case = record['case']
        fact = expected[record['lesson']]
        assert record['source']['stage'] == 'S1'
        assert record['source']['suite'] == 'S1_DIAG'
        assert record['source']['original_path'] == record['path']
        assert record['source']['original_case_sha256'] == record['original_case_sha256']
        assert len(record['operation_id']) == 64
        assert record['inner_case_unchanged'] is True
        assert record['case_sha256_matches'] is True
        assert record['case_sha256_recomputed'] == case['case_sha256']
        assert record['hard_timeout_steps']['selected'] == 'original_v07_Q018'
        assert record['hard_timeout_steps']['original_v07'] == fact['timeout']
        assert record['hard_timeout_steps']['reserve_batch_padding'] == 9432
        assert case['case_id'] == 0
        assert case['stair_count'] == 6
        assert case['geometry']['step_depth'] == 0.30
        assert case['geometry']['width'] == 0.80
        assert case['geometry']['h'] == pytest.approx(fact['height'])
        assert case['geometry']['platform_length'] == 2.0
        assert case['speed'] == 0.5
        assert case['termination_contract'] == 'v07_original_stairs'
        position = record['original_failure_position']
        assert position['terminal_route_x'] == pytest.approx(fact['x'])
        assert position['surface'] == fact['surface']
        assert position['source_field'] == 'episode.forward_distance_end'
        assert position['platform_front_local_x'] == pytest.approx(1.8)
        assert record['replay']['terminal_platform_support'] is not None
        assert any('qpos_raw' in item for item in record['missing_diagnostic_fields'])
        assert any('rng_state_raw' in item for item in record['missing_diagnostic_fields'])


def test_h_wrapper_inner_case_is_not_mutated():
    import copy
    manifest = s2_cases.build_h_series_case0_manifest(ROOT)
    broken = copy.deepcopy(manifest)
    broken['records'][0]['case']['speed'] = 0.6
    with pytest.raises(ValueError):
        s2_cases.validate_h_series_manifest(broken, ROOT)
    record = manifest['records'][0]
    assert 'operation_id' not in record['case']
    assert 'source' not in record['case']
    assert record['case']['stage'] == 'S1'
    assert record['case']['suite'] == 'S1_DIAG'


def test_s2_terrain_rows_keep_real_wide_platform_bounds():
    rows = s2_cases.s2_terrain_rows()
    assert len(rows) == 3
    flat, narrow, wide = rows
    assert flat['kind'] == 'flat' and flat['N'] == 0
    assert flat['frame_origin_world'] == [0.0, 0.0, 0.0]
    for row, width in ((narrow, 0.8), (wide, 4.0)):
        assert row['kind'] == 's2_stairs'
        assert row['N'] == 40
        assert row['platform_length'] == 20.0
        assert row['platform_width'] == width
        assert row['platform_height'] == 8.0
        assert row['valid_bounds_local'][1] == 28.0
        assert row['frame_origin_world'][1] in (12.0, 28.0)
    assert narrow['frame_origin_world'] != wide['frame_origin_world']


def test_s1_constants_and_terrain_are_untouched():
    assert s1_spec.SDS_SHA256 == '500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976'
    assert original.TERRAIN_BY_LESSON['H00'].step_depth == 0.30
    from src.tasks.velocity.v09 import cases
    assert cases.GENERATOR == 'v09-s1-sha256-u53-v1'


def test_supersede_register_is_idempotent_and_lookupable(tmp_path):
    old_manifest = 'a' * 64
    old_file = 'b' * 64
    new_manifest = 'c' * 64
    new_file = 'd' * 64
    receipt_content = 'e' * 64
    receipt_file = 'f' * 64
    kwargs = dict(old_suite='S2_0', old_manifest_sha256=old_manifest,
                  old_file_sha256=old_file, new_suite='S2_DIAG',
                  new_manifest_sha256=new_manifest, new_file_sha256=new_file,
                  new_receipt_path='receipt.json',
                  new_receipt_content_sha256=receipt_content,
                  new_receipt_file_sha256=receipt_file,
                  reason='Q022 更名、旧清单零物理', ruling='DSH-Q026',
                  written_at='2026-11-22T00:00:00+00:00')
    first = s2_cases.register_supersede(tmp_path, **kwargs)
    assert first['already_registered'] is False
    second = s2_cases.register_supersede(tmp_path, **kwargs)
    assert second['already_registered'] is True
    lines = (tmp_path / s2_cases.SUPERSEDE_REGISTER).read_text().splitlines()
    assert len(lines) == 1
    record = __import__('json').loads(lines[0])
    assert record['old_suite'] == 'S2_0'
    assert record['new_suite'] == 'S2_DIAG'
    assert record['reason'] == 'Q022 更名、旧清单零物理'
    assert record['ruling'] == 'DSH-Q026'
    assert s2_cases.lookup_supersede(tmp_path, old_manifest)['new_manifest_sha256'] == new_manifest
    assert s2_cases.lookup_supersede(tmp_path, old_file)['new_receipt_content_sha256'] == receipt_content
    assert s2_cases.lookup_supersede(tmp_path, old_file)['new_receipt_file_sha256'] == receipt_file
    assert s2_cases.lookup_supersede(tmp_path, 'f' * 64) is None


def test_real_supersede_register_maps_old_freeze_to_new_receipt():
    root = ROOT
    record = s2_cases.lookup_supersede(root, s2_spec.S2_0_SUPERSEDES['manifest_sha256'])
    assert record is not None
    assert record['new_manifest_sha256'] == s2_cases.build_s2_0_manifest()['manifest_sha256']
    assert record['new_receipt_path'] == 'logs/a2_v09/s2_0/cases/freeze_receipt_S2_DIAG.json'
    assert record['new_receipt_content_sha256'] == (
        '5ec1c89732f71d0927ad3e947f38901978df453812b83ba59a26363f4161462f')
    assert record['new_receipt_file_sha256'] == (
        '6f80b6fe7a4f9373d4c57ff0b5350246880cc6db2dc3cd284a2f3bcdb0dc5e2e')


def test_history_projection_is_derived_only(tmp_path):
    old_manifest = 'a' * 64
    record = dict(old_suite='S2_0', old_manifest_sha256=old_manifest,
                  old_file_sha256='b' * 64, new_suite='S2_DIAG',
                  new_manifest_sha256='c' * 64, new_file_sha256='d' * 64,
                  new_receipt_path='r.json',
                  new_receipt_content_sha256='e' * 64,
                  new_receipt_file_sha256='f' * 64,
                  reason='Q022 更名、旧清单零物理', ruling='DSH-Q026',
                  written_at='2026-11-22T00:00:00+00:00')
    s2_cases.register_supersede(tmp_path, **record)
    projection = s2_cases.history_projection(tmp_path)
    assert projection['read_only_projection'] is True
    assert projection['entry_count'] == 1
    assert projection['entries'][0]['new_receipt_file_sha256'] == 'f' * 64
    digest = s2_cases.rebuild_history_projection(tmp_path)
    assert len(digest) == 64
    # A hand-made/non-derived index is refused by the only writer.
    with pytest.raises(ValueError):
        s2_cases.write_history_projection(tmp_path, {'entries': []})
    broken = dict(projection)
    broken['read_only_projection'] = False
    with pytest.raises(ValueError):
        s2_cases.write_history_projection(tmp_path, broken)
    # Rebuild is deterministic after the projection exists.
    assert s2_cases.rebuild_history_projection(tmp_path) == digest
