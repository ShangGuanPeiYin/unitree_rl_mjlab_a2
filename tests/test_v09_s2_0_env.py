"""CPU-only S2 adapter/evidence tests; no simulator or env.step."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_env


def case_for(task):
    manifest = s2_cases.build_s2_0_manifest()
    return next(row for row in manifest['cases'] if row['task'] == task)


def frame(step, *, task='stairs_walk', top=False, four=False, failure=False,
          sample=None):
    return dict(
        step=step, route_x=1.0 + 0.1 * step, base_xy=[1.0 + 0.1 * step, 0.0],
        feet_xy=[[1.0, 0.1], [1.0, -0.1], [0.9, 0.1], [0.9, -0.1]],
        feet_route_x=[1.0, 1.0, 0.9, 0.9], pitch=0.0,
        foot_on_platform=[False, False, False, False],
        foot_contact=[True, True, False, False], g_level=0.0,
        top_arrived=top, failure=failure, stop_request=False, terminal=False,
        c_exec=0.5, c_request=0.5, a_request=[0.0] * 12,
        a_applied=[0.0] * 12, velocity_route=0.5, pitch_rate=0.0,
        foot_normal_force=[10.0, 10.0, 0.0, 0.0],
        contact_pairs=[dict(foot=0, geom='tread_1', normal_force=10.0,
                            duration_s=0.02, position_xy=[1.0, 0.1],
                            surface='tread', valid=True)],
        pd_target=[0.1] * 12, joint_position=[0.1] * 12,
        joint_velocity=[0.0] * 12,
        map_valid=np.ones((40, 30), dtype=bool), map_sample_time_s=0.4,
        native_sample_id=sample)


def test_adapter_builds_all_three_s2_0_tasks():
    for task in ('flat_walk', 'stairs_walk', 'platform_walk'):
        adapter = s2_env.S2TaskAdapter.from_case(case_for(task))
        machine = adapter.state_machine()
        gate = adapter.level_gate()
        assert adapter.config.timeout_steps == case_for(task)['hard_timeout_steps']
        assert machine.config.task == task
        assert gate.state_dict()['value'] == (1.0 if task == 'flat_walk' else 0.0)
    platform = s2_env.S2TaskAdapter.from_case(case_for('platform_walk'))
    assert platform.platform['front_x'] == pytest.approx(8.0)
    assert platform.platform['y_half_width'] in (0.4, 2.0)
    assert platform.zero_point() == pytest.approx(9.0)
    stairs = s2_env.S2TaskAdapter.from_case(case_for('stairs_walk'))
    assert stairs.platform['top_height'] == pytest.approx(8.0)
    assert stairs.zero_point() is None


def test_adapter_accepts_original_h_geometry():
    record = s2_cases.build_h_series_case0_manifest(ROOT)['records'][0]
    adapter = s2_env.S2TaskAdapter.from_case(record['case'])
    assert adapter.terrain.terrain_key.startswith('legacy_H00')
    assert adapter.config.timeout_steps == 1301
    assert adapter.config.post_distance is None
    assert adapter.config.legacy_h is True
    assert adapter.platform['front_x'] == pytest.approx(1.8)


def test_evidence_recorder_roundtrip_and_contract(tmp_path):
    case = case_for('stairs_walk')
    adapter = s2_env.S2TaskAdapter.from_case(case)
    recorder = s2_env.S2EvidenceRecorder(case=case, adapter=adapter)
    sample = dict(refresh_index=4, sample_sim_time_s=0.4, produced_at_actor_step=2)
    recorder.append(frame(0, sample=sample))
    recorder.append(frame(1, top=True, four=True, sample=sample))
    assert recorder.validate()['rows'] == 2
    trace = recorder.trace()
    assert trace['feet_xy'].shape == (2, 4, 2)
    assert trace['contact_pairs'][0][0]['surface'] == 'tread'
    written = recorder.write(tmp_path)
    assert Path(written['npz']).exists() and Path(written['metadata']).exists()
    metadata = json.loads(Path(written['metadata']).read_text())
    assert metadata['rows'] == 2
    assert metadata['native_samples'][0]['actor_step'] == 0
    assert metadata['contract']['required_fields'] == s2_env.evidence_contract()['required_fields']
    with pytest.raises(FileExistsError):
        recorder.write(tmp_path)


def test_evidence_recorder_rejects_missing_fields_and_bad_pairs():
    case = case_for('flat_walk')
    adapter = s2_env.S2TaskAdapter.from_case(case)
    recorder = s2_env.S2EvidenceRecorder(case=case, adapter=adapter)
    broken = frame(0)
    broken.pop('map_sample_time_s')
    with pytest.raises(ValueError):
        recorder.append(broken)
    bad_pair = frame(0)
    bad_pair['contact_pairs'] = [dict(foot=0)]
    with pytest.raises(ValueError):
        recorder.append(bad_pair)
    with pytest.raises(ValueError):
        recorder.trace()


def test_case_record_carries_metrics_and_map_evidence():
    case = case_for('flat_walk')
    adapter = s2_env.S2TaskAdapter.from_case(case)
    recorder = s2_env.S2EvidenceRecorder(case=case, adapter=adapter)
    recorder.append(frame(0))
    record = s2_env.case_record(case=case, adapter=adapter, recorder=recorder,
                                metrics={'quality_pass': True},
                                map_evidence={'branch': 'real'},
                                state_summary={'state': 'POST_WALK'},
                                failure_context={'present': False})
    assert record['case_sha256'] == case['case_sha256']
    assert record['metrics']['quality_pass'] is True
    assert record['map_evidence']['branch'] == 'real'
    assert len(record['config_sha256']) == 64
    assert len(record['evidence_contract_sha256']) == 64


def test_evidence_contract_contains_metric_and_attribution_fields():
    contract = s2_env.evidence_contract()
    for key in ('feet_xy', 'feet_route_x', 'foot_contact', 'contact_pairs',
                'pd_target', 'joint_position', 'map_valid',
                'map_sample_time_s', 'native_sample_id', 'c_exec'):
        assert key in contract['required_fields']
    assert contract['native_sample_id'] == [
        'refresh_index', 'sample_sim_time_s', 'produced_at_actor_step']


def test_branch_plan_pairs_every_case_and_shares_prefix():
    from src.tasks.velocity.v09 import s2_cases
    matrix = s2_cases.build_s2_0_manifest()
    plan = s2_cases.build_s2_diag_run_plan(matrix)
    branch = s2_env.branch_execution_plan(matrix, plan)
    assert branch['pair_count'] == 75
    assert branch['pairs'][0]['real_operation_id'] != branch['pairs'][0]['plane_operation_id']
    assert branch['pairs'][0]['sample_streams'] == branch['pairs'][0]['sample_streams']


def test_branch_prefix_proof_is_elementwise():
    trace = {key: np.arange(6) for key in ('step', 'route_x', 'g_level')}
    identical = {key: value.copy() for key, value in trace.items()}
    assert len(s2_env.assert_branches_identical_until(
        trace, identical, end_index=4)) == 64
    changed = {key: value.copy() for key, value in trace.items()}
    changed['route_x'][2] = 99
    with pytest.raises(ValueError):
        s2_env.assert_branches_identical_until(trace, changed, end_index=4)
    assert s2_env.assert_branches_identical_until(
        trace, changed, end_index=2) == s2_env.pre_intervention_digest(trace, end_index=2)


def test_b_v5_2_all_six_h_adapters_keep_original_timeout_and_legacy_mode():
    records = s2_cases.build_h_series_case0_manifest(ROOT)['records']
    assert [record['case']['hard_timeout_steps'] for record in records] == [
        1301, 1302, 1306, 1315, 1318, 1332]
    for record in records:
        adapter = s2_env.S2TaskAdapter.from_case(record['case'])
        assert adapter.config.legacy_h is True
        assert adapter.config.timeout_steps == record['case']['hard_timeout_steps']
        assert adapter.config.post_distance is None
