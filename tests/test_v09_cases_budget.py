"""CPU-only frozen-input/budget tests; no simulator and no behavior evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec as original
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09 import cases, budget


@pytest.fixture(scope='module')
def manifests():
    return {suite: cases.build_s1_manifest(suite)
            for suite in cases.GATE_SUITES + cases.FINAL_SUITES}


@pytest.fixture(scope='module')
def plan(manifests):
    extras = {n: [dict(num_envs=n, control_steps=48, reason='save_restore_rollout'),
                  dict(num_envs=100, control_steps=5005, reason='L40_vx0.3_evaluation'),
                  dict(num_envs=1, control_steps=5005, reason='L40_vx0.3_video'),
                  dict(num_envs=n, control_steps=48, reason='restore_after_external_evidence')]
              for n in spec.ENV_CANDIDATES}
    return budget.s1_reservation_plan(manifests,
        calibration_manifest=cases.build_s1_manifest('train_calibration', calibration_cases_per_group=8),
        diagnostic_manifest=cases.build_s1_manifest('S1_DIAG'),
        probe_extra_costs=extras,
                                     engineering_extra_samples=1000)


def reserve(ledger, operation_id='op', **kwargs):
    arguments = dict(segment_id='segment', operation_id=operation_id, stage='S1',
        course='S1_RESOURCE', category='engineering', num_envs=20, max_control_steps=10,
        checkpoint_sha256='a' * 64, reason='CPU accounting fixture')
    arguments.update(kwargs)
    return ledger.reserve(**arguments)


def test_groups_retain_all_original_geometry_and_actual_timeouts():
    groups = cases.s1_groups()
    assert len(groups) == 75
    stairs = [row for row in groups if row['task'] == 'stairs']
    assert len(stairs) == 66
    assert {r['lesson'] for r in stairs} == {r.lesson_id for r in original.TERRAIN_TABLE if r.N}
    for row in stairs:
        tile = original.TERRAIN_BY_LESSON[row['lesson']]
        assert row['geometry']['platform_length'] == tile.platform_length == 2.
        assert row['geometry']['frame_origin_world'] == tile.frame_origin_world
        assert row['hard_timeout_steps'] == original.episode_step_limit(tile.tile_id, row['speed'])
    assert sum(row['hard_timeout_steps'] for row in groups) == 118225
    assert {r['hard_timeout_steps'] for r in groups if r['task'] != 'stairs'} == {1001, 1401, 1501}


def test_manifest_counts_stream_separation_and_full_spawn(manifests):
    all_ids = set()
    for suite, manifest in manifests.items():
        assert cases.validate_manifest(manifest)
        assert manifest['cases_per_group'] == (20 if suite in cases.GATE_SUITES else 100)
        assert len(manifest['cases']) == manifest['cases_per_group'] * 75
        ids = {identity for case in manifest['cases'] for identity in case['streams'].values()}
        assert len(ids) == 7 * len(manifest['cases'])
        assert all_ids.isdisjoint(ids)
        all_ids.update(ids)
        spawn = manifest['cases'][0]['spawn']
        assert spawn['local_position'][2] == .4
        assert len(spawn['quaternion_wxyz']) == 4
        assert spawn['joint_position'] == list(original.A2_DEFAULT_JOINT_POS)
    train = cases.build_s1_manifest('train_calibration', calibration_cases_per_group=1)
    assert all_ids.isdisjoint(identity for c in train['cases'] for identity in c['streams'].values())
    assert train['training_rules']['seed'] == 42
    with pytest.raises(ValueError):
        cases.build_s1_manifest('train_calibration')


def test_manifest_freeze_hash_and_access_control(tmp_path, manifests):
    path = tmp_path / 'final.json'
    final = manifests['final_1']
    cases.freeze_manifest(path, final)
    with pytest.raises(FileExistsError):
        cases.freeze_manifest(path, final)
    with pytest.raises(PermissionError):
        cases.load_manifest(path, purpose='training')
    with pytest.raises(PermissionError):
        cases.load_manifest(path, purpose='gate')
    assert cases.load_manifest(path, purpose='acceptance')['manifest_sha256'] == final['manifest_sha256']
    changed = deepcopy(final)
    changed['cases'][0]['spawn']['local_position'][0] += .01
    with pytest.raises(ValueError, match='SHA256'):
        cases.validate_manifest(changed)


def test_batch_cost_includes_finished_slot_wait_and_no_empty_padding(manifests):
    rows = deepcopy(manifests['gate_A']['cases'][:20])
    rows[0]['hard_timeout_steps'] = 1100
    result = budget.batch_worst_cost(rows, batch_size=20)
    assert result['control_samples'] == 20 * 1100
    assert result['physics_substeps'] == 4 * 20 * 1100
    assert budget.batch_worst_cost(rows, batch_size=10)['control_samples'] == 10 * (1100 + 1001)
    with pytest.raises(ValueError, match='equally divide'):
        budget.batch_worst_cost(rows, batch_size=8)


def test_stage_plan_real_cost_and_all_mandatory_obligations(plan):
    assert plan['admitted']
    assert plan['total_control_samples'] == 124142190 + 1000
    assert plan['baseline_subtotal'] == 41196500
    assert plan['candidate_gates'] == 26009500
    assert plan['candidate_finals'] == 35467500
    assert plan['original_6400_baseline'] == 40196500
    assert sum(p['rollout_samples'] for p in plan['probes']) == 13516800
    assert sum(v['control_samples'] for v in plan['obligations'].values()) == plan['total_control_samples']
    assert len([k for k in plan['obligations'] if k.startswith('candidate_gate')]) == 11
    assert len([k for k in plan['obligations'] if k.startswith('candidate_final')]) == 3
    assert all(len(p['extra_operations']) == 4 for p in plan['probes'])


def test_missing_probe_bound_and_overcap_refuse_admission(manifests):
    with pytest.raises(ValueError):
        budget.s1_reservation_plan(manifests, probe_extra_costs={}, engineering_extra_samples=0)
    calibration = cases.build_s1_manifest('train_calibration', calibration_cases_per_group=8)
    diagnostic = cases.build_s1_manifest('S1_DIAG')
    extras = {n: [] for n in spec.ENV_CANDIDATES}
    with pytest.raises(ValueError, match='must be explicit'):
        budget.s1_reservation_plan(manifests, calibration_manifest=calibration, diagnostic_manifest=diagnostic, probe_extra_costs=extras, engineering_extra_samples=0)
    extras = {n: [dict(num_envs=n, control_steps=48, reason='S1.2'),
                  dict(num_envs=100, control_steps=5005, reason='eval'),
                  dict(num_envs=1, control_steps=5005, reason='video'),
                  dict(num_envs=n, control_steps=48, reason='restore')] for n in spec.ENV_CANDIDATES}
    assert not budget.s1_reservation_plan(manifests, calibration_manifest=calibration, diagnostic_manifest=diagnostic, probe_extra_costs=extras,
                                         engineering_extra_samples=40_000_000)['admitted']


def test_no_physics_before_stage_atomic_reservation(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    for category in ('learning', 'engineering', 'evaluation'):
        with pytest.raises(ValueError, match='full stage plan'):
            reserve(ledger, category, category=category, course='S1.1')
    assert ledger.snapshot()['total_control_samples'] == 0
    ledger.reserve_stage(plan)
    with pytest.raises(ValueError, match='already admitted'):
        ledger.reserve_stage(plan)


def test_account_partial_unknown_duplicate_and_checkpoint_no_rewind(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    checkpoint = ledger.checkpoint_metadata()
    operation = reserve(ledger)
    ledger.commit(operation, actual_control_steps=3, status='interrupted')
    with pytest.raises(ValueError, match='already settled'):
        ledger.commit(operation, actual_control_steps=3)
    with pytest.raises(ValueError, match='already exists'):
        reserve(ledger)
    unknown = reserve(ledger, 'unknown')
    ledger.commit(unknown, status='unknown')
    state = ledger.validate_checkpoint(checkpoint)
    assert state['total_control_samples'] == 20 * (3 + 10)
    assert state['operations'][unknown]['physics_substeps'] == 800
    ledger.advance_state('S1', gate_attempts=2, final_suites_used=1, final_recoveries=1,
                         next_gate_threshold=200000000, episode_flat=97)
    with pytest.raises(ValueError, match='cannot rewind'):
        ledger.advance_state('S1', gate_attempts=1)
    assert ledger.validate_checkpoint(checkpoint)['stages']['S1']['progress']['gate_attempts'] == 2
    assert ledger.snapshot()['stages']['S1']['next_gate_suite'] == 'gate_A'
    reopened = budget.AppendOnlyLedger(ledger.path, run_id='run')
    assert reopened.snapshot()['total_control_samples'] == 260


def test_future_finals_protected_from_engineering_and_unexecuted_release(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    with pytest.raises(ValueError, match='obligation exhausted'):
        reserve(ledger, max_control_steps=51)
    with pytest.raises(ValueError, match='unexecuted'):
        ledger.close_obligation('S1', 'candidate_final_1', reason='not actually run')
    with pytest.raises(ValueError, match='different accounting category'):
        reserve(ledger, obligation_id='candidate_final_1')
    assert ledger.snapshot()['stages']['S1']['obligations']['candidate_final_1']['limit'] == 11822500


def test_eval_actual_case_coverage_cannot_be_duplicated_to_release(tmp_path, plan, manifests):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    manifest = manifests['gate_A']
    batch = manifest['cases'][:20]
    arguments = dict(category='evaluation', obligation_id='candidate_gate_01',
        case_sha256=manifest['manifest_sha256'], case_sha256s=[r['case_sha256'] for r in batch],
        num_envs=20, max_control_steps=1001)
    first = reserve(ledger, **arguments)
    with pytest.raises(ValueError, match='in flight'):
        reserve(ledger, 'same_inflight', **arguments)
    ledger.commit(first, actual_control_steps=900)
    with pytest.raises(ValueError, match='already completed'):
        reserve(ledger, 'same_completed', **arguments)
    with pytest.raises(ValueError, match='incomplete'):
        ledger.close_obligation('S1', 'candidate_gate_01', reason='single batch is insufficient')
    with pytest.raises(ValueError, match='manifest SHA'):
        reserve(ledger, 'wrong_manifest', **dict(arguments, case_sha256='bad'))
    ledger.invalidate_obligation('S1', 'candidate_gate_01', reason='engineering fault invalidated entire suite')
    assert not ledger.snapshot()['stages']['S1']['obligations']['candidate_gate_01']['completed_cases']
    ledger.extend_stage('S1', 18000, reason='within-stage engineering remainder')
    ledger.fund_obligation_retry('S1', 'candidate_gate_01', 18000, reason='full-suite engineering rerun')
    retry = reserve(ledger, 'retry', **arguments)
    ledger.commit(retry, actual_control_steps=901)
    assert ledger.snapshot()['total_control_samples'] == 36020


def test_unknown_probe_is_charged_but_not_completion_evidence(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    key = reserve(ledger, obligation_id='probe_512', num_envs=512, max_control_steps=48,
                  work_item_id='rollout_001')
    ledger.commit(key, status='unknown')
    duty = ledger.snapshot()['stages']['S1']['obligations']['probe_512']
    assert duty['spent'] == 24576
    assert not duty['completed_items']
    with pytest.raises(ValueError):
        ledger.close_obligation('S1', 'probe_512', reason='unknown cannot finish probe')
    duty = ledger.snapshot()['stages']['S1']['obligations']['probe_512']
    assert not duty['closed'] and not duty['completed_items']
    assert duty['spent'] == 24576


def test_course_caps_cannot_borrow_final_recovery(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    with pytest.raises(ValueError, match='allocation required'):
        reserve(ledger, category='learning', course='S1.2', num_envs=512, max_control_steps=48)
    with pytest.raises(ValueError, match='allocation exhausted'):
        reserve(ledger, category='learning', course='S1.2', allocation_id='S1.2_main',
                num_envs=1, max_control_steps=630000001)
    key = reserve(ledger, category='learning', course='S1.1', num_envs=512, max_control_steps=48)
    ledger.commit(key, actual_control_steps=20, status='interrupted')
    assert ledger.snapshot()['stages']['S1']['allocations']['S1.1'] == 512 * 20


def test_atomic_competing_reservations_do_not_overspend(tmp_path, plan):
    path = tmp_path / 'ledger.jsonl'
    ledger = budget.AppendOnlyLedger(path, run_id='run')
    ledger.reserve_stage(plan)

    def attempt(index):
        try:
            reserve(budget.AppendOnlyLedger(path, run_id='run'), str(index), max_control_steps=30)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = list(pool.map(attempt, range(2)))
    assert sum(accepted) == 1  # each reserves 600 from a 1000-sample engineering duty
    assert ledger.snapshot()['stages']['S1']['reserved']['engineering'] == 600


def test_hash_chain_and_partial_tail_fail_closed(tmp_path, plan):
    path = tmp_path / 'ledger.jsonl'
    ledger = budget.AppendOnlyLedger(path, run_id='run')
    ledger.reserve_stage(plan)
    with path.open('a') as handle:
        handle.write('{"unfinished":')
    with pytest.raises(ValueError, match='incomplete ledger tail'):
        ledger.snapshot()


def test_training_role_rounding_and_episode_stream_independent_of_N():
    for count in (2048, 1536, 1024, 512):
        allocated = cases.allocate_training_roles(count)
        assert sum(allocated.values()) == count
        for role, weight in zip(cases.TRAINING_ROLES, (.2, .2, .3, .3)):
            assert abs(allocated[role] - count * weight) < 1
    assert cases.allocate_training_roles(2048) == dict(flat_walk=410, flat_stop=410,
        H00_H15=614, D25_D22_D20_L12_L20_L40=614)
    for role in cases.TRAINING_ROLES:
        saved = cases.training_case(role, 93)
        for count in (2048, 512):
            cases.allocate_training_roles(count)
            assert cases.training_case(role, 93) == saved
        assert cases.training_case(role, 94)['streams'] != saved['streams']
    for role in cases.TRAINING_ROLES:
        seen = [cases.training_case(role, i) for i in range(300)]
        assert {row['speed'] for row in seen} == {.3, .5, .6}
        if role == 'H00_H15':
            assert {row['lesson'] for row in seen} == {f'H{i:02d}' for i in range(16)}
        if role == 'flat_stop':
            assert {row['stop_mode'] for row in seen} == {'instant', 'ramp_2s'}


def test_fixed_evaluation_candidate_and_probe_complete_duration(tmp_path, plan, manifests):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    manifest = manifests['gate_A']
    first_batch = manifest['cases'][:20]
    args = dict(category='evaluation', obligation_id='candidate_gate_01',
        case_sha256=manifest['manifest_sha256'], num_envs=20, max_control_steps=1001)
    first = reserve(ledger, case_sha256s=[r['case_sha256'] for r in first_batch], **args)
    ledger.commit(first, actual_control_steps=900)
    with pytest.raises(ValueError, match='frozen checkpoint SHA'):
        reserve(ledger, 'other_model', checkpoint_sha256='b' * 64,
                case_sha256s=[r['case_sha256'] for r in manifest['cases'][20:40]], **args)
    key = reserve(ledger, 'probe', obligation_id='probe_512', num_envs=512,
                  max_control_steps=48, work_item_id='rollout_001')
    with pytest.raises(ValueError, match='full planned'):
        ledger.commit(key, actual_control_steps=1)
    ledger.commit(key, actual_control_steps=48)
    assert ledger.snapshot()['stages']['S1']['obligations']['probe_512']['completed_items'] == {'rollout_001': True}
    evaluation = reserve(ledger, 'probe_eval', obligation_id='probe_512', num_envs=100,
                         max_control_steps=5005, work_item_id='extra_002')
    ledger.commit(evaluation, actual_control_steps=900)
    assert ledger.snapshot()['stages']['S1']['obligations']['probe_512']['completed_items']['extra_002']


def test_s10_subcap_and_abandoned_segment_recovery(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    ledger.extend_stage('S1', 10_000_000, reason='available stage remainder')
    with pytest.raises(ValueError, match='S1.0'):
        ledger.fund_obligation_retry('S1', 'calibration', 9_000_000,
                                     reason='would exceed baseline subcap')
    key = reserve(ledger, 'abandoned')
    assert ledger.settle_pending_unknown(segment_id='segment', reason='process died') == [key]
    assert ledger.snapshot()['total_control_samples'] == 200
    assert ledger.settle_pending_unknown(segment_id='segment', reason='idempotent review') == []


def test_q005_unattempted_probe_release_is_not_resource_success(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    before = ledger.snapshot()['stages']['S1']
    release = before['obligations']['probe_1536']['limit']
    ledger.skip_unattempted_probe('S1', 'probe_1536', reason='not_attempted; larger candidate selected',
                                 evidence_sha256='e' * 64)
    state = ledger.snapshot()['stages']['S1']
    assert state['obligations']['probe_1536']['skipped']
    assert not state['obligations']['probe_1536']['completed_items']
    assert state['obligations']['engineering']['limit'] == 1000 + release
    assert state['envelope'] == before['envelope']
    assert state['spent'] == before['spent']
    key = reserve(ledger, 'failed_probe', obligation_id='probe_512', num_envs=512,
                  max_control_steps=48, work_item_id='rollout_001')
    ledger.commit(key, actual_control_steps=7, status='interrupted')
    with pytest.raises(ValueError, match='never-attempted'):
        ledger.skip_unattempted_probe('S1', 'probe_512', reason='attempted must stay charged')
    assert ledger.snapshot()['total_control_samples'] == 512 * 7
    ledger.fail_probe_obligation('S1', 'probe_512', reason='resource check failed',
                                 evidence_sha256='f' * 64)
    duty = ledger.snapshot()['stages']['S1']['obligations']['probe_512']
    assert duty['failed'] and duty['closed']
    assert duty['limit'] == duty['spent'] == 512 * 7
    assert not duty['completed_items']
    assert ledger.snapshot()['total_control_samples'] == 512 * 7


def test_stage_cap_and_resource_course_independent_s10_cap(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    remainder = plan['remaining_nonlearning']
    ledger.extend_stage('S1', remainder, reason='all remaining same-stage engineering capacity')
    assert ledger.snapshot()['stages']['S1']['envelope'] == 150000000
    with pytest.raises(ValueError, match='budget exhausted'):
        ledger.extend_stage('S1', 1, reason='would exceed stage cap')
    with pytest.raises(ValueError, match='S1_RESOURCE'):
        reserve(ledger, 'wrong_probe_course', course='S1.0', obligation_id='probe_2048',
                num_envs=2048, max_control_steps=48, work_item_id='rollout_001')
    key = reserve(ledger, 'resource', obligation_id='probe_2048', num_envs=2048,
                  max_control_steps=48, work_item_id='rollout_001')
    ledger.commit(key, actual_control_steps=48)
    state = ledger.snapshot()['stages']['S1']
    assert state['baseline_commitment'] == 41196500
    assert state['spent']['engineering'] == 98304
    assert state['obligations']['probe_2048']['spent'] == 98304
    assert ledger.snapshot()['total_control_samples'] == 98304  # parent reservation not double billed


def test_calibration_allowance_separate_from_complete_frozen_target(tmp_path, plan):
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    calibration = cases.build_s1_manifest('train_calibration', calibration_cases_per_group=8)
    duty = ledger.snapshot()['stages']['S1']['obligations']['calibration']
    assert duty['limit'] == 1000000 and duty['target'] == 945800
    assert len(duty['expected_cases']) == 600
    batches = budget.batch_worst_cost(calibration['cases'], batch_size=8)['batches']
    # Shorten the fixture to one entire fixed group while retaining actual
    # complete-coverage logic; the production manifest above remains 600 cases.
    small = deepcopy(plan)
    keep = set(batches[0]['case_sha256s'])
    small['case_catalogue']['train_calibration'] = {k: v for k, v in
        small['case_catalogue']['train_calibration'].items() if k in keep}
    small['obligations']['calibration']['target_control_samples'] = batches[0]['control_samples']
    small['plan_sha256'] = cases.sha256({k: v for k, v in small.items() if k != 'plan_sha256'})
    test_ledger = budget.AppendOnlyLedger(tmp_path / 'small.jsonl', run_id='small')
    test_ledger.reserve_stage(small)
    batch = batches[0]
    key = test_ledger.reserve(segment_id='cal', operation_id='batch', stage='S1', course='S1.0',
        category='engineering', num_envs=8, max_control_steps=batch['control_steps'],
        checkpoint_sha256=spec.SOURCE_SHA256, reason='fixed training calibration',
        case_sha256=calibration['manifest_sha256'], obligation_id='calibration',
        case_sha256s=batch['case_sha256s'])
    test_ledger.commit(key, actual_control_steps=900)
    test_ledger.close_obligation('S1', 'calibration', reason='all fixed cases covered')
    state = test_ledger.snapshot()['stages']['S1']
    assert state['obligations']['calibration']['closed']
    assert state['obligations']['calibration']['spent'] == 7200
    assert state['obligations']['engineering']['limit'] == 1000 + 1000000 - 7200


def test_q008_diagnostic_stream_and_evidence_obligations(tmp_path, plan, manifests):
    diag = cases.build_s1_manifest('S1_DIAG')
    assert len(diag['cases']) == 1500
    ids = {value for row in diag['cases'] for value in row['streams'].values()}
    assert all(ids.isdisjoint(value for row in manifest['cases'] for value in row['streams'].values())
               for manifest in manifests.values())
    assert plan['diagnostic_samples'] == 4729000
    assert plan['video_samples'] == 709350
    ledger = budget.AppendOnlyLedger(tmp_path / 'ledger.jsonl', run_id='run')
    ledger.reserve_stage(plan)
    with pytest.raises(ValueError, match='S1_EVIDENCE'):
        reserve(ledger, obligation_id='diag_before', max_control_steps=1001, num_envs=20,
                case_sha256=diag['manifest_sha256'],
                case_sha256s=[row['case_sha256'] for row in diag['cases'][:20]])
    duty = ledger.snapshot()['stages']['S1']['obligations']['video_init']
    assert len(duty['expected_cases']) == 75 and duty['limit'] == 118225
    assert ledger.snapshot()['stages']['S1']['baseline_commitment'] == 41196500


def test_diagnostic_and_video_roles_cannot_unseal_any_final_suite(tmp_path, manifests):
    for suite in cases.FINAL_SUITES:
        path = tmp_path / (suite + '.json')
        cases.freeze_manifest(path, manifests[suite])
        for purpose in ('diagnostic', 'evidence'):
            with pytest.raises(PermissionError, match='sealed'):
                cases.load_manifest(path, purpose=purpose)
    path = tmp_path / 'diag.json'
    cases.freeze_manifest(path, cases.build_s1_manifest('S1_DIAG'))
    assert cases.load_manifest(path, purpose='diagnostic')['suite'] == 'S1_DIAG'
    assert cases.load_manifest(path, purpose='evidence')['suite'] == 'S1_DIAG'


def test_q009_resource_layout_and_true_total(plan):
    assert sum(row['control_samples'] for row in plan['probes']) == 16030340
    assert plan['total_control_samples'] == 124142190 + 1000
    assert plan['remaining_nonlearning'] == 25857810 - 1000
    for n in spec.ENV_CANDIDATES:
        work = plan['obligations'][f'probe_{n}']['work_items']
        assert len(work) == 59
        assert [(work[f'extra_{i:03d}']['num_envs'], work[f'extra_{i:03d}']['control_steps'])
                for i in range(1,5)] == [(n,48),(100,5005),(1,5005),(n,48)]
        assert work['extra_001']['require_full_steps']
        assert work['extra_004']['require_full_steps']
