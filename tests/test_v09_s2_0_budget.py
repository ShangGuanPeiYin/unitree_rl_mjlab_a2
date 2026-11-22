"""CPU-only Q020/Q021 S2 reservation assertions; no ledger write or physics."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_budget, s2_spec

CASES = ROOT / 'logs/a2_v09/s2_0/cases'


@pytest.fixture(scope='module')
def manifest():
    return s2_cases.load_frozen_manifest(CASES / 'S2_DIAG.json', ROOT)


@pytest.fixture(scope='module')
def h_manifest():
    return s2_cases.load_frozen_manifest(CASES / 'h_series_case0.json', ROOT)


@pytest.fixture(scope='module')
def plan(manifest, h_manifest):
    return s2_budget.s2_stage_reservation_draft(manifest, h_manifest)


def test_new_s2_diag_freeze_uses_frozen_artifact_and_preserves_old(manifest):
    assert manifest['suite'] == 'S2_DIAG'
    assert manifest['manifest_sha256'] == (
        'fab7372ec20f44a8f0f8c2a75fe5f179c7beafc390ad22daaf73d8aa42c183a1')
    old = s2_cases.load_frozen_manifest(CASES / 'S2_0.json', ROOT, historical=True)
    assert old['suite'] == 'S2_0'
    assert old['manifest_sha256'] == (
        '5efda5abc35ad714dcf163b1648e2ac4ceac24edc12852889ee6cd3d3739acdf')
    assert manifest['supersedes']['manifest_sha256'] == old['manifest_sha256']
    run_plan = s2_cases.load_frozen_manifest(CASES / 'S2_DIAG_run_plan.json', ROOT) \
        if False else None
    # The run plan has its own validator; check the frozen bytes directly.
    import json
    plan_doc = json.loads((CASES / 'S2_DIAG_run_plan.json').read_text())
    assert plan_doc['native_trajectories'] == 150
    assert plan_doc['plan_sha256'] == (
        '3f09c0091f2ccaab8d87ae362f8fe1595c3e560dce55e4ab02a1fa0f3c98e126')


def test_s2_0_subcap_uses_new_75_case_manifest(manifest, h_manifest):
    matrix = s2_cases.s2_0_matrix_cost(manifest)
    assert matrix['real_branch_control_samples'] == 210_195
    assert matrix['plane_branch_control_samples'] == 210_195
    assert matrix['total_control_samples'] == 420_390
    assert matrix['batches'] == 30
    assert matrix['batch_size'] == 5
    assert matrix['native_trajectories'] == 150

    h_series = s2_cases.h_series_reservation(h_manifest)
    assert h_series['actual_timeout_sum'] == 7_874
    assert h_series['same_batch_upper_bound'] == 7_992
    assert h_series['actual_batch_control_samples'] == 7_992
    assert h_series['reserved_control_samples'] == 9_432

    draft = s2_cases.s2_0_reservation_draft(manifest, h_manifest)
    assert draft['total_control_samples_with_h'] == 429_822
    assert draft['remaining_nonlearning'] == 50_000_000 - 429_822
    assert draft['admitted'] is True


def test_gate_matrices_have_the_six_one_one_group_counts_and_cases():
    gates = s2_budget.gate_suite_costs()
    assert gates['S2.1']['groups'] == 36
    assert gates['S2.2']['groups'] == 33
    assert gates['S2.3']['groups'] == 51
    assert gates['S2.1']['cases'] == 36 * 20
    assert gates['S2.2']['cases'] == 33 * 20
    assert gates['S2.3']['cases'] == 51 * 20
    assert gates['S2.1']['control_samples'] == 1_620_000
    assert gates['S2.2']['control_samples'] == 2_061_980
    assert gates['S2.3']['control_samples'] == 4_348_500
    assert gates['S2.1']['breakdown']['flat'] == 234_180


def test_final_acceptance_has_27_groups_and_three_sets():
    final = s2_budget.final_acceptance_cost()
    assert final['groups'] == 27
    assert final['candidate_cases'] == 2_700
    assert final['candidate_samples'] == 13_249_500
    assert final['s1_reference_samples'] == 13_249_500
    assert final['original_6400_flat_samples'] == 1_170_900
    assert final['per_set_samples'] == 27_669_900
    assert final['three_sets_samples'] == 83_009_700


def test_l40_milestone_cache_is_a_single_frozen_comparison():
    cache = s2_budget.l40_milestone_cache_cost()
    assert cache['groups_per_suite'] == 3
    assert cache['suites'] == 2
    assert cache['cases_per_group'] == 20
    assert cache['per_suite_samples'] == 219_220
    assert cache['control_samples'] == 438_440
    assert 'P10/P50/P90' in cache['content']


def test_resource_probes_cover_both_1024_and_512():
    assert s2_budget.resource_probe_cost(1024, worst_timeout_steps=9472) == dict(
        num_envs=1024, warmup_and_measurement_rollouts=55,
        warmup_and_measurement_samples=2_703_360,
        evaluation_samples=947_200, video_samples=9_472,
        restore_rollout_samples=49_152, control_samples=3_709_184,
        full_admission=True)
    assert s2_budget.resource_probe_cost(512, worst_timeout_steps=9472)[
        'control_samples'] == 2_332_928
    with pytest.raises(ValueError):
        s2_budget.resource_probe_cost(2048, worst_timeout_steps=9472)


def test_stage_reservation_itemized_total_stays_under_four_hundred_million(plan):
    items = plan['items']
    assert sum(items.values()) == plan['total_control_samples']
    assert plan['total_control_samples'] == 263_888_370
    assert plan['total_control_samples'] <= s2_spec.S2_NONLEARNING_CAP
    assert plan['remaining_nonlearning'] == 400_000_000 - 263_888_370
    assert items['resource_probe_1024'] == 3_709_184
    assert items['resource_probe_512'] == 2_332_928
    assert items['s2_0_matrix_two_branches'] == 420_390
    assert items['h_series_baseline_supplement'] == 9_432
    assert items['h_series_candidate_retest_3_models'] == 28_296
    assert items['l40_milestone_frozen_cache_once'] == 438_440
    assert items['gate_admissions_up_to_40'] == 173_940_000
    assert items['final_sets_3_with_frozen_comparisons'] == 83_009_700
    assert plan['q021_reservation_written'] is False
    assert plan['admitted'] is True
    assert plan['s2_0_remaining'] == 50_000_000 - 429_822


def test_billing_and_probe_caliber_are_explicit(plan):
    billing = plan['billing']
    assert billing['padding_included'] is True
    assert billing['completed_slot_wait_steps_billed'] is True
    assert billing['s1_samples_not_charged_to_s2'] is True
    assert billing['physics_substeps'] == 4 * plan['total_control_samples']
    probe = billing['probe']
    assert probe['gpu_peak_mib_cap'] == 7168
    assert probe['warmups'] == 5
    assert probe['measurements'] == 50
    assert probe['restore_check'] == 1
    assert probe['evaluation_envs'] == 100
    assert probe['independent_copy'] is True


def test_learning_plan_reports_both_environment_counts():
    learning = s2_budget.learning_plan()
    rows = learning['courses']
    assert rows['S2_1']['rollouts_1024'] == 16_276
    assert rows['S2_1']['rollouts_512'] == 32_552
    assert rows['S2_2']['rollouts_1024'] == 24_414
    assert rows['S2_2']['rollouts_512'] == 48_828
    assert rows['S2_3']['rollouts_1024'] == 24_414
    assert rows['S2_3']['rollouts_512'] == 48_828
    assert learning['allocated_learning_cap'] == 3_200_000_000
    assert learning['stage_learning_cap'] == 3_600_000_000
    assert learning['unallocated_learning_cap'] == 400_000_000


def test_worst_gate_suite_is_s2_3_and_40_admissions_are_reserved(plan):
    assert plan['worst_gate_suite_samples'] == 4_348_500
    assert plan['gate_limit'] == 40
    assert plan['gates']['S2.3']['groups'] == 51
    assert plan['q025']['timeout_changed'] is False
    assert plan['q025']['sensitivity_extra_first_meter_steps'] == {
        '0.3': 334, '0.5': 200, '0.6': 167}
