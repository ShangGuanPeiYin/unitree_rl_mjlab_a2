"""CPU-only Q016/Q023/Q024 plane ablation tests; no simulator."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_map_ablation as ablation


def make_map(valid=None):
    real = np.zeros((3, 4, 4))
    real[0] = 0.25
    real[1] = 1.0
    real[2] = 0.0
    if valid is None:
        valid = np.ones((4, 4), dtype=bool)
    real[0][~valid] = 0.0
    real[1][~valid] = 0.0
    real[2][~valid] = 1.0
    return real, valid


def make_native_map():
    real = np.zeros((3, 40, 30))
    real[0] = 0.25
    real[1] = 1.0
    real[2] = 0.0
    return real, np.ones((40, 30), dtype=bool)


def test_only_valid_cell_height_changes_and_invalid_is_0_0_1():
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    valid[3, 3] = False
    real, valid = make_map(valid)
    out, info = ablation.apply_plane_height_channel(real, valid, 8.0, 8.2)
    raw = (8.0 - 8.2) / 1.5
    assert out[0, 1, 1] == pytest.approx(np.clip(raw, -1, 1))
    assert out[0, 0, 0] == 0.0
    assert out[1, 0, 0] == 0.0
    assert out[2, 0, 0] == 1.0
    assert np.all(out[1, 1:, :] == real[1, 1:, :])
    assert np.all(out[2, 1:, :] == real[2, 1:, :])
    assert info['channels_unchanged'] == ('valid', 'age')
    assert info['invalid_cells'] == '0/0/1'
    assert info['replaced_cells'] == 14
    assert info['valid_cells'] == 14
    assert info['valid_fraction'] == pytest.approx(14 / 16)


def test_q023_saturation_and_exact_clip_use_fixed_thresholds():
    real, valid = make_map()
    out, info = ablation.apply_plane_height_channel(real, valid, 12.0, 8.0)
    assert np.all(out[0] == 1.0)
    assert info['saturation_numerator'] == 16
    assert info['saturation_denominator'] == 16
    assert info['saturation'] == 1.0
    assert info['exact_clip_numerator'] == 16
    assert info['exact_clip_fraction'] == 1.0
    # |h_input| = 0.9995: saturation yes, exact clip no (raw < 1).
    plane = 8.0 + 1.5 * 0.9995
    out, info = ablation.apply_plane_height_channel(real, valid, plane, 8.0)
    assert np.allclose(out[0], 0.9995)
    assert info['saturation_numerator'] == 16
    assert info['saturation'] == 1.0
    assert info['exact_clip_numerator'] == 0
    assert info['exact_clip_fraction'] == 0.0


def test_validity_and_age_are_not_replaced_even_when_stale():
    valid = np.ones((4, 4), dtype=bool)
    real, valid = make_map(valid)
    real[2] = 0.6
    out, info = ablation.apply_plane_height_channel(real, valid, 8.0, 8.0)
    assert np.all(out[2] == 0.6)
    assert np.all(out[1] == 1.0)
    assert info['replaced_cells'] == 16


def test_replace_mask_limits_height_cells():
    real, valid = make_map()
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 2] = True
    out, info = ablation.apply_plane_height_channel(real, valid, 8.0, 8.0,
                                                    replace_mask=mask)
    assert info['replaced_cells'] == 1
    assert out[0, 2, 2] == 0.0
    assert out[0, 0, 0] == 0.25


def test_q024_trigger_frame_uses_real_and_next_fresh_step_replaces():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    out, info = switch.observe(frame_index=0, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=True)
    assert info['active'] is False
    assert np.array_equal(out, real)
    out, info = switch.observe(frame_index=1, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=True,
                               front_foot_top_support=True)
    assert info['trigger_frame'] == 1
    assert info['trigger_frame_uses_real'] is True
    assert info['active'] is False           # Q024: t must consume the real map
    assert np.array_equal(out, real)
    out, info = switch.observe(frame_index=2, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=True)
    assert info['active'] is True
    assert info['trigger_frame_uses_real'] is False
    assert switch.first_consumed_replacement_frame == 2
    assert switch.first_consumed_replacement_frame - switch.trigger_frame == 1
    assert np.allclose(out[0], (8.0 - 8.2) / 1.5)


def test_q024_holds_until_a_normal_refresh_after_the_trigger():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True)
    out, info = switch.observe(frame_index=1, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=False,
                               front_foot_top_support=True)
    assert info['active'] is False and np.array_equal(out, real)
    out, info = switch.observe(frame_index=2, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=False)
    assert info['active'] is False and np.array_equal(out, real)
    out, info = switch.observe(frame_index=3, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=True)
    assert info['active'] is True
    assert switch.first_consumed_replacement_frame == 3
    assert switch.first_consumed_replacement_frame - switch.trigger_frame == 2


def test_q024_terminal_before_exposure_is_reported_not_fabricated():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True)
    out, info = switch.observe(frame_index=1, real_map=real, valid=valid,
                               base_sample_z=8.2, refreshed=True,
                               front_foot_top_support=True, terminal=True)
    assert info['trigger_frame_uses_real'] is True
    summary = switch.summary()
    assert summary['exposed'] is False
    assert summary['first_consumed_replacement_frame'] is None
    assert summary['note'] == 'not_exposed_before_terminal'


def test_q024_birth_frame_exposure_for_platform_and_flat():
    real, valid = make_native_map()
    for task, plane in (('platform_walk', 8.0), ('flat_walk', 0.0)):
        switch = ablation.S2PlaneAblation(task=task, plane_world_z=plane)
        out, info = switch.observe(frame_index=0, real_map=real, valid=valid,
                                   base_sample_z=plane + 0.4, refreshed=True)
        assert info['active'] is True
        assert switch.first_consumed_replacement_frame == 0
        assert np.allclose(out[0], (plane - (plane + 0.4)) / 1.5)


def test_nonfresh_frames_reuse_the_same_policy_map_and_count_q023_frames():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    first, info = switch.observe(frame_index=0, real_map=real, valid=valid,
                                 base_sample_z=0.4, refreshed=True)
    second, info = switch.observe(frame_index=1, real_map=real, valid=valid,
                                  base_sample_z=0.4, refreshed=False)
    assert np.array_equal(first, second)
    summary = switch.summary()
    assert summary['frames_consumed'] == 2
    assert summary['valid_fraction_numerator'] == 2 * 40 * 30
    assert summary['valid_fraction_denominator'] == 2 * 40 * 30
    assert summary['valid_fraction'] == 1.0
    assert summary['saturation_denominator'] == 2 * 40 * 30


def test_q023_group_aggregation_sums_numerators_and_handles_na():
    real, valid = make_native_map()
    first = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    first.observe(frame_index=0, real_map=real, valid=valid,
                  base_sample_z=0.4, refreshed=True, terminal=True)
    second = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    second.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=True, terminal=True)
    aggregate = ablation.aggregate_evidence([first.summary(), second.summary()])
    assert aggregate['valid_fraction_denominator'] == 2 * 40 * 30
    assert aggregate['valid_fraction_numerator'] == 2 * 40 * 30
    assert aggregate['saturation_denominator'] == 2 * 40 * 30
    assert aggregate['not_exposed'] == 0
    empty = np.zeros((40, 30), dtype=bool)
    empty_real = real.copy()
    empty_real[0] = 0.0
    empty_real[1] = 0.0
    empty_real[2] = 1.0
    zero = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    zero.observe(frame_index=0, real_map=empty_real, valid=empty,
                 base_sample_z=0.4, refreshed=True, terminal=True)
    aggregate = ablation.aggregate_evidence([zero.summary()])
    assert aggregate['valid_fraction'] == 0.0
    assert aggregate['saturation'] is None
    assert aggregate['exact_clip_fraction'] is None
    assert aggregate['zero_saturation_denominator'] == 1


def test_q024_pairing_and_config_evidence_fields():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    pairing = dict(case_sha256='a' * 64, stream_sha256='e' * 64,
                   weight_sha256='b' * 64, code_sha256='c' * 64,
                   physics_sha256='f' * 64, reward_sha256='0' * 64,
                   termination_sha256='1' * 64, state_sha256='d' * 64,
                   command=[0.5, 0.0, 0.0], action=[0.0] * 12,
                   sample_frame=17)
    out, info = switch.observe(frame_index=0, real_map=real, valid=valid,
                               base_sample_z=0.4, refreshed=True,
                               terminal=True, pairing=pairing)
    summary = switch.summary()
    assert len(summary['config_sha256']) == 64
    assert len(summary['input_pairing_sha256']) == 64
    assert len(summary['pre_intervention_pairing_sha256']) == 64
    assert summary['paired_proof']['passed'] is True
    assert summary['paired_proof']['valid_channel_unchanged'] is True
    assert summary['paired_proof']['age_channel_unchanged'] is True
    assert summary['paired_proof']['only_valid_height_changed'] is True
    assert len(summary['trigger_pairing_sha256']) == 64
    assert all(field in pairing for field in ablation.Q024_REQUIRED_PAIRING_FIELDS
               if field not in ('real_map_sha256', 'policy_map_sha256'))


def test_only_height_changed_guard_rejects_other_channel_changes():
    real, valid = make_map()
    policy = real.copy()
    policy[1, 0, 0] = 0.0
    proof = ablation.assert_only_height_changed(real, policy, valid)
    assert proof['passed'] is False
    assert proof['valid_channel_unchanged'] is False


def test_invalid_shape_and_bad_scale_are_rejected():
    real, valid = make_map()
    with pytest.raises(ValueError):
        ablation.apply_plane_height_channel(real[:2], valid, 8.0, 8.2)
    with pytest.raises(ValueError):
        ablation.apply_plane_height_channel(real, valid[:2], 8.0, 8.2)
    with pytest.raises(ValueError):
        ablation.plane_height_value(8.0, 8.2, scale=0.0)
    with pytest.raises(ValueError):
        ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0,
                                 shape=(3, 48, 32))


def test_bv03_terminal_callback_after_finished_does_not_count():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    # The counter starts at the trigger frame for this regression.
    switch.observe(frame_index=30, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True,
                   front_foot_top_support=True)
    # Step 31: Actor consumes; physics then reports the first terminal.
    switch.observe(frame_index=31, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False, terminal=True)
    # Step 32: a post-terminal callback must not change any count.
    switch.observe(frame_index=32, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False)
    summary = switch.summary()
    assert summary['frames_consumed'] == 2          # 30 and 31 only
    assert summary['replacement_frames_consumed'] == 0
    assert summary['saturation_denominator'] == 0
    assert summary['saturation'] is None
    assert summary['first_consumed_replacement_frame'] is None
    assert summary['exposed'] is False


def test_bv03_trigger_and_terminal_same_frame_counts_once():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True)
    switch.observe(frame_index=1, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True,
                   front_foot_top_support=True, terminal=True)
    switch.observe(frame_index=2, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True)
    summary = switch.summary()
    assert summary['frames_consumed'] == 2   # frame 0 + trigger/terminal frame 1
    assert summary['exposed'] is False
    assert summary['replacement_frames_consumed'] == 0


def test_bv03_unconsumed_terminal_callback_is_not_counted():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True)
    switch.observe(frame_index=1, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True,
                   front_foot_top_support=True)
    switch.observe(frame_index=2, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False, consumed=True)
    switch.observe(frame_index=3, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False, consumed=False,
                   terminal=True)
    switch.observe(frame_index=4, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False, consumed=True)
    assert switch.summary()['frames_consumed'] == 3  # frames 0,1,2 only
    assert switch.finished is True


def test_reset_prevents_cross_case_accounting():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=True)
    switch.observe(frame_index=1, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=False)
    assert switch.summary()['frames_consumed'] == 2
    switch.reset()
    assert switch.summary()['frames_consumed'] == 0
    sample = dict(refresh_index=0, sample_sim_time_s=0.0, produced_at_actor_step=0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=True, sample_id=sample)
    assert switch.summary()['frames_consumed'] == 1
    assert switch.summary()['sample_reference_count'] == 1


def test_per_frame_valid_age_and_invalid_cells_are_checked_each_frame():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=True)
    # Non-refresh frame: age changes and one cell becomes invalid; the policy
    # map must follow the current valid/age channels and force invalid 0/0/1.
    stale = real.copy()
    stale[2] = 0.6
    valid2 = valid.copy()
    valid2[0, 0] = False
    stale[:, 0, 0] = 0.0
    stale[2, 0, 0] = 1.0
    switch.observe(frame_index=1, real_map=stale, valid=valid2,
                   base_sample_z=0.4, refreshed=False)
    summary = switch.summary()
    assert summary['frame_checks'] == 2
    assert summary['frame_check_failures'] == 0
    # A deliberately corrupted age channel must fail the per-frame proof.
    bad = stale.copy()
    bad[2] = 0.0
    with pytest.raises(ValueError):
        switch.observe(frame_index=2, real_map=bad, valid=valid2,
                       base_sample_z=0.4, refreshed=False)


def test_native_sample_references_are_kept_per_consumed_frame():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    sample = dict(refresh_index=4, sample_sim_time_s=0.4, produced_at_actor_step=2)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=True, sample_id=sample)
    switch.observe(frame_index=1, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=False, sample_id=sample)
    summary = switch.summary()
    assert summary['native_sample_id'] == sample
    assert summary['sample_reference_count'] == 2
    assert len(summary['sample_references_sha256']) == 64
    assert summary['first_sample_reference'][0] == 0
    assert summary['last_sample_reference'][0] == 1


def test_real_branch_uses_its_own_event_and_refresh_window():
    real, valid = make_native_map()
    evidence = ablation.S2RealBranchEvidence(task='stairs_walk')
    for frame in range(2):
        evidence.observe(frame_index=frame, real_map=real, valid=valid,
                         refreshed=(frame % 5 == 0))
    evidence.observe(frame_index=2, real_map=real, valid=valid,
                     refreshed=True, front_foot_top_support=True)
    evidence.observe(frame_index=3, real_map=real, valid=valid,
                     refreshed=False)
    evidence.observe(frame_index=4, real_map=real, valid=valid,
                     refreshed=True)
    summary = evidence.summary()
    assert summary['branch'] == 'real'
    assert summary['trigger_frame'] == 2
    assert summary['window_start_frame'] == 4
    assert summary['s_minus_t'] == 2
    assert summary['replacement_applied'] is False
    assert summary['first_replacement_frame'] is None
    assert summary['exact_clip_fraction'] is None
    assert summary['exact_clip_missing_reason'] == 'real_map_stores_only_clipped_height'


def test_real_branch_birth_window_and_not_exposed():
    real, valid = make_native_map()
    flat = ablation.S2RealBranchEvidence(task='flat_walk')
    flat.observe(frame_index=0, real_map=real, valid=valid, refreshed=True)
    summary = flat.summary()
    assert summary['window_start_frame'] == 0
    assert summary['exposed'] is True

    stair = ablation.S2RealBranchEvidence(task='stairs_walk')
    stair.observe(frame_index=0, real_map=real, valid=valid, refreshed=True)
    summary = stair.summary()
    assert summary['exposed'] is False
    assert summary['note'] == 'not_exposed_before_terminal'


def test_q024_minimal_evidence_validation_rejects_missing_or_forbidden_diff():
    base = dict(case_sha256='a' * 64, stream_sha256='e' * 64,
                weight_sha256='b' * 64, code_sha256='c' * 64,
                physics_sha256='f' * 64, reward_sha256='0' * 64,
                termination_sha256='1' * 64, state_sha256='d' * 64,
                command=[0.5, 0.0, 0.0], action=[0.0] * 12,
                sample_frame=17)
    assert ablation.validate_pairing(base) is True
    broken = dict(base)
    broken.pop('physics_sha256')
    with pytest.raises(ValueError):
        ablation.validate_pairing(broken)
    forbidden = dict(base, config_diff={'map_branch': ['real', 'plane']})
    with pytest.raises(ValueError):
        ablation.validate_pairing(forbidden)
    allowed = dict(base, config_diff={'diagnostic_map_branch': ['real', 'plane'],
                                      'record_id': ['a', 'b']})
    assert ablation.validate_pairing(allowed) is True


def test_q024_branch_pair_requires_shared_identity():
    base = dict(case_sha256='a' * 64, stream_sha256='e' * 64,
                weight_sha256='b' * 64, code_sha256='c' * 64,
                physics_sha256='f' * 64, reward_sha256='0' * 64,
                termination_sha256='1' * 64, state_sha256='d' * 64,
                command=[0.5, 0.0, 0.0], action=[0.0] * 12,
                sample_frame=17)
    proof = ablation.validate_branch_pair(base, dict(base, state_sha256='9' * 64))
    assert len(proof['shared_identity_sha256']) == 64
    with pytest.raises(ValueError):
        ablation.validate_branch_pair(base, dict(base, reward_sha256='2' * 64))


def test_bv03_terminal_after_the_consumed_step_is_not_counted():
    real, valid = make_native_map()
    switch = ablation.S2PlaneAblation(task='stairs_walk', plane_world_z=8.0)
    switch.observe(frame_index=30, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=True,
                   front_foot_top_support=True)
    switch.observe(frame_index=31, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False)
    # Physics reports first termination after the step-31 Actor consumption.
    switch.observe(frame_index=32, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False, consumed=False,
                   terminal=True)
    assert switch.summary()['frames_consumed'] == 2
    switch.observe(frame_index=33, real_map=real, valid=valid,
                   base_sample_z=8.2, refreshed=False, consumed=True)
    assert switch.summary()['frames_consumed'] == 2


def test_f_v04_mixed_aggregate_does_not_dilute_plane_exact_clip():
    real = dict(frames_consumed=1, valid_fraction_denominator=1080,
                valid_cell_frame_sum=1080, valid_fraction=1.0,
                saturation_numerator=0, saturation_denominator=0,
                saturation=None, exact_clip_numerator=0,
                exact_clip_denominator=None, exact_clip_fraction=None,
                exposed=True, replacement_frames_consumed=0,
                missing_reason=None)
    plane = dict(frames_consumed=1, valid_fraction_denominator=1080,
                 valid_cell_frame_sum=1080, valid_fraction=1.0,
                 saturation_numerator=540, saturation_denominator=1080,
                 saturation=0.5, exact_clip_numerator=540,
                 exact_clip_denominator=1080, exact_clip_fraction=0.5,
                 exposed=True, replacement_frames_consumed=1,
                 missing_reason=None)
    aggregate = ablation.aggregate_evidence([real, plane])
    assert aggregate['saturation_denominator'] == 1080
    assert aggregate['exact_clip_denominator'] == 1080
    assert aggregate['exact_clip_fraction'] == pytest.approx(0.5)


def test_config_identity_declares_every_required_pairing_field():
    identity = ablation.config_identity()
    assert identity['q024']['required_pairing_field_count'] == 11
    assert len(identity['q024']['required_pairing_fields']) == 11
    assert len(identity['q024']['shared_identity_fields']) == 7
    assert set(ablation.PAIRING_REQUIRED_FIELDS) == set(
        identity['q024']['required_pairing_fields'])
