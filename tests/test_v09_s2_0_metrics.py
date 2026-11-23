"""CPU-only S2.0 event/segment metric tests; no simulator or physics rollout."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_metrics


PLATFORM = dict(front_x=8.0, length=20.0, y_half_width=0.4)


def make_trace(*, failure_at=None, stop_at=None, contact=True, route_x=None,
               front_at=2, four_at=5, single_segment=False):
    n = 30
    step = np.arange(n)
    route = 7.5 + 0.1 * step if route_x is None else np.asarray(route_x, dtype=float)
    base = np.stack((route, np.zeros(n)), axis=-1)
    feet = np.zeros((n, 4, 2))
    feet[:, :, 0] = 7.0
    if not single_segment:
        feet[front_at:, 0, 0] = 8.2
        feet[front_at:, 1, 0] = 8.2
        feet[four_at:, 2, 0] = 8.3
        feet[four_at:, 3, 0] = 8.3
    else:
        feet[front_at:, :, 0] = 8.2
    foot_on = np.zeros((n, 4), dtype=bool)
    foot_on[front_at:, 0:2] = True
    if single_segment:
        foot_on[front_at:, :] = True
    else:
        foot_on[four_at:, 2:4] = True
    if contact:
        # Two-frame toggling gives several complete landing intervals.
        foot_contact = ((step[:, None] + np.array([0, 0, 1, 1])) % 2 == 1)
    else:
        foot_contact = np.zeros((n, 4), dtype=bool)
    g_level = np.zeros(n)
    g_level[7] = 0.5
    g_level[8:] = 1.0
    pitch = np.zeros(n)
    pitch[10] = 0.1
    pitch[12] = -0.05
    c_exec = np.full(n, 0.5)
    velocity = np.full(n, 5.0)
    if stop_at is not None:
        c_exec[stop_at:] = 0.0
        velocity[stop_at:] = 0.0
    trace = dict(
        step=step, route_x=route, base_xy=base, feet_xy=feet,
        foot_on_platform=foot_on, foot_contact=foot_contact, g_level=g_level,
        pitch=pitch, c_exec=c_exec, velocity_route=velocity,
        foot_normal_force=np.full((n, 4), 10.0),
        top_arrived=np.zeros(n, dtype=bool), failure=np.zeros(n, dtype=bool),
        stop_request=np.zeros(n, dtype=bool), terminal=np.zeros(n, dtype=bool),
    )
    trace['top_arrived'][8:] = True
    if failure_at is not None:
        trace['failure'][failure_at] = True
    if stop_at is not None:
        trace['stop_request'][stop_at] = True
    return trace


def test_event_boundaries():
    trace = make_trace()
    events = s2_metrics.case_events(trace, platform=PLATFORM)
    assert events['front_foot_entry']['index'] == 2
    assert events['front_foot_entry']['step'] == 2
    assert events['four_feet_platform']['index'] == 5
    assert events['g_level_positive']['index'] == 7
    assert events['g_level_full']['index'] == 8
    assert events['top_arrived']['index'] == 8
    assert events['failure']['present'] is False
    assert events['stop_request']['present'] is False
    assert events['terminal']['index'] == 29


def test_platform_polygon_excludes_feet_one_centimetre_outside():
    trace = make_trace()
    trace['feet_xy'][5:, 2:4, 1] = 0.40 + 0.011  # lateral exit beyond narrow platform
    events = s2_metrics.case_events(trace, platform=PLATFORM)
    assert events['four_feet_platform']['present'] is False


def test_segment_bounds_and_first_meter_boundary():
    trace = make_trace()
    events = s2_metrics.case_events(trace, platform=PLATFORM)
    bounds = s2_metrics.segment_bounds(trace, events)
    assert bounds['front_entry_to_four_entry']['start'] == 2
    assert bounds['front_entry_to_four_entry']['end'] == 5
    assert bounds['front_entry_to_four_entry']['completed'] is True
    assert bounds['first_meter']['start'] == 5
    assert bounds['first_meter']['end'] == 15
    assert bounds['first_meter']['completed'] is True
    assert bounds['post_first_meter']['start'] == 15
    assert bounds['post_first_meter']['end'] == 29
    assert bounds['after_stop_request']['entered'] is False


def test_exact_one_metre_counts_and_0999_does_not():
    route = 7.5 + 0.1 * np.arange(30)
    route[5] = 0.0
    for index in range(6, 30):
        route[index] = (index - 5) * 0.1
    route[15] = 1.0
    events = s2_metrics.case_events(make_trace(route_x=route), platform=PLATFORM)
    bounds = s2_metrics.segment_bounds(make_trace(route_x=route), events)
    assert bounds['first_meter']['end'] == 15
    assert bounds['first_meter']['completed'] is True

    route2 = route.copy()
    route2[15] = 0.999
    route2[16] = 1.0
    events2 = s2_metrics.case_events(make_trace(route_x=route2), platform=PLATFORM)
    bounds2 = s2_metrics.segment_bounds(make_trace(route_x=route2), events2)
    assert bounds2['first_meter']['end'] == 16
    assert bounds2['first_meter']['completed'] is True


def test_same_frame_failure_wins_over_first_meter():
    trace = make_trace(failure_at=15)
    result = s2_metrics.summarize_case(trace, platform=PLATFORM)
    first = result['segments']['first_meter']
    assert result['first_meter_reached'] is False
    assert first['completed'] is False
    assert first['failed'] is True
    assert first['end_reason'] == 'failure'
    assert result['failure_step'] == 15


def test_failure_before_four_foot_entry_keeps_the_terminal_frame():
    trace = make_trace(failure_at=4)
    result = s2_metrics.summarize_case(trace, platform=PLATFORM)
    front = result['segments']['front_entry_to_four_entry']
    assert front['start_step'] == 2
    assert front['end_step'] == 4
    assert front['failed'] is True
    assert front['completed'] is False
    assert result['segments']['first_meter']['entered'] is False
    assert result['segments']['first_meter']['not_reached'] is True


def test_gait_cycles_and_missing_marker():
    result = s2_metrics.summarize_case(make_trace(), platform=PLATFORM)
    first = result['segments']['first_meter']
    assert first['gait']['valid'] is True
    assert first['gait']['complete_cycles'] >= 2
    assert first['gait']['stride_median_m'] is not None
    assert first['gait']['frequency_median_hz'] is not None
    assert first['valid_gait_samples'] is True

    missing = s2_metrics.summarize_case(make_trace(contact=False), platform=PLATFORM)
    first_missing = missing['segments']['first_meter']
    assert first_missing['gait']['valid'] is False
    assert first_missing['gait']['stride_median_m'] is None
    assert first_missing['gait']['missing_reason'] == 'insufficient_complete_landing_cycles'
    assert first_missing['valid_gait_samples'] is False


def test_single_sample_segment_reports_missing_rate_and_cycles():
    result = s2_metrics.summarize_case(make_trace(single_segment=True),
                                       platform=PLATFORM)
    front = result['segments']['front_entry_to_four_entry']
    assert front['sample_count'] == 1
    assert front['pitch_rate_abs_max_deg_s'] is None
    assert front['missing']['pitch_rate'] == 'single_sample_segment'
    assert front['gait']['valid'] is False


def test_pitch_sign_is_reported_without_reinterpretation():
    result = s2_metrics.summarize_case(make_trace(), platform=PLATFORM)
    first = result['segments']['first_meter']
    assert first['pitch_max_deg_signed'] == pytest.approx(0.1 * 180 / np.pi)
    assert first['pitch_min_deg_signed'] == pytest.approx(-0.05 * 180 / np.pi)
    assert first['pitch_abs_max_deg'] == pytest.approx(0.1 * 180 / np.pi)
    assert 'positive' in result['sign_convention']


def test_g_level_delays_are_relative_to_four_foot_entry():
    result = s2_metrics.summarize_case(make_trace(), platform=PLATFORM)
    assert result['g_level_delay_seconds']['g_level_positive_seconds_from_zero'] == pytest.approx(0.04)
    assert result['g_level_delay_seconds']['g_level_full_seconds_from_zero'] == pytest.approx(0.06)


def test_stop_request_is_separated_from_post_first_meter():
    trace = make_trace(stop_at=20)
    result = s2_metrics.summarize_case(trace, platform=PLATFORM)
    assert result['stop_request_step'] == 20
    assert result['segments']['post_first_meter']['end_step'] == 20
    after = result['segments']['after_stop_request']
    assert after['entered'] is True
    assert after['start_step'] == 20
    assert after['end_step'] == 29
    assert after['c_exec_max'] == 0.0
    assert after['actual_forward_speed_mean'] == 0.0


def test_platform_start_uses_spawn_as_zero():
    trace = make_trace()
    trace['route_x'] = 9.0 + 0.1 * np.arange(30)
    trace['base_xy'][:, 0] = trace['route_x']
    trace['feet_xy'][:, :, 0] = 8.0 + trace['feet_xy'][:, :, 0] - 7.0
    events = s2_metrics.case_events(trace, platform=PLATFORM)
    result = s2_metrics.summarize_case(trace, platform=PLATFORM, zero_anchor='spawn')
    assert result['zero_index'] == 0
    bounds = s2_metrics.segment_bounds(trace, events, zero_anchor='spawn')
    assert bounds['first_meter']['start'] == 0
    assert bounds['first_meter']['end'] == 10


def test_signed_net_displacement_can_be_negative():
    route = 8.0 - 0.05 * np.arange(30)
    trace = make_trace(route_x=route)
    result = s2_metrics.summarize_case(trace, platform=PLATFORM)
    first = result['segments']['first_meter']
    assert first['signed_distance_m'] < 0
    assert result['first_meter_reached'] is False


def test_group_counts_keep_failures_and_denominators():
    good = s2_metrics.summarize_case(make_trace(), platform=PLATFORM)
    bad = s2_metrics.summarize_case(make_trace(failure_at=6), platform=PLATFORM)
    summary = s2_metrics.summarize_group([good, bad], speed=0.5)
    assert summary['case_count'] == 2
    assert summary['physical_failures'] == 1
    assert summary['first_meter_reached'] == 1
    first = summary['segments']['first_meter']
    assert first['entered'] == 2
    assert first['failed'] == 1
    assert first['completed'] == 1
    assert first['valid_gait_samples'] >= 1
    assert first['sample_count'] > 0


def test_flat_gait_comparison_preserves_missing():
    group = s2_metrics.summarize_group(
        [s2_metrics.summarize_case(make_trace(), platform=PLATFORM)], speed=0.5)
    missing = s2_metrics.summarize_group(
        [s2_metrics.summarize_case(make_trace(contact=False), platform=PLATFORM)],
        speed=0.5)
    compare = s2_metrics.compare_gait_to_flat(group, missing)
    assert compare['speed'] == 0.5
    for key in ('stride_median_m', 'frequency_median_hz'):
        assert compare['metrics'][key]['available'] is False
        assert compare['metrics'][key]['ratio'] is None
    with pytest.raises(ValueError):
        s2_metrics.compare_gait_to_flat(group, dict(missing, speed=0.6))


def test_trace_validation_rejects_broken_ordering():
    trace = make_trace()
    trace['step'][3] = trace['step'][2]
    with pytest.raises(ValueError):
        s2_metrics.case_events(trace, platform=PLATFORM)
    broken = make_trace()
    broken['g_level'][0] = 1.5
    with pytest.raises(ValueError):
        s2_metrics.case_events(broken, platform=PLATFORM)


def test_front_foot_top_support_alias_is_the_same_latched_event():
    events = s2_metrics.case_events(make_trace(), platform=PLATFORM)
    assert events['front_foot_top_support'] == events['front_foot_entry']


def test_d_post_zero_uses_latched_top_arrived_frame():
    trace = make_trace()
    result = s2_metrics.summarize_case(trace, platform=PLATFORM)
    assert result['d_post_zero_index'] == 8
    # route[8] = 8.3, route[-1] = 10.4
    assert result['d_post_terminal_m'] == pytest.approx(2.1)
    assert result['d_post_segments_m']['first_meter'] == pytest.approx(9.0 - 8.3)
    assert result['first_meter_zero_index'] == 5


def test_d_post_uses_spawn_for_platform_start():
    trace = make_trace()
    trace['route_x'] = 9.0 + 0.1 * np.arange(30)
    trace['base_xy'][:, 0] = trace['route_x']
    trace['feet_xy'][:, :, 0] = 8.0 + trace['feet_xy'][:, :, 0] - 7.0
    result = s2_metrics.summarize_case(trace, platform=PLATFORM, zero_anchor='spawn')
    assert result['d_post_zero_index'] == 0
    assert result['d_post_terminal_m'] == pytest.approx(2.9)


def test_failure_context_distinguishes_platform_and_stairs():
    platform = s2_metrics.summarize_case(make_trace(failure_at=20), platform=PLATFORM)
    context = platform['failure_context']
    assert context['surface'] == 'platform'
    assert context['classification'] == 'platform_cruise_failure'
    assert context['after_four_foot_entry'] is True
    assert context['onto_platform_m'] > 0

    route = 0.0 + 0.1 * np.arange(30)
    stairs_trace = make_trace(route_x=route, failure_at=5)
    stairs_trace['feet_xy'][:, :, 0] = 0.5
    stairs = s2_metrics.summarize_case(stairs_trace, platform=PLATFORM)
    assert stairs['failure_context']['surface'] == 'stairs'
    assert stairs['failure_context']['classification'] == 'stair_failure'
    assert stairs['failure_context']['after_four_foot_entry'] is False
    assert stairs['failure_context']['signed_from_platform_front_m'] < 0

    approach_route = -1.0 + 0.1 * np.arange(30)
    approach_trace = make_trace(route_x=approach_route, failure_at=2)
    approach_trace['feet_xy'][:, :, 0] = -0.5
    approach = s2_metrics.summarize_case(approach_trace, platform=PLATFORM)
    assert approach['failure_context']['surface'] == 'approach'


def test_original_failure_coverage_and_surface():
    result = s2_metrics.summarize_case(make_trace(), platform=PLATFORM)
    covered = s2_metrics.original_failure_coverage(
        result, dict(route_x=2.0, surface='platform'))
    assert covered['covered'] is True
    assert covered['status'] == 'covered'
    not_covered = s2_metrics.original_failure_coverage(
        result, dict(route_x=12.0, surface='platform'))
    assert not_covered['covered'] is False
    assert not_covered['note'] == 'not_verified_before_original_failure_position'
    with pytest.raises(ValueError):
        s2_metrics.original_failure_coverage(result, dict(surface='platform'))


def test_group_counts_failure_surfaces():
    platform_fail = s2_metrics.summarize_case(make_trace(failure_at=20), platform=PLATFORM)
    stairs_fail = s2_metrics.summarize_case(
        make_trace(route_x=0.0 + 0.1 * np.arange(30), failure_at=5), platform=PLATFORM)
    summary = s2_metrics.summarize_group([platform_fail, stairs_fail], speed=0.5)
    assert summary['failure_on_platform'] == 1
    assert summary['failure_on_stairs'] == 1
    assert summary['failure_on_approach'] == 0


def test_failure_context_separates_body_terminal_from_four_foot_entry():
    # Body is over the platform but the four-foot event never happens.
    trace = make_trace(failure_at=20, single_segment=True)
    trace['foot_on_platform'][:] = False
    trace['feet_xy'][:, :, 0] = 7.0
    result = s2_metrics.summarize_case(trace, platform=PLATFORM)
    context = result['failure_context']
    assert context['body_terminal_surface'] == 'platform'
    assert context['four_feet_on_platform_at_failure'] is False
    assert context['feet_on_platform_at_failure'] == 0
    assert context['classification'] == 'platform_failure_before_four_feet'
    assert context['after_four_foot_entry'] is False


def test_q025_platform_quality_window_is_one_to_five_metres():
    trace = make_trace()
    trace['route_x'] = 9.0 + 0.5 * np.arange(30)
    trace['base_xy'][:, 0] = trace['route_x']
    trace['feet_xy'][:, :, 0] = 8.0 + trace['feet_xy'][:, :, 0] - 7.0
    result = s2_metrics.summarize_case(trace, platform=PLATFORM, zero_anchor='spawn')
    window = result['quality_window']
    assert window['entered'] is True
    assert window['start_index'] == 2          # spawn + 1 m
    assert window['end_index'] == 10           # spawn + 5 m
    assert window['start_m'] == pytest.approx(1.0)
    assert window['end_m'] == pytest.approx(5.0)
    assert window['nominal_length_m'] == pytest.approx(4.0)
    assert result['first_meter_reached'] is True
    assert result['d_post_zero_index'] == 0


def test_q025_climb_quality_window_starts_after_four_foot_entry_plus_one_metre():
    result = s2_metrics.summarize_case(make_trace(), platform=PLATFORM)
    window = result['quality_window']
    assert window['entered'] is True
    assert window['start_index'] == result['first_meter_zero_index'] + 10 + 0  # 4-foot 5 + 10
    assert window['start_m'] == pytest.approx(1.0)
    assert window['end_index'] == 29
    assert result['d_post_zero_index'] == 8


def test_q025_timeout_is_unchanged_by_the_quality_window():
    from src.tasks.velocity.v09 import s2_spec
    assert s2_spec.timeout_steps(task='platform_walk', speed=0.3,
                                 post_distance=5.0) == 2167
    assert s2_spec.PLATFORM_QUALITY_WINDOW_END_M == 5.0


def test_map_evidence_attaches_to_case_and_group_with_raw_sums():
    from src.tasks.velocity.v09 import s2_map_ablation as ablation
    real = np.zeros((3, 40, 30))
    real[0] = 0.25
    real[1] = 1.0
    valid = np.ones((40, 30), dtype=bool)
    switch = ablation.S2PlaneAblation(task='flat_walk', plane_world_z=0.0)
    sample = dict(refresh_index=0, sample_sim_time_s=0.0, produced_at_actor_step=0)
    switch.observe(frame_index=0, real_map=real, valid=valid,
                   base_sample_z=0.4, refreshed=True, sample_id=sample,
                   terminal=True)
    evidence = switch.summary()
    case = s2_metrics.summarize_case(make_trace(), platform=PLATFORM,
                                     map_evidence=evidence)
    record = case['map_evidence']
    assert record['branch'] == 'plane_diagnostic'
    assert record['valid_fraction_numerator'] == 40 * 30
    assert record['valid_fraction_denominator'] == 40 * 30
    assert record['saturation_numerator'] >= 0
    assert record['event_step'] == 0
    assert record['native_sample_id'] == sample
    assert record['replacement_frames_consumed'] == 1
    assert record['sample_reference_count'] == 1
    group = s2_metrics.summarize_group([case, case], speed=0.5)
    aggregate = group['map_evidence']
    assert aggregate['valid_fraction_denominator'] == 2 * 40 * 30
    assert aggregate['valid_fraction_numerator'] == 2 * 40 * 30
    assert aggregate['replacement_frames_consumed'] == 2
    assert aggregate['sample_reference_count'] == 2


def test_map_evidence_na_and_missing_fields():
    base = dict(branch='real', frames_consumed=1, valid_fraction_numerator=0,
                valid_fraction_denominator=0, valid_fraction=None,
                saturation_numerator=0, saturation_denominator=0,
                saturation=None, exact_clip_numerator=0,
                exact_clip_denominator=None, exact_clip_fraction=None,
                event_step=None, native_sample_id=None,
                first_replacement_actor_step=None, first_replacement_frame=None,
                s_minus_t=None, exposed=False, replacement_applied=False,
                replacement_frames_consumed=0, sample_reference_count=0,
                sample_references_sha256=None, first_sample_reference=None,
                last_sample_reference=None)
    record = s2_metrics.normalize_map_evidence(base)
    assert record['valid_fraction'] is None
    assert record['saturation'] is None
    assert record['exact_clip_fraction'] is None
    assert record['zero_valid_denominator'] is True
    assert record['zero_saturation_denominator'] is True
    assert record['not_exposed'] is True
    broken = dict(base)
    broken.pop('saturation_numerator')
    with pytest.raises(ValueError):
        s2_metrics.normalize_map_evidence(broken)


def _evidence(**overrides):
    base = dict(branch='real', frames_consumed=1, valid_fraction_numerator=1080,
                valid_fraction_denominator=1080, valid_fraction=1.0,
                saturation_numerator=0, saturation_denominator=0,
                saturation=None, exact_clip_numerator=0,
                exact_clip_denominator=None, exact_clip_fraction=None,
                event_step=0, native_sample_id=None,
                first_replacement_actor_step=None, first_replacement_frame=None,
                s_minus_t=None, exposed=True, replacement_applied=False,
                replacement_frames_consumed=0, sample_reference_count=0,
                sample_references_sha256=None, first_sample_reference=None,
                last_sample_reference=None,
                exact_clip_missing_reason='real_map_stores_only_clipped_height',
                missing_reason=None)
    base.update(overrides)
    return base


def test_mixed_branch_exact_clip_uses_only_the_plane_denominator():
    real = _evidence()
    plane = _evidence(branch='plane_diagnostic', replacement_applied=True,
                      saturation_numerator=540, saturation_denominator=1080,
                      saturation=0.5, exact_clip_numerator=540,
                      exact_clip_denominator=1080, exact_clip_fraction=0.5,
                      exact_clip_missing_reason=None, replacement_frames_consumed=1)
    aggregate = s2_metrics.summarize_map_evidence([real, plane])
    assert aggregate['saturation_denominator'] == 1080
    assert aggregate['exact_clip_denominator'] == 1080
    assert aggregate['exact_clip_fraction'] == pytest.approx(540 / 1080)
    assert aggregate['exact_clip_available_cases'] == 1


def test_missing_reason_counts_are_written_into_records_and_groups():
    reached = _evidence(exposed=False, event_step=10,
                        missing_reason='no_valid_refresh')
    not_reached = _evidence(exposed=False, event_step=None,
                            missing_reason='event_not_reached')
    aggregate = s2_metrics.summarize_map_evidence([reached, not_reached])
    assert aggregate['not_exposed'] == 2
    assert aggregate['missing_reason_counts'] == {'event_not_reached': 1,
                                                  'no_valid_refresh': 1}
    result = s2_metrics.summarize_case(make_trace(), platform=PLATFORM,
                                       map_evidence=reached)
    assert result['map_evidence']['missing_reason'] == 'no_valid_refresh'
    group = s2_metrics.summarize_group([result], speed=0.5)
    assert group['map_evidence']['missing_reason_counts']['no_valid_refresh'] == 1


def test_zero_valid_denominator_is_a_distinct_reason_code():
    zero = _evidence(saturation_numerator=0, saturation_denominator=0,
                     saturation=None, exact_clip_numerator=0,
                     exact_clip_denominator=None, exact_clip_fraction=None,
                     zero_valid_denominator_reason='zero_valid_denominator')
    normalized = s2_metrics.normalize_map_evidence(zero)
    assert normalized['zero_saturation_denominator'] is True
    assert normalized['zero_valid_denominator_reason'] == 'zero_valid_denominator'
    aggregate = s2_metrics.summarize_map_evidence([zero])
    assert aggregate['zero_valid_denominator_cases'] == 1
    assert aggregate['zero_valid_denominator_reason_counts'] == {
        'zero_valid_denominator': 1}


def _metric_evidence(n, task, *, support_entry=None):
    import torch
    step = torch.arange(n, dtype=torch.long)
    position = torch.zeros(n, 3)
    position[:, 0] = -1.0 + 0.01 * step.float()
    support = torch.ones(n, 4, dtype=torch.bool)
    if support_entry is not None:
        support[:support_entry] = False
    return dict(
        step=step, position=position,
        initial_position=position[0].clone(),
        roll=torch.zeros(n), pitch=torch.zeros(n),
        velocity_b=torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        velocity_w=torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1),
        omega=torch.zeros(n, 3),
        foot_x=position[:, 0].unsqueeze(-1).repeat(1, 4),
        foot_velocity_xy=torch.zeros(n, 4, 2),
        raw_contact=(step[:, None] + torch.tensor([0, 0, 1, 1])) % 2 == 1,
        filtered=(step[:, None] + torch.tensor([0, 0, 1, 1])) % 2 == 1,
        support=support,
        action=torch.zeros(n, 12), previous_action=torch.zeros(n, 12),
        height=torch.full((n,), 0.4), nonfoot_contact=torch.zeros(n, dtype=torch.bool),
        command=torch.full((n,), 0.5), is_cruise=torch.ones(n, dtype=torch.bool))


def test_f3_flat_pitch_15_latch_is_window_scoped_in_gate_metrics():
    from src.tasks.velocity.v09 import metrics
    evidence = _metric_evidence(1000, 'flat_walk')
    evidence['pitch'][50:60] = 16 * 3.141592653589793 / 180
    result = metrics.summarize_trajectory(
        evidence, task='flat_walk', stop_mode='none', cruise=0.5)
    assert result['quality_window_start_step'] == 100
    assert result['quality_window_end_step'] == 999
    assert result['pitch_over_15_max_frames'] == 0
    assert 'pitch_continuous' not in result['quality_reasons']

    evidence = _metric_evidence(1000, 'flat_walk')
    evidence['pitch'][300:310] = 16 * 3.141592653589793 / 180
    result = metrics.summarize_trajectory(
        evidence, task='flat_walk', stop_mode='none', cruise=0.5)
    assert result['pitch_over_15_max_frames'] == 10
    assert 'pitch_continuous' in result['quality_reasons']


def test_f3_stairs_pitch_15_latch_starts_after_entry_plus_one_metre():
    from src.tasks.velocity.v09 import metrics
    evidence = _metric_evidence(700, 'stairs_walk', support_entry=100)
    evidence['pitch'][150:160] = 16 * 3.141592653589793 / 180
    result = metrics.summarize_trajectory(
        evidence, task='stairs_walk', stop_mode='none', cruise=0.5)
    assert result['quality_window_start_step'] == 200
    assert result['pitch_over_15_max_frames'] == 0
    assert 'pitch_continuous' not in result['quality_reasons']

    evidence = _metric_evidence(700, 'stairs_walk', support_entry=100)
    evidence['pitch'][300:310] = 16 * 3.141592653589793 / 180
    result = metrics.summarize_trajectory(
        evidence, task='stairs_walk', stop_mode='none', cruise=0.5)
    assert result['pitch_over_15_max_frames'] == 10
    assert 'pitch_continuous' in result['quality_reasons']
