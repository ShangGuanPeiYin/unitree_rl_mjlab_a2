"""S2 itemized atomic reservation (Q021), computed from frozen inputs.

This is a CPU planning module.  It writes no ledger event and starts no
physics.  Every batch cost is ``num_envs × max(hard_timeout_steps)`` including
the padding of a completed slot, per §2.5.  Gate/final group compositions are
the §6.1.1/§6.5 matrices; their case manifests are not frozen yet, so this
plan is the admission upper bound and must be recomputed from the real manifests
before Q021 reservation.

Q021 items are kept separate and never double counted: 1024/512 resource
probes, S2.0 15×5×2, six H baseline supplemental sampling, three final
candidate re-tests, up to 40 gate admissions, and three final suites with their
frozen comparisons.
"""
from __future__ import annotations

from ..v07 import spec as original
from . import s2_cases, s2_spec
from .cases import sha256
from . import budget as ledger_budget

GATE_CASES = 20
FINAL_CASES = 100
GATE_LIMIT = 40
FINAL_SETS = 3
FINAL_GROUP_COUNT = 27
L40_MILESTONE_GROUPS = 3
L40_MILESTONE_SUITES = 2
L40_MILESTONE_CASES = 20
RESOURCE_PROBE_EVAL_ENVS = 100
RESOURCE_PROBE_WARMUP_AND_MEASURE = 55


def _flat_group_cost():
    # 3 speeds × walk/instant/ramp.
    return 3 * s2_spec.FLAT_WALK_HARD_TIMEOUT_STEPS + \
        3 * s2_spec.FLAT_STOP_HARD_TIMEOUT_STEPS['instant'] + \
        3 * s2_spec.FLAT_STOP_HARD_TIMEOUT_STEPS['ramp_2s']


def _stairs_timeout(speed, stair_count, post_distance, *, stop=False, ramp=False):
    return s2_spec.timeout_steps(
        task='stairs_stop' if stop else 'stairs_walk', speed=speed,
        stair_count=stair_count, step_height=s2_spec.STAIR_HEIGHT,
        step_depth=s2_spec.STAIR_DEPTH, approach_length=s2_spec.T_APPROACH_MAX,
        post_distance=post_distance,
        ramp_seconds=2.0 if ramp else 0.0,
        hold_seconds=20.0 if stop else 0.0)


def _platform_timeout(speed, post_distance, *, stop=False, ramp=False):
    return s2_spec.timeout_steps(
        task='platform_stop' if stop else 'platform_walk', speed=speed,
        post_distance=post_distance, ramp_seconds=2.0 if ramp else 0.0,
        hold_seconds=20.0 if stop else 0.0)


def _sum(values):
    return sum(int(value) for value in values)


def _review_short_climb():
    # 10/20 levels × two widths × three speeds, solid walk 5 m.
    total = 0
    for levels in (10, 20):
        for speed in s2_spec.SPEEDS:
            total += 2 * GATE_CASES * _stairs_timeout(speed, levels, 5.0)
    return total


def gate_suite_costs():
    """Per-course 20-case gate matrix (§6.1.1), all groups included."""
    flat = GATE_CASES * _flat_group_cost()
    review = _review_short_climb()
    platform_short = 2 * 2 * _sum(GATE_CASES * _platform_timeout(speed, 5.0)
                                  for speed in s2_spec.SPEEDS)
    l40_walk_10 = 2 * _sum(GATE_CASES * _stairs_timeout(speed, 40, 10.0)
                           for speed in s2_spec.SPEEDS)
    platform_10 = 2 * _sum(GATE_CASES * _platform_timeout(speed, 10.0)
                           for speed in s2_spec.SPEEDS)
    l40_stop_10 = 2 * _sum(
        GATE_CASES * (_stairs_timeout(speed, 40, 10.0, stop=True)
                      + _stairs_timeout(speed, 40, 10.0, stop=True, ramp=True))
        for speed in s2_spec.SPEEDS)
    platform_stop_10 = 2 * _sum(
        GATE_CASES * (_platform_timeout(speed, 10.0, stop=True)
                      + _platform_timeout(speed, 10.0, stop=True, ramp=True))
        for speed in s2_spec.SPEEDS)
    milestone = GATE_CASES * _sum(
        original.episode_step_limit(original.FINAL_TILE_ID, speed)
        for speed in s2_spec.SPEEDS)
    suites = {
        'S2.1': dict(groups=36, cases=36 * GATE_CASES,
                     breakdown=dict(flat=flat, review_short=review,
                                    platform_short=platform_short,
                                    l40_milestone=milestone)),
        'S2.2': dict(groups=33, cases=33 * GATE_CASES,
                     breakdown=dict(flat=flat, l40_walk_10=l40_walk_10,
                                    review_short=review, platform_8m_walk_10=platform_10)),
        'S2.3': dict(groups=51, cases=51 * GATE_CASES,
                     breakdown=dict(flat=flat, l40_stop_10=l40_stop_10,
                                    l40_walk_10=l40_walk_10,
                                    review_short=review, platform_8m_stop_10=platform_stop_10)),
    }
    for name, expected in (('S2.1', 36), ('S2.2', 33), ('S2.3', 51)):
        assert suites[name]['groups'] == expected
        suites[name]['control_samples'] = sum(suites[name]['breakdown'].values())
    return suites


def final_acceptance_cost():
    """§6.5 per final set: candidate 2700 + S1 2700 + 6400 flat 900."""
    flat = FINAL_CASES * _flat_group_cost()
    l40_stop = 2 * _sum(
        FINAL_CASES * (_stairs_timeout(speed, 40, 10.0, stop=True)
                       + _stairs_timeout(speed, 40, 10.0, stop=True, ramp=True))
        for speed in s2_spec.SPEEDS)
    l40_walk = 2 * _sum(FINAL_CASES * _stairs_timeout(speed, 40, 10.0)
                        for speed in s2_spec.SPEEDS)
    candidate = flat + l40_stop + l40_walk
    s1_reference = candidate
    original_6400_flat = flat
    return dict(groups=FINAL_GROUP_COUNT,
                candidate_cases=27 * FINAL_CASES,
                candidate_samples=candidate,
                s1_reference_samples=s1_reference,
                original_6400_flat_samples=original_6400_flat,
                per_set_samples=candidate + s1_reference + original_6400_flat,
                sets=FINAL_SETS,
                three_sets_samples=FINAL_SETS * (candidate + s1_reference + original_6400_flat))


def l40_milestone_cache_cost():
    """§6.1.1 one-time frozen S1 comparison cache over the same A/B cases."""
    per_suite = L40_MILESTONE_CASES * sum(
        original.episode_step_limit(original.FINAL_TILE_ID, speed)
        for speed in s2_spec.SPEEDS)
    return dict(groups_per_suite=L40_MILESTONE_GROUPS,
                suites=L40_MILESTONE_SUITES,
                cases_per_group=L40_MILESTONE_CASES,
                per_suite_samples=per_suite,
                control_samples=L40_MILESTONE_SUITES * per_suite,
                content='success count, P10/P50/P90, paired net distance, '
                        'physical failure classes')


def resource_probe_cost(num_envs, *, worst_timeout_steps):
    """§2.3 full admission: 5 warmups + 50 measurements + restore + eval/video."""
    if num_envs not in s2_spec.S2_ENV_CANDIDATES:
        raise ValueError('probe environment count must be 1024 or 512')
    warm_measure = num_envs * RESOURCE_PROBE_WARMUP_AND_MEASURE * s2_spec.ROLLOUT_STEPS
    evaluation = RESOURCE_PROBE_EVAL_ENVS * int(worst_timeout_steps)
    video = int(worst_timeout_steps)
    restore = num_envs * s2_spec.ROLLOUT_STEPS
    return dict(num_envs=num_envs,
                warmup_and_measurement_rollouts=RESOURCE_PROBE_WARMUP_AND_MEASURE,
                warmup_and_measurement_samples=warm_measure,
                evaluation_samples=evaluation,
                video_samples=video,
                restore_rollout_samples=restore,
                control_samples=warm_measure + evaluation + video + restore,
                full_admission=True)


def learning_plan():
    caps = s2_spec.S2_COURSE_LEARNING_CAPS
    rows = {}
    for course, cap in caps.items():
        rows[course] = dict(
            learning_cap=cap,
            rollouts_1024=s2_spec.complete_rollouts(cap, 1024),
            rollouts_512=s2_spec.complete_rollouts(cap, 512),
            samples_per_rollout_1024=1024 * s2_spec.ROLLOUT_STEPS,
            samples_per_rollout_512=512 * s2_spec.ROLLOUT_STEPS)
    total = sum(caps.values())
    if total > s2_spec.S2_LEARNING_CAP:
        raise ValueError('S2 course learning caps exceed the stage learning cap')
    return dict(courses=rows, allocated_learning_cap=total,
                stage_learning_cap=s2_spec.S2_LEARNING_CAP,
                unallocated_learning_cap=s2_spec.S2_LEARNING_CAP - total)


def s2_stage_reservation_draft(s2_0_manifest, h_manifest):
    """Q021 itemized upper bound; raises unless every cap holds."""
    gates = gate_suite_costs()
    finals = final_acceptance_cost()
    s2_0 = s2_cases.s2_0_matrix_cost(s2_0_manifest)
    h_baseline = s2_cases.h_series_reservation(h_manifest)['reserved_control_samples']
    worst_gate = max(row['control_samples'] for row in gates.values())
    worst_timeout = max(
        _stairs_timeout(speed, 40, 10.0, stop=True, ramp=True)
        for speed in s2_spec.SPEEDS)
    probes = {str(envs): resource_probe_cost(envs, worst_timeout_steps=worst_timeout)
              for envs in s2_spec.S2_ENV_CANDIDATES}
    l40_cache = l40_milestone_cache_cost()
    items = {
        'resource_probe_1024': probes['1024']['control_samples'],
        'resource_probe_512': probes['512']['control_samples'],
        's2_0_matrix_two_branches': s2_0['total_control_samples'],
        'h_series_baseline_supplement': h_baseline,
        'h_series_candidate_retest_3_models': FINAL_SETS * h_baseline,
        'l40_milestone_frozen_cache_once': l40_cache['control_samples'],
        'gate_admissions_up_to_40': GATE_LIMIT * worst_gate,
        'final_sets_3_with_frozen_comparisons': finals['three_sets_samples'],
    }
    total = sum(items.values())
    learning = learning_plan()
    nonlearning_ok = total <= s2_spec.S2_NONLEARNING_CAP
    s2_0_ok = (s2_0['total_control_samples'] + h_baseline
               <= s2_spec.S2_0_NONLEARNING_SUBCAP)
    learning_ok = learning['allocated_learning_cap'] <= s2_spec.S2_LEARNING_CAP
    if not (nonlearning_ok and s2_0_ok and learning_ok):
        raise ValueError('S2 atomic reservation exceeds an allowed cap')
    return dict(
        stage=s2_spec.STAGE, sds_sha256=s2_spec.SDS_SHA256,
        s2_0_manifest_sha256=s2_0_manifest['manifest_sha256'],
        h_manifest_sha256=h_manifest['manifest_sha256'],
        ledger_namespace=s2_spec.LEDGER_NAMESPACE,
        ledger_hint=s2_spec.LEDGER_HINT,
        q021_reservation_written=False,
        items=items, total_control_samples=total,
        physics_substeps=4 * total,
        nonlearning_cap=s2_spec.S2_NONLEARNING_CAP,
        remaining_nonlearning=s2_spec.S2_NONLEARNING_CAP - total,
        s2_0_subcap=s2_spec.S2_0_NONLEARNING_SUBCAP,
        s2_0_remaining=s2_spec.S2_0_NONLEARNING_SUBCAP - (
            s2_0['total_control_samples'] + h_baseline),
        gates=gates, gate_limit=GATE_LIMIT, worst_gate_suite_samples=worst_gate,
        final_acceptance=finals, probes=probes,
        l40_milestone_cache=l40_cache,
        billing=dict(
            unit='environment control samples',
            physics_substeps=4 * total,
            padding_included=True,
            completed_slot_wait_steps_billed=True,
            reset_after_terminal_excluded=True,
            s1_samples_not_charged_to_s2=True,
            probe=dict(gpu_peak_mib_cap=7168, host_free_mib_min=4096,
                       warmups=5, measurements=50, restore_check=1,
                       independent_copy=True,
                       evaluation_envs=RESOURCE_PROBE_EVAL_ENVS,
                       evidence='5 warmups + 50 measurements + one full 48-step '
                                'rollout with 20 Adam updates + save/eval/restore'),
        ),
        q025=dict(platform_quality_window='[spawn+1 m, spawn+5 m]',
                  platform_nominal_window_m=4.0,
                  climb_quality_window='four-foot entry +1 m to decel/end',
                  timeout_changed=False,
                  sensitivity_extra_first_meter_steps={
                      str(speed): s2_spec.extra_first_meter_steps(speed)
                      for speed in s2_spec.SPEEDS}),
        learning=learning,
        admitted=bool(nonlearning_ok and s2_0_ok and learning_ok),
        note='upper bound only; Q021 still requires the real gate/final manifests before reservation',
    )


def build_reservation_plan(s2_0_manifest, h_manifest):
    """Q021/T4 itemized plan, recomputed from the frozen manifests."""
    draft = s2_stage_reservation_draft(s2_0_manifest, h_manifest)
    l40 = l40_milestone_cache_cost()
    assert l40['per_suite_samples'] * 2 == l40['control_samples']
    items = []

    def add(obligation_id, name, category, samples, basis, **extra):
        items.append(dict(obligation_id=obligation_id, name=name,
                          category=category, control_samples=int(samples),
                          basis=basis, **extra))

    add('resource_probe_1024', '1024 full admission probe', 'engineering',
        draft['items']['resource_probe_1024'], 'SDS §2.3')
    add('resource_probe_512', '512 fallback probe', 'engineering',
        draft['items']['resource_probe_512'], 'SDS §2.3')
    add('s2_0_matrix_two_branches', 'S2.0 75-case real+plane branches',
        'engineering', draft['items']['s2_0_matrix_two_branches'],
        'SDS §4.3 / DSH-Q020', subcap='S2_0')
    add('h_series_baseline_supplement',
        'six H baseline missing-field supplement (<=6)', 'engineering',
        draft['items']['h_series_baseline_supplement'],
        'SDS §4.3 / DSH-Q018', subcap='S2_0')
    add('h_series_candidate_retest_3_models',
        'six H candidate retest for up to 3 final candidates', 'evaluation',
        draft['items']['h_series_candidate_retest_3_models'], 'SDS §4.3')
    add('l40_milestone_cache_gate_A',
        '§6.1.1 L40 milestone frozen comparison cache A', 'evaluation',
        l40['per_suite_samples'], 'SDS §6.1.1 / DSH-Q021')
    add('l40_milestone_cache_gate_B',
        '§6.1.1 L40 milestone frozen comparison cache B', 'evaluation',
        l40['per_suite_samples'], 'SDS §6.1.1 / DSH-Q021')
    add('gate_admissions_up_to_40', 'up to 40 gate admissions (worst course)',
        'evaluation', draft['items']['gate_admissions_up_to_40'],
        'SDS §2.5 / §6.1.1 / DSH-Q021')
    add('final_sets_3_with_frozen_comparisons',
        'three final sets: candidate + S1 + original 6400 flat', 'evaluation',
        draft['items']['final_sets_3_with_frozen_comparisons'],
        'SDS §6.5 / DSH-Q021')
    add('waiting_empty_steps_no_extra_charge',
        'completed-slot waiting/padding is embedded, not double-charged',
        'engineering', 0, 'SDS §2.5 (embedded in each batch upper bound)',
        embedded_in_batch_bounds=True)
    nonlearning_total = sum(row['control_samples'] for row in items)
    learning_items = [
        dict(obligation_id=f'learning_{course.replace(".", "_")}',
             name=f'{course} learning allocation', category='learning',
             control_samples=int(cap), basis='SDS §2.5 / §6.2-6.4')
        for course, cap in s2_spec.S2_COURSE_LEARNING_CAPS.items()]
    learning_total = sum(row['control_samples'] for row in learning_items)
    plan = dict(
        schema_version=s2_spec.CHECKPOINT_SCHEMA, stage='S2',
        course='S2.0-S2.3', sds_sha256=s2_spec.SDS_SHA256,
        sds_version='v1.2', namespace=s2_spec.LEDGER_NAMESPACE,
        s2_0_manifest_sha256=s2_0_manifest['manifest_sha256'],
        h_manifest_sha256=h_manifest['manifest_sha256'],
        items=items, learning_items=learning_items,
        total_control_samples=nonlearning_total,
        learning_total_control_samples=learning_total,
        stage_total_control_samples=nonlearning_total + learning_total,
        physics_substeps=4 * (nonlearning_total + learning_total),
        billing=dict(
            unit='environment control samples',
            padding_included=True, waiting_empty_steps_billed=True,
            final_reset_frames_excluded=True,
            s1_samples_reused=False, double_counting=False,
            probe=dict(gpu_peak_mib_cap=7168, host_free_mib_min=4096,
                       warmups=5, measurements=50, restore_checks=1,
                       independent_copy=True, evaluation_envs=100),
            note='each item is a worst-case upper bound; actual commitments '
                 'settle against the obligation limits'),
        s2_0_subcap=s2_spec.S2_0_NONLEARNING_SUBCAP,
        s2_0_subcap_used=sum(row['control_samples'] for row in items
                             if row.get('subcap') == 'S2_0'),
        nonlearning_cap=s2_spec.S2_NONLEARNING_CAP,
        learning_cap=s2_spec.S2_LEARNING_CAP,
        total_cap=s2_spec.S2_TOTAL_CAP,
        admitted=True,
    )
    plan['plan_sha256'] = sha256({key: value for key, value in plan.items()
                                  if key != 'plan_sha256'})
    return plan
