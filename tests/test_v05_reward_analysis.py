#!/usr/bin/env python3
"""CPU tests for the A2 stairs v0.5 reward counterexample / time-scaling analysis.

The test imports the bootstrap before any ``src.tasks`` module (the analysis
module does the same at its own import time), then verifies the invariants the
delivery depends on:

* the seven SDS 9.3 counterexamples exist and the three necessary conditions
  hold (matched speed above standing still, the repeated-lift allowance at most
  0.60, the early fall below standing still);
* the three SDS 9.3 reference scalars (3.973384 / -2.482749 / -6.839384) are
  reproduced from the fixed 20 s / 1000-step / c=0.4 case within 1e-5;
* the time-scaling grid has the exact 3x6x5 = 90 points and each point's
  ``t = ceil((x+1)/(v*dt)-1e-9)`` / ``gamma**(t-1)`` discount matches an
  independent recomputation;
* the acknowledged N=40/c=0.4 over-speed trade-off is reported as measured and
  never turned into a fabricated passing order;
* both trace sections list their missing fields explicitly and never claim a
  full swing/stall state return;
* the report is reproducible: two runs over the same retained inputs produce the
  same digest and the same JSON payload apart from wall-clock bookkeeping.

Run with::

    CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python -m pytest \
        tests/test_v05_reward_analysis.py -q
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT / 'scripts', PROJECT_ROOT / 'scripts' / 'diagnostics',
              PROJECT_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(str(PROJECT_ROOT))

import analyze_a2_v05_rewards as analysis  # noqa: E402

REQUIRED_CASES = ('stationary', 'slow_drag', 'overspeed_impact',
                  'pitch_forward_no_landing', 'repeated_lift', 'normal_ascent',
                  'early_fall')


def _by_name(report):
    section = report.get('counterexamples', report)
    return {case['name']: case for case in section['cases']}


def test_seven_counterexamples_and_required_invariants():
    report = analysis.counterexamples()
    cases = _by_name(report)
    assert set(cases) == set(REQUIRED_CASES)
    assert len(cases) == 7
    for name in REQUIRED_CASES:
        case = cases[name]
        assert case['horizon_steps'] == analysis.STEPS
        assert case['terms_undiscounted']
        assert math.isfinite(case['total_undiscounted'])
        assert math.isfinite(case['total_discounted'])
        assert case['dt'] == analysis.DT
        assert case['gamma'] == analysis.GAMMA
        assert case['terminal_observation_bootstrap'] is False
    invariants = report['invariants']
    assert invariants['matched_above_stationary']['passed'] is True
    assert invariants['normal_ascent_above_stationary']['passed'] is True
    assert invariants['repeated_lift_allowance']['passed'] is True
    assert invariants['early_failure_below_stationary']['passed'] is True
    assert invariants['repeated_lift_allowance']['swing_undiscounted'] <= 0.60 + 1e-12
    assert cases['early_fall']['total_discounted'] < cases['stationary']['total_discounted']
    assert (cases['normal_ascent']['total_discounted']
            > cases['stationary']['total_discounted'])
    assert report['all_required_passed'] is True


def test_reference_scalars_match_sds_anchor():
    report = analysis.counterexamples()
    references = report['references']
    assert abs(references['matched_speed'] - 3.973384) <= 1e-5
    assert abs(references['stationary'] - (-2.482749)) <= 1e-5
    assert abs(references['early_failure'] - (-6.839384)) <= 1e-5
    for name, check in report['reference_check'].items():
        assert check['passed'], (name, check)


def test_discount_conventions_and_event_step_rule():
    assert analysis.discount_at(1) == 1.0
    assert analysis.discount_at(2) == 0.995
    assert analysis.discount_sum(1) == 1.0
    steps = 7
    assert abs(analysis.discount_sum(steps)
               - math.fsum(0.995 ** index for index in range(steps))) <= 1e-12
    # t = ceil((x+1)/(v*dt) - 1e-9) with the exact rational ceiling.
    assert analysis._ceil_step(1.2, 0.4) == 150
    assert analysis._ceil_step(9.8, 0.4) == 1225
    assert analysis._ceil_step(0.4, 0.4) == 50
    fraction = Fraction('9.8') / (Fraction('0.4') * Fraction('0.02'))
    assert analysis._ceil_step(9.8, 0.4) == -(-fraction.numerator // fraction.denominator)
    # An exact boundary must not be pushed to the next step by the 1e-9 offset.
    assert analysis._ceil_step(1.0, 0.4) == 125


def test_stationary_discounted_total_recomputed_independently():
    report = analysis.counterexamples()
    stationary = _by_name(report)['stationary']
    expected = (analysis.speed_amount(0.0, 0.0, analysis.C, False)
                * analysis.discount_sum(analysis.STEPS)
                + analysis.spec.STALL_REWARD
                * math.fsum(analysis.discount_at(step)
                            for step in range(99, analysis.STEPS + 1)))
    assert abs(stationary['total_discounted'] - expected) <= 1e-9
    early = _by_name(report)['early_fall']
    expected_early = (analysis.speed_amount(0.0, 0.0, analysis.C, False)
                      * analysis.discount_sum(100)
                      + analysis.spec.STALL_REWARD
                      * math.fsum(analysis.discount_at(step) for step in (99, 100))
                      + analysis.spec.FAILURE_AMOUNT * analysis.discount_at(100))
    assert abs(early['total_discounted'] - expected_early) <= 1e-9
    assert early['terminal_step'] == 100


def test_speed_term_keeps_negative_part_when_gate_closed():
    assert abs(analysis.speed_rate(0.0, 0.0, analysis.C, False)
               - analysis.spec.STILL_SPEED_RATE) <= 1e-12
    assert analysis.speed_rate(analysis.C, 0.0, analysis.C, False) == 0.0
    assert analysis.speed_rate(analysis.C, 0.0, analysis.C, True) == 1.0
    for row in analysis.speed_rate_table():
        assert row['passed'], row


def test_time_scaling_grid_point_count_and_discounts():
    scaling = analysis.time_scaling()
    grid = scaling['grid']
    assert grid['expected_cell_count'] == 3 * 6 * 5 == 90
    assert grid['cell_count'] == 90 == len(scaling['cells'])
    assert grid['c_values'] == [0.2, 0.4, 0.6]
    assert grid['N_values'] == [1, 4, 8, 12, 20, 40]
    assert grid['ratio_values'] == [0.5, 1.0, 1.2, 1.5, 2.0]
    seen = set()
    for cell in scaling['cells']:
        key = (cell['c'], cell['N'], cell['ratio'])
        assert key not in seen
        seen.add(key)
        expected_t = analysis._ceil_step(analysis.STAIR_STEP_DEPTH * cell['N'] + 0.8
                                         - analysis.CLIMB_START_X,
                                         cell['ratio'] * cell['c'])
        assert cell['t_success'] == expected_t
        assert abs(cell['discount_success']
                   - analysis.GAMMA ** (cell['t_success'] - 1)) <= 1e-15
        assert len(cell['landing_steps']) == cell['N']
        assert len(cell['odd_landing_steps']) == (cell['N'] + 1) // 2
        expected_swing = (analysis.spec.SWING_AMOUNT_PER_TARGET * 4
                          * math.fsum(analysis.GAMMA ** (step - 1)
                                      for step in cell['landing_steps']))
        assert abs(cell['swing_full_discounted'] - expected_swing) <= 1e-12
        assert cell['swing_off_discounted'] == 0.0
        assert set(cell['variants']) == {f'{swing}|{rate:+.2f}'
                                        for swing in ('full', 'off', 'odd')
                                        for rate in (0.0, -0.25, -0.5)}
        for rate in ('+0.00', '-0.25', '-0.50'):
            assert cell['variants'][f'full|{rate}']['continuous'] == \
                cell['continuous_discounted'][rate]


def test_known_n40_c04_tradeoff_is_reported_not_faked():
    scaling = analysis.time_scaling()
    known = scaling['known_tradeoff']
    assert known['cell'] == 'c=0.4,N=40'
    assert known['sds_acknowledged'] is True
    assert known['blocking_for_0_5'] is False
    assert known['expected_above_matched'] == ['1.5', '2.0']
    row = known['rows'][0]
    assert row['matched_is_max'] is False
    assert {'1.5', '2.0'}.issubset(set(row['ratios_above_matched']))
    assert row['totals']['1.5'] > row['matched_total']
    assert row['totals']['2.0'] > row['matched_total']
    assert 'speed-optimal' in known['note']
    assert scaling['grid']['note'].startswith('idealised scalar grid')


def test_missing_fields_are_explicit_for_both_traces():
    report = analysis.analyze()
    for name in ('parking', 'high_speed'):
        fields = report['traces'][name]['missing_fields']
        assert len(fields) >= 5
        for entry in fields:
            assert entry['field']
            assert entry['reason']
    parking_fields = {entry['field'] for entry in report['traces']['parking']['missing_fields']}
    high_fields = {entry['field'] for entry in report['traces']['high_speed']['missing_fields']}
    assert 'terminal_observation' in parking_fields
    assert any(field.startswith('swing.') for field in parking_fields)
    assert 'body_vx[t]' in high_fields
    assert 'swing.*[t]' in high_fields
    assert report['checks']['traces_declare_missing'] is True


def test_parking_trace_is_partial_and_never_invents_swing_or_stall_state():
    report = analysis.analyze()
    parking = report['traces']['parking']
    assert parking['available'] is True
    assert parking['swing_terms_reconstructed'] is False
    assert 'swing' not in parking['reconstructible_terms']
    assert 'stall' in parking['reconstructible_terms']
    assert parking['stall_first_penalty_step'] == 99
    assert abs(parking['executed_c'] - 0.4) <= 1e-6
    assert parking['control_steps'] == 1266
    assert parking['timeout'] is True
    for name, value in parking['reconstructed_discounted'].items():
        assert math.isfinite(value), name
    assert 'bootstrap' in parking['note']


def test_high_speed_trace_recomputes_episode_level_speed_rate():
    report = analysis.analyze()
    high = report['traces']['high_speed']
    assert high['available'] is True
    assert high['failing_episodes'] >= 100
    assert 0.0 < high['negative_rate_fraction'] <= 1.0
    assert high['failure_reasons']
    for episode in high['episodes'][:50]:
        expected = analysis.speed_rate(episode['mean_vx'], 0.0,
                                       episode['executed_c'], True)
        assert abs(episode['speed_rate'] - expected) <= 1e-12
        assert abs(episode['speed_amount'] - 0.02 * expected) <= 1e-15
    assert 'bootstrap' in high['note']


def test_report_is_reproducible_for_the_same_inputs():
    first = analysis.analyze()
    second = analysis.analyze()
    assert analysis.report_digest(first) == analysis.report_digest(second) == first['digest']
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        report_one, json_one, markdown_one = analysis.execute(one)
        report_two, json_two, markdown_two = analysis.execute(two)
        assert report_one['digest'] == report_two['digest']
        stored_one = json.loads(Path(json_one).read_text())
        stored_two = json.loads(Path(json_two).read_text())
        for payload in (stored_one, stored_two):
            for key in analysis.VOLATILE_KEYS:
                payload.pop(key, None)
        assert stored_one == stored_two
        assert Path(markdown_one).read_text() == Path(markdown_two).read_text()
        assert report_one['digest'] in Path(markdown_one).read_text()


def test_report_checks_and_digest_shape():
    report = analysis.analyze()
    assert report['schema'] == analysis.ANALYSIS_SCHEMA
    assert report['all_checks_passed'] is True
    assert len(report['digest']) == 64
    assert int(report['digest'], 16) >= 0
    assert report['checks']['speed_truth_table'] is True
    assert all(report['checks']['signature_match'].values())
    assert report['reward_config_sha256'] == analysis.spec.reward_config_sha256()
    assert report['fixed']['amounts']['failure'] == -10.0
    assert report['fixed']['amounts']['footfall'] == 0.50
    assert report['fixed']['amounts']['swing'] == 0.15
