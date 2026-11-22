#!/usr/bin/env python3
"""CPU tests for the A2 stairs 0.5-r2 reward counterexample / anchor re-run (W2).

The test imports the bootstrap before any ``src.tasks`` module (the analysis
module does the same at its own import time), then verifies the invariants the
delivery depends on:

* the fixed counterexample set is exactly the Q003-r2 §4 "seven scenarios"
  (S1..S6 plus the two S7 failure variants) and every pass carries finite totals;
* the three r1 anchors (``3.9733841257`` / ``-2.4827493695`` / ``-6.8393842614``),
  the S3 rate ``0.738215683449317`` and the S3 discounted total
  ``-0.6548597670`` reproduce through the **real r2** pure functions/state
  machine within ``1e-6``;
* the real stall state machine charges S2/S7b first at control step 99 and S6 at
  step 384, never at the stopping frame;
* the required ordering facts hold (matched beats stationary, an early fall is
  worse than standing still, the 0.08 m/s side motion is net negative, stopping
  is worse than walking, correcting the yaw beats holding it);
* every missing term is declared explicitly (this table is **not** a complete
  training return);
* the evidence is bound to the r2 source closure SHA, the r2 contract SHA and
  both r2 decision SHAs, and the report digest is reproducible;
* the older defective tables are marked ``superseded`` (with unmodified bytes) and
  ``current_evidence=false``; r1 evidence is declared non-reusable.

Run with::

    CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
      scripts/test_a2_ideal_stairs_v05_r2.py --pattern 'test_v05_r2_reward_analysis.py'
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT / 'scripts', PROJECT_ROOT / 'scripts' / 'diagnostics',
              PROJECT_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v05_r2_bootstrap  # noqa: E402

a2_v05_r2_bootstrap.activate(str(PROJECT_ROOT))

import analyze_a2_v05_r2_rewards as analysis  # noqa: E402

REQUIRED_SCENARIOS = ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7a', 'S7b')
_CACHE = {}


def report():
    """One cached analysis run: the script drives the r2 functions twice (f32/f64).

    The r2 source closure is read once per process.  If another worker edits a bound
    r2 file inside that ~0.4 s window, the analysis flags
    ``source_closure_stable=false``; the snapshot is then dropped and read again so
    the cached report always describes one self-consistent source snapshot.
    """
    if 'report' not in _CACHE:
        value = analysis.analyze(write_index=True)
        if not value['provenance']['source_closure_stable']:
            analysis.reset_provenance_cache()
            value = analysis.analyze(write_index=True)
        _CACHE['report'] = value
    return _CACHE['report']


def _by_id(section):
    return {entry['id']: entry for entry in section['entries']}


def test_counterexample_set_is_the_fixed_seven_scenarios():
    entries = analysis.SCENARIOS
    assert tuple(scenario.sid for scenario in entries) == REQUIRED_SCENARIOS
    # Seven frozen scenario themes; S7 (the 100-step failure) has two variants.
    assert len(entries) == 8
    pairs = report()['passes']['float64']['entries']
    assert tuple(entry['id'] for entry in pairs) == REQUIRED_SCENARIOS
    for entry in pairs:
        assert entry['steps'] == (100 if entry['id'] in ('S7a', 'S7b') else 1000)
        assert entry['terminal_step'] == (100 if entry['id'] in ('S7a', 'S7b') else None)
        for key in ('undiscounted_mean', 'undiscounted_sum', 'discounted_sum'):
            assert math.isfinite(entry['total'][key]), (entry['id'], key)
        assert set(entry['terms']) == set(analysis.MODELED_TERMS)


def test_three_r1_anchors_reproduce_within_tolerance():
    checks = [item for item in report()['checks']
              if item['dtype'] == 'float64' and item['label'] in (
                  'S1 matched speed discounted', 'S2 stationary discounted',
                  'S7b early failure discounted')]
    assert len(checks) == 3
    for item in checks:
        assert item['passed'], item
        assert item['abs_error'] <= analysis.ANCHOR_TOLERANCE
    entries = _by_id(report()['passes']['float64'])
    assert abs(entries['S1']['total']['discounted_sum'] - 3.9733841257) <= 1e-6
    assert abs(entries['S2']['total']['discounted_sum'] - (-2.4827493695)) <= 1e-6
    assert abs(entries['S7b']['total']['discounted_sum'] - (-6.8393842614)) <= 1e-6


def test_s3_rate_and_discounted_total():
    torch = analysis.torch
    rate = float(analysis.r2_rewards.speed_rate(
        torch.tensor([analysis.C], dtype=torch.float64),
        torch.tensor([0.08], dtype=torch.float64),
        torch.tensor([analysis.C], dtype=torch.float64),
        torch.tensor([True]), enabled=torch.tensor([True])))
    assert abs(rate - 0.738215683449317) <= 1e-9
    # A hard-coded ey=0 would return exactly 1.0 for the same sample.
    no_side = float(analysis.r2_rewards.speed_rate(
        torch.tensor([analysis.C], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([analysis.C], dtype=torch.float64),
        torch.tensor([True]), enabled=torch.tensor([True])))
    assert abs(no_side - 1.0) <= 1e-12
    total = _by_id(report()['passes']['float64'])['S3']['total']['discounted_sum']
    assert abs(total - (-0.6548597670)) <= 1e-6
    float32_total = _by_id(report()['passes']['float32'])['S3']['total']['discounted_sum']
    assert abs(float32_total - (-0.6548597670)) <= analysis.FLOAT32_SUM_TOLERANCE


def test_stall_first_penalty_steps_use_the_real_state_machine():
    entries = _by_id(report()['passes']['float64'])
    assert entries['S2']['stall_state']['first_penalty_step'] == 99
    assert entries['S7b']['stall_state']['first_penalty_step'] == 99
    assert entries['S6']['stall_state']['first_penalty_step'] == 384
    # The 0.4 parking stop happens at step 288 and must not be charged there.
    assert entries['S6']['kinematics']['moving_steps'] == 288
    assert entries['S6']['stall_state']['first_penalty_step'] > 288
    assert entries['S6']['stall_state']['penalty_steps'] > 0
    assert entries['S2']['stall_state']['penalty_steps'] > 0
    for sid in ('S1', 'S3', 'S4', 'S5', 'S7a'):
        assert entries[sid]['stall_state']['first_penalty_step'] is None
    assert entries['S2']['stall_state']['window_samples'] == 51
    assert entries['S2']['stall_state']['window_intervals'] == 50
    assert entries['S2']['stall_state']['window_seconds'] == 1.0
    # Same first-penalty steps in the training dtype.
    float32_entries = _by_id(report()['passes']['float32'])
    for sid, expected in analysis.STALL_FIRST_STEP.items():
        assert float32_entries[sid]['stall_state']['first_penalty_step'] == expected


def test_required_ordering_holds():
    ordering = report()['ordering']
    assert all(item['passed'] for item in ordering.values()), ordering
    assert ordering['matched_above_stationary']['matched'] > \
        ordering['matched_above_stationary']['stationary']
    assert ordering['early_failure_below_stationary']['early_failure'] < \
        ordering['early_failure_below_stationary']['stationary']
    assert ordering['s3_negative']['s3'] < 0.0
    assert ordering['s6_stop_below_s1']['s6'] < ordering['s6_stop_below_s1']['s1']
    assert ordering['s5_turn_above_s4_hold']['s5'] > \
        ordering['s5_turn_above_s4_hold']['s4']


def test_missing_terms_are_explicit_and_scope_is_declared():
    missing = report()['missing_terms']
    assert missing['not_complete_return'] is True
    for term in ('track_angular_velocity', 'pose', 'orientation', 'body_ang_vel',
                 'angular_momentum', 'joint_acc', 'joint_limits', 'action_rate',
                 'slip', 'soft_landing', 'edge', 'nonfoot'):
        assert term in missing['regularizers_not_modeled']
    structurally = {item['term'] for item in missing['structurally_zero_here']}
    assert {'stand_still', 'footfall', 'swing', 'success'} <= structurally
    for term in ('timeout_bootstrap', 'failure_terminal_bootstrap',
                 'GAE_value_baseline', 'entropy_bonus', 'PPO_clip_ratio'):
        assert term in missing['rl_objective_not_modeled']
    relevant = missing['relevant_per_sequence']
    assert 'track_angular_velocity' in relevant['S5']
    assert 'failure_terminal_bootstrap' in relevant['S7a']
    assert 'failure_terminal_bootstrap' in relevant['S7b']
    statement = report()['scope_statement']
    assert '不是完整训练回报' in statement
    assert '局部合成收益' in statement


def test_evidence_is_bound_to_the_r2_source_contract_and_decisions():
    provenance = report()['provenance']
    assert provenance['chain'] == 'r2'
    for key in ('r2_source_sha256', 'r2_contract_sha256'):
        value = provenance[key]
        assert isinstance(value, str) and len(value) == 64
        assert int(value, 16) >= 0
    assert provenance['implementation_decision_sha256'].startswith('0cd996e7')
    assert provenance['design_decision_sha256'].startswith('78515bb9')
    assert provenance['sds_sha256'].startswith('9ff356b0')
    # The closure snapshot is self-consistent (two bracketing scans agree).
    assert provenance['source_closure_stable'] is True
    assert provenance['source_closure_divergent_paths'] == []
    # The contract SHA is the r2 spec value, not a re-declared constant.  A
    # concurrent edit by another worker is the only acceptable reason for a
    # difference, and then the analysis itself must have flagged it.
    recomputed = analysis.spec.contract_sha256()
    if recomputed != provenance['r2_contract_sha256']:
        assert report()['required_checks']['source_closure_stable_during_run'] is False
    else:
        assert provenance['r2_contract_sha256'] == recomputed
    # The closure SHA is the digest of the bound (path, sha256) source map.
    entries = sorted(provenance['r2_source_files'].items())
    expected = analysis.sha256_bytes(analysis.canonical(entries).encode('utf-8'))
    assert provenance['r2_source_sha256'] == expected
    assert provenance['r2_source_file_count'] == len(entries)
    assert 'src/tasks/velocity/v05_r2/rewards.py' in provenance['r2_source_files']
    assert 'scripts/diagnostics/analyze_a2_v05_r2_rewards.py' in provenance['r2_source_files']
    # The r1 namespace must never appear in the bound closure.
    assert not any(path.startswith('src/tasks/velocity/v05/')
                   for path in provenance['r2_source_files'])
    assert provenance['identity']['recipe'] == '0.5-r2'
    assert provenance['checkpoint_schema'] == 7
    assert provenance['architecture'] == 'v05-route-state-v1'
    payload = report()
    assert payload['digest'] == analysis.report_digest(payload)
    assert len(payload['digest']) == 64
    assert payload['all_required_passed'] is True


def test_old_tables_are_superseded_and_never_current_evidence():
    index = report()['index']
    assert index is not None
    assert index['r1_evidence_reusable'] is False
    statuses = {row['path']: row['status'] for row in index['historical']}
    superseded = ('logs/a2_ideal_stairs_v05/implementation/coordination/q003_followup/'
                  'r2_counterexamples.json')
    script = ('logs/a2_ideal_stairs_v05/implementation/coordination/q003_followup/'
              'r2_counterexamples.py')
    assert statuses[superseded] == 'superseded'
    assert statuses[script] == 'superseded'
    assert any(row['status'] == 'not_r2_evidence' for row in index['historical'])
    for row in index['historical']:
        assert row['current_evidence'] is False, row['path']
        assert row['overwritten_by_this_script'] is False
        assert row['reason']
        if row['documented_sha256'] is not None and row['exists']:
            assert row['observed_sha256'] == row['documented_sha256'], row['path']
            assert row['bytes_unchanged'] is True
    current = {entry['file'] for entry in index['current_evidence']}
    assert {'a2_v05_r2_reward_analysis.json', 'a2_v05_r2_reward_analysis.md',
            'analysis_index.json'} == current
    assert all(entry['current_evidence'] for entry in index['current_evidence'])
    assert index['prohibited']
    assert 'superseded' in index['statement']


def test_discount_conventions_and_additivity():
    assert analysis.discount_at(1) == 1.0
    assert analysis.discount_at(2) == 0.995
    assert analysis.discount_sum(1) == 1.0
    assert abs(analysis.discount_sum(7)
               - math.fsum(0.995 ** index for index in range(7))) <= 1e-12
    entries = _by_id(report()['passes']['float64'])
    for entry in entries.values():
        for key in ('undiscounted_sum', 'discounted_sum'):
            reconstructed = math.fsum(entry['terms'][term][key]
                                      for term in analysis.MODELED_TERMS)
            assert abs(reconstructed - entry['total'][key]) <= 1e-9, (entry['id'], key)
    # S1 is a constant dt*speed_rate at the matched speed, so its discounted total
    # is exactly dt * sum(gamma**(t-1)) and must not depend on the series length
    # beyond the horizon.
    matched = entries['S1']
    expected_s1 = analysis.DT * analysis.discount_sum(analysis.STEPS)
    assert abs(matched['total']['discounted_sum'] - expected_s1) <= 1e-9
    assert abs(expected_s1 - 3.9733841257) <= 1e-9
    # S2 is the stationary speed floor for the whole horizon plus the real stall
    # penalty from its first penalised step on.
    stationary = entries['S2']
    still_rate = float(analysis.r2_rewards.speed_rate(
        analysis.torch.tensor([0.0], dtype=analysis.torch.float64),
        analysis.torch.tensor([0.0], dtype=analysis.torch.float64),
        analysis.torch.tensor([analysis.C], dtype=analysis.torch.float64),
        analysis.torch.tensor([False]), enabled=analysis.torch.tensor([True]))[0])
    expected_still = (analysis.DT * still_rate * analysis.discount_sum(analysis.STEPS))
    assert abs(stationary['terms']['speed']['discounted_sum'] - expected_still) <= 1e-9
    # The r2 StallState keeps its per-step amount in torch.float32, so the 1000-step
    # stall sum carries that representation residue.  r1's counterexample report used
    # 1e-7 for the stall discounted-sum comparison; keep that convention and record
    # the exact float32 identity separately instead of loosening anything else.
    exact_stall = analysis.spec.STALL_REWARD
    expected_stall = exact_stall * math.fsum(
        analysis.discount_at(step) for step in range(99, 1001))
    stall_residue = abs(stationary['terms']['stall']['discounted_sum'] - expected_stall)
    assert stall_residue <= analysis.STALL_SUM_TOLERANCE, stall_residue
    assert analysis.STALL_SUM_TOLERANCE == 1e-7
    stall_per_step = float(analysis.torch.tensor(analysis.spec.STALL_REWARD,
                                                 dtype=analysis.torch.float32))
    assert abs(stall_per_step - exact_stall) < 1e-9
    assert abs(stationary['terms']['stall']['discounted_sum']
               - stall_per_step * math.fsum(analysis.discount_at(step)
                                            for step in range(99, 1001))) <= 1e-12
    early = entries['S7b']
    assert abs(early['terms']['failure']['discounted_sum']
               - analysis.spec.FAILURE_AMOUNT * analysis.discount_at(100)) <= 1e-9


def test_route_evidence_is_observation_only():
    route = report()['route']
    assert route['dim'] == 3
    assert route['key'] == 'route_error'
    assert abs(route['normalized_value_at_0_20_m'] - 1.0) <= 1e-6
    assert abs(route['normalized_value_at_0_80_m'] - 4.0) <= 1e-6
    assert route['yaw_axis'] == 0.0
    assert all(route['checks'].values()), route['checks']
    assert len(route['scenarios']) == len(REQUIRED_SCENARIOS)
    for row in route['scenarios']:
        assert len(row['route_error']) == 3
        assert all(math.isfinite(value) for value in row['route_error'])
    # route never appears as a reward term.
    assert 'route_error' not in analysis.MODELED_TERMS
    for entry in report()['passes']['float64']['entries']:
        assert 'route_error' not in entry['terms']


def test_report_is_reproducible_for_the_same_source():
    first = report()
    second = analysis.analyze(write_index=True)
    assert analysis.report_digest(first) == analysis.report_digest(second) == first['digest']
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        report_one, json_one, markdown_one, index_one = analysis.execute(one)
        report_two, json_two, markdown_two, index_two = analysis.execute(two)
        assert report_one['digest'] == report_two['digest']
        stored_one = json.loads(Path(json_one).read_text())
        stored_two = json.loads(Path(json_two).read_text())
        for payload in (stored_one, stored_two):
            for key in analysis.VOLATILE_KEYS:
                payload.pop(key, None)
        assert stored_one == stored_two
        # The index file records its own directory, everything else must match.
        index_payload_one = json.loads(Path(index_one).read_text())
        index_payload_two = json.loads(Path(index_two).read_text())
        index_payload_one.pop('directory', None)
        index_payload_two.pop('directory', None)
        assert index_payload_one == index_payload_two
        assert Path(markdown_one).read_text() == Path(markdown_two).read_text()
        assert report_one['digest'] in Path(markdown_one).read_text()
        # A second run into a populated directory must refuse to overwrite.
        try:
            analysis.execute(one)
        except SystemExit as error:
            assert 'already exists' in str(error)
        else:  # pragma: no cover - the guard must fire
            raise AssertionError('execute() overwrote an existing r2 analysis output')
