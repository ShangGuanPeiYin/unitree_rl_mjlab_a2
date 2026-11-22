#!/usr/bin/env python3
"""CPU tests for the A2 stairs v0.4 T6 reward counterexample/ranking diagnostic.

The test imports the bootstrap before any ``src.tasks`` module, then verifies:

* the exact SDS T6 matrix/event timing and every required ranking;
* an independent recomputation of all required scores and event timings;
* the required counterexamples and synthetic extra checks;
* that a corrupted score cannot be hidden by the reporting layer;
* that a tampered event time is rejected;
* that old replay comparison is partial-only and never presented as a full
  v0.4 return.

Run with::

    CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python -m unittest \
        tests/test_v04_reward_analysis.py -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import a2_v04_bootstrap  # noqa: E402

a2_v04_bootstrap.activate(str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'diagnostics'))
import analyze_a2_v04_rewards as analysis  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.v04 import rewards, spec  # noqa: E402


def _ceil_fraction(value):
    return -(-value.numerator // value.denominator)


def _reference_trace(c, N, m, offsets, gate_mode):
    """Independent implementation of the T6 timing/gate/score rules."""
    dt = Fraction('0.02')
    vx_fraction = Fraction(str(m)) * Fraction(str(c))
    distance = Fraction(1) + Fraction(N) * Fraction('0.2') + Fraction('0.8')
    T = _ceil_fraction(distance / (vx_fraction * dt))
    gate = [True] * T
    landings = []
    swings = []
    for k in range(1, N + 1):
        for foot, offset in enumerate(offsets):
            off = Fraction(str(offset))
            L = max(1, _ceil_fraction(
                (Fraction(1) + Fraction(k - 1) * Fraction('0.2')
                 + Fraction('0.10') + off) / (vx_fraction * dt)))
            U = max(1, _ceil_fraction(
                (Fraction(1) + Fraction(k - 1) * Fraction('0.2')
                 - Fraction('0.05') + off) / (vx_fraction * dt)))
            if L > T or U > T:
                raise AssertionError('reference event outside trace')
            landings.append((L, foot, k))
            swings.append((U, foot, k))
    if gate_mode == 'C':
        for U, _, _ in swings:
            gate[U - 1] = False
            gate[min(U + 1, T) - 1] = False
    elif gate_mode == 'E':
        for L, _, _ in landings:
            gate[L - 1] = False
            gate[min(L + 1, T) - 1] = False

    config = spec.reward_config()
    footfall_amount = float(config['footfall']['amount'])
    swing_amount = float(config['swing']['amount_per_target'])
    success_amount = float(config['events']['success'])
    vx = float(m) * float(c)
    speed = rewards.speed_reward(
        torch.tensor([vx] * T, dtype=torch.float64),
        torch.zeros(T, dtype=torch.float64),
        torch.full((T,), float(c), dtype=torch.float64),
        torch.tensor(gate, dtype=torch.bool),
        dt=float(config['dt'])).tolist()
    amounts = [0.0] * T
    timing = []
    for L, foot, k in landings:
        amounts[L - 1] += footfall_amount
        timing.append([L, analysis.EVENT_FOOTFALL, foot, k])
    paid_swings = 0
    consumed_swings = 0
    for U, foot, k in swings:
        timing.append([U, analysis.EVENT_SWING, foot, k])
        if gate[U - 1]:
            amounts[U - 1] += swing_amount
            paid_swings += 1
        else:
            consumed_swings += 1
    amounts[T - 1] += success_amount
    timing.append([T, analysis.EVENT_SUCCESS, -1, -1])
    timing.sort()
    score = math.fsum(
        (analysis.GAMMA ** index) * (speed[index] + amounts[index])
        for index in range(T))
    return {
        'T': T,
        'score': score,
        'timing': timing,
        'paid_swings': paid_swings,
        'consumed_swings': consumed_swings,
    }


def _fixture_row(vx, x, max_body_x, progress, total):
    return {
        'stage': 'S1',
        'diagnostics': {
            'stage': 'S1',
            'rewards': {
                'reward_raw/track_linear_velocity': 2.0,
                'reward_weighted/track_linear_velocity': 0.04,
                'reward_raw/orientation': 0.0,
                'reward_weighted/orientation': 0.0,
                'reward_raw/nonfoot': 0.0,
                'reward_weighted/nonfoot': 0.0,
                'reward_progress': progress,
                'reward_event/footfall': 0.0,
                'reward_event/success': 0.0,
                'reward_event/failure': 0.0,
                'reward_total': total,
            },
            'linear_velocity': [vx, 0.0, 0.0],
            'linear_velocity_body': [vx, 0.0, 0.0],
            'command': [vx, 0.0, 0.0],
            'command_executed': [vx, 0.0, 0.0],
            'quaternion': [1.0, 0.0, 0.0, 0.0],
            'nonfoot_force': [[0.0, 0.0, 0.0] for _ in range(9)],
            'foot_force': [[0.0, 0.0, 0.0] for _ in range(4)],
            'support': [True, True, False, False],
            'support_step': [0, 0, 0, 0],
            'stop_latched': False,
            'failure': False,
            'success': False,
            'truncated': False,
            'reason_code': 0,
            'position': [x, 0.0, 0.4],
            'local_position': [x, 0.0, 0.4],
            'max_body_x': max_body_x,
            'pitch': 0.0,
            'yaw': 0.0,
            'world_vx': vx,
        },
    }


class RewardAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analysis.analyze()
        cls.report_json = analysis.canonical_json(cls.report)

    def test_bootstrap_is_active_and_isolated(self):
        self.assertTrue(a2_v04_bootstrap.is_active(str(PROJECT_ROOT)))
        tasks = sys.modules['src.tasks']
        self.assertIsNone(tasks.__file__)
        self.assertEqual(getattr(tasks, '__a2_v04_bootstrap_root__'),
                         str(PROJECT_ROOT))

    def test_config_sha_and_assertions(self):
        report = self.report
        self.assertEqual(report['versions']['reward_config_sha256'],
                         spec.reward_config_sha256())
        self.assertTrue(report['config_assertions']['pass'])
        self.assertEqual(report['config_assertions']['defects'], [])
        for record in report['config_assertions']['checks']:
            self.assertTrue(record['pass'], record)
            self.assertIsNotNone(record['line'], record)
        self.assertEqual(report['versions']['dt'], 0.02)
        self.assertEqual(report['versions']['gamma'], 0.99)

    def test_input_file_shas_are_current(self):
        records = {record['role']: record for record in self.report['input_files']}
        for role in ('sds', 'reward_spec', 'reward_functions', 'bootstrap',
                     'frozen_footfall_state', 'analysis_script'):
            record = records[role]
            self.assertTrue(record['present'], role)
            payload = (PROJECT_ROOT / record['path']).read_bytes()
            self.assertEqual(record['sha256'],
                             hashlib.sha256(payload).hexdigest(), role)

    def test_exact_matrix_dimensions_and_counts(self):
        matrix = self.report['matrix']
        self.assertEqual(set(matrix['variants']), {'A', 'B', 'C', 'D', 'E'})
        self.assertEqual(matrix['definition']['required_variants'], ['A', 'B', 'C'])
        self.assertEqual(matrix['definition']['extra_synthetic_variants'], ['D', 'E'])
        expected_cells = {(c, N) for c in (0.2, 0.4, 0.6)
                          for N in (4, 8, 12, 20, 40)}
        for name, variant in matrix['variants'].items():
            self.assertEqual(variant['summary']['cells'], 15, name)
            self.assertEqual(variant['summary']['traces'], 75, name)
            cells = {(cell['c'], cell['N']) for cell in variant['cells']}
            self.assertEqual(cells, expected_cells, name)
            for cell in variant['cells']:
                self.assertEqual([trace['m'] for trace in cell['traces']],
                                 list(analysis.M_VALUES), (name, cell['c'], cell['N']))
                self.assertEqual(set(trace['m'] for trace in cell['traces']),
                                 set(analysis.M_VALUES))
                for trace in cell['traces']:
                    self.assertEqual(trace['T'],
                                     analysis.control_samples(cell['c'], cell['N'],
                                                              trace['m']))
                    self.assertEqual(trace['event_counts']['footfall'],
                                     4 * cell['N'])
                    self.assertEqual(trace['event_counts']['swing_events'],
                                     4 * cell['N'])
                    self.assertEqual(trace['event_counts']['success'], 1)
                    if name == 'C':
                        self.assertEqual(trace['event_counts']['swing_paid'], 0)
                        self.assertEqual(trace['event_counts']['swing_consumed_unpaid'],
                                         4 * cell['N'])
                    elif name == 'E':
                        self.assertEqual(
                            trace['event_counts']['swing_paid']
                            + trace['event_counts']['swing_consumed_unpaid'],
                            4 * cell['N'])
                    else:
                        self.assertEqual(trace['event_counts']['swing_paid'],
                                         4 * cell['N'])
                    self.assertTrue(trace['timing_check']['pass'])
                    self.assertLess(trace['speed_formula_max_abs_error'], 1e-12)

    def test_required_rankings_and_comparison_count(self):
        matrix = self.report['matrix']
        self.assertTrue(matrix['required_ordering_pass'])
        self.assertTrue(matrix['extra_ordering_pass'])
        self.assertTrue(matrix['timing_integrity_pass'])
        required_comparisons = sum(
            variant['summary']['required_comparisons']
            for variant in matrix['variants'].values() if variant['required'])
        self.assertEqual(required_comparisons, 45 * 3)
        required_traces = sum(
            variant['summary']['traces']
            for variant in matrix['variants'].values() if variant['required'])
        self.assertEqual(required_traces, 3 * 75)
        for name in ('A', 'B', 'C'):
            for cell in matrix['variants'][name]['cells']:
                scores = {trace['m']: trace['score'] for trace in cell['traces']}
                for other in (0.5, 1.2, 2.0):
                    self.assertGreater(scores[1.0], scores[other] + 1e-6,
                                       (name, cell['c'], cell['N'], other))
                ordering = cell['ordering']
                self.assertTrue(ordering['pass'])
                for record in ordering['required']:
                    self.assertTrue(record['pass'])
                    self.assertGreater(record['margin'], record['tolerance'])
                # SDS T6 deliberately does not constrain m=1.05 against m=1.
                self.assertFalse(ordering['m105_relation']['required'])
                self.assertIn(ordering['m105_relation']['pass'], (True, False, None))

    def test_score_and_timing_fingerprints_match_report(self):
        matrix = self.report['matrix']
        score_payload = []
        timing_payload = []
        for name in ('A', 'B', 'C', 'D', 'E'):
            for cell in matrix['variants'][name]['cells']:
                for trace in cell['traces']:
                    score_payload.append([name, trace['c'], trace['N'], trace['m'],
                                          trace['score']])
                    timing_payload.append([name, trace['c'], trace['N'], trace['m'],
                                           trace['T'], trace['event_timing']])
        self.assertEqual(matrix['score_fingerprint_sha256'],
                         analysis.sha256_bytes(
                             analysis.canonical_json(score_payload).encode('utf-8')))
        self.assertEqual(matrix['timing_fingerprint_sha256'],
                         analysis.sha256_bytes(
                             analysis.canonical_json(timing_payload).encode('utf-8')))

    def test_independent_score_and_timing_recomputation(self):
        matrix = self.report['matrix']
        for name, offsets, gate_mode in (
                ('A', analysis.OFFSETS_A, 'A'),
                ('B', analysis.OFFSETS_B, 'B'),
                ('C', analysis.OFFSETS_A, 'C')):
            for cell in matrix['variants'][name]['cells']:
                for trace in cell['traces']:
                    reference = _reference_trace(cell['c'], cell['N'], trace['m'],
                                                 offsets, gate_mode)
                    self.assertAlmostEqual(trace['score'], reference['score'],
                                           delta=1e-9,
                                           msg=(name, cell['c'], cell['N'], trace['m']))
                    self.assertEqual(trace['event_timing'], reference['timing'])
                    self.assertEqual(trace['event_counts']['swing_paid'],
                                     reference['paid_swings'])
                    self.assertEqual(
                        trace['event_counts']['swing_consumed_unpaid'],
                        reference['consumed_swings'])

    def test_known_golden_scores(self):
        # Values are the exact float64 results of this frozen 0.4-r1 matrix.
        goldens = {
            ('A', 0.2, 4): {0.5: 0.37143087232118677, 1.0: 1.2602218937830663,
                            1.05: 1.2916364242185392, 1.2: 0.6972150562726027,
                            2.0: -2.1487236100923672},
            ('A', 0.6, 40): {0.5: 1.2736656222509928, 1.0: 4.8348256287734195,
                             1.05: 5.161234994902296, 1.2: 4.275102274617364,
                             2.0: -0.12780311716574136},
            ('C', 0.2, 4): {0.5: 0.36104418491349144, 1.0: 1.152632201137275,
                            1.05: 1.1709922918714832, 1.2: 0.5356325307174812,
                            2.0: -2.5144226093162314},
            ('B', 0.4, 8): None,
        }
        for (name, c, N), expected in goldens.items():
            cell = next(cell for cell in self.report['matrix']['variants'][name]['cells']
                        if cell['c'] == c and cell['N'] == N)
            if expected is None:
                # B is checked by the independent recomputation; no duplicate golden.
                continue
            scores = {trace['m']: trace['score'] for trace in cell['traces']}
            for m, value in expected.items():
                self.assertAlmostEqual(scores[m], value, delta=1e-9,
                                       msg=(name, c, N, m))

    def test_counterexamples(self):
        counter = self.report['counterexamples']
        self.assertTrue(counter['all_assertions_pass'])
        self.assertFalse(counter['dynamics_feasible'])
        self.assertFalse(counter['training_improvement_claimed'])
        for key in ('stationary', 'front_lift_no_landing',
                    'body_pitch_no_landing', 'body_roll_no_landing',
                    'same_step_loop'):
            self.assertTrue(counter[key]['pass'], key)
            self.assertTrue(counter[key]['synthetic'], key)
        self.assertEqual(counter['stationary']['new_total'], 0.0)
        self.assertEqual(counter['stationary']['counts']['footfall'], 0)
        self.assertEqual(counter['front_lift_no_landing']['counts']['footfall'], 0)
        self.assertEqual(counter['front_lift_no_landing']['counts']['swing_paid'], 2)
        self.assertEqual(counter['body_pitch_no_landing']['counts']['footfall'], 0)
        self.assertEqual(counter['body_roll_no_landing']['counts']['footfall'], 0)
        loop = counter['same_step_loop']
        self.assertEqual(loop['repeat_attempts_fl_step1'], 5)
        self.assertAlmostEqual(loop['paid_amount'], 0.75, delta=1e-12)
        self.assertAlmostEqual(loop['naive_amount_all_landings'], 1.75, delta=1e-12)
        self.assertGreater(loop['naive_amount_all_landings'], loop['paid_amount'])
        sweep = counter['success_speed_sweep']
        self.assertTrue(sweep['all_required_pass'])
        for cell in sweep['cells']:
            self.assertTrue(cell['target_beats_0_5'])
            self.assertTrue(cell['target_beats_1_2'])
            self.assertTrue(cell['target_beats_2'])
            self.assertIn(cell['m105_vs_m1'], ('above', 'below', 'tie'))
        self.assertFalse(sweep['deadband_note'].lower().startswith('required'))

    def test_tampered_timing_is_rejected(self):
        trace = analysis.build_trace(0.2, 4, 1.0, analysis.OFFSETS_A, 'A', 'A')
        self.assertTrue(analysis.verify_trace_timing(trace)['pass'])
        tampered = copy.deepcopy(trace)
        tampered['events'][0]['t'] += 1
        check = analysis.verify_trace_timing(tampered)
        self.assertFalse(check['pass'])
        self.assertTrue(check['mismatches'])
        with self.assertRaises(analysis.TimingIntegrityError):
            analysis.score_trace(tampered)

    def test_swapped_timings_are_rejected(self):
        trace = analysis.build_trace(0.4, 8, 1.2, analysis.OFFSETS_B, 'B', 'B')
        self.assertTrue(analysis.verify_trace_timing(trace)['pass'])
        tampered = copy.deepcopy(trace)
        # Swap two events with distinct times; a timing swap must never be used
        # to make the ordering pass.
        first = tampered['events'][0]
        last = tampered['events'][-2]
        self.assertNotEqual(first['t'], last['t'])
        first['t'], last['t'] = last['t'], first['t']
        check = analysis.verify_trace_timing(tampered)
        self.assertFalse(check['pass'])
        self.assertTrue(check['mismatches'])
        with self.assertRaises(analysis.TimingIntegrityError):
            analysis.score_trace(tampered)

    def test_failure_cannot_be_hidden_by_reporting(self):
        def corrupt(trace):
            scored = analysis.score_trace(trace)
            if (trace['variant'] == 'A' and trace['c'] == 0.2
                    and trace['N'] == 4 and trace['m'] == 1.0):
                scored = dict(scored)
                scored['score'] -= 10.0
            return scored

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'bad_report.json'
            report, digest = analysis.execute(output, include_trace=False,
                                              score_fn=corrupt)
            self.assertEqual(report['status']['exit_code'], 2)
            self.assertFalse(report['status']['required_ordering_pass'])
            self.assertFalse(report['status']['pass'])
            self.assertTrue(report['matrix']['failures'])
            failure = next(item for item in report['matrix']['failures']
                           if item['variant'] == 'A' and item['c'] == 0.2
                           and item['N'] == 4)
            self.assertTrue(failure['required'])
            self.assertLess(failure['scores'][1.0], failure['scores'][2.0])
            self.assertTrue(any(item['kind'] == 'required_ordering_failure'
                                for item in report['defects']))
            self.assertTrue(output.is_file())
            parsed = json.loads(output.read_text(encoding='utf-8'))
            self.assertFalse(parsed['status']['required_ordering_pass'])
            sidecar = output.with_suffix('.sha256')
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar.read_text(encoding='utf-8').split()[0], digest)

    def test_old_trace_comparison_is_partial_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for vx in (0.2, 0.4, 0.6):
                directory = root / f'vx_{vx}'
                directory.mkdir()
                rows = []
                for index in range(3):
                    x = -1.0 + index * vx * 0.02
                    progress = 2.0 * vx * 0.02 if index > 0 else 0.0
                    rows.append(json.dumps(
                        _fixture_row(vx, x, x, progress, 0.04 + progress),
                        sort_keys=True))
                (directory / 'steps_env0.jsonl').write_text(
                    '\n'.join(rows) + '\n', encoding='utf-8')
            comparison = analysis.compare_traces(root, verify_identity=False)
            self.assertEqual(comparison['status'], 'completed')
            self.assertTrue(comparison['old_terms_all_pass'])
            self.assertTrue(comparison['complete_speed_set'])
            self.assertFalse(comparison['full_v04_return_reconstructable'])
            self.assertTrue(comparison['partial_only'])
            fields = {item['field'] for item in comparison['missing_fields']}
            self.assertTrue({'swing', 'safe_support', 'contact_truth'} <= fields)
            self.assertEqual(len(comparison['files']), 3)
            for file_report in comparison['files']:
                self.assertTrue(file_report['old_terms_pass'])
                for term, stats in file_report['old_terms'].items():
                    self.assertTrue(stats['pass'], (file_report['vx'], term, stats))
                reconstruct = file_report['reconstructability']
                self.assertFalse(reconstruct['full_v04_return_reconstructable'])
                self.assertFalse(reconstruct['gate_reconstructable'])
                self.assertFalse(reconstruct['safe_support_reconstructable'])
                self.assertFalse(reconstruct['swing_reconstructable'])
                self.assertFalse(reconstruct['contact_truth_reconstructable'])
                self.assertNotIn('new_total', file_report)
                self.assertNotIn('v04_return', file_report)
                new_terms = file_report['new_terms']
                self.assertIn('speed_true_bounds', new_terms)
                self.assertEqual(len(new_terms['speed_true_bounds']['undiscounted']), 2)
                self.assertIn('proxy', new_terms['speed_proxy_legacy_support_gate']['note'])
                self.assertEqual(file_report['replay_identity']['pass'], None)
                self.assertFalse(file_report['replay_identity']['enforced'])

    def test_deterministic_subset_report(self):
        first = analysis.analyze(variant_names=('A',))
        second = analysis.analyze(variant_names=('A',))
        self.assertEqual(analysis.canonical_json(first),
                         analysis.canonical_json(second))
        self.assertNotIn('generated_at', first)
        self.assertNotIn('timestamp', first)

    def test_report_is_machine_readable_and_json_clean(self):
        parsed = json.loads(self.report_json)
        self.assertEqual(parsed['schema'], analysis.ANALYSIS_SCHEMA)
        self.assertTrue(parsed['status']['pass'])
        self.assertEqual(parsed['status']['exit_code'], 0)
        self.assertEqual(parsed['defects'], [])
        self.assertFalse(parsed['scope']['gpu_used'])
        self.assertFalse(parsed['scope']['network_used'])
        self.assertFalse(parsed['scope']['training_used'])
        self.assertFalse(parsed['scope']['dynamics_feasibility_claimed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
