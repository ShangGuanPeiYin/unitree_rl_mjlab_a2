"""SDS 9.1 counterexample/curriculum contract for the v0.6 reward line.

This covers the CPU-only items owned by A that are not pure reward algebra:

* the H tick / 64-episode FIFO / EMA / cooldown / advance-vs-regress precedence
  constants and pure helpers (SDS 5.5, 9.1 items 13-14);
* the D/L cumulative substage budgets and the "old pass is invalid" rule
  (SDS 5.6, 9.1 item 15);
* the D1 seed offsets added exactly once (SDS 8.3, 9.1 item 18);
* the ten-scenario offline counterexample report (SDS 9.1 closing requirement),
  including its explicit "not counted" labels for missing real state.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
try:  # SDS 7.3: entry points bootstrap first; tolerate an already-imported package.
    import a2_v06_bootstrap

    a2_v06_bootstrap.activate(ROOT)
except (ImportError, RuntimeError):
    pass

from src.tasks.velocity.v06 import spec  # noqa: E402

SCRIPT = ROOT / 'scripts/diagnostics/analyze_a2_v06_rewards.py'


def load_analysis():
    module_spec = importlib.util.spec_from_file_location(
        'analyze_a2_v06_rewards', SCRIPT)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def outcome(step, env_id):
    return dict(step=step, env_id=env_id, success=False, failure=False,
                timeout=False, role='frontier', tile_id=1, tracking_score=0.,
                command=.4)


class CurriculumWindowTests(unittest.TestCase):
    def test_window_is_64_and_one_per_update(self):
        self.assertEqual(spec.CURRICULUM_WINDOW, 64)
        self.assertEqual(spec.CURRICULUM_WINDOWS_PER_UPDATE, 1)
        self.assertEqual(spec.CURRICULUM['window_scope'], 'frontier_role_only')
        self.assertEqual(spec.CURRICULUM['unfinished_windows'],
                         'retained_never_dropped')

    def test_fifo_orders_by_end_step_then_env_id(self):
        fifo, ready = [], False
        for step, env in ((5, 2), (3, 9), (5, 1), (4, 7)):
            fifo, ready = spec.fifo_append(fifo, outcome(step, env))
        self.assertEqual([(entry['step'], entry['env_id']) for entry in fifo],
                         [(3, 9), (4, 7), (5, 1), (5, 2)])
        self.assertFalse(ready)

    def test_partial_window_is_retained(self):
        fifo = [outcome(index, 0) for index in range(70)]
        window, remaining = spec.fifo_pop_window(fifo)
        self.assertEqual(len(window), 64)
        self.assertEqual(len(remaining), 6)
        # A second update consumes the retained remainder only once it is full.
        fifo = list(remaining)
        ready = False
        for _ in range(60):
            fifo, ready = spec.fifo_append(fifo, outcome(100, 0))
        self.assertTrue(ready)
        window, rest = spec.fifo_pop_window(fifo)
        self.assertEqual(len(window), 64)
        self.assertEqual(len(rest), 2)


class EmaAndTransitionTests(unittest.TestCase):
    def test_first_batch_initialises_then_ema(self):
        self.assertEqual(spec.curriculum_ema(None, .5), .5)
        self.assertAlmostEqual(spec.curriculum_ema(.5, .9), .97 * .5 + .03 * .9,
                               places=15)

    def test_advance_requires_five_ready_checks_and_zero_cooldown(self):
        tick, ready, hard, cooldown = 0, 0, 0, 0
        events = []
        for _ in range(5):
            result = spec.curriculum_transition(tick, .9, .9, .0, ready, hard, cooldown)
            tick = result['level_tick']
            ready, hard, cooldown = (result['ready_streak'], result['hard_streak'],
                                     result['cooldown'])
            events.append(result['event'])
        self.assertEqual(events, ['hold', 'hold', 'hold', 'hold', 'advance'])
        self.assertEqual(tick, 1)
        self.assertEqual(cooldown, 5)
        # During cooldown a ready streak rebuilds but cannot advance.
        result = spec.curriculum_transition(tick, .9, .9, .0, 4, 0, 3)
        self.assertEqual(result['event'], 'hold')
        self.assertEqual(result['cooldown'], 2)

    def test_regress_beats_advance_and_ignores_cooldown(self):
        result = spec.curriculum_transition(50, .9, .9, .45, 5, 1, 4)
        self.assertEqual(result['event'], 'regress')
        self.assertEqual(result['level_tick'], 47)
        self.assertEqual(result['cooldown'], 5)
        # Regress is `F>=0.40 OR S<=0.20` and needs two consecutive checks.
        first = spec.curriculum_transition(50, .2, .9, .0, 0, 0, 0)
        self.assertEqual(first['event'], 'hold')
        self.assertEqual(first['hard_streak'], 1)
        second = spec.curriculum_transition(50, .2, .9, .0, 0, 1, 0)
        self.assertEqual(second['event'], 'regress')

    def test_level_clamped_and_generation_row(self):
        self.assertEqual(spec.curriculum_transition(1, .0, .0, .9, 0, 1, 0)['level_tick'], 0)
        self.assertEqual(spec.curriculum_transition(99, .9, .9, .0, 4, 0, 0)['level_tick'], 100)
        self.assertEqual(spec.curriculum_frontier_row(0), 0)
        self.assertEqual(spec.curriculum_frontier_row(7), 1)
        self.assertEqual(spec.curriculum_frontier_row(100), 15)
        self.assertTrue(spec.generation_starts(0, 1))
        self.assertFalse(spec.generation_starts(1, 1))
        self.assertEqual(spec.CURRICULUM['generation']['old_generation'],
                         'log_only_not_counted')
        self.assertEqual(spec.CURRICULUM['generation']['clears'],
                         ('ema', 'fifo', 'ready_streak', 'hard_streak'))
        self.assertEqual(spec.CURRICULUM['terrain_commit'],
                         'full_update_boundary_only')


class SubstageTests(unittest.TestCase):
    def test_cumulative_budgets_and_no_borrowing(self):
        self.assertEqual(spec.D_SUBSTAGE_BUDGETS, (1500, 1500))
        self.assertEqual(spec.L_SUBSTAGE_BUDGETS, (1500, 1500, 3000))
        rules = spec.CURRICULUM['substages']
        self.assertEqual((rules['advance_streak'], rules['regress_streak']), (2, 2))
        self.assertTrue(rules['budgets_cumulative'])
        self.assertTrue(rules['regress_keeps_spent'])
        self.assertTrue(rules['no_borrowing'])
        self.assertTrue(rules['exhausted_return_stops'])
        self.assertTrue(rules['old_pass_invalid'])
        self.assertTrue(rules['no_cross_phase_regress'])
        self.assertTrue(rules['lowest_substage_holds'])
        self.assertEqual(rules['hard_condition'], 'same_as_H')

    def test_v_r_gate_streak_decoupled_from_ramp(self):
        self.assertEqual((spec.V_MIN_UPDATES, spec.V_GATE_STREAK_START),
                         (1200, 1000))
        self.assertEqual((spec.R_MIN_UPDATES, spec.R_GATE_STREAK_START),
                         (600, 400))
        self.assertTrue(spec.CURRICULUM['v']['gates_use_final_speeds_from_first_day'])
        self.assertTrue(spec.CURRICULUM['r']['gates_use_final_spawn_from_first_day'])
        self.assertTrue(spec.CURRICULUM['r']['flat_keeps_small_spawn'])

    def test_near_easy_sampling_and_v_r_anchors(self):
        self.assertEqual(spec.NEAR_OFFSETS, (-2, -1))
        self.assertEqual(spec.CURRICULUM['near']['k_below_2'], 'T[k]')
        self.assertEqual(spec.CURRICULUM['easy']['range'], 'T[0]..T[max(k-3,0)]')
        self.assertEqual(spec.V_R_FRONTIER_TILE_ID, 21)
        self.assertEqual(spec.V_R_NEAR_TILES, ('L12', 'L20'))
        self.assertEqual(spec.V_R_EASY_LAST_TILE_ID, 18)


class SeedOffsetTests(unittest.TestCase):
    def test_offset_added_exactly_once(self):
        raw = 42
        self.assertEqual(spec.evaluation_seed('validation', raw), 200342)
        self.assertEqual(spec.evaluation_seed('diagnostic', raw), 500342)
        self.assertEqual(spec.evaluation_seed('perturbed', raw), 300342)
        self.assertEqual(spec.evaluation_seed('final', raw), 400342)
        # Applying the offset twice would be 400642 / 700642 / ...: never legal.
        self.assertNotEqual(spec.evaluation_seed('validation', raw), 400642)
        self.assertNotEqual(spec.evaluation_seed('diagnostic', raw), 700642)
        self.assertEqual(spec.evaluation_seeds(raw), {
            'validation': 200342, 'diagnostic': 500342,
            'perturbed': 300342, 'final': 400342})
        # The final test seed is fixed, never raw plus an offset.
        self.assertEqual(spec.evaluation_seed('final', 0), 400342)
        self.assertEqual(spec.evaluation_seed('final', 100000), 400342)


class CounterexampleReportTests(unittest.TestCase):
    SCENARIOS = ('stationary', 'forward_normal', 'backward', 'lateral_drift',
                 'overspeed', 'swing_without_landing', 'repeated_stepping',
                 'normal_stair_climb', 'fall', 'near_goal_dawdle')

    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()
        cls.report = cls.analysis.build_report()

    def test_all_ten_scenarios_present_with_both_sums(self):
        names = tuple(row['name'] for row in self.report['scenarios'])
        self.assertEqual(names, self.SCENARIOS)
        self.assertEqual(self.report['scene_count'], 10)
        for row in self.report['scenarios']:
            self.assertGreater(row['steps'], 0)
            self.assertIsInstance(row['undiscounted_sum'], float)
            self.assertIsInstance(row['discounted_sum'], float)
            self.assertFalse(math.isnan(row['undiscounted_sum']))
            self.assertFalse(math.isnan(row['discounted_sum']))
            self.assertIn('forward_progress', row['per_term'])

    def test_missing_state_is_declared_not_zero_filled(self):
        for row in self.report['scenarios']:
            self.assertIn('joint_tracking', row['not_counted'])
            self.assertIn('similar_to_default', row['not_counted'])
            self.assertIn('joint_limits', row['not_counted'])
            self.assertEqual(sorted(row['counted']),
                             sorted(set(row['counted'])))
        stationary = self.report['scenarios'][0]
        self.assertIn('foot_slip', stationary['not_counted'])
        self.assertIn('foot_clearance', stationary['not_counted'])

    def test_local_invariants_hold(self):
        checks = self.report['invariants']['checks']
        self.assertTrue(self.report['invariants']['all_required_passed'], checks)
        self.assertTrue(checks['ten_scenarios'])
        self.assertTrue(checks['failure_minus_10'])
        self.assertTrue(checks['success_plus_10'])
        self.assertTrue(checks['forward_per_metre_0p4'])
        self.assertTrue(checks['stationary_tracking_rate_1p8465'])

    def test_script_writes_json_and_markdown_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            code = self.analysis.main(['--output-dir', str(output)])
            self.assertEqual(code, 0)
            json_path = output / 'v06_reward_counterexamples.json'
            md_path = output / 'v06_reward_counterexamples.md'
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            report = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(report['scene_count'], 10)
            self.assertEqual(report['sds']['sha256'], spec.SDS_SHA256)
            self.assertEqual(report['reward_config_sha256'],
                             spec.reward_config_sha256())
            text = md_path.read_text(encoding='utf-8')
            for name in self.SCENARIOS:
                self.assertIn(name, text)
            self.assertIn('未计入', text)


if __name__ == '__main__':
    unittest.main()
