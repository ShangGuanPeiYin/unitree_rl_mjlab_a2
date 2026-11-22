"""V0.5 identities, stage/budget table, slot allocation and explicit source closure.

The source contract is an explicit product-entry closure: it binds the bootstrap,
the migration script and every existing ``src/tasks/velocity/v05/*.py`` while
refusing to grow with unrelated task directories and never binding the legacy
``src/tasks/__init__.py`` scanner.  Synthetic temporary projects exercise that
closure without importing any task code.
"""
import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)

from src.tasks.velocity.v05 import spec  # noqa: E402

V05_FILES = {str(path.relative_to(spec.ROOT))
             for path in (spec.ROOT / 'src/tasks/velocity/v05').glob('*.py')}


def stable_contract(**kwargs):
    """Return a contract whose source closure was identical on two consecutive reads.

    Several workers share this checkout, so a file can be written between the
    contract hash and its verification.  Three identical back-to-back reads prove
    the closure was quiescent for a whole hash pass without weakening the check.
    """
    attempts = []
    for _ in range(3):
        first = spec.contract(**kwargs)
        second = spec.contract(**kwargs)
        if first == second:
            return first
        attempts.append(sorted(set(first['behavior_sha256'])
                               ^ set(second['behavior_sha256'])))
    raise AssertionError(f'the v0.5 source closure kept changing during the test: {attempts}')


class IdentityTests(unittest.TestCase):
    def test_version_identity_and_paths(self):
        self.assertEqual((spec.CODE_VERSION, spec.RECIPE_VERSION, spec.SDS_REVISION,
                          spec.SCHEMA_VERSION, spec.TASK, spec.PROTOCOL_ID,
                          spec.FLAT_PROTOCOL_ID, spec.LOG_ROOT),
                         ('0.5.0', '0.5-r1', '1.1', 6,
                          'Unitree-A2-IdealHeightMap-Stairs-V05',
                          'v05-stair-climb-first-v1', 'v05-flat-speed-v1',
                          'logs/a2_ideal_stairs_v05'))
        self.assertEqual((spec.CONTROL_DT, spec.PHYSICS_DT, spec.DECIMATION, spec.MAP_HZ,
                          spec.ACTION_SCALE), (.02, .005, 4, 10, .25))
        self.assertEqual(spec.ROOT, ROOT)
        self.assertEqual(spec.INIT_PATH,
                         '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt')
        self.assertEqual(spec.INIT_SHA256,
                         '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722')
        self.assertEqual(spec.sha256(spec.INIT_PATH), spec.INIT_SHA256)
        self.assertEqual(spec.OBSERVATION,
                         {'proprio': [45], 'history': [5, 45], 'heightmap': [3, 48, 32],
                          'privileged': [35]})
        self.assertEqual(spec.FOOT_ORDER, ('FL', 'FR', 'RL', 'RR'))
        self.assertEqual(spec.SPEEDS, (.2, .4, .6))


class StageTableTests(unittest.TestCase):
    def test_ten_stage_rows(self):
        rows = [(stage.id, stage.tile_id, stage.N, stage.h, stage.budget, stage.gate)
                for stage in spec.STAGES]
        self.assertEqual(rows, [
            ('F0', 0, 0, 0., 2000, 100),
            ('P1', 1, 1, .02, 1000, 100),
            ('P2', 2, 1, .04, 1000, 100),
            ('P3', 3, 1, .06, 1000, 100),
            ('P4', 4, 1, .08, 2000, 100),
            ('S1', 5, 4, .08, 3000, 200),
            ('S2', 6, 8, .12, 4000, 200),
            ('S3', 7, 12, .16, 4000, 200),
            ('S4', 8, 20, .20, 5000, 200),
            ('S5', 9, 40, .20, 7000, 200)])
        self.assertEqual(spec.STAGE_ORDER,
                         ('F0', 'P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4', 'S5'))
        self.assertEqual(set(spec.STAGE_BY_ID), set(spec.STAGE_ORDER))
        self.assertEqual(set(spec.STAGE_BY_TILE), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        self.assertEqual(spec.STAGE_BY_TILE[0].id, 'F0')
        self.assertEqual(spec.STAGE_BY_TILE[9].id, 'S5')
        # The flat alias is metadata only; it is never a training stage.
        self.assertNotIn('flat', spec.STAGE_BY_ID)
        self.assertEqual(spec.FLAT_ALIAS, 'flat')
        self.assertEqual(spec.FLAT_TILE_ID, 0)

    def test_total_budget_and_gate_intervals(self):
        self.assertEqual(sum(stage.budget for stage in spec.STAGES), 30000)
        self.assertEqual(spec.TOTAL_BUDGET, 30000)
        self.assertEqual(sum(stage.gate for stage in spec.STAGES), 1500)
        self.assertEqual(spec.FORMAL_STAGE_BUDGETS,
                         'F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,S2=4000,'
                         'S3=4000,S4=5000,S5=7000')
        self.assertEqual(spec.PILOT_STAGE_BUDGETS, 'F0=2000,P1=500')
        self.assertEqual(spec.PILOT_P1_UPDATES, 500)
        self.assertEqual(spec.PILOT_MAX_UPDATES, 2500)
        for stage in spec.STAGES:
            self.assertEqual(spec.previous_stage_id(stage.id),
                             spec.STAGE_ORDER[spec.STAGE_ORDER.index(stage.id) - 1]
                             if stage.id != 'F0' else None)

    def test_episode_step_limit_is_single_source(self):
        """SDS 5.1: the one episode-cap implementation (flat 1500, stair formula)."""
        self.assertEqual(spec.F0_EPISODE_STEPS, 1500)
        # Printed real values, frozen here so a formula edit cannot drift silently.
        self.assertEqual(spec.stair_episode_steps(4, .08, .4), 1266)
        self.assertEqual(spec.stair_episode_steps(0, 0., .4), 1500)
        self.assertEqual(spec.stair_episode_steps(40, .2, .2), 7257)
        self.assertEqual(spec.stair_episode_steps(40, .2, .6), 2753)
        # The degenerate h == 0 case is the flat cap, not the stair formula.
        self.assertEqual(spec.stair_episode_steps(4, 0., .4), 1500)
        self.assertEqual(spec.stair_episode_steps(40, 0., .2), 1500)
        for value in (spec.stair_episode_steps(4, .08, .4), spec.episode_step_limit(4, .08, .4),
                      spec.stair_episode_steps(0, 0., .4), spec.episode_step_limit(0, 0., .4)):
            self.assertIsInstance(value, int)
        # Per-slot dispatch follows the slot's real N/h/c for every stage.
        for stage in spec.STAGES:
            for speed in spec.SPEEDS:
                expected = (spec.F0_EPISODE_STEPS if stage.N == 0
                            else spec.stair_episode_steps(stage.N, stage.h, speed))
                self.assertEqual(spec.episode_step_limit(stage.N, stage.h, speed),
                                 expected, msg=(stage.id, speed))
        # Slower commands allow more steps; the S5 spread is exactly the reference pair.
        self.assertEqual(spec.episode_step_limit(40, .2, .4), 3879)
        self.assertEqual(spec.episode_step_limit(4, .08, .2), 2031)
        self.assertEqual(spec.episode_step_limit(4, .08, .6), 1011)
        self.assertGreater(spec.episode_step_limit(40, .2, .2),
                           spec.episode_step_limit(40, .2, .6))
        # Contract carries the same implementation and reference cases.
        timeout = stable_contract()['timeout']
        self.assertEqual(timeout['flat_steps'], spec.F0_EPISODE_STEPS)
        self.assertEqual(timeout['implementation'],
                         'spec.episode_step_limit(tile_n, tile_h, executed_c) -> int')
        self.assertEqual(timeout['stair_implementation'],
                         'spec.stair_episode_steps(stage_n, stage_h, executed_c) -> int')
        self.assertEqual(timeout['reference_cases'],
                         [[4, .08, .4, 1266], [0, 0., .4, 1500],
                          [40, .2, .2, 7257], [40, .2, .6, 2753]])
        for count, height, command, expected in timeout['reference_cases']:
            self.assertEqual(spec.episode_step_limit(count, height, command), expected)
            self.assertEqual(spec.stair_episode_steps(count, height, command), expected)
        self.assertEqual(timeout['flat_exception'], 'N == 0 or h == 0 -> F0_EPISODE_STEPS')
        self.assertFalse(timeout['stagnation_termination'])
        # Argument validation is explicit, never a silent clamp.
        for args in ((-1, .08, .4), (4.5, .08, .4), (True, .08, .4),
                     (4, -.08, .4), (4, float('nan'), .4), (4, float('inf'), .4),
                     (4, .08, 0.), (4, .08, -.4), (4, .08, float('nan')),
                     (4, .08, float('inf')), (4, .08, True), (4, .08, 'fast')):
            with self.assertRaises(ValueError, msg=str(args)):
                spec.stair_episode_steps(*args)
            with self.assertRaises(ValueError, msg=str(args)):
                spec.episode_step_limit(*args)

    def test_512_formal_slot_counts_are_exact(self):
        expected = {
            'F0': {'flat': 512, 'previous': 0, 'current': 0},
            'P1': {'flat': 102, 'previous': 0, 'current': 410},
        }
        for stage in spec.STAGES[2:]:
            expected[stage.id] = {'flat': 51, 'previous': 102, 'current': 359}
        for stage in spec.STAGES:
            counts = spec.slot_counts(stage.id, 512)
            self.assertEqual(counts, expected[stage.id], msg=stage.id)
            self.assertEqual(sum(counts.values()), 512)
            layout = spec.slot_layout(stage.id, 512)
            self.assertEqual(len(layout), 512)
            self.assertEqual(layout.count('flat'), counts['flat'])
            self.assertEqual(layout.count('previous'), counts['previous'])
            self.assertEqual(layout.count('current'), counts['current'])
            self.assertEqual(layout, sorted(layout, key=('flat', 'previous', 'current').index))
            tiles = spec.slot_tiles(stage.id, 512)
            self.assertEqual(len(tiles), 512)
            self.assertEqual(tiles.count(0), counts['flat'])
            if stage.id == 'F0':
                self.assertEqual(set(tiles), {0})
            else:
                self.assertEqual(tiles.count(stage.tile_id), counts['current'])

    def test_8_environment_largest_remainder_allocation(self):
        self.assertEqual(spec.slot_counts('F0', 8), {'flat': 8, 'previous': 0, 'current': 0})
        self.assertEqual(spec.slot_counts('P1', 8), {'flat': 2, 'previous': 0, 'current': 6})
        self.assertEqual(spec.slot_layout('P1', 8),
                         ['flat', 'flat'] + ['current'] * 6)
        for stage in spec.STAGES[2:]:
            self.assertEqual(spec.slot_counts(stage.id, 8),
                             {'flat': 1, 'previous': 1, 'current': 6}, msg=stage.id)
            self.assertEqual(spec.slot_layout(stage.id, 8),
                             ['flat', 'previous'] + ['current'] * 6, msg=stage.id)
            self.assertEqual(spec.slot_tiles(stage.id, 8),
                             [0, spec.STAGE_BY_TILE[stage.tile_id - 1].tile_id]
                             + [stage.tile_id] * 6, msg=stage.id)
        # Every legal environment count is conserved and deterministic.
        for count in (1, 2, 3, 4, 7, 8, 51, 512, 513, 1024):
            for stage in spec.STAGES:
                values = spec.slot_counts(stage.id, count)
                self.assertEqual(sum(values.values()), count, msg=(count, stage.id))
                self.assertEqual(list(values), list(spec.SLOT_CLASSES))
                self.assertEqual(spec.slot_counts(stage.id, count), values)

    def test_slot_allocation_rejects_bad_arguments(self):
        for args in ((0, 'F0'), (-1, 'S1'), (1.5, 'S1'), (True, 'S1'), (10, 'flat'),
                     (10, 'R0'), (10, 'S6'), (10, 's1'), (10, None)):
            with self.assertRaises(ValueError, msg=str(args)):
                spec.slot_counts(*args)
        for bad in ('flat', 'S6', None, 5):
            with self.assertRaises(ValueError, msg=str(bad)):
                spec.slot_layout(bad, 8)


class RewardRecipeTests(unittest.TestCase):
    def test_frozen_0_5_r1_numbers(self):
        cfg = spec.reward_config()
        self.assertEqual((cfg['version'], cfg['dt']), ('0.5-r1', .02))
        self.assertTrue(cfg['dt_multiplied_once'])
        self.assertFalse(cfg['total_reward_clamped'])
        self.assertFalse(cfg['alive_bonus'])
        self.assertEqual(cfg['active_definition'], 'executed_c > 0.1')
        speed = cfg['speed']
        self.assertEqual((speed['coefficient'], speed['exp_coefficient'],
                          speed['exp_scale'], speed['exp_offset'],
                          speed['over_deadband'], speed['over_coefficient'],
                          speed['over_cap'], speed['backward_coefficient'],
                          speed['backward_cap'], speed['side_coefficient'],
                          speed['side_cap'], speed['side_scale']),
                         (.02, 1.5, 4., -.5, 1.2, .5, 4., .5, 4., .25, 4., .2))
        self.assertTrue(speed['positive_part_only_gated'])
        self.assertTrue(speed['includes_dt'])
        self.assertEqual(speed['active_nonpositive_c'], 'error')
        self.assertEqual(speed['disabled_zero_c'], 'zero_with_explicit_enabled_false')
        self.assertAlmostEqual(speed['still_rate'], 1.5 * math.exp(-4.) - .5, places=15)
        self.assertAlmostEqual(speed['still_rate'], -0.4725265416668987, places=15)
        # The 0.02 printed in SDS 3.3/3.4 is the control dt applied once: the
        # per-step amount is dt*rate, so the exported coefficient equals dt.
        self.assertEqual((speed['dt'], speed['per_step_formula']),
                         (.02, 'dt * speed_rate'))
        self.assertEqual(speed['coefficient'], spec.CONTROL_DT)
        self.assertIn('control dt', speed['coefficient_note'])
        self.assertAlmostEqual(speed['still_r_speed'],
                               spec.CONTROL_DT * (1.5 * math.exp(-4.) - .5), places=17)
        self.assertAlmostEqual(speed['still_r_speed'], -0.009450530833337975, places=17)
        self.assertEqual(speed['match_rate'], 1.)
        self.assertAlmostEqual(speed['match_r_speed'], spec.CONTROL_DT, places=17)
        checks = speed['discount_checks']
        self.assertEqual((checks['gamma'], checks['steps'], checks['per_step']),
                         (.995, 1000, 'dt*rate'))
        self.assertEqual((checks['stationary'], checks['matched'], checks['early_failure']),
                         (-2.482749, 3.973384, -6.839384))
        self.assertEqual((spec.SPEED_COEFFICIENT, spec.STALL_COEFFICIENT,
                          spec.SPEED_DT, spec.STALL_DT),
                         (spec.CONTROL_DT, spec.CONTROL_DT, .02, .02))
        self.assertAlmostEqual(spec.MATCH_R_SPEED, .02, places=17)
        self.assertEqual(spec.STALL_REWARD, spec.CONTROL_DT * spec.STALL_RATE)
        stall = cfg['stall']
        self.assertEqual((stall['coefficient'], stall['rate'], stall['per_step_amount'],
                          stall['history_samples'], stall['intervals'],
                          stall['consecutive_samples'], stall['first_penalty_sample'],
                          stall['highwater_dx'], stall['dx_fraction'], stall['dx_scale']),
                         (.02, -.25, -.005, 51, 50, 50, 99, .03, .15, 1.))
        self.assertEqual((stall['dt'], stall['per_step_formula']),
                         (.02, 'dt * stall_rate'))
        self.assertTrue(stall['climbed_disables'])
        self.assertFalse(stall['gated'])
        self.assertFalse(stall['termination_added'])
        self.assertEqual(cfg['events']['success'], 10.)
        self.assertEqual(cfg['events']['failure'], -10.)
        self.assertEqual(cfg['events']['timeout_amount'], 0.)
        self.assertTrue(cfg['events']['failure_precedence'])
        self.assertEqual(cfg['footfall']['amount'], .50)
        self.assertEqual(cfg['footfall']['recompute'], 'events.sum(-1)*0.50')
        self.assertTrue(cfg['footfall']['top_platform_shares_last_step'])
        self.assertTrue(cfg['footfall']['no_dt_multiplier'])
        swing = cfg['swing']
        self.assertEqual((swing['amount_per_target'], swing['target_window_m'],
                          swing['body_height_scale'], swing['target_height_margin'],
                          swing['max_steps'], swing['safe_inset'], swing['foot_radius']),
                         (.15, .60, .04, .04, 40, .042, .032))
        self.assertTrue(swing['consume_without_gate'])
        self.assertTrue(swing['target_lock_never_moves'])
        self.assertEqual(swing['paid_fraction_scope'], 'episode_foot_target')
        self.assertEqual(cfg['pose']['weight'], .05)
        self.assertEqual(cfg['pose']['sigma_active'], [.15, .35, .5] * 4)
        self.assertEqual(cfg['pose']['sigma_zero_command'], [.05, .1, .15] * 4)
        self.assertEqual(cfg['action_rate'] if 'action_rate' in cfg else
                         cfg['continuous_weights']['action_rate'], -.02)
        self.assertEqual(cfg['continuous_weights'], {
            'track_angular_velocity': 1., 'pose': .05, 'orientation': -1.,
            'body_ang_vel': -.05, 'angular_momentum': -.025, 'joint_acc': -2.5e-7,
            'joint_limits': -10., 'action_rate': -.02, 'slip': -.25,
            'soft_landing': -.001, 'stand_still': -1., 'center': -.5, 'heading': -.5,
            'edge': -1., 'nonfoot': -1.})
        self.assertEqual(cfg['orientation']['lower_deadband'], math.radians(5.))
        self.assertEqual(cfg['orientation']['upper_deadband'], math.radians(55.))
        self.assertEqual(cfg['nonfoot']['force_threshold'], 10.)
        self.assertFalse(cfg['nonfoot']['averaging'])
        self.assertFalse(cfg['slip']['averaged_over_feet'])
        self.assertIn('active_command_including_flat_and_approach', cfg['center']['scope'])
        self.assertIn('active_command_including_flat_and_approach', cfg['heading']['scope'])
        self.assertEqual(cfg['edge']['inset'], .042)
        self.assertTrue(cfg['edge']['last_step_and_platform_union'])
        self.assertEqual(cfg['removed']['body_x_progress']['new_contribution'], 0.)
        self.assertEqual(cfg['removed']['body_x_progress']['visibility'], 'diagnostic_only')
        self.assertEqual(cfg['continuous_weights']['track_angular_velocity'], 1.)
        self.assertEqual(cfg['track_angular_velocity']['baseline_subtract'], 1.)
        self.assertEqual(len(cfg['speed']['truth_table_per_second_rate']), 8)

    def test_recipe_hash_is_canonical_and_immutable(self):
        digest = spec.reward_config_sha256()
        self.assertRegex(digest, r'^[0-9a-f]{64}$')
        self.assertEqual(digest, hashlib.sha256(
            spec.canonical_json(spec.REWARD_CONFIG).encode('utf-8')).hexdigest())
        self.assertEqual(digest, spec.reward_config_sha256())
        copy = spec.reward_config()
        copy['version'] = 'mutated'
        copy['continuous_weights']['pose'] = 9.
        self.assertEqual(spec.REWARD_CONFIG['version'], '0.5-r1')
        self.assertEqual(spec.REWARD_CONFIG['continuous_weights']['pose'], .05)
        self.assertEqual(digest, spec.reward_config_sha256())
        recipe = spec.recipe_config()
        self.assertEqual(recipe['recipe'], '0.5-r1')
        self.assertEqual(recipe['reward_config_sha256'], digest)
        self.assertEqual((recipe['std_min'], recipe['std_max'], recipe['initial_std']),
                         (.10, .30, .15))
        self.assertEqual(json.loads(spec.canonical_json(recipe)), recipe)
        with self.assertRaises(ValueError):
            spec.recipe_config('0.4-r1')

    def test_speed_gate_thresholds_match_sds_6_1(self):
        """SDS 6.1 / v03 precedent: mean<=0.20, smooth_mae<=0.30, deviation<=50."""
        self.assertEqual((spec.MEAN_RELATIVE_ERROR_MAX, spec.SMOOTH_MAE_MAX,
                          spec.LONGEST_DEVIATION_MAX, spec.DEVIATION_THRESHOLD,
                          spec.SPEED_EPSILON),
                         (.20, .30, 50, .50, 1e-6))
        # The two relative-error limits must never be swapped back.
        self.assertLess(spec.MEAN_RELATIVE_ERROR_MAX, spec.SMOOTH_MAE_MAX)
        evaluation = stable_contract()['evaluation']
        self.assertEqual((evaluation['mean_relative_error_max'],
                          evaluation['smooth_mae_max'],
                          evaluation['longest_deviation_max'],
                          evaluation['deviation_threshold'],
                          evaluation['epsilon']),
                         (.20, .30, 50, .50, 1e-6))
        self.assertEqual(evaluation['speed_window'], 25)
        self.assertTrue(evaluation['speed_not_a_gate_for_stairs'])

    def test_ppo_contract_numbers(self):
        ppo = spec.PPO_CONTRACT
        self.assertEqual((ppo['num_envs'], ppo['rollout'], ppo['epochs'],
                          ppo['minibatches'], ppo['updates_adam_steps']),
                         (512, 48, 5, 4, 20))
        self.assertEqual((ppo['learning_rate'], ppo['min_lr'], ppo['max_lr'],
                          ppo['desired_kl']), (5e-5, 1e-5, 1e-4, .01))
        self.assertEqual((ppo['gamma'], ppo['gae_lambda'], ppo['clip'],
                          ppo['value_coef'], ppo['max_grad_norm']),
                         (.995, .95, .2, 1., 1.))
        self.assertTrue(ppo['clipped_value_loss'])
        self.assertEqual((ppo['entropy'], ppo['initial_std'], ppo['std_min'],
                          ppo['std_max']), (.005, .15, .10, .30))
        self.assertEqual(ppo['std_constrained_in'],
                         ['action_distribution', 'constrain_std'])
        self.assertEqual(ppo['mean_activation'], 'none_no_tanh_no_clip')
        self.assertEqual(ppo['control_samples_per_update'], 24576)
        self.assertEqual((spec.PPO_INITIAL_LR, spec.PPO_GAMMA, spec.INITIAL_STD,
                          spec.STD_MIN, spec.STD_MAX), (5e-5, .995, .15, .10, .30))


class SourceClosureTests(unittest.TestCase):
    def test_contract_identity_reward_sha_and_settings(self):
        contract = spec.contract()
        self.assertEqual((contract['code_version'], contract['recipe_version'],
                          contract['sds_revision'], contract['checkpoint_schema'],
                          contract['task'], contract['protocol_id'],
                          contract['flat_protocol_id']),
                         ('0.5.0', '0.5-r1', '1.1', 6,
                          'Unitree-A2-IdealHeightMap-Stairs-V05',
                          'v05-stair-climb-first-v1', 'v05-flat-speed-v1'))
        self.assertEqual(contract['reward_config'], spec.REWARD_CONFIG)
        self.assertEqual(contract['reward_config_sha256'], spec.reward_config_sha256())
        self.assertEqual(contract['source_eth_path'], spec.INIT_PATH)
        self.assertEqual(contract['source_eth_sha256'], spec.INIT_SHA256)
        self.assertEqual(contract['initialization'],
                         'explicit_eth_actor_mlp_and_frozen_normalizer')
        self.assertEqual(contract['stages'], [asdict(stage) for stage in spec.STAGES])
        self.assertEqual(contract['total_budget'], 30000)
        self.assertEqual(contract['training_num_envs'], 512)
        self.assertEqual(contract['evaluation_num_envs'], 8)
        self.assertEqual(contract['slot_classes'], ['flat', 'previous', 'current'])
        self.assertEqual(contract['allocation'],
                         'integer_largest_remainder_flat_previous_current')
        self.assertEqual(contract['slot_counts_512']['P1'],
                         {'flat': 102, 'previous': 0, 'current': 410})
        self.assertEqual(contract['slot_layout_8']['P1'], ['flat', 'flat'] + ['current'] * 6)
        self.assertEqual(contract['command']['range_per_episode'], [.2, .6])
        self.assertEqual(contract['command']['vy'], 0.)
        self.assertEqual(contract['command']['wz'], 0.)
        self.assertEqual(contract['timeout']['flat_steps'], 1500)
        self.assertFalse(contract['timeout']['stagnation_termination'])
        self.assertEqual(contract['ppo'], spec.PPO_CONTRACT)
        self.assertEqual(contract['evaluation']['validation_seed'], 200342)
        self.assertEqual(contract['evaluation']['perturbed_seed'], 300342)
        self.assertEqual(contract['evaluation']['diagnostic_seed'], 500342)
        self.assertEqual(contract['evaluation']['final_test_seed'], 400342)
        self.assertEqual(contract['resume_mode'], 'new_episode')
        self.assertEqual(contract['recovery']['unknown_tail'], 'recovery_blocked')
        self.assertEqual(contract['monitoring']['group'], 'v05')
        self.assertEqual(contract['monitoring']['heartbeat_kind'], 'v05_heartbeat')
        self.assertEqual(json.loads(spec.canonical_json(contract)), contract)
        self.assertEqual(json.loads(json.dumps(contract)), contract)
        self.assertEqual(spec.contract_sha256(), hashlib.sha256(
            spec.canonical_json(contract).encode('utf-8')).hexdigest())
        for kwargs in ({'recipe': 'A'}, {'recipe': '0.4-r1'}, {'mode': 'comparison'}):
            with self.assertRaises(ValueError):
                spec.contract(**kwargs)

    def test_seed_and_mode_are_the_only_contract_variants(self):
        allowed = {'seed', 'mode', 'evaluation'}
        for _ in range(3):
            base = stable_contract()
            other = stable_contract(mode='pilot', seed=43)
            differing = {key for key in base if base[key] != other[key]}
            if differing <= allowed:
                break
        else:  # pragma: no cover - only a concurrent writer can reach this
            self.fail(f'contract variants differed beyond seed/mode/evaluation: {differing}')
        self.assertEqual(differing, allowed)
        self.assertEqual(other['evaluation']['validation_seed'], 200343)
        self.assertEqual(other['evaluation']['perturbed_seed'], 300343)
        self.assertEqual(other['evaluation']['diagnostic_seed'], 500343)
        self.assertEqual(other['evaluation']['final_test_seed'], 400342)
        for key in base:
            if key in ('seed', 'mode', 'evaluation'):
                continue
            self.assertEqual(base[key], other[key], msg=key)

    def test_closure_binds_every_existing_product_file_and_bootstrap(self):
        contract = stable_contract()
        bound = contract['behavior_sha256']
        self.assertTrue(V05_FILES, 'no v05 module exists')
        self.assertLessEqual(V05_FILES, set(bound))
        for required in ('scripts/a2_v05_bootstrap.py',
                         'scripts/check_a2_ideal_stairs_v05_migration.py',
                         'src/tasks/velocity/v05/__init__.py',
                         'src/tasks/velocity/v05/spec.py',
                         'src/tasks/velocity/v05/model.py',
                         'src/tasks/velocity/v05/rewards.py'):
            if (spec.ROOT / required).is_file():
                self.assertIn(required, bound)
        self.assertNotIn('src/tasks/__init__.py', bound)
        for relative, digest in bound.items():
            self.assertEqual(spec.sha256(spec.ROOT / relative), digest, msg=relative)
        # The pending list is exactly the expected product entries absent from the
        # same snapshot (single contract read, no second directory scan).
        self.assertEqual(contract['pending_source_entries'],
                         [relative for relative in spec.SOURCE_ENTRIES
                          if relative not in bound])
        self.assertEqual(contract['source_closure_complete'],
                         not contract['pending_source_entries'])
        self.assertEqual(spec.DYNAMIC_DEPENDENCIES, ())
        self.assertIn('src/tasks/velocity/__init__.py', bound)
        self.assertIn('src/__init__.py', bound)
        # The shared network module is a real execution dependency of model.py.
        self.assertIn('src/tasks/velocity/rl/ideal_heightmap_actor_critic.py', bound)

    def test_behavior_hashes_matches_source_hashes(self):
        self.assertEqual(spec.product_entries(), sorted(spec.product_entries()))
        for relative in spec.product_entries():
            self.assertTrue((spec.ROOT / relative).is_file(), msg=relative)
        for _ in range(3):
            first = spec.behavior_hashes()
            if first == spec.source_hashes():
                break
        else:  # pragma: no cover - only a concurrent writer can reach this
            self.fail('behavior_hashes/source_hashes disagreed for three reads')
        self.assertLessEqual(set(spec.product_entries()), set(first))
        self.assertNotIn('src/tasks/__init__.py', first)

    def test_explicit_closure_does_not_scan_the_task_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                'src/__init__.py': '',
                'src/tasks/__init__.py': 'raise RuntimeError("legacy scanner must not run")\n',
                'src/tasks/velocity/__init__.py': '',
                'src/tasks/velocity/v05/__init__.py': '',
                'src/tasks/velocity/v05/entry.py': 'from .helper import value\n',
                'src/tasks/velocity/v05/helper.py': 'from .leaf import value as leaf\nvalue = 1\n',
                'src/tasks/velocity/v05/leaf.py': 'value = 1\n',
                'src/tasks/unrelated/__init__.py': '',
                'src/tasks/unrelated/env.py': 'value = 1\n',
                'src/tasks/new_task/__init__.py': '',
                'src/tasks/new_task/env.py': 'value = 1\n',
                'src/tasks/velocity/config/a2/env_cfgs.py': 'value = 1\n',
                'scripts/entry.py': 'from src.tasks.velocity.v05.entry import value\n',
            }
            for relative, contents in files.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents)
            first = spec.source_hashes(['src/tasks/velocity/v05/entry.py'], root=root)
            self.assertEqual(set(first), {
                'src/__init__.py', 'src/tasks/velocity/__init__.py',
                'src/tasks/velocity/v05/__init__.py', 'src/tasks/velocity/v05/entry.py',
                'src/tasks/velocity/v05/helper.py', 'src/tasks/velocity/v05/leaf.py'})
            self.assertNotIn('src/tasks/__init__.py', first)
            self.assertNotIn('src/tasks/unrelated/env.py', first)
            self.assertNotIn('src/tasks/new_task/env.py', first)
            self.assertNotIn('src/tasks/velocity/config/a2/env_cfgs.py', first)
            for relative, digest in first.items():
                self.assertEqual(digest, hashlib.sha256((root / relative).read_bytes()).hexdigest())
            (root / 'src/tasks/velocity/v05/leaf.py').write_text('value = 2\n')
            second = spec.source_hashes(['src/tasks/velocity/v05/entry.py'], root=root)
            self.assertNotEqual(first['src/tasks/velocity/v05/leaf.py'],
                                second['src/tasks/velocity/v05/leaf.py'])
            (root / 'src/tasks/unrelated/env.py').write_text('value = 2\n')
            (root / 'src/tasks/new_task/extra.py').write_text('value = 2\n')
            (root / 'src/tasks/another_robot').mkdir()
            (root / 'src/tasks/another_robot/__init__.py').write_text('')
            self.assertEqual(second,
                             spec.source_hashes(['src/tasks/velocity/v05/entry.py'], root=root))
            with self.assertRaises(FileNotFoundError):
                spec.source_hashes(['src/tasks/velocity/v05/missing.py'], root=root)
            with self.assertRaises(ValueError):
                spec.source_hashes(['../escape.py'], root=root)
            script = spec.source_hashes(['scripts/entry.py'], root=root)
            self.assertIn('src/tasks/velocity/v05/entry.py', script)
            self.assertNotIn('src/tasks/__init__.py', script)
            # The default path binds exactly the existing product entries.
            default = spec.source_hashes(root=root)
            self.assertLessEqual(set(spec.product_entries(root)), set(default))
            self.assertEqual(set(default) - set(spec.product_entries(root)),
                             {'src/__init__.py', 'src/tasks/velocity/__init__.py'})
            self.assertIn('src/tasks/velocity/v05/leaf.py', default)
            self.assertNotIn('src/tasks/__init__.py', default)


if __name__ == '__main__':
    unittest.main()
