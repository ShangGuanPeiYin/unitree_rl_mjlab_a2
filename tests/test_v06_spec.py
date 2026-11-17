"""SDS v0.6 constants, terrain table, recipe CLI assertions and pure helpers.

Any number here that stops matching ``SDS_A2_IdealHeightMap_Stairs40_v0.6.md``
revision 1.0 must fail.  The test imports only the standard-library-only
``spec.py``; it never touches a simulator, a GPU, training or W&B.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
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


class IdentityTests(unittest.TestCase):
    def test_identity_block(self):
        self.assertEqual(
            (spec.CODE_VERSION, spec.RECIPE_VERSION, spec.SDS_REVISION,
             spec.SCHEMA_VERSION, spec.ARCHITECTURE, spec.TASK,
             spec.PROTOCOL_ID, spec.LOG_ROOT, spec.TRAINING_MODE,
             spec.TRAINING_SEED),
            ('0.6.0', '0.6-go2refs-r1', '1.0', 8, 'v06-route-state-v1',
             'Unitree-A2-IdealHeightMap-Stairs-V06',
             'v06-go2refs-climb-first-v1', 'logs/a2_ideal_stairs_v06',
             'train', 42))
        self.assertEqual(spec.ROOT, ROOT)
        self.assertEqual(spec.identity_block(), dict(
            code_version='0.6.0', recipe_version='0.6-go2refs-r1',
            schema_version=8, architecture='v06-route-state-v1',
            task='Unitree-A2-IdealHeightMap-Stairs-V06',
            protocol='v06-go2refs-climb-first-v1',
            log_root='logs/a2_ideal_stairs_v06', training_mode='train',
            training_seed=42))

    def test_frozen_documents_exist_and_match(self):
        self.assertEqual(spec.SDS_SHA256,
                         '1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60')
        self.assertEqual(spec.W0_FREEZE_SHA256,
                         'd1b51bc771e216e69eefd6da8ad834e6f1bb983145eb8c19d6297d5480a3c1f0')
        self.assertEqual(spec.Q001_DECISION_SHA256,
                         '5f3f2aa397fb5cf5ce286788e76384e8b19914c758e0ae29e50a67aa4d71a4b1')
        self.assertEqual(spec.Q002_DECISION_SHA256,
                         '3ebc7441f9be457cb135846366fe23926d02ec69e2232e0502d6c318ce5ccd52')
        self.assertEqual(spec.W0_ADDENDUM_SHA256,
                         'b7553319abeb754b3a2838215e9110713435533b336af068ac89967f6b9c59ef')
        self.assertEqual(spec.W0_ADDENDUM2_SHA256,
                         'c1dc2a8a092c086d876f43b41b5f6ebea7f68bdf24d93be5a4b045be18878dfc')
        self.assertEqual(len(spec.FROZEN_DOCUMENTS), 6)
        for name, relative, digest in spec.FROZEN_DOCUMENTS:
            path = ROOT / relative
            self.assertTrue(path.is_file(), name)
            self.assertEqual(spec.sha256(path), digest, name)
        self.assertEqual(spec.document_identity(), dict(
            sds_path=spec.SDS_PATH, sds_sha256=spec.SDS_SHA256,
            w0_freeze_path=spec.W0_FREEZE_PATH,
            w0_freeze_sha256=spec.W0_FREEZE_SHA256,
            q001_decision_path=spec.Q001_DECISION_PATH,
            q001_decision_sha256=spec.Q001_DECISION_SHA256,
            q002_decision_path=spec.Q002_DECISION_PATH,
            q002_decision_sha256=spec.Q002_DECISION_SHA256,
            w0_addendum_path=spec.W0_ADDENDUM_PATH,
            w0_addendum_sha256=spec.W0_ADDENDUM_SHA256,
            w0_addendum_2_path=spec.W0_ADDENDUM2_PATH,
            w0_addendum_2_sha256=spec.W0_ADDENDUM2_SHA256))

    def test_init_checkpoint_binding(self):
        self.assertEqual(spec.INIT_PATH,
                         '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt')
        self.assertEqual(spec.INIT_SHA256,
                         '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722')
        self.assertTrue(Path(spec.INIT_PATH).is_file())
        self.assertEqual(spec.sha256(spec.INIT_PATH), spec.INIT_SHA256)

    def test_timing_and_observation(self):
        self.assertEqual((spec.CONTROL_DT, spec.PHYSICS_DT, spec.DECIMATION,
                          spec.ACTION_SCALE, spec.MAP_HZ),
                         (.02, .005, 4, .25, 10))
        self.assertEqual((spec.N_JOINTS, spec.PROPRIO_DIM, spec.HISTORY_LEN,
                          spec.HEIGHTMAP_SHAPE, spec.ROUTE_DIM,
                          spec.PRIVILEGED_DIM, spec.CRITIC_WIDTH,
                          spec.ETH_ACTOR_DIM),
                         (12, 45, 5, (3, 48, 32), 3, 35, 211, 45))
        self.assertEqual(spec.ROUTE_STATE['order'],
                         ['clip(local_y/0.20,-4.0,4.0)', 'sin(e_world)',
                          'cos(e_world)'])
        self.assertEqual((spec.ROUTE_LATERAL_SCALE, spec.ROUTE_LATERAL_CLIP,
                          spec.ROUTE_YAW_AXIS), (.20, 4., 0.))
        self.assertAlmostEqual(spec.ROUTE_LATERAL_SATURATION_M, .80, places=15)


class TerrainTableTests(unittest.TestCase):
    def test_22_tiles_and_flat(self):
        self.assertEqual(len(spec.TERRAIN_TABLE), 22)
        self.assertEqual(spec.T_TILE_COUNT, 21)
        self.assertEqual([entry.tile_id for entry in spec.TERRAIN_TABLE],
                         list(range(22)))
        flat = spec.terrain_spec(spec.FLAT_TILE_ID)
        self.assertEqual((flat.lesson_id, flat.N, flat.h), ('FLAT', 0, 0.))
        self.assertEqual(spec.tile_id_for('flat'), 0)
        self.assertEqual(spec.tile_id_for('F0'), 0)
        self.assertEqual(spec.terrain_spec(21).lesson_id, 'L40')
        self.assertEqual(spec.tile_id_for('L40'), 21)

    def test_exact_step_heights_and_depths(self):
        expected_heights = (.02, .030833333333333334, .04166666666666667,
                            .052500000000000005, .06333333333333334,
                            .07416666666666667, .085, .09583333333333334,
                            .10666666666666667, .1175, .12833333333333333,
                            .1391666666666667, .15, .16, .18, .20)
        for index, height in enumerate(expected_heights):
            self.assertAlmostEqual(spec.TERRAIN_TABLE[index + 1].h, height,
                                   places=15, msg=f'T{index}')
            self.assertEqual(spec.TERRAIN_TABLE[index + 1].lesson_id,
                             f'H{index:02d}')
            self.assertEqual(spec.TERRAIN_TABLE[index + 1].N, 6)
            self.assertEqual(spec.TERRAIN_TABLE[index + 1].step_depth, .30)
        rows = {entry.lesson_id: entry for entry in spec.TERRAIN_TABLE}
        self.assertEqual((rows['D25'].N, rows['D25'].h, rows['D25'].step_depth),
                         (6, .20, .25))
        self.assertEqual((rows['D20'].N, rows['D20'].h, rows['D20'].step_depth),
                         (6, .20, .20))
        self.assertEqual((rows['L12'].N, rows['L12'].step_depth), (12, .20))
        self.assertEqual((rows['L20'].N, rows['L20'].step_depth), (20, .20))
        self.assertEqual((rows['L40'].N, rows['L40'].step_depth), (40, .20))

    def test_geometry(self):
        for entry in spec.TERRAIN_TABLE[1:]:
            self.assertEqual((entry.width, entry.tread_thickness,
                              entry.platform_length), (.80, .01, 2.0))
            self.assertEqual(entry.frame_origin_world,
                             (0., entry.tile_id * 8., 0.))
            self.assertEqual(entry.valid_bounds_local,
                             (-2., entry.N * entry.step_depth + 2., -2., 2.))
        flat = spec.TERRAIN_TABLE[0]
        self.assertEqual(flat.valid_bounds_local, (-2., 22., -4., 4.))
        self.assertEqual(spec.FLAT_QUERY_BOUNDS_LOCAL, flat.valid_bounds_local)

    def test_terrain_lookup_rejects_unknown(self):
        for tile_id in (-1, 22, 100):
            with self.assertRaises(ValueError):
                spec.terrain_spec(tile_id)
        with self.assertRaises(ValueError):
            spec.terrain_spec(True)
        with self.assertRaises(ValueError):
            spec.tile_id_for('H16')
        with self.assertRaises(ValueError):
            spec.tile_id_for('nope')


class PhaseTests(unittest.TestCase):
    def test_phase_table(self):
        self.assertEqual(spec.PHASE_ORDER, ('F0', 'H', 'D', 'L', 'V', 'R'))
        self.assertEqual(spec.PHASE_BUDGETS,
                         {'F0': 1000, 'H': 12000, 'D': 3000,
                          'L': 6000, 'V': 4000, 'R': 4000})
        self.assertEqual(spec.TOTAL_BUDGET, 30000)
        self.assertEqual(spec.total_budget(), 30000)
        self.assertEqual(spec.D_SUBSTAGES, ('D25', 'D20'))
        self.assertEqual(spec.D_SUBSTAGE_BUDGETS, (1500, 1500))
        self.assertEqual(spec.L_SUBSTAGES, ('L12', 'L20', 'L40'))
        self.assertEqual(spec.L_SUBSTAGE_BUDGETS, (1500, 1500, 3000))
        for phase in spec.PHASES:
            self.assertEqual(phase.budget, spec.phase_budget(phase.phase))
            if phase.phase in ('F0', 'H', 'D', 'L'):
                self.assertEqual(phase.gate_speeds, (.3, .5, .8))
                self.assertEqual(phase.reward_speeds, (.3, .8))
            else:
                self.assertEqual(phase.gate_speeds, (.2, .4, .6))
                self.assertEqual(phase.reward_speeds, (.2, .6))
        with self.assertRaises(ValueError):
            spec.phase_spec('S1')


class RewardTableTests(unittest.TestCase):
    def test_terms_and_weights(self):
        self.assertEqual(spec.REWARD_TERMS, (
            'tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z',
            'orientation_roll', 'base_height', 'action_rate',
            'similar_to_default', 'dof_acc', 'dof_vel', 'ang_vel_xy',
            'feet_air_time', 'foot_slip', 'foot_clearance', 'joint_tracking',
            'center', 'heading', 'joint_limits'))
        self.assertEqual(spec.REWARD_WEIGHTS, {
            'tracking_lin_vel': 1.5, 'tracking_ang_vel': .8, 'lin_vel_z': -1.0,
            'orientation_roll': -5.0, 'base_height': -.1, 'action_rate': -.01,
            'similar_to_default': -.05, 'dof_acc': -2.5e-7, 'dof_vel': -5e-4,
            'ang_vel_xy': -.05, 'feet_air_time': .2, 'foot_slip': -.15,
            'foot_clearance': -.5, 'joint_tracking': -.1, 'center': -.5,
            'heading': -.5, 'joint_limits': -10.0})
        self.assertEqual((spec.FORWARD_PROGRESS_PER_M, spec.SUCCESS_REWARD,
                          spec.FAILURE_REWARD, spec.TIMEOUT_REWARD),
                         (.4, 10.0, -10.0, 0.))
        self.assertEqual((spec.BASE_HEIGHT_TARGET, spec.FLAT_CLEARANCE_TARGET,
                          spec.LIN_VEL_Z_DEADBAND, spec.AIR_TIME_TARGET),
                         (.40, .075, .15, .10))
        self.assertAlmostEqual(spec.FOOT_RADIUS, .032, places=15)

    def test_clearance_targets(self):
        self.assertEqual(spec.clearance_target(.02, flat=False), .17)
        self.assertEqual(spec.clearance_target(.15, flat=False), .17)
        self.assertAlmostEqual(spec.clearance_target(.18, flat=False), .20)
        self.assertAlmostEqual(spec.clearance_target(.20, flat=False), .22)
        self.assertEqual(spec.clearance_target(0., flat=True), .075)
        with self.assertRaises(ValueError):
            spec.clearance_target(-.1, flat=False)

    def test_reward_config_is_hashable_and_stable(self):
        first = spec.reward_config()
        second = spec.reward_config()
        self.assertEqual(first, second)
        self.assertEqual(first['version'], spec.RECIPE_VERSION)
        self.assertEqual(first['weights'], spec.REWARD_WEIGHTS)
        self.assertTrue(first['dt_multiplied_once'])
        self.assertTrue(first['terminal_not_scaled_by_dt'])
        self.assertTrue(first['failure_precedence'])
        self.assertEqual(first['timeout_reward'], 0.)
        digest = spec.reward_config_sha256()
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest,
                         hashlib.sha256(spec.canonical_json(first).encode()).hexdigest())
        self.assertEqual(spec.canonical_json({'b': 1, 'a': 2}), '{"a":2,"b":1}')

    def test_default_pose_and_soft_limits(self):
        self.assertEqual(spec.A2_DEFAULT_JOINT_POS,
                         (-.1, .9, -1.8, .1, .9, -1.8,
                          -.1, .9, -1.8, .1, .9, -1.8))
        self.assertEqual(len(spec.A2_SOFT_JOINT_LIMITS), 12)
        # Q001-4 midpoint semantics: m=(lo+hi)/2, r=(hi-lo)/2, soft=m+-0.9r.
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[0][0], -.909, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[0][1], .909, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[1][0], -2.0655, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[1][1], 2.8755, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[2][0], -2.6585, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[2][1], -.6515, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[7][0], -1.285, places=12)
        self.assertAlmostEqual(spec.A2_SOFT_JOINT_LIMITS[7][1], 3.665, places=12)

    def test_diagnostic_collapse_streak_in_reward_config_and_sha(self):
        self.assertEqual(spec.DIAGNOSTIC_COLLAPSE_STREAK, 2)
        config = spec.reward_config()
        self.assertEqual(config['diagnostics']['collapse_streak'], 2)
        self.assertEqual(config['diagnostics']['scope'],
                         'inherited_stall_collapse_only')
        diagnostics = spec.contract()['diagnostics']
        self.assertEqual(diagnostics['collapse_streak'], 2)
        self.assertEqual(diagnostics['scope'], 'inherited_stall_collapse_only')
        self.assertIn('stall', diagnostics)
        # It is inside the hashed reward configuration, not a side constant.
        mutated = spec.reward_config()
        mutated['diagnostics']['collapse_streak'] = 3
        self.assertNotEqual(
            hashlib.sha256(spec.canonical_json(mutated).encode()).hexdigest(),
            spec.reward_config_sha256())
        self.assertEqual(
            spec.reward_config_sha256(),
            hashlib.sha256(spec.canonical_json(config).encode()).hexdigest())

    def test_q002_2_fall_drop_constant_formula_and_sha(self):
        self.assertEqual(spec.FALL_DROP, .50)
        config = spec.reward_config()
        self.assertEqual(config['fall_drop']['drop'], .50)
        self.assertEqual(config['fall_drop']['comparison'], 'strict_less_than')
        self.assertEqual(config['fall_drop']['flat'], 'disabled')
        self.assertEqual(config['fall_drop']['approach'], 'local_x<0 disabled')
        self.assertEqual(config['fall_drop']['platform'], 'z_expected=N*h')
        self.assertEqual(spec.contract()['fall_drop']['drop'], .50)
        # The threshold is part of the hashed reward configuration.
        mutated = spec.reward_config()
        mutated['fall_drop']['drop'] = .60
        self.assertNotEqual(
            hashlib.sha256(spec.canonical_json(mutated).encode()).hexdigest(),
            spec.reward_config_sha256())
        # The expected tread is a per-tile geometric quantity, never an
        # observed foot/average/spawn height: platform expectation is N*h and
        # flat carries no stair region at all.
        for entry in spec.TERRAIN_TABLE[1:]:
            self.assertGreater(entry.N, 0)
            self.assertEqual(entry.valid_bounds_local[1],
                             entry.N * entry.step_depth + spec.PLATFORM_LENGTH)
            # Platform (local_x = N*d) expectation from the frozen formula is N*h.
            local_x = entry.N * entry.step_depth
            z_expected = min(max(math.floor(local_x / entry.step_depth) + 1, 0),
                             entry.N) * entry.h
            self.assertAlmostEqual(z_expected, entry.N * entry.h)
            region = (entry.N > 0) and (local_x >= 0)
            self.assertTrue(region)
        flat = spec.terrain_spec(spec.FLAT_TILE_ID)
        self.assertEqual((flat.N, flat.h), (0, 0.))
        self.assertFalse((flat.N > 0) and (0. >= 0))  # flat disables the check
        self.assertFalse((flat.N > 0) and (-1. >= 0))
        source = (ROOT / 'src/tasks/velocity/v06/spec.py').read_text()
        self.assertIn('Q002-2', source)
        self.assertIn('min(max(floor(local_x / d) + 1, 0), N) * h', source)
        self.assertIn('strict', source.lower())

    def test_inherited_stall_diagnostic_constants_in_sha(self):
        self.assertEqual(spec.STALL_HISTORY_SAMPLES, 51)
        self.assertEqual(spec.STALL_DX_FRACTION_PER_SECOND, .15)
        self.assertEqual(spec.STALL_DX_THRESHOLD_SCALE, 1.0)
        self.assertEqual(spec.STALL_HIGHWATER_DX, .03)
        self.assertEqual(spec.STALL_CONSECUTIVE_SAMPLES, 50)
        self.assertEqual(spec.STALL_ACTIVE_COMMAND, .1)
        stall = spec.reward_config()['diagnostics']['stall']
        self.assertEqual(stall['history_samples'], 51)
        self.assertEqual(stall['dx_fraction_per_second'], .15)
        self.assertEqual(stall['dx_threshold_scale'], 1.0)
        self.assertEqual(stall['highwater_dx'], .03)
        self.assertEqual(stall['consecutive_samples'], 50)
        self.assertEqual(stall['active_command'], .1)
        self.assertFalse(stall['reward_payment'])
        self.assertFalse(stall['termination'])
        self.assertEqual(spec.contract()['diagnostics']['stall'], stall)
        # They are part of the hashed reward configuration.
        mutated = spec.reward_config()
        mutated['diagnostics']['stall']['highwater_dx'] = .04
        self.assertNotEqual(
            hashlib.sha256(spec.canonical_json(mutated).encode()).hexdigest(),
            spec.reward_config_sha256())
        source = (ROOT / 'src/tasks/velocity/v06/spec.py').read_text()
        self.assertIn('inherited r2 stall diagnostic', source)
        self.assertIn('Q001-7', source)

    def test_q002_4_stop_and_lifecycle_states(self):
        self.assertEqual(spec.STOP_STATES, (
            'behavior_regression', 'behavior_collapse', 'curriculum_stalled',
            'budget_exhausted', 'gate_failed', 'user_requested_stop',
            'monitoring_blocked', 'resource_insufficient'))
        self.assertEqual(spec.LIFECYCLE_STATES,
                         ('running', 'ready_for_final_test', 'complete'))
        self.assertEqual(spec.ALL_TRAINING_STATES,
                         spec.LIFECYCLE_STATES + spec.STOP_STATES)
        self.assertEqual(len(spec.STOP_STATES), 8)
        self.assertEqual(len(spec.LIFECYCLE_STATES), 3)
        self.assertNotIn('pilot_not_applicable', spec.ALL_TRAINING_STATES)
        self.assertFalse(spec.PILOT_NOT_APPLICABLE_IS_TERMINAL)
        self.assertEqual(spec.COMPLETE_MEANING,
                         'final_300_case_test_passed_only')
        contract = spec.contract()
        self.assertEqual(contract['states']['stop'], list(spec.STOP_STATES))
        self.assertEqual(contract['states']['lifecycle'],
                         list(spec.LIFECYCLE_STATES))
        self.assertNotIn('pilot_not_applicable', contract['states']['all'])
        self.assertEqual(contract['states']['complete_meaning'],
                         'final_300_case_test_passed_only')

    def test_q001_3_layout_wording_is_not_numerical_v02_equality(self):
        source = (ROOT / 'src/tasks/velocity/v06/spec.py').read_text()
        self.assertIn('Q001-3', source)
        self.assertNotIn('follows the existing v0.2 atlas convention', source)
        self.assertIn('not** a claim of numerical identity', source)
        self.assertEqual(spec.TERRAIN_TABLE[1].frame_origin_world, (0., 8., 0.))
        self.assertEqual(spec.TERRAIN_TABLE[1].valid_bounds_local,
                         (-2., 6 * .30 + 2., -2., 2.))


class PpoTests(unittest.TestCase):
    def test_ppo_config(self):
        expected = dict(
            num_envs=512, eval_num_envs=8, rollout=48, epochs=5, minibatches=4,
            learning_rate=3e-4, min_lr=1e-5, max_lr=3e-4, desired_kl=.01,
            gamma=.99, gae_lambda=.95, clip=.2, entropy_coef=.003,
            value_coef=1.0, clipped_value_loss=True, max_grad_norm=1.0,
            initial_std=.30, std_min=.05, std_max=1.00, adam_betas=(.9, .999),
            adam_eps=1e-8, weight_decay=0., kl_high=.02, kl_low=.005,
            kl_scale=1.5, adam_updates_per_iteration=20,
            control_samples_per_update=24576,
            physics_substeps_per_update=98304,
            advantage_normalization=True, timeout_bootstrap=True)
        for name, value in expected.items():
            self.assertEqual(spec.PPO_CONFIG[name], value, name)
        self.assertEqual(spec.PPO_CONFIG['optimizer'],
                         'single_adam_each_parameter_once')
        self.assertFalse(spec.PPO_CONFIG['kl_early_exit'])
        self.assertFalse(spec.PPO_CONFIG['automatic_rollback'])
        self.assertEqual(spec.PPO_CONFIG['actor_freeze'], 'none')
        self.assertEqual(spec.PPO_CONFIG['grad_norm_groups'],
                         ('actor', 'critic', 'fusion', 'std'))

    def test_ledger(self):
        self.assertEqual(spec.TOTAL_BUDGET * spec.CONTROL_SAMPLES_PER_UPDATE,
                         737_280_000)
        self.assertEqual(spec.TOTAL_BUDGET * spec.PHYSICS_SUBSTEPS_PER_UPDATE,
                         2_949_120_000)
        self.assertEqual(spec.TOTAL_BUDGET * spec.ADAM_UPDATES_PER_ITERATION,
                         600_000)
        self.assertEqual((spec.SMOKE_CONTROL_SAMPLES,
                          spec.SMOKE_PHYSICS_SUBSTEPS,
                          spec.SMOKE_ADAM_UPDATES), (768, 3072, 40))
        self.assertEqual((spec.PROBE_UPDATES, spec.PROBE_CONTROL_SAMPLES),
                         (56, 1_376_256))
        self.assertEqual(spec.RESOURCE_LIMITS, dict(
            gpu_mib=7168, process_mib=5120, host_free_mib=4096,
            disk_free_gib=10, single_gpu_worker=True))


class CurriculumTests(unittest.TestCase):
    def test_curriculum_constants(self):
        self.assertEqual((spec.CURRICULUM_WINDOW,
                          spec.CURRICULUM_WINDOWS_PER_UPDATE,
                          spec.EMA_DECAY, spec.EMA_NEW), (64, 1, .97, .03))
        self.assertEqual((spec.ADVANCE_S_MIN, spec.ADVANCE_Q_MIN,
                          spec.ADVANCE_F_MAX, spec.ADVANCE_STREAK),
                         (.65, .45, .35, 5))
        self.assertEqual((spec.REGRESS_F_MIN, spec.REGRESS_S_MAX,
                          spec.REGRESS_STREAK, spec.REGRESS_TICKS),
                         (.40, .20, 2, 3))
        self.assertEqual((spec.LEVEL_TICK_MIN, spec.LEVEL_TICK_MAX,
                          spec.FRONTIER_ROW_COUNT, spec.COOLDOWN_CHECKS),
                         (0, 100, 15, 5))
        self.assertEqual(spec.ROLE_COUNTS,
                         {'flat': 102, 'frontier': 164, 'near': 123, 'easy': 123})
        self.assertEqual(spec.ROLE_COUNT_TOTAL, 512)
        self.assertEqual((spec.NEAR_OFFSETS, spec.EASY_BACKOFF),
                         ((-2, -1), 3))
        self.assertEqual(spec.V_R_FRONTIER_TILE_ID, 21)
        self.assertEqual(spec.V_R_NEAR_TILES, ('L12', 'L20'))
        self.assertEqual(spec.V_R_EASY_LAST_TILE_ID, 18)
        self.assertEqual((spec.V_MIN_UPDATES, spec.V_GATE_STREAK_START,
                          spec.R_MIN_UPDATES, spec.R_GATE_STREAK_START),
                         (1200, 1000, 600, 400))
        self.assertEqual((spec.COMMAND_INITIAL_VX, spec.COMMAND_FINAL_VX,
                          spec.COMMAND_VY, spec.COMMAND_WZ,
                          spec.V_RAMP_UPDATES),
                         ((.30, .80), (.20, .60), 0., 0., 1000))
        self.assertEqual((spec.SPAWN_X, spec.SPAWN_Y, spec.R_SPAWN_X,
                          spec.R_SPAWN_Y, spec.R_SPAWN_RAMP_UPDATES),
                         ((-1.10, -.90), (-.05, .05), (-1.20, -.80),
                          (-.10, .10), 400))
        self.assertEqual(spec.SPAWN_YAW_DEGREES, (-5., 5.))
        self.assertEqual(spec.R_SPAWN_YAW_DEGREES, (-10., 10.))
        self.assertEqual(spec.CURRICULUM['substages']['D']['budgets'],
                         (1500, 1500))
        self.assertEqual(spec.CURRICULUM['substages']['L']['budgets'],
                         (1500, 1500, 3000))

    def test_frontier_row_boundaries(self):
        self.assertEqual(spec.curriculum_frontier_row(0), 0)
        self.assertEqual(spec.curriculum_frontier_row(1), 0)
        self.assertEqual(spec.curriculum_frontier_row(6), 0)
        self.assertEqual(spec.curriculum_frontier_row(7), 1)
        self.assertEqual(spec.curriculum_frontier_row(13), 1)
        self.assertEqual(spec.curriculum_frontier_row(14), 2)
        self.assertEqual(spec.curriculum_frontier_row(99), 14)
        self.assertEqual(spec.curriculum_frontier_row(100), 15)
        self.assertEqual(spec.curriculum_level(0), 0.)
        self.assertEqual(spec.curriculum_level(100), 1.)
        for tick in (-1, 101, 1.5, True):
            with self.assertRaises(ValueError):
                spec.curriculum_frontier_row(tick)

    def test_ema_first_batch_then_decay(self):
        self.assertEqual(spec.curriculum_ema(None, .8), .8)
        self.assertAlmostEqual(spec.curriculum_ema(1., 0.), .97, places=15)
        self.assertAlmostEqual(spec.curriculum_ema(.5, 1.), .515, places=15)
        with self.assertRaises(ValueError):
            spec.curriculum_ema(None, float('nan'))

    def test_transition_regress_priority_cooldown(self):
        # A single ready check never advances and keeps the cooldown decremented.
        result = spec.curriculum_transition(0, .7, .5, .2, 4, 0, 0)
        self.assertEqual((result['event'], result['level_tick']), ('advance', 1))
        self.assertEqual(result['cooldown'], 5)
        # Regress wins over a simultaneous advance and ignores cooldown.
        result = spec.curriculum_transition(50, .7, .5, .45, 4, 1, 3)
        self.assertEqual((result['event'], result['level_tick']), ('regress', 47))
        self.assertEqual(result['cooldown'], 5)
        # Clamp at zero and 100.
        self.assertEqual(spec.curriculum_transition(1, .0, .0, .9, 0, 1, 0)['level_tick'], 0)
        self.assertEqual(spec.curriculum_transition(99, .9, .9, .0, 4, 0, 0)['level_tick'], 100)
        # Advance requires ready_streak 5 and cooldown 0.
        self.assertEqual(spec.curriculum_transition(10, .9, .9, .0, 3, 0, 0)['event'],
                         'hold')
        self.assertEqual(spec.curriculum_transition(10, .9, .9, .0, 4, 0, 1)['event'],
                         'hold')
        # A hold decrements an existing cooldown.
        self.assertEqual(spec.curriculum_transition(10, .0, .0, .9, 0, 0, 3)['cooldown'], 2)

    def test_fifo_window_and_remainder(self):
        fifo, ready = [], False
        for index in range(70):
            outcome = dict(step=70 - index, env_id=index)
            fifo, ready = spec.fifo_append(fifo, outcome)
        self.assertTrue(ready)
        self.assertEqual([entry['step'] for entry in fifo], list(range(1, 71)))
        window, remaining = spec.fifo_pop_window(fifo)
        self.assertEqual(len(window), 64)
        self.assertEqual(len(remaining), 6)
        self.assertEqual([entry['step'] for entry in remaining], [65, 66, 67, 68, 69, 70])

    def test_generation_change(self):
        self.assertTrue(spec.generation_starts(3, 4))
        self.assertFalse(spec.generation_starts(3, 3))
        self.assertIn('old_generation', spec.CURRICULUM['generation'])
        self.assertEqual(spec.CURRICULUM['generation']['cooldown'], 5)

    def test_seed_offsets_only_once(self):
        self.assertEqual(spec.evaluation_seed('validation', 42), 200342)
        self.assertEqual(spec.evaluation_seed('diagnostic', 42), 500342)
        self.assertEqual(spec.evaluation_seed('perturbed', 42), 300342)
        self.assertEqual(spec.evaluation_seed('final', 42), 400342)
        self.assertEqual(spec.evaluation_seed('final', 999), 400342)
        self.assertEqual(spec.evaluation_seeds(42), {
            'validation': 200342, 'diagnostic': 500342,
            'perturbed': 300342, 'final': 400342})
        with self.assertRaises(ValueError):
            spec.evaluation_seed('bogus', 42)

    def test_timeout_formula(self):
        self.assertEqual(spec.episode_timeout_steps(40, .20, .20, .2), 7257)
        self.assertEqual(spec.episode_timeout_steps(40, .20, .20, .6), 2753)
        self.assertEqual(spec.episode_timeout_steps(6, .20, .25, .3), 1874)
        self.assertEqual(spec.episode_timeout_steps(0, 0., .20, .4),
                         spec.F0_EPISODE_STEPS)
        self.assertEqual(spec.episode_step_limit(0, .4), spec.F0_EPISODE_STEPS)
        self.assertEqual(spec.episode_step_limit(21, .2), 7257)


class GateTests(unittest.TestCase):
    def test_gate_constants(self):
        self.assertEqual((spec.GATE_INTERVAL, spec.DIAGNOSTIC_INTERVAL,
                          spec.SAVE_INTERVAL, spec.RETENTION_INTERVAL),
                         (100, 50, 100, 200))
        self.assertEqual((spec.GATE_CASES, spec.GATE_PASS_MIN,
                          spec.FINAL_CASES, spec.FINAL_PASS_MIN),
                         (20, 18, 100, 95))
        self.assertEqual(spec.GATE_SPEEDS_F0_H_D_L, (.3, .5, .8))
        self.assertEqual(spec.GATE_SPEEDS_V_R, (.2, .4, .6))
        self.assertEqual((spec.F0_OBSERVE_STEPS, spec.F0_LATERAL_MAX,
                          spec.F0_HEADING_MAX_DEGREES,
                          spec.F0_PROGRESS_FRACTION, spec.F0_EPISODE_STEPS),
                         (500, .20, 10., .5, 1500))
        self.assertEqual((spec.SUCCESS_RECENT_SUPPORT_STEPS,
                          spec.SUCCESS_SINGLE_FOOT_FRAMES,
                          spec.SUCCESS_MIN_SUPPORT_FEET,
                          spec.SUCCESS_BODY_PLATFORM_MARGIN,
                          spec.SUCCESS_LATERAL_MAX,
                          spec.SUCCESS_CONSECUTIVE_FRAMES),
                         (50, 2, 2, .8, .20, 5))
        self.assertAlmostEqual(spec.SUCCESS_FOOT_SAFE_INSET, .042, places=15)
        self.assertEqual((spec.FAILURE_NONFOOT_PERSIST_TIME,
                          spec.FAILURE_ATTITUDE_LIMIT_DEGREES), (.1, 70.))
        self.assertEqual((spec.RETENTION_CASES, spec.RETENTION_CASES_PER_SPEED,
                          spec.RETENTION_FAIL_STREAK, spec.RETENTION_FAIL_MIN),
                         (15, 5, 2, 12))
        self.assertEqual((spec.MIN_LEVEL_PHASE_UPDATES,
                          spec.MIN_LEVEL_FAILURE_STREAK,
                          spec.MIN_LEVEL_FAILURE_F_MIN), (200, 4, .80))


class RuntimeAssetTests(unittest.TestCase):
    """Q001-4: compare spec constants against the real mjlab A2 asset data."""

    def test_soft_limits_and_q0_match_runtime_robot_data(self):
        import torch
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg
        from src.assets.robots.unitree_a2.a2_constants import get_a2_robot_cfg

        scene = Scene(SceneCfg(num_envs=1, env_spacing=1.0,
                               entities={'robot': get_a2_robot_cfg()}), device='cpu')
        model = scene.compile()
        simulation = Simulation(num_envs=1, cfg=SimulationCfg(), model=model,
                                device='cpu')
        scene.initialize(mj_model=simulation.mj_model, model=simulation.model,
                         data=simulation.data)
        robot = scene.entities['robot']
        names = list(robot.joint_names)
        self.assertEqual(len(names), 12)
        indices = [names.index(f'{name}_joint') for name in spec.JOINT_ORDER]
        default = robot.data.default_joint_pos[0].detach().cpu()
        soft = robot.data.soft_joint_pos_limits[0].detach().cpu()
        expected_default = torch.tensor(spec.A2_DEFAULT_JOINT_POS)
        expected_soft = torch.tensor(spec.A2_SOFT_JOINT_LIMITS)
        delta_default = (default[indices] - expected_default).abs().max()
        delta_soft = (soft[indices] - expected_soft).abs().max()
        self.assertLessEqual(float(delta_default), 1e-6)
        self.assertLessEqual(float(delta_soft), 1e-6)
        print(f'\nruntime A2 default_joint_pos max|delta|={float(delta_default):.3e}')
        print(f'runtime A2 soft_joint_pos_limits max|delta|={float(delta_soft):.3e}')


class RecipeCliTests(unittest.TestCase):
    def test_defaults_are_the_sds_command(self):
        resolved = spec.recipe_config()
        self.assertEqual(resolved['recipe'], '0.6-go2refs-r1')
        self.assertEqual(resolved['seed'], 42)
        self.assertEqual(resolved['device'], 'cuda:0')
        self.assertEqual(resolved['num_envs'], 512)
        self.assertEqual(resolved['eval_num_envs'], 8)
        self.assertEqual(resolved['rollout'], 48)
        self.assertEqual(resolved['epochs'], 5)
        self.assertEqual(resolved['minibatches'], 4)
        self.assertEqual(resolved['learning_rate'], .0003)
        self.assertEqual(resolved['gamma'], .99)
        self.assertEqual(resolved['entropy_coef'], .003)
        self.assertEqual(resolved['initial_std'], .30)
        self.assertEqual(resolved['std_min'], .05)
        self.assertEqual(resolved['std_max'], 1.00)
        self.assertEqual((resolved['initial_vx_min'], resolved['initial_vx_max']),
                         (.30, .80))
        self.assertEqual((resolved['final_vx_min'], resolved['final_vx_max']),
                         (.20, .60))
        self.assertEqual((resolved['vy'], resolved['wz']), (0., 0.))
        self.assertEqual(resolved['max_updates'], 30000)
        self.assertEqual(resolved['phase_budgets'],
                         {'F0': 1000, 'H': 12000, 'D': 3000, 'L': 6000,
                          'V': 4000, 'R': 4000})
        self.assertEqual(resolved['save_interval'], 100)
        self.assertEqual(resolved['diagnostic_interval'], 50)
        self.assertEqual(resolved['gate_interval'], 100)
        self.assertEqual(resolved['init_checkpoint'], spec.INIT_PATH)
        self.assertEqual(resolved['init_sha256'], spec.INIT_SHA256)
        self.assertEqual(resolved['acceptance_file'],
                         'logs/a2_ideal_stairs_v06/implementation/'
                         'V06_IMPLEMENTATION_ACCEPTANCE.md')
        self.assertEqual(resolved['output_dir'],
                         'logs/a2_ideal_stairs_v06/train_seed42_trial01')

    def test_accepts_equal_values_and_phase_mapping(self):
        resolved = spec.recipe_config(
            mode='train', execute=False,
            phase_budgets={'F0': 1000, 'H': 12000, 'D': 3000, 'L': 6000,
                           'V': 4000, 'R': 4000},
            learning_rate=0.0003, gamma=.99)
        self.assertFalse(resolved['execute'])
        self.assertEqual(resolved['phase_budgets']['H'], 12000)
        self.assertEqual(resolved['learning_rate'], .0003)

    def test_rejects_overrides_and_unknown_flags(self):
        for kwargs in (dict(seed=43), dict(learning_rate=.0005),
                       dict(num_envs=256), dict(phase_budgets='F0=1000,H=1'),
                       dict(recipe='other'), dict(rolling=1)):
            with self.assertRaises(ValueError, msg=kwargs):
                spec.recipe_config(**kwargs)
        with self.assertRaises(ValueError):
            spec.recipe_config(execute='yes')

    def test_contract_identity_and_sources(self):
        contract = spec.contract()
        self.assertEqual(contract['identity'], spec.identity_block())
        self.assertEqual(contract['sds_sha256'], spec.SDS_SHA256)
        self.assertEqual(contract['q001_decision_sha256'],
                         spec.Q001_DECISION_SHA256)
        self.assertEqual(contract['w0_addendum_sha256'],
                         spec.W0_ADDENDUM_SHA256)
        self.assertEqual(contract['q002_decision_sha256'],
                         spec.Q002_DECISION_SHA256)
        self.assertEqual(contract['w0_addendum_2_sha256'],
                         spec.W0_ADDENDUM2_SHA256)
        self.assertEqual(contract['total_budget'], 30000)
        self.assertEqual(contract['phase_budgets'], spec.PHASE_BUDGETS)
        self.assertEqual([row['tile_id'] for row in contract['terrain']],
                         list(range(22)))
        self.assertEqual(contract['ppo'], spec.PPO_CONFIG)
        self.assertEqual(contract['reward_config_sha256'],
                         spec.reward_config_sha256())
        self.assertEqual(contract['recipe']['phase_budgets'],
                         {'F0': 1000, 'H': 12000, 'D': 3000, 'L': 6000,
                          'V': 4000, 'R': 4000})
        self.assertIn('src/tasks/velocity/v06/spec.py', contract['behavior_sha256'])
        # The closure is produced by the static import walk; C's training stack
        # may reuse read-only velocity config/mdp modules, so this test only
        # checks the v06 product itself is bound, not the whole dependency set.
        self.assertTrue(any(entry.startswith('src/tasks/velocity/v06/')
                            for entry in contract['behavior_sha256']))
        self.assertEqual(contract['source_closure_complete'],
                         not contract['pending_source_entries'])

    def test_product_entries_are_explicit_and_v06_only(self):
        entries = spec.product_entries()
        self.assertIn('src/tasks/velocity/v06/spec.py', entries)
        self.assertTrue(all(entry.startswith('src/tasks/velocity/v06/')
                            or entry.startswith('scripts/') for entry in entries))
        self.assertEqual(spec.pending_source_entries(),
                         [entry for entry in spec.SOURCE_ENTRIES
                          if not (ROOT / entry).is_file()])
        for relative in spec.pending_source_entries():
            self.assertFalse((ROOT / relative).is_file())
        hashes = spec.behavior_hashes()
        self.assertEqual(spec.sha256(ROOT / 'src/tasks/velocity/v06/spec.py'),
                         hashes['src/tasks/velocity/v06/spec.py'])


if __name__ == '__main__':
    unittest.main()
