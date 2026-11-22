"""v0.5-r2 identities, stage/budget table, slot allocation, reward identity and the
explicit product-entry source closure.

The r2 contract must be *identical* to r1 everywhere SDS 1.2 says "明确不变":
this module compares the complete reward configuration, stage table, slot counts
and episode caps against the read-only r1 module (``src/tasks/velocity/v05/spec.py``)
and asserts that the only reward difference is the recipe ``version`` field.

The source contract is an explicit product-entry closure: it binds the bootstrap,
the migration script and every existing ``src/tasks/velocity/v05_r2/*.py`` while
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
import a2_v05_r2_bootstrap  # noqa: E402

a2_v05_r2_bootstrap.activate(ROOT)

from src.tasks.velocity.v05 import spec as r1_spec  # noqa: E402  (read-only reference)
from src.tasks.velocity.v05_r2 import spec  # noqa: E402

V05R2_FILES = {str(path.relative_to(spec.ROOT))
               for path in (spec.ROOT / 'src/tasks/velocity/v05_r2').glob('*.py')}


def stable_contract(**kwargs):
    """Return a contract whose source closure was identical on two consecutive reads.

    Several workers share this checkout, so a file can be written between the
    contract hash and its verification.  Identical back-to-back reads prove the
    closure was quiescent for a whole hash pass without weakening the check.
    """
    attempts = []
    for _ in range(3):
        first = spec.contract(**kwargs)
        second = spec.contract(**kwargs)
        if first == second:
            return first
        attempts.append(sorted(set(first['behavior_sha256'])
                               ^ set(second['behavior_sha256'])))
    raise AssertionError(f'the v0.5-r2 source closure kept changing during the test: {attempts}')


class IdentityTests(unittest.TestCase):
    def test_version_identity_and_paths(self):
        self.assertEqual((spec.CODE_VERSION, spec.RECIPE_VERSION, spec.SDS_REVISION,
                          spec.SCHEMA_VERSION, spec.ARCHITECTURE, spec.TASK,
                          spec.PROTOCOL_ID, spec.FLAT_PROTOCOL_ID, spec.LOG_ROOT),
                         ('0.5.1', '0.5-r2', '1.0', 7, 'v05-route-state-v1',
                          'Unitree-A2-IdealHeightMap-Stairs-V05R2',
                          'v05-r2-stair-climb-first-v1', 'v05-r2-flat-world-route-v1',
                          'logs/a2_ideal_stairs_v05_r2'))
        self.assertEqual((spec.CONTROL_DT, spec.PHYSICS_DT, spec.DECIMATION, spec.MAP_HZ,
                          spec.ACTION_SCALE), (.02, .005, 4, 10, .25))
        self.assertEqual(spec.ROOT, ROOT)
        self.assertEqual(spec.INIT_PATH,
                         '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt')
        self.assertEqual(spec.INIT_SHA256,
                         '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722')
        self.assertEqual(spec.sha256(spec.INIT_PATH), spec.INIT_SHA256)
        self.assertEqual(spec.SMOKE_CONTAINER, 'v05-r2-smoke-state-v1')
        self.assertFalse(spec.SMOKE_FORMAL_RESUMABLE)
        self.assertEqual(spec.FOOT_ORDER, ('FL', 'FR', 'RL', 'RR'))
        self.assertEqual(spec.SPEEDS, (.2, .4, .6))

    def test_frozen_document_sha256_bound_and_verified(self):
        """The contract binds the SDS and both Q003-r2 decisions by real SHA256."""
        self.assertEqual(spec.SDS_SHA256,
                         '9ff356b022e6205f0a181ff4048ef0f6e773742250d3174bf9b087986c8a4abc')
        self.assertEqual(spec.DESIGN_DECISION_SHA256,
                         '78515bb951a96a8c69b9f10b90043a6312ed39124bc7898bd08b89e046d3a0fc')
        self.assertEqual(spec.IMPLEMENTATION_DECISION_SHA256,
                         '0cd996e72c494a6cd981db067230710dbc8a5c0930d97bed1e86069a86895bf1')
        self.assertEqual(len(spec.FROZEN_DOCUMENTS), 3)
        for name, relative, digest in spec.FROZEN_DOCUMENTS:
            path = spec.ROOT / relative
            self.assertTrue(path.is_file(), (name, relative))
            self.assertEqual(spec.sha256(path), digest, name)
        contract = stable_contract()
        self.assertEqual(contract['sds_path'], spec.SDS_PATH)
        self.assertEqual(contract['sds_sha256'], spec.SDS_SHA256)
        self.assertEqual(contract['document_identity'], spec.document_identity())
        self.assertEqual(contract['document_identity']['sds_sha256'], spec.SDS_SHA256)
        self.assertEqual(contract['document_identity']['design_decision_sha256'],
                         spec.DESIGN_DECISION_SHA256)
        self.assertEqual(contract['document_identity']['implementation_decision_sha256'],
                         spec.IMPLEMENTATION_DECISION_SHA256)
        self.assertEqual(contract['design_decision_path'], spec.DESIGN_DECISION_PATH)
        self.assertEqual(contract['design_decision_sha256'], spec.DESIGN_DECISION_SHA256)
        self.assertEqual(contract['implementation_decision_path'],
                         spec.IMPLEMENTATION_DECISION_PATH)
        self.assertEqual(contract['implementation_decision_sha256'],
                         spec.IMPLEMENTATION_DECISION_SHA256)

    def test_observation_contract_is_five_keys_with_route(self):
        self.assertEqual(spec.OBSERVATION,
                         {'proprio': [45], 'history': [5, 45], 'heightmap': [3, 48, 32],
                          'privileged': [35], 'route_error': [3]})
        self.assertEqual(spec.OBSERVATION_KEYS,
                         ('proprio', 'history', 'heightmap', 'privileged', 'route_error'))
        self.assertEqual(spec.ACTOR_OBSERVATION_KEYS,
                         ('proprio', 'history', 'heightmap', 'route_error'))
        self.assertEqual(spec.CRITIC_OBSERVATION_KEYS,
                         ('proprio', 'history', 'heightmap', 'privileged', 'route_error'))
        self.assertEqual(spec.ROUTE_ERROR_KEY, 'route_error')
        self.assertEqual((spec.ROUTE_ERROR_DIM, spec.ROUTE_LATERAL_SCALE,
                          spec.ROUTE_LATERAL_CLIP, spec.ROUTE_YAW_AXIS),
                         (3, .20, 4., 0.))
        self.assertAlmostEqual(spec.ROUTE_LATERAL_SATURATION_M, .80, places=15)
        # r2 adds exactly one key: no key may be merged into proprio/history.
        self.assertEqual(set(spec.OBSERVATION) - set(r1_spec.OBSERVATION),
                         {'route_error'})
        self.assertEqual(set(r1_spec.OBSERVATION) - set(spec.OBSERVATION), set())
        self.assertEqual(spec.CRITIC_INPUT_DIMS, 211)
        self.assertEqual((spec.ETH_NORMALIZER_DIMS, spec.CRITIC_NORMALIZER_DIMS), (45, 80))

    def test_route_state_contract(self):
        route = spec.route_state()
        self.assertEqual(route, spec.ROUTE_STATE)
        self.assertEqual(route['architecture'], spec.ARCHITECTURE)
        self.assertEqual(route['key'], 'route_error')
        self.assertEqual(route['dim'], 3)
        self.assertEqual(route['dtype'], 'float32')
        self.assertEqual(route['yaw_axis'], 0.)
        self.assertEqual(route['lateral_scale_m'], .20)
        self.assertEqual(route['lateral_clip'], 4.)
        self.assertEqual(route['angle_unit'], 'radians')
        self.assertEqual(route['order'], ['clip(local_y/0.20,-4.0,4.0)',
                                          'sin(e_world)', 'cos(e_world)'])
        self.assertIn('wrap_to_pi', route['e_world'])
        self.assertTrue(route['per_environment_tile'])
        self.assertFalse(route['in_eth_normalizer'])
        self.assertFalse(route['in_critic_normalizer'])
        self.assertEqual(route['normalizer'],
                         'none_fixed_scale_no_running_mean_no_running_var')
        self.assertEqual(route['terminal_observation'], 'pre-reset pose')
        self.assertEqual(route['deployment_label'], 'ideal_route_state')
        self.assertIn('no hardware validation', route['deployment'])
        self.assertEqual(route['consumer'],
                         ['actor.route_fusion', 'critic input tail [208:211]'])
        self.assertIn('reward', route['not_in'])
        # A copy never aliases the frozen table.
        route['dim'] = 99
        self.assertEqual(spec.ROUTE_STATE['dim'], 3)

    def test_network_key_contract(self):
        self.assertEqual(spec.ROUTE_ERROR_DIM, 3)
        self.assertEqual(spec.ROUTE_FUSION_KEYS,
                         ('actor.route_fusion.weight', 'actor.route_fusion.bias'))
        self.assertEqual(spec.ROUTE_FUSION_SHAPES,
                         {'actor.route_fusion.weight': (512, 3),
                          'actor.route_fusion.bias': (512,)})
        self.assertEqual(spec.FUSION_KEYS, ('actor.fusion.weight', 'actor.fusion.bias'))
        self.assertEqual(spec.CRITIC_FIRST_KEY, 'critic.0.weight')
        self.assertEqual(spec.CRITIC_FIRST_SHAPE, (512, 211))
        self.assertEqual(spec.CRITIC_HIDDEN, (512, 256, 128))
        self.assertEqual(spec.ACTOR_HIDDEN, (512, 256, 128))
        self.assertEqual(spec.ACTION_DIM, 12)
        self.assertEqual(len(spec.ETH_ACTOR_SOURCE_KEYS), 13)
        self.assertIn('critic_state_dict', spec.ETH_FORBIDDEN_SOURCE_BLOCKS)


class StageTableTests(unittest.TestCase):
    def test_ten_stage_rows_match_r1_exactly(self):
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
        self.assertEqual(rows, [(stage.id, stage.tile_id, stage.N, stage.h, stage.budget,
                                 stage.gate) for stage in r1_spec.STAGES])
        self.assertEqual(spec.STAGE_ORDER, r1_spec.STAGE_ORDER)
        self.assertEqual(spec.STAGE_BY_TILE[0].id, 'F0')
        self.assertEqual(spec.STAGE_BY_TILE[9].id, 'S5')
        self.assertNotIn('flat', spec.STAGE_BY_ID)
        self.assertEqual(spec.FLAT_ALIAS, 'flat')
        self.assertEqual(spec.FLAT_TILE_ID, 0)
        self.assertEqual(sum(stage.budget for stage in spec.STAGES), 30000)
        self.assertEqual(sum(stage.gate for stage in spec.STAGES), 1500)
        self.assertEqual(spec.FORMAL_STAGE_BUDGETS,
                         'F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,S2=4000,'
                         'S3=4000,S4=5000,S5=7000')
        self.assertEqual(spec.PILOT_STAGE_BUDGETS, 'F0=2000,P1=500')
        self.assertEqual((spec.PILOT_P1_UPDATES, spec.PILOT_MAX_UPDATES), (500, 2500))
        self.assertEqual(spec.PILOT_STAGES, ('F0', 'P1'))
        for stage in spec.STAGES:
            self.assertEqual(spec.previous_stage_id(stage.id),
                             spec.STAGE_ORDER[spec.STAGE_ORDER.index(stage.id) - 1]
                             if stage.id != 'F0' else None)

    def test_512_and_8_slot_counts_match_r1_exactly(self):
        expected = {
            'F0': {'flat': 512, 'previous': 0, 'current': 0},
            'P1': {'flat': 102, 'previous': 0, 'current': 410},
        }
        for stage in spec.STAGES[2:]:
            expected[stage.id] = {'flat': 51, 'previous': 102, 'current': 359}
        for stage in spec.STAGES:
            counts = spec.slot_counts(stage.id, 512)
            self.assertEqual(counts, expected[stage.id], stage.id)
            self.assertEqual(spec.slot_counts(stage.id, 512),
                             r1_spec.slot_counts(stage.id, 512), stage.id)
            self.assertEqual(spec.slot_layout(stage.id, 512),
                             r1_spec.slot_layout(stage.id, 512), stage.id)
            self.assertEqual(spec.slot_tiles(stage.id, 512),
                             r1_spec.slot_tiles(stage.id, 512), stage.id)
            self.assertEqual(sum(counts.values()), 512)
        for count in (1, 2, 3, 4, 7, 8, 51, 512, 513, 1024):
            for stage in spec.STAGES:
                values = spec.slot_counts(stage.id, count)
                self.assertEqual(sum(values.values()), count, (count, stage.id))
                self.assertEqual(values, r1_spec.slot_counts(stage.id, count),
                                 (count, stage.id))
        for args in ((0, 'F0'), (-1, 'S1'), (1.5, 'S1'), (True, 'S1'), (10, 'flat'),
                     (10, 'R0'), (10, 'S6'), (10, 's1'), (10, None)):
            with self.assertRaises(ValueError, msg=str(args)):
                spec.slot_counts(*args)

    def test_episode_step_limit_is_single_source(self):
        self.assertEqual(spec.F0_EPISODE_STEPS, 1500)
        self.assertEqual(spec.stair_episode_steps(4, .08, .4), 1266)
        self.assertEqual(spec.stair_episode_steps(0, 0., .4), 1500)
        self.assertEqual(spec.stair_episode_steps(40, .2, .2), 7257)
        self.assertEqual(spec.stair_episode_steps(40, .2, .6), 2753)
        self.assertEqual(spec.stair_episode_steps(4, 0., .4), 1500)
        self.assertEqual(spec.episode_step_limit(40, .2, .4), 3879)
        self.assertEqual(spec.episode_step_limit(4, .08, .2), 2031)
        self.assertEqual(spec.episode_step_limit(4, .08, .6), 1011)
        for stage in spec.STAGES:
            for speed in spec.SPEEDS:
                self.assertEqual(spec.episode_step_limit(stage.N, stage.h, speed),
                                 r1_spec.episode_step_limit(stage.N, stage.h, speed),
                                 (stage.id, speed))
        for args in ((-1, .08, .4), (4.5, .08, .4), (True, .08, .4), (4, -.08, .4),
                     (4, float('nan'), .4), (4, float('inf'), .4), (4, .08, 0.),
                     (4, .08, -.4), (4, .08, float('nan')), (4, .08, True), (4, .08, 'fast')):
            with self.assertRaises(ValueError, msg=str(args)):
                spec.episode_step_limit(*args)


class ModeBudgetTests(unittest.TestCase):
    def test_pilot_and_formal_single_source(self):
        self.assertEqual(spec.MODE_BUDGETS,
                         {'pilot': dict(total=2500, stages={'F0': 2000, 'P1': 500}),
                          'formal': dict(total=30000,
                                         stages={stage.id: stage.budget
                                                 for stage in spec.STAGES})})
        pilot = spec.mode_budget('pilot')
        self.assertEqual(pilot['total'], 2500)
        self.assertEqual(pilot['stages'], {'F0': 2000, 'P1': 500})
        self.assertEqual(pilot['source'], 'mode_budget')
        self.assertTrue(pilot['formal_resumable'])
        formal = spec.mode_budget('formal')
        self.assertEqual(formal['total'], 30000)
        self.assertEqual(formal['stages'], {stage.id: stage.budget for stage in spec.STAGES})
        self.assertEqual(sum(formal['stages'].values()), 30000)
        # The r1 defect must be impossible: the pilot cap is not the ten-stage sum.
        self.assertNotEqual(pilot['total'], sum(stage.budget for stage in spec.STAGES))
        self.assertEqual(spec.stage_budget('pilot', 'P1'), 500)
        self.assertEqual(spec.stage_budget('formal', 'P1'), 1000)
        self.assertEqual(spec.stage_budget('formal', 'S5'), 7000)
        with self.assertRaises(ValueError):
            spec.stage_budget('pilot', 'S2')

    def test_probe_and_smoke_never_inherit_the_formal_cap(self):
        probe = spec.mode_budget('probe')
        self.assertEqual(probe['total'], 56)
        self.assertEqual(probe['stages'], spec.MODE_BUDGETS['formal']['stages'])
        self.assertEqual((spec.PROBE_WARMUP_UPDATES, spec.PROBE_MEASURE_UPDATES,
                          spec.PROBE_RECOVERY_UPDATES), (5, 50, 1))
        self.assertEqual(probe['total'],
                         spec.PROBE_WARMUP_UPDATES + spec.PROBE_MEASURE_UPDATES
                         + spec.PROBE_RECOVERY_UPDATES)
        self.assertEqual(probe['total'] * 48 * 512, 1376256)
        smoke = spec.mode_budget('smoke')
        self.assertEqual(smoke['total'], 2)
        self.assertEqual(smoke['stages'], {})
        self.assertFalse(smoke['formal_resumable'])
        self.assertNotIn('smoke', spec.MODE_BUDGETS)
        self.assertNotIn('probe', spec.MODE_BUDGETS)
        for mode in ('pilot', 'formal', 'probe'):
            self.assertEqual(spec.mode_budget(mode)['total'], spec.contract(mode=mode)['total_budget'])

    def test_mode_budget_is_a_deep_copy_and_rejects_unknown_modes(self):
        record = spec.mode_budget('formal')
        record['stages']['F0'] = 1
        record['total'] = 1
        self.assertEqual(spec.MODE_BUDGETS['formal']['stages']['F0'], 2000)
        self.assertEqual(spec.MODE_BUDGETS['formal']['total'], 30000)
        for mode in ('', 'test', 'PILOT', None, 42, False, 'formal '):
            with self.assertRaises(ValueError, msg=repr(mode)):
                spec.mode_budget(mode)

    def test_contract_budget_fields_are_the_same_source(self):
        for mode in spec.CONTRACT_MODES:
            contract = stable_contract(mode=mode)
            budget = spec.mode_budget(mode)
            self.assertEqual(contract['total_budget'], budget['total'], mode)
            self.assertEqual(contract['update_cap'], budget['total'], mode)
            self.assertEqual(contract['stage_budgets'], budget['stages'], mode)
            self.assertEqual(contract['effective_cap_source'], budget['source'], mode)
            self.assertEqual(contract['mode'], mode)
            self.assertEqual(contract['mode_budgets'],
                             {name: dict(total=record['total'],
                                         stages=record['stages'])
                              for name, record in spec.MODE_BUDGETS.items()})
        pilot = stable_contract(mode='pilot')
        self.assertEqual((pilot['total_budget'], pilot['update_cap']), (2500, 2500))
        self.assertEqual(pilot['effective_cap_source'], 'mode_budget')
        self.assertEqual(pilot['stage_budgets'], {'F0': 2000, 'P1': 500})
        self.assertEqual(pilot['pilot_stage_budgets'], 'F0=2000,P1=500')
        self.assertEqual(pilot['pilot_p1_updates'], 500)
        self.assertEqual(pilot['pilot_max_updates'], 2500)
        formal = stable_contract(mode='formal')
        self.assertEqual((formal['total_budget'], formal['update_cap']), (30000, 30000))
        with self.assertRaises(ValueError):
            spec.contract(mode='comparison')


class RewardRecipeTests(unittest.TestCase):
    def test_reward_config_is_r1_plus_the_version_identity_only(self):
        cfg = spec.reward_config()
        r1_cfg = r1_spec.reward_config()
        self.assertEqual((cfg['version'], cfg['dt']), ('0.5-r2', .02))
        self.assertEqual(r1_cfg['version'], '0.5-r1')
        # SDS 3.9.2: byte-identical after removing the identity field.
        self.assertEqual(dict(cfg, version='0.5-r1'), r1_cfg)
        self.assertEqual(spec.canonical_json(dict(cfg, version='0.5-r1')),
                         spec.canonical_json(r1_cfg))
        # Explicit: the weights SDS 3.7 freezes must never move to -2.0.
        self.assertEqual((cfg['continuous_weights']['center'],
                          cfg['continuous_weights']['heading']), (-.5, -.5))
        self.assertEqual((spec.CENTER_WEIGHT, spec.HEADING_WEIGHT), (-.5, -.5))
        self.assertEqual(cfg['continuous_weights'], r1_cfg['continuous_weights'])
        self.assertEqual(cfg['center']['weight'], -.5)
        self.assertEqual(cfg['heading']['weight'], -.5)
        self.assertIn('min((local_y/0.2)^2,4)', cfg['center']['formula'])
        self.assertTrue(cfg['dt_multiplied_once'])
        self.assertFalse(cfg['total_reward_clamped'])
        self.assertFalse(cfg['alive_bonus'])
        self.assertEqual(cfg['events']['success'], 10.)
        self.assertEqual(cfg['events']['failure'], -10.)
        self.assertEqual(cfg['footfall']['amount'], .50)
        self.assertEqual(cfg['swing']['amount_per_target'], .15)
        self.assertEqual(cfg['swing']['target_window_m'], .60)
        self.assertTrue(cfg['swing']['consume_without_gate'])
        self.assertEqual(cfg['stall']['per_step_amount'], -.005)
        self.assertEqual((cfg['stall']['history_samples'], cfg['stall']['intervals'],
                          cfg['stall']['consecutive_samples'],
                          cfg['stall']['first_penalty_sample']), (51, 50, 50, 99))
        self.assertEqual(cfg['pose']['weight'], .05)
        self.assertEqual(cfg['pose']['sigma_active'], [.15, .35, .5] * 4)
        self.assertEqual(cfg['orientation']['weight'], -1.)
        self.assertAlmostEqual(cfg['orientation']['lower_deadband'], math.radians(5.))
        self.assertAlmostEqual(cfg['orientation']['upper_deadband'], math.radians(55.))
        self.assertEqual(cfg['nonfoot']['force_threshold'], 10.)
        self.assertEqual(cfg['edge']['inset'], .042)
        self.assertFalse(cfg['slip']['averaged_over_feet'])
        self.assertEqual(cfg['stand_still']['weight'], -1.)
        self.assertEqual(cfg['removed']['body_x_progress']['new_contribution'], 0.)
        self.assertEqual(len(cfg['continuous_weights']), 15)

    def test_reward_sha_is_canonical_and_immutable(self):
        digest = spec.reward_config_sha256()
        self.assertRegex(digest, r'^[0-9a-f]{64}$')
        self.assertEqual(digest, hashlib.sha256(
            spec.canonical_json(spec.REWARD_CONFIG).encode('utf-8')).hexdigest())
        # r2 identity differs from r1 exactly because of the version field (SDS 3.9.2).
        self.assertNotEqual(digest, r1_spec.reward_config_sha256())
        stripped = spec.canonical_json(dict(spec.REWARD_CONFIG, version='0.5-r1'))
        self.assertEqual(hashlib.sha256(stripped.encode('utf-8')).hexdigest(),
                         r1_spec.reward_config_sha256())
        copy = spec.reward_config()
        copy['version'] = 'mutated'
        copy['continuous_weights']['pose'] = 9.
        self.assertEqual(spec.REWARD_CONFIG['version'], '0.5-r2')
        self.assertEqual(spec.REWARD_CONFIG['continuous_weights']['pose'], .05)
        self.assertEqual(digest, spec.reward_config_sha256())
        recipe = spec.recipe_config()
        self.assertEqual(recipe['recipe'], '0.5-r2')
        self.assertEqual(recipe['reward_config_sha256'], digest)
        self.assertEqual((recipe['std_min'], recipe['std_max'], recipe['initial_std']),
                         (.10, .30, .15))
        self.assertEqual(json.loads(spec.canonical_json(recipe)), recipe)
        with self.assertRaises(ValueError):
            spec.recipe_config('0.5-r1')

    def test_ppo_contract_numbers_match_r1(self):
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
        self.assertEqual(ppo, r1_spec.PPO_CONTRACT)
        self.assertEqual((spec.PPO_INITIAL_LR, spec.PPO_GAMMA, spec.INITIAL_STD,
                          spec.STD_MIN, spec.STD_MAX), (5e-5, .995, .15, .10, .30))

    def test_evaluation_thresholds_match_r1(self):
        self.assertEqual((spec.MEAN_RELATIVE_ERROR_MAX, spec.SMOOTH_MAE_MAX,
                          spec.LONGEST_DEVIATION_MAX, spec.DEVIATION_THRESHOLD,
                          spec.SPEED_EPSILON), (.20, .30, 50, .50, 1e-6))
        self.assertEqual((spec.EPISODES_PER_SPEED, spec.CASE_PASS_REQUIRED,
                          spec.CONSECUTIVE_GATES), (20, 18, 2))
        self.assertEqual((spec.EPISODES_PER_SPEED_FINAL, spec.CASE_PASS_REQUIRED_FINAL),
                         (100, 95))
        self.assertEqual((spec.VALIDATION_SEED_OFFSET, spec.PERTURBED_SEED_OFFSET,
                          spec.DIAGNOSTIC_SEED_OFFSET, spec.FINAL_TEST_SEED),
                         (200300, 300300, 500300, 400342))
        self.assertEqual((spec.F0_OBSERVE_STEPS, spec.F0_WINDOW_STEPS,
                          spec.F0_EPISODE_STEPS), (500, (101, 500), 1500))
        self.assertAlmostEqual(spec.F0_HEADING_MAX, math.radians(10.))
        self.assertEqual(spec.F0_LATERAL_MAX, .20)
        evaluation = stable_contract()['evaluation']
        self.assertEqual((evaluation['mean_relative_error_max'],
                          evaluation['smooth_mae_max'],
                          evaluation['longest_deviation_max'],
                          evaluation['deviation_threshold'],
                          evaluation['epsilon']), (.20, .30, 50, .50, 1e-6))
        self.assertEqual(evaluation['validation_seed'], 200342)
        self.assertEqual(evaluation['perturbed_seed'], 300342)
        self.assertEqual(evaluation['diagnostic_seed'], 500342)
        self.assertEqual(evaluation['final_test_seed'], 400342)
        self.assertTrue(evaluation['speed_not_a_gate_for_stairs'])
        self.assertEqual(evaluation['raw_route_evidence'],
                         ['route_error', 'local_y', 'yaw'])


class SourceClosureTests(unittest.TestCase):
    def test_contract_identity_reward_sha_and_settings(self):
        contract = stable_contract()
        self.assertEqual((contract['code_version'], contract['recipe_version'],
                          contract['sds_revision'], contract['checkpoint_schema'],
                          contract['architecture'], contract['task'],
                          contract['protocol_id'], contract['flat_protocol_id']),
                         ('0.5.1', '0.5-r2', '1.0', 7, 'v05-route-state-v1',
                          'Unitree-A2-IdealHeightMap-Stairs-V05R2',
                          'v05-r2-stair-climb-first-v1', 'v05-r2-flat-world-route-v1'))
        self.assertEqual(contract['identity'], spec.identity_block())
        self.assertEqual(set(contract['identity']),
                         {'task', 'recipe', 'code_version', 'protocol', 'sds_revision'})
        self.assertEqual(contract['reward_config'], spec.REWARD_CONFIG)
        self.assertEqual(contract['reward_config_sha256'], spec.reward_config_sha256())
        self.assertEqual(contract['source_eth_path'], spec.INIT_PATH)
        self.assertEqual(contract['source_eth_sha256'], spec.INIT_SHA256)
        self.assertEqual(contract['initialization'],
                         'explicit_eth_actor_mlp_and_frozen_normalizer')
        self.assertEqual(contract['critic_initialization'], 'fresh_seeded_211')
        self.assertEqual(contract['route_fusion_initialization'], 'zeros_after_construction')
        self.assertEqual(contract['stages'], [asdict(stage) for stage in spec.STAGES])
        self.assertEqual(contract['training_num_envs'], 512)
        self.assertEqual(contract['evaluation_num_envs'], 8)
        self.assertEqual(contract['slot_classes'], ['flat', 'previous', 'current'])
        self.assertEqual(contract['allocation'],
                         'integer_largest_remainder_flat_previous_current')
        self.assertEqual(contract['slot_counts_512']['P1'],
                         {'flat': 102, 'previous': 0, 'current': 410})
        self.assertEqual(contract['slot_layout_8']['P1'], ['flat', 'flat'] + ['current'] * 6)
        self.assertEqual(contract['command']['range_per_episode'], [.2, .6])
        self.assertEqual((contract['command']['vy'], contract['command']['wz']), (0., 0.))
        self.assertEqual(contract['timeout']['flat_steps'], 1500)
        self.assertFalse(contract['timeout']['stagnation_termination'])
        self.assertEqual(contract['ppo'], spec.PPO_CONTRACT)
        self.assertEqual(contract['observation'], spec.OBSERVATION)
        self.assertEqual(contract['route_state'], spec.ROUTE_STATE)
        self.assertEqual(contract['resume_mode'], 'new_episode')
        self.assertEqual(contract['recovery']['unknown_tail'], 'recovery_blocked')
        self.assertEqual(contract['architecture_detail']['critic'].split()[0], '211-512-256-128-1')
        self.assertEqual(contract['monitoring']['group'], 'v05_r2')
        self.assertEqual(contract['monitoring']['heartbeat_kind'], 'v05_r2_heartbeat')
        self.assertEqual(contract['monitoring']['panel_count'], 8)
        self.assertEqual(contract['monitoring']['unconfirmed_seconds'], 120)
        self.assertEqual(contract['monitoring']['route_namespace'], 'route/*')
        self.assertEqual(contract['monitoring']['route_panel_group'], '运动/接触')
        self.assertEqual(contract['monitoring']['extra_panel_groups'], 0)
        self.assertEqual(contract['smoke_container'], 'v05-r2-smoke-state-v1')
        self.assertFalse(contract['smoke_formal_resumable'])
        spawn = contract['spawn']['flat']
        self.assertEqual((spawn['x_range'], spawn['y_range']), ([-1.1, -.9], [-.05, .05]))
        self.assertEqual(spawn['yaw_degrees'], [-5., 5.])
        self.assertEqual(spawn['center'], [-1., 0., 0.])
        self.assertEqual(spawn['sample_order'], 'x,y,yaw,command')
        self.assertTrue(spawn['initial_cases_precedence'])
        self.assertEqual(spawn['initial_pose_source'], 'r2_flat_uniform')
        self.assertIn('x=0', spawn['b1'])
        self.assertIn('not an r1 behavior', spawn['b2'])
        self.assertEqual(json.loads(spec.canonical_json(contract)), contract)
        self.assertEqual(spec.contract_sha256(), hashlib.sha256(
            spec.canonical_json(contract).encode('utf-8')).hexdigest())
        for kwargs in ({'recipe': 'A'}, {'recipe': '0.5-r1'}, {'mode': 'comparison'}):
            with self.assertRaises(ValueError):
                spec.contract(**kwargs)
        for seed in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError, msg=repr(seed)):
                spec.contract(seed=seed)
        with self.assertRaises(ValueError):
            spec.contract(device='')

    def test_seed_and_mode_are_the_only_contract_variants(self):
        allowed = {'seed', 'mode', 'evaluation', 'total_budget', 'update_cap',
                   'stage_budgets', 'effective_cap_source'}
        for _ in range(3):
            base = stable_contract()
            other = stable_contract(mode='pilot', seed=43)
            differing = {key for key in base if base[key] != other[key]}
            if differing <= allowed:
                break
        else:  # pragma: no cover - only a concurrent writer can reach this
            self.fail(f'contract variants differed beyond seed/mode/budget: {differing}')
        self.assertLessEqual(differing, allowed)
        # Seed, mode and the effective budget record must all change here; the
        # ``source`` label happens to read the same for pilot and formal.
        self.assertLessEqual({'seed', 'mode', 'total_budget', 'update_cap',
                              'stage_budgets', 'evaluation'}, differing)
        self.assertEqual(other['evaluation']['validation_seed'], 200343)
        self.assertEqual(other['evaluation']['perturbed_seed'], 300343)
        self.assertEqual(other['evaluation']['diagnostic_seed'], 500343)
        self.assertEqual(other['evaluation']['final_test_seed'], 400342)
        for key in base:
            if key not in allowed:
                self.assertEqual(base[key], other[key], msg=key)

    def test_closure_binds_every_existing_product_file_and_scripts(self):
        contract = stable_contract()
        bound = contract['behavior_sha256']
        self.assertTrue(V05R2_FILES, 'no v05_r2 module exists')
        self.assertLessEqual(V05R2_FILES, set(bound))
        for required in ('scripts/a2_v05_r2_bootstrap.py',
                         'scripts/check_a2_ideal_stairs_v05_r2_migration.py',
                         'src/tasks/velocity/v05_r2/__init__.py',
                         'src/tasks/velocity/v05_r2/spec.py',
                         'src/tasks/velocity/v05_r2/model.py'):
            self.assertTrue((spec.ROOT / required).is_file(), required)
            self.assertIn(required, bound)
        self.assertNotIn('src/tasks/__init__.py', bound)
        for relative, digest in bound.items():
            self.assertEqual(spec.sha256(spec.ROOT / relative), digest, msg=relative)
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
        # The r1 files are read-only references, never part of the r2 closure.
        self.assertFalse(any(relative.startswith('src/tasks/velocity/v05/')
                             for relative in bound))

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
                'src/tasks/velocity/v05_r2/__init__.py': '',
                'src/tasks/velocity/v05_r2/entry.py': 'from .helper import value\n',
                'src/tasks/velocity/v05_r2/helper.py': ('from .leaf import value as leaf\n'
                                                        'value = 1\n'),
                'src/tasks/velocity/v05_r2/leaf.py': 'value = 1\n',
                'src/tasks/unrelated/__init__.py': '',
                'src/tasks/unrelated/env.py': 'value = 1\n',
                'src/tasks/new_task/__init__.py': '',
                'src/tasks/new_task/env.py': 'value = 1\n',
                'src/tasks/velocity/config/a2/env_cfgs.py': 'value = 1\n',
                'scripts/entry.py': 'from src.tasks.velocity.v05_r2.entry import value\n',
            }
            for relative, contents in files.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents)
            first = spec.source_hashes(['src/tasks/velocity/v05_r2/entry.py'], root=root)
            self.assertEqual(set(first), {
                'src/__init__.py', 'src/tasks/velocity/__init__.py',
                'src/tasks/velocity/v05_r2/__init__.py',
                'src/tasks/velocity/v05_r2/entry.py',
                'src/tasks/velocity/v05_r2/helper.py',
                'src/tasks/velocity/v05_r2/leaf.py'})
            self.assertNotIn('src/tasks/__init__.py', first)
            self.assertNotIn('src/tasks/unrelated/env.py', first)
            self.assertNotIn('src/tasks/new_task/env.py', first)
            for relative, digest in first.items():
                self.assertEqual(digest, hashlib.sha256((root / relative).read_bytes()).hexdigest())
            (root / 'src/tasks/velocity/v05_r2/leaf.py').write_text('value = 2\n')
            second = spec.source_hashes(['src/tasks/velocity/v05_r2/entry.py'], root=root)
            self.assertNotEqual(first['src/tasks/velocity/v05_r2/leaf.py'],
                                second['src/tasks/velocity/v05_r2/leaf.py'])
            (root / 'src/tasks/unrelated/env.py').write_text('value = 2\n')
            (root / 'src/tasks/new_task/extra.py').write_text('value = 2\n')
            (root / 'src/tasks/another_robot').mkdir()
            (root / 'src/tasks/another_robot/__init__.py').write_text('')
            self.assertEqual(second,
                             spec.source_hashes(['src/tasks/velocity/v05_r2/entry.py'], root=root))
            with self.assertRaises(FileNotFoundError):
                spec.source_hashes(['src/tasks/velocity/v05_r2/missing.py'], root=root)
            with self.assertRaises(ValueError):
                spec.source_hashes(['../escape.py'], root=root)
            script = spec.source_hashes(['scripts/entry.py'], root=root)
            self.assertIn('src/tasks/velocity/v05_r2/entry.py', script)
            self.assertNotIn('src/tasks/__init__.py', script)
            default = spec.source_hashes(root=root)
            self.assertLessEqual(set(spec.product_entries(root)), set(default))
            self.assertEqual(set(default) - set(spec.product_entries(root)),
                             {'src/__init__.py', 'src/tasks/velocity/__init__.py'})
            self.assertNotIn('src/tasks/__init__.py', default)


if __name__ == '__main__':
    unittest.main()
