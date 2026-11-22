"""V0.4 identities, reward configuration, stage/budget and explicit source closure.

The product source contract is an explicit closure: it binds the bootstrap and
v0.4 entries while refusing to grow with unrelated task directories and while
never binding the legacy ``src/tasks/__init__.py`` scanner.  Synthetic temporary
projects exercise that closure without importing any task code.
"""
import hashlib
import importlib
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


class SpecTests(unittest.TestCase):
    def spec(self):
        try:
            return importlib.import_module('src.tasks.velocity.v04.spec')
        except ModuleNotFoundError:
            self.fail('V04 specification implementation is missing')

    def contract_or_skip(self):
        s = self.spec()
        missing = [p for p in s.SOURCE_ENTRIES if not (s.ROOT / p).is_file()]
        if missing:
            self.skipTest('v04 contract source closure is incomplete because other v04 '
                          'modules are still being written: ' + ', '.join(missing))
        return s.contract()

    @staticmethod
    def write_project(root, files):
        for relative, contents in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)

    def test_identity_and_frozen_reward_configuration(self):
        s = self.spec()
        self.assertEqual((s.CODE_VERSION, s.RECIPE_VERSION, s.SDS_REVISION, s.SCHEMA_VERSION,
                          s.TASK, s.PROTOCOL_ID),
                         ('0.4.0', '0.4-r1', '1.1', 5, 'Unitree-A2-IdealHeightMap-Stairs-V04',
                          'v04-stair-body-speed-v1'))
        self.assertEqual((s.CONTROL_DT, s.PHYSICS_DT, s.SPEEDS, s.SLOT_CLASSES),
                         (.02, .005, (.2, .4, .6), ('previous_stairs', 'current_stairs')))
        self.assertEqual((s.FOOT_RADIUS, s.SWING_TOP_TOLERANCE, s.SUPPORT_FORCE_GT,
                          s.SAFE_INSET, s.MAX_SWING_STEPS), (.032, .005, 5., .042, 40))
        cfg = s.reward_config()
        self.assertEqual((cfg['version'], cfg['dt']), ('0.4-r1', .02))
        self.assertIn('r_speed + dt*sum(other_continuous_weighted_rates)',
                      cfg['composition'])
        speed = cfg['speed']
        self.assertEqual((speed['frame'], speed['exp_scale'], speed['exp_coefficient'],
                          speed['over_absolute_coefficient'], speed['over_deadband_fraction'],
                          speed['backward_absolute_coefficient'], speed['lateral_tanh_coefficient']),
                         ('body', 4., .5, 10., .05, 2., .25))
        self.assertAlmostEqual(speed['exp_offset_subtract'], .5 * math.exp(-4.), places=15)
        self.assertAlmostEqual(speed['exp_zero_error_value'], .5 * (1. - math.exp(-4.)), places=15)
        self.assertTrue(speed['positive_part_only_gated'])
        self.assertTrue(speed['includes_dt'])
        self.assertEqual(speed['disabled_zero_c'], 'zero_with_explicit_enabled_false')
        self.assertEqual(speed['active_nonpositive_c'], 'error')
        self.assertIn('|roll|<=30deg', speed['gate'])
        self.assertIn('-10deg<=pitch<=55deg', speed['gate'])
        orientation = cfg['orientation']
        self.assertEqual((orientation['weight'], orientation['roll_coefficient'],
                          orientation['lower_coefficient']), (-1., 1., 4.))
        self.assertAlmostEqual(orientation['lower_deadband'], math.radians(5.), places=15)
        self.assertAlmostEqual(orientation['upper_deadband'], math.radians(55.), places=15)
        nonfoot = cfg['nonfoot']
        self.assertEqual((nonfoot['weight'], nonfoot['force_threshold'],
                          nonfoot['aggregation']), (-1., 10., 'any_monitored_link'))
        events = cfg['events']
        self.assertEqual((events['success'], events['failure']), (10., -4.))
        self.assertTrue(events['failure_precedence'])
        self.assertTrue(events['one_time'])
        self.assertEqual(cfg['footfall']['amount'], .25)
        swing = cfg['swing']
        self.assertEqual((swing['amount_per_target'], swing['max_fraction'],
                          swing['target_window_m'], swing['support_force_gt'],
                          swing['safe_inset'], swing['top_tolerance'], swing['debounce'],
                          swing['body_height_scale'], swing['target_height_margin'],
                          swing['max_steps'], swing['command_active_min']),
                         (.10, 1., .30, 5., .042, .005, 2, .04, .04, 40, .1))
        self.assertTrue(swing['no_dt_multiplier'])
        self.assertTrue(swing['consume_without_gate'])
        self.assertTrue(swing['target_lock_after_window_entry'])
        self.assertEqual(swing['paid_fraction_scope'], 'episode_foot_target')
        self.assertEqual(cfg['removed']['body_x_progress']['new_contribution'], 0.)
        self.assertEqual(cfg['removed']['old_linear_velocity'], 'diagnostic_only')
        self.assertEqual(cfg['unchanged']['edge'], -1.)  # frozen edge term, not removed
        # Reward config hashing is canonical JSON and not derived from a caller copy.
        digest = s.reward_config_sha256()
        self.assertRegex(digest, r'^[0-9a-f]{64}$')
        self.assertEqual(digest, hashlib.sha256(
            s.canonical_json(s.REWARD_CONFIG).encode('utf-8')).hexdigest())
        copy = s.reward_config()
        copy['version'] = 'mutated'
        self.assertEqual(s.REWARD_CONFIG['version'], '0.4-r1')
        self.assertEqual(digest, s.reward_config_sha256())
        with self.assertRaises(ValueError):
            s.recipe_config('0.3')

    def test_stage_table_and_23000_budget(self):
        s = self.spec()
        self.assertEqual([(x.id, x.tile_id, x.N, x.h, x.budget) for x in s.STAGES],
                         [('S1', 5, 4, .08, 3000), ('S2', 6, 8, .12, 4000),
                          ('S3', 7, 12, .16, 4000), ('S4', 8, 20, .20, 5000),
                          ('S5', 9, 40, .20, 7000)])
        self.assertEqual(sum(x.budget for x in s.STAGES), 23000)
        self.assertEqual(set(s.STAGE_BY_ID), {'S1', 'S2', 'S3', 'S4', 'S5'})
        self.assertEqual(set(s.STAGE_BY_TILE), {5, 6, 7, 8, 9})
        self.assertNotIn('flat', s.STAGE_BY_ID)
        self.assertNotIn('R0', s.STAGE_BY_ID)

    def test_512_formal_slot_allocation_and_remainders(self):
        s = self.spec()
        for num_envs, expected in ((1024, [205, 819]), (512, [102, 410]), (258, [52, 206]),
                                   (1, [0, 1]), (2, [0, 2]), (3, [1, 2]), (4, [1, 3])):
            for stage, previous, current_tile in (('S2', 'S1', 6), ('S5', 'S4', 9)):
                tiles, classes, counts = s.allocate_stage_slots(num_envs, stage, previous)
                self.assertEqual(counts.tolist(), expected, msg=(num_envs, stage))
                self.assertEqual(classes.tolist(), [0] * expected[0] + [1] * expected[1])
                self.assertEqual(tiles.tolist(),
                                 [current_tile - 1] * expected[0] + [current_tile] * expected[1])
                self.assertEqual(tiles.dtype.name, 'int64')
            tiles, classes, counts = s.allocate_stage_slots(num_envs, 'S1')
            self.assertEqual(counts.tolist(), [0, num_envs])
            self.assertTrue((tiles == 5).all())
            self.assertTrue((classes == 1).all())
        self.assertEqual(s.allocate_stage_slots(512, 'S5', 'S4')[2].tolist(), [102, 410])
        for args in ((0, 'S1'), (-1, 'S1'), (1.5, 'S1'), (True, 'S1'), (10, 'flat'),
                     (10, 'R0'), (10, 'S3', 'S1'), (10, 'S1', 'S5')):
            with self.assertRaises(ValueError, msg=str(args)):
                s.allocate_stage_slots(*args)

    def test_contract_identity_reward_sha_and_512_formal_settings(self):
        s = self.spec()
        c = self.contract_or_skip()
        self.assertEqual((c['code_version'], c['recipe_version'], c['checkpoint_schema'],
                          c['sds_revision'], c['task'], c['protocol_id']),
                         ('0.4.0', '0.4-r1', 5, '1.1',
                          'Unitree-A2-IdealHeightMap-Stairs-V04', 'v04-stair-body-speed-v1'))
        self.assertEqual(c['reward_config'], s.REWARD_CONFIG)
        self.assertEqual(c['reward_config_sha256'], s.reward_config_sha256())
        self.assertEqual(c['source_c_sha256'], s.INIT_SHA256)
        self.assertEqual(s.sha256(s.INIT_PATH), s.INIT_SHA256)
        self.assertEqual((c['source_code_version'], c['source_checkpoint_schema'],
                          c['source_completed_updates'], c['source_optimizer_steps']),
                         ('0.2.1', 3, 3000, 60000))
        self.assertEqual(c['stages'], [asdict(stage) for stage in s.STAGES])
        self.assertEqual(c['total_budget'], 23000)
        self.assertEqual((c['rollout'], c['epochs'], c['minibatches'], c['save_interval'],
                          c['formal_eval_interval']), (48, 5, 4, 100, 200))
        self.assertEqual((c['seed'], c['device'], c['vx_range'], c['vy'], c['wz']),
                         (42, 'cuda:0', [.2, .6], 0., 0.))
        self.assertEqual((c['physics_dt'], c['control_dt'], c['action_scale']), (.005, .02, .25))
        self.assertEqual(c['observation'],
                         {'proprio': [45], 'history': [5, 45], 'heightmap': [3, 48, 32],
                          'privileged': [35]})
        # Formal 512-env settings and the eight-env evaluation contract.
        self.assertEqual(c['resources']['candidate_num_envs'], [512])
        self.assertEqual(c['resources']['evaluation_num_envs'], [8])
        self.assertEqual(c['slot_classes'], ['previous_stairs', 'current_stairs'])
        self.assertEqual((c['mix_S1'], c['mix_later']), ([0., 1.], [.2, .8]))
        self.assertEqual(c['allocation'], 'integer_largest_remainder_previous_first')
        self.assertEqual(c['flat_slots'], 0)
        self.assertEqual(s.allocate_stage_slots(512, 'S5', 'S4')[2].tolist(), [102, 410])
        self.assertEqual(c['evaluation']['speeds'], [.2, .4, .6])
        self.assertEqual(c['evaluation']['seeds'],
                         {'validation': 200342, 'perturbed': 300342, 'test': 400342})
        self.assertEqual(c['evaluation']['episodes_per_speed'], 20)
        self.assertEqual(c['evaluation']['joint_successes_required'], 18)
        self.assertEqual(c['evaluation']['final_episodes_per_speed'], 100)
        self.assertEqual(c['evaluation']['final_joint_successes_required'], 95)
        self.assertEqual(c['geometry']['step_depth'], .2)
        self.assertEqual(c['geometry']['stair_width'], .8)
        self.assertEqual(c['monitoring']['group'], 'v04')
        self.assertEqual(c['monitoring']['heartbeat_kind'], 'v04_heartbeat')
        self.assertEqual(c['monitoring']['panel_count'], 8)
        self.assertTrue(c['monitoring']['single_writer'])
        # Explicit closure: bootstrap/v0.4 entries bound, legacy scanner excluded.
        bound = c['behavior_sha256']
        self.assertEqual(set(s.SOURCE_ENTRIES) - set(bound), set())
        self.assertIn('scripts/a2_v04_bootstrap.py', bound)
        self.assertIn('src/tasks/velocity/v04/spec.py', bound)
        self.assertIn('src/tasks/velocity/v04/rewards.py', bound)
        self.assertIn('src/tasks/velocity/v04/env.py', bound)
        self.assertNotIn('src/tasks/__init__.py', bound)
        for relative, digest in bound.items():
            self.assertEqual(s.sha256(s.ROOT / relative), digest, msg=relative)
        # The contract is canonical JSON, and the fixed recipe/mode are enforced.
        self.assertEqual(json.loads(s.canonical_json(c)), c)
        self.assertEqual(json.loads(json.dumps(c)), c)
        for kwargs in ({'recipe': 'A'}, {'mode': 'comparison'}, {'recipe': '0.4'}):
            with self.assertRaises(ValueError):
                s.contract(**kwargs)

    def test_source_hashes_is_an_explicit_closure_in_a_temp_project(self):
        s = self.spec()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root, {
                'src/__init__.py': '',
                'src/tasks/__init__.py': 'raise RuntimeError("legacy scanner must not run")\n',
                'src/tasks/velocity/__init__.py': '',
                'src/tasks/velocity/v04/__init__.py': '',
                'src/tasks/velocity/v04/entry.py': 'from .helper import value\n',
                'src/tasks/velocity/v04/helper.py': 'from .leaf import value as leaf\n'
                                                   'value = 1\n',
                'src/tasks/velocity/v04/leaf.py': 'value = 1\n',
                'src/tasks/unrelated/__init__.py': '',
                'src/tasks/unrelated/env.py': 'value = 1\n',
                'src/tasks/new_task/__init__.py': '',
                'src/tasks/new_task/env.py': 'value = 1\n',
                'src/tasks/velocity/config/a2/env_cfgs.py': 'value = 1\n',
                'scripts/entry.py': 'from src.tasks.velocity.v04.entry import value\n',
            })
            first = s.source_hashes(['src/tasks/velocity/v04/entry.py'], root=root)
            self.assertEqual(set(first), {
                'src/__init__.py', 'src/tasks/velocity/__init__.py',
                'src/tasks/velocity/v04/__init__.py', 'src/tasks/velocity/v04/entry.py',
                'src/tasks/velocity/v04/helper.py', 'src/tasks/velocity/v04/leaf.py'})
            # The legacy root scanner and unrelated new tasks never enter.
            self.assertNotIn('src/tasks/__init__.py', first)
            self.assertNotIn('src/tasks/unrelated/env.py', first)
            self.assertNotIn('src/tasks/new_task/env.py', first)
            self.assertNotIn('src/tasks/velocity/config/a2/env_cfgs.py', first)
            # Every digest is the real SHA of the bound file.
            for relative, digest in first.items():
                self.assertEqual(digest, hashlib.sha256((root / relative).read_bytes()).hexdigest())
            # A bound transitive dependency changes the closure digest.
            (root / 'src/tasks/velocity/v04/leaf.py').write_text('value = 2\n')
            second = s.source_hashes(['src/tasks/velocity/v04/entry.py'], root=root)
            self.assertNotEqual(first['src/tasks/velocity/v04/leaf.py'],
                                second['src/tasks/velocity/v04/leaf.py'])
            # Unrelated task edits and additions leave the closure untouched.
            (root / 'src/tasks/unrelated/env.py').write_text('value = 2\n')
            (root / 'src/tasks/new_task/extra.py').write_text('value = 2\n')
            (root / 'src/tasks/another_robot').mkdir()
            (root / 'src/tasks/another_robot/__init__.py').write_text('')
            third = s.source_hashes(['src/tasks/velocity/v04/entry.py'], root=root)
            self.assertEqual(second, third)
            # Missing required files fail closed; escaping paths are rejected.
            with self.assertRaises(FileNotFoundError):
                s.source_hashes(['src/tasks/velocity/v04/missing.py'], root=root)
            with self.assertRaises(ValueError):
                s.source_hashes(['../escape.py'], root=root)
            # A static absolute import from a script resolves the same closure.
            script = s.source_hashes(['scripts/entry.py'], root=root)
            self.assertIn('src/tasks/velocity/v04/entry.py', script)
            self.assertNotIn('src/tasks/__init__.py', script)

    def test_real_repository_explicit_closure_binds_bootstrap_and_v04(self):
        s = self.spec()
        paths = ['scripts/a2_v04_bootstrap.py', 'src/tasks/velocity/v04/spec.py',
                 'src/tasks/velocity/v04/rewards.py']
        hashes = s.source_hashes(paths)
        for relative in paths:
            self.assertIn(relative, hashes)
            self.assertEqual(hashes[relative], s.sha256(s.ROOT / relative))
        self.assertIn('src/tasks/velocity/v04/__init__.py', hashes)
        self.assertIn('src/tasks/velocity/__init__.py', hashes)
        self.assertNotIn('src/tasks/__init__.py', hashes)
        # The frozen entry list itself is explicit: bootstrap and every v04
        # product module are present, the legacy root initializer is absent and
        # there is no dynamic-dependency escape hatch.
        entries = set(s.SOURCE_ENTRIES)
        self.assertIn('scripts/a2_v04_bootstrap.py', entries)
        for module in ('__init__', 'spec', 'rewards', 'env', 'model', 'runner', 'recovery',
                       'speed', 'evaluation', 'connection', 'telemetry', 'orchestration'):
            self.assertIn(f'src/tasks/velocity/v04/{module}.py', entries)
        self.assertNotIn('src/tasks/__init__.py', entries)
        self.assertEqual(s.DYNAMIC_DEPENDENCIES, ())
        # The default closure is the frozen entry list; only exercise it when
        # every entry exists so this check does not duplicate the contract skip.
        if all((s.ROOT / p).is_file() for p in s.SOURCE_ENTRIES):
            default = s.source_hashes()
            self.assertIn('scripts/a2_v04_bootstrap.py', default)
            self.assertNotIn('src/tasks/__init__.py', default)


if __name__ == '__main__':
    unittest.main()
