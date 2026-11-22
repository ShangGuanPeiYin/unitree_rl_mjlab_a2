"""CPU-only V0.5-r2 model contract: route bypass wiring, 211-dimensional Critic,
dual std projection, schema-7 validation and ETH initialization.

These tests do not simulate, train or touch a GPU.  They pin the r2 identity, the
``[0.10, 0.30]`` std range in both places SDS 2.5 names, the fresh
Critic/statistics/Adam state, the exact ``route_fusion`` wiring inside the first
activation, the route tail of the 211-input Critic and the explicit schema-7
validators the runner/recovery call sites use (including the refusal of r1,
schema<=6 and smoke containers - never via ``strict=False``).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_r2_bootstrap  # noqa: E402

a2_v05_r2_bootstrap.activate(ROOT)

import copy  # noqa: E402
import math  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import (  # noqa: E402
    IdealHeightmapActorCritic)
from src.tasks.velocity.v05_r2 import model as target  # noqa: E402
from src.tasks.velocity.v05_r2 import spec  # noqa: E402

V05R2 = target.V05R2ActorCritic


class PublicAPITests(unittest.TestCase):
    def test_model_api_and_identity(self):
        self.assertTrue(callable(V05R2.from_eth))
        for name in ('validate_model_state', 'validate_model_tensors', 'validate_contract',
                     'validate_initialization', 'validate_schema7_keys', 'validate_identity',
                     'load_model_state', 'mean_equivalence', 'map_history_invariance',
                     'route_zero_invariance', 'route_fusion_zero_audit',
                     'route_bypass_trainability', 'critic_211_audit',
                     'critic_fresh_reference', 'tensor_audit', 'synthetic_observations',
                     'route_error_from_pose', 'audit_initialization'):
            self.assertTrue(callable(getattr(target, name)), name)
        self.assertEqual(V05R2.architecture_version, 'v05-route-state-v1')
        self.assertEqual(V05R2.architecture_version, spec.ARCHITECTURE)
        self.assertEqual((V05R2.std_min, V05R2.std_max), (.10, .30))
        self.assertEqual(len(target.ETH_ACTOR_KEYS), 13)
        self.assertEqual(len(target.NORMALIZER_SOURCE_KEYS), 4)
        # The r1 class name stays importable for the sibling r2 modules.
        self.assertIs(target.V05ActorCritic, V05R2)
        self.assertTrue(issubclass(V05R2, IdealHeightmapActorCritic))
        self.assertEqual(target.SCHEMA7_TOP_LEVEL_KEYS,
                         ('schema', 'identity', 'contract_sha256', 'model', 'optimizer',
                          'normalizer', 'learning', 'rng', 'curriculum', 'ledger',
                          'evaluation', 'source', 'recipe', 'resume_mode'))


class NetworkStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.observations = target.synthetic_observations(4, seed=7)

    def test_route_bypass_layer_shape_and_zero_init(self):
        fusion = self.model.actor.route_fusion
        self.assertIsInstance(fusion, nn.Linear)
        self.assertEqual((fusion.in_features, fusion.out_features), (3, 512))
        self.assertFalse(fusion.weight.any())
        self.assertFalse(fusion.bias.any())
        self.assertEqual(tuple(self.model.state_dict()['actor.route_fusion.weight'].shape),
                         (512, 3))
        self.assertEqual(tuple(self.model.state_dict()['actor.route_fusion.bias'].shape),
                         (512,))
        # Exactly two new tensors relative to r1's shape (51 -> 53 state keys).
        self.assertEqual(len(self.model.state_dict()), 53)
        self.assertEqual(set(spec.ROUTE_FUSION_KEYS) - set(self.model.state_dict()), set())

    def test_forward_is_exactly_the_documented_three_way_sum(self):
        obs = self.observations
        actor = self.model.actor
        with torch.no_grad():
            features = actor.features(obs)
            expected = actor.mlp[2:](F.elu(actor.mlp[0](actor.normalizer(obs['proprio']))
                                          + actor.fusion(features)
                                          + actor.route_fusion(obs['route_error'])))
            actual = self.model(obs)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(tuple(actual.shape), (4, 12))
        # The bypass really feeds the activation: non-zero route weights change it.
        with torch.no_grad():
            actor.route_fusion.weight.normal_(0., .01)
            routed = self.model(obs)
            self.assertFalse(torch.equal(routed, actual))
            other = self.model(dict(obs, route_error=-obs['route_error']))
            self.assertFalse(torch.equal(routed, other))
            actor.route_fusion.weight.zero_()
            self.assertTrue(torch.equal(self.model(obs), actual))

    def test_actor_mean_ignores_route_map_history_at_zero_fusion(self):
        obs = self.observations
        with torch.no_grad():
            base = self.model(obs)
            changed = self.model(dict(obs, route_error=torch.zeros_like(obs['route_error']),
                                      heightmap=torch.zeros_like(obs['heightmap']),
                                      history=torch.zeros_like(obs['history'])))
        self.assertEqual(float((base - changed).abs().max()), 0.)

    def test_critic_is_211_inputs_with_route_in_the_tail(self):
        critic = self.model.critic
        self.assertEqual(critic[0].in_features, 211)
        self.assertEqual(critic[0].out_features, 512)
        self.assertEqual(tuple(critic[0].weight.shape), (512, 211))
        self.assertEqual(int(self.model.critic_normalizer.mean.numel()), 80)
        self.assertEqual(int(self.model.critic_normalizer.mean.numel()),
                         spec.CRITIC_NORMALIZER_DIMS)
        captured = {}

        def pre_hook(_module, inputs):
            captured['input'] = inputs[0]

        handle = critic.register_forward_pre_hook(pre_hook)
        try:
            with torch.no_grad():
                self.model.evaluate(self.observations)
        finally:
            handle.remove()
        self.assertEqual(tuple(captured['input'].shape), (4, 211))
        self.assertTrue(torch.equal(captured['input'][:, 208:211],
                                    self.observations['route_error']))
        self.assertEqual((critic[2].in_features, critic[2].out_features), (512, 256))
        self.assertEqual((critic[4].in_features, critic[4].out_features), (256, 128))
        self.assertEqual((critic[6].in_features, critic[6].out_features), (128, 1))
        # The map/history encoders keep the r1 shape.
        self.assertEqual(self.model.critic_map_encoder[0].in_channels, 3)
        self.assertEqual(self.model.critic_history_encoder[1][0].in_features, 225)

    def test_critic_value_depends_on_route_map_and_history(self):
        obs = self.observations
        generator = torch.Generator(device='cpu').manual_seed(9)
        with torch.no_grad():
            base = self.model.evaluate(obs)
            route = self.model.evaluate(dict(
                obs, route_error=torch.rand(obs['route_error'].shape, generator=generator)))
            mapped = self.model.evaluate(dict(
                obs, heightmap=torch.randn(obs['heightmap'].shape, generator=generator),
                history=torch.randn(obs['history'].shape, generator=generator)))
        self.assertGreater(float((base - route).abs().max()), 0.)
        self.assertGreater(float((base - mapped).abs().max()), 0.)
        self.assertEqual(tuple(base.shape), (4, 1))


class StdConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.observations = target.synthetic_observations(4, seed=7)

    def tearDown(self):
        with torch.no_grad():
            self.model.std.fill_(spec.INITIAL_STD)

    def test_action_distribution_clamps_both_ends_without_mutating_parameter(self):
        for raw, expected in ((.05, .10), (.10, .10), (.15, .15), (.30, .30), (.50, .30)):
            with torch.no_grad():
                self.model.std.fill_(raw)
            distribution = self.model.action_distribution(self.observations)
            self.assertTrue(torch.allclose(
                distribution.stddev,
                torch.full((4, 12), expected, dtype=torch.float32), atol=1e-6), msg=raw)
            with torch.no_grad():
                reference = self.model(self.observations)
            self.assertTrue(torch.equal(distribution.mean, reference))
            self.assertTrue(all(abs(float(value) - raw) <= 1e-7
                                for value in self.model.std.detach()))

    def test_constrain_std_projects_the_parameter_after_adam(self):
        for raw, expected in ((.05, .10), (.50, .30), (.20, .20)):
            with torch.no_grad():
                self.model.std.fill_(raw)
            self.model.constrain_std()
            self.assertTrue(torch.allclose(self.model.std,
                                           torch.full((12,), expected), atol=1e-7))
        self.assertTrue(self.model.std.requires_grad)

    def test_initial_std_is_exactly_point_fifteen(self):
        for _ in range(2):
            model = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
            self.assertTrue(torch.equal(model.std.detach().cpu(), torch.full((12,), .15)))
            distribution = model.action_distribution(self.observations)
            self.assertTrue(torch.allclose(distribution.stddev,
                                           torch.full((4, 12), .15), atol=1e-6))


class FreshStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_critic_is_fresh_and_matches_a_seed42_r2_construction(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            fresh = V05R2()
        state, reference = self.model.state_dict(), fresh.state_dict()
        fresh_keys = [key for key in reference if key != 'std' and not key.startswith('actor.')]
        self.assertEqual(len(fresh_keys), 24)
        for key in fresh_keys:
            self.assertTrue(torch.equal(reference[key], state[key]), key)
        self.assertEqual(self.model.critic_normalizer.count.item(), 0)
        self.assertEqual(float(self.model.critic_normalizer.mean.abs().sum()), 0.)
        self.assertTrue(torch.equal(self.model.critic_normalizer.var,
                                    torch.ones_like(self.model.critic_normalizer.var)))
        # The r2 211-input Critic never inherits the ETH 72-dim Critic block.
        eth = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=True)
        self.assertIn('critic_state_dict', eth)
        self.assertEqual(tuple(eth['critic_state_dict']['mlp.0.weight'].shape), (512, 72))
        self.assertNotEqual(tuple(state['critic.0.weight'].shape), (512, 72))
        self.assertEqual(self.model.transfer_metadata['forbidden_source_blocks_migrated'], [])

    def test_adam_state_is_empty_and_counters_are_zero(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=spec.PPO_INITIAL_LR)
        self.assertEqual(optimizer.state, {})
        meta = self.model.transfer_metadata
        self.assertEqual(meta['optimizer'], 'fresh_adam')
        self.assertEqual(meta['critic'], 'fresh_seeded_211')
        self.assertEqual(meta['critic_input_dims'], 211)
        self.assertEqual(meta['route_fusion'], 'zeros_after_migration')
        self.assertEqual(meta['route_fusion_tensor_count'], 2)
        self.assertEqual(meta['forbidden_source_blocks'], ['critic_state_dict',
                                                           'optimizer_state_dict'])
        self.assertFalse(meta['resume'])
        self.assertEqual(meta['scope'], 'initialization')
        self.assertEqual(meta['architecture'], spec.ARCHITECTURE)

    def test_recipe_identity_is_bound_on_the_model(self):
        self.assertEqual(self.model.recipe, spec.RECIPE_VERSION)
        self.assertEqual(self.model.recipe_settings, spec.recipe_config())


class ObservationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.observations = target.synthetic_observations(3, seed=11)

    def test_every_consumer_rejects_a_missing_route_error(self):
        partial = {key: value for key, value in self.observations.items()
                   if key != 'route_error'}
        for name, call in (
                ('forward', lambda obs: self.model(obs)),
                ('evaluate', lambda obs: self.model.evaluate(obs)),
                ('action_distribution', lambda obs: self.model.action_distribution(obs)),
                ('act', lambda obs: self.model.act(obs, deterministic=True)),
                ('update_critic_statistics',
                 lambda obs: self.model.update_critic_statistics(obs))):
            with self.subTest(consumer=name), self.assertRaises(ValueError) as caught:
                call(partial)
            message = str(caught.exception)
            self.assertIn('route_error', message)
            self.assertIn(name, message)
            self.assertIn('never zero filled', message)

    def test_every_consumer_rejects_a_missing_privileged_key(self):
        partial = {key: value for key, value in self.observations.items()
                   if key != 'privileged'}
        for name, call in (('evaluate', lambda obs: self.model.evaluate(obs)),
                           ('update_critic_statistics',
                            lambda obs: self.model.update_critic_statistics(obs))):
            with self.subTest(consumer=name), self.assertRaises(ValueError) as caught:
                call(partial)
            self.assertIn('privileged', str(caught.exception))
        # The Actor path does not require privileged.
        self.assertEqual(tuple(self.model(partial).shape), (3, 12))

    def test_bad_route_shapes_dtypes_batches_devices_and_values_are_rejected(self):
        route = self.observations['route_error']
        mutations = {
            'two_columns': route[:, :2],
            'four_columns': torch.cat((route, route[:, :1]), -1),
            'three_d': route.unsqueeze(-1),
            'float64': route.double(),
            'batch': route[:2],
            'nan': torch.full_like(route, float('nan')),
            'inf': torch.full_like(route, float('inf')),
        }
        for name, value in mutations.items():
            with self.subTest(mutation=name), self.assertRaises(ValueError):
                self.model(dict(self.observations, route_error=value))
            with self.subTest(mutation=name, critic=True), self.assertRaises(ValueError):
                self.model.evaluate(dict(self.observations, route_error=value))

    def test_shape_error_message_prints_expected_and_actual(self):
        route = self.observations['route_error'][:, :2]
        with self.assertRaises(ValueError) as caught:
            self.model(dict(self.observations, route_error=route))
        self.assertIn('B,3', str(caught.exception))
        self.assertIn('[3, 2]', str(caught.exception))


class RouteFormulaTests(unittest.TestCase):
    def test_formula_units_and_saturation(self):
        local_y = torch.tensor([-.8001, -.8, -.4, -.2, 0., .2, .4, .8, .8001])
        yaw = torch.zeros_like(local_y)
        route = target.route_error_from_pose(local_y, yaw)
        self.assertEqual(tuple(route.shape), (9, 3))
        self.assertEqual(route.dtype, torch.float32)
        self.assertTrue(torch.isfinite(route).all())
        self.assertTrue(torch.allclose(route[:, 0],
                                       torch.tensor([-4., -4., -2., -1., 0., 1., 2., 4., 4.]),
                                       atol=1e-6))
        self.assertTrue(torch.allclose(route[:, 1], torch.zeros(9), atol=1e-7))
        self.assertTrue(torch.allclose(route[:, 2], torch.ones(9), atol=1e-7))
        for degrees in (0., 5., -5., 10., -10., 90., -90., 180., -180.):
            e_world = math.radians(degrees)
            row = target.route_error_from_pose(torch.tensor([0., .2]), torch.tensor(
                [e_world, e_world]))
            self.assertAlmostEqual(float(row[0, 1]), math.sin(e_world), places=6)
            self.assertAlmostEqual(float(row[0, 2]), math.cos(e_world), places=6)
            self.assertAlmostEqual(float(row[0, 0]), 0., places=7)
            self.assertAlmostEqual(float(row[1, 0]), 1., places=6)
        # Wrapping: 370 degrees is 10 degrees, not 370.
        wrapped = target.route_error_from_pose(torch.zeros(1),
                                               torch.tensor([math.radians(370.)]))
        self.assertAlmostEqual(float(wrapped[0, 1]), math.sin(math.radians(10.)), places=5)
        with self.assertRaises(ValueError):
            target.route_error_from_pose(torch.tensor([float('nan')]), torch.zeros(1))
        with self.assertRaises(ValueError):
            target.route_error_from_pose(torch.zeros(2), torch.zeros(3))

    def test_non_zero_yaw_axis_fails_closed(self):
        """SDS 4.2/13.5: the atlas is translation-only, so ``yaw_axis`` is frozen at 0.

        ``spec`` already rejects any other axis and the r2 environment asserts the
        translation-only metadata; this reference helper must fail closed too
        instead of silently computing an unsupported (rotated-axis) observation.
        """
        self.assertEqual(spec.ROUTE_YAW_AXIS, 0.)
        zeros = torch.zeros(2)
        rejected = (0.1, -0.1, math.radians(5.), -math.radians(5.), math.pi, 1., True,
                    float('nan'), float('inf'), float('-inf'), 'x', None)
        for axis in rejected:
            with self.subTest(yaw_axis=axis), self.assertRaises(ValueError) as caught:
                target.route_error_from_pose(zeros, zeros, yaw_axis=axis)
            self.assertIn('yaw_axis', str(caught.exception))
            if axis in (0.1, -0.1, math.radians(5.), -math.radians(5.), math.pi, 1.):
                self.assertIn('translations only', str(caught.exception))
                self.assertIn('13.5', str(caught.exception))
        # The frozen zero axis keeps working through both the default and the kwarg.
        row = target.route_error_from_pose(torch.tensor([0., .2]), torch.zeros(2))
        self.assertTrue(torch.equal(
            row, target.route_error_from_pose(torch.tensor([0., .2]), torch.zeros(2),
                                              yaw_axis=0.)))
        self.assertAlmostEqual(float(row[1, 0]), 1., places=6)

    def test_synthetic_observations_cover_the_audit_grid(self):
        observations = target.synthetic_observations(128, seed=42)
        self.assertEqual(set(observations), set(spec.OBSERVATION))
        for key, shape in spec.OBSERVATION.items():
            self.assertEqual(tuple(observations[key].shape[1:]), tuple(shape), key)
            self.assertEqual(observations[key].dtype, torch.float32, key)
            self.assertTrue(torch.isfinite(observations[key]).all(), key)
        route = observations['route_error']
        self.assertEqual(tuple(route.shape), (128, 3))
        self.assertTrue(torch.equal(route[:, 0], route[:, 0].clamp(-4., 4.)))
        coverage = target.route_coverage(route)
        self.assertTrue(coverage['zero_present'])
        self.assertTrue(coverage['lateral_plus_clip'])
        self.assertTrue(coverage['lateral_minus_clip'])
        self.assertEqual(sorted(coverage['heading_radians']),
                         sorted(target.ROUTE_AUDIT_E_WORLD))
        self.assertTrue(coverage['finite'])
        # Mixed batch positions: the first four route rows are not a contiguous
        # slice of the audit grid.
        self.assertFalse(torch.equal(route[:4, 0],
                                     torch.tensor([target.ROUTE_AUDIT_LOCAL_Y[0]] * 4)))


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.r1_model = IdealHeightmapActorCritic()

    def test_tensor_mode_accepts_only_the_exact_r2_state(self):
        state = self.model.state_dict()
        self.assertEqual(target.validate_model_state(state, self.model), 'model')
        with torch.device('meta'):
            template = V05R2()
        self.assertEqual(target.validate_model_state(state, template), 'model')
        with self.assertRaises(ValueError):
            target.validate_model_state({'contract': {}, 'model': state}, template)
        bad = copy.deepcopy(state)
        bad.pop('actor.route_fusion.bias')
        with self.assertRaises(ValueError):
            target.validate_model_state(bad, template)
        extra = copy.deepcopy(state)
        extra['actor.bogus'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            target.validate_model_state(extra, template)
        # An r1 (schema-6, 208-Critic, no bypass) state is rejected, not adapted.
        with self.assertRaises(ValueError):
            target.validate_model_state(self.r1_model.state_dict(), template)
        for mutation in ('std_low', 'std_high', 'nan', 'dtype', 'norm_std', 'norm_var',
                         'critic_var', 'critic_count', 'continuous', 'actor_mean_dims',
                         'critic_dims', 'route_shape', 'critic_shape'):
            changed = copy.deepcopy(state)
            if mutation == 'std_low':
                changed['std'] = torch.full((12,), .0999)
            if mutation == 'std_high':
                changed['std'] = torch.full((12,), .3001)
            if mutation == 'nan':
                changed['critic.0.weight'] = changed['critic.0.weight'].clone()
                changed['critic.0.weight'].view(-1)[0] = float('nan')
            if mutation == 'dtype':
                changed['critic.0.weight'] = changed['critic.0.weight'].double()
            if mutation == 'norm_std':
                changed['actor.normalizer._std'] = changed['actor.normalizer._std'].clone()
                changed['actor.normalizer._std'].view(-1)[0] = 0.
            if mutation == 'norm_var':
                changed['actor.normalizer._var'] = changed['actor.normalizer._var'].clone()
                changed['actor.normalizer._var'].view(-1)[0] = -1.
            if mutation == 'critic_var':
                changed['critic_normalizer.var'] = changed['critic_normalizer.var'].clone()
                changed['critic_normalizer.var'].view(-1)[0] = -1.
            if mutation == 'critic_count':
                changed['critic_normalizer.count'] = torch.tensor(-1., dtype=torch.float64)
            if mutation == 'continuous':
                changed['critic_normalizer.continuous'] = changed[
                    'critic_normalizer.continuous'].clone()
                changed['critic_normalizer.continuous'][48] = True
            if mutation == 'actor_mean_dims':
                changed['actor.normalizer._mean'] = torch.zeros(1, 48)
            if mutation == 'critic_dims':
                changed['critic_normalizer.mean'] = torch.zeros(208)
            if mutation == 'route_shape':
                changed['actor.route_fusion.weight'] = torch.zeros(512, 4)
            if mutation == 'critic_shape':
                changed['critic.0.weight'] = torch.zeros(512, 208)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                target.validate_model_state(changed, template)

    def test_schema7_envelope_mode_rejects_unknown_tails(self):
        contract = spec.contract()
        state = self.model.state_dict()
        payload = {'contract': contract, 'model': state}
        self.assertEqual(
            target.validate_model_state(payload, self.model, contract=contract,
                                        require_closure=False), 'schema7')
        strict_fields = {'contract', 'model', 'rng', 'ledger', 'optimizer',
                         'learning_state', 'curriculum'}
        for extra in ('rng', 'ledger', 'optimizer', 'unknown_tail'):
            bad = dict(payload)
            bad[extra] = {}
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                target.validate_model_state(bad, self.model, contract=contract,
                                            require_closure=False)
        with self.assertRaises(ValueError):
            target.validate_model_state(payload, self.model, contract=contract,
                                        required_fields=strict_fields, require_closure=False)
        full = dict(payload, rng={}, ledger={}, optimizer={}, learning_state={}, curriculum={})
        self.assertEqual(
            target.validate_model_state(full, self.model, contract=contract,
                                        required_fields=strict_fields,
                                        require_closure=False), 'schema7')
        other = spec.contract(mode='pilot')
        with self.assertRaises(ValueError):
            target.validate_model_state(payload, self.model, contract=other,
                                        require_closure=False)

    def test_schema7_keys_reject_r1_schema_le_6_and_smoke(self):
        contract = spec.contract()
        state = dict(
            schema=7, identity=spec.identity_block(),
            contract_sha256=spec.contract_sha256(contract),
            model=self.model.state_dict(), optimizer={}, normalizer={}, learning={},
            rng={}, curriculum={}, ledger={}, evaluation={}, source={},
            recipe=spec.recipe_config(), resume_mode='new_episode')
        self.assertIs(target.validate_schema7_keys(state), state)
        self.assertEqual(set(state), set(target.SCHEMA7_TOP_LEVEL_KEYS))
        for schema in (6, 5, 0, -1):
            bad = dict(state, schema=schema)
            with self.subTest(schema=schema), self.assertRaises(target.LegacySchemaError):
                target.validate_schema7_keys(bad)
            with self.assertRaises(ValueError):
                target.validate_schema7_keys(bad)
        for schema in (True, 7.0, '7', None):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                target.validate_schema7_keys(dict(state, schema=schema))
        unlabelled = {key: value for key, value in state.items() if key != 'schema'}
        with self.assertRaises(target.LegacySchemaError):
            target.validate_schema7_keys(unlabelled)
        with self.assertRaises(target.LegacySchemaError):
            target.validate_schema7_keys(dict(unlabelled, contract=contract))
        for schema in (8, 9):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                target.validate_schema7_keys(dict(state, schema=schema))
        for extra in ('unknown_tail', 'formal_resumable'):
            bad = dict(state)
            bad[extra] = True
            with self.assertRaises(ValueError):
                target.validate_schema7_keys(bad)
        missing = dict(state)
        missing.pop('evaluation')
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(missing)
        for container in ('v05-smoke-state-v1', spec.SMOKE_CONTAINER):
            with self.subTest(container=container), self.assertRaises(ValueError):
                target.validate_schema7_keys(dict(state, container=container))
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, formal_resumable=False))
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, scope='initialization'))
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, resume_mode='continue'))
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, identity={'task': spec.TASK}))
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, recipe=dict(spec.recipe_config(),
                                                                recipe='0.5-r1')))
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, contract_sha256='not-a-digest'))
        # An r1/schema-6 model body cannot slip through the tensor key gate.
        r1_body = dict(state, model=self.r1_model.state_dict())
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(r1_body)
        body = dict(self.model.state_dict())
        body.pop('actor.route_fusion.weight')
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, model=body))
        body = dict(self.model.state_dict())
        body['critic.0.weight'] = torch.zeros(512, 208)
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, model=body))
        shaped = dict(self.model.state_dict())
        shaped['actor.route_fusion.weight'] = torch.zeros(512, 4)
        with self.assertRaises(ValueError):
            target.validate_schema7_keys(dict(state, model=shaped))

    def test_identity_block_is_exact(self):
        self.assertEqual(spec.identity_block(), dict(
            task=spec.TASK, recipe=spec.RECIPE_VERSION, code_version=spec.CODE_VERSION,
            protocol=spec.PROTOCOL_ID, sds_revision=spec.SDS_REVISION))
        self.assertEqual(target.validate_identity(spec.identity_block()),
                         spec.identity_block())
        for key in ('task', 'recipe', 'code_version', 'protocol', 'sds_revision'):
            bad = dict(spec.identity_block())
            bad[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                target.validate_identity(bad)
        with self.assertRaises(ValueError):
            target.validate_identity(dict(spec.identity_block(),
                                          architecture=spec.ARCHITECTURE))

    def test_contract_identity_recipe_and_closure_sha(self):
        contract = spec.contract()
        self.assertEqual(target.validate_contract(contract, require_closure=False), contract)
        mutations = {
            'schema6': ('checkpoint_schema', 6),
            'schema8': ('checkpoint_schema', 8),
            'code': ('code_version', '0.5.0'),
            'architecture': ('architecture', 'v05'),
            'task': ('task', 'Unitree-A2-IdealHeightMap-Stairs-V05'),
            'protocol': ('protocol_id', 'v05-stair-climb-first-v1'),
            'flat_protocol': ('flat_protocol_id', 'v05-flat-speed-v1'),
            'recipe': ('recipe_version', '0.5-r1'),
            'reward_version': ('reward_version', '0.5-r1'),
            'reward_sha': ('reward_config_sha256', '0' * 64),
            'sds': ('sds_revision', '1.1'),
            'sds_sha': ('sds_sha256', '0' * 64),
            'design_sha': ('design_decision_sha256', '0' * 64),
            'implementation_sha': ('implementation_decision_sha256', '0' * 64),
            'identity': ('identity', {'task': spec.TASK}),
            'route_state': ('route_state', dict(spec.ROUTE_STATE, yaw_axis=.5)),
            'observation': ('observation', dict(spec.OBSERVATION, route_error=[2])),
            'budget': ('total_budget', 29500),
            'cap': ('update_cap', 29500),
        }
        for name, (key, value) in mutations.items():
            bad = copy.deepcopy(contract)
            bad[key] = value
            with self.subTest(mutation=name), self.assertRaises(ValueError):
                target.validate_contract(bad, require_closure=False)
        bad = copy.deepcopy(contract)
        bad['behavior_sha256'] = {}
        with self.assertRaises(ValueError):
            target.validate_contract(bad, require_closure=False)
        bad = copy.deepcopy(contract)
        bad['behavior_sha256'] = dict(bad['behavior_sha256'], **{'../escape.py': '0' * 64})
        with self.assertRaises(ValueError):
            target.validate_contract(bad, require_closure=False)
        bad = copy.deepcopy(contract)
        bad['behavior_sha256'] = dict(bad['behavior_sha256'], **{'x.py': 'not-a-digest'})
        with self.assertRaises(ValueError):
            target.validate_contract(bad, require_closure=False)
        stale = copy.deepcopy(contract)
        stale['behavior_sha256'] = dict(stale['behavior_sha256'])
        stale['behavior_sha256']['src/tasks/velocity/v05_r2/spec.py'] = '0' * 64
        target.validate_contract(stale, require_closure=False)
        with self.assertRaisesRegex(ValueError, 'closure SHA mismatch'):
            target.validate_contract(stale, require_closure=True)
        with self.assertRaises(ValueError):
            target.validate_contract(spec.contract(seed=43), require_closure=False, seed=42)
        with self.assertRaises(ValueError):
            target.validate_contract(spec.contract(mode='pilot'), require_closure=False,
                                     mode='formal')

    def test_loader_is_strict_and_never_uses_strict_false(self):
        state = self.model.state_dict()
        target.load_model_state(self.model, state)
        bad = dict(state)
        bad.pop('actor.route_fusion.bias')
        with self.assertRaises(ValueError) as caught:
            target.load_model_state(self.model, bad)
        self.assertIn('strict=False is never used', str(caught.exception))
        with self.assertRaises(ValueError):
            target.load_model_state(self.model, self.r1_model.state_dict())

    def test_model_validator_never_accepts_an_envelope_in_tensor_mode(self):
        full = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=True)
        with self.assertRaises(ValueError):
            target.validate_model_state(full, self.model)


class MigrationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=True)

    def test_wrong_sha_and_unapproved_source_are_rejected_before_loading(self):
        with patch.object(target.torch, 'load') as loader:
            for expected in ('0' * 64, 'f' * 64, None, '', spec.INIT_SHA256.upper()):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    V05R2.from_eth(spec.INIT_PATH, expected, 42)
            loader.assert_not_called()
        with patch.object(target, 'sha256', return_value='0' * 64):
            with self.assertRaises(ValueError):
                V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        with self.assertRaises(FileNotFoundError):
            V05R2.from_eth(ROOT / 'logs' / 'missing.pt', spec.INIT_SHA256, 42)
        for seed in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError, msg=repr(seed)):
                V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, seed)

    def test_corrupt_source_state_is_rejected(self):
        mutations = {
            'missing_mlp': lambda state: state.pop('mlp.2.bias'),
            'missing_normalizer': lambda state: state.pop('obs_normalizer._mean'),
            'missing_std': lambda state: state.pop('std'),
            'extra': lambda state: state.update({'actor.bogus': torch.zeros(1)}),
            'shape': lambda state: state.update({'mlp.0.weight': torch.zeros(1, 45)}),
            'dtype': lambda state: state.update({'mlp.4.weight': state['mlp.4.weight'].double()}),
            'nan': lambda state: state['mlp.0.weight'].view(-1).__setitem__(0, float('nan')),
            'inf': lambda state: state['std'].view(-1).__setitem__(1, float('inf')),
            'zero_normalizer_std': lambda state: state['obs_normalizer._std'].fill_(0.),
            'negative_variance': lambda state: state['obs_normalizer._var'].fill_(-1.),
            'negative_std': lambda state: state['std'].fill_(-1.),
            'bad_count': lambda state: state['obs_normalizer.count'].fill_(-1),
        }
        for name, apply in mutations.items():
            with self.subTest(mutation=name):
                bad = copy.deepcopy(self.source)
                apply(bad['actor_state_dict'])
                with patch.object(target, 'sha256', return_value=spec.INIT_SHA256), \
                     patch.object(target.torch, 'load', return_value=bad):
                    with self.assertRaises(ValueError):
                        V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        bad = {'iter': 1}
        with patch.object(target, 'sha256', return_value=spec.INIT_SHA256), \
             patch.object(target.torch, 'load', return_value=bad):
            with self.assertRaises(ValueError):
                V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_migration_is_deterministic_and_seed_independent_where_documented(self):
        torch.manual_seed(123456)
        first = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        torch.manual_seed(987654)
        second = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        other_seed = V05R2.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 43)
        self.assertFalse(torch.equal(first.critic[0].weight, other_seed.critic[0].weight))
        for key in first.state_dict():
            if key.startswith('actor.mlp.') or key.startswith('actor.normalizer.'):
                self.assertTrue(torch.equal(first.state_dict()[key],
                                            other_seed.state_dict()[key]), key)
            if key in spec.ROUTE_FUSION_KEYS:
                self.assertFalse(first.state_dict()[key].any(), key)


if __name__ == '__main__':
    unittest.main()
