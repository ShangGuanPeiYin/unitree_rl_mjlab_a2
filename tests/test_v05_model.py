"""CPU-only V0.5 model contract: dual std projection, schema-6 validation, ETH start.

These tests do not simulate, train or touch a GPU.  They pin the v0.5 identity,
the ``[0.10, 0.30]`` std range in both places SDS 2.2 names, the fresh
Critic/Adam/statistics state, and the explicit schema-6 model-state validator the
runner/recovery call sites use.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)

import copy  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import (  # noqa: E402
    IdealHeightmapActorCritic)
from src.tasks.velocity.v05 import model as target  # noqa: E402
from src.tasks.velocity.v05 import spec  # noqa: E402

V05 = target.V05ActorCritic


class PublicAPITests(unittest.TestCase):
    def test_v05_model_api_and_identity(self):
        self.assertTrue(callable(V05.from_eth))
        self.assertTrue(callable(target.validate_model_state))
        self.assertTrue(callable(target.validate_model_tensors))
        self.assertTrue(callable(target.validate_contract))
        self.assertTrue(callable(target.validate_initialization))
        self.assertTrue(callable(target.mean_equivalence))
        self.assertTrue(callable(target.map_history_invariance))
        self.assertTrue(callable(target.tensor_audit))
        self.assertEqual(V05.architecture_version, 'v05')
        self.assertEqual((V05.std_min, V05.std_max), (.10, .30))
        self.assertEqual(V05.std_min, spec.STD_MIN)
        self.assertEqual(V05.std_max, spec.STD_MAX)
        self.assertEqual(len(target.ETH_ACTOR_KEYS), 13)
        self.assertEqual(len(target.NORMALIZER_SOURCE_KEYS), 4)
        self.assertEqual(spec.INIT_SHA256,
                         '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722')


class StdConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.observations = target.synthetic_observations(4, seed=7)

    def tearDown(self):
        with torch.no_grad():
            self.model.std.fill_(spec.INITIAL_STD)

    def test_action_distribution_clamps_both_ends_without_mutating_parameter(self):
        mean = None
        for raw, expected in ((.05, .10), (.10, .10), (.15, .15), (.30, .30), (.50, .30)):
            with torch.no_grad():
                self.model.std.fill_(raw)
            distribution = self.model.action_distribution(self.observations)
            self.assertTrue(torch.allclose(
                distribution.stddev,
                torch.full((4, 12), expected, dtype=torch.float32), atol=1e-6),
                msg=raw)
            with torch.no_grad():
                reference = self.model(self.observations)
            self.assertTrue(torch.equal(distribution.mean, reference))
            self.assertTrue(all(abs(float(value) - raw) <= 1e-7
                                for value in self.model.std.detach()))
            mean = reference
        self.assertIsNotNone(mean)

    def test_constrain_std_projects_the_parameter_after_adam(self):
        with torch.no_grad():
            self.model.std.fill_(.05)
        self.model.constrain_std()
        self.assertTrue(torch.allclose(self.model.std, torch.full((12,), .10), atol=1e-7))
        with torch.no_grad():
            self.model.std.fill_(.50)
        self.model.constrain_std()
        self.assertTrue(torch.allclose(self.model.std, torch.full((12,), .30), atol=1e-7))
        with torch.no_grad():
            self.model.std.fill_(.20)
        self.model.constrain_std()
        self.assertTrue(torch.allclose(self.model.std, torch.full((12,), .20), atol=1e-7))
        self.assertTrue(self.model.std.requires_grad)

    def test_initial_std_is_exactly_point_fifteen(self):
        for _ in range(2):
            model = V05.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
            self.assertTrue(torch.equal(model.std.detach().cpu(),
                                        torch.full((12,), .15)))
            distribution = model.action_distribution(self.observations)
            self.assertTrue(torch.allclose(distribution.stddev,
                                           torch.full((4, 12), .15), atol=1e-6))
            self.assertTrue((distribution.stddev >= .10 - 1e-6).all())
            self.assertTrue((distribution.stddev <= .30 + 1e-6).all())


class FreshStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_critic_is_fresh_and_matches_a_seed42_construction(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            fresh = IdealHeightmapActorCritic()
        state, reference = self.model.state_dict(), fresh.state_dict()
        fresh_keys = [key for key in reference if key != 'std' and not key.startswith('actor.')]
        self.assertEqual(len(fresh_keys), 24)
        for key in fresh_keys:
            self.assertTrue(torch.equal(reference[key], state[key]), key)
        self.assertEqual(self.model.critic_normalizer.count.item(), 0)
        self.assertEqual(float(self.model.critic_normalizer.mean.abs().sum()), 0.)
        self.assertTrue(torch.equal(self.model.critic_normalizer.var,
                                    torch.ones_like(self.model.critic_normalizer.var)))
        self.assertTrue(torch.equal(self.model.critic_normalizer.continuous,
                                    reference['critic_normalizer.continuous']))

    def test_adam_state_is_empty_and_counters_are_zero(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=spec.PPO_INITIAL_LR)
        self.assertEqual(optimizer.state, {})
        self.assertTrue(all(len(group['params']) > 0 for group in optimizer.param_groups))
        meta = self.model.transfer_metadata
        self.assertEqual(meta['optimizer'], 'fresh_adam')
        self.assertEqual(meta['critic_statistics'], 'fresh')
        self.assertFalse(meta['resume'])
        self.assertEqual(meta['scope'], 'initialization')

    def test_recipe_identity_is_bound_on_the_model(self):
        self.assertEqual(self.model.recipe, spec.RECIPE_VERSION)
        self.assertEqual(self.model.recipe_settings, spec.recipe_config())


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_tensor_mode_accepts_only_the_exact_model_state(self):
        state = self.model.state_dict()
        self.assertEqual(target.validate_model_state(state, self.model), 'model')
        with torch.device('meta'):
            template = V05()
        self.assertEqual(target.validate_model_state(state, template), 'model')
        with self.assertRaises(ValueError):
            target.validate_model_state({'contract': {}, 'model': state}, template)
        bad = copy.deepcopy(state)
        bad.pop('actor.fusion.bias')
        with self.assertRaises(ValueError):
            target.validate_model_state(bad, template)
        extra = copy.deepcopy(state)
        extra['actor.bogus'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            target.validate_model_state(extra, template)
        for mutation in ('std_low', 'std_high', 'nan', 'dtype', 'norm_std', 'norm_var',
                         'critic_var', 'critic_count', 'continuous'):
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
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                target.validate_model_state(changed, template)

    def test_schema6_envelope_mode_rejects_unknown_tails(self):
        contract = spec.contract()
        state = self.model.state_dict()
        payload = {'contract': contract, 'model': state}
        self.assertEqual(
            target.validate_model_state(payload, self.model, contract=contract,
                                        require_closure=False), 'schema6')
        strict_fields = {'contract', 'model', 'rng', 'ledger', 'optimizer',
                         'learning_state', 'curriculum'}
        self.assertEqual(
            target.validate_model_state(payload, self.model, contract=contract,
                                        required_fields={'contract', 'model'},
                                        require_closure=False), 'schema6')
        for extra in ('rng', 'ledger', 'optimizer', 'unknown_tail'):
            bad = dict(payload)
            bad[extra] = {}
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                target.validate_model_state(bad, self.model, contract=contract,
                                            require_closure=False)
        for missing in (('contract',), ('model',)):
            bad = dict(payload)
            bad.pop(missing[0])
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                target.validate_model_state(bad, self.model, contract=contract,
                                            require_closure=False)
        # A caller that declares its full envelope must get an exact field match.
        with self.assertRaises(ValueError):
            target.validate_model_state(payload, self.model, contract=contract,
                                        required_fields=strict_fields, require_closure=False)
        full = dict(payload, rng={}, ledger={}, optimizer={}, learning_state={}, curriculum={})
        self.assertEqual(
            target.validate_model_state(full, self.model, contract=contract,
                                        required_fields=strict_fields,
                                        require_closure=False), 'schema6')
        # A different expected contract is rejected even when the shape matches.
        other = spec.contract(mode='pilot')
        with self.assertRaises(ValueError):
            target.validate_model_state(payload, self.model, contract=other,
                                        require_closure=False)

    def test_contract_identity_recipe_and_closure_sha(self):
        contract = spec.contract()
        self.assertEqual(target.validate_contract(contract, require_closure=False), contract)
        mutations = {
            'schema': ('checkpoint_schema', 5),
            'schema7': ('checkpoint_schema', 7),
            'code': ('code_version', '0.4.0'),
            'task': ('task', 'Unitree-A2-IdealHeightMap-Stairs-V04'),
            'protocol': ('protocol_id', 'v04-stair-body-speed-v1'),
            'flat_protocol': ('flat_protocol_id', 'v05-flat-speed-v2'),
            'recipe': ('recipe_version', '0.4-r1'),
            'reward_version': ('reward_version', '0.4-r1'),
            'reward_sha': ('reward_config_sha256', '0' * 64),
            'sds': ('sds_revision', '1.0'),
        }
        for name, (key, value) in mutations.items():
            bad = copy.deepcopy(contract)
            bad[key] = value
            with self.subTest(mutation=name), self.assertRaises(ValueError):
                target.validate_contract(bad, require_closure=False)
        for key in ('behavior_sha256',):
            bad = copy.deepcopy(contract)
            bad[key] = {}
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
        # A stale manifest fails closed only when the current closure is demanded.
        stale = copy.deepcopy(contract)
        stale['behavior_sha256'] = dict(stale['behavior_sha256'])
        stale['behavior_sha256']['src/tasks/velocity/v05/spec.py'] = '0' * 64
        target.validate_contract(stale, require_closure=False)
        with self.assertRaisesRegex(ValueError, 'closure SHA mismatch'):
            target.validate_contract(stale, require_closure=True)
        # Seed/mode checks are explicit, not implicit.
        with self.assertRaises(ValueError):
            target.validate_contract(spec.contract(seed=43), require_closure=False, seed=42)
        with self.assertRaises(ValueError):
            target.validate_contract(spec.contract(mode='pilot'), require_closure=False,
                                     mode='formal')

    def test_model_validator_never_accepts_an_envelope_in_tensor_mode(self):
        full = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=True)
        with self.assertRaises(ValueError):
            target.validate_model_state(full, self.model)


class MeanParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_synthetic_actor_mean_and_frozen_statistics(self):
        reference = target.load_eth_actor_reference(spec.INIT_PATH)
        observations = target.synthetic_observations(8, seed=812)
        result = target.mean_equivalence(self.model, reference, observations)
        self.assertTrue(result['passed'])
        self.assertLessEqual(result['max_abs_error'], 1e-6)
        self.assertTrue(result['finite'])
        statistics = {key: value.clone() for key, value in self.model.state_dict().items()
                      if 'normalizer' in key}
        with torch.no_grad():
            self.model(observations)
            self.model.evaluate(observations)
        for key, value in statistics.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]), key)

    def test_privileged_and_map_inputs_cannot_change_the_initial_mean(self):
        observations = target.synthetic_observations(8, seed=812)
        altered = dict(observations, privileged=observations['privileged'] + 100.)
        self.model.eval()
        try:
            with torch.no_grad():
                self.assertTrue(torch.equal(self.model(observations), self.model(altered)))
                self.assertTrue(target.map_history_invariance(
                    self.model, observations)['passed'])
        finally:
            self.model.train()

    def test_wrong_sha_and_unapproved_source_are_rejected_before_loading(self):
        with patch.object(target.torch, 'load') as loader:
            for expected in ('0' * 64, 'f' * 64, None, '', spec.INIT_SHA256.upper()):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    V05.from_eth(spec.INIT_PATH, expected, 42)
            loader.assert_not_called()
        with patch.object(target, 'sha256', return_value='0' * 64):
            with self.assertRaises(ValueError):
                V05.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        with self.assertRaises(FileNotFoundError):
            V05.from_eth(ROOT / 'logs' / 'missing.pt', spec.INIT_SHA256, 42)


if __name__ == '__main__':
    unittest.main()
