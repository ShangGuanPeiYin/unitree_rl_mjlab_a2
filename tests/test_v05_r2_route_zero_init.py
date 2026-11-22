"""SDS 2.6 / 9.3 route-bypass zero-initialization evidence (CPU only).

The five initialization audits that gate the r2 network:

A1  128 synthetic observations with *changing* ``route_error`` give an Actor mean
    within ``1e-6`` of the unmodified ETH Actor MLP + frozen normalizer;
A2  changing route/map/history cannot change the zero-fusion initial output;
A3  the two new ``actor.route_fusion`` tensors are elementwise zero and are the
    only new tensors relative to r1;
B   both bypass tensors are in ``torch.optim.Adam(model.parameters())`` and
    receive a finite gradient from one residual backward pass (no optimizer step);
C   the Critic is 211-wide with ``route_error`` as its last three features and
    equals a same-seed, same-construction-order fresh r2 reference per tensor.

Only *initialization mean equivalence* is claimed: the ETH source has no
``route_fusion``, so "all tensors equal" is never asserted.  ``route_error`` is
also proven never to touch either normalizer, and the bypass is proven to consume
its input un-normalized (no clamp inside the network).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_r2_bootstrap  # noqa: E402

a2_v05_r2_bootstrap.activate(ROOT)

import math  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.v05_r2 import model as target  # noqa: E402
from src.tasks.velocity.v05_r2 import spec  # noqa: E402

V05R2ActorCritic = target.V05R2ActorCritic


class RouteZeroInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.source = torch.load(spec.INIT_PATH, map_location='cpu',
                                weights_only=True)['actor_state_dict']
        cls.observations = target.synthetic_observations(128, seed=42)
        cls.reference = target.load_eth_actor_reference(spec.INIT_PATH)

    def test_a1_initial_actor_mean_matches_eth_with_varying_route(self):
        route = self.observations['route_error']
        self.assertEqual(tuple(route.shape), (128, 3))
        # The 128 observations really do vary the route input: 81 distinct
        # (local_y, e_world) triples are cycled over 128 mixed batch positions.
        self.assertEqual(int(torch.unique(route[:, 0]).numel()), 7)
        self.assertGreater(int(torch.unique(route[:, 1]).numel()), 8)
        self.assertGreater(int(torch.unique(route[:, 2]).numel()), 4)
        triples = {tuple(row) for row in route.tolist()}
        self.assertGreater(len(triples), 40)
        self.assertFalse(bool((route == 0).all()))
        result = target.mean_equivalence(self.model, self.reference, self.observations)
        self.assertEqual(result['observations'], 128)
        self.assertTrue(result['finite'])
        self.assertTrue(result['allclose'])
        self.assertLessEqual(result['max_abs_error'], 1e-6)
        self.assertTrue(result['passed'])
        # Only initialization mean equivalence is claimed.
        self.assertEqual(result['compared'], 'deterministic_actor_mean')
        self.assertIn('initialization_mean_equivalence_only', result['claim'])

    def test_a2_zero_fusion_output_ignores_route_map_and_history(self):
        matrix = target.map_history_invariance(self.model, self.observations)
        self.assertTrue(matrix['passed'])
        self.assertEqual(matrix['max_abs_difference'], 0.)
        route = target.route_zero_invariance(self.model, self.observations)
        self.assertTrue(route['passed'])
        self.assertEqual(route['max_abs_difference'], 0.)
        self.assertEqual(set(route['variants']),
                         {'negated', 'zeroed', 'clipped_ends', 'randomized'})
        for name, item in route['variants'].items():
            self.assertEqual(item['max_abs_difference'], 0., name)
            self.assertTrue(item['passed'], name)
        # A deliberately extreme route/map/history must still leave the mean
        # bit-identical while both bypass weight sets are zero.
        altered = dict(self.observations)
        altered['route_error'] = torch.full_like(self.observations['route_error'], 123.)
        altered['heightmap'] = torch.full_like(self.observations['heightmap'], -7.)
        altered['history'] = torch.full_like(self.observations['history'], 0.75)
        self.model.eval()
        try:
            with torch.no_grad():
                self.assertTrue(torch.equal(self.model(self.observations),
                                            self.model(altered)))
        finally:
            self.model.train()

    def test_a3_new_tensors_are_elementwise_zero_and_exactly_two(self):
        audit = target.route_fusion_zero_audit(self.model)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['tensor_count'], 2)
        self.assertEqual(audit['keys'], ['actor.route_fusion.weight',
                                         'actor.route_fusion.bias'])
        self.assertEqual(audit['shapes'], {'actor.route_fusion.weight': [512, 3],
                                           'actor.route_fusion.bias': [512]})
        self.assertTrue(all(audit['elementwise_zero'].values()))
        state = self.model.state_dict()
        self.assertEqual(int(state['actor.route_fusion.weight'].numel()), 512 * 3)
        self.assertEqual(int(state['actor.route_fusion.bias'].numel()), 512)
        # Exactly two new tensors relative to the r1 state (51 -> 53).
        r1_keys = set(state) - set(spec.ROUTE_FUSION_KEYS)
        self.assertEqual(len(state), 53)
        self.assertEqual(len(r1_keys), 51)
        self.assertEqual(set(spec.ROUTE_FUSION_KEYS) - set(r1_keys),
                         {'actor.route_fusion.weight', 'actor.route_fusion.bias'})
        # The r1 fusion stays zero too (SDS 2.2 step 4: four zeroed tensors).
        for key in spec.FUSION_KEYS:
            self.assertEqual(float(state[key].abs().sum()), 0., key)

    def test_b_bypass_is_in_adam_and_receives_a_finite_gradient(self):
        audit = target.route_bypass_trainability(self.model, self.observations)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['parameters'], ['actor.route_fusion.weight',
                                               'actor.route_fusion.bias'])
        self.assertTrue(all(audit['in_optimizer'].values()))
        self.assertTrue(audit['loss_finite'])
        self.assertFalse(audit['optimizer_step_called'])
        for name, row in audit['gradient_rows'].items():
            self.assertTrue(row['present'], name)
            self.assertTrue(row['finite'], name)
            self.assertGreater(row['abs_max'], 0., name)
        self.assertTrue(audit['gradient_nonzero'])
        # The audit leaves no gradient residue behind.
        self.assertIsNone(self.model.actor.route_fusion.weight.grad)
        self.assertIsNone(self.model.actor.route_fusion.bias.grad)
        # Direct Adam membership check by identity and storage pointer.
        parameters = {id(parameter): parameter
                      for parameter in self.model.parameters()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=spec.PPO_INITIAL_LR)
        ids = {id(parameter) for group in optimizer.param_groups
               for parameter in group['params']}
        ptrs = {parameter.data_ptr() for group in optimizer.param_groups
                for parameter in group['params']}
        for name in spec.ROUTE_FUSION_KEYS:
            parameter = dict(self.model.named_parameters())[name]
            self.assertIs(parameters[id(parameter)], parameter)
            self.assertIn(id(parameter), ids)
            self.assertIn(parameter.data_ptr(), ptrs)
            self.assertTrue(parameter.requires_grad)

    def test_c_critic_211_and_fresh_reference_per_tensor(self):
        audit = target.critic_211_audit(self.model, self.observations, seed=42)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['critic_in_features'], 211)
        self.assertEqual(audit['critic_out_features'], 512)
        self.assertEqual(audit['critic_first_layer_shape'], [512, 211])
        self.assertEqual(audit['critic_normalizer_dims'], 80)
        self.assertEqual(audit['critic_input_row_dims'], spec.CRITIC_INPUT_DIMS)
        self.assertTrue(audit['route_tail_equal'])
        self.assertEqual(audit['route_tail_index'], [208, 211])
        self.assertEqual(audit['route_tail_shape'], [128, 211])
        self.assertTrue(audit['route_changes_value'])
        self.assertGreater(audit['route_value_max_abs_difference'], 0.)
        self.assertTrue(audit['map_history_changes_value'])
        self.assertGreater(audit['map_history_value_max_abs_difference'], 0.)
        self.assertTrue(audit['fresh_all_equal'])
        self.assertEqual(len(audit['fresh_keys']), 24)
        self.assertTrue(all(audit['fresh_equal'].values()))
        # A different seed must produce a different fresh Critic.
        other = target.critic_211_audit(self.model, self.observations, seed=43)
        self.assertFalse(other['fresh_all_equal'])
        self.assertIn('critic.0.weight',
                      [key for key, equal in other['fresh_equal'].items() if not equal])

    def test_aggregate_audit_reports_all_five_items(self):
        audit = target.audit_initialization(
            self.model, source_state=self.source, reference=self.reference,
            observations=self.observations, seed=42)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['observations'], 128)
        self.assertEqual(audit['new_tensor_count'], 2)
        self.assertEqual(audit['actor_copied_tensors'], 12)
        self.assertTrue(audit['actor_all_copied_equal'])
        for name in ('actor_mean_equivalence', 'zero_fusion_invariance',
                     'route_zero_invariance', 'route_fusion_zero',
                     'route_bypass_trainability', 'critic_211'):
            self.assertTrue(audit[name]['passed'], name)
        self.assertEqual(audit['route_coverage']['samples'], 128)
        self.assertTrue(audit['route_coverage']['zero_present'])
        self.assertEqual(len(audit['route_coverage']['heading_radians']),
                         len(target.ROUTE_AUDIT_E_WORLD))
        with self.assertRaises(ValueError):
            target.audit_initialization(self.model, source_state=self.source,
                                        reference=None, observations=self.observations,
                                        seed=42)


class RouteNormalizerIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = V05R2ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        cls.observations = target.synthetic_observations(16, seed=5)

    def test_route_never_enters_either_normalizer(self):
        self.assertEqual(spec.ETH_NORMALIZER_DIMS, 45)
        self.assertEqual(spec.CRITIC_NORMALIZER_DIMS, 80)
        state = self.model.state_dict()
        self.assertEqual(tuple(state['actor.normalizer._mean'].shape), (1, 45))
        self.assertEqual(tuple(state['actor.normalizer._var'].shape), (1, 45))
        self.assertEqual(tuple(state['actor.normalizer._std'].shape), (1, 45))
        self.assertEqual(tuple(state['critic_normalizer.mean'].shape), (80,))
        self.assertEqual(tuple(state['critic_normalizer.var'].shape), (80,))
        self.assertEqual(int(state['critic_normalizer.mean'].numel()), 80)
        mask = state['critic_normalizer.continuous']
        self.assertTrue(torch.equal(mask[48:52], torch.zeros(4, dtype=torch.bool)))
        # The frozen ETH normalizer never moves, whatever the route says.
        snapshot = {key: value.clone() for key, value in state.items()
                    if key.startswith('actor.normalizer.')}
        routes = (torch.zeros_like(self.observations['route_error']),
                  torch.full_like(self.observations['route_error'], 4.),
                  torch.full_like(self.observations['route_error'], -4.),
                  torch.randn_like(self.observations['route_error']) * 1e3)
        for route in routes:
            changed = dict(self.observations, route_error=route)
            with torch.no_grad():
                self.model(changed)
                self.model.evaluate(changed)
        for key, value in snapshot.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]), key)

    def test_critic_statistics_ignore_the_route_channel(self):
        """Two fresh models fed the same 80-dim data but different routes agree."""
        routes = (torch.zeros_like(self.observations['route_error']),
                  torch.full_like(self.observations['route_error'], -4.))
        statistics = []
        for route in routes:
            model = V05R2ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
            model.update_critic_statistics(dict(self.observations, route_error=route))
            statistics.append({key: value.clone()
                               for key, value in model.state_dict().items()
                               if 'normalizer' in key})
        self.assertEqual(set(statistics[0]), set(statistics[1]))
        for key, value in statistics[0].items():
            self.assertTrue(torch.equal(value, statistics[1][key]), key)
        # The statistic did move (proprio/privileged are real data) ...
        self.assertGreater(float(statistics[0]['critic_normalizer.mean'].abs().sum()), 0.)
        self.assertEqual(statistics[0]['critic_normalizer.mean'].numel(), 80)
        self.assertGreater(float(statistics[0]['critic_normalizer.count']), 0.)

    def test_route_bypass_input_is_not_normalized_or_clipped(self):
        """SDS 2.3: only the formula clamps; the network must not clamp again."""
        model = V05R2ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        observations = target.synthetic_observations(4, seed=8)
        with torch.no_grad():
            model.actor.route_fusion.weight.fill_(1.)
            model.actor.route_fusion.bias.zero_()
            base = model(observations)
            clipped = model(dict(observations, route_error=torch.tensor(
                [[4., 0., 1.]] * 4, dtype=torch.float32)))
            beyond = model(dict(observations, route_error=torch.tensor(
                [[40., 0., 1.]] * 4, dtype=torch.float32)))
        self.assertFalse(torch.equal(clipped, beyond))
        self.assertGreater(float((clipped - beyond).abs().max()), 0.)
        self.assertFalse(torch.equal(base, clipped))


if __name__ == '__main__':
    unittest.main()
