"""CPU-only v0.5-r2 PPO tests: frozen math, route storage and SDS 7.6 diagnostics.

Three groups:

* the frozen r2 loop must be tensor-identical to ``v03.ppo.PPO.update`` under a
  fixed storage and seed (SDS 2.5/7.6: r2 measures, it never retunes);
* ``route_error`` must enter the rollout buffer as ``[48*B,3]``, reach the actor
  and the critic as a ``[B/4,3]`` minibatch slice at all 20 Adam steps, and be
  provably read-only with respect to advantage, reward, loss and parameters
  (SDS 4.4/9.3-26);
* missing / mis-shaped / non-finite route is a hard error with the key and the
  consumer named, never a zero fill (SDS 4.7).
"""
import copy
from dataclasses import asdict, replace
import math

import torch
from torch import nn
from torch.distributions import Normal
import unittest

from src.tasks.velocity.v03.ppo import PPO as V03PPO
from src.tasks.velocity.v03.ppo import PPOConfig as V03Config
from src.tasks.velocity.v05_r2 import ppo as module
from src.tasks.velocity.v05_r2 import spec
from src.tasks.velocity.v05_r2.ppo import (ADAM_STEPS_PER_UPDATE, CONTROL_SAMPLES_PER_UPDATE,
                                           PPO, PPOConfig, ROUTE_ERROR_KEY,
                                           explained_variance, measure, physics_substeps,
                                           route_diagnostics, update_math,
                                           validate_config, validate_route_observation)

STD_LOW, STD_HIGH = spec.STD_MIN, spec.STD_MAX


class TinyModel(nn.Module):
    """Minimal model with the two-place std contract; records every route slice."""
    def __init__(self, std=.15):
        super().__init__()
        self.actor_weight = nn.Parameter(torch.tensor(.01))
        self.value_weight = nn.Parameter(torch.tensor(1.))
        self.std = nn.Parameter(torch.full((12,), float(std)))
        self.stats_updates = 0
        self.constrain_calls = 0
        self.actor_routes = []
        self.critic_routes = []

    def forward(self, obs):
        return (self.actor_weight * obs['proprio'][:, :1]).expand(-1, 12)

    def action_distribution(self, obs):
        self.actor_routes.append(obs[ROUTE_ERROR_KEY].detach().clone())
        return Normal(self(obs), self.std.clamp(STD_LOW, STD_HIGH))

    def evaluate(self, obs):
        self.critic_routes.append(obs[ROUTE_ERROR_KEY].detach().clone())
        return self.value_weight * obs['proprio'][:, :1]

    def update_critic_statistics(self, obs):
        self.stats_updates += 1

    @torch.no_grad()
    def constrain_std(self):
        self.constrain_calls += 1
        self.std.clamp_(STD_LOW, STD_HIGH)


class TinyEnv:
    """Stand-in for the r2 env: exactly the five observation keys, route per step."""
    device = 'cpu'
    num_envs = 2

    def __init__(self, route=None, *, with_route=True):
        self.steps = 0
        self.rng = torch.Generator()
        self.with_route = with_route
        self.route_sequence = []

    def observation(self, value=1., route_row=None):
        batch = self.num_envs
        if route_row is None:
            route_row = [0.25, 0.0, 1.0]
        obs = {'proprio': torch.full((batch, 45), float(value)),
               'history': torch.full((batch, 5, 45), float(value)),
               'heightmap': torch.zeros(batch, 3, 2, 2),
               'privileged': torch.zeros(batch, 35)}
        if self.with_route:
            route = torch.tensor([route_row], dtype=torch.float32).repeat(batch, 1)
            # Each control step (and each environment) has a distinct lateral
            # component, so a storage row is traceable to exactly one frame.
            route[:, 0] += (torch.arange(batch, dtype=torch.float32)
                            + .01 * (self.steps % 48))
            obs[ROUTE_ERROR_KEY] = route
        return obs

    def reset(self):
        self.steps = 0
        self.route_sequence = []
        obs = self.observation()
        if self.with_route:
            self.route_sequence.append(obs[ROUTE_ERROR_KEY].clone())
        return obs, {}

    def step(self, actions):
        self.steps += 1
        obs = self.observation()
        terminal = self.observation(value=7., route_row=[-4.0, 0.6, 0.8])
        if self.with_route:
            self.route_sequence.append(obs[ROUTE_ERROR_KEY].clone())
        return (obs, torch.full((self.num_envs,), 2.),
                torch.tensor([False, True]), torch.tensor([True, False]),
                dict(terminal_observation=terminal,
                     episodes=[{'step': self.steps}],
                     reward_terms={'speed': 2., 'stall': -.5},
                     gate_open=torch.tensor([True, False])))


def make_runner(seed=11, env=None):
    torch.manual_seed(seed)
    return PPO(TinyModel(), env or TinyEnv())


class PPOTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


class ConfigTests(PPOTestCase):
    def test_explicit_sds_25_configuration(self):
        expected = dict(num_envs=512, rollout=48, epochs=5, minibatches=4,
                        learning_rate=5e-5, min_lr=1e-5, max_lr=1e-4, desired_kl=.01,
                        gamma=.995, gae_lambda=.95, clip=.2, value_coef=1.,
                        max_grad_norm=1., clipped_value_loss=True, entropy=.005,
                        init_std=.15, std_min=.10, std_max=.30)
        self.assertEqual(asdict(PPOConfig()), expected)
        self.assertEqual(module.SDS_PPO, expected)
        self.assertEqual(spec.PPO_CONTRACT['learning_rate'], 5e-5)
        self.assertEqual(CONTROL_SAMPLES_PER_UPDATE, 48 * 512)
        self.assertEqual(CONTROL_SAMPLES_PER_UPDATE, 24576)
        self.assertEqual(ADAM_STEPS_PER_UPDATE, 20)

    def test_config_cannot_be_overridden_silently(self):
        for override in (dict(gamma=.99), dict(learning_rate=1e-4), dict(std_min=.05),
                         dict(rollout=24), dict(max_lr=3e-4), dict(desired_kl=.02)):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    validate_config(replace(PPOConfig(), **override))
                with self.assertRaises(ValueError):
                    PPO(TinyModel(), TinyEnv(), replace(PPOConfig(), **override))

    def test_unit_conversion_is_three_way_bound(self):
        self.assertEqual(physics_substeps(0), 0)
        self.assertEqual(physics_substeps(CONTROL_SAMPLES_PER_UPDATE),
                         4 * CONTROL_SAMPLES_PER_UPDATE)
        self.assertEqual(physics_substeps(CONTROL_SAMPLES_PER_UPDATE), 98304)
        with self.assertRaises(ValueError):
            physics_substeps(-1)
        with self.assertRaises(ValueError):
            physics_substeps(10, substeps=5)
        instance = make_runner()
        self.assertEqual(instance.control_samples_per_update, 96)  # 48 * 2 envs
        self.assertEqual(instance.physics_substeps_per_update, 384)
        self.assertEqual(instance.learning, dict(update=0, optimizer_step=0,
                                                 control_samples=0, physics_substeps=0))
        instance.collect()
        metrics = instance.update()
        self.assertEqual(metrics['control_samples'], 96)
        self.assertEqual(metrics['physics_substeps'], 384)
        self.assertEqual(instance.learning, dict(update=1, optimizer_step=20,
                                                 control_samples=96, physics_substeps=384))

    def test_adam_is_constructed_with_the_new_initial_learning_rate(self):
        instance = make_runner()
        self.assertEqual(instance.optimizer.param_groups[0]['lr'], 5e-5)
        self.assertFalse(instance.optimizer.state)


class RouteStorageTests(PPOTestCase):
    def test_route_enters_the_rollout_buffer_as_48b3(self):
        env = TinyEnv()
        instance = make_runner(env=env)
        instance.collect()
        route = instance.storage['obs'][ROUTE_ERROR_KEY]
        self.assertEqual(tuple(route.shape), (48 * env.num_envs, 3))
        self.assertEqual(route.dtype, torch.float32)
        # SDS 4.4 item 2: the row at t is exactly what the env returned before
        # the action of step t was sampled (same frame as proprio).
        expected = torch.stack(env.route_sequence[:48], 0).flatten(0, 1)
        self.assertTrue(torch.equal(route, expected))
        self.assertEqual(instance.last_route_diagnostics['samples'], 96)
        self.assertEqual(instance.last_route_diagnostics['nonfinite_count'], 0)

    def test_every_minibatch_slice_reaches_actor_and_critic(self):
        env = TinyEnv()
        torch.manual_seed(21)
        instance = make_runner(seed=21, env=env)
        instance.collect()
        route = instance.storage['obs'][ROUTE_ERROR_KEY].clone()
        instance.model.actor_routes.clear()
        instance.model.critic_routes.clear()
        instance.update()
        self.assertEqual(len(instance.model.actor_routes), 20)
        self.assertEqual(len(instance.model.critic_routes), 20)
        rows = route.shape[0]
        self.assertEqual(rows, 96)
        seen = []
        for actor, critic in zip(instance.model.actor_routes, instance.model.critic_routes):
            self.assertEqual(tuple(actor.shape), (rows // 4, 3))
            self.assertEqual(tuple(critic.shape), (rows // 4, 3))
            # The actor and the critic of one Adam step receive the same slice.
            self.assertTrue(torch.equal(actor, critic))
            for row in actor:
                self.assertIn(row.tolist(), route.tolist())
            seen.extend(actor.tolist())
        # Five epochs cover every rollout row exactly five times.
        self.assertEqual(len(seen), 5 * rows)
        for row in route:
            self.assertEqual(seen.count(row.tolist()), 5)

    def test_route_is_read_only_for_advantage_reward_and_gradients(self):
        # Compare two complete updates whose only difference is the route tensor.
        torch.manual_seed(5)
        runner_a = make_runner(seed=5)
        runner_a.collect()
        torch.manual_seed(5)
        runner_b = make_runner(seed=5)
        runner_b.collect()
        self.assertTrue(torch.equal(runner_a.storage['obs'][ROUTE_ERROR_KEY],
                                    runner_b.storage['obs'][ROUTE_ERROR_KEY]))
        # Only the route tensor changes (scaled/reordered, still in range).
        runner_b.storage['obs'][ROUTE_ERROR_KEY] = (
            runner_b.storage['obs'][ROUTE_ERROR_KEY] * -0.5)
        torch.manual_seed(818)
        metrics_a = runner_a.update()
        torch.manual_seed(818)
        metrics_b = runner_b.update()
        for key, value in runner_a.model.state_dict().items():
            self.assertTrue(torch.equal(value, runner_b.model.state_dict()[key]), key)
        for key in ('loss', 'policy_loss', 'value_loss', 'entropy', 'kl',
                    'gradient_norm', 'clip_fraction', 'lr'):
            self.assertEqual(metrics_a[key], metrics_b[key], key)
        self.assertEqual(metrics_a['reward_terms'], metrics_b['reward_terms'])
        self.assertTrue(torch.equal(runner_a.storage['advantage'],
                                    runner_b.storage['advantage']))
        self.assertTrue(torch.equal(runner_a.storage['returns'],
                                    runner_b.storage['returns']))
        # The route tensors differ, so the storage-derived read-only aggregates
        # differ while the learning math above stayed identical.
        self.assertNotEqual(route_diagnostics(
            runner_a.storage['obs'][ROUTE_ERROR_KEY])['abs_y_mean'],
            route_diagnostics(runner_b.storage['obs'][ROUTE_ERROR_KEY])['abs_y_mean'])

    def test_step_route_aggregation_is_exact_when_local_y_is_exposed(self):
        class ExactEnv(TinyEnv):
            def step(self, actions):
                obs, reward, terminated, truncated, info = super().step(actions)
                local = torch.tensor([[0., .35], [0., -.95]])
                info['diagnostics'] = {'local_position': local,
                                       'heading': torch.tensor([.1, -.2])}
                return obs, reward, terminated, truncated, info

        instance = make_runner(env=ExactEnv())
        instance.collect()
        block = instance.last_route_diagnostics
        # mean(|local_y|) is exact from the environment, not the clipped route.
        self.assertAlmostEqual(block['abs_y_mean'], (0.35 + 0.95) / 2, places=6)
        self.assertAlmostEqual(block['abs_y_max'], .95, places=6)
        self.assertEqual(block['abs_y_source'], 'env_local_y')
        self.assertEqual(block['samples'], 96)

    def test_route_aggregation_marks_the_clipped_reconstruction(self):
        instance = make_runner()
        instance.collect()
        block = instance.last_route_diagnostics
        self.assertEqual(block['abs_y_source'], 'route_component_clipped')
        # The per-step pre-reset terminal route saturates at -4 (0.80 m), which
        # the explicit clipped marker records instead of pretending exactness.
        self.assertAlmostEqual(block['abs_y_max'], .80, places=6)
        self.assertEqual(block['samples'], 96)

    def test_route_diagnostics_match_the_sds_48_definitions(self):
        route = torch.tensor([[0., 0., 1.], [1., 0.5, math.sqrt(.75)],
                              [4., 0., 1.], [-2., 0., -1.]], dtype=torch.float32)
        block = route_diagnostics(route)
        self.assertAlmostEqual(block['abs_y_mean'],
                               (0. + .2 + .8 + .4) / 4., places=7)
        self.assertAlmostEqual(block['abs_y_max'], .8, places=7)
        self.assertAlmostEqual(block['normalized_abs_y_max'], 4., places=7)
        self.assertAlmostEqual(block['saturation_rate'], .25, places=7)
        self.assertAlmostEqual(block['heading_deg_max'], 180., places=6)
        self.assertAlmostEqual(block['heading_deg_mean'],
                               (0. + 30. + 0. + 180.) / 4., places=5)
        self.assertEqual(block['nonfinite_count'], 0)
        with self.assertRaises(ValueError):
            route_diagnostics(torch.tensor([[float('nan'), 0., 1.]]))

    def test_changed_route_does_not_change_initial_actor_mean(self):
        """SDS 2.6 A at the PPO boundary: route cannot leak into the loss."""
        model = TinyModel()
        base = TinyEnv().observation()
        first = model(base).detach().clone()
        altered = {key: value.clone() for key, value in base.items()}
        altered[ROUTE_ERROR_KEY] = torch.tensor([[3.5, -0.4, 0.9],
                                                 [-2.0, 0.2, -0.9]])
        self.assertTrue(torch.equal(first, model(altered).detach()))
        # The value head also ignores route in this controllable stand-in.
        self.assertTrue(torch.equal(model.evaluate(base), model.evaluate(altered)))


class StepWithoutRouteEnv(TinyEnv):
    """Has route at reset but drops it from the post-step observation."""

    def step(self, actions):
        obs = self.observation()
        del obs[ROUTE_ERROR_KEY]
        return (obs, torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool),
                torch.zeros(self.num_envs, dtype=torch.bool),
                dict(terminal_observation=self.observation(value=7.),
                     episodes=[], reward_terms={}))


class MissingRouteTests(PPOTestCase):
    def test_missing_route_in_a_step_observation_is_rejected(self):
        instance = PPO(TinyModel(), StepWithoutRouteEnv())
        with self.assertRaises(ValueError) as caught:
            instance.collect()
        self.assertIn(ROUTE_ERROR_KEY, str(caught.exception))
        self.assertIn('PPO.collect.step', str(caught.exception))

    def test_missing_route_in_the_terminal_observation_is_rejected(self):
        env = TinyEnv()
        instance = make_runner(env=env)
        original = env.step

        def step(actions):
            obs, reward, terminated, truncated, info = original(actions)
            del info['terminal_observation'][ROUTE_ERROR_KEY]
            return obs, reward, terminated, truncated, info

        env.step = step
        with self.assertRaises(ValueError) as caught:
            instance.collect()
        self.assertIn(ROUTE_ERROR_KEY, str(caught.exception))
        self.assertIn('terminal_observation', str(caught.exception))

    def test_missing_route_is_rejected_by_collect_and_update(self):
        env = TinyEnv(with_route=False)
        instance = PPO(TinyModel(), env)
        with self.assertRaises(ValueError) as caught:
            instance.reset()
        self.assertIn(ROUTE_ERROR_KEY, str(caught.exception))
        self.assertIn('PPO.reset', str(caught.exception))

    def test_missing_route_in_the_stored_rollout_is_rejected(self):
        instance = make_runner()
        instance.collect()
        del instance.storage['obs'][ROUTE_ERROR_KEY]
        with self.assertRaises(ValueError) as caught:
            instance.update()
        self.assertIn(ROUTE_ERROR_KEY, str(caught.exception))
        self.assertIn('PPO.update', str(caught.exception))
        self.assertEqual(instance.learning_state['completed_ppo_updates'], 0)

    def test_shape_dtype_batch_and_nonfinite_are_rejected(self):
        base = TinyEnv().observation()
        cases = {
            'shape2': torch.zeros(2, 2),
            'shape4': torch.zeros(2, 4),
            'shape3d': torch.zeros(2, 3, 1),
            'float64': torch.zeros(2, 3, dtype=torch.float64),
            'batch': torch.zeros(3, 3),
            'nan': torch.tensor([[0., float('nan'), 1.], [0., 0., 1.]]),
            'inf': torch.tensor([[0., float('inf'), 1.], [0., 0., 1.]]),
            'range': torch.tensor([[5., 0., 1.], [0., 2., 1.]]),
        }
        for name, route in cases.items():
            with self.subTest(case=name):
                obs = dict(base, **{ROUTE_ERROR_KEY: route})
                with self.assertRaises(ValueError):
                    validate_route_observation(obs, consumer='unit')
        with self.assertRaises(ValueError):
            validate_route_observation(base, consumer='unit', row_count=99)

    def test_measure_requires_a_route_tensor(self):
        rows = [dict(epoch=1, minibatch=index + 1, kl=.001, lr=5e-5,
                     grad_norm_preclip=.5, clip_fraction=.0, loss=1., policy_loss=1.,
                     value_loss=1., entropy=1., std_min=.15, std_max=.15,
                     std_touch_min=0, std_touch_max=0, std_dimensions=12,
                     route_samples=8)
                for index in range(20)]
        storage = dict(value=torch.zeros(8), returns=torch.arange(8, dtype=torch.float64))
        with self.assertRaises(ValueError):
            measure(minibatch_rows=rows, storage=storage, std=torch.full((12,), .15),
                    num_envs=8)


class EquivalentMathTests(PPOTestCase):
    def test_instrumented_update_is_tensor_identical_to_v03(self):
        torch.manual_seed(135)
        reference = V03PPO(TinyModel(std=.20), TinyEnv())
        reference.collect()
        storage = copy.deepcopy(reference.storage)
        reference_model = copy.deepcopy(reference.model)
        candidate_model = copy.deepcopy(reference.model)
        reference.model = reference_model
        reference.optimizer = torch.optim.Adam(reference_model.parameters(),
                                               lr=V03Config().learning_rate)
        candidate_optimizer = torch.optim.Adam(candidate_model.parameters(),
                                               lr=V03Config().learning_rate)
        torch.manual_seed(817)
        expected = reference.update()
        torch.manual_seed(817)
        metrics, diagnostics = update_math(model=candidate_model,
                                           optimizer=candidate_optimizer,
                                           cfg=V03Config(), storage=storage, device='cpu')
        for key, value in candidate_model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.model.state_dict()[key]), key)
        candidate_state = candidate_optimizer.state_dict()['state']
        reference_state = reference.optimizer.state_dict()['state']
        self.assertEqual(set(candidate_state), set(reference_state))
        for index in candidate_state:
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(candidate_state[index][field],
                                            reference_state[index][field]),
                                (index, field))
        self.assertEqual(candidate_optimizer.param_groups[0]['lr'],
                         reference.optimizer.param_groups[0]['lr'])
        self.assertEqual(metrics, {key: expected[key] for key in metrics})
        self.assertEqual(diagnostics['minibatch_count'], 20)

    def test_class_math_is_deterministic_and_shared(self):
        torch.manual_seed(135)
        first = make_runner(seed=135)
        first.collect()
        torch.manual_seed(135)
        second = make_runner(seed=135)
        second.collect()
        self.assertEqual(first.storage['advantage'].tolist(),
                         second.storage['advantage'].tolist())
        torch.manual_seed(817)
        first_metrics = first.update()
        torch.manual_seed(817)
        second_metrics = second.update()
        for key, value in first.model.state_dict().items():
            self.assertTrue(torch.equal(value, second.model.state_dict()[key]), key)
        self.assertEqual(first_metrics, second_metrics)


class DiagnosticsTests(PPOTestCase):
    def test_twenty_minibatch_fields_and_min_mean_max(self):
        instance = make_runner()
        instance.collect()
        metrics = instance.update()
        diagnostics = instance.last_update_diagnostics
        self.assertEqual(diagnostics['minibatch_count'], 20)
        self.assertEqual(diagnostics['adam_steps'], 20)
        self.assertEqual(len(diagnostics['minibatches']), 20)
        epochs = [row['epoch'] for row in diagnostics['minibatches']]
        self.assertEqual(epochs, sorted(epochs))
        self.assertEqual(set(epochs), {1, 2, 3, 4, 5})
        for prefix in ('kl', 'lr', 'grad_norm_preclip', 'clip_fraction'):
            rows = [row[prefix] for row in diagnostics['minibatches']]
            low, mean, high = (diagnostics[f'{prefix}_min'],
                               diagnostics[f'{prefix}_mean'],
                               diagnostics[f'{prefix}_max'])
            self.assertLessEqual(low, mean)
            self.assertLessEqual(mean, high)
            self.assertAlmostEqual(low, min(rows), places=12)
            self.assertAlmostEqual(high, max(rows), places=12)
            self.assertAlmostEqual(mean, sum(rows) / len(rows), places=12)
        self.assertAlmostEqual(diagnostics['clip_fraction_mean'],
                               metrics['clip_fraction'], places=12)
        ratio = diagnostics['gradient_clip_ratio']
        expected_ratio = sum(1 for row in diagnostics['minibatches']
                             if row['grad_norm_preclip'] > 1.) / 20
        self.assertEqual(ratio, expected_ratio)
        self.assertEqual(len(diagnostics['std']), 12)
        self.assertEqual(diagnostics['std_dimensions'], 12)
        self.assertEqual(set(diagnostics['std']), {f'{index:02d}' for index in range(12)})
        self.assertEqual(diagnostics['route']['samples'], 96)
        self.assertEqual(diagnostics['route_error_contract']['key'], ROUTE_ERROR_KEY)
        self.assertEqual(diagnostics['route_error_contract']['shape'], [3])
        # The same field names as r1 (plus the read-only route block).
        for name in ('kl_min', 'kl_mean', 'kl_max', 'lr_min', 'grad_norm_preclip_max',
                     'clip_fraction_mean', 'gradient_clip_ratio', 'value_mean',
                     'return_mean', 'explained_variance', 'std_touch_min_ratio',
                     'std_touch_max_ratio', 'adam_steps', 'control_samples',
                     'physics_substeps'):
            self.assertIn(name, diagnostics)

    def test_value_return_statistics_and_explained_variance(self):
        instance = make_runner()
        instance.collect()
        instance.update()
        diagnostics = instance.last_update_diagnostics
        values = instance.storage['value'].double()
        returns = instance.storage['returns'].double()
        self.assertAlmostEqual(diagnostics['value_mean'], float(values.mean()), places=10)
        self.assertAlmostEqual(diagnostics['return_mean'], float(returns.mean()), places=10)
        expected = 1. - (float(torch.var(returns - values, unbiased=False))
                         / float(torch.var(returns, unbiased=False)))
        self.assertAlmostEqual(diagnostics['explained_variance'], expected, places=10)
        self.assertIsNone(diagnostics['explained_variance_reason'])

    def test_explained_variance_null_when_denominator_is_zero(self):
        value, reason = explained_variance(torch.ones(5), torch.ones(5))
        self.assertIsNone(value)
        self.assertEqual(reason, 'zero_return_variance')
        value, reason = explained_variance([1., 2., 3.], [1., 2., 3.])
        self.assertAlmostEqual(value, 1.)
        self.assertIsNone(reason)
        value, reason = explained_variance([float('nan')], [1.])
        self.assertIsNone(value)
        self.assertEqual(reason, 'nonfinite_value_or_return')
        instance = make_runner()
        instance.collect()
        instance.storage['returns'].fill_(3.)
        instance.storage['value'].fill_(3.)
        instance.update()
        diagnostics = instance.last_update_diagnostics
        self.assertIsNone(diagnostics['explained_variance'])
        self.assertEqual(diagnostics['explained_variance_reason'], 'zero_return_variance')

    def test_kl_warning_does_not_stop_the_twenty_adam_steps(self):
        instance = make_runner()
        instance.collect()
        with torch.no_grad():
            instance.model.actor_weight.mul_(60.)
        with self.assertWarns(RuntimeWarning):
            metrics = instance.update()
        diagnostics = instance.last_update_diagnostics
        self.assertGreater(diagnostics['kl_max'], module.MAX_KL_WARNING)
        self.assertTrue(diagnostics['kl_warning'])
        self.assertEqual(metrics['adam_steps'], 20)
        self.assertEqual(instance.learning_state,
                         dict(optimizer_initialized=True, completed_ppo_updates=1,
                              optimizer_steps=20))
        codes = [item['code'] for item in instance.update_warnings]
        self.assertIn('max_kl_above_0.10', codes)

    def test_nonfinite_is_still_a_hard_error(self):
        instance = make_runner()
        instance.collect()
        instance.storage['returns'][0] = float('nan')
        with self.assertRaises(FloatingPointError):
            instance.update()
        self.assertEqual(instance.learning_state['completed_ppo_updates'], 0)
        self.assertIsNone(instance.last_update_diagnostics)

    def test_first_twenty_updates_keep_all_minibatch_rows(self):
        instance = make_runner()
        for update in range(1, 22):
            instance.collect()
            instance.update()
            self.assertEqual(instance.last_update_diagnostics['update'], update)
        self.assertEqual(len(instance.update_history), 21)
        self.assertEqual(len(instance.minibatch_history), 20)
        self.assertTrue(all(len(rows) == 20 for rows in instance.minibatch_history))
        self.assertNotIn('minibatches', instance.update_history[-1])
        record = instance.diagnostics_jsonl_record()
        self.assertEqual(record['control_samples'], 96)
        self.assertEqual(record['physics_substeps'], 384)
        self.assertNotIn('minibatches', record)
        self.assertEqual(set(record['route']), {
            'abs_y_mean', 'abs_y_max', 'normalized_abs_y_max', 'saturation_rate',
            'heading_deg_mean', 'heading_deg_max', 'nonfinite_count', 'samples',
            'abs_y_source', 'component_scale_m', 'clip'})


class StdTests(PPOTestCase):
    def test_two_place_std_constraint_and_projection_after_each_adam(self):
        instance = make_runner()
        with torch.no_grad():
            instance.model.std.fill_(.95)
        instance.collect()
        metrics = instance.update()
        self.assertEqual(instance.model.constrain_calls, 20)
        self.assertGreaterEqual(float(instance.model.std.detach().min()), STD_LOW - 1e-6)
        self.assertLessEqual(float(instance.model.std.detach().max()), STD_HIGH + 1e-6)
        self.assertGreaterEqual(metrics['std_min'], STD_LOW - 1e-6)
        self.assertLessEqual(metrics['std_max'], STD_HIGH + 1e-6)
        diagnostics = instance.last_update_diagnostics
        self.assertEqual(diagnostics['std_touch_tolerance'], 1e-6)
        self.assertEqual(diagnostics['std_touch_max_count'], 12)
        with torch.no_grad():
            instance.model.std.fill_(9.)
        distribution = instance.model.action_distribution(instance.obs)
        self.assertLessEqual(float(distribution.stddev.detach().max()),
                             STD_HIGH + 1e-6)

    def test_native_r2_model_uses_the_same_two_std_bounds(self):
        from src.tasks.velocity.v05_r2.model import V05R2ActorCritic, synthetic_observations
        torch.manual_seed(42)
        model = V05R2ActorCritic()
        self.assertEqual(module.INITIAL_STD, .15)
        obs = synthetic_observations(batch=4)
        with torch.no_grad():
            model.std.fill_(.05)
            model.constrain_std()
            self.assertGreaterEqual(float(model.std.detach().min()), STD_LOW - 1e-6)
            model.std.fill_(.90)
            model.constrain_std()
            self.assertLessEqual(float(model.std.detach().max()), STD_HIGH + 1e-6)
            self.assertLessEqual(
                float(model.action_distribution(obs).stddev.detach().max()),
                STD_HIGH + 1e-6)

    def test_ppo_has_no_checkpoint_or_early_stop_api(self):
        self.assertFalse(hasattr(PPO, 'save'))
        self.assertFalse(hasattr(PPO, 'load'))
        self.assertFalse(hasattr(PPO, 'early_stop'))
        self.assertIn('no early stop', module.__doc__.lower())


if __name__ == '__main__':
    unittest.main()
