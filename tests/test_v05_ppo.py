"""CPU-only v0.5 PPO measurement tests.

The v0.5 adaptor may not change the frozen PPO mathematics, so the central test
compares the instrumented update loop with ``v03.ppo.PPO.update`` under fixed
storage and a fixed RNG seed: parameters, Adam moments, learning state and every
shared metric must be identical.  Everything else checks the read-only SDS 7.5
diagnostics (20 minibatch rows, min/mean/max, gradient clipping ratio,
value/return statistics, explained variance with a null reason, warning-only
KL/clip thresholds) and the two-place std projection.
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
from src.tasks.velocity.v05 import ppo as module
from src.tasks.velocity.v05.ppo import (ADAM_STEPS_PER_UPDATE, CONTROL_SAMPLES_PER_UPDATE,
                                        PPO, PPOConfig, explained_variance, measure,
                                        physics_substeps, update_math, validate_config)

STD_LOW, STD_HIGH = .10, .30


class TinyModel(nn.Module):
    """Minimal model with the v0.5 two-place std constraint contract."""
    def __init__(self, std=.15):
        super().__init__()
        self.actor_weight = nn.Parameter(torch.tensor(.01))
        self.value_weight = nn.Parameter(torch.tensor(1.))
        self.std = nn.Parameter(torch.full((12,), float(std)))
        self.stats_updates = 0
        self.constrain_calls = 0

    def forward(self, obs):
        return (self.actor_weight * obs['proprio'][:, :1]).expand(-1, 12)

    def action_distribution(self, obs):
        return Normal(self(obs), self.std.clamp(STD_LOW, STD_HIGH))

    def evaluate(self, obs):
        return self.value_weight * obs['proprio'][:, :1]

    def update_critic_statistics(self, obs):
        self.stats_updates += 1

    @torch.no_grad()
    def constrain_std(self):
        self.constrain_calls += 1
        self.std.clamp_(STD_LOW, STD_HIGH)


class TinyEnv:
    device = 'cpu'
    num_envs = 2

    def __init__(self):
        self.steps = 0
        self.rng = torch.Generator()

    def observation(self, value=1.):
        return dict(proprio=torch.full((2, 45), float(value)),
                    heightmap=torch.zeros(2, 3, 2, 2))

    def reset(self):
        self.steps = 0
        return self.observation(), {}

    def step(self, actions):
        self.steps += 1
        return (self.observation(), torch.full((2,), 2.),
                torch.tensor([False, True]), torch.tensor([True, False]),
                dict(terminal_observation=self.observation(7),
                     episodes=[{'step': self.steps}],
                     reward_terms={'speed': 2., 'stall': -.5},
                     gate_open=torch.tensor([True, False])))


def make_runner(seed=11):
    torch.manual_seed(seed)
    return PPO(TinyModel(), TinyEnv())


class PPOTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


class ConfigTests(PPOTestCase):
    def test_explicit_sds_22_configuration(self):
        expected = dict(num_envs=512, rollout=48, epochs=5, minibatches=4,
                        learning_rate=5e-5, min_lr=1e-5, max_lr=1e-4, desired_kl=.01,
                        gamma=.995, gae_lambda=.95, clip=.2, value_coef=1.,
                        max_grad_norm=1., clipped_value_loss=True, entropy=.005,
                        init_std=.15, std_min=.10, std_max=.30)
        self.assertEqual(asdict(PPOConfig()), expected)
        self.assertEqual(module.SDS_PPO, expected)
        self.assertEqual(CONTROL_SAMPLES_PER_UPDATE, 48 * 512)
        self.assertEqual(CONTROL_SAMPLES_PER_UPDATE, 24576)
        self.assertEqual(ADAM_STEPS_PER_UPDATE, 20)

    def test_config_cannot_be_overridden_silently(self):
        for override in (dict(gamma=.99), dict(learning_rate=1e-4), dict(std_min=.05),
                         dict(rollout=24), dict(max_lr=3e-4)):
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


class EquivalenceTests(PPOTestCase):
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
        # update_math returns the frozen core metrics; ``V03PPO.update`` adds the
        # rollout metrics on top, so the shared keys must match exactly.
        self.assertEqual(metrics, {key: expected[key] for key in metrics})
        self.assertEqual(diagnostics['minibatch_count'], 20)

    def test_v05_class_math_is_deterministic_and_shared(self):
        torch.manual_seed(135)
        first = PPO(TinyModel(std=.20), TinyEnv())
        first.collect()
        torch.manual_seed(135)
        second = PPO(TinyModel(std=.20), TinyEnv())
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

    def test_class_update_runs_the_measured_loop(self):
        torch.manual_seed(4)
        instance = PPO(TinyModel(), TinyEnv())
        instance.collect()
        storage = copy.deepcopy(instance.storage)
        torch.manual_seed(99)
        expected, diagnostics = update_math(
            model=instance.model, optimizer=instance.optimizer, cfg=instance.cfg,
            storage=storage, device='cpu', learning_state=instance.learning_state,
            notify=instance._notify_update)
        torch.manual_seed(4)
        other = PPO(TinyModel(), TinyEnv())
        other.collect()
        torch.manual_seed(99)
        metrics = other.update()
        for key, value in instance.model.state_dict().items():
            self.assertTrue(torch.equal(value, other.model.state_dict()[key]), key)
        for key in expected:
            self.assertEqual(metrics[key], expected[key], key)
        self.assertEqual(other.last_update_diagnostics['minibatch_count'], 20)
        self.assertEqual(other.last_update_diagnostics['kl_min'], diagnostics['kl_min'])


class MeasurementTests(PPOTestCase):
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
        # Equal minibatches: the mean of per-minibatch fractions equals the
        # sample-weighted update fraction that v0.3 reports.
        self.assertAlmostEqual(diagnostics['clip_fraction_mean'],
                               metrics['clip_fraction'], places=12)
        ratio = diagnostics['gradient_clip_ratio']
        self.assertGreaterEqual(ratio, 0.)
        self.assertLessEqual(ratio, 1.)
        expected_ratio = sum(1 for row in diagnostics['minibatches']
                             if row['grad_norm_preclip'] > 1.) / 20
        self.assertEqual(ratio, expected_ratio)
        self.assertEqual(len(diagnostics['std']), 12)
        self.assertEqual(diagnostics['std_dimensions'], 12)
        self.assertEqual(set(diagnostics['std']), {f'{index:02d}' for index in range(12)})

    def test_value_return_statistics_and_explained_variance(self):
        instance = make_runner()
        instance.collect()
        instance.update()
        diagnostics = instance.last_update_diagnostics
        values = instance.storage['value'].double()
        returns = instance.storage['returns'].double()
        self.assertAlmostEqual(diagnostics['value_mean'], float(values.mean()), places=10)
        self.assertAlmostEqual(diagnostics['return_mean'], float(returns.mean()), places=10)
        self.assertAlmostEqual(diagnostics['value_std'],
                               float(values.std(unbiased=False)), places=10)
        self.assertAlmostEqual(diagnostics['return_std'],
                               float(returns.std(unbiased=False)), places=10)
        self.assertEqual(diagnostics['value_min'], float(values.min()))
        self.assertEqual(diagnostics['return_max'], float(returns.max()))
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
        value, reason = explained_variance([5., 5., 5.], [1., 2., 3.])
        self.assertAlmostEqual(value, 0.)
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
        self.assertEqual(diagnostics['return_std'], 0.)

    def test_clip_fraction_threshold_warns_without_tuning(self):
        rows = [dict(epoch=1, minibatch=index + 1, kl=.001, lr=5e-5,
                     grad_norm_preclip=.5, clip_fraction=.8, loss=1., policy_loss=1.,
                     value_loss=1., entropy=1., std_min=.15, std_max=.15,
                     std_touch_min=0, std_touch_max=0, std_dimensions=12)
                for index in range(20)]
        storage = dict(value=torch.zeros(8), returns=torch.arange(8, dtype=torch.float64))
        diagnostics = measure(minibatch_rows=rows, storage=storage,
                              std=torch.full((12,), .15), num_envs=8)
        self.assertTrue(diagnostics['clip_fraction_warning'])
        self.assertFalse(diagnostics['kl_warning'])
        self.assertEqual(diagnostics['warning_codes'], ['mean_clip_fraction_above_0.50'])
        self.assertEqual(diagnostics['gradient_clip_ratio'], 0.)

    def test_large_kl_warns_but_all_twenty_adam_steps_run(self):
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
        self.assertTrue(all(item['update'] == 1 for item in instance.update_warnings))
        self.assertEqual(instance.cfg, PPOConfig())
        self.assertEqual(len(diagnostics['minibatches']), 20)

    def test_nonfinite_is_still_a_hard_error(self):
        instance = make_runner()
        instance.collect()
        instance.storage['returns'][0] = float('nan')
        with self.assertRaises(FloatingPointError):
            instance.update()
        self.assertEqual(instance.learning_state['completed_ppo_updates'], 0)
        self.assertIsNone(instance.last_update_diagnostics)

    def test_per_minibatch_gradient_preclip_norm_and_clip_ratio(self):
        instance = make_runner()
        instance.collect()
        instance.update()
        rows = instance.last_update_diagnostics['minibatches']
        self.assertTrue(all(row['grad_norm_preclip'] >= 0. for row in rows))
        self.assertTrue(all(math.isfinite(row['grad_norm_preclip']) for row in rows))
        ratio = instance.last_update_diagnostics['gradient_clip_ratio']
        self.assertEqual(ratio,
                         sum(1 for row in rows if row['grad_norm_preclip'] > 1.) / 20)

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
        self.assertEqual(instance.update_history[-1]['minibatch_count'], 20)
        record = instance.diagnostics_jsonl_record()
        self.assertEqual(record['control_samples'], 96)
        self.assertEqual(record['physics_substeps'], 384)
        self.assertNotIn('minibatches', record)


class StdAndGateTests(PPOTestCase):
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
        self.assertEqual(diagnostics['std_touch_max_ratio'], 1.)
        with torch.no_grad():
            instance.model.std.fill_(9.)
        distribution = instance.model.action_distribution(instance.obs)
        self.assertLessEqual(float(distribution.stddev.detach().max()),
                             STD_HIGH + 1e-6)

    def test_native_v05_model_uses_the_same_two_std_bounds(self):
        try:
            from src.tasks.velocity.v05.model import V05ActorCritic
        except ImportError:  # the model is implemented by another work package
            self.skipTest('v05.model is not implemented in this work package')
        torch.manual_seed(42)
        model = V05ActorCritic()
        # The 0.15 initial value belongs to the ETH migration (SDS 2.1 step 5);
        # the bare constructor value is not the v0.5 claim under test here.
        self.assertEqual(module.INITIAL_STD, .15)
        with torch.no_grad():
            model.std.fill_(.05)
            model.constrain_std()
            self.assertGreaterEqual(float(model.std.detach().min()), STD_LOW - 1e-6)
            model.std.fill_(.90)
            model.constrain_std()
            self.assertLessEqual(float(model.std.detach().max()), STD_HIGH + 1e-6)
            self.assertLessEqual(
                float(model.action_distribution({
                    'proprio': torch.zeros(1, 45),
                    'history': torch.zeros(1, 5, 45),
                    'heightmap': torch.zeros(1, 3, 48, 32),
                    'privileged': torch.zeros(1, 35)}).stddev.detach().max()),
                STD_HIGH + 1e-6)

    def test_gate_rate_and_reward_contribution_are_read_only(self):
        instance = make_runner()
        instance.collect()
        metrics = instance.update()
        self.assertAlmostEqual(metrics['gate_open_rate'], .5)
        self.assertEqual(instance.last_update_diagnostics['gate_samples'], 48 * 2)
        self.assertEqual(metrics['reward_terms'], {'speed': 2., 'stall': -.5})
        self.assertEqual(metrics['map_metrics'],
                         {'map_valid_fraction': 0., 'map_age_mean': 0.})
        self.assertAlmostEqual(metrics['speed'], 2.)
        self.assertAlmostEqual(metrics['stall'], -.5)

    def test_diagnostics_do_not_change_the_learning_state_stream(self):
        instance = make_runner()
        events = []
        instance.update_callback = lambda event, payload: events.append((event, payload))
        instance.collect()
        instance.update()
        self.assertEqual([event for event, _ in events],
                         ['optimizer_started'] + ['optimizer_step_completed'] * 20
                         + ['update_completed'])
        for event, payload in events:
            # The callback payload set is exactly the v0.3 one: no diagnostic field.
            self.assertLessEqual(set(payload), {'learning_state',
                                                'expected_optimizer_steps', 'epoch_index',
                                                'minibatch_index',
                                                'optimizer_steps_in_update'})
            self.assertEqual(set(payload['learning_state']),
                             {'update', 'optimizer_step', 'control_samples',
                              'physics_substeps'})
        self.assertEqual(events[0][1]['learning_state']['optimizer_step'], 0)
        self.assertEqual(events[-1][1]['learning_state']['optimizer_step'], 20)
        self.assertEqual([payload['learning_state']['optimizer_step']
                          for event, payload in events[1:21]], list(range(1, 21)))

    def test_ppo_has_no_checkpoint_or_early_stop_api(self):
        self.assertFalse(hasattr(PPO, 'save'))
        self.assertFalse(hasattr(PPO, 'load'))
        self.assertFalse(hasattr(PPO, 'early_stop'))
        self.assertIn('no early stop', module.__doc__.lower())


if __name__ == '__main__':
    unittest.main()
