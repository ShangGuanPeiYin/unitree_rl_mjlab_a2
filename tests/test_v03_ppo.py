"""Small CPU rollout tests exercise real PPO and Adam, without a simulator."""
import copy
import importlib
import unittest

import torch
from torch import nn
from torch.distributions import Normal
from src.tasks.velocity.rl.ideal_stairs_runner import PPO as LegacyPPO, PPOConfig
from src.tasks.velocity.v02_formal.runner import PPO as FormalPPO

try:
    target = importlib.import_module('src.tasks.velocity.v03.ppo')
except ModuleNotFoundError:
    target = None


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_weight = nn.Parameter(torch.tensor(.01))
        self.value_weight = nn.Parameter(torch.tensor(1.))
        self.std = nn.Parameter(torch.full((12,), .2))
        self.stats_updates = 0
    def forward(self, obs):
        return (self.actor_weight * obs['proprio'][:, :1]).expand(-1, 12)
    def action_distribution(self, obs):
        return Normal(self(obs), self.std.clamp(.05, .30))
    def evaluate(self, obs):
        return self.value_weight * obs['proprio'][:, :1]
    def update_critic_statistics(self, obs):
        self.stats_updates += 1
    @torch.no_grad()
    def constrain_std(self):
        self.std.clamp_(.05, .30)


class TinyEnv:
    device = 'cpu'
    num_envs = 2
    def __init__(self):
        self.steps = 0
    def observation(self, value):
        return dict(proprio=torch.full((2, 45), float(value)), heightmap=torch.zeros(2, 3, 2, 2))
    def reset(self):
        return self.observation(1), {}
    def step(self, actions):
        self.steps += 1
        # env0 timeout, env1 real termination; reset observations intentionally differ.
        return (self.observation(1), torch.full((2,), 2.), torch.tensor([False, True]),
                torch.tensor([True, False]), dict(terminal_observation=self.observation(7),
                episodes=[{'step': self.steps}], reward_terms={'controlled': 2.}))


class PublicAPI(unittest.TestCase):
    def test_v03_ppo_core_exists_without_old_checkpoint_protocol(self):
        self.assertIsNotNone(target, 'V3 PPO core is not implemented')
        self.assertFalse(issubclass(target.PPO, LegacyPPO))
        self.assertFalse(hasattr(target.PPO, 'save'))
        self.assertFalse(hasattr(target.PPO, 'load'))


@unittest.skipIf(target is None, 'V3 implementation absent during RED')
class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def runner(self):
        return target.PPO(TinyModel(), TinyEnv())

    def test_fixed_configuration_and_fresh_empty_adam(self):
        runner = self.runner()
        self.assertEqual(runner.cfg, PPOConfig())
        self.assertFalse(runner.optimizer.state)
        self.assertEqual(runner.learning_state, dict(optimizer_initialized=False, completed_ppo_updates=0, optimizer_steps=0))
        from dataclasses import replace
        with self.assertRaises(ValueError):
            target.PPO(TinyModel(), TinyEnv(), replace(PPOConfig(), gamma=.98))

    def test_timeout_terminal_bootstrap_and_actual_step_callbacks(self):
        runner = self.runner()
        events, diagnostics = [], []
        def observe(event):
            self.assertEqual(runner.env.steps, event['step_index'])
            events.append(event)
        runner.control_step_callback = observe
        runner.diagnostics_callback = diagnostics.append
        records = runner.collect()
        self.assertEqual(len(events), 48)
        self.assertEqual(len(diagnostics), 48)
        self.assertEqual(len(records), 48)
        self.assertEqual(sum(e['physics_samples_increment'] for e in events), 96)
        self.assertTrue(all(e['num_envs'] == 2 for e in events))
        self.assertIs(events[0]['info'], diagnostics[0])
        self.assertEqual(runner.model.stats_updates, 1)
        self.assertTrue(torch.allclose(runner.storage['returns'][::2], torch.full((48,), 8.93), atol=1e-6))
        self.assertTrue(torch.equal(runner.storage['returns'][1::2], torch.full((48,), 2.)))
        runner.reset()
        self.assertIsNone(runner.storage)

    def test_gae_continues_across_live_steps_and_stops_at_timeout(self):
        runner = self.runner()
        def live_step(actions):
            runner.env.steps += 1
            done = runner.env.steps == 48
            return (runner.env.observation(1), torch.full((2,), 2.), torch.zeros(2, dtype=torch.bool),
                    torch.full((2,), done), dict(terminal_observation=runner.env.observation(7 if done else 1)))
        runner.env.step = live_step
        runner.collect()
        # Hand recurrence: delta=1.99; final timeout delta=7.93, multiplier=.9405.
        expected = 7.93
        for _ in range(47): expected = 1.99 + .9405 * expected
        self.assertAlmostEqual(runner.storage['returns'][0].item(), expected + 1., places=4)

    def test_twenty_actual_adam_steps_and_callback_order(self):
        runner = self.runner()
        runner.collect()
        before = runner.model.actor_weight.detach().clone()
        events = []
        def observed(event, payload):
            if event == 'optimizer_started':
                self.assertFalse(runner.optimizer.state)
                self.assertEqual(runner.learning_state['optimizer_steps'], 0)
            events.append((event, copy.deepcopy(payload)))
        runner.update_callback = observed
        metrics = runner.update()
        self.assertEqual([e for e, _ in events], ['optimizer_started'] + ['optimizer_step_completed'] * 20 + ['update_completed'])
        self.assertEqual(runner.learning_state, dict(optimizer_initialized=True, completed_ppo_updates=1, optimizer_steps=20))
        self.assertTrue(all(v['step'].item() == 20 for v in runner.optimizer.state.values()))
        self.assertFalse(torch.equal(before, runner.model.actor_weight))
        self.assertGreaterEqual(metrics['clip_fraction'], 0.)
        self.assertLessEqual(metrics['clip_fraction'], 1.)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(torch.all((runner.model.std >= .05) & (runner.model.std <= .30)))

    def test_update_math_matches_frozen_formal_core_and_actual_sample_clip(self):
        runner = self.runner()
        torch.manual_seed(135)
        runner.collect()
        # Deliberately heterogeneous ratios: counting abs(ratio-1)>clip cannot be
        # reconstructed from the ratio mean. Shift half the behavior logp.
        runner.storage['logp'][::2] += 1.
        reference = FormalPPO(copy.deepcopy(runner.model), TinyEnv())
        reference.storage = copy.deepcopy(runner.storage)
        reference.rollout_metrics = dict(runner.rollout_metrics)
        counts = []
        original_distribution = runner.model.action_distribution
        def recorded(obs):
            dist = original_distribution(obs)
            original_log_prob = dist.log_prob
            def log_prob(action):
                result = original_log_prob(action)
                counts.append(result.detach().sum(-1))
                return result
            dist.log_prob = log_prob
            return dist
        runner.model.action_distribution = recorded
        permutations = []
        original_randperm = torch.randperm
        from unittest.mock import patch
        def capture_permutation(*args, **kwargs):
            result = original_randperm(*args, **kwargs)
            permutations.append(result.clone())
            return result
        torch.manual_seed(817)
        with patch.object(target.torch, 'randperm', side_effect=capture_permutation):
            metrics = runner.update()
        torch.manual_seed(817)
        expected = reference.update()
        for key, value in runner.model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.model.state_dict()[key]), key)
        for key in ('loss', 'policy_loss', 'value_loss', 'entropy', 'kl', 'gradient_norm', 'lr'):
            self.assertEqual(metrics[key], expected[key], key)
        clipped, total = 0, 0
        for epoch, order in enumerate(permutations):
            for mini, index in enumerate(order.chunk(4)):
                ratio = torch.exp(counts[epoch * 4 + mini] - runner.storage['logp'][index])
                clipped += int(((ratio - 1).abs() > .2).sum())
                total += len(index)
        self.assertEqual(metrics['clip_fraction'], clipped / total)
        self.assertGreater(metrics['clip_fraction'], .1)

    def test_failed_step_or_optimizer_notification_never_claims_completion(self):
        runner = self.runner()
        events = []
        runner.control_step_callback = events.append
        def failed(actions):
            runner.env.steps += 1
            raise RuntimeError('simulator failed after possibly advancing physics')
        runner.env.step = failed
        with self.assertRaises(RuntimeError): runner.collect()
        self.assertEqual(events, [])  # Ledger must interpret this as unknown, not zero physics.
        self.assertEqual(runner.learning_state['completed_ppo_updates'], 0)
        runner = self.runner()
        runner.collect()
        def reject(event, payload):
            raise RuntimeError('durable ledger unavailable')
        runner.update_callback = reject
        with self.assertRaises(RuntimeError): runner.update()
        self.assertFalse(runner.optimizer.state)
        self.assertEqual(runner.learning_state['completed_ppo_updates'], 0)

    def test_partial_adam_failure_preserves_partial_steps_without_complete_event(self):
        runner = self.runner()
        runner.collect()
        events = []
        def stop_after_third(event, payload):
            events.append(event)
            if event == 'optimizer_step_completed' and payload['learning_state']['optimizer_steps'] == 3:
                raise RuntimeError('interrupted after third Adam step')
        runner.update_callback = stop_after_third
        with self.assertRaises(RuntimeError): runner.update()
        self.assertEqual(runner.learning_state, dict(optimizer_initialized=True, completed_ppo_updates=0, optimizer_steps=3))
        self.assertNotIn('update_completed', events)
        self.assertTrue(all(v['step'].item() == 3 for v in runner.optimizer.state.values()))

    def test_nonfinite_loss_and_gradient_fail_without_complete_update(self):
        runner = self.runner()
        runner.collect()
        runner.storage['returns'][0] = float('nan')
        with self.assertRaises(FloatingPointError): runner.update()
        self.assertEqual(runner.learning_state['completed_ppo_updates'], 0)
        runner = self.runner()
        runner.collect()
        runner.model.actor_weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        with self.assertRaises(FloatingPointError): runner.update()
        self.assertEqual(runner.learning_state['completed_ppo_updates'], 0)
