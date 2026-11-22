"""CPU-only end-to-end route plumbing: env obs -> storage -> actor/critic -> evidence.

The simulated r2 environment is GPU-only (SDS 9.2), so this test uses a
faithful lightweight stand-in that implements the *route semantics* the real
``V05R2Env`` must implement:

* every returned observation carries ``route_error [B,3] float32``;
* ``info['terminal_observation']`` is the pre-reset route while the returned
  ``obs`` is the post-reset route of the new episode;
* per-episode evidence carries ``route_final`` (pre-reset) and ``route_initial``
  (post-reset).

The model, the PPO loop and the runner are the real r2 implementations, so the
chain under test is the real one: env observation -> rollout storage
``[48*B,3]`` -> minibatch slice ``[B/4,3]`` into ``actor.route_fusion`` and into
the last three columns of the 211-input critic -> read-only ``route/*``
diagnostics emitted with the training update.
"""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from src.tasks.velocity.v05_r2 import ppo as ppo_module
from src.tasks.velocity.v05_r2 import spec
from src.tasks.velocity.v05_r2.model import V05R2ActorCritic, synthetic_observations
from src.tasks.velocity.v05_r2.ppo import PPO
from src.tasks.velocity.v05_r2.recovery import AttemptLedger
from src.tasks.velocity.v05_r2.runner import Curriculum, V05Runner

ROUTE = ppo_module.ROUTE_ERROR_KEY


class RouteEvidenceEnv:
    """Faithful five-key route stand-in (CPU) with pre/post-reset semantics."""
    device = 'cpu'

    def __init__(self, num_envs=2, seed=7, batch=2):
        self.num_envs = int(num_envs)
        self.rng = torch.Generator().manual_seed(seed)
        self.call = 0
        self.reset_count = 0
        self.stage = 'P1'
        self.stage_iteration = 0
        self.frames = []          # every observation returned (reset + steps)
        self.terminals = []       # every pre-reset terminal observation
        self.reset_initial = []   # route_initial of each episode
        self.episodes = []        # episode evidence records

    # --------------------------------------------------------- observations
    def _base(self):
        obs = synthetic_observations(batch=self.num_envs)
        return {key: value.detach().clone() for key, value in obs.items()}

    def _set_route(self, obs, value):
        route = torch.zeros(self.num_envs, 3)
        route[:, 0] = value + .01 * torch.arange(self.num_envs)
        route[:, 1] = torch.sin(torch.tensor(.05 * value))
        route[:, 2] = torch.cos(torch.tensor(.05 * value))
        obs[ROUTE] = route
        return obs

    def _obs(self, value):
        return self._set_route(self._base(), value)

    # ------------------------------------------------------------- lifecycle
    def set_stage(self, stage, stage_iteration=0, previous_stage=None):
        self.stage = stage
        self.stage_iteration = int(stage_iteration)

    def reset(self):
        self.reset_count += 1
        obs = self._obs(.02)
        self.frames.append(obs[ROUTE].clone())
        self.reset_initial.append(obs[ROUTE].clone())
        return obs, {}

    def step(self, action):
        self.call += 1
        # Terminal route is the pre-reset frame of the current episode; the
        # returned observation is the next episode's post-reset route.
        terminal = self._obs(3.9)
        terminal_route = terminal[ROUTE].clone()
        self.terminals.append(terminal_route.clone())
        next_obs = self._obs(.02 + .01 * (self.call % 40))
        self.frames.append(next_obs[ROUTE].clone())
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        probe = float(terminal_route[:, 0].abs().max())
        self.episodes.append(dict(
            slot_class='current', duration=.02,
            route_final=terminal_route[0].tolist(),
            route_initial=self.reset_initial[-1][0].tolist(),
            route_max_abs_y=probe * .20,
            route_max_abs_heading_deg=float(
                torch.rad2deg(torch.atan2(terminal_route[0, 1],
                                          terminal_route[0, 2])).abs()),
            reward_sums={'reward_total': -1.}))
        return next_obs, torch.full((self.num_envs,), -.5), zeros, zeros, {
            'terminal_observation': terminal,
            'episodes': [self.episodes[-1]],
            'reward_terms': {'center': -.1},
        }


class RoutePlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        torch.manual_seed(42)
        self.model = V05R2ActorCritic()
        self.env = RouteEvidenceEnv(num_envs=2)
        self.ppo = PPO(self.model, self.env)
        self.actor_inputs = []
        self.critic_inputs = []
        self.ppo.model.actor.route_fusion.register_forward_hook(
            lambda module, inputs, output: self.actor_inputs.append(
                inputs[0].detach().clone()))
        self.ppo.model.critic[0].register_forward_hook(
            lambda module, inputs, output: self.critic_inputs.append(
                inputs[0].detach().clone()))
        torch.manual_seed(1234)
        self.records = self.ppo.collect()

    def test_env_observation_reaches_the_rollout_buffer_unchanged(self):
        stored = self.ppo.storage['obs'][ROUTE]
        self.assertEqual(tuple(stored.shape), (48 * 2, 3))
        self.assertEqual(stored.dtype, torch.float32)
        expected = torch.stack(self.env.frames[:48], 0).flatten(0, 1)
        self.assertTrue(torch.equal(stored, expected))
        # The route is a per-control-step observation: 48 distinct frames, one
        # write each, no repeat/cache (SDS 4.4 item 2).
        self.assertFalse(torch.equal(self.env.frames[0], self.env.frames[1]))

    def test_terminal_and_post_reset_routes_are_distinct_and_both_reach_the_critic(self):
        # During collect the critic saw both the per-step route and the pre-reset
        # terminal route (3.9 offset) through its last three input columns.
        seen = torch.cat([value[:, 208:211] for value in self.critic_inputs], 0)
        pre_reset = torch.stack(self.env.terminals, 0).flatten(0, 1)
        for row in pre_reset:
            self.assertTrue(any(torch.equal(row, candidate) for candidate in seen))
        self.assertTrue(any(bool((row[0] > 3.).item()) for row in seen))
        # Returned observation after the automatic reset is the new episode's
        # route, not the pre-reset terminal value (SDS 4.5).
        post_reset = self.ppo.obs[ROUTE]
        self.assertTrue(torch.equal(post_reset, self.env.frames[-1]))
        self.assertFalse(torch.equal(post_reset, self.env.terminals[-1]))
        self.assertEqual(self.env.episodes[-1]['route_final'],
                         self.env.terminals[-1][0].tolist())
        self.assertNotEqual(self.env.episodes[-1]['route_final'],
                            post_reset[0].tolist())

    def test_actor_and_critic_receive_the_same_minibatch_route_slices(self):
        stored = self.ppo.storage['obs'][ROUTE].clone()
        self.actor_inputs.clear()
        self.critic_inputs.clear()
        torch.manual_seed(99)
        self.ppo.update()
        self.assertEqual(len(self.actor_inputs), 20)
        self.assertEqual(len(self.critic_inputs), 20)
        for actor, critic in zip(self.actor_inputs, self.critic_inputs):
            self.assertEqual(tuple(actor.shape), (24, 3))
            # The 211-input critic appends route_error at [208:211] (SDS 2.4).
            self.assertTrue(torch.equal(critic[:, 208:211], actor))
            self.assertEqual(critic.shape[1], 211)
            for row in actor:
                self.assertIn(row.tolist(), stored.tolist())
        self.assertEqual(self.ppo.model.actor.route_fusion.in_features, 3)
        self.assertEqual(self.ppo.model.critic[0].in_features, 211)
        self.assertEqual(self.ppo.model.critic[0].out_features, 512)

    def test_route_never_enters_the_normalizers(self):
        before = {name: value.clone() for name, value in
                  self.ppo.model.state_dict().items()
                  if name.startswith(('actor.normalizer.', 'critic_normalizer.'))}
        self.ppo.collect()
        torch.manual_seed(7)
        self.ppo.update()
        after = self.ppo.model.state_dict()
        actor_before = self.ppo.model.actor.normalizer
        self.assertEqual(actor_before._mean.numel(), 45)
        self.assertEqual(self.ppo.model.critic_normalizer.mean.numel(), 80)
        # The ETH actor normalizer is frozen; the critic normalizer only sees the
        # 80 proprio+privileged inputs once per rollout.
        self.assertTrue(torch.equal(before['actor.normalizer._mean'],
                                    after['actor.normalizer._mean']))
        self.assertEqual(self.ppo.model.critic_normalizer.mean.numel(), 80)

    def test_route_diagnostics_flow_into_the_runner_train_event_and_episode_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(42)
            model = V05R2ActorCritic()
            env = RouteEvidenceEnv(num_envs=2)
            ppo = PPO(model, env)
            ppo.obs, _ = env.reset()
            ledger = AttemptLedger(directory, run_id='r2-plumbing', mode='formal')
            curriculum = Curriculum(ledger, mode='formal', seed=42)
            emitted = []

            class Capture:
                def emit(self, kind, metrics):
                    emitted.append((kind, metrics))

            runner = V05Runner(model=model, env=env, ppo=ppo, curriculum=curriculum,
                               ledger=ledger, device='cpu', seed=42,
                               output_dir=Path(directory) / 'run',
                               save_checkpoints=False, final_test=False,
                               telemetry=Capture())
            runner.reset_episode()
            runner.train_step()
            train = [metrics for kind, metrics in emitted if kind == 'train']
            self.assertEqual(len(train), 1)
            route = train[0]['route']
            self.assertEqual(route['nonfinite_count'], 0)
            self.assertGreaterEqual(route['abs_y_max'], 0.)
            record = ppo.diagnostics_jsonl_record()
            self.assertEqual(record['route']['nonfinite_count'], 0)
            self.assertEqual(record['physics_substeps'], 4 * record['control_samples'])
            # The episode evidence keeps both the pre-reset final and post-reset
            # initial route so the raw route is traceable outside the checkpoint.
            episode = env.episodes[-1]
            self.assertEqual(len(episode['route_final']), 3)
            self.assertEqual(len(episode['route_initial']), 3)
            self.assertEqual(episode['route_final'],
                             env.terminals[-1][0].tolist())
            self.assertEqual(episode['route_initial'],
                             env.reset_initial[-1][0].tolist())
            # The checkpoint policy state contains the r2 bypass tensors.
            state = runner.policy_state
            self.assertEqual(tuple(state['actor.route_fusion.weight'].shape), (512, 3))
            self.assertEqual(tuple(state['actor.route_fusion.bias'].shape), (512,))
            self.assertEqual(tuple(state['critic.0.weight'].shape), (512, 211))
            self.assertEqual(ppo.learning['update'], 1)
            self.assertEqual(ppo.learning['physics_substeps'],
                             4 * ppo.learning['control_samples'])


if __name__ == '__main__':
    unittest.main()
