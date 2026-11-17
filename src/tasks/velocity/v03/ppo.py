"""Checkpoint-free PPO core with read-only control/optimizer ledger hooks.

V03 runner owns save/load, attempts, durable ledgers and all recovery decisions.
A failed env.step may already have advanced physics: absence of a completed-step
callback never proves zero samples. Callbacks propagate failures to the owner.
"""
import numpy as np
import torch
from src.tasks.velocity.rl.ideal_stairs_runner import PPO as LegacyPPO, PPOConfig, finite


class PPO:
    """Frozen PPO mathematics, with no inheritance of old checkpoint protocols.

    Hooks (all optional and synchronous):
      control_step_callback(event): after env.step returns, before another step;
        event = {step_index: 1..48, num_envs, physics_samples_increment, info}.
      diagnostics_callback(info): the same unmodified environment info.
      update_callback(name, payload): optimizer_started before any Adam/LR
        mutation; optimizer_step_completed after each Adam step/std projection;
        update_completed only after all 20 Adam steps. Payload contains a copied
        learning_state and relevant step counts. These hooks never write files.
    """
    def __init__(self, model, env, cfg=PPOConfig()):
        LegacyPPO.__init__(self, model, env, cfg)
        self.control_step_callback = None
        self.diagnostics_callback = None
        self.update_callback = None
        self.learning_state = dict(optimizer_initialized=False,
                                   completed_ppo_updates=0, optimizer_steps=0)

    def reset(self):
        LegacyPPO.reset(self)

    def _notify_update(self, event, **payload):
        if self.update_callback is not None:
            self.update_callback(event, dict(learning_state=dict(self.learning_state), **payload))

    def collect(self):
        if self.obs is None:
            self.reset()
        c = self.cfg
        n = self.env.num_envs
        device = self.env.device
        obs = {k: torch.empty((c.rollout, *v.shape), dtype=v.dtype, device=device)
               for k, v in self.obs.items()}
        fields = {k: torch.empty((c.rollout, n), device=device)
                  for k in ('logp', 'value', 'reward', 'terminated', 'done', 'next_value')}
        actions = torch.empty((c.rollout, n, 12), device=device)
        means = torch.empty_like(actions)
        stds = torch.empty_like(actions)
        records = []
        reward_sums = {}
        self.model.train()
        # Exactly once before rollout; frozen throughout all 20 Adam minibatches.
        self.model.update_critic_statistics(self.obs)
        for t in range(c.rollout):
            with torch.no_grad():
                for key, value in self.obs.items():
                    obs[key][t].copy_(value)
                dist = self.model.action_distribution(self.obs)
                action = dist.sample()
                finite('actions', action)
                value = self.model.evaluate(self.obs).squeeze(-1)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if self.control_step_callback is not None:
                    self.control_step_callback(dict(step_index=t + 1, num_envs=n,
                        physics_samples_increment=n, info=info))
                if self.diagnostics_callback is not None:
                    self.diagnostics_callback(info)
                terminal_obs = info.get('terminal_observation', next_obs)
                next_value = self.model.evaluate(terminal_obs).squeeze(-1)
                for key, val in [('logp', dist.log_prob(action).sum(-1)), ('value', value),
                        ('reward', reward), ('terminated', terminated),
                        ('done', terminated | truncated), ('next_value', next_value)]:
                    fields[key][t].copy_(val)
                actions[t].copy_(action)
                means[t].copy_(dist.mean)
                stds[t].copy_(dist.stddev)
                records.extend(info.get('episodes', []))
                self.obs = next_obs
                for key, val in info.get('reward_terms', {}).items():
                    reward_sums[key] = reward_sums.get(key, 0.) + val
        advantage = torch.zeros_like(fields['reward'])
        last = torch.zeros(n, device=device)
        for t in reversed(range(c.rollout)):
            delta = (fields['reward'][t] + c.gamma * (1 - fields['terminated'][t])
                     * fields['next_value'][t] - fields['value'][t])
            last = delta + c.gamma * c.gae_lambda * (1 - fields['done'][t]) * last
            advantage[t] = last
        returns = advantage + fields['value']
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        self.storage = dict(obs={k: v.flatten(0, 1) for k, v in obs.items()},
            actions=actions.flatten(0, 1), means=means.flatten(0, 1), stds=stds.flatten(0, 1),
            advantage=advantage.flatten(), returns=returns.flatten(),
            **{k: v.flatten() for k, v in fields.items()})
        self.rollout_metrics = {key: val / c.rollout for key, val in reward_sums.items()}
        self.rollout_metrics.update(map_valid_fraction=float(self.obs['heightmap'][:, 1].mean()),
                                    map_age_mean=float(self.obs['heightmap'][:, 2].mean()))
        return records

    def update(self):
        c = self.cfg
        s = self.storage
        if s is None:
            raise RuntimeError('Missing rollout')
        batch = len(s['advantage'])
        if batch % c.minibatches:
            raise ValueError('Rollout not divisible into minibatches')
        result = []
        clipped_samples = 0
        total_samples = 0
        # Durable owner must acknowledge the uncertain-optimizer boundary before
        # even adaptive LR changes Adam param_groups; exceptions abort here.
        self._notify_update('optimizer_started', expected_optimizer_steps=c.epochs * c.minibatches)
        for epoch in range(c.epochs):
            for minibatch, idx in enumerate(torch.randperm(batch, device=self.env.device).chunk(c.minibatches)):
                obs = {k: v[idx] for k, v in s['obs'].items()}
                dist = self.model.action_distribution(obs)
                value = self.model.evaluate(obs).squeeze(-1)
                with torch.no_grad():
                    kl = (torch.log(dist.stddev / s['stds'][idx])
                        + (s['stds'][idx] ** 2 + (s['means'][idx] - dist.mean) ** 2)
                        / (2 * dist.stddev ** 2) - .5).sum(-1).mean()
                    finite('KL', kl)
                    lr = self.optimizer.param_groups[0]['lr']
                    if kl > 2 * c.desired_kl:
                        lr = max(c.min_lr, lr / 1.5)
                    elif 0 < kl < c.desired_kl / 2:
                        lr = min(c.max_lr, lr * 1.5)
                    for group in self.optimizer.param_groups:
                        group['lr'] = lr
                ratio = torch.exp(dist.log_prob(s['actions'][idx]).sum(-1) - s['logp'][idx])
                policy = torch.maximum(-s['advantage'][idx] * ratio,
                    -s['advantage'][idx] * ratio.clamp(1 - c.clip, 1 + c.clip)).mean()
                clipped = s['value'][idx] + (value - s['value'][idx]).clamp(-c.clip, c.clip)
                value_loss = torch.maximum((value - s['returns'][idx]) ** 2,
                                            (clipped - s['returns'][idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy + c.value_coef * value_loss - c.entropy * entropy
                finite('loss', loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None:
                        finite(f'gradient:{name}', parameter.grad)
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                finite('gradient norm', grad)
                self.optimizer.step()
                self.learning_state['optimizer_steps'] += 1
                self.learning_state['optimizer_initialized'] = True
                self.model.constrain_std()
                self._notify_update('optimizer_step_completed', epoch_index=epoch + 1,
                                    minibatch_index=minibatch + 1)
                clipped_samples += int(((ratio.detach() - 1).abs() > c.clip).sum())
                total_samples += idx.numel()
                result.append([float(loss.detach()), float(policy.detach()), float(value_loss.detach()),
                               float(entropy.detach()), float(kl), float(grad)])
        self.learning_state['completed_ppo_updates'] += 1
        self._notify_update('update_completed', optimizer_steps_in_update=c.epochs * c.minibatches)
        values = np.mean(result, axis=0)
        std = self.model.std.detach()
        return dict(zip(('loss', 'policy_loss', 'value_loss', 'entropy', 'kl', 'gradient_norm'), values.tolist()),
            clip_fraction=clipped_samples / total_samples,
            lr=self.optimizer.param_groups[0]['lr'], std_mean=float(std.mean()),
            std_min=float(std.min()), std_max=float(std.max()),
            **{f'std_{i}': float(v) for i, v in enumerate(std)}, **self.rollout_metrics)
