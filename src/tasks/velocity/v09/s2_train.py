"""S2.1 PPO training stack (CPU/testable; real GPU env injected later)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from . import s2_checkpoint, s2_loss, s2_runner, s2_spec, s2_train_env
from .cases import sha256

CHECKPOINT_EVERY_SAMPLES = 5_000_000
S2_1_SAMPLES_PER_ROLLOUT = 1024 * s2_runner.ROLLOUT_STEPS
S2_1_MAX_ROLLOUTS = s2_spec.complete_rollouts(
    s2_spec.S2_COURSE_LEARNING_CAPS['S2_1'], 1024)
S2_1_CHECKPOINT_ROLLOUTS = math.ceil(CHECKPOINT_EVERY_SAMPLES / S2_1_SAMPLES_PER_ROLLOUT)


def rollout_plan(num_envs=1024, course='S2.1'):
    course = s2_spec.course_key(course)
    cap = s2_spec.S2_COURSE_LEARNING_CAPS[course]
    samples = num_envs * s2_runner.ROLLOUT_STEPS
    return dict(num_envs=num_envs, rollout_steps=s2_runner.ROLLOUT_STEPS,
                samples_per_rollout=samples,
                max_rollouts=s2_spec.complete_rollouts(cap, num_envs),
                checkpoint_every_rollouts=math.ceil(
                    CHECKPOINT_EVERY_SAMPLES / samples),
                learning_cap=cap)


def _state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(
            value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def flatten_time_major_old_logs(transitions):
    return torch.stack([row['log_probs'] for row in transitions], 0).reshape(-1)


def _obs_batch(obs, index, device):
    return {key: torch.as_tensor(value[index:index + 1], dtype=torch.float32,
                                 device=device) for key, value in obs.items()}


class S2LearningBudget:
    """One reserve→commit (or release) per full 1024×48 rollout."""

    def __init__(self, ledger, *, obligation_id='learning_S2_1',
                 samples_per_rollout=S2_1_SAMPLES_PER_ROLLOUT):
        self.ledger = ledger
        self.obligation_id = obligation_id
        self.samples_per_rollout = int(samples_per_rollout)
        self.rollouts = 0

    def begin(self, attempt_id):
        self.ledger.reserve(
            self.obligation_id, self.samples_per_rollout,
            operation_id=f'{attempt_id}:rollout{self.rollouts}:reserve',
            reason='S2.1 learning rollout')
        return self.samples_per_rollout

    def finish(self, attempt_id, actual_samples):
        actual = int(actual_samples)
        if actual < 0 or actual > self.samples_per_rollout:
            raise ValueError('actual samples outside rollout reservation')
        base = f'{attempt_id}:rollout{self.rollouts}'
        if actual:
            self.ledger.commit(self.obligation_id, actual,
                               operation_id=f'{base}:commit',
                               reason='S2.1 learning actual samples')
        unused = self.samples_per_rollout - actual
        if unused:
            self.ledger.release(self.obligation_id, unused,
                                operation_id=f'{base}:release',
                                reason='S2.1 learning unused reservation')
        self.rollouts += 1
        return dict(committed=actual, released=unused)


class S2Trainer:
    """PPO update/core with logical minibatches and one Adam step each."""

    def __init__(self, *, model, env, ledger=None, output_dir=None,
                 device='cpu', attempt_id='s2.1', learning_rate=3e-5,
                 teacher=None, beta_flat=s2_loss.BETA_FLAT,
                 ledger_run_id=None, course='S2.1'):
        course = s2_spec.course_key(course)
        self.model = model.to(device)
        self.env = env
        self.device = device
        self.ledger = ledger
        self.output_dir = Path(output_dir) if output_dir else None
        self.attempt_id = attempt_id
        self.course = course.replace('_', '.')
        self.ledger_run_id = ledger_run_id or 's2.1'
        self.teacher = teacher.to(device).eval() if teacher is not None else None
        self.beta_flat = float(beta_flat)
        self.model.train()
        self.model.set_s1_course('S1.2')
        self.model.std.requires_grad_(True)
        self.model.actor.normalizer.eval()
        self.model.critic_normalizer.eval()
        self.optimizer = torch.optim.Adam(
            self.model.optimizer_parameters(), lr=float(learning_rate),
            betas=s2_runner.ADAM_BETAS, eps=s2_runner.ADAM_EPS,
            weight_decay=s2_runner.WEIGHT_DECAY)
        self.learning_rate = float(learning_rate)
        self.rollouts = 0
        self.learning_samples = 0
        self.evaluation_samples = 0
        self.engineering_samples = 0
        self.rollback_samples = 0
        self.gates_passed = []
        self.gate_uses = 0
        self.final_uses = 0
        self.engineering_rerun_reasons = []
        self.random_state = None
        self._obs = None
        self._rollout_steps = 0
        if self.teacher is not None:
            self.teacher.requires_grad_(False)

    def _policy_obs(self, obs, index):
        return _obs_batch(obs, index, self.device)

    def collect_rollout(self):
        # Episodes persist across rollout boundaries.  The env resets only the
        # environments whose episode terminates, not the whole batch.
        self._rollout_steps = 0
        if self._obs is None:
            self._obs = self.env.reset()
        obs = self._obs
        transitions = []
        n = self.env.num_envs
        for _ in range(s2_runner.ROLLOUT_STEPS):
            item = {key: value for key, value in obs.items()}
            dist = self.model.action_distribution(item)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(-1)
            values = self.model.evaluate(item).squeeze(-1)
            next_obs, reward, done, info = self.env.step(actions.detach())
            transitions.append(dict(obs=obs, actions=actions,
                                    log_probs=log_probs, values=values,
                                    reward=reward, done=done,
                                    terminated=bool(info.get('terminated', False))
                                    if hasattr(info, 'get') else False,
                                    truncated=bool(info.get('truncated', False))
                                    if hasattr(info, 'get') else False,
                                    terminated_mask=(info.get('terminated_mask')
                                                     if hasattr(info, 'get') else None),
                                    truncated_mask=(info.get('truncated_mask')
                                                    if hasattr(info, 'get') else None),
                                    failure_mask=(info.get('failure_mask')
                                                  if hasattr(info, 'get') else None),
                                    timeout_mask=(info.get('timeout_mask')
                                                  if hasattr(info, 'get') else None),
                                    failure_class=(info.get('failure_class')
                                                   if hasattr(info, 'get') else None),
                                    teacher_obs=(info.get('teacher_observation')
                                                 if hasattr(info, 'get') else None)))
            obs = next_obs
            self._rollout_steps += 1
        self._obs = obs
        rewards = torch.stack([row['reward'] for row in transitions], 1)
        values = torch.stack([row['values'] for row in transitions], 1)
        done = torch.stack([row['done'] for row in transitions], 1)
        def _mask(name, fallback):
            rows = [row.get(name) for row in transitions]
            if any(value is None for value in rows):
                return fallback
            return torch.stack([value if torch.is_tensor(value)
                                else torch.as_tensor(value, device=self.device)
                                for value in rows], 1).to(self.device)
        terminated = _mask('terminated_mask', torch.zeros_like(done))
        truncated = _mask('truncated_mask', torch.zeros_like(done))
        failure = _mask('failure_mask', torch.zeros_like(done))
        timeout = _mask('timeout_mask', torch.zeros_like(done))
        # Non-terminal sampling boundary bootstraps from V(next_obs); terminal
        # rows keep the zero bootstrap in GAE.
        with torch.no_grad():
            next_values = self.model.evaluate(
                {key: value for key, value in self._obs.items()}).squeeze(-1)
        adv, returns = [], []
        for index in range(n):
            a, r = s2_runner.gae(
                rewards[index].detach().cpu().numpy(),
                values[index].detach().cpu().numpy(),
                terminated=terminated[index].cpu().numpy(),
                timeout=(truncated[index] | failure[index] | timeout[index]).cpu().numpy(),
                sampling_boundary=np.ones(len(transitions), dtype=bool),
                last_value=(0.0 if bool(done[index, -1]) else
                            float(next_values[index].detach().cpu())))
            adv.append(torch.as_tensor(a, dtype=torch.float32, device=self.device))
            returns.append(torch.as_tensor(r, dtype=torch.float32, device=self.device))
        # [N,T] -> [T,N] so flattening matches time-major transitions.
        return (transitions,
                torch.stack(adv, 1).transpose(0, 1).contiguous(),
                torch.stack(returns, 1).transpose(0, 1).contiguous())

    def update(self, transitions, advantages, returns):
        old_logs = flatten_time_major_old_logs(transitions)
        # flatten logical rollout; keep one logical batch for KL/LR
        n = self.env.num_envs
        role_rows = (getattr(self.env, 'role_plan', {}) or {}).get('envs') or []
        flat = [str(row.get('task', '')).startswith('flat_') for row in role_rows]
        if len(flat) != n:
            flat = [False] * n
        clean = s2_loss.clean_flat_mask(dict(
            ordinary_flat=flat, nominal_dynamics=[True] * n,
            no_noise=[True] * n, no_delay=[True] * n, no_force=[True] * n))
        clean_env = torch.tensor(clean, dtype=torch.bool, device=self.device)
        for epoch in range(s2_runner.EPOCHS):
            # deterministic logical minibatches over flattened samples
            flat_obs = {}
            for key in transitions[0]['obs']:
                flat_obs[key] = torch.cat([row['obs'][key] for row in transitions], 0)
            flat_teacher = {}
            if self.teacher is not None and transitions[0].get('teacher_obs') is not None:
                for key in transitions[0]['teacher_obs']:
                    flat_teacher[key] = torch.cat(
                        [row['teacher_obs'][key] for row in transitions], 0)
            flat_actions = torch.stack([row['actions'] for row in transitions], 0).reshape(-1, 12)
            flat_old = old_logs
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)
            # complete logical minibatch determines lr before gradient.
            # Chunk only for memory; the KL/lr decision still uses all samples.
            new_log = []
            for start in range(0, flat_actions.shape[0], s2_runner.MICROBATCH_MAX):
                chunk = slice(start, start + s2_runner.MICROBATCH_MAX)
                item = {key: value[chunk] for key, value in flat_obs.items()}
                with torch.no_grad():
                    logp = self.model.action_distribution(item)                        .log_prob(flat_actions[chunk]).sum(-1)
                new_log.append(logp)
            kl = s2_runner.approximate_kl(
                flat_old.detach().cpu().numpy(), torch.cat(new_log).detach().cpu().numpy())
            self.learning_rate = s2_runner.adjust_learning_rate(self.learning_rate, kl)
            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rate
            flat_clean = clean_env.unsqueeze(0).expand(
                len(transitions), n).reshape(-1)
            n_clean = int(flat_clean.sum())
            order = torch.arange(flat_actions.shape[0], device=self.device)
            per = math.ceil(len(order) / s2_runner.MINIBATCHES)
            for start in range(0, len(order), per):
                index = order[start:start + per]
                self.optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, len(index), s2_runner.MICROBATCH_MAX):
                    micro = index[micro_start:micro_start + s2_runner.MICROBATCH_MAX]
                    item = {key: value[micro] for key, value in flat_obs.items()}
                    actions = flat_actions[micro]
                    dist = self.model.action_distribution(item)
                    logp = dist.log_prob(actions).sum(-1)
                    ratio = torch.exp(logp - flat_old[micro].detach())
                    unclipped = ratio * flat_adv[micro]
                    clipped = torch.clamp(ratio, 1 - s2_runner.CLIP,
                                          1 + s2_runner.CLIP) * flat_adv[micro]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value = self.model.evaluate(item).squeeze(-1)
                    value_loss = (value - flat_ret[micro]).pow(2).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = policy_loss + s2_runner.VALUE_COEF * value_loss \
                        - s2_runner.ENTROPY_COEF * entropy
                    scaled_loss = loss * (len(micro) / len(index))
                    if self.teacher is not None and n_clean:
                        clean_micro = micro[flat_clean[micro]]
                        if len(clean_micro):
                            clean_item = {key: value[clean_micro]
                                          for key, value in flat_obs.items()}
                            teacher_item = ({key: value[clean_micro]
                                             for key, value in flat_teacher.items()}
                                            if flat_teacher else clean_item)
                            mu_student = self.model(clean_item)
                            with torch.no_grad():
                                mu_teacher = self.teacher(teacher_item).detach()
                            anchor = torch.sum((mu_student - mu_teacher) ** 2) / (12.0 * n_clean)
                            scaled_loss = scaled_loss + self.beta_flat * anchor
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                          s2_runner.GRAD_CLIP)
                self.optimizer.step()
                self.model.constrain_std()
        self.rollouts += 1
        self.learning_samples += self.env.num_envs * s2_runner.ROLLOUT_STEPS
        return dict(rollouts=self.rollouts, learning_rate=self.learning_rate,
                    std=self.model.std.detach().cpu().tolist())

    def train_rollout(self, budget=None):
        if budget is not None:
            budget.begin(self.attempt_id)
        try:
            transitions, adv, ret = self.collect_rollout()
            result = self.update(transitions, adv, ret)
            if budget is not None:
                budget.finish(self.attempt_id,
                              self.env.num_envs * s2_runner.ROLLOUT_STEPS)
            return result
        except Exception:
            if budget is not None:
                env = getattr(self, 'env', None)
                actual = (int(getattr(env, 'num_envs', 0))
                          * int(getattr(self, '_rollout_steps', 0)))
                budget.finish(self.attempt_id, actual)
            raise

    def checkpoint_metadata(self, *, case_manifest_sha256='0' * 64,
                            manifest_sha256='0' * 64, code_sha256=None):
        code_sha256 = code_sha256 or hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()
        return dict(
            stage='S2', course=self.course,
            curriculum=dict(role=self.course, role_counts=self.env.role_plan.get('counts'),
                            speeds=list(s2_train_env.SPEEDS),
                            platform_widths=list(s2_train_env.PLATFORM_WIDTHS)),
            strength='nominal', model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            policy_std=self.model.std.detach().cpu().tolist(),
            normalizer_state=self.model.actor.normalizer.state_dict(),
            teacher_source=dict(sha256=s2_spec.S1_ENTRY_WEIGHT_SHA256),
            beta_flat=self.beta_flat,
            learning_samples=self.learning_samples,
            evaluation_samples=self.evaluation_samples,
            engineering_samples=self.engineering_samples,
            rollback_samples=self.rollback_samples,
            gates_passed=list(self.gates_passed),
            case_manifest_sha256=case_manifest_sha256,
            candidate_sha256=_state_sha(self.model),
            gate_ab_sequence='gate_A',
            gate_uses=self.gate_uses, final_uses=self.final_uses,
            engineering_rerun_reasons=list(self.engineering_rerun_reasons),
            random_state=self.random_state or dict(torch=torch.get_rng_state().tolist()),
            role_assignment=self.env.role_plan,
            randomization={'profile': 'nominal', 'force_family': 'none',
                           'noise': 'none'},
            command_flow={'vx': list(s2_train_env.SPEEDS), 'vy': 0.0, 'wz': 0.0,
                          'post_distance_m': s2_train_env.POST_DISTANCE},
            delay_config={'control_delay_steps': 0, 'profile': 'nominal'},
            entry_weight=dict(sha256=s2_spec.S1_ENTRY_WEIGHT_SHA256),
            ledger_run_id=self.ledger_run_id,
            ledger_cursor=dict(learning_samples=self.learning_samples,
                               rollouts=self.rollouts),
            code_sha256=code_sha256, manifest_sha256=manifest_sha256)

    def save(self, path, **metadata_kwargs):
        metadata = self.checkpoint_metadata(**metadata_kwargs)
        return s2_checkpoint.save_checkpoint(
            path, self.model.state_dict(), self.optimizer.state_dict(), metadata)

    def resume(self, path, *, device='cpu'):
        from . import s2_r1
        s2_r1.reject_invalid_checkpoint(path)
        payload = s2_checkpoint.load_checkpoint(path, device=device)
        self.model.load_state_dict(payload['model'])
        self.optimizer.load_state_dict(payload['optimizer'])
        std = payload['metadata']['policy_std']
        self.model.std.data.copy_(torch.as_tensor(std, dtype=self.model.std.dtype,
                                                 device=self.model.std.device))
        self.learning_rate = float(self.optimizer.param_groups[0]['lr'])
        self.learning_samples = int(payload['metadata']['learning_samples'])
        self.rollouts = self.learning_samples // (
            self.env.num_envs * s2_runner.ROLLOUT_STEPS)
        return payload['metadata']


def gate_groups():
    """S2.1 36 groups: flat 9 + stairs 12 + platform 12 + L40 3."""
    groups = []
    for speed in s2_train_env.SPEEDS:
        for mode in ('walk', 'stop_instant', 'stop_ramp_2s'):
            groups.append(dict(id=f'flat/{mode}/{speed}', task='flat',
                               speed=speed, mode=mode))
    for levels in s2_train_env.STAIR_COUNTS:
        for width in s2_train_env.PLATFORM_WIDTHS:
            for speed in s2_train_env.SPEEDS:
                groups.append(dict(id=f'stairs/{levels}/{width}/{speed}',
                                   task='stairs_walk', stair_count=levels,
                                   platform_width=width, speed=speed,
                                   post_distance=s2_train_env.POST_DISTANCE))
    for height in s2_train_env.PLATFORM_HEIGHTS:
        for width in s2_train_env.PLATFORM_WIDTHS:
            for speed in s2_train_env.SPEEDS:
                groups.append(dict(id=f'platform/{height}/{width}/{speed}',
                                   task='platform_walk', platform_height=height,
                                   platform_width=width, speed=speed,
                                   post_distance=s2_train_env.POST_DISTANCE))
    for speed in s2_train_env.SPEEDS:
        groups.append(dict(id=f'l40/{speed}', task='legacy_l40',
                           stair_count=40, speed=speed,
                           post_distance=s2_train_env.POST_DISTANCE,
                           cache_suites=['l40_milestone_cache_gate_A',
                                         'l40_milestone_cache_gate_B']))
    if len(groups) != 36:
        raise ValueError('S2.1 gate matrix must contain 36 groups')
    return groups


def gate_plan(cases_per_group=20, threshold=18):
    groups = gate_groups()
    return dict(groups=len(groups), cases_per_group=int(cases_per_group),
                threshold=int(threshold), suites=['gate_A', 'gate_B'],
                total_cases=len(groups) * int(cases_per_group) * 2,
                gate_admission='gate_admissions_up_to_40',
                l40_cache=['l40_milestone_cache_gate_A',
                           'l40_milestone_cache_gate_B'])


def evaluate_gate_results(results, *, threshold=18):
    """Validate 36×2 group results; 18/20 or better passes."""
    seen = {}
    for row in results:
        key = (row.get('group'), row.get('suite'))
        if key in seen:
            raise ValueError(f'duplicate gate result {key}')
        seen[key] = row
    expected = {(group['id'], suite) for group in gate_groups()
                for suite in ('gate_A', 'gate_B')}
    if set(seen) != expected:
        raise ValueError('gate results do not cover 36 groups × A/B')
    failures = []
    for key, row in seen.items():
        successes = int(row.get('successes', 0))
        total = int(row.get('total', 20))
        if total != 20 or successes < int(threshold):
            failures.append(dict(group=key[0], suite=key[1],
                                 successes=successes, total=total))
    return dict(groups=36, suites=2, total=len(seen),
                passed=not failures, failures=failures)
