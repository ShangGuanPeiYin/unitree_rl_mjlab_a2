"""S2 PPO configuration/math and CPU dry-run plan (§2.4/Q021 gated)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from . import s2_spec

ROLLOUT_STEPS = 48
EPOCHS = 5
MINIBATCHES = 4
MICROBATCH_MAX = 512
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
VALUE_COEF = 1.0
ENTROPY_COEF = 0.003
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
LR_INITIAL = 3e-5
LR_MIN = 1e-5
LR_MAX = 1e-4
KL_HIGH = 0.02
KL_LOW = 0.005
STD_MIN = 0.05
STD_MAX = 1.0
GRAD_CLIP = 1.0
ENV_CANDIDATES = s2_spec.S2_ENV_CANDIDATES


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = ROLLOUT_STEPS
    epochs: int = EPOCHS
    minibatches: int = MINIBATCHES
    microbatch_max: int = MICROBATCH_MAX
    gamma: float = GAMMA
    lam: float = LAMBDA
    clip: float = CLIP
    value_coef: float = VALUE_COEF
    entropy_coef: float = ENTROPY_COEF
    learning_rate: float = LR_INITIAL
    lr_min: float = LR_MIN
    lr_max: float = LR_MAX
    std_min: float = STD_MIN
    std_max: float = STD_MAX
    grad_clip: float = GRAD_CLIP


def adjust_learning_rate(current, kl):
    if isinstance(kl, bool) or not isinstance(kl, (int, float)) or not math.isfinite(kl):
        raise ValueError('KL must be finite')
    if kl < 0:
        raise ValueError('KL must be nonnegative')
    if kl > KL_HIGH:
        updated = current / 1.5
    elif 0 < kl < KL_LOW:
        updated = current * 1.5
    else:
        updated = current
    return min(LR_MAX, max(LR_MIN, updated))


def project_std(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('std must be a number')
    if not math.isfinite(value):
        raise ValueError('std must be finite')
    return min(STD_MAX, max(STD_MIN, float(value)))


def normalize_advantage(advantage):
    values = np.asarray(advantage, dtype=np.float64)
    if values.size == 0:
        raise ValueError('empty advantage')
    return (values - values.mean()) / (values.std() + 1e-8)


def clipped_ppo_loss(ratio, advantage, *, clip=CLIP):
    ratio = np.asarray(ratio, dtype=np.float64)
    advantage = np.asarray(advantage, dtype=np.float64)
    if ratio.shape != advantage.shape:
        raise ValueError('ratio/advantage shape mismatch')
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * advantage
    return -float(np.mean(np.minimum(unclipped, clipped)))


def value_loss(prediction, target, *, clip=CLIP, old_prediction=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError('value shape mismatch')
    if old_prediction is None:
        return float(np.mean((prediction - target) ** 2))
    old = np.asarray(old_prediction, dtype=np.float64)
    clipped = old + np.clip(prediction - old, -clip, clip)
    return float(np.mean(np.maximum((prediction - target) ** 2,
                                    (clipped - target) ** 2)))


def gae(rewards, values, *, terminated, timeout, sampling_boundary,
        last_value=0.0, gamma=GAMMA, lam=LAMBDA):
    """Task failure/timeout bootstrap 0; only the sampler boundary bootstraps."""
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    done = np.asarray(terminated, dtype=bool) | np.asarray(timeout, dtype=bool)
    boundary = np.asarray(sampling_boundary, dtype=bool)
    if not (r.shape == v.shape == done.shape == boundary.shape):
        raise ValueError('GAE arrays must share shape')
    n = len(r)
    advantages = np.zeros(n)
    last = 0.0
    for t in range(n - 1, -1, -1):
        non_terminal = 0.0 if done[t] else 1.0
        if t == n - 1:
            next_value = float(last_value) if boundary[t] and not done[t] else 0.0
            delta = r[t] + gamma * next_value - v[t]
        else:
            delta = r[t] + gamma * v[t + 1] * non_terminal - v[t]
        last = delta + gamma * lam * non_terminal * last
        advantages[t] = last
    returns = advantages + v
    return advantages, returns


def weighted_accumulate(rows):
    """Weight micro-batch gradients by sample count; one logical mean."""
    total = 0.0
    count = 0
    for gradient, samples in rows:
        if samples <= 0:
            raise ValueError('sample count must be positive')
        total += np.asarray(gradient, dtype=np.float64) * samples
        count += samples
    return total / count if count else None


def logical_minibatch_plan(num_samples, *, minibatches=MINIBATCHES,
                           microbatch_max=MICROBATCH_MAX):
    if num_samples <= 0:
        raise ValueError('positive sample count required')
    per = math.ceil(num_samples / minibatches)
    micro = math.ceil(per / microbatch_max)
    return dict(logical_size=per, microbatches=micro,
                micro_sizes=[min(microbatch_max, per - i * microbatch_max)
                             for i in range(micro)],
                adam_updates_per_epoch=minibatches,
                epochs=EPOCHS)


def dry_run_plan(*, num_envs=1024):
    if num_envs not in ENV_CANDIDATES:
        raise ValueError('S2 dry-run supports 1024 or 512')
    caps = s2_spec.S2_COURSE_LEARNING_CAPS
    return dict(
        stage='S2', sds_sha256=s2_spec.SDS_SHA256, num_envs=num_envs,
        samples_per_rollout=num_envs * ROLLOUT_STEPS,
        config=asdict(PPOConfig()),
        learning_caps=caps,
        complete_rollouts={course: s2_spec.complete_rollouts(cap, num_envs)
                           for course, cap in caps.items()},
        std_unfreeze_from='S2.1', std_range=[STD_MIN, STD_MAX],
        new_adam_at='S2.1', physics_started=False, training_started=False,
        note='CPU dry-run only; execution requires Q021 reservation and a GPU window')


def gaussian_entropy(std, *, action_dim=12):
    std = np.asarray(std, dtype=np.float64)
    if np.any(std <= 0):
        raise ValueError('std must be positive')
    return float(0.5 * action_dim * (1.0 + math.log(2.0 * math.pi))
                 + np.sum(np.log(std)))


def approximate_kl(old_log_prob, new_log_prob):
    old = np.asarray(old_log_prob, dtype=np.float64)
    new = np.asarray(new_log_prob, dtype=np.float64)
    if old.shape != new.shape or old.size == 0:
        raise ValueError('KL inputs must share a nonempty shape')
    log_ratio = new - old
    return float(np.mean((np.exp(log_ratio) - 1.0) - log_ratio))


def assert_frozen(*, teacher_eval, teacher_grad_enabled, normalizer_updates,
                  stage='S2'):
    if not teacher_eval:
        raise ValueError('teacher must stay in eval mode')
    if teacher_grad_enabled:
        raise ValueError('teacher must have no gradients')
    if normalizer_updates != 0:
        raise ValueError('student normalizer buffers are frozen in S2')
    if stage not in ('S2', 'S2.1', 'S2.2', 'S2.3'):
        raise ValueError('frozen contract applies to S2 stages')
    return True


class PPOTrainerState:
    """Minimal logical-minibatch update state for CPU numeric tests."""

    def __init__(self, *, learning_rate=LR_INITIAL, std=0.3, course='S2.0'):
        self.learning_rate = float(learning_rate)
        self.std = project_std(std)
        self.course = course
        self.adam_id = 'adam_s2.1' if course == 'S2.1' else 'adam_s2.0'
        self.optimizer_steps = 0

    def logical_minibatch_update(self, *, old_log_prob, new_log_prob,
                                 microbatch_gradients):
        kl = approximate_kl(old_log_prob, new_log_prob)
        self.learning_rate = adjust_learning_rate(self.learning_rate, kl)
        accumulated = weighted_accumulate(microbatch_gradients)
        self.optimizer_steps += 1
        return dict(kl=kl, learning_rate=self.learning_rate,
                    gradient=accumulated, optimizer_steps=self.optimizer_steps,
                    adam_updates=1)

    def switch_course(self, course):
        if course not in ('S2.1', 'S2.2', 'S2.3'):
            raise ValueError('unknown S2 course')
        previous = (self.learning_rate, self.std, self.course)
        self.course = course
        if course == 'S2.1':
            self.adam_id = 'adam_s2.1'
        self.learning_rate = previous[0]
        self.std = previous[1]
        return dict(previous=previous, learning_rate=self.learning_rate,
                    std=self.std, adam_id=self.adam_id,
                    reset_learning_rate=False, reset_std=False)
