"""CPU-only §5.4 loss and §2.4 runner tests."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_loss, s2_runner


def test_anchor_exact_aggregation_and_zero_clean():
    student = np.zeros((3, 12))
    teacher = np.ones((3, 12))
    clean = np.array([True, False, True])
    assert s2_loss.flat_anchor_loss(student, teacher, clean) == pytest.approx(1.0)
    assert s2_loss.flat_anchor_loss(student, teacher, np.zeros(3, dtype=bool)) == 0.0
    assert s2_loss.BETA_FLAT == 0.1


def test_anchor_microbatch_accumulation_equals_single_batch():
    rng = np.random.default_rng(0)
    student = rng.normal(size=(7, 12))
    teacher = rng.normal(size=(7, 12))
    clean = np.array([True, True, False, True, False, True, True])
    single = s2_loss.flat_anchor_loss(student, teacher, clean)
    accumulator = s2_loss.AnchorAccumulator()
    for sl in (slice(0, 3), slice(3, 6), slice(6, 7)):
        accumulator.add(student[sl], teacher[sl], clean[sl])
    assert accumulator.value() == pytest.approx(single, abs=1e-15)
    assert accumulator.n_clean == 5


def test_clean_flat_mask_requires_all_flags():
    flags = dict(ordinary_flat=[True, True], nominal_dynamics=[True, True],
                 no_noise=[True, False], no_delay=[True, True],
                 no_force=[True, True])
    assert s2_loss.clean_flat_mask(flags).tolist() == [True, False]
    with pytest.raises(ValueError):
        s2_loss.clean_flat_mask({'ordinary_flat': [True]})


def test_ppo_config_and_kl_learning_rate():
    config = s2_runner.PPOConfig()
    assert config.rollout_steps == 48 and config.epochs == 5
    assert config.minibatches == 4 and config.microbatch_max == 512
    assert config.gamma == 0.99 and config.lam == 0.95 and config.clip == 0.2
    assert s2_runner.adjust_learning_rate(3e-5, 0.03) == pytest.approx(2e-5)
    assert s2_runner.adjust_learning_rate(3e-5, 0.001) == pytest.approx(4.5e-5)
    assert s2_runner.adjust_learning_rate(3e-5, 0.01) == pytest.approx(3e-5)
    assert s2_runner.adjust_learning_rate(9e-5, 0.001) <= 1e-4


def test_std_projection_and_clipped_loss():
    assert s2_runner.project_std(0.01) == 0.05
    assert s2_runner.project_std(2.0) == 1.0
    assert s2_runner.clipped_ppo_loss(np.array([1.3]), np.array([1.0])) == pytest.approx(-1.2)
    assert s2_runner.clipped_ppo_loss(np.array([2.0]), np.array([-1.0])) == pytest.approx(2.0)


def test_gae_task_failure_and_timeout_do_not_bootstrap():
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.5, 0.5])
    terminated = np.array([False, True, False])
    timeout = np.array([False, False, True])
    boundary = np.array([False, False, True])
    advantages, returns = s2_runner.gae(rewards, values, terminated=terminated,
                                        timeout=timeout, sampling_boundary=boundary,
                                        last_value=99.0)
    assert advantages.shape == returns.shape == (3,)
    # At t=1 terminated: no gamma*V2; at t=2 timeout: no last value bootstrap.
    assert advantages[1] == pytest.approx(1.0 + 0.0 - 0.5)


def test_gae_sampler_boundary_can_bootstrap_last_value():
    rewards = np.array([1.0])
    values = np.array([0.5])
    advantages, _ = s2_runner.gae(
        rewards, values, terminated=[False], timeout=[False],
        sampling_boundary=[True], last_value=2.0)
    assert advantages[0] == pytest.approx(1.0 + 0.99 * 2.0 - 0.5)


def test_weighted_accumulation_and_logical_minibatch():
    accumulated = s2_runner.weighted_accumulate([
        (np.array([1.0, 1.0]), 2), (np.array([3.0, 3.0]), 6)])
    assert accumulated.tolist() == [2.5, 2.5]
    plan = s2_runner.logical_minibatch_plan(2048)
    assert plan['logical_size'] == 512
    assert plan['microbatches'] == 1 and plan['adam_updates_per_epoch'] == 4


def test_runner_dry_run_supports_1024_and_512_not_2048():
    plan = s2_runner.dry_run_plan(num_envs=1024)
    assert plan['physics_started'] is False and plan['training_started'] is False
    assert plan['complete_rollouts']['S2_1'] == 16276
    assert s2_runner.dry_run_plan(num_envs=512)['samples_per_rollout'] == 24576
    with pytest.raises(ValueError):
        s2_runner.dry_run_plan(num_envs=2048)
