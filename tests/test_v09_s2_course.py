"""CPU/fake tests for S2.2/S2.3 course and stop-column contracts."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_course, s2_spec, s2_train, s2_train_env


def test_s2_2_course_config_and_budget():
    cfg = s2_course.validate_course('S2.2')
    assert cfg['main_levels'] == (40,)
    assert cfg['post_range'] == (5.0, 10.0)
    assert cfg['platform_post'] == 10.0
    assert cfg['review_levels'] == (10, 20)
    plan = s2_train.rollout_plan(512, course='S2.2')
    assert plan['max_rollouts'] == 48_828
    assert plan['samples_per_rollout'] == 24_576
    assert plan['max_rollouts'] * plan['samples_per_rollout'] <= 1_200_000_000


def test_s2_3_course_config_and_roles():
    cfg = s2_course.validate_course('S2.3')
    assert cfg['main_task'] == 'stairs_stop'
    assert cfg['post_range'] == (3.0, 10.0)
    plan = s2_train_env.build_role_plan(512, course='S2.3')
    assert {row['task'] for row in plan['envs']
            if row['role'] == 'stairs_walk'} == {'stairs_stop'}
    assert {row['task'] for row in plan['envs']
            if row['role'] == 'platform_walk'} == {'platform_stop'}
    assert {row['stop_mode'] for row in plan['envs']
            if row['task'] == 'stairs_stop'} == {'instant', 'ramp_2s'}
    assert {row['stair_count'] for row in plan['envs']
            if row['role'] == 'l40_milestone'} == {10, 20}


def test_instant_stop_next_actor_and_stand_mask():
    state = s2_course.StopState(0.6, 2.0, 'instant')
    state.advance(distance=1.99)
    assert state.reward_mask()['I_cruise'] is True
    out = state.advance(distance=2.0)
    assert out['c_exec'] == 0.0
    assert out['stop_requested'] is True
    mask = state.reward_mask()
    assert mask['I_cruise'] is False
    assert mask['forward_progress'] is False
    assert mask['air_time'] is False
    assert mask['stable_stand'] is True


def test_ramp_stop_reaches_zero_after_one_hundred_steps_and_holds():
    state = s2_course.StopState(0.6, 2.0, 'ramp_2s')
    state.advance(distance=2.0)
    assert state.last_c_exec == pytest.approx(0.6)
    for _ in range(99):
        state.advance()
    assert state.last_c_exec > 0.0
    state.advance()
    assert state.last_c_exec == 0.0
    assert state.phase in ('HOLD', 'STAND')
    for _ in range(500):
        state.advance()
    assert state.phase == 'STAND'
    for _ in range(500):
        state.advance()
    assert state.phase == 'DONE_STAND'


def test_stop_mask_does_not_change_reward_weights():
    weights = s2_course.reward_weights()
    assert weights == __import__('src.tasks.velocity.v07.spec',
                                 fromlist=['REWARD_WEIGHTS']).REWARD_WEIGHTS
