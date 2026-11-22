"""Unit tests for the S2 §5 reward formula and precedence."""
from __future__ import annotations
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_rewards


def base_frame():
    return dict(
        c_request=0.5, c_exec=0.5, phase='CRUISE',
        gravity_b=[0.0, 0.0, -1.0], velocity_b=[0.5, 0.0, 0.0],
        omega_b=[0.0, 0.0, 0.0], joint_position=[0.0] * 12,
        default_joint_position=[0.0] * 12, joint_velocity=[0.0] * 12,
        joint_velocity_prev=[0.0] * 12, action=[0.0] * 12,
        previous_action=[0.0] * 12, q_target_applied=[0.0] * 12,
        contact_filtered=[0.0] * 4, foot_velocity_xy=[[0.0, 0.0]] * 4,
        foot_height=[s2_rewards.FLAT_CLEARANCE] * 4,
        foot_surfaces=['ground'] * 4, foot_step_heights=[0.0] * 4,
        g_level=0.0, base_height=0.40, surface_height=0.0,
        air_time_reward=0.0, local_y=0.0, yaw_error=0.0,
        joint_limit_violation=0.0, v_heading=0.5, dx=0.1,
        failure=False, timeout=False, top_once=False, success_once=False,
        stable_stand=False)


def test_total_formula_and_one_dt():
    result = s2_rewards.compute_reward(base_frame())
    expected_continuous = 0.02 * (1.5 * 1.0 + 0.8 * 1.0)
    assert result['continuous'] == pytest.approx(expected_continuous)
    assert result['forward_progress'] == pytest.approx(0.04)
    assert result['total'] == pytest.approx(expected_continuous + 0.04)


def test_failure_precedence_and_one_time_terms():
    frame = base_frame()
    frame.update(failure=True, top_once=True, success_once=True)
    result = s2_rewards.compute_reward(frame)
    assert result['top_reward'] == 0.0
    assert result['success_reward'] == 0.0
    assert result['failure_reward'] == -10.0
    assert result['total'] == pytest.approx(result['continuous'] + result['forward_progress'] - 10.0)


def test_top_and_success_once():
    frame = dict(base_frame(), top_once=True, success_once=True)
    result = s2_rewards.compute_reward(frame)
    assert result['top_reward'] == 2.0
    assert result['success_reward'] == 8.0
    assert result['total'] == pytest.approx(result['continuous'] + 0.04 + 10.0)


def test_icruise_requires_positive_request_nonstop_and_executed_request():
    assert s2_rewards.is_cruise(c_request=0.5, c_exec=0.5, phase='CRUISE')
    assert not s2_rewards.is_cruise(c_request=0.0, c_exec=0.0, phase='CRUISE')
    assert not s2_rewards.is_cruise(c_request=0.5, c_exec=0.5, phase='HOLD')
    assert not s2_rewards.is_cruise(c_request=0.5, c_exec=0.2, phase='CRUISE')


def test_decel_closes_forward_air_and_only_zero_command_stands():
    frame = dict(base_frame(), c_exec=0.0, phase='HOLD',
                 air_time_reward=1.0, stable_stand=True)
    result = s2_rewards.compute_reward(frame)
    assert result['forward_progress'] == 0.0
    assert result['raw_terms']['feet_air_time'] == 0.0
    assert result['raw_terms']['stable_stand'] == 1.0
    frame['c_exec'] = 0.2
    result = s2_rewards.compute_reward(frame)
    assert result['raw_terms']['stable_stand'] == 0.0


def test_batch_reward_matches_scalar_and_all_twenty_terms():
    frame = base_frame()
    scalar = s2_rewards.compute_reward(frame)
    batch_frame = {key: ([value] if not isinstance(value, list) or
                         (value and isinstance(value[0], (int, float)))
                         else [value]) for key, value in frame.items()}
    # Build explicit batch fields for nested arrays.
    batch_frame = dict(
        c_request=[frame['c_request']], c_exec=[frame['c_exec']],
        phase=[frame['phase']], gravity_b=[frame['gravity_b']],
        velocity_b=[frame['velocity_b']], omega_b=[frame['omega_b']],
        joint_position=[frame['joint_position']],
        default_joint_position=[frame['default_joint_position']],
        joint_velocity=[frame['joint_velocity']],
        joint_velocity_prev=[frame['joint_velocity_prev']],
        action=[frame['action']], previous_action=[frame['previous_action']],
        q_target_applied=[frame['q_target_applied']],
        contact_filtered=[frame['contact_filtered']],
        foot_velocity_xy=[frame['foot_velocity_xy']],
        foot_height=[frame['foot_height']],
        target_clearance=[[s2_rewards.FLAT_CLEARANCE] * 4],
        g_level=[0.0], base_height=[0.4], surface_height=[0.0],
        air_time_reward=[0.0], local_y=[0.0], yaw_error=[0.0],
        joint_limit_violation=[0.0], v_heading=[0.5], dx=[0.1],
        failure=[False], timeout=[False], top_once=[False],
        success_once=[False], stable_stand=[False])
    batch = s2_rewards.compute_reward_batch(batch_frame)
    assert set(batch['terms']) == set(s2_rewards.WEIGHTS)
    assert batch['total'][0] == pytest.approx(scalar['total'])


def test_g_level_ramp_and_failure_reentry():
    assert s2_rewards.g_level_at(10, 10) == 0.0
    assert s2_rewards.g_level_at(11, 10) == pytest.approx(0.04)
    assert s2_rewards.g_level_at(34, 10) == pytest.approx(0.96)
    assert s2_rewards.g_level_at(35, 10) == 1.0
    assert s2_rewards.g_level_at(40, 10) == 1.0
    assert s2_rewards.g_level_at(41, -1) == 0.0
