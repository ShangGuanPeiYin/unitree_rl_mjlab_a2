"""CPU-only §5 reward tests."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_rewards as rewards


def frame(**overrides):
    base = dict(c_request=0.5, c_exec=0.5, phase='POST_WALK',
                gravity_b=[0.0, 0.0, -1.0], velocity_b=[0.5, 0.0, 0.0],
                omega_b=[0.0, 0.0, 0.0], joint_position=[0.0] * 12,
                default_joint_position=[0.0] * 12,
                joint_velocity=[0.0] * 12, joint_velocity_prev=[0.0] * 12,
                action=[0.0] * 12, previous_action=[0.0] * 12,
                q_target_applied=[0.0] * 12,
                contact_filtered=[0.0] * 4,
                foot_velocity_xy=[[0.0, 0.0]] * 4, foot_height=[0.0] * 4,
                foot_surfaces=['ground'] * 4, foot_step_heights=[0.0] * 4,
                g_level=0.0, dx=0.1, v_heading=0.5, local_y=0.0,
                yaw_error=0.0, base_height=0.4, surface_height=0.0,
                stable_stand=False, air_time_reward=0.0, top_once=False,
                success_once=False, failure=False, timeout=False)
    base.update(overrides)
    return base


def test_continuous_sum_multiplies_dt_once():
    result = rewards.compute_reward(frame())
    assert result['continuous'] == pytest.approx(0.02 * (1.5 + 0.8))
    assert result['forward_progress'] == pytest.approx(0.04)
    assert result['total'] == pytest.approx(0.046 + 0.04)


def test_g_level_gates_pitch_and_overspeed_with_same_value():
    no_g = rewards.compute_reward(frame(gravity_b=[0.1, 0.0, -1.0],
                                       v_heading=1.0, g_level=0.0))
    with_g = rewards.compute_reward(frame(gravity_b=[0.1, 0.0, -1.0],
                                         v_heading=1.0, g_level=1.0))
    assert with_g['raw_terms']['orientation_pitch_level'] == pytest.approx(0.01)
    assert no_g['raw_terms']['orientation_pitch_level'] == 0.0
    assert with_g['raw_terms']['horizontal_overspeed'] == pytest.approx(0.35 ** 2)
    assert no_g['raw_terms']['horizontal_overspeed'] == 0.0


def test_same_frame_failure_pays_minus_ten_and_suppresses_top_success():
    result = rewards.compute_reward(frame(top_once=True, success_once=True,
                                          failure=True))
    assert result['top_reward'] == 0.0
    assert result['success_reward'] == 0.0
    assert result['failure_reward'] == -10.0
    assert result['total'] == pytest.approx(0.046 + 0.04 - 10.0)


def test_timeout_is_failure():
    result = rewards.compute_reward(frame(timeout=True))
    assert result['failure_reward'] == -10.0


def test_stop_closes_forward_and_air_and_clears_accumulator():
    result = rewards.compute_reward(frame(phase='DECEL', dx=1.0,
                                          air_time_reward=5.0))
    assert result['forward_progress'] == 0.0
    assert result['raw_terms']['feet_air_time'] == 0.0
    assert result['clear_air_accumulator'] is True


def test_cruise_requires_positive_reached_command():
    assert rewards.is_cruise(c_request=0.5, c_exec=0.5, phase='POST_WALK') is True
    assert rewards.is_cruise(c_request=0.0, c_exec=0.0, phase='POST_WALK') is False
    assert rewards.is_cruise(c_request=0.5, c_exec=0.4, phase='POST_WALK') is False
    assert rewards.is_cruise(c_request=0.5, c_exec=0.5, phase='DECEL') is False


def test_clearance_target_per_true_surface():
    assert rewards.clearance_target('ground', 0.0) == pytest.approx(0.075)
    assert rewards.clearance_target('platform', 0.2) == pytest.approx(0.075)
    assert rewards.clearance_target('tread', 0.2) == pytest.approx(0.22)


def test_twenty_weighted_columns_match_spec():
    assert len(rewards.WEIGHTS) == 20
    assert rewards.WEIGHTS['ang_vel_xy'] == (-0.05, -0.20)
    assert rewards.WEIGHTS['feet_air_time'] == (0.2, 0.0)
    assert rewards.WEIGHTS['stable_stand'] == (0.0, 1.0)
