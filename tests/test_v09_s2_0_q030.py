"""Q030 pitch sign and derived fractions (CPU only)."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07.events import euler_heading
from src.tasks.velocity.v09 import s2_env
import run_a2_v09_s2_0 as run_cli


def _quat_y(angle):
    return torch.tensor([[math.cos(angle / 2), 0.0,
                          math.sin(angle / 2), 0.0]], dtype=torch.float64)


def _forward_z(quat):
    w, x, y, z = quat[0].tolist()
    # (R_WB e_x)_z from the standard wxyz rotation matrix.
    return 2.0 * (x * z - w * y)


def test_q030_pitch_sign_is_nose_up_positive_by_independent_geometry():
    level = _quat_y(0.0)
    nose_up = _quat_y(-math.radians(20.0))
    nose_down = _quat_y(math.radians(20.0))
    for quat, expected in ((level, 0.0), (nose_up, math.radians(20.0)),
                           (nose_down, -math.radians(20.0))):
        pitch = float(euler_heading(quat)[1][0])
        assert pitch == pytest.approx(expected, abs=1e-12)
        assert pitch == pytest.approx(math.asin(_forward_z(quat)), abs=1e-12)


def test_q030_fractions_include_zero_in_denominator_and_use_raw_sign():
    rows = [dict(pitch=-0.2), dict(pitch=0.0), dict(pitch=0.1)]
    stats = run_cli._pitch_stats(rows)
    assert stats['pitch_sign_convention'] == s2_env.PITCH_SIGN_CONVENTION
    assert stats['frames'] == 3
    assert stats['nose_down_fraction'] == pytest.approx(1 / 3)
    assert stats['nose_up_fraction'] == pytest.approx(1 / 3)
    assert stats['level_fraction'] == pytest.approx(1 / 3)
    assert stats['min_deg'] == pytest.approx(-0.2 * 180 / math.pi)
    assert stats['max_deg'] == pytest.approx(0.1 * 180 / math.pi)



def test_v82_four_windows_exclude_post_reset_and_start_after_events():
    rows = [
        dict(step=0, route_x=0.0, pitch=0.0, foot_on_platform=[False]*4,
             top_arrived=False, terminal=False, post_reset=False),
        dict(step=1, route_x=1.0, pitch=-0.1, foot_on_platform=[False]*4,
             top_arrived=False, terminal=False, post_reset=False),
        dict(step=2, route_x=1.5, pitch=-0.2, foot_on_platform=[True]*4,
             top_arrived=False, terminal=False, post_reset=False),
        dict(step=3, route_x=1.6, pitch=-0.3, foot_on_platform=[True]*4,
             top_arrived=True, terminal=False, post_reset=False),
        dict(step=4, route_x=1.7, pitch=-0.4, foot_on_platform=[True]*4,
             top_arrived=True, terminal=False, post_reset=False),
        dict(step=5, route_x=1.8, pitch=0.5, foot_on_platform=[True]*4,
             top_arrived=True, terminal=True, post_reset=True),
    ]
    windows = run_cli._four_windows(
        rows, platform_front=1.0, case={},
        events=[dict(name='four_feet_platform', step=3),
                dict(name='top_arrived', step=4)])
    assert windows['front_edge']['frames'] == 4
    assert windows['front_edge']['step_min'] == 1
    assert windows['front_edge']['step_max'] == 4
    assert windows['four_feet']['step_min'] == 2
    assert windows['four_feet']['frames'] == 3
    assert windows['top_arrived']['step_min'] == 3
    assert windows['top_arrived']['frames'] == 2
    assert windows['four_feet']['first_true_step'] == 2
    assert windows['four_feet']['latched_event_step'] == 3
    assert windows['top_arrived']['first_true_step'] == 3
    assert windows['top_arrived']['latched_event_step'] == 4
    assert windows['top_arrived']['nose_down_fraction'] == 1.0
    assert windows['s6_5_cruise']['status'] == 'NA'
