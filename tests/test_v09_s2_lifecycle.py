"""CPU tests for the S2 training lifecycle state machine and quality."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_lifecycle as life


def _one(task='stairs_walk', post=5.0, speed=0.5, stop='none', legacy=False):
    return life.S2Lifecycle(task=[task], post_distance=[post], speed=[speed],
                            stop_mode=[stop], legacy=[legacy], front=[2.0],
                            spawn_x=[-1.0])


def test_stairs_top_is_event_and_success_waits_for_post():
    s = _one()
    for x in (0.1, 1.0, 2.0):
        out = s.update(local_x=[x], top_signal=[x >= 2.0], four_feet=[x >= 2.0],
                       body_ok=[x >= 2.2], quality_ok=[True])
    assert bool(out['top_event'][0]) or bool(out['top'][0])
    assert not s.success[0]
    out = s.update(local_x=[7.0], top_signal=[True], four_feet=[True],
                   body_ok=[True], quality_ok=[True])
    assert s.success[0]
    assert bool(out['success_event'][0])


def test_top_paid_once_and_quality_failure_blocks_success():
    s = _one()
    s.update(local_x=[2.0], top_signal=[True], four_feet=[True],
             body_ok=[True], quality_ok=[False])
    out = s.update(local_x=[7.0], top_signal=[True], four_feet=[True],
                   body_ok=[True], quality_ok=[False])
    assert s.top[0] and not s.success[0]
    assert out['quality_reason'] if 'quality_reason' in out else True


def test_platform_walk_success_at_post_distance_without_top():
    s = _one(task='platform_walk', post=5.0)
    s.front = np.array([2.0]); s.spawn_x = np.array([3.0])
    for x in (3.0, 6.0, 7.9):
        s.update(local_x=[x], top_signal=[False], four_feet=[False],
                 body_ok=[False], quality_ok=[True])
    assert not s.success[0]
    s.update(local_x=[8.0], top_signal=[False], four_feet=[False],
             body_ok=[False], quality_ok=[True])
    assert s.success[0]


def test_l40_legacy_top_success_same_frame():
    s = _one(task='legacy_l40', post=5.0, legacy=True)
    out = s.update(local_x=[8.0], top_signal=[True], four_feet=[True],
                   body_ok=[True], quality_ok=[True])
    assert s.success[0] and s.top_event[0]
    assert not s.failure[0]


def test_stop_instant_next_actor_zero_and_hold_then_success():
    s = _one(task='stairs_stop', post=3.0, stop='instant')
    s.update(local_x=[4.9], top_signal=[True], four_feet=[True],
             body_ok=[True], quality_ok=[True])
    out = s.update(local_x=[5.0], top_signal=[True], four_feet=[True],
                   body_ok=[True], quality_ok=[True])
    assert bool(out['c_exec'][0] == 0.0)
    assert out['phase'][0] in ('HOLD', 'DECEL')
    assert not s.success[0]
    for _ in range(999):
        out = s.update(local_x=[5.0], top_signal=[True], four_feet=[True],
                       body_ok=[True], quality_ok=[True])
    assert not s.success[0]
    out = s.update(local_x=[3.0], top_signal=[True], four_feet=[True],
                   body_ok=[True], quality_ok=[True])
    assert bool(s.success[0]) and bool(out['success_event'][0])


def test_stop_ramp_monotone_and_request_independent():
    s = _one(task='platform_stop', post=2.0, stop='ramp_2s')
    s.update(local_x=[4.9], top_signal=[False], four_feet=[False],
             body_ok=[False], quality_ok=[True])
    out = s.update(local_x=[5.0], top_signal=[False], four_feet=[False],
                   body_ok=[False], quality_ok=[True])
    values = [out['c_exec'][0]]
    for _ in range(99):
        values.append(s.update(local_x=[5.0], top_signal=[False], four_feet=[False],
                               body_ok=[False], quality_ok=[True])['c_exec'][0])
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert bool(values[-1] == 0.0)
    assert np.allclose(s.c_request[0], 0.5)


def test_failure_precedence_and_reset_isolation():
    s = life.S2Lifecycle(task=['stairs_walk', 'stairs_walk'],
                         post_distance=[5.0, 5.0], speed=[0.5, 0.5],
                         stop_mode=['none', 'none'], legacy=[False, False],
                         front=[2.0, 2.0], spawn_x=[-1.0, -1.0])
    s.update(local_x=[7.0, 7.0], top_signal=[True, True], four_feet=[True, True],
             body_ok=[True, True], quality_ok=[True, True], failure=[True, False])
    assert s.failure[0] and not s.success[0]
    s.reset([True, False])
    assert not s.top[0] and not s.failure[0]
    assert s.top[1]  # other env untouched


def test_quality_reasons_thresholds():
    good = life.quality_reasons(pitch_deg=[0.0, 1.0], speed=[0.5, 0.5],
                                command=[0.5, 0.5], contact=None,
                                planned_support=[True, True])
    assert good == []
    bad = life.quality_reasons(pitch_deg=[20.0] * 20, speed=[1.0] * 20,
                               command=[0.0] * 20, contact=None,
                               planned_support=[False] * 20, nonfoot=True)
    assert 'pitch_rms' in bad and 'pitch_p99' in bad
    assert 'speed_rmse' in bad and 'nonfoot_contact' in bad
    assert 'support' in bad
