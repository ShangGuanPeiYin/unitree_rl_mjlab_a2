"""CPU-only S2 state machine / g_level timing tests; no env.step."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_state as state


def frame(route_x, *, first=False, top=False, four=False, failure=False,
          a_request=None, a_applied=None):
    return dict(route_x=route_x, first_step=first, top_arrived=top,
                four_feet_on_platform=four, failure=failure,
                a_request=([0.0] * 12 if a_request is None else a_request),
                a_applied=([0.0] * 12 if a_applied is None else a_applied))


def stairs_config(**overrides):
    base = dict(task='stairs_walk', speed=0.5, stair_count=40,
                initial_route_x=-1.0, timeout_steps=1000, post_distance=5.0,
                platform_start=False)
    base.update(overrides)
    return state.S2TaskConfig(**base)


def test_state_order_and_single_top_latch():
    machine = state.S2StateMachine(stairs_config())
    machine.step(frame(-1.0))
    assert machine.state == 'APPROACH'
    machine.step(frame(0.1, first=True))
    assert machine.state == 'CLIMB'
    machine.step(frame(4.0, top=True))
    assert machine.state == 'TOP_TRANSITION'
    assert machine.top_arrived is True
    assert machine.top_route_x == 4.0
    machine.step(frame(5.0, top=True, four=True))  # repeated top is ignored
    assert machine.state == 'POST_WALK'
    assert sum(1 for event in machine.events if event['name'] == 'top_arrived') == 1
    machine.step(frame(8.9))  # d_post = 4.9, not yet
    assert machine.success is False
    machine.step(frame(9.0))  # d_post = 5.0 at the step end
    assert machine.success is True
    assert machine.state == 'SUCCESS'
    assert machine.terminal is True
    assert [event['name'] for event in machine.events].count('success') == 1


def test_same_frame_failure_wins_over_top_and_success():
    machine = state.S2StateMachine(stairs_config())
    machine.step(frame(0.1, first=True))
    events = machine.step(frame(4.0, top=True, four=True, failure=True))
    assert machine.failure is True
    assert machine.top_arrived is False
    assert machine.success is False
    assert machine.state == 'FAILURE'
    assert [event['name'] for event in events] == ['failure']


def test_timeout_uses_the_frozen_config_value_and_pays_failure():
    machine = state.S2StateMachine(stairs_config(timeout_steps=3))
    machine.step(frame(-1.0))
    machine.step(frame(0.0))
    assert machine.failure is False
    machine.step(frame(0.1, first=True))
    assert machine.failure is True
    assert machine.timeout is True
    assert machine.completed_steps == 3
    assert machine.state == 'FAILURE'


def test_platform_start_uses_spawn_zero_and_no_top_event():
    config = state.S2TaskConfig(task='platform_walk', speed=0.5, stair_count=40,
                                initial_route_x=9.0, timeout_steps=1000,
                                post_distance=5.0, platform_start=True)
    machine = state.S2StateMachine(config)
    assert machine.state == 'POST_WALK'
    machine.step(frame(13.9))
    assert machine.success is False
    assert machine.d_post(13.9) == pytest.approx(4.9)
    machine.step(frame(14.0))
    assert machine.success is True
    assert machine.top_arrived is False
    assert all(event['name'] != 'top_arrived' for event in machine.events)


def test_flat_walk_uses_completion_steps_not_distance():
    config = state.S2TaskConfig(task='flat_walk', speed=0.5, stair_count=0,
                                initial_route_x=-1.0, timeout_steps=1001,
                                post_distance=None, completion_steps=3)
    machine = state.S2StateMachine(config)
    machine.step(frame(-1.0))
    machine.step(frame(-0.5))
    assert machine.success is False
    machine.step(frame(0.0))
    assert machine.success is True
    assert machine.state == 'SUCCESS'


def test_flat_same_frame_failure_wins_over_completion():
    config = state.S2TaskConfig(task='flat_walk', speed=0.5, stair_count=0,
                                initial_route_x=-1.0, timeout_steps=1001,
                                post_distance=None, completion_steps=2)
    machine = state.S2StateMachine(config)
    machine.step(frame(-1.0))
    machine.step(frame(-0.5, failure=True))
    assert machine.failure is True and machine.success is False


def test_c_exec_chain_is_recorded_every_step():
    machine = state.S2StateMachine(stairs_config())
    machine.step(frame(-1.0, a_request=[0.1] * 12, a_applied=[0.2] * 12))
    machine.step(frame(0.1, first=True, a_request=[0.3] * 12,
                       a_applied=[0.4] * 12))
    trace = machine.command_trace()
    assert trace[0]['c_exec'] == pytest.approx(0.5)
    assert trace[1]['c_exec'] == pytest.approx(0.5)
    assert trace[1]['c_request'] == pytest.approx(0.5)
    assert trace[1]['a_applied'][0] == pytest.approx(0.4)


def test_level_gate_holds_half_second_then_ramps_and_resets():
    gate = state.LevelGate()
    for _ in range(state.G_LEVEL_HOLD_STEPS):
        assert gate.update(True) == 0.0
    assert gate.update(True) == pytest.approx(1 / state.G_LEVEL_RAMP_STEPS)
    for _ in range(state.G_LEVEL_RAMP_STEPS):
        value = gate.update(True)
    assert value == 1.0
    assert gate.ever_full is True
    assert gate.update(False) == 0.0
    assert gate.held == 0
    for _ in range(state.G_LEVEL_HOLD_STEPS):
        gate.update(True)
    assert gate.update(True) > 0.0


def test_level_condition_requires_four_feet_on_true_top_platform():
    platform = dict(platform_front_x=8.0, platform_length=20.0,
                    platform_y_half_width=0.4)
    assert state.level_condition(base_xy=(8.5, 0.0),
                                 feet_xy=[(8.2, 0.1), (8.3, -0.1),
                                          (8.4, 0.0), (8.1, 0.2)],
                                 **platform) is True
    assert state.level_condition(base_xy=(8.2, 0.0),   # not 0.4 m past edge
                                 feet_xy=[(8.2, 0.1), (8.3, -0.1),
                                          (8.4, 0.0), (8.1, 0.2)],
                                 **platform) is False
    assert state.level_condition(base_xy=(8.5, 0.0),
                                 feet_xy=[(8.2, 0.5), (8.3, -0.1),
                                          (8.4, 0.0), (8.1, 0.2)],
                                 **platform) is False
    wide = dict(platform_front_x=8.0, platform_length=20.0,
                platform_y_half_width=2.0)
    assert state.level_condition(base_xy=(8.5, 1.0),
                                 feet_xy=[(8.2, 1.5), (8.3, -1.0),
                                          (8.4, 0.0), (8.1, 1.9)],
                                 **wide) is True
