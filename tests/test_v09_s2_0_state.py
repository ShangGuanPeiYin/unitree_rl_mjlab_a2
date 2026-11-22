"""CPU-only S2 state machine / Q027 g_level timing tests; no env.step."""
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
          v07_success=False, a_request=None, a_applied=None):
    return dict(route_x=route_x, first_step=first, top_arrived=top,
                four_feet_on_platform=four, failure=failure,
                v07_success=v07_success,
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
    machine.step(frame(5.0, top=True, four=True))
    assert machine.state == 'POST_WALK'
    assert sum(1 for event in machine.events if event['name'] == 'top_arrived') == 1
    machine.step(frame(8.9))
    assert machine.success is False
    machine.step(frame(9.0))
    assert machine.success is True
    assert machine.state == 'SUCCESS'
    assert machine.terminal is True


def test_same_frame_failure_wins_over_top_and_success():
    machine = state.S2StateMachine(stairs_config())
    machine.step(frame(0.1, first=True))
    events = machine.step(frame(4.0, top=True, four=True, failure=True))
    assert machine.failure is True
    assert machine.top_arrived is False
    assert machine.success is False
    assert machine.state == 'FAILURE'
    assert [event['name'] for event in events] == ['failure']


def test_timeout_is_task_failure():
    machine = state.S2StateMachine(stairs_config(timeout_steps=3))
    machine.step(frame(-1.0))
    machine.step(frame(0.0))
    machine.step(frame(0.1, first=True))
    assert machine.failure is True
    assert machine.timeout is True
    assert machine.completed_steps == 3


def test_platform_start_uses_spawn_zero_and_no_top_event():
    config = state.S2TaskConfig(task='platform_walk', speed=0.5, stair_count=40,
                                initial_route_x=9.0, timeout_steps=1000,
                                post_distance=5.0, platform_start=True)
    machine = state.S2StateMachine(config)
    assert machine.state == 'POST_WALK'
    machine.step(frame(13.9))
    assert machine.success is False
    machine.step(frame(14.0))
    assert machine.success is True
    assert machine.top_arrived is False


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
    assert trace[1]['a_applied'][0] == pytest.approx(0.4)


def test_q027_manual_sequence_t0_t1_t24_t25_and_full_hold():
    gate = state.LevelGate()
    assert gate.update(True, step_index=10) == 0.0
    assert gate.t0_step == 10
    assert gate.update(True, step_index=11) == pytest.approx(1 / 25)
    for step in range(12, 35):       # t0+2 .. t0+24
        value = gate.update(True, step_index=step)
    assert step == 34 and value == pytest.approx(24 / 25)
    assert gate.update(True, step_index=35) == 1.0   # t0+25
    assert gate.first_full_step == 35
    assert gate.update(True, step_index=36) == 1.0
    assert gate.ever_full is True


def test_q027_exact_offsets_and_no_first_frame_precount():
    gate = state.LevelGate()
    assert gate.update(True, step_index=0) == 0.0       # t0
    assert gate.update(True, step_index=1) == pytest.approx(0.04)
    gate = state.LevelGate()
    gate.update(True, step_index=0)
    for step in range(1, 25):
        gate.update(True, step_index=step)
    assert gate.value == pytest.approx(0.96)            # t0+24
    assert gate.update(True, step_index=25) == 1.0      # t0+25
    assert gate.first_full_step == 25


def test_q027_interruption_and_reentry_restarts_t0():
    gate = state.LevelGate()
    gate.update(True, step_index=0)
    for step in range(1, 6):
        gate.update(True, step_index=step)
    assert gate.interruptions == 0
    assert gate.update(False, step_index=6) == 0.0
    assert gate.interruptions == 1
    assert gate.t0_step is None
    assert gate.update(True, step_index=10) == 0.0      # new t0
    assert gate.t0_step == 10
    assert gate.update(True, step_index=11) == pytest.approx(0.04)
    assert gate.restarts == 2


def test_q027_full_then_fail_reentry():
    gate = state.LevelGate()
    gate.update(True, step_index=0)
    for step in range(1, 26):
        gate.update(True, step_index=step)
    assert gate.value == 1.0
    assert gate.update(False, step_index=26) == 0.0
    assert gate.interruptions == 1
    assert gate.update(True, step_index=30) == 0.0
    assert gate.t0_step == 30
    assert gate.update(True, step_index=31) == pytest.approx(0.04)


def test_q027_flat_is_one_and_platform_start_first_sample_is_t0():
    flat = state.LevelGate(flat=True)
    assert flat.update(False) == 1.0
    assert flat.update(False) == 1.0
    platform = state.LevelGate()
    assert platform.update(True, step_index=0) == 0.0
    assert platform.update(True, step_index=1) == pytest.approx(0.04)


def test_q027_report_and_snapshot_version():
    gate = state.LevelGate()
    gate.update(True, step_index=5)
    gate.update(True, step_index=6)
    report = state.g_level_report(gate, four_foot_step=4)
    assert report['first_full_geometry_step'] == 5
    assert report['first_g_positive_step'] == 6
    assert report['delay_seconds_from_four_foot'] == pytest.approx(0.04)
    assert report['first_g_full_step'] is None
    assert 'g_level_never_full' in report['na_reasons']
    snapshot = gate.state_dict()
    assert snapshot['version'] == state.G_LEVEL_SNAPSHOT_VERSION
    restored = state.LevelGate().load_state(snapshot)
    assert restored.value == gate.value
    old = dict(snapshot, version=state.G_LEVEL_PRE_Q027_SNAPSHOT)
    with pytest.raises(ValueError):
        state.LevelGate().load_state(old)


def test_level_condition_requires_strict_four_foot_true_platform():
    platform = dict(platform_front_x=8.0, platform_length=20.0,
                    platform_y_half_width=0.4)
    feet = [(8.2, 0.1), (8.3, -0.1), (8.4, 0.0), (8.1, 0.2)]
    assert state.level_condition(base_xy=(8.5, 0.0), feet_xy=feet,
                                 **platform) is True
    assert state.level_condition(base_xy=(8.4, 0.0), feet_xy=feet,
                                 **platform) is False
    assert state.level_condition(base_xy=(8.5, 0.0),
                                 feet_xy=[(8.2, 0.5)] + feet[1:],
                                 **platform) is False
    wide = dict(platform_front_x=8.0, platform_length=20.0,
                platform_y_half_width=2.0)
    assert state.level_condition(base_xy=(8.5, 1.0),
                                 feet_xy=[(8.2, 1.5), (8.3, -1.0),
                                          (8.4, 0.0), (8.1, 1.9)],
                                 **wide) is True


def test_b_v5_2_legacy_h_uses_v07_success_not_s2_distance():
    config = state.S2TaskConfig(task='legacy_stairs', speed=0.5,
                                stair_count=6, initial_route_x=-1.0,
                                timeout_steps=1301, post_distance=None,
                                legacy_h=True)
    machine = state.S2StateMachine(config)
    machine.step(frame(0.1, first=True))
    machine.step(frame(1.8, top=True, four=True))
    # Body only 0.2 m past the 1.8 front edge: v07 success is false.
    machine.step(frame(2.0, v07_success=False))
    assert machine.success is False
    # At 2.6 m the caller reports the v07 support/attitude success.
    machine.step(frame(2.6, v07_success=True))
    assert machine.success is True
    assert machine.terminal is True
    assert any(event['name'] == 'legacy_v07_success' for event in machine.events)


def test_b_v5_2_legacy_helper_and_timeout_identity():
    assert state.legacy_v07_success(2.0, platform_front_x=1.8) is False
    assert state.legacy_v07_success(2.6, platform_front_x=1.8) is True
    assert state.legacy_v07_success(2.6, platform_front_x=1.8,
                                    supported=False) is False
    for timeout in (1301, 1302, 1306, 1315, 1318, 1332):
        config = state.S2TaskConfig(task='legacy_stairs', speed=0.5,
                                    stair_count=6, initial_route_x=-1.0,
                                    timeout_steps=timeout, post_distance=None,
                                    legacy_h=True)
        assert config.timeout_steps == timeout
