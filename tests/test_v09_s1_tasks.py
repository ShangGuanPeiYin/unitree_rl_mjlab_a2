"""Synthetic-only S1 timeline/quality/reset-evidence checks; no simulation."""
from dataclasses import dataclass
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap  # noqa: E402
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import metrics as m  # noqa: E402
from src.tasks.velocity.v09.env import EpisodeIds, V09S1Env, V07Env, S1StallDiagnostics  # noqa: E402


def trajectory(task='flat_walk', stop_mode='none', n=None, cruise=.3):
    total = m.flat_schedule(task, stop_mode)['completion_steps'] if n is None else n
    actor = torch.arange(total)
    command = torch.tensor([m.flat_command(task, stop_mode, cruise, int(i)) for i in actor])
    velocity = torch.zeros(total, 3)
    velocity[:, 0] = command
    position = torch.zeros(total, 3)
    position[:, 0] = command.cumsum(0) * .02
    contact = ((actor[:, None] + torch.arange(4)[None] * 3) % 25) < 12
    return dict(step=actor + 1, position=position, initial_position=torch.zeros(3),
                is_cruise=torch.ones(total, dtype=torch.bool) if task == 'flat_walk' else actor < 400,
                roll=torch.zeros(total), pitch=torch.zeros(total),
                velocity_b=velocity.clone(), velocity_w=velocity.clone(), omega=torch.zeros(total, 3),
                foot_x=position[:, :1].expand(-1, 4).clone(),
                foot_velocity_xy=torch.zeros(total, 4, 2), raw_contact=contact,
                filtered=contact.clone(), support=torch.ones(total, 4, dtype=torch.bool),
                action=torch.zeros(total, 12), previous_action=torch.zeros(total, 12),
                height=torch.full((total,), .4), height_valid=torch.ones(total, dtype=torch.bool),
                nonfoot_contact=torch.zeros(total, dtype=torch.bool), command=command)


def summarize(e, task='flat_walk', stop_mode='none', **kwargs):
    return m.summarize_trajectory(e, task=task, stop_mode=stop_mode, cruise=.3, **kwargs)


def test_q002_exact_instant_and_ramp_actor_indices():
    assert m.flat_command('flat_stop', 'instant', .6, 399) == .6
    assert m.flat_command('flat_stop', 'instant', .6, 400) == 0
    assert m.flat_command('flat_stop', 'ramp_2s', .6, 399) == .6
    assert m.flat_command('flat_stop', 'ramp_2s', .6, 400) == pytest.approx(.594)
    assert m.flat_command('flat_stop', 'ramp_2s', .6, 498) == pytest.approx(.006)
    assert m.flat_command('flat_stop', 'ramp_2s', .6, 499) == 0
    assert m.flat_command('flat_stop', 'ramp_2s', .6, 500) == 0
    ramp = m.flat_schedule('flat_stop', 'ramp_2s')
    assert ramp == dict(completion_steps=1500, hard_timeout_steps=1501,
                        hold_actor_index=500, first_zero_actor_index=499)
    instant = m.flat_schedule('flat_stop', 'instant')
    assert instant['completion_steps'] == 1400 and instant['hard_timeout_steps'] == 1401


@pytest.mark.parametrize('task,mode,hold', [('flat_walk', 'none', None),
                                         ('flat_stop', 'instant', 1000),
                                         ('flat_stop', 'ramp_2s', 1000)])
def test_complete_windows_and_per_foot_cycles(task, mode, hold):
    result = summarize(trajectory(task, mode), task, mode)
    assert result['quality_pass']
    assert result['cruise_sample_count'] == (900 if task == 'flat_walk' else 300)
    assert result['pitch_rms_deg'] == 0
    assert result['gait_valid'] and all(len(cycles) >= 2 for cycles in result['gait_cycles'])
    assert result['stride'] == pytest.approx(.15, abs=1e-6)
    assert result['frequency'] == pytest.approx(2.)
    if hold is not None:
        assert result['hold_sample_count'] == hold
        assert result['stand_sample_count'] == 500
        assert result['drift'] == pytest.approx(0)
        assert result['recovery_seconds'] == pytest.approx(.5)


def test_early_failure_cannot_turn_a_short_prefix_into_complete_teacher_window():
    result = summarize(trajectory(n=150), physical_failure=True)
    assert result['cruise_sample_count'] == 50
    assert not result['cruise_window_valid']
    assert result['speed_rmse'] is None and result['pitch_rms_deg'] is None
    assert not result['gait_valid'] and result['stride'] is None
    assert all(not result['metric_validity'][key]['valid']
               for key in ('speed_rmse', 'pitch_rms_deg', 'slip', 'action_change', 'stride', 'frequency'))


def test_window_then_failure_retains_teacher_metric_validity_but_not_missing_stand():
    result = summarize(trajectory('flat_stop', 'instant', n=500), 'flat_stop', 'instant', physical_failure=True)
    assert result['metric_validity']['speed_rmse']['valid']
    assert result['metric_validity']['stride']['valid']
    assert not result['metric_validity']['drift']['valid']
    assert result['drift'] is None and not result['quality_pass']


def test_no_landings_never_becomes_zero_stride_or_frequency():
    evidence = trajectory()
    evidence['raw_contact'][:] = True
    result = summarize(evidence)
    assert result['quality_pass']  # absolute quality and relative gait coverage differ
    assert not result['relative_metrics_complete']
    assert result['stride'] is None and result['frequency'] is None
    assert result['metric_validity']['stride']['missing_reason'] == 'insufficient_landing_cycles'


def test_ramp_recovery_uses_first_zero_decel_interval_and_302s_fails():
    e = trajectory('flat_stop', 'ramp_2s')
    # First 25 stable intervals start at i625, complete at i649's step end:
    # (650 - 499)*.02 = 3.02s, so formal-HOLD origin500 would wrongly pass.
    e['support'][499:625] = False
    result = summarize(e, 'flat_stop', 'ramp_2s')
    assert result['recovery_seconds'] == pytest.approx(3.02)
    assert 'recovery' in result['quality_reasons'] and not result['quality_pass']
    e['support'][624] = True
    result = summarize(e, 'flat_stop', 'ramp_2s')
    assert result['recovery_seconds'] == pytest.approx(3.0)
    assert result['quality_pass']


def test_q001_drift_uses_true_last_10_second_window_start():
    e = trajectory('flat_stop', 'instant')
    # Window is Actor900..1399 / end-step901..1400; origin is end-step900.
    e['position'][900:, 1] = .11
    result = summarize(e, 'flat_stop', 'instant')
    assert result['drift'] == pytest.approx(.11)
    assert 'stand_displacement' in result['quality_reasons']


def test_stall_continuous_pitch_and_full_window_metrics_do_not_mask_failures():
    e = trajectory()
    e['velocity_b'][400:500, 0] = 0
    e['velocity_w'][400:500, 0] = 0
    e['position'][:, 0] = e['velocity_w'][:, 0].cumsum(0) * .02
    result = summarize(e)
    assert result['speed_rmse'] == pytest.approx(.1, abs=1e-7)
    assert 'cruise_stall' in result['quality_reasons'] and not result['quality_pass']
    e = trajectory()
    e['pitch'][300:310] = 16 / m.RAD_TO_DEG
    result = summarize(e)
    assert 'pitch_continuous' in result['quality_reasons']


def test_q003_stall_starts_at_true_birth_x0_and_49_50_boundary_before_two_seconds():
    for n, expected_count, latched in ((98, 49, False), (99, 50, True)):
        e = trajectory(n=n)
        e['position'][:] = 0
        result = m.cruise_stall(e, .3)
        assert result['stall_max_consecutive_frames'] == expected_count
        assert result['cruise_stall_latched'] == latched
        if latched:
            assert result['stall_latched_step'] == 99
    # A false first-sample x0 would count from n51, delaying the latch by 1 step.
    e = trajectory(n=99)
    e['position'][:] = 0
    e['initial_position'][0] = -.2
    result = m.cruise_stall(e, .3)
    assert result['stall_max_consecutive_frames'] == 49
    assert not result['cruise_stall_latched']


def test_q003_decel_clears_pending_stall_and_latched_failure_persists():
    e = trajectory('flat_stop', 'ramp_2s', n=600)
    e['position'][:] = 0
    result = m.cruise_stall(e, .3)
    assert result['stall_latched_step'] == 99
    assert result['cruise_stall_latched']
    # A fresh cruise segment immediately before DECEL cannot combine evidence
    # across DECEL, even if its positive ramp command stays above .1.
    e['is_cruise'][:] = False
    e['is_cruise'][350:400] = True
    e['is_cruise'][450:500] = True
    result = m.cruise_stall(e, .3)
    assert not result['cruise_stall_latched']
    assert result['stall_max_consecutive_frames'] == 1
    # Phase labels can change at a top transition without changing CRUISE:
    # no discontinuity in is_cruise means the same 99-step latch is retained.
    e['is_cruise'][:99] = True
    assert m.cruise_stall(e, .3)['stall_latched_step'] == 99


def test_q004_pool_all_foot_cycles_equally_after_each_foot_coverage_gate():
    e = trajectory()
    e['raw_contact'][:] = False
    # Foot0: 9 landings /8 cycles, 0.30m; each other foot3 landings/2 cycles,.60m.
    for foot, landings, stride in ((0, list(range(150, 951, 100)), .30),
                                   (1, [200, 500, 800], .60),
                                   (2, [210, 510, 810], .60),
                                   (3, [220, 520, 820], .60)):
        for event, index in enumerate(landings):
            e['raw_contact'][index, foot] = True
            e['foot_x'][index, foot] = event * stride
    result = summarize(e)
    assert result['gait_valid']
    assert [len(cycles) for cycles in result['gait_cycles']] == [8, 2, 2, 2]
    assert result['stride'] == pytest.approx(.30, abs=1e-6)


def test_support_requires_top_force_safe_inset_and_two_consecutive_frames():
    points = torch.tensor([[[.1, .1], [.041, .1], [.1, .1], [.1, .1]]])
    forces = torch.tensor([[[6.], [6.], [5.], [6.]]])
    bounds, tiles, ids = torch.tensor([[0., 1., 0., 1.]]), torch.tensor([0]), torch.tensor([0])
    support, counts = m.support_from_geometry(points, forces, bounds, tiles, ids, torch.zeros(1, 4, dtype=torch.long))
    assert not support.any()
    support, counts = m.support_from_geometry(points, forces, bounds, tiles, ids, counts)
    assert support.tolist() == [[True, False, False, True]]


def test_episode_ids_and_case_streams_do_not_depend_on_slot_or_environment_count():
    ids = EpisodeIds({'flat_walk': 8})
    assert ids.reserve('flat_walk') == 8
    assert ids.reserve('stairs') == 0
    assert ids.reserve('flat_walk') == 9
    assert ids.state_dict() == {'flat_walk': 10, 'stairs': 1}


def test_flat_stall_adapter_allows_command_changes_without_changing_stair_diagnostic():
    flat = torch.tensor([True, False])
    stall = S1StallDiagnostics(2, 'cpu', flat, torch.tensor([.3, .3]))
    stall.reset([0, 1], torch.zeros(2))
    args = (torch.zeros(2), torch.tensor([.3, .3]), torch.zeros(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    stall.update(*args)
    stall.update(args[0], torch.tensor([0., .3]), *args[2:])
    assert stall.active_samples.tolist() == [0, 2]
    with pytest.raises(ValueError):
        stall.update(args[0], torch.tensor([0., .5]), *args[2:])


def test_trace_buffer_exports_pre_reset_snapshot_and_retains_independent_slots():
    trace = m.EpisodeTraceBuffer(2, 5)
    for step in range(1, 4):
        trace.append(dict(step=torch.tensor([step, step]), action=torch.full((2, 12), float(step))))
    evidence = trace.evidence(0)
    trace.reset(torch.tensor([0]))
    assert evidence['step'].tolist() == [1, 2, 3]
    assert evidence['previous_action'][:, 0].tolist() == [0., 1., 2.]
    assert trace.counts.tolist() == [0, 3]
    assert trace.storage_bytes > 0


@dataclass(frozen=True)
class FakeState:
    success: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    reason: torch.Tensor


@pytest.mark.parametrize('step,physical,quality,expected', [
    (1399, False, True, (False, False, False)),
    (1400, False, True, (True, False, False)),
    (1400, False, False, (False, True, False)),
    (1400, True, True, (False, True, False)),
    (1401, False, True, (False, True, False)),
])
def test_flat_complete_boundary_and_failure_priority_without_physics(step, physical, quality, expected):
    class SyntheticEnv(V09S1Env):
        @property
        def num_envs(self):
            return 1
        @property
        def device(self):
            return 'cpu'
    env = object.__new__(SyntheticEnv)
    env._v09_ready = True
    env.episode_length_buf = torch.tensor([step])
    env.flat_mask = env.flat_stop_mask = torch.tensor([True])
    env.ramp_mask = torch.tensor([False])
    env.completion_steps, env.hard_timeout_steps = torch.tensor([1400]), torch.tensor([1401])
    env.start_vx = torch.tensor([.3])
    env.command = torch.tensor([[0., 0., 0.]])
    env.v09_cases = [dict(task='flat_stop', stop_mode='instant', speed=.3)]
    env._completed_evidence, env._completed_metrics = {}, {}
    evidence = trajectory('flat_stop', 'instant')
    if not quality:
        evidence['pitch'][:] = .3
    class FakeTrace:
        def append(self, value):
            pass
        def evidence(self, index):
            return evidence
    env.trace = FakeTrace()
    env._metric_frame = lambda state: {}
    state = FakeState(torch.tensor([False]), torch.tensor([physical]),
                      torch.tensor([step >= 1500]), torch.tensor([5 if physical else 0]))
    with patch.object(V07Env, '_task_state', return_value=state):
        actual = env._task_state()
    assert tuple(bool(value[0]) for value in (actual.success, actual.failure, actual.timeout)) == expected
