"""CPU tests for the shared per-case S2 quality criterion (DSH-Q036 / SDS §6.5).

Includes an independent cross-check of all 24 hand-derived cases in
``logs/a2_v09/s2/q036_verify/quality_cases.json``; the frames are rebuilt from
the JSON spec and the production ``s2_quality`` machine must reproduce the
hand-computed outcome, endpoint, window frame count, reasons and metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import s2_quality as q

CASES = ROOT / 'logs/a2_v09/s2/q036_verify/quality_cases.json'


# --------------------------------------------------------------- frame model
def _patch(patches, n, default):
    for start, count, value in patches or ():
        if start <= n < start + count:
            return value
    return default


def _xc(segments, n):
    offset = 0
    for count, x0, dx in segments:
        if n <= offset + count:
            return x0 + dx * (n - offset)
        offset += count
    return x0 + dx * count


def _c_exec(case, n, speed):
    spec = case['c_exec']
    if spec == 'cruise':
        return speed
    if spec == 'stop_instant':
        return 0.0 if n > case.get('request_interval', 400) else speed
    if spec == 'stop_ramp':
        request = case.get('request_interval', 400)
        if n <= request:
            return speed
        k = n - request
        return speed * (1.0 - k / 100.0) if k <= 100 else 0.0
    if isinstance(spec, dict) and 'segs' in spec:
        offset = 0
        for count, value in spec['segs']:
            if n <= offset + count:
                return float(value)
            offset += count
        return 0.0
    raise ValueError(spec)


def build_frames(case):
    speed = float(case['speed'])
    n_frames = int(case['n'])
    spawn_x = float(case.get('spawn_x', -1.0))
    segments = case.get('x') or [[n_frames, spawn_x + 0.01, 0.01]]
    frames = []
    for n in range(1, n_frames + 1):
        c_exec = _c_exec(case, n, speed)
        frames.append(dict(
            n=n, t=0.02 * n, c_exec=c_exec,
            speed=_patch(case.get('speed_patches'), n, c_exec),
            x=_xc(segments, n),
            y=_patch(case.get('y_patches'), n, 0.0),
            pitch_deg=_patch(case.get('pitch_patches'), n, 0.0),
            roll_deg=_patch(case.get('roll_patches'), n, 0.0),
            omega=_patch(case.get('omega_patches'), n, 0.0),
            height=_patch(case.get('height_patches'), n, 0.40),
            foot_speed=_patch(case.get('foot_speed_patches'), n, 0.0),
            support=_patch(case.get('support_patches'), n, True),
            four_feet_platform=(n in set(case.get('four_feet_platform_at') or ())),
            top=(n in set(case.get('top_at') or ())),
            failure=(n in set(case.get('failure_at') or ())),
            timeout=(n in set(case.get('timeout_at') or ()))))
    return frames


def _outcome(out):
    if not out['endpoint'][0]:
        return 'no_endpoint'
    if out['physical_failure'][0]:
        return 'physical_failure'
    if out['timeout'][0]:
        return 'timeout_fail'
    if out['quality_failure'][0]:
        return 'quality_fail'
    return 'success' if out['quality_ok'][0] else 'no_outcome'


def run_case(case):
    frames = build_frames(case)
    out, machine = q.evaluate_frames(
        frames, task=case['task'], speed=float(case['speed']),
        spawn_x=float(case.get('spawn_x', -1.0)),
        post_distance=float(case.get('post', 5.0)),
        stop_mode=case.get('stop_mode', 'none'),
        legacy=bool(case.get('legacy', False)))
    count = int(out['hold_samples'][0])
    take = min(500, count) if count >= 1000 else count
    stand = dict(
        frames=take,
        mean_speed=out['stop_mean_speed'][0],
        peak_speed=out['stop_peak_speed'][0],
        displacement=out['stop_max_displacement'][0],
        support_ratio=out['stop_support_ratio'][0],
        tilt_max=max(out['stop_max_roll'][0], out['stop_max_pitch'][0])
        if take else None,
        height_min=out['stop_min_height'][0])
    return dict(
        outcome=_outcome(out),
        endpoint_step=int(out['endpoint_step'][0]),
        window_frames=int(out['cruise_samples'][0]),
        reasons=list(out['reasons'][0]),
        first_zero=int(out['first_zero_step'][0]),
        cruise_metrics=dict(
            frames=int(out['cruise_samples'][0]),
            pitch_rms=out['cruise_pitch_rms'][0],
            pitch_p99=out['cruise_pitch_p99'][0],
            speed_rmse=out['cruise_speed_rmse'][0]),
        stand_metrics=stand)


def _document():
    return json.loads(CASES.read_text(encoding='utf-8'))


def test_all_twenty_four_independent_reference_cases_pass():
    document = _document()
    assert len(document['cases']) == 24
    failures = []
    for case in document['cases']:
        expected = case['expected']
        actual = run_case(case)
        if expected.get('outcome') != actual['outcome']:
            failures.append(f"{case['name']} outcome {actual['outcome']} != {expected['outcome']}")
        if ('endpoint_step' in expected
                and expected['endpoint_step'] != actual['endpoint_step']):
            failures.append(f"{case['name']} endpoint {actual['endpoint_step']} != {expected['endpoint_step']}")
        if ('window_frames' in expected
                and expected['window_frames'] != actual['window_frames']):
            failures.append(f"{case['name']} frames {actual['window_frames']} != {expected['window_frames']}")
        want = sorted(expected.get('reasons') or [])
        got = sorted(actual['reasons'])
        if want != got:
            failures.append(f"{case['name']} reasons {got} != {want}")
        for group in ('cruise_metrics', 'stand_metrics'):
            for key, value in (expected.get(group) or {}).items():
                measured = (actual[group] or {}).get(key)
                if measured is None or abs(float(measured) - float(value)) > 1e-9:
                    failures.append(
                        f"{case['name']} {group}.{key} {measured} != {value}")
        if 'first_zero' in expected and expected['first_zero'] != actual['first_zero']:
            failures.append(f"{case['name']} first_zero {actual['first_zero']} != {expected['first_zero']}")
    assert not failures, '\n'.join(failures)


# ------------------------------------------------------------- small helpers
def _step(qm, *, x, y=0.0, pitch=0.0, roll=0.0, speed=0.5, v_heading=0.5,
          c_exec=0.5, stop=False, entry=False, top=False, height=0.4,
          support=True, foot_speed=0.0, omega=0.0, failure=False,
          timeout=False):
    return qm.update(
        local_x=[x], local_y=[y], pitch_deg=[pitch], roll_deg=[roll],
        horizontal_speed=[speed], v_heading=[v_heading], c_exec=[c_exec],
        stop_requested=[stop], four_feet_entry=[entry], top_arrived=[top],
        base_height=[height], surface_height=[0.0],
        foot_support=[[support] * 4], foot_speed_xy=[[foot_speed] * 4],
        angular_speed=[omega], failure=[failure], timeout=[timeout])


def _flat_walk(**overrides):
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    out = None
    for step in range(1, 1001):
        kwargs = dict(x=-1.0 + 0.01 * step)
        kwargs.update(overrides)
        out = _step(qm, **kwargs)
    return qm, out


# ------------------------------------------------------------------- T-1
def test_zero_command_flat_walk_must_not_succeed():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        out = _step(qm, x=-1.0 + 0.01 * step, c_exec=0.0, v_heading=0.0,
                    speed=0.0)
    assert out['cruise_samples'][0] == 0
    assert out['endpoint'][0]
    assert out['quality_failure'][0]
    assert out['reasons'][0] == ['missing_window']


# ------------------------------------------------------------------- T-2
def _late_zero_run(delay):
    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['instant'])
    zero_at = 401 + delay
    out = None
    for step in range(1, 1401):
        stop = step >= 401
        c = 0.0 if step >= zero_at else 0.5
        out = _step(qm, x=-1.0, c_exec=c, v_heading=c, speed=c, stop=stop)
    return qm, out


def test_hold_anchored_to_actual_first_zero_matching_endpoint():
    qm, out = _late_zero_run(0)
    assert out['first_zero_step'][0] == 401
    assert out['hold_samples'][0] == 1000
    assert out['endpoint'][0] and out['quality_ok'][0]


def test_hold_anchored_to_actual_first_zero_late_zero_fails_at_planned_time():
    qm, out = _late_zero_run(100)
    assert out['first_zero_step'][0] == 501
    assert out['endpoint'][0]
    assert int(out['endpoint_step'][0]) == 1400
    assert out['hold_samples'][0] == 900
    assert out['quality_failure'][0]
    assert out['reasons'][0] == ['hold_incomplete']


# ------------------------------------------------------------------- T-6
def test_pitch_over15_latch_is_window_scoped():
    # outside the 2 s..20 s window: a 10-frame excursion must not latch
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        pitch = 20.0 if 20 <= step < 30 else 0.0
        out = _step(qm, x=-1.0 + 0.01 * step, pitch=pitch)
    assert not out['latched'][0]
    assert out['quality_ok'][0]

    # inside the window: the same excursion latches irreversibly
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        pitch = 20.0 if 300 <= step < 310 else 0.0
        out = _step(qm, x=-1.0 + 0.01 * step, pitch=pitch)
        if step == 308:
            assert not out['latched'][0]
        if step == 309:
            assert out['latched'][0]
            assert out['latched_reason'][0] == 'pitch_over15'
    assert out['quality_failure'][0]
    assert 'pitch_over15' in out['reasons'][0]


def test_latch_is_irreversible_and_survives_clean_frames():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        pitch = 20.0 if 300 <= step < 310 else 0.0
        out = _step(qm, x=-1.0 + 0.01 * step, pitch=pitch)
    assert out['endpoint'][0] and out['quality_failure'][0]
    assert out['latched'][0]


# --------------------------------------------------------------- priority
def test_same_frame_failure_priority():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        out = _step(qm, x=-1.0 + 0.01 * step, failure=step == 1000)
    assert out['physical_failure'][0]
    assert not out['quality_failure'][0]
    assert not out['quality_ok'][0]
    assert out['reasons'][0] == ['physical_failure']

    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        out = _step(qm, x=-1.0 + 0.01 * step, timeout=step == 1000)
    assert out['timeout'][0]
    assert not out['quality_failure'][0]
    assert out['reasons'][0] == ['timeout']


# ------------------------------------------------------- pending vs closed
def test_pending_versus_closed_windows():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 501):
        out = _step(qm, x=-1.0 + 0.01 * step)
    assert out['cruise_state'][0] == 'pending'
    assert out['cruise_samples'][0] == 401  # intervals 100..500
    assert np.isnan(out['cruise_pitch_rms'][0])
    for step in range(501, 1001):
        out = _step(qm, x=-1.0 + 0.01 * step)
    assert out['cruise_state'][0] == 'closed'
    assert out['cruise_samples'][0] == 900
    assert not np.isnan(out['cruise_pitch_rms'][0])


def test_stop_state_pending_then_closed():
    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['instant'])
    for step in range(1, 901):
        c = 0.5 if step < 401 else 0.0
        out = _step(qm, x=-1.0, c_exec=c, v_heading=c, speed=c,
                    stop=step >= 401)
    assert out['stop_state'][0] == 'pending'
    assert out['hold_samples'][0] == 500
    for step in range(901, 1401):
        out = _step(qm, x=-1.0, c_exec=0.0, v_heading=0.0, speed=0.0,
                    stop=True)
    assert out['stop_state'][0] == 'closed'
    assert out['hold_samples'][0] == 1000
    assert out['quality_ok'][0]


# --------------------------------------------------------------- reset
def test_reset_clears_only_selected_environment():
    qm = q.S2Quality(task=['flat_walk', 'flat_walk'], speed=[0.5, 0.5],
                     front=[0.0, 0.0], spawn_x=[-1.0, -1.0])
    for step in range(1, 301):
        out = qm.update(
            local_x=[-1.0 + 0.01 * step] * 2, local_y=[0.0, 0.0],
            pitch_deg=[0.0, 0.0], roll_deg=[0.0, 0.0],
            horizontal_speed=[0.5, 0.5], v_heading=[0.5, 0.5],
            c_exec=[0.5, 0.5], foot_support=[[True] * 4, [True] * 4],
            base_height=[0.4, 0.4], surface_height=[0.0, 0.0],
            foot_speed_xy=[[0.0] * 4, [0.0] * 4], angular_speed=[0.0, 0.0])
    assert out['cruise_samples'][0] == 201
    qm.reset([True, False])
    assert qm.step[0] == 0 and qm.cruise_samples[0] == 0
    assert qm.step[1] == 300 and qm.cruise_samples[1] == 201


# ------------------------------------------------------------- t=0 convention
def test_t0_row_is_not_an_interval():
    case = next(row for row in _document()['cases']
                if row['name'] == 'flat_walk_pass_baseline')
    frames = build_frames(case)
    rows = [dict(frames[0], n=0, t=0.0)] + frames
    out, _ = q.evaluate_frames(rows, task='flat_walk', speed=0.5,
                               spawn_x=-1.0)
    assert out['cruise_samples'][0] == 900
    assert int(out['endpoint_step'][0]) == 1000
    assert out['quality_ok'][0]


# --------------------------------------------------------- step-count table
def test_step_count_table_and_hold_samples():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        out = _step(qm, x=-1.0)
    assert qm.step[0] == 1000 and out['endpoint'][0]

    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['instant'])
    for step in range(1, 1401):
        c = 0.5 if step < 401 else 0.0
        out = _step(qm, x=-1.0, c_exec=c, v_heading=c, speed=c)
    assert qm.step[0] == 1400
    assert out['first_zero_step'][0] == 401
    assert out['hold_samples'][0] == 1000
    assert out['endpoint'][0]

    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['ramp_2s'])
    for step in range(1, 1501):
        if step <= 400:
            c = 0.5
        else:
            c = max(0.0, 0.5 * (1.0 - (step - 400) / 100.0))
        out = _step(qm, x=-1.0, c_exec=c, v_heading=c, speed=c)
    assert qm.step[0] == 1500
    assert out['first_zero_step'][0] == 500
    assert out['hold_samples'][0] == 1001  # zero frame + 1000 HOLD intervals
    assert out['endpoint'][0] and out['quality_ok'][0]


# ------------------------------------------------- Q037.3 pitch window rules
def test_pitch_nine_then_ten_inside_same_window():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        pitch = 20.0 if 300 <= step < 309 or 310 <= step < 320 else 0.0
        out = _step(qm, x=-1.0 + 0.01 * step, pitch=pitch)
        if step == 318:
            assert not out['latched'][0]  # 9 frames in the second run
        if step == 319:
            assert out['latched'][0]
            assert out['latched_reason'][0] == 'pitch_over15'
    assert out['quality_failure'][0]


def test_pitch_counter_does_not_span_window_boundary():
    # flat_stop cruise window is 2..8 s (intervals 100..399).  Nine frames at
    # the very end of the window plus many at 20 deg in HOLD must NOT latch.
    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['instant'])
    for step in range(1, 1401):
        stop = step >= 400
        c = 0.5 if not stop else 0.0
        pitch = 20.0 if (391 <= step <= 399 or step >= 400) else 0.0
        out = _step(qm, x=-1.0, pitch=pitch, c_exec=c, v_heading=c,
                    speed=c, stop=stop)
    assert 'pitch_over15' not in out['reasons'][0]
    assert out['latched_reason'][0] != 'pitch_over15'


def test_pitch_exactly_15_degrees_does_not_latch():
    qm = q.S2Quality(task=['flat_walk'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0])
    for step in range(1, 1001):
        out = _step(qm, x=-1.0 + 0.01 * step,
                    pitch=15.0 if 300 <= step < 350 else 0.0)
    assert not out['latched'][0]
    assert 'pitch_over15' not in out['reasons'][0]


# ------------------------------------------------------------ legacy L40
def test_legacy_l40_safe_summit_is_success_without_cruise_window():
    qm = q.S2Quality(task=['legacy_l40'], speed=[0.5], front=[8.0],
                     spawn_x=[-1.0], post_distance=[5.0])
    for step in range(1, 1200):
        x = -1.0 + 0.01 * step
        out = _step(qm, x=x, top=x >= 8.8, height=8.4)
        if out['endpoint'][0]:
            break
    assert out['endpoint'][0]
    assert out['cruise_state'][0] == 'none'
    assert out['quality_ok'][0] and not out['quality_failure'][0]


# ------------------------------------------------ recovery 25/150 boundary
def test_recovery_completes_25_stable_frames_within_deadline():
    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['instant'])
    for step in range(1, 500):
        stop = step >= 401
        c = 0.0 if stop else 0.5
        x = -1.0 + 0.01 * min(step, 400)
        out = _step(qm, x=x, speed=0.0 if stop else 0.5, c_exec=c,
                    v_heading=c, stop=stop)
    assert out['stable_done'][0]
    assert 401 <= int(out['stable_step'][0]) <= 426
    assert not out['latched'][0]


def test_recovery_deadline_latches_at_first_zero_plus_150():
    qm = q.S2Quality(task=['flat_stop'], speed=[0.5], front=[0.0],
                     spawn_x=[-1.0], stop_mode=['instant'])
    for step in range(1, 700):
        stop = step >= 401
        c = 0.0 if stop else 0.5
        x = -1.0 + 0.01 * min(step, 400)
        out = _step(qm, x=x, speed=0.2 if stop else 0.5, c_exec=c,
                    v_heading=c, stop=stop)
    assert out['first_zero_step'][0] == 401
    assert out['latched'][0] and out['latched_reason'][0] == 'recovery'
    assert int(qm.latched_step[0]) == 551
