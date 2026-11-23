"""S2 per-case absolute quality state machine (DSH-Q036, SDS §4.2/§6.5).

This module is the single implementation of the shared per-case quality
criterion used by training, gate and final acceptance.  It is pure Python /
numpy: the training adapter feeds already-measured per-env frames and the
gate/final drivers can reuse :func:`evaluate_frames` on recorded trajectories.

Conventions (pinned by DSH-Q036 B and the independent verifier reference
``logs/a2_v09/s2/q036_verify/quality_reference.py``):

* one interval = one completed ``0.02 s`` control interval; interval ``n`` is
  1-based and the row at ``t = 0.02*n`` is the state at the end of interval
  ``n``.  A logged ``t = 0`` row is the pre-step sample and is never counted;
* flat cruise windows are half-open: ``2.0 <= t < 20.0`` for ``flat_walk``
  (intervals 100..999, 900 frames) and ``2.0 <= t < 8.0`` for ``flat_stop``
  (intervals 100..399, 300 frames);
* stairs/platform windows open at the first interval with ``D_post >= 1.0 m``
  after the four-foot simultaneous platform entry (spawn+1 m for the platform
  direct start) and run through the stop-request interval / endpoint;
* only intervals with an actively executed positive command (``c_exec > 0``)
  can be cruise samples (T-1); an empty or <1 s window can never pass;
* the HOLD window is anchored to the **actual** first consumed ``c_exec == 0``
  (T-2), inclusive; a full HOLD is at least 1000 intervals and the stand window
  is its last 500 intervals;
* irreversible latches: ``|pitch| > 15 deg`` for 10 consecutive *window*
  frames (T-6), the inherited §6.5 stall detector (POST_WALK cruise only) and
  the recovery deadline (25 consecutive §5.3 stable frames within 150 intervals
  of the first zero);
* same-frame priority is physical failure > quality failure/timeout > success >
  standalone top event.
"""
from __future__ import annotations

import math

import numpy as np

TASKS = ('flat_walk', 'flat_stop', 'stairs_walk', 'stairs_stop',
         'platform_walk', 'platform_stop', 'legacy_l40')
FLAT_TASKS = ('flat_walk', 'flat_stop')
STAIR_TASKS = ('stairs_walk', 'stairs_stop', 'legacy_l40')
PLATFORM_TASKS = ('platform_walk', 'platform_stop')
STOP_TASKS = ('flat_stop', 'stairs_stop', 'platform_stop')
NONSTOP_TASKS = ('flat_walk', 'stairs_walk', 'platform_walk', 'legacy_l40')

CONTROL_DT = 0.02
FLAT_WALK_WINDOW_S = (2.0, 20.0)
FLAT_STOP_WINDOW_S = (2.0, 8.0)
FLAT_WALK_STEPS = 1000
FLAT_STOP_INSTANT_STEPS = 1400
FLAT_STOP_RAMP_STEPS = 1500
STOP_REQUEST_INTERVAL = 400
RAMP_STEPS = 100
HOLD_INTERVALS = 1000
STAND_WINDOW = 500
MIN_WINDOW_INTERVALS = 50

PITCH_RMS_MAX = 5.0
PITCH_P99_MAX = 10.0
SPEED_RMSE_MAX = 0.10
STOP_MEAN_SPEED = 0.05
STOP_PEAK_SPEED = 0.10
STOP_DISPLACEMENT = 0.10
STOP_TILT_DEG = 10.0
STOP_HEIGHT = 0.30
STOP_SUPPORT_RATIO = 0.90

PITCH_LATCH_DEG = 15.0
PITCH_LATCH_FRAMES = 10
STALL_WINDOW = 51
STALL_DX_FRACTION_PER_SECOND = 0.15
STALL_DX_THRESHOLD_SCALE = 1.0
STALL_HIGHWATER_DX = 0.03
STALL_CONSECUTIVE = 50
STALL_ACTIVE_COMMAND = 0.1
RECOVERY_DEADLINE = 150
STABLE_CONSECUTIVE = 25
STABLE_SPEED = 0.05
STABLE_OMEGA = 0.25
STABLE_HEIGHT = 0.30
STABLE_FOOT_SPEED = 0.05
SUPPORT_DEBOUNCE = 2

#: Canonical reason names (shared by training, gate and final).
REASON_NAMES = (
    'pitch_over15', 'stall', 'recovery',
    'missing_window', 'window_short', 'pitch_rms', 'pitch_p99', 'speed_rmse',
    'hold_incomplete', 'stand_mean_speed', 'stand_peak_speed',
    'stand_displacement', 'stand_tilt', 'stand_height', 'stand_support',
    'physical_failure', 'timeout', 'protection_expired')


class S2QualityError(ValueError):
    pass


# --------------------------------------------------------------------- shared
def cruise_metric_values(pitch_deg, speed, command):
    """Closed-window pitch/speed metrics from raw per-interval arrays."""
    pitch = np.asarray(pitch_deg, dtype=np.float64).reshape(-1)
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    command = np.asarray(command, dtype=np.float64).reshape(-1)
    if pitch.size == 0:
        return dict(frames=0, pitch_rms=None, pitch_p99=None, speed_rmse=None)
    if not (pitch.size == speed.size == command.size):
        raise S2QualityError('cruise metric arrays must have equal length')
    return dict(
        frames=int(pitch.size),
        pitch_rms=float(math.sqrt(float(np.mean(pitch ** 2)))),
        pitch_p99=float(np.quantile(np.abs(pitch), 0.99)),
        speed_rmse=float(math.sqrt(float(np.mean((speed - command) ** 2)))))


def cruise_reasons(metrics, *, min_frames=MIN_WINDOW_INTERVALS):
    """Canonical cruise-window reasons; empty means pass."""
    reasons = []
    frames = int(metrics.get('frames') or 0)
    if frames == 0:
        return ['missing_window']
    if frames < int(min_frames):
        return ['window_short']
    if metrics.get('pitch_rms') is None or metrics['pitch_rms'] > PITCH_RMS_MAX:
        reasons.append('pitch_rms')
    if metrics.get('pitch_p99') is None or metrics['pitch_p99'] > PITCH_P99_MAX:
        reasons.append('pitch_p99')
    if metrics.get('speed_rmse') is None or metrics['speed_rmse'] > SPEED_RMSE_MAX:
        reasons.append('speed_rmse')
    return reasons


def stand_metric_values(speeds, rolls, pitches, heights, supports, xs, ys):
    """Stand-window metrics from the last 500 HOLD intervals."""
    speeds = np.asarray(speeds, dtype=np.float64).reshape(-1)
    rolls = np.asarray(rolls, dtype=np.float64).reshape(-1)
    pitches = np.asarray(pitches, dtype=np.float64).reshape(-1)
    heights = np.asarray(heights, dtype=np.float64).reshape(-1)
    supports = np.asarray(supports, dtype=bool).reshape(-1)
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    size = speeds.size
    if size == 0:
        return dict(frames=0)
    displacement = np.hypot(xs - xs[0], ys - ys[0])
    finite = np.where(np.isfinite(heights), heights, np.inf)
    return dict(
        frames=int(size),
        mean_speed=float(np.mean(speeds)),
        peak_speed=float(np.max(speeds)),
        displacement=float(np.max(displacement)),
        support_ratio=float(np.mean(supports)),
        tilt_max=float(np.max(np.maximum(np.abs(rolls), np.abs(pitches)))),
        height_min=float(np.min(finite)))


def stand_reasons(metrics, *, hold_frames, hold_intervals=HOLD_INTERVALS):
    """Canonical stop/stand reasons; empty means pass."""
    if int(hold_frames) < int(hold_intervals):
        return ['hold_incomplete']
    if int(metrics.get('frames') or 0) < STAND_WINDOW:
        return ['hold_incomplete']
    reasons = []
    if metrics['mean_speed'] > STOP_MEAN_SPEED:
        reasons.append('stand_mean_speed')
    if metrics['peak_speed'] > STOP_PEAK_SPEED:
        reasons.append('stand_peak_speed')
    if metrics['displacement'] > STOP_DISPLACEMENT:
        reasons.append('stand_displacement')
    if metrics['tilt_max'] > STOP_TILT_DEG:
        reasons.append('stand_tilt')
    if metrics['height_min'] < STOP_HEIGHT:
        reasons.append('stand_height')
    if metrics['support_ratio'] < STOP_SUPPORT_RATIO:
        reasons.append('stand_support')
    return reasons


def acceptance_reasons(metrics, *, stop=False):
    """Gate/final acceptance view of the same shared metric contract.

    Keeps the public ``s2_acceptance.quality_reasons`` reason vocabulary while
    the thresholds and comparison rules come from this module only.
    """
    reasons = []
    if metrics.get('valid_cruise_window') is not True:
        reasons.append('missing_cruise_window')
    if metrics.get('pitch_rms_deg') is None or metrics['pitch_rms_deg'] > PITCH_RMS_MAX:
        reasons.append('pitch_rms')
    if metrics.get('pitch_p99_deg') is None or metrics['pitch_p99_deg'] > PITCH_P99_MAX:
        reasons.append('pitch_p99')
    if (metrics.get('pitch_over15_max_frames', 0) * CONTROL_DT
            >= PITCH_LATCH_FRAMES * CONTROL_DT):
        reasons.append('pitch_over15')
    if metrics.get('speed_rmse') is None or metrics['speed_rmse'] > SPEED_RMSE_MAX:
        reasons.append('speed_rmse')
    if metrics.get('nonfoot_contact', False):
        reasons.append('nonfoot_contact')
    if metrics.get('cruise_stall_latched', False):
        reasons.append('cruise_stall')
    if stop:
        if (metrics.get('recovery_seconds') is None
                or metrics['recovery_seconds'] > RECOVERY_DEADLINE * CONTROL_DT):
            reasons.append('recovery')
        if (metrics.get('stand_mean_speed') is None
                or metrics['stand_mean_speed'] > STOP_MEAN_SPEED):
            reasons.append('stand_mean_speed')
        if (metrics.get('stand_peak_speed') is None
                or metrics['stand_peak_speed'] > STOP_PEAK_SPEED):
            reasons.append('stand_peak_speed')
        if (metrics.get('stand_max_displacement') is None
                or metrics['stand_max_displacement'] > STOP_DISPLACEMENT):
            reasons.append('stand_displacement')
        if (metrics.get('stand_support_fraction') is None
                or metrics['stand_support_fraction'] < STOP_SUPPORT_RATIO):
            reasons.append('stand_support')
        for key in ('stand_roll_abs_max_deg', 'stand_pitch_abs_max_deg'):
            if metrics.get(key) is None or metrics[key] > STOP_TILT_DEG:
                reasons.append('stand_attitude')
        if metrics.get('stand_height_min') is None or metrics['stand_height_min'] < STOP_HEIGHT:
            reasons.append('stand_height')
    return reasons


def _as_array(value, n, dtype, name, default=0.0):
    if value is None:
        return np.full(n, default, dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(n, array.item(), dtype=dtype)
    if array.shape[0] != n:
        raise S2QualityError(f'{name} length {array.shape[0]} != {n}')
    return array


class S2Quality:
    """Vectorized per-env S2 quality state machine (numpy only)."""

    def __init__(self, *, task, speed, front=0.0, spawn_x=-1.0,
                 post_distance=5.0, platform_length=0.0, platform_width=8.0,
                 stop_mode='none', legacy=None, control_dt=CONTROL_DT,
                 flat_walk_steps=FLAT_WALK_STEPS,
                 cruise_after_steps=STOP_REQUEST_INTERVAL,
                 ramp_steps=RAMP_STEPS, hold_steps=HOLD_INTERVALS,
                 stand_steps=STAND_WINDOW,
                 min_window_samples=MIN_WINDOW_INTERVALS):
        self.task = np.asarray(task, dtype=object)
        n = int(self.task.shape[0])
        if n == 0:
            raise S2QualityError('empty task batch')
        unknown = sorted(set(str(v) for v in self.task) - set(TASKS))
        if unknown:
            raise S2QualityError(f'unknown S2 task(s): {unknown}')
        self.num_envs = n
        self.speed = _as_array(speed, n, np.float64, 'speed')
        self.front = _as_array(front, n, np.float64, 'front')
        self.spawn_x = _as_array(spawn_x, n, np.float64, 'spawn_x')
        self.post = _as_array(post_distance, n, np.float64, 'post_distance')
        self.platform_length = _as_array(platform_length, n, np.float64,
                                         'platform_length')
        self.platform_width = _as_array(platform_width, n, np.float64,
                                        'platform_width')
        self.stop_mode = _as_array(stop_mode, n, object, 'stop_mode',
                                   default='none')
        self.legacy = (np.asarray([str(t) == 'legacy_l40' for t in self.task],
                                  dtype=bool) if legacy is None
                       else np.asarray(legacy, dtype=bool))
        self.control_dt = float(control_dt)
        self.flat_walk_steps = int(flat_walk_steps)
        self.cruise_after_steps = int(cruise_after_steps)
        self.ramp_steps = int(ramp_steps)
        self.hold_steps = int(hold_steps)
        self.stand_steps = int(stand_steps)
        self.min_window_samples = int(min_window_samples)
        self.is_flat = np.isin(self.task, FLAT_TASKS)
        self.is_stairs = np.isin(self.task, STAIR_TASKS)
        self.is_platform = np.isin(self.task, PLATFORM_TASKS)
        self.is_stop = np.isin(self.task, STOP_TASKS)
        self.is_legacy = self.legacy
        self.is_climb = np.isin(self.task, ('stairs_walk', 'stairs_stop',
                                            'legacy_l40'))
        self.ramp_len = np.where(
            self.is_stop & (self.stop_mode == 'ramp_2s'),
            self.ramp_steps, 0).astype(np.int64)
        self.flat_end_step = np.where(self.task == 'flat_walk',
                                      self.flat_walk_steps,
                                      self.cruise_after_steps)
        self.planned_stop_end = (self.cruise_after_steps + self.ramp_len
                                 + self.hold_steps)
        self._max_hold = self.hold_steps + 2
        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self):
        n = self.num_envs
        self.step = np.zeros(n, dtype=np.int64)
        self.latched = np.zeros(n, dtype=bool)
        self.latched_reason = np.full(n, '', dtype=object)
        self.latched_step = np.full(n, -1, dtype=np.int64)
        self.pitch_run = np.zeros(n, dtype=np.int64)
        self.stall_x = np.zeros((n, STALL_WINDOW), dtype=np.float64)
        self.stall_m = np.zeros((n, STALL_WINDOW), dtype=np.float64)
        self.stall_count = np.zeros(n, dtype=np.int64)
        self.stall_consec = np.zeros(n, dtype=np.int64)
        self.top_step = np.full(n, -1, dtype=np.int64)
        self.entry_step = np.full(n, -1, dtype=np.int64)
        self.entry_x = np.zeros(n, dtype=np.float64)
        self.cruise_start_step = np.full(n, -1, dtype=np.int64)
        self.cruise_start_x = np.zeros(n, dtype=np.float64)
        self.cruise_samples = np.zeros(n, dtype=np.int64)
        self.cruise_closed = np.zeros(n, dtype=bool)
        self.cruise_close_step = np.full(n, -1, dtype=np.int64)
        self.cruise_pitch = np.zeros((n, max(self.flat_walk_steps, 4096)),
                                     dtype=np.float64)
        self.cruise_speed_err_sq = np.zeros(n, dtype=np.float64)
        self.cruise_pitch_rms = np.full(n, np.nan, dtype=np.float64)
        self.cruise_pitch_p99 = np.full(n, np.nan, dtype=np.float64)
        self.cruise_speed_rmse = np.full(n, np.nan, dtype=np.float64)
        self.support_count = np.zeros((n, 4), dtype=np.int64)
        self.stop_request_step = np.full(n, -1, dtype=np.int64)
        self.first_zero_step = np.full(n, -1, dtype=np.int64)
        self.hold_count = np.zeros(n, dtype=np.int64)
        self.stable_run = np.zeros(n, dtype=np.int64)
        self.stable_done = np.zeros(n, dtype=bool)
        self.stable_step = np.full(n, -1, dtype=np.int64)
        self.stop_closed = np.zeros(n, dtype=bool)
        self.hold_speed = np.zeros((n, self._max_hold), dtype=np.float64)
        self.hold_roll = np.zeros((n, self._max_hold), dtype=np.float64)
        self.hold_pitch = np.zeros((n, self._max_hold), dtype=np.float64)
        self.hold_height = np.zeros((n, self._max_hold), dtype=np.float64)
        self.hold_support = np.zeros((n, self._max_hold), dtype=bool)
        self.hold_x = np.zeros((n, self._max_hold), dtype=np.float64)
        self.hold_y = np.zeros((n, self._max_hold), dtype=np.float64)
        self.stand_mean_speed = np.full(n, np.nan, dtype=np.float64)
        self.stand_peak_speed = np.full(n, np.nan, dtype=np.float64)
        self.stand_max_displacement = np.full(n, np.nan, dtype=np.float64)
        self.stand_max_roll = np.full(n, np.nan, dtype=np.float64)
        self.stand_max_pitch = np.full(n, np.nan, dtype=np.float64)
        self.stand_min_height = np.full(n, np.nan, dtype=np.float64)
        self.stand_support_ratio = np.full(n, np.nan, dtype=np.float64)
        self.hold_end_step = np.full(n, -1, dtype=np.int64)
        self.endpoint = np.zeros(n, dtype=bool)
        self.endpoint_step = np.full(n, -1, dtype=np.int64)
        self.endpoint_done = np.zeros(n, dtype=bool)
        self.quality_ok = np.ones(n, dtype=bool)
        self.quality_failure = np.zeros(n, dtype=bool)
        self.quality_reason = np.full(n, '', dtype=object)
        self.reasons = np.empty(n, dtype=object)
        for i in range(n):
            self.reasons[i] = []
        self.physical_failure = np.zeros(n, dtype=bool)
        self.timeout = np.zeros(n, dtype=bool)

    def reset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise S2QualityError('reset mask shape mismatch')
        if not mask.any():
            return
        self.step[mask] = 0
        self.latched[mask] = False
        self.latched_reason[mask] = ''
        self.latched_step[mask] = -1
        self.pitch_run[mask] = 0
        self.stall_x[mask] = 0
        self.stall_m[mask] = 0
        self.stall_count[mask] = 0
        self.stall_consec[mask] = 0
        self.top_step[mask] = -1
        self.entry_step[mask] = -1
        self.entry_x[mask] = 0
        self.cruise_start_step[mask] = -1
        self.cruise_start_x[mask] = 0
        self.cruise_samples[mask] = 0
        self.cruise_closed[mask] = False
        self.cruise_close_step[mask] = -1
        self.cruise_pitch[mask] = 0
        self.cruise_speed_err_sq[mask] = 0
        self.cruise_pitch_rms[mask] = np.nan
        self.cruise_pitch_p99[mask] = np.nan
        self.cruise_speed_rmse[mask] = np.nan
        self.support_count[mask] = 0
        self.stop_request_step[mask] = -1
        self.first_zero_step[mask] = -1
        self.hold_count[mask] = 0
        self.stable_run[mask] = 0
        self.stable_done[mask] = False
        self.stable_step[mask] = -1
        self.stop_closed[mask] = False
        self.hold_speed[mask] = 0
        self.hold_roll[mask] = 0
        self.hold_pitch[mask] = 0
        self.hold_height[mask] = 0
        self.hold_support[mask] = False
        self.hold_x[mask] = 0
        self.hold_y[mask] = 0
        self.stand_mean_speed[mask] = np.nan
        self.stand_peak_speed[mask] = np.nan
        self.stand_max_displacement[mask] = np.nan
        self.stand_max_roll[mask] = np.nan
        self.stand_max_pitch[mask] = np.nan
        self.stand_min_height[mask] = np.nan
        self.stand_support_ratio[mask] = np.nan
        self.hold_end_step[mask] = -1
        self.endpoint[mask] = False
        self.endpoint_step[mask] = -1
        self.endpoint_done[mask] = False
        self.quality_ok[mask] = True
        self.quality_failure[mask] = False
        self.quality_reason[mask] = ''
        self.physical_failure[mask] = False
        self.timeout[mask] = False
        for i in np.nonzero(mask)[0].tolist():
            self.reasons[i] = []

    # ------------------------------------------------------------------ latch
    def _latch(self, mask, reason, step):
        mask = np.asarray(mask, dtype=bool) & ~self.latched
        if not mask.any():
            return
        self.latched[mask] = True
        self.latched_reason[mask] = reason
        self.latched_step[mask] = np.asarray(step)[mask]

    def _latch_one(self, env, reason, step):
        if self.latched[env]:
            return
        self.latched[env] = True
        self.latched_reason[env] = reason
        self.latched_step[env] = int(step)

    # ----------------------------------------------------------------- update
    def update(self, *, local_x, local_y=None, pitch_deg, roll_deg,
               horizontal_speed, v_heading, c_exec, c_request=None,
               stop_requested=None, four_feet_entry=None, top_arrived=None,
               base_height=None, surface_height=None, foot_support=None,
               foot_speed_xy=None, angular_speed=None, failure=False,
               timeout=False):
        """Advance exactly one completed 0.02 s control interval.

        ``c_exec`` is the command actually consumed by this Actor tick.  Stop
        request timing is derived from the same measured frames (time/distance)
        so the quality machine does not depend on the training lifecycle's
        one-frame command pipeline (T-8).  ``stop_requested`` is accepted for
        diagnostics/consistency only.
        """
        n = self.num_envs
        self.step += 1
        step = self.step
        local_x = _as_array(local_x, n, np.float64, 'local_x')
        local_y = _as_array(local_y, n, np.float64, 'local_y')
        pitch = _as_array(pitch_deg, n, np.float64, 'pitch_deg')
        roll = _as_array(roll_deg, n, np.float64, 'roll_deg')
        speed = _as_array(horizontal_speed, n, np.float64, 'horizontal_speed')
        v_heading = _as_array(v_heading, n, np.float64, 'v_heading')
        c_exec = _as_array(c_exec, n, np.float64, 'c_exec')
        failure = _as_array(failure, n, bool, 'failure')
        timeout = _as_array(timeout, n, bool, 'timeout')
        top_arrived = _as_array(top_arrived, n, bool, 'top_arrived')
        four_feet_entry = _as_array(four_feet_entry, n, bool,
                                    'four_feet_entry')
        if base_height is not None and surface_height is not None:
            height = (_as_array(base_height, n, np.float64, 'base_height')
                      - _as_array(surface_height, n, np.float64,
                                  'surface_height'))
        else:
            height = np.full(n, np.nan, dtype=np.float64)
        if angular_speed is not None:
            angular_speed = _as_array(angular_speed, n, np.float64,
                                      'angular_speed')
        else:
            angular_speed = np.zeros(n, dtype=np.float64)
        if foot_speed_xy is not None:
            foot_speed_xy = np.asarray(foot_speed_xy, dtype=np.float64)
            if foot_speed_xy.shape != (n, 4):
                raise S2QualityError('foot_speed_xy must be [n,4]')
            foot_speed_mean = foot_speed_xy.mean(-1)
        else:
            foot_speed_mean = np.zeros(n, dtype=np.float64)
        if foot_support is not None:
            # §5.3 requires two consecutive control frames of real top-face
            # support; the caller supplies that already-debounced per-foot
            # flag (training debounces in s2_train_real, gate/final reuse the
            # recorded support channel).
            valid_support = np.asarray(foot_support, dtype=bool)
            if valid_support.shape != (n, 4):
                raise S2QualityError('foot_support must be [n,4]')
        else:
            valid_support = np.ones((n, 4), dtype=bool)

        self.physical_failure = failure
        self.timeout = timeout
        top_new = top_arrived & ~failure & ~timeout & (self.top_step < 0)
        self.top_step[top_new] = step[top_new]

        # ------------------------------------------------ four-foot entry
        new_entry = (four_feet_entry & ~self.is_platform & ~self.is_flat
                     & ~self.is_legacy & (self.entry_step < 0))
        if new_entry.any():
            self.entry_step[new_entry] = step[new_entry]
            self.entry_x[new_entry] = local_x[new_entry]

        # -------------------------------------------------- stop request
        request_now = np.zeros(n, dtype=bool)
        flat_req = ((self.task == 'flat_stop')
                    & (step == self.cruise_after_steps)
                    & (self.stop_request_step < 0))
        request_now |= flat_req
        distance = self._distance(local_x)
        dist_req = (np.isin(self.task, ('stairs_stop', 'platform_stop'))
                    & (distance >= self.post) & (self.stop_request_step < 0)
                    & ~failure & ~timeout)
        request_now |= dist_req
        self.stop_request_step = np.where(request_now, step,
                                          self.stop_request_step)

        # ------------------------------------------ actual first c_exec == 0
        zero_now = ((self.stop_request_step >= 0)
                    & (step > self.stop_request_step)
                    & (self.first_zero_step < 0) & (c_exec <= 1e-12))
        self.first_zero_step = np.where(zero_now, step, self.first_zero_step)

        # -------------------------------------------------- cruise window
        t = step.astype(np.float64) * self.control_dt
        cruise = c_exec > 1e-7
        flat_open = ((self.task == 'flat_walk')
                     & (t >= FLAT_WALK_WINDOW_S[0]) & (t < FLAT_WALK_WINDOW_S[1]))
        flat_stop_open = ((self.task == 'flat_stop')
                          & (t >= FLAT_STOP_WINDOW_S[0])
                          & (t < FLAT_STOP_WINDOW_S[1]))
        stairs_open = (self.is_climb & ~self.is_legacy
                       & (self.entry_step >= 0)
                       & (local_x - self.entry_x >= 1.0))
        platform_open = self.is_platform & (local_x - self.spawn_x >= 1.0)
        stop_before_request = ((self.stop_request_step < 0)
                               | (step <= self.stop_request_step))
        in_window = ((flat_open | flat_stop_open)
                     | ((stairs_open | platform_open)
                        & stop_before_request))
        in_window &= (cruise & ~self.is_legacy & ~failure & ~timeout
                      & ~self.cruise_closed)
        newly = in_window & (self.cruise_start_step < 0)
        if newly.any():
            self.cruise_start_step[newly] = step[newly]
            self.cruise_start_x[newly] = local_x[newly]
        if in_window.any():
            index = np.nonzero(in_window)[0]
            counts = self.cruise_samples[index]
            writable = counts < self.cruise_pitch.shape[1]
            index = index[writable]
            counts = counts[writable]
            self.cruise_pitch[index, counts] = pitch[index]
            self.cruise_speed_err_sq[index] += (
                (v_heading[index] - c_exec[index]) ** 2)
            self.cruise_samples[index] = counts + 1

        # window-scoped pitch latch (T-6)
        over = in_window & (np.abs(pitch) > PITCH_LATCH_DEG)
        self.pitch_run = np.where(over, self.pitch_run + 1, 0)
        self._latch(self.pitch_run >= PITCH_LATCH_FRAMES, 'pitch_over15', step)

        # ------------------------------------------------ §6.5 stall latch
        stall_active = (in_window & (self.top_step >= 0)
                        & (c_exec > STALL_ACTIVE_COMMAND))
        self._update_stall(local_x, c_exec, stall_active, step)

        # --------------------------------------- §5.3 stable-stand recovery
        stable = ((speed <= STABLE_SPEED)
                  & (np.abs(roll) <= STOP_TILT_DEG)
                  & (np.abs(pitch) <= STOP_TILT_DEG)
                  & (angular_speed <= STABLE_OMEGA)
                  & np.isfinite(height) & (height >= STABLE_HEIGHT)
                  & (foot_speed_mean <= STABLE_FOOT_SPEED)
                  & valid_support.all(-1))
        in_recovery = self.first_zero_step >= 0
        self.stable_run = np.where(in_recovery & stable,
                                   self.stable_run + 1, 0)
        completed = (in_recovery & (self.stable_run >= STABLE_CONSECUTIVE)
                     & ~self.stable_done)
        if completed.any():
            self.stable_done[completed] = True
            self.stable_step[completed] = step[completed]
        overdue = (in_recovery & ~self.stable_done
                   & (step - self.first_zero_step >= RECOVERY_DEADLINE))
        self._latch(overdue, 'recovery', step)

        # ------------------------------------------------- HOLD window (T-2)
        in_hold = (self.first_zero_step >= 0) & ~failure & ~timeout
        if in_hold.any():
            index = np.nonzero(in_hold)[0]
            slots = self.hold_count[index]
            writable = slots < self._max_hold
            index, slots = index[writable], slots[writable]
            self.hold_speed[index, slots] = speed[index]
            self.hold_roll[index, slots] = roll[index]
            self.hold_pitch[index, slots] = pitch[index]
            self.hold_height[index, slots] = height[index]
            self.hold_support[index, slots] = valid_support[index].all(-1)
            self.hold_x[index, slots] = local_x[index]
            self.hold_y[index, slots] = local_y[index]
            self.hold_count[index] = slots + 1

        # ------------------------------------------------------- endpoint
        endpoint = self._endpoint(local_x)
        endpoint_now = endpoint & ~self.endpoint_done
        self.endpoint = endpoint_now
        self.endpoint_step[endpoint_now] = step[endpoint_now]
        self.endpoint_done |= endpoint_now

        # close cruise / stop windows when the evaluation is complete
        close_cruise = endpoint_now | (
            self.is_stop & (self.stop_request_step >= 0)
            & ~self.cruise_closed)
        self._close_cruise(close_cruise & ~self.cruise_closed, step)
        close_stop = endpoint_now & self.is_stop & ~self.stop_closed
        self._close_stop(close_stop, step)

        # ------------------------------------------------------- decision
        self._decide(step)
        return self.result()

    # --------------------------------------------------------------- helpers
    def _distance(self, local_x):
        stairs = np.isin(self.task, ('stairs_walk', 'stairs_stop',
                                     'legacy_l40'))
        zero = np.where(self.entry_step >= 0, self.entry_x, local_x)
        return np.where(self.is_platform, local_x - self.spawn_x,
                        np.where(stairs, local_x - zero,
                                 local_x - self.spawn_x))

    def _endpoint(self, local_x):
        self._current_x = np.asarray(local_x, dtype=np.float64)
        endpoint = np.zeros(self.num_envs, dtype=bool)
        endpoint |= (self.task == 'flat_walk') & (
            self.step >= self.flat_walk_steps)
        endpoint |= (self.task == 'flat_stop') & (
            self.step >= self.planned_stop_end)
        # Endpoint distance matches S2Lifecycle exactly: platform direct walks
        # from spawn, stairs walk from the platform front edge (D_post).
        dist = np.where(
            self.is_platform, local_x - self.spawn_x,
            np.where(np.isin(self.task, ('stairs_walk', 'platform_walk')),
                     local_x - self.front, local_x - self.spawn_x))
        endpoint |= (np.isin(self.task, ('stairs_walk', 'platform_walk'))
                     & (dist >= self.post))
        endpoint |= self.is_legacy & (self.top_step >= 0)
        endpoint |= (np.isin(self.task, ('stairs_stop', 'platform_stop'))
                     & (self.first_zero_step >= 0)
                     & (self.step >= self.first_zero_step + self.hold_steps - 1))
        return endpoint

    def _close_cruise(self, close, step):
        if not close.any():
            return
        self.cruise_closed |= close
        self.cruise_close_step[close] = step[close]
        for env in np.nonzero(close)[0]:
            count = int(self.cruise_samples[env])
            if count:
                values = self.cruise_pitch[env, :count].astype(np.float64)
                self.cruise_pitch_rms[env] = float(
                    np.sqrt(np.mean(values ** 2)))
                self.cruise_pitch_p99[env] = float(
                    np.quantile(np.abs(values), 0.99))
                self.cruise_speed_rmse[env] = float(np.sqrt(
                    self.cruise_speed_err_sq[env] / count))

    def _close_stop(self, close, step):
        if not close.any():
            return
        self.stop_closed |= close
        for env in np.nonzero(close)[0]:
            count = int(self.hold_count[env])
            if count:
                take = min(STAND_WINDOW, count)
                values = stand_metric_values(
                    self.hold_speed[env, count - take:count],
                    self.hold_roll[env, count - take:count],
                    self.hold_pitch[env, count - take:count],
                    self.hold_height[env, count - take:count],
                    self.hold_support[env, count - take:count],
                    self.hold_x[env, count - take:count],
                    self.hold_y[env, count - take:count])
                self.stand_mean_speed[env] = values.get('mean_speed', np.nan)
                self.stand_peak_speed[env] = values.get('peak_speed', np.nan)
                self.stand_max_displacement[env] = values.get(
                    'displacement', np.nan)
                self.stand_max_roll[env] = float(np.max(
                    np.abs(self.hold_roll[env, count - take:count]))) \
                    if take else np.nan
                self.stand_max_pitch[env] = float(np.max(
                    np.abs(self.hold_pitch[env, count - take:count]))) \
                    if take else np.nan
                self.stand_min_height[env] = values.get('height_min', np.nan)
                self.stand_support_ratio[env] = values.get(
                    'support_ratio', np.nan)

    def _decide(self, step):
        n = self.num_envs
        current = self.endpoint
        # pre-endpoint quality_ok is "not latched"; the endpoint consumes it.
        self.quality_ok = np.where(current, False, ~self.latched & ~self.physical_failure)
        self.quality_failure = np.zeros(n, dtype=bool)
        if not current.any():
            self.quality_reason = np.where(
                current, '', np.where(self.latched, self.latched_reason, ''))
            return
        for env in np.nonzero(current)[0].tolist():
            reasons = []
            if self.physical_failure[env]:
                reasons = ['physical_failure']
            elif self.timeout[env]:
                reasons = ['timeout']
            else:
                if self.latched[env]:
                    reasons.append(self.latched_reason[env])
                if self.is_legacy[env]:
                    pass
                else:
                    if int(self.cruise_samples[env]) == 0:
                        reasons.append('missing_window')
                    else:
                        metrics = dict(
                            frames=int(self.cruise_samples[env]),
                            pitch_rms=float(self.cruise_pitch_rms[env]),
                            pitch_p99=float(self.cruise_pitch_p99[env]),
                            speed_rmse=float(self.cruise_speed_rmse[env]))
                        reasons.extend(cruise_reasons(
                            metrics, min_frames=self.min_window_samples))
                    if self.is_stop[env]:
                        reasons.extend(self._stop_reasons(env))
            reasons = sorted(set(reasons), key=REASON_NAMES.index)
            self.reasons[env] = reasons
            self.quality_reason[env] = '+'.join(reasons)
            failed = (not self.physical_failure[env]
                      and not self.timeout[env] and bool(reasons))
            self.quality_failure[env] = failed
            self.quality_ok[env] = bool(
                not self.physical_failure[env] and not self.timeout[env]
                and not reasons)

    def _stop_reasons(self, env):
        hold = int(self.hold_count[env])
        if hold < self.hold_steps:
            return ['hold_incomplete']
        take = min(STAND_WINDOW, hold)
        metrics = stand_metric_values(
            self.hold_speed[env, hold - take:hold],
            self.hold_roll[env, hold - take:hold],
            self.hold_pitch[env, hold - take:hold],
            self.hold_height[env, hold - take:hold],
            self.hold_support[env, hold - take:hold],
            self.hold_x[env, hold - take:hold],
            self.hold_y[env, hold - take:hold])
        return stand_reasons(metrics, hold_frames=hold,
                             hold_intervals=self.hold_steps)

    def _update_stall(self, local_x, c_exec, active, step):
        fresh = self.stall_count == 0
        if fresh.any():
            self.stall_x[fresh] = local_x[fresh, None]
            self.stall_m[fresh] = local_x[fresh, None]
            self.stall_count[fresh] = 1
        high = np.maximum(self.stall_m[:, -1], local_x)
        self.stall_x = np.concatenate((self.stall_x[:, 1:],
                                       local_x[:, None]), axis=1)
        self.stall_m = np.concatenate((self.stall_m[:, 1:], high[:, None]),
                                      axis=1)
        self.stall_count = np.minimum(self.stall_count + 1, STALL_WINDOW)
        ready = self.stall_count >= STALL_WINDOW
        dx = self.stall_x[:, -1] - self.stall_x[:, 0]
        dm = self.stall_m[:, -1] - self.stall_m[:, 0]
        threshold = (STALL_DX_FRACTION_PER_SECOND * c_exec
                     * STALL_DX_THRESHOLD_SCALE)
        qualifying = (active & ready & (dx < threshold)
                      & (dm < STALL_HIGHWATER_DX))
        self.stall_consec = np.where(qualifying, self.stall_consec + 1, 0)
        self._latch(qualifying
                    & (self.stall_consec >= STALL_CONSECUTIVE), 'stall', step)

    # ---------------------------------------------------------------- result
    def result(self):
        return dict(
            quality_ok=self.quality_ok.copy(),
            quality_failure=self.quality_failure.copy(),
            quality_reason=self.quality_reason.copy(),
            reasons=list(self.reasons),
            physical_failure=self.physical_failure.copy(),
            timeout=self.timeout.copy(),
            latched=self.latched.copy(),
            latched_reason=self.latched_reason.copy(),
            step=self.step.copy(),
            endpoint=self.endpoint.copy(),
            endpoint_step=self.endpoint_step.copy(),
            cruise_state=np.where(self.is_legacy, 'none',
                                  np.where(self.cruise_closed, 'closed',
                                           'pending')).astype(object),
            stop_state=np.where(self.is_stop,
                                np.where(self.stop_closed, 'closed', 'pending'),
                                'none').astype(object),
            four_feet_entry_step=self.entry_step.copy(),
            four_feet_entry_x=self.entry_x.copy(),
            cruise_start_step=self.cruise_start_step.copy(),
            cruise_close_step=self.cruise_close_step.copy(),
            cruise_samples=self.cruise_samples.copy(),
            cruise_pitch_rms=self.cruise_pitch_rms.copy(),
            cruise_pitch_p99=self.cruise_pitch_p99.copy(),
            cruise_speed_rmse=self.cruise_speed_rmse.copy(),
            stop_request_step=self.stop_request_step.copy(),
            first_zero_step=self.first_zero_step.copy(),
            hold_samples=self.hold_count.copy(),
            stable_done=self.stable_done.copy(),
            stable_step=self.stable_step.copy(),
            stand_samples=np.minimum(self.hold_count, self.stand_steps)
            .astype(np.int64),
            stop_mean_speed=self.stand_mean_speed.copy(),
            stop_peak_speed=self.stand_peak_speed.copy(),
            stop_max_displacement=self.stand_max_displacement.copy(),
            stop_max_roll=self.stand_max_roll.copy(),
            stop_max_pitch=self.stand_max_pitch.copy(),
            stop_min_height=self.stand_min_height.copy(),
            stop_support_ratio=self.stand_support_ratio.copy(),
        )


# --------------------------------------------------------------- offline path
def _support_rows(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return [[bool(value)] * 4]
    array = np.asarray(value, dtype=bool)
    if array.ndim == 1 and array.size == 4:
        return array.reshape(1, 4)
    return array


def _foot_speed_rows(value):
    if value is None:
        return None
    if np.isscalar(value):
        return [[float(value)] * 4]
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.size == 4:
        return array.reshape(1, 4)
    return array


def frame_to_update(frame):
    """Map one recorded frame to :meth:`S2Quality.update` keyword arguments."""
    return dict(
        local_x=frame.get('local_x', frame.get('x')),
        local_y=frame.get('local_y', frame.get('y', 0.0)),
        pitch_deg=frame.get('pitch_deg', 0.0),
        roll_deg=frame.get('roll_deg', 0.0),
        horizontal_speed=frame.get('horizontal_speed', frame.get('speed', 0.0)),
        v_heading=frame.get('v_heading', frame.get('speed', 0.0)),
        c_exec=frame.get('c_exec', 0.0),
        c_request=frame.get('c_request'),
        stop_requested=frame.get('stop_requested', False),
        four_feet_entry=frame.get('four_feet_entry',
                                  frame.get('four_feet_platform', False)),
        top_arrived=frame.get('top_arrived', frame.get('top', False)),
        base_height=frame.get('base_height', frame.get('height', 0.4)),
        surface_height=frame.get('surface_height', 0.0),
        foot_support=_support_rows(frame.get('foot_support',
                                              frame.get('support'))),
        foot_speed_xy=_foot_speed_rows(frame.get('foot_speed_xy',
                                                 frame.get('foot_speed'))),
        angular_speed=frame.get('angular_speed', frame.get('omega')),
        failure=frame.get('failure', False),
        timeout=frame.get('timeout', False))


def evaluate_frames(frames, *, task, speed, front=0.0, spawn_x=-1.0,
                    post_distance=5.0, platform_length=0.0,
                    platform_width=8.0, stop_mode='none', legacy=None):
    """Offline evaluation of one recorded episode through the shared machine.

    ``frames`` is an iterable of per-interval mappings (see
    :func:`frame_to_update`).  Returns the final machine dict plus the metrics
    that gate/final acceptance needs.  The training path uses the incremental
    :class:`S2Quality` instance directly; both share every check.
    """
    machine = S2Quality(task=[task], speed=[speed], front=[front],
                        spawn_x=[spawn_x], post_distance=[post_distance],
                        platform_length=[platform_length],
                        platform_width=[platform_width],
                        stop_mode=[stop_mode], legacy=[legacy])
    out = machine.result()
    for frame in frames:
        # Q036 B2: a logged t=0 row is the pre-step sample, never an interval.
        if int(frame.get('n', 0) or 0) == 0 or float(frame.get('t', 1) or 0) == 0.0:
            continue
        out = machine.update(**frame_to_update(frame))
        if out['endpoint'][0]:
            break
    return out, machine
