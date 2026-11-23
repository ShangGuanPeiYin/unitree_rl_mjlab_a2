"""Vectorized S2 training lifecycle and quality admission (CPU-testable)."""
from __future__ import annotations

import numpy as np

TASK_CODES = dict(flat_walk=0, flat_stop=1, stairs_walk=2, platform_walk=3,
                  stairs_stop=4, platform_stop=5, legacy_l40=6)
CRUISE_STEPS = 400  # flat_stop: 8 s of walking at 0.02 s control
RAMP_STEPS = 100
HOLD_STEPS = 1000
FLAT_STOP_STEP = CRUISE_STEPS  # retained name for the old tests/readers
#: DSH-Q036 T-5: the exact ``c_request`` semantics (whether the external
#: request becomes zero at the stop request or keeps the cruise target) is
#: still with the planner.  Keep the current WIP behaviour in ONE switch so a
#: ruling can flip it without touching the executed-command timeline.  Tests
#: must read this constant instead of hardwiring 0.5/0.0.
C_REQUEST_ZERO_AT_STOP_REQUEST = True
PHASES = ('APPROACH', 'CLIMB', 'TOP_TRANSITION', 'POST_WALK', 'DECEL', 'HOLD')
APPROACH, CLIMB, TOP_TRANSITION, POST_WALK, DECEL, HOLD = range(6)


class S2Lifecycle:
    def __init__(self, *, task, post_distance, speed, stop_mode='none',
                 legacy=False, front=0.0, spawn_x=-1.0,
                 ramp_steps=RAMP_STEPS, hold_steps=HOLD_STEPS,
                 completion_steps=None):
        self.task = np.asarray(task, dtype=object)
        n = self.task.shape[0]
        self.post = np.asarray(post_distance, dtype=float)
        self.speed = np.asarray(speed, dtype=float)
        self.stop_mode = np.asarray(stop_mode, dtype=object)
        self.legacy = np.asarray(legacy, dtype=bool)
        self.front = np.asarray(front, dtype=float)
        self.spawn_x = np.asarray(spawn_x, dtype=float)
        self.ramp_steps = int(ramp_steps)
        self.hold_steps = int(hold_steps)
        # ``completion`` is the planned S2 endpoint (control intervals), never
        # the +1 hard-protection limit: flat_walk 1000, flat_stop instant 1400,
        # flat_stop ramp 1500 (400 cruise + 100 ramp + 1000 HOLD).
        if completion_steps is None:
            default = [self._planned_completion(i) for i in range(n)]
        else:
            default = completion_steps
        self.completion = np.asarray(default, dtype=int)
        self.flat = np.isin(self.task, ('flat_walk', 'flat_stop'))
        self.is_stop = np.isin(self.task, ('flat_stop', 'stairs_stop',
                                           'platform_stop'))
        self.ramp_len = np.where(self.is_stop & (self.stop_mode == 'ramp_2s'),
                                 self.ramp_steps, 0).astype(int)
        self.phase = np.zeros(n, dtype=int)
        self.top = np.zeros(n, dtype=bool)
        self.top_event = np.zeros(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        self.failure = np.zeros(n, dtype=bool)
        self.timeout = np.zeros(n, dtype=bool)
        self.quality_failure = np.zeros(n, dtype=bool)
        self.stop_requested = np.zeros(n, dtype=bool)
        self.stop_step = np.full(n, -1, dtype=int)
        self.hold_step = np.full(n, -1, dtype=int)
        self.first_zero_step = np.full(n, -1, dtype=int)
        self.hold_samples = np.zeros(n, dtype=int)
        self.ramp_k = np.zeros(n, dtype=int)
        self.consumed_c_exec = self.speed.copy()
        self.consumed_c_request = self.speed.copy()
        self.consumed_stop_requested = np.zeros(n, dtype=bool)
        self.consumed_phase = self.phase.copy()
        self.consumed_ramp_k = np.zeros(n, dtype=int)
        self.step = np.zeros(n, dtype=int)
        self.c_request = self.speed.copy()
        self.c_exec = self.speed.copy()
        self.g_step = np.full(n, -1, dtype=int)
        self.g_level = np.zeros(n)
        self.air_time = np.zeros((n, 4))
        self.prev_contact = np.zeros((n, 4), dtype=bool)
        self.prev_qdot = np.zeros((n, 12))
        self.quality_reason = [''] * n

    def _planned_completion(self, index):
        task = self.task[index]
        if task == 'flat_walk':
            return 1000
        if task == 'flat_stop':
            ramp = self.ramp_steps if self.stop_mode[index] == 'ramp_2s' else 0
            return CRUISE_STEPS + ramp + self.hold_steps
        return -1

    def reset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.phase[mask] = APPROACH
        self.top[mask] = False
        self.top_event[mask] = False
        self.success[mask] = False
        self.failure[mask] = False
        self.timeout[mask] = False
        self.quality_failure[mask] = False
        self.stop_requested[mask] = False
        self.stop_step[mask] = -1
        self.hold_step[mask] = -1
        self.first_zero_step[mask] = -1
        self.hold_samples[mask] = 0
        self.ramp_k[mask] = 0
        self.consumed_c_exec[mask] = self.speed[mask]
        self.consumed_c_request[mask] = self.speed[mask]
        self.consumed_stop_requested[mask] = False
        self.consumed_phase[mask] = APPROACH
        self.consumed_ramp_k[mask] = 0
        self.step[mask] = 0
        self.c_request[mask] = self.speed[mask]
        self.c_exec[mask] = self.speed[mask]
        self.g_step[mask] = -1
        self.g_level[mask] = 0.0
        self.air_time[mask] = 0
        self.prev_contact[mask] = False
        self.prev_qdot[mask] = 0

    def update(self, *, local_x, top_signal, four_feet, body_ok, quality_ok,
               failure=False, timeout=False, contacts=None, qdot=None,
               ramp_override=False, quality_failure=False,
               quality_reason=None):
        local_x = np.asarray(local_x, dtype=float)
        top_signal = np.asarray(top_signal, dtype=bool)
        four_feet = np.asarray(four_feet, dtype=bool)
        body_ok = np.asarray(body_ok, dtype=bool)
        quality_ok = np.asarray(quality_ok, dtype=bool)
        failure = np.asarray(failure, dtype=bool)
        timeout = np.asarray(timeout, dtype=bool)
        quality_failure = np.asarray(quality_failure, dtype=bool)
        n = local_x.shape[0]
        # T-8 ordering: ``consumed_*`` is what this Actor tick actually used
        # (written by the previous update).  Quality consumes the same-frame
        # values, never a recomputed timeline.
        self.consumed_c_exec = self.c_exec.copy()
        self.consumed_c_request = self.c_request.copy()
        self.consumed_stop_requested = self.stop_requested.copy()
        self.consumed_phase = self.phase.copy()
        self.consumed_ramp_k = self.ramp_k.copy()
        self.step += 1
        top_now = top_signal & ~self.top & ~failure & ~timeout
        self.top_event = top_now
        self.top |= top_now
        dpost = local_x - self.front
        start_dist = local_x - self.spawn_x
        # Phase / g-level geometry gate.
        g_ready = self.top & four_feet & body_ok & ~failure & ~timeout
        self.g_step = np.where(g_ready & (self.g_step < 0), self.step, self.g_step)
        self.g_step = np.where(~g_ready, -1, self.g_step)
        self.g_level = np.where(
            self.g_step < 0, 0.0,
            np.minimum(1.0, (self.step - self.g_step) / 25.0))
        # Stop request after the distance; instant next Actor, ramp 100.
        stop_distance = np.where(
            np.isin(self.task, ['platform_walk', 'platform_stop']),
            start_dist, dpost)
        # flat_stop requests after 400 completed cruise intervals (8 s); the
        # stairs/platform stop tasks request on distance.
        flat_stop_time = (self.task == 'flat_stop') & (self.step >= CRUISE_STEPS)
        requested = ((~self.stop_requested) & ~failure & ~timeout
                     & (((np.isin(self.task, ['stairs_stop', 'platform_stop']))
                         & (stop_distance >= self.post))
                        | flat_stop_time))
        self.stop_requested |= requested
        self.stop_step = np.where(requested, self.step, self.stop_step)
        # External request becomes zero at the request frame; the executed
        # command still follows the instant/ramp contract below.  The exact
        # c_request semantics is T-5 pending, isolated in one module switch.
        if C_REQUEST_ZERO_AT_STOP_REQUEST:
            self.c_request = np.where(self.stop_requested, 0.0, self.speed)
        else:
            self.c_request = self.speed.copy()
        # The request is latched at the end of this control step; the next
        # Actor tick consumes it.  Ramp k = 1..100 with the 100th exactly 0.
        elapsed = self.step - self.stop_step
        ramp_k = np.maximum(1, elapsed + 1)
        ramp_value = self.speed * np.maximum(
            0.0, 1.0 - ramp_k / float(self.ramp_steps))
        instant = self.stop_requested & (self.stop_mode == 'instant')
        ramp_mode = self.stop_requested & (self.stop_mode == 'ramp_2s')
        self.ramp_k = np.where(ramp_mode, np.minimum(ramp_k, self.ramp_steps),
                               np.zeros_like(ramp_k))
        self.c_exec = np.where(
            self.stop_requested,
            np.where(instant, 0.0, ramp_value),
            self.speed)
        self.c_exec = np.where(self.c_exec <= 1e-12, 0.0, self.c_exec)
        # T-2: anchor the HOLD window to the *actual* consumed first zero
        # (inclusive), not to the planned request+ramp frame.
        zero_now = (self.consumed_stop_requested & (self.first_zero_step < 0)
                    & (self.consumed_c_exec <= 1e-12))
        self.first_zero_step = np.where(zero_now, self.step,
                                        self.first_zero_step)
        self.hold_step = self.first_zero_step.copy()
        in_hold = self.first_zero_step >= 0
        self.hold_samples = np.where(in_hold, self.hold_samples + 1,
                                     self.hold_samples)
        # Endpoint: flat tasks use the planned time; stairs/platform stop use
        # the actual first zero + 1000 intervals - 1 (T-2).
        flat_endpoint = self.flat & (self.completion > 0) & (
            self.step >= self.completion)
        stop_endpoint = (np.isin(self.task, ('stairs_stop', 'platform_stop'))
                         & (self.first_zero_step >= 0)
                         & (self.step >= self.first_zero_step
                            + self.hold_steps - 1))
        endpoint = flat_endpoint | stop_endpoint
        hold_done = endpoint & (self.hold_samples >= self.hold_steps)
        # success
        top_success = self.legacy & self.top
        # Flat tasks succeed on their planned time window, never on distance.
        distance_ok = ((~self.legacy)
                       & np.isin(self.task, ['platform_walk', 'platform_stop',
                                             'stairs_walk'])
                       & (np.where(
                           np.isin(self.task, ['platform_walk', 'platform_stop']),
                           start_dist, dpost) >= self.post))
        nonstop_ok = distance_ok & ~np.isin(self.task, ['stairs_stop', 'platform_stop'])
        stop_ok = hold_done & np.isin(self.task, ['stairs_stop', 'platform_stop'])
        flat_walk_ok = (self.task == 'flat_walk') & (
            self.step >= self.completion)
        flat_stop_ok = (self.task == 'flat_stop') & hold_done
        flat_ok = flat_walk_ok | flat_stop_ok
        success_now = ((top_success | nonstop_ok | stop_ok | flat_ok)
                       & quality_ok & ~failure & ~timeout & ~quality_failure
                       & ~self.success)
        # phase
        phase = np.zeros(n, dtype=int)
        phase = np.where(self.top, POST_WALK, phase)
        phase = np.where((~self.top) & (local_x >= 0) & np.isin(
            self.task, ['stairs_walk', 'stairs_stop']), CLIMB, phase)
        phase = np.where(self.stop_requested & (self.c_exec > 1e-7), DECEL, phase)
        phase = np.where((self.stop_requested & (self.c_exec <= 1e-7)
                          & ~success_now), HOLD, phase)
        self.phase = phase
        self.success |= success_now
        self.failure |= failure & ~self.success
        self.quality_failure |= quality_failure & ~failure & ~self.success
        self.failure |= self.quality_failure & ~self.success
        # Hard-protection limits (1001/1401/1501) are backup only: the normal
        # endpoint is an explicit success or quality failure above.
        hard = (self.flat & (self.completion > 0)
                & (self.step >= self.completion + 1))
        self.timeout |= (timeout | hard) & ~self.success & ~self.failure
        if contacts is not None:
            contact = np.asarray(contacts, dtype=bool)
            self.air_time = np.where(contact, 0.0,
                                     np.where(self.prev_contact, self.air_time + 0.02,
                                              self.air_time))
            self.prev_contact = contact.copy()
        if qdot is not None:
            self.prev_qdot = np.asarray(qdot, dtype=float).copy()
        if quality_reason is not None:
            self.quality_reason = [str(v) for v in quality_reason]
        return dict(top=self.top,
                    top_event=self.top_event, success=self.success,
                    failure=self.failure, timeout=self.timeout,
                    quality_failure=self.quality_failure,
                    physical_failure=failure,
                    quality_reason=list(self.quality_reason),
                    terminated=self.success | self.failure | self.timeout,
                    truncated=np.zeros(n, dtype=bool),
                    v_cruise=self.speed,
                    c_request=self.c_request,
                    c_exec=self.c_exec,
                    c_request_at_actor=self.consumed_c_request,
                    c_request_after_step=self.c_request,
                    c_exec_at_actor=self.consumed_c_exec,
                    c_exec_after_step=self.c_exec,
                    ramp_k_at_actor=self.consumed_ramp_k,
                    ramp_k_after_step=self.ramp_k,
                    phase=[PHASES[i] for i in self.consumed_phase],
                    phase_next=[PHASES[i] for i in self.phase],
                    consumed_c_exec=self.consumed_c_exec,
                    consumed_stop_requested=self.consumed_stop_requested,
                    g_level=self.g_level, stop_requested=self.stop_requested,
                    stop_requested_at_actor=self.consumed_stop_requested,
                    stop_requested_after_step=self.stop_requested,
                    stop_step=self.stop_step, hold_step=self.first_zero_step,
                    first_zero_step=self.first_zero_step,
                    hold_samples=self.hold_samples,
                    endpoint=endpoint, hold_done=hold_done,
                    success_event=success_now)


def quality_reasons(*, pitch_deg, speed, command, contact, planned_support,
                    nonfoot=False, recovery_seconds=None, stand=None):
    """P1-b §6.5 per-case quality (T-3: delegates to the shared core).

    Returns a list, empty means pass.  This compatibility wrapper keeps the
    historical names used by the S2.0/P1-b callers while every threshold and
    metric computation comes from :mod:`s2_quality`.
    """
    from . import s2_quality as _quality
    pitch = np.asarray(pitch_deg, dtype=float)
    speed = np.asarray(speed, dtype=float)
    command = np.asarray(command, dtype=float)
    if pitch.size == 0:
        return ['missing_window']
    metrics = _quality.cruise_metric_values(pitch, speed, command)
    reasons = [name for name in _quality.cruise_reasons(metrics, min_frames=1)
               if name not in ('missing_window', 'window_short')]
    over = maximum = 0
    for value in pitch:
        over = over + 1 if abs(value) > _quality.PITCH_LATCH_DEG else 0
        maximum = max(maximum, over)
    if maximum * _quality.CONTROL_DT >= (
            _quality.PITCH_LATCH_FRAMES * _quality.CONTROL_DT):
        reasons.append('pitch_over15')
    if bool(np.asarray(nonfoot).any()) if hasattr(nonfoot, '__len__') else bool(nonfoot):
        reasons.append('nonfoot_contact')
    if planned_support is not None:
        support = np.asarray(planned_support, dtype=bool)
        if support.size and float(support.mean()) < _quality.STOP_SUPPORT_RATIO:
            reasons.append('support')
    if (recovery_seconds is not None
            and float(recovery_seconds) > _quality.RECOVERY_DEADLINE
            * _quality.CONTROL_DT):
        reasons.append('recovery')
    if stand:
        for key in ('mean_speed', 'peak_speed'):
            limit = (_quality.STOP_MEAN_SPEED if key == 'mean_speed'
                     else _quality.STOP_PEAK_SPEED)
            if stand.get(key) is not None and stand[key] > limit:
                reasons.append('stand_' + key)
        if (stand.get('support_fraction') is not None
                and stand['support_fraction'] < _quality.STOP_SUPPORT_RATIO):
            reasons.append('stand_support')
        if (stand.get('height_min') is not None
                and stand['height_min'] < _quality.STOP_HEIGHT):
            reasons.append('stand_height')
    return reasons
