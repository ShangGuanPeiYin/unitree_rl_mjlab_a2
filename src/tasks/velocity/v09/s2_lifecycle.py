"""Vectorized S2 training lifecycle and quality admission (CPU-testable)."""
from __future__ import annotations

import numpy as np

TASK_CODES = dict(flat_walk=0, flat_stop=1, stairs_walk=2, platform_walk=3,
                  stairs_stop=4, platform_stop=5, legacy_l40=6)
FLAT_STOP_STEP = 400  # 8 s of walking at 0.02 s control
PHASES = ('APPROACH', 'CLIMB', 'TOP_TRANSITION', 'POST_WALK', 'DECEL', 'HOLD')
APPROACH, CLIMB, TOP_TRANSITION, POST_WALK, DECEL, HOLD = range(6)


class S2Lifecycle:
    def __init__(self, *, task, post_distance, speed, stop_mode='none',
                 legacy=False, front=0.0, spawn_x=-1.0,
                 ramp_steps=100, hold_steps=1000, completion_steps=None):
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
        self.completion = (np.asarray(
            [1000 if self.task[i] == 'flat_walk' else 1401 for i in range(n)],
            dtype=int) if completion_steps is None
            else np.asarray(completion_steps, dtype=int))
        self.phase = np.zeros(n, dtype=int)
        self.top = np.zeros(n, dtype=bool)
        self.top_event = np.zeros(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        self.failure = np.zeros(n, dtype=bool)
        self.timeout = np.zeros(n, dtype=bool)
        self.stop_requested = np.zeros(n, dtype=bool)
        self.stop_step = np.full(n, -1, dtype=int)
        self.hold_step = np.full(n, -1, dtype=int)
        self.step = np.zeros(n, dtype=int)
        self.c_request = self.speed.copy()
        self.c_exec = self.speed.copy()
        self.g_step = np.full(n, -1, dtype=int)
        self.g_level = np.zeros(n)
        self.air_time = np.zeros((n, 4))
        self.prev_contact = np.zeros((n, 4), dtype=bool)
        self.prev_qdot = np.zeros((n, 12))
        self.quality_reason = [''] * n

    def reset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.phase[mask] = APPROACH
        self.top[mask] = False
        self.top_event[mask] = False
        self.success[mask] = False
        self.failure[mask] = False
        self.timeout[mask] = False
        self.stop_requested[mask] = False
        self.stop_step[mask] = -1
        self.hold_step[mask] = -1
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
               ramp_override=False):
        local_x = np.asarray(local_x, dtype=float)
        top_signal = np.asarray(top_signal, dtype=bool)
        four_feet = np.asarray(four_feet, dtype=bool)
        body_ok = np.asarray(body_ok, dtype=bool)
        quality_ok = np.asarray(quality_ok, dtype=bool)
        failure = np.asarray(failure, dtype=bool)
        timeout = np.asarray(timeout, dtype=bool)
        n = local_x.shape[0]
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
        # flat_stop stops after 8 s of walking (SDS 6.2: 20 s walk, then stop,
        # then a 20 s hold); stairs/platform stop tasks stop on distance.
        flat_stop_time = (self.task == 'flat_stop') & (self.step >= FLAT_STOP_STEP)
        requested = ((~self.stop_requested) & ~failure & ~timeout
                     & (((np.isin(self.task, ['stairs_stop', 'platform_stop']))
                         & (stop_distance >= self.post))
                        | flat_stop_time))
        self.stop_requested |= requested
        self.stop_step = np.where(requested, self.step, self.stop_step)
        elapsed = np.maximum(0, self.step - self.stop_step)
        ramp = self.speed * np.maximum(0.0, 1.0 - elapsed / self.ramp_steps)
        instant = self.stop_requested & (self.stop_mode == 'instant')
        self.c_exec = np.where(
            self.stop_requested,
            np.where(instant, 0.0, ramp),
            self.speed)
        self.c_exec = np.where(self.c_exec <= 1e-12, 0.0, self.c_exec)
        first_zero = self.stop_requested & (self.hold_step < 0) & (self.c_exec <= 1e-12)
        self.hold_step = np.where(first_zero, self.step, self.hold_step)
        # success
        top_success = self.legacy & self.top
        # Flat tasks succeed on their planned time window, never on distance.
        distance_ok = ((~self.legacy)
                       & np.isin(self.task, ['platform_walk', 'platform_stop',
                                             'stairs_walk'])
                       & (np.where(
                           np.isin(self.task, ['platform_walk', 'platform_stop']),
                           start_dist, dpost) >= self.post))
        hold_ok = self.stop_requested & (self.hold_step >= 0) & (
            self.step - self.hold_step >= self.hold_steps)
        nonstop_ok = distance_ok & ~np.isin(self.task, ['stairs_stop', 'platform_stop'])
        stop_ok = hold_ok & np.isin(self.task, ['stairs_stop', 'platform_stop'])
        flat_stop_hold = (self.task == 'flat_stop') & self.stop_requested \
            & (self.hold_step >= 0) & (self.step - self.hold_step >= self.hold_steps)
        flat_ok = (((self.task == 'flat_walk') & (self.step >= self.completion))
                   | ((self.task == 'flat_stop') & (self.step >= self.completion)
                      & flat_stop_hold))
        success_now = ((top_success | nonstop_ok | stop_ok | flat_ok)
                       & quality_ok & ~failure & ~timeout & ~self.success)
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
        self.timeout |= timeout & ~self.success & ~self.failure
        if contacts is not None:
            contact = np.asarray(contacts, dtype=bool)
            self.air_time = np.where(contact, 0.0,
                                     np.where(self.prev_contact, self.air_time + 0.02,
                                              self.air_time))
            self.prev_contact = contact.copy()
        if qdot is not None:
            self.prev_qdot = np.asarray(qdot, dtype=float).copy()
        return dict(phase=[PHASES[i] for i in self.phase], top=self.top,
                    top_event=self.top_event, success=self.success,
                    failure=self.failure, timeout=self.timeout,
                    terminated=self.success | self.failure | self.timeout,
                    truncated=np.zeros(n, dtype=bool),
                    c_request=self.c_request, c_exec=self.c_exec,
                    g_level=self.g_level, stop_requested=self.stop_requested,
                    hold_step=self.hold_step, success_event=success_now)


def quality_reasons(*, pitch_deg, speed, command, contact, planned_support,
                    nonfoot=False, recovery_seconds=None, stand=None):
    """P1-b §6.5 per-case quality; returns a list, empty means pass."""
    import math
    pitch = np.asarray(pitch_deg, dtype=float)
    speed = np.asarray(speed, dtype=float)
    command = np.asarray(command, dtype=float)
    reasons = []
    if pitch.size == 0:
        return ['missing_window']
    if float(np.sqrt(np.mean(pitch ** 2))) > 5.0:
        reasons.append('pitch_rms')
    if float(np.quantile(np.abs(pitch), .99)) > 10.0:
        reasons.append('pitch_p99')
    over = maximum = 0
    for value in pitch:
        over = over + 1 if abs(value) > 15.0 else 0
        maximum = max(maximum, over)
    if maximum * 0.02 >= 0.2:
        reasons.append('pitch_over15')
    if float(np.sqrt(np.mean((speed - command) ** 2))) > 0.10:
        reasons.append('speed_rmse')
    if bool(np.asarray(nonfoot).any()) if hasattr(nonfoot, '__len__') else bool(nonfoot):
        reasons.append('nonfoot_contact')
    if planned_support is not None:
        support = np.asarray(planned_support, dtype=bool)
        if support.size and float(support.mean()) < 0.90:
            reasons.append('support')
    if recovery_seconds is not None and float(recovery_seconds) > 3.0:
        reasons.append('recovery')
    if stand:
        for key in ('mean_speed', 'peak_speed'):
            if stand.get(key) is not None and stand[key] > (0.05 if key == 'mean_speed' else 0.10):
                reasons.append('stand_' + key)
        if stand.get('support_fraction') is not None and stand['support_fraction'] < 0.90:
            reasons.append('stand_support')
        if stand.get('height_min') is not None and stand['height_min'] < 0.30:
            reasons.append('stand_height')
    return reasons
