"""S2 task state machine, distance trigger and ``g_level`` gate (pure CPU).

S2.0 uses ``flat_walk``, ``stairs_walk`` and ``platform_walk``.  DECEL/HOLD are
implemented for the later stop courses but S2.0 never enables them.

Semantics implemented here:
* §4.2 state order APPROACH → CLIMB → TOP_TRANSITION → POST_WALK → DECEL →
  HOLD → SUCCESS;
* ``top_arrived`` is latched exactly once and never reset by the task;
* ``D_post`` uses the Q019 zero: the latched ``top_arrived`` body route_x for a
  complete climb, or the spawn route_x for a platform start;
* distance/stop triggers happen at the *end* of a control step, so the next
  Actor call receives the new command;
* a same-frame failure wins over top/success/distance;
* the hard timeout is an input from :mod:`s2_spec`, never recomputed here;
* ``g_level`` requires the true-terrain four-foot platform condition plus the
  base being 0.4 m past the platform front edge; it holds 0.5 s then ramps
  linearly to 1 and resets to 0 when the condition fails.

The 0.5 s g_level *hold* is explicit in §5.2; the ramp duration is not numeric
there.  ``G_LEVEL_RAMP_STEPS`` is a named provisional constant (0.5 s) so a
planning ruling can change one value without touching the state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math

from . import s2_spec

STATE_ORDER = ('APPROACH', 'CLIMB', 'TOP_TRANSITION', 'POST_WALK', 'DECEL',
               'HOLD', 'SUCCESS', 'FAILURE')
WALK_TASKS = ('flat_walk', 'stairs_walk', 'platform_walk')
STOP_TASKS = ('flat_stop', 'stairs_stop', 'platform_stop')
# Q027: the 0.5 s is the *whole* 0→1 ramp; there is no extra hold interval.
G_LEVEL_RAMP_STEPS = 25          # 25 × 0.02 s = 0.5 s, centralised here
G_LEVEL_BASE_MARGIN = 0.4        # strict > past the platform front edge
G_LEVEL_SNAPSHOT_VERSION = 'q027_ramp_total_0p5s'
G_LEVEL_PRE_Q027_SNAPSHOT = 'pre_q027_hold_plus_ramp'
DT = s2_spec.CONTROL_DT


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


@dataclass(frozen=True)
class S2TaskConfig:
    task: str
    speed: float
    stair_count: int
    initial_route_x: float
    timeout_steps: int
    post_distance: float | None
    stop_mode: str = 'none'
    ramp_steps: int = 100
    hold_steps: int = 1000
    completion_steps: int | None = None
    platform_start: bool = False

    @classmethod
    def from_case(cls, case):
        task = case['task']
        if task == 'stairs':
            return cls(task='stairs_walk', speed=float(case['speed']),
                       stair_count=int(case['stair_count']),
                       initial_route_x=float(case['spawn']['local_position_xy'][0]),
                       timeout_steps=int(case['hard_timeout_steps']),
                       post_distance=0.0, stop_mode='none',
                       completion_steps=None, platform_start=False)
        if task not in WALK_TASKS + STOP_TASKS:
            raise ValueError('unknown S2 task')
        command = case.get('command', {})
        stop_mode = case.get('stop_mode', 'none')
        if stop_mode == 'none' and task.endswith('_stop'):
            raise ValueError('stop task requires a stop mode')
        if task.startswith('flat_'):
            schedule = (s2_spec.flat_schedule('flat_walk') if task == 'flat_walk'
                        else s2_spec.flat_stop_schedule(stop_mode))
            completion = schedule['completion_steps']
            post = None
        else:
            completion = None
            post = command.get('post_distance_m')
            if post is None:
                post = s2_spec.POST_TOP_DIAGNOSTIC_DISTANCE
        return cls(
            task=task, speed=float(case['speed']),
            stair_count=int(case['stair_count']),
            initial_route_x=float(case['spawn']['local_position_xy'][0]),
            timeout_steps=int(case['hard_timeout_steps']),
            post_distance=(None if post is None else float(post)),
            stop_mode=stop_mode,
            ramp_steps=int(command.get('ramp_steps', 100 if stop_mode == 'ramp_2s' else 0)),
            hold_steps=int(command.get('hold_steps', 0)),
            completion_steps=completion,
            platform_start=task.startswith('platform_'))

    @property
    def is_stop(self):
        return self.stop_mode != 'none' and self.task.endswith('_stop')

    @property
    def initial_state(self):
        if self.task == 'flat_walk':
            return 'POST_WALK'
        if self.task.startswith('platform_'):
            return 'POST_WALK'
        return 'APPROACH'


def level_condition(*, platform_front_x, platform_length, platform_y_half_width,
                    base_xy, feet_xy, base_margin=G_LEVEL_BASE_MARGIN):
    """Q027 condition; strict ``>`` past the front edge, true geometry only."""
    base = (float(base_xy[0]), float(base_xy[1]))
    x0 = float(platform_front_x)
    x1 = x0 + float(platform_length)
    half = float(platform_y_half_width)
    if not (base[0] > x0 + float(base_margin) and base[0] <= x1):
        return False
    if abs(base[1]) > half:
        return False
    for x, y in feet_xy:
        if not (x0 <= float(x) <= x1 and abs(float(y)) <= half):
            return False
    return True


class LevelGate:
    """Q027: ``g(t0)=0``, ``g(t0+k)=min(k/25,1)``, whole 0.5 s ramp.

    ``update`` is called once per post-physics control-step sample.  Flat terrain
    is fixed at 1 by ``flat=True``.  A failed condition zeroes the gate and
    clears the start; the next valid frame defines a new t0 (even after full).
    """

    def __init__(self, *, flat=False):
        self.flat = bool(flat)
        self.started = False
        self.k = None
        self.value = 1.0 if self.flat else 0.0
        self.t0_step = 0 if self.flat else None
        self.first_positive_step = 0 if self.flat else None
        self.first_full_step = 0 if self.flat else None
        self.interruptions = 0
        self.restarts = 0
        self.total_steps = 0
        self.on_steps = 0
        self.off_run = 0
        self.max_off_run = 0
        self.ever_full = self.flat
        self.interruption_log = []
        self.restart_log = []

    def update(self, condition, *, step_index=None):
        if self.flat:
            self.total_steps += 1
            self.on_steps += 1
            self.value = 1.0
            return self.value
        self.total_steps += 1
        if not condition:
            if self.started:
                self.interruptions += 1
                self.interruption_log.append(dict(step=step_index, k=self.k,
                                                  value=self.value))
            self.started = False
            self.k = None
            self.t0_step = None
            self.value = 0.0
            self.off_run += 1
            self.max_off_run = max(self.max_off_run, self.off_run)
            return self.value
        self.off_run = 0
        self.on_steps += 1
        if not self.started:
            self.started = True
            self.restarts += 1
            self.k = 0
            self.t0_step = step_index
            self.value = 0.0
            self.restart_log.append(dict(t0=step_index))
            return self.value
        self.k += 1
        self.value = min(self.k / float(G_LEVEL_RAMP_STEPS), 1.0)
        if self.value > 0.0 and self.first_positive_step is None:
            self.first_positive_step = step_index
        if self.value >= 1.0:
            self.ever_full = True
            if self.first_full_step is None:
                self.first_full_step = step_index
        return self.value

    @property
    def duty_cycle(self):
        return (self.on_steps / self.total_steps) if self.total_steps else None

    def state_dict(self):
        return dict(version=G_LEVEL_SNAPSHOT_VERSION, flat=self.flat,
                    started=self.started, k=self.k, value=self.value,
                    t0_step=self.t0_step,
                    first_positive_step=self.first_positive_step,
                    first_full_step=self.first_full_step,
                    interruptions=self.interruptions, restarts=self.restarts,
                    total_steps=self.total_steps, on_steps=self.on_steps,
                    off_run=self.off_run, max_off_run=self.max_off_run,
                    ever_full=self.ever_full,
                    interruption_log=list(self.interruption_log),
                    restart_log=list(self.restart_log))

    def load_state(self, state):
        """Reject snapshots with a different g_level timing contract."""
        if not isinstance(state, dict):
            raise ValueError('gate snapshot must be a mapping')
        version = state.get('version', G_LEVEL_PRE_Q027_SNAPSHOT)
        if version != G_LEVEL_SNAPSHOT_VERSION:
            raise ValueError(
                f'g_level snapshot version {version!r} is not compatible with '
                f'{G_LEVEL_SNAPSHOT_VERSION!r}; do not silently mix timing rules')
        for key, value in state.items():
            if key == 'version':
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def g_level_report(gate, *, four_foot_step=None, four_foot_step_end=None):
    """Four Q027 timestamps and the measured delay from four-foot entry."""
    report = dict(
        four_foot_entry_step=four_foot_step,
        first_full_geometry_step=gate.t0_step,
        first_g_positive_step=gate.first_positive_step,
        first_g_full_step=gate.first_full_step,
        delay_seconds_from_four_foot=None,
        interruptions=gate.interruptions, restarts=gate.restarts,
        duty_cycle=gate.duty_cycle, max_off_run_steps=gate.max_off_run,
        ever_full=gate.ever_full,
        interruptions_detail=list(gate.interruption_log),
        restarts_detail=list(gate.restart_log),
        na_reasons=[])
    if four_foot_step is None:
        report['na_reasons'].append('four_foot_entry_not_reached')
    if gate.t0_step is None:
        report['na_reasons'].append('full_geometry_not_reached')
    if gate.first_positive_step is None:
        report['na_reasons'].append('g_level_never_positive')
    if gate.first_full_step is None:
        report['na_reasons'].append('g_level_never_full')
    if four_foot_step is not None and gate.first_positive_step is not None:
        report['delay_seconds_from_four_foot'] = (
            (gate.first_positive_step - four_foot_step) * DT)
    return report


class S2StateMachine:
    """Per-episode state/command/event machine; one instance per case."""

    def __init__(self, config):
        if not isinstance(config, S2TaskConfig):
            raise ValueError('config must be S2TaskConfig')
        self.config = config
        self.state = config.initial_state
        self.completed_steps = 0
        self.top_arrived = False
        self.top_route_x = None
        self.four_feet_platform = False
        self.distance_reached = False
        self.stop_requested = False
        self.first_zero_step = None
        self.hold_completed = False
        self.success = False
        self.failure = False
        self.timeout = False
        self.terminal = False
        self.ramp_index = 0
        self.c_request = float(config.speed)
        self.c_exec = float(config.speed)
        self.events = []
        self._command_trace = []

    @property
    def d_post_zero(self):
        if self.config.platform_start or self.config.task == 'flat_walk':
            return float(self.config.initial_route_x)
        return self.top_route_x

    def d_post(self, route_x):
        zero = self.d_post_zero
        if zero is None:
            return None
        return float(route_x) - float(zero)

    def _event(self, name, route_x, **extra):
        self.events.append(dict(name=name, step=self.completed_steps,
                                route_x=(None if route_x is None else float(route_x)),
                                state=self.state, **extra))

    def step(self, frame):
        """Advance one control step; returns events latched at this step end.

        ``frame`` fields: ``route_x``, ``first_step`` (bool), ``top_arrived``
        (signal), ``four_feet_on_platform`` (bool), ``failure`` (bool),
        ``a_request``, ``a_applied``.  ``c_exec`` recorded for this step is the
        command that was actually sent to the Actor at the start of the step.
        """
        if self.terminal:
            raise RuntimeError('state machine already terminal')
        route_x = _finite(frame.get('route_x'), 'route_x')
        self._command_trace.append(dict(step=self.completed_steps, state=self.state,
                                        c_request=self.c_request, c_exec=self.c_exec,
                                        a_request=frame.get('a_request'),
                                        a_applied=frame.get('a_applied')))
        self.completed_steps += 1
        failure = bool(frame.get('failure', False))
        timeout = self.completed_steps >= self.config.timeout_steps
        # §4.2: failure and timeout take precedence over every success event.
        if failure or timeout:
            self.failure = True
            self.timeout = bool(timeout and not failure)
            self.state = 'FAILURE'
            self._event('failure', route_x, timeout=self.timeout)
            self.terminal = True
            return list(self.events[-1:])
        if frame.get('top_arrived') and not self.top_arrived:
            self.top_arrived = True
            self.top_route_x = route_x
            self.state = 'TOP_TRANSITION'
            self._event('top_arrived', route_x)
        if frame.get('four_feet_on_platform') and not self.four_feet_platform:
            self.four_feet_platform = True
            self._event('four_feet_platform', route_x)
        if self.state == 'APPROACH' and frame.get('first_step'):
            self.state = 'CLIMB'
            self._event('enter_climb', route_x)
        if self.state == 'CLIMB' and self.top_arrived:
            self.state = 'TOP_TRANSITION'
        if self.state == 'TOP_TRANSITION' and self.four_feet_platform:
            self.state = 'POST_WALK'
        if self.state == 'POST_WALK':
            if self.config.task == 'flat_walk':
                if self.completed_steps >= self.config.completion_steps:
                    self.success = True
                    self.state = 'SUCCESS'
                    self.terminal = True
                    self._event('success', route_x)
            elif self.config.post_distance is not None:
                distance = self.d_post(route_x)
                if distance >= self.config.post_distance and not self.distance_reached:
                    self.distance_reached = True
                    self._event('distance_reached', route_x, d_post=distance)
                    if self.config.is_stop:
                        self.stop_requested = True
                        self._event('stop_request', route_x, d_post=distance)
                        self.state = 'DECEL'
                        self.ramp_index = 0
                        if self.config.stop_mode == 'instant':
                            self._set_zero()
                        else:
                            self._ramp_next()
                    else:
                        self.success = True
                        self.state = 'SUCCESS'
                        self.terminal = True
                        self._event('success', route_x, d_post=distance)
            if self.state in ('POST_WALK', 'SUCCESS') and self.four_feet_platform is False \
                    and self.config.task.startswith('stairs_') and self.top_arrived:
                # Four-foot entry is expected after top; keep POST_WALK until it happens.
                pass
        if self.state == 'DECEL':
            if self.c_exec <= 1e-12:
                if self.first_zero_step is None:
                    self.first_zero_step = self.completed_steps
                self.state = 'HOLD'
                self.hold_completed = True
                self._event('enter_hold', route_x)
                self.success = True
                self.state = 'SUCCESS'
                self._event('success', route_x)
            else:
                self._ramp_next()
        return list(self.events)

    def _set_zero(self):
        self.c_request = 0.0
        self.c_exec = 0.0
        self.ramp_index = self.config.ramp_steps

    def _ramp_next(self):
        self.c_request = 0.0
        self.ramp_index += 1
        self.c_exec = float(self.config.speed) * max(
            0.0, 1.0 - self.ramp_index / float(self.config.ramp_steps))
        if self.c_exec <= 1e-12:
            self.c_exec = 0.0

    def command_trace(self):
        return list(self._command_trace)

    def summary(self):
        return dict(task=self.config.task, initial_state=self.config.initial_state,
                    state=self.state, completed_steps=self.completed_steps,
                    top_arrived=self.top_arrived, top_route_x=self.top_route_x,
                    d_post_zero=self.d_post_zero,
                    four_feet_platform=self.four_feet_platform,
                    distance_reached=self.distance_reached,
                    stop_requested=self.stop_requested,
                    first_zero_step=self.first_zero_step,
                    success=self.success, failure=self.failure,
                    timeout=self.timeout, terminal=self.terminal,
                    c_request=self.c_request, c_exec=self.c_exec,
                    events=list(self.events), command_trace=self.command_trace())
