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
G_LEVEL_HOLD_STEPS = 25          # 0.5 s at 50 Hz, explicit in §5.2
G_LEVEL_RAMP_STEPS = 25          # provisional; §5.2 gives no numeric ramp length
G_LEVEL_BASE_MARGIN = 0.4        # §5.2 body past the platform front edge
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
            # Original H/L regression wrapper: success at top, original timeout.
            command = case.get('command', {})
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
    """§5.2 condition on true terrain: four feet XY on the same top platform."""
    base = (float(base_xy[0]), float(base_xy[1]))
    x0 = float(platform_front_x)
    x1 = x0 + float(platform_length)
    half = float(platform_y_half_width)
    if not (base[0] >= x0 + float(base_margin) and base[0] <= x1):
        return False
    if abs(base[1]) > half:
        return False
    for x, y in feet_xy:
        if not (x0 <= float(x) <= x1 and abs(float(y)) <= half):
            return False
    return True


class LevelGate:
    """0.5 s hold then linear 0→1; any failed condition resets to zero."""

    def __init__(self, *, hold_steps=G_LEVEL_HOLD_STEPS, ramp_steps=G_LEVEL_RAMP_STEPS):
        if hold_steps < 0 or ramp_steps <= 0:
            raise ValueError('hold must be >=0 and ramp >0')
        self.hold_steps, self.ramp_steps = int(hold_steps), int(ramp_steps)
        self.held = 0
        self.value = 0.0
        self.ever_full = False

    def update(self, condition):
        if not condition:
            self.held = 0
            self.value = 0.0
            return self.value
        self.held += 1
        if self.held <= self.hold_steps:
            self.value = 0.0
        else:
            progress = (self.held - self.hold_steps) / self.ramp_steps
            self.value = min(1.0, progress)
        if self.value >= 1.0:
            self.ever_full = True
        return self.value

    def state_dict(self):
        return dict(held=self.held, value=self.value, ever_full=self.ever_full,
                    hold_steps=self.hold_steps, ramp_steps=self.ramp_steps)


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
