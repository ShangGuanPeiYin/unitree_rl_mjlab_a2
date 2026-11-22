"""S2.2/S2.3 course contracts and stop-column activation (CPU-testable)."""
from __future__ import annotations

from dataclasses import dataclass

from . import s2_spec
from . import s2_train_env

COURSE_CONFIG = {
    'S2.1': dict(main_task='stairs_walk', main_levels=(10, 20),
                 post_range=(5.0, 5.0), platform_height= (2.0, 4.0),
                 platform_post=5.0, stop=False, review_levels=(40,),
                 review_post=5.0),
    'S2.2': dict(main_task='stairs_walk', main_levels=(40,),
                 post_range=(5.0, 10.0), platform_height=(8.0,),
                 platform_post=10.0, stop=False, review_levels=(10, 20),
                 review_post=5.0),
    'S2.3': dict(main_task='stairs_stop', main_levels=(40,),
                 post_range=(3.0, 10.0), platform_height=(8.0,),
                 platform_post=(3.0, 10.0), stop=True,
                 review_levels=(10, 20), review_post=5.0),
}


@dataclass
class StopState:
    speed: float
    post_distance: float
    stop_mode: str
    ramp_steps: int = 100
    hold_steps: int = 1000
    distance: float = 0.0
    steps: int = 0
    stop_requested: bool = False
    stop_step: int | None = None
    zero_step: int | None = None
    phase: str = 'CRUISE'
    last_c_exec: float | None = None

    def __post_init__(self):
        if self.stop_mode not in ('instant', 'ramp_2s'):
            raise ValueError('stop_mode must be instant or ramp_2s')
        if self.post_distance < 0:
            raise ValueError('post_distance must be nonnegative')

    def advance(self, *, distance: float | None = None):
        self.steps += 1
        if distance is not None:
            self.distance = float(distance)
        if not self.stop_requested and self.distance >= self.post_distance:
            self.stop_requested = True
            self.stop_step = self.steps  # next Actor command
        if self.stop_requested:
            if self.stop_mode == 'instant' or self.steps >= (self.stop_step or 0) + self.ramp_steps:
                c_exec = 0.0
                if self.zero_step is None:
                    self.zero_step = self.steps
            else:
                elapsed = self.steps - (self.stop_step or 0)
                c_exec = self.speed * max(0.0, 1.0 - elapsed / self.ramp_steps)
        else:
            c_exec = self.speed
        self.last_c_exec = c_exec
        if self.zero_step is not None and self.steps - self.zero_step >= self.hold_steps:
            self.phase = 'DONE_STAND'
        elif self.zero_step is not None and self.steps - self.zero_step >= self.hold_steps - 500:
            self.phase = 'STAND'
        elif self.zero_step is not None:
            self.phase = 'HOLD'
        elif self.stop_requested:
            self.phase = 'DECEL'
        else:
            self.phase = 'CRUISE'
        return dict(c_exec=c_exec, stop_requested=self.stop_requested,
                    phase=self.phase, stop_step=self.stop_step,
                    zero_step=self.zero_step, hold_progress=(
                        None if self.zero_step is None else self.steps - self.zero_step))

    def reward_mask(self):
        stopped = self.stop_requested
        zero = self.last_c_exec is not None and abs(self.last_c_exec) <= 1e-8
        return dict(I_cruise=(not stopped), forward_progress=(not stopped),
                    air_time=(not stopped), stable_stand=bool(zero),
                    stop_phase=self.phase)


def reward_weights():
    """Reward weights are unchanged from §5; only columns are gated."""
    from ..v07 import spec as original
    return dict(original.REWARD_WEIGHTS)


def validate_course(course):
    if course not in COURSE_CONFIG:
        raise ValueError(f'unknown S2 course {course!r}')
    return COURSE_CONFIG[course]


def assert_course_budget(course, *, num_envs=512):
    if course not in s2_spec.S2_COURSE_LEARNING_CAPS:
        raise ValueError('unknown S2 course')
    cap = s2_spec.S2_COURSE_LEARNING_CAPS[course]
    plan = s2_train_env.build_role_plan(num_envs, course=course)
    samples = num_envs * s2_spec.ROLLOUT_STEPS
    complete = cap // samples
    if complete * samples > cap:
        raise AssertionError('course budget exceeds cap')
    return dict(cap=cap, samples_per_rollout=samples, complete_rollouts=complete,
                role_counts=plan['counts'])
