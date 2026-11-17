"""v0.6 speed groups, gate thresholds, F0 flat criteria and retention-set helpers.

Everything numeric is read from the frozen spec module (single number source,
agent A).  When the frozen ``spec`` surface does not yet expose a curriculum
field, the value is resolved from a documented SDS fallback and recorded in
``CURRICULUM_SETTINGS['source']`` so the coordination note can list it
(``implementation/coordination/QC_training_stack.md``, status "待 Codex 裁决").
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from . import spec

# ------------------------------------------------------------------ identity
TASK_ID = getattr(spec, 'TASK_ID', spec.TASK)
RECIPE_VERSION = spec.RECIPE_VERSION
CODE_VERSION = spec.CODE_VERSION
SCHEMA_VERSION = int(getattr(spec, 'CHECKPOINT_SCHEMA', spec.SCHEMA_VERSION))
PROTOCOL_ID = spec.PROTOCOL_ID
CONTROL_DT = float(spec.CONTROL_DT)

GATE_CASES = int(spec.GATE_CASES)
GATE_PASS_MIN = int(spec.GATE_PASS_MIN)
FINAL_CASES = int(spec.FINAL_CASES)
FINAL_PASS_MIN = int(spec.FINAL_PASS_MIN)
GATE_INTERVAL = int(spec.GATE_INTERVAL)
DIAGNOSTIC_INTERVAL = int(spec.DIAGNOSTIC_INTERVAL)
SAVE_INTERVAL = int(spec.SAVE_INTERVAL)
RETENTION_INTERVAL = int(spec.RETENTION_INTERVAL)

CASE_KINDS = ('validation', 'diagnostic', 'perturbed', 'test',
              'flat_retention', 'stair_retention')
SPEED_TOLERANCE = 1e-9

#: SDS 8.3 D1 anti-regression offsets: applied exactly once, in case construction.
_SEED_OFFSET_FALLBACK = {'validation': 200300, 'diagnostic': 500300,
                         'perturbed': 300300, 'test': 400342}
#: SDS 5.4 spawn ranges (metres / radians) and SDS 6.1 F0 criteria.
_CURRICULUM_FALLBACK = {
    'spawn_standard': ((-1.10, -0.90), (-0.05, 0.05), (-5.0, 5.0)),
    'spawn_expanded': ((-1.20, -0.80), (-0.10, 0.10), (-10.0, 10.0)),
    'spawn_r_updates': 400,
    'f0_gate_steps': 500,
    'f0_max_abs_y': 0.20,
    'f0_max_yaw_deg': 10.0,
    'f0_forward_factor': 0.5,
    'f0_horizon_s': 10.0,
    'retention_cases': 15,
    'retention_per_speed': 5,
    'retention_fail_cases': 12,
    'retention_consecutive': 2,
    'diagnostic_cases': 15,
    'level_tick_min': 0,
    'level_tick_max': 100,
    'gate_streak': 2,
}
_CURRICULUM_PATHS = {
    'spawn_standard': ('spawn_standard', 'spawn.standard', 'initial_spawn',
                       'spawns.standard', 'birth.standard'),
    'spawn_expanded': ('spawn_expanded', 'spawn.expanded', 'perturbed_spawn',
                       'spawns.expanded', 'birth.expanded'),
    'spawn_r_updates': ('spawn_r_updates', 'r_spawn_updates', 'r_ramp_updates'),
    'f0_gate_steps': ('f0_gate_steps', 'f0_gate.steps', 'f0.steps', 'flat_gate_steps'),
    'f0_max_abs_y': ('f0_max_abs_y', 'f0_gate.max_abs_y', 'f0.max_abs_y'),
    'f0_max_yaw_deg': ('f0_max_yaw_deg', 'f0_gate.max_yaw_deg', 'f0.max_yaw_deg'),
    'f0_forward_factor': ('f0_forward_factor', 'f0_gate.forward_factor',
                          'f0.forward_factor'),
    'f0_horizon_s': ('f0_horizon_s', 'f0_gate.horizon_s', 'f0.horizon_s'),
    'retention_cases': ('retention_cases', 'retention.cases', 'retention_total'),
    'retention_per_speed': ('retention_per_speed', 'retention.per_speed'),
    'retention_fail_cases': ('retention_fail_cases', 'retention.fail_cases',
                             'retention_fail_threshold'),
    'retention_consecutive': ('retention_consecutive', 'retention.consecutive',
                              'retention_fail_streak'),
    'diagnostic_cases': ('diagnostic_cases', 'diagnostic.cases'),
    'level_tick_min': ('level_tick_min', 'level.tick_min'),
    'level_tick_max': ('level_tick_max', 'level.tick_max', 'level_tick_limit'),
    'gate_streak': ('gate_streak', 'gate.consecutive'),
}


def _resolve_seed_offsets():
    """D1 offsets: read the frozen spec constants, fall back to the SDS literals."""
    resolved, fallbacks = {}, []
    module_names = {'validation': ('VALIDATION_SEED_OFFSET',),
                    'diagnostic': ('DIAGNOSTIC_SEED_OFFSET',),
                    'perturbed': ('PERTURBED_SEED_OFFSET',),
                    'test': ('FINAL_TEST_SEED',)}
    mapping = getattr(spec, 'SEED_OFFSETS', None)
    mapping = mapping if isinstance(mapping, dict) else {}
    for kind, names in module_names.items():
        value = None
        for name in names:
            value = getattr(spec, name, None)
            if value is not None:
                break
        if value is None:
            for alias in ((kind,) if kind != 'test' else ('test', 'final')):
                if alias in mapping:
                    value = mapping[alias]
                    break
        if value is None:
            value = _SEED_OFFSET_FALLBACK[kind]
            fallbacks.append(kind)
        resolved[kind] = int(value)
    return resolved, tuple(fallbacks)


SEED_OFFSETS, SEED_OFFSET_FALLBACKS = _resolve_seed_offsets()


def _find(mapping, path):
    current = mapping
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


#: Flat module attributes the frozen spec exposes for each curriculum field.
_SPEC_ATTRS = {
    'spawn_standard': ('SPAWN_X', 'SPAWN_Y', 'SPAWN_YAW_DEGREES'),
    'spawn_expanded': ('R_SPAWN_X', 'R_SPAWN_Y', 'R_SPAWN_YAW_DEGREES'),
    'spawn_r_updates': ('R_SPAWN_RAMP_UPDATES',),
    'f0_gate_steps': ('F0_OBSERVE_STEPS',),
    'f0_max_abs_y': ('F0_LATERAL_MAX',),
    'f0_max_yaw_deg': ('F0_HEADING_MAX_DEGREES',),
    'f0_forward_factor': ('F0_PROGRESS_FRACTION',),
    'f0_horizon_s': ('F0_PROGRESS_SECONDS',),
    'retention_cases': ('RETENTION_CASES',),
    'retention_per_speed': ('RETENTION_CASES_PER_SPEED',),
    'retention_fail_cases': ('RETENTION_FAIL_MIN',),
    'retention_consecutive': ('RETENTION_FAIL_STREAK',),
    'diagnostic_cases': ('DIAGNOSTIC_CASES',),
    'level_tick_min': ('LEVEL_TICK_MIN',),
    'level_tick_max': ('LEVEL_TICK_MAX',),
    'gate_streak': ('GATE_STREAK_REQUIRED',),
}


def _resolve_curriculum():
    raw = getattr(spec, 'CURRICULUM', {})
    if not isinstance(raw, dict):
        raw = {}
    settings, sources = {}, {}
    for name, fallback in _CURRICULUM_FALLBACK.items():
        value, source = None, 'fallback'
        attributes = _SPEC_ATTRS.get(name, ())
        if name in ('spawn_standard', 'spawn_expanded') and all(
                hasattr(spec, attribute) for attribute in attributes):
            value = tuple(getattr(spec, attribute) for attribute in attributes)
            source = 'spec.' + '/'.join(attributes)
        if value is None:
            for attribute in attributes:
                found = getattr(spec, attribute, None)
                if found is not None:
                    value, source = found, f'spec.{attribute}'
                    break
        if value is None:
            for path in _CURRICULUM_PATHS[name]:
                found = _find(raw, path)
                if found is not None:
                    value, source = found, f'spec.CURRICULUM[{path!r}]'
                    break
        if value is None:
            value = fallback
        if name in ('spawn_standard', 'spawn_expanded'):
            value = tuple(tuple(float(item) for item in pair) for pair in value)
        settings[name] = value
        sources[name] = source
    for name, value in settings.items():
        if name in ('spawn_standard', 'spawn_expanded'):
            continue
        if name.endswith(('_cases', '_steps', '_updates', '_consecutive', '_speed',
                          '_min', '_max')) or name == 'gate_streak':
            settings[name] = int(value)
        else:
            settings[name] = float(value)
    settings['source'] = sources
    return settings


CURRICULUM_SETTINGS = _resolve_curriculum()
#: Q003: the one effective forward-distance definition.  Recorded in every
#: evaluation descriptor/identity; a net displacement, signed, never clamped.
FORWARD_DISTANCE_DEFINITION_FALLBACK = 'net_displacement_pre_reset_500_steps'


def forward_distance_definition():
    """The frozen Q003 forward-distance definition (spec first, Q003 fallback)."""
    for name in ('FORWARD_DISTANCE_DEFINITION', 'forward_distance_definition'):
        value = getattr(spec, name, None)
        if isinstance(value, str) and value:
            return value
        if callable(value):
            return str(value())
    return FORWARD_DISTANCE_DEFINITION_FALLBACK


def net_displacement(start_x, end_x):
    """Signed ``x_local[500, pre_reset] - x_local[0, after_reset]`` (Q003)."""
    start = float(start_x)
    end = float(end_x)
    for name, value in (('start_x', start), ('end_x', end)):
        if not math.isfinite(value):
            raise ValueError(f'Q003 net displacement requires a finite {name}')
    return end - start


def positive_path_length(positions):
    """Cumulative forward path length (diagnostic only; never a gate criterion)."""
    values = [float(value) for value in positions]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('positive_path_length requires finite positions')
    return float(sum(max(current - previous, 0.0)
                     for previous, current in zip(values, values[1:])))


F0_GATE_STEPS = CURRICULUM_SETTINGS['f0_gate_steps']
F0_MAX_ABS_Y = CURRICULUM_SETTINGS['f0_max_abs_y']
F0_MAX_YAW_DEG = CURRICULUM_SETTINGS['f0_max_yaw_deg']
F0_FORWARD_FACTOR = CURRICULUM_SETTINGS['f0_forward_factor']
F0_HORIZON_S = CURRICULUM_SETTINGS['f0_horizon_s']
RETENTION_CASES = CURRICULUM_SETTINGS['retention_cases']
RETENTION_PER_SPEED = CURRICULUM_SETTINGS['retention_per_speed']
RETENTION_FAIL_CASES = CURRICULUM_SETTINGS['retention_fail_cases']
RETENTION_CONSECUTIVE = CURRICULUM_SETTINGS['retention_consecutive']
DIAGNOSTIC_CASES = CURRICULUM_SETTINGS['diagnostic_cases']


class MissingConstant(RuntimeError):
    """A Q001-7 required constant is absent or invalid in the frozen spec.

    Missing configuration is a hard error (preflight/construction), never a silent
    default: the collapse streak may only come from ``spec.DIAGNOSTIC_COLLAPSE_STREAK``
    and it must be part of the configuration/SHA record.
    """


def diagnostic_collapse_streak():
    """The inherited stall-collapse streak (Q001-7), required, never defaulted.

    Scope is exactly the inherited r2 stagnation collapse: 15 diagnostic cases
    (5 per protocol speed), per-case ``stall_samples/active_samples >= 0.80``
    counted for >=12 cases, stairs additionally >=12 cases without a valid first
    step (the two counts are independent, not an intersection).  Flat has no
    first-step condition.  This number must **not** be reused for any other
    diagnostic anomaly (retention and minimum-level streaks are separate).
    """
    value = getattr(spec, 'DIAGNOSTIC_COLLAPSE_STREAK', None)
    if value is None:
        raise MissingConstant(
            'spec.DIAGNOSTIC_COLLAPSE_STREAK is required by Q001-7 and must not be '
            'defaulted; the frozen spec module does not define it')
    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 2:
        raise MissingConstant(
            f'spec.DIAGNOSTIC_COLLAPSE_STREAK must be an integer >= 2, got {value!r}')
    return int(value)


def _find_collapse_entry(value, streak):
    if isinstance(value, dict):
        for key, item in value.items():
            if 'collapse' in str(key).lower() and item == streak \
                    and not isinstance(item, bool):
                return str(key)
            found = _find_collapse_entry(item, streak)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_collapse_entry(item, streak)
            if found is not None:
                return found
    return None


def require_diagnostic_collapse_streak():
    """Preflight assertion: the constant exists, is valid and is in the config/SHA.

    ``spec.DIAGNOSTIC_COLLAPSE_STREAK`` must be part of the frozen configuration
    record (``spec.reward_config()``/``spec.contract()``) so it is covered by the
    configuration SHA; otherwise the run may not start.
    """
    streak = diagnostic_collapse_streak()
    checks = {}
    sources = []
    for name in ('reward_config', 'contract'):
        hook = getattr(spec, name, None)
        if not callable(hook):
            continue
        try:
            config = hook()
        except Exception as error:  # noqa: BLE001 - a broken config is a hard error
            raise MissingConstant(
                f'spec.{name}() cannot be evaluated while checking the collapse '
                f'constant: {error!r}') from error
        entry = _find_collapse_entry(config, streak)
        checks[name] = entry is not None
        if entry is not None:
            sources.append(f'spec.{name}()[{entry!r}]')
    if not any(checks.values()):
        raise MissingConstant(
            'spec.DIAGNOSTIC_COLLAPSE_STREAK is not part of the frozen config/SHA '
            'record (spec.reward_config()/spec.contract())')
    return dict(value=streak, checks=checks, sources=sources,
                reward_config_sha256=(spec.reward_config_sha256()
                                      if callable(getattr(spec, 'reward_config_sha256', None))
                                      else None))


# ------------------------------------------------------------------- phases
def phase_order():
    return tuple(entry.phase for entry in spec.PHASES)


def is_stairs_phase(phase):
    return str(phase).upper() != 'F0'


def gate_speeds(phase):
    """The gate speeds of a phase: F0/H/D/L 0.3/0.5/0.8, V/R 0.2/0.4/0.6."""
    return tuple(float(value) for value in spec.phase_spec(phase).gate_speeds)


def gate_speed_group(phase):
    """``flat`` for F0 (0.3/0.5/0.8) and ``climb`` for the stair phases."""
    return 'flat' if str(phase).upper() == 'F0' else 'climb'


def phase_budget(phase):
    return int(spec.phase_budget(phase))


def total_budget():
    return int(spec.total_budget())


def required_cases(kind, phase=None, episodes=None):
    """Per-speed pass count required by the SDS gate table."""
    if kind == 'test':
        return int(FINAL_PASS_MIN)
    if kind in ('diagnostic',):
        return None
    return GATE_PASS_MIN


def diagnostic_per_speed(phase=None):
    """Q001-7: 15 diagnostic cases total = five per protocol speed."""
    speeds = gate_speeds(phase or 'H')
    if not speeds:
        raise MissingConstant('the diagnostic speed group is empty')
    per_speed, remainder = divmod(int(DIAGNOSTIC_CASES), len(speeds))
    if remainder:
        raise EvaluationErrorLike(
            f'DIAGNOSTIC_CASES={DIAGNOSTIC_CASES} is not divisible by '
            f'{len(speeds)} protocol speeds')
    return per_speed


class EvaluationErrorLike(RuntimeError):
    """The frozen diagnostic case count cannot form an equal per-speed batch."""


def case_episodes(kind, phase=None):
    """Cases **per speed** for one table kind."""
    if kind == 'test':
        return int(FINAL_CASES)
    if kind == 'diagnostic':
        return diagnostic_per_speed(phase)
    if kind in ('flat_retention', 'stair_retention'):
        return int(RETENTION_PER_SPEED)
    return int(GATE_CASES)


def speed_episode_count(kind, phase=None):
    return case_episodes(kind, phase)


def spawn_range(phase, kind=None):
    """(x, y, yaw_deg) spawn ranges for a case table."""
    phase = str(phase).upper()
    if kind in ('diagnostic',):
        return ((-1.0, -1.0), (0.0, 0.0), (0.0, 0.0))
    if kind == 'test' or phase == 'R':
        return CURRICULUM_SETTINGS['spawn_expanded']
    if kind == 'perturbed':
        return CURRICULUM_SETTINGS['spawn_expanded']
    return CURRICULUM_SETTINGS['spawn_standard']


def measurement_stage(phase, kind):
    """The physical phase a case table measures (retention pins flat/H00)."""
    if kind == 'flat_retention':
        return 'F0'
    if kind == 'stair_retention':
        return 'H'
    return str(phase).upper()


def measurement_kind(stage):
    return 'flat' if str(stage).upper() == 'F0' else 'stair'


# -------------------------------------------------------------------- F0 gate
@dataclass(frozen=True)
class FlatCaseMetrics:
    steps: int
    max_abs_y: float
    max_abs_yaw_deg: float
    forward: float
    failure: bool
    required_forward: float

    def as_dict(self):
        return dict(steps=self.steps, max_abs_y=self.max_abs_y,
                    max_abs_yaw_deg=self.max_abs_yaw_deg, forward=self.forward,
                    forward_distance=self.forward,
                    forward_distance_definition=forward_distance_definition(),
                    failure=self.failure, required_forward=self.required_forward)


def flat_case_metrics(*, speed, steps, max_abs_y, max_abs_yaw_deg, forward=None,
                      failure=False, start_x=None, end_x=None):
    """The SDS 6.1 flat criteria with the Q003 signed net displacement.

    ``forward`` is ``D_net = x_local[500, pre_reset] - x_local[0, after_reset]``:
    signed, never clamped or absolute-valued.  Callers may pass ``start_x`` and
    ``end_x`` from the frozen pre-reset episode snapshot instead; passing both is
    allowed only when they agree exactly.
    """
    required = F0_FORWARD_FACTOR * float(speed) * F0_HORIZON_S
    if start_x is not None or end_x is not None:
        if start_x is None or end_x is None:
            raise ValueError('Q003 net displacement needs both start_x and end_x')
        derived = net_displacement(start_x, end_x)
        if forward is not None and float(forward) != derived:
            raise ValueError('forward differs from the Q003 start/end net displacement')
        forward = derived
    if forward is None:
        raise ValueError('flat case metrics require the Q003 net forward distance')
    metrics = FlatCaseMetrics(
        steps=int(steps), max_abs_y=float(max_abs_y),
        max_abs_yaw_deg=float(max_abs_yaw_deg), forward=float(forward),
        failure=bool(failure), required_forward=float(required))
    for name, value in metrics.as_dict().items():
        if name in ('failure', 'required_forward', 'steps',
                    'forward_distance_definition'):
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f'flat case metric {name} is not finite')
    return metrics


def flat_case_pass(metrics):
    """SDS 6.1 + Q003: 500 steps, |y|<=0.20, |yaw|<=10 deg, signed D_net >= 5*c."""
    if not isinstance(metrics, FlatCaseMetrics):
        raise TypeError('flat_case_pass needs FlatCaseMetrics')
    return bool(metrics.steps == F0_GATE_STEPS
                and not metrics.failure
                and metrics.max_abs_y <= F0_MAX_ABS_Y + 1e-12
                and metrics.max_abs_yaw_deg <= F0_MAX_YAW_DEG + 1e-12
                and metrics.forward >= metrics.required_forward - 1e-9)


# ------------------------------------------------------------------ retention
def retention_due(phase, global_update, passed):
    """Whether an SDS 6.3 retention check is due at this complete update boundary.

    ``passed`` maps ``'flat'``/``'h00'`` to whether that set has already passed
    the full gate (retention only starts after the corresponding pass).
    """
    if int(global_update) <= 0 or int(global_update) % int(RETENTION_INTERVAL):
        return None
    if not any(bool(value) for value in passed.values()):
        return None
    return 'flat_retention' if passed.get('flat') else 'stair_retention'


def retention_decision(streak, failed_cases, cases=RETENTION_CASES):
    """SDS 6.3: two consecutive sets with >=12/15 failures stop training."""
    if int(cases) != int(RETENTION_CASES):
        raise ValueError('retention set size must be the frozen 15 cases')
    streak = int(streak) + 1 if int(failed_cases) >= int(RETENTION_FAIL_CASES) else 0
    return dict(failed_cases=int(failed_cases), cases=int(cases),
                streak=streak,
                stop=bool(streak >= RETENTION_CONSECUTIVE),
                state='behavior_regression' if streak >= RETENTION_CONSECUTIVE else None)


__all__ = [
    'CASE_KINDS', 'CURRICULUM_SETTINGS', 'CODE_VERSION', 'CONTROL_DT',
    'FORWARD_DISTANCE_DEFINITION_FALLBACK',
    'DIAGNOSTIC_CASES', 'DIAGNOSTIC_INTERVAL', 'F0_FORWARD_FACTOR', 'F0_GATE_STEPS',
    'F0_HORIZON_S', 'F0_MAX_ABS_Y', 'F0_MAX_YAW_DEG', 'FINAL_CASES', 'FINAL_PASS_MIN',
    'FlatCaseMetrics', 'GATE_CASES', 'GATE_INTERVAL', 'GATE_PASS_MIN', 'PROTOCOL_ID',
    'RECIPE_VERSION', 'RETENTION_CASES', 'RETENTION_CONSECUTIVE', 'RETENTION_FAIL_CASES',
    'MissingConstant', 'RETENTION_INTERVAL', 'RETENTION_PER_SPEED', 'SAVE_INTERVAL',
    'SCHEMA_VERSION',
    'SEED_OFFSETS', 'SEED_OFFSET_FALLBACKS', 'SPEED_TOLERANCE', 'TASK_ID',
    'case_episodes', 'diagnostic_collapse_streak', 'flat_case_metrics', 'flat_case_pass',
    'forward_distance_definition', 'net_displacement', 'positive_path_length',
    'gate_speed_group',
    'gate_speeds', 'is_stairs_phase', 'measurement_kind', 'measurement_stage',
    'phase_budget', 'phase_order', 'required_cases', 'retention_decision',
    'require_diagnostic_collapse_streak', 'retention_due', 'spawn_range',
    'speed_episode_count', 'total_budget',
]
