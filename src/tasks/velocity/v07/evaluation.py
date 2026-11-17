"""v0.6 evaluation: fixed case tables, D1 seed offsets, gates and isolation proof.

SDS 8.3 D1 anti-regression (implemented exactly once, here, in the case
construction layer):

===================  ==========================  =========================
kind                 table seed                  note
===================  ==========================  =========================
``validation``       ``seed + 200300``           every full gate
``diagnostic``       ``seed + 500300``           every 50 updates
``perturbed``        ``seed + 300300``           perturbation set
``test``             ``400342`` (fixed)          the final frozen test
===================  ==========================  =========================

The runner always passes the **raw** training seed; no other module adds an
offset.  Gates require 18/20 per speed (95/100 for the final test).  Evaluation
runs deterministic action means and must leave the model, the normalizer
buffers, the optimizer and the counters untouched: :func:`evaluate_cases`
compares a serializable snapshot before and after and reports the real
``statistics_unchanged`` flag instead of hard-coding it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable, Optional

import numpy as np
import torch

from . import spec
from . import speed

OFFSET_SOURCE = 'case_construction' if not speed.SEED_OFFSET_FALLBACKS else 'fallback'
VALIDATION_SEED_OFFSET = speed.SEED_OFFSETS['validation']
DIAGNOSTIC_SEED_OFFSET = speed.SEED_OFFSETS['diagnostic']
PERTURBED_SEED_OFFSET = speed.SEED_OFFSETS['perturbed']
FINAL_TEST_SEED = speed.SEED_OFFSETS['test']


class EvaluationError(RuntimeError):
    """An evaluation result is incomplete or violates its frozen table."""


class EvaluationIsolationError(RuntimeError):
    """Evaluation changed trainable/persistent training state."""


def _canonical(value):
    return spec.canonical_json(jsonable(value))


def jsonable(value):
    """Convert tensors/numpy scalars into plain JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('evaluation evidence cannot contain NaN/Inf')
        return value
    return value


#: Case kind -> the frozen spec seed kind (retention reuses the validation table).
SEED_KIND = {'validation': 'validation', 'diagnostic': 'diagnostic',
             'perturbed': 'perturbed', 'test': 'final',
             'flat_retention': 'validation', 'stair_retention': 'validation'}


def seed_for(kind, seed):
    """The one and only seed-offset rule (SDS 8.3), delegated to the frozen spec."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError('training seed must be a positive integer')
    if kind not in SEED_KIND:
        raise ValueError(f'unknown case kind: {kind!r}')
    frozen = getattr(spec, 'evaluation_seed', None)
    if callable(frozen):
        return int(frozen(SEED_KIND[kind], seed))
    if kind == 'validation':
        return VALIDATION_SEED_OFFSET + seed
    if kind == 'perturbed':
        return PERTURBED_SEED_OFFSET + seed
    if kind == 'diagnostic':
        return DIAGNOSTIC_SEED_OFFSET + seed
    if kind == 'test':
        return FINAL_TEST_SEED
    return VALIDATION_SEED_OFFSET + seed


# ---------------------------------------------------------------- case table
@dataclass
class CaseTable:
    """One frozen (phase, kind) case table."""

    phase: str
    kind: str
    seed: int
    table_seed: int
    arrays: dict
    cases: list
    episodes: int
    measurement_stage: str
    measurement: str
    sha256: str = ''
    substage: Optional[str] = None
    round: Optional[int] = None
    #: Q002-3: the case generator's explicit spawn identity (never inferred).
    spawn_profile: str = 'standard'
    spawn_parameters: dict = field(default_factory=dict)
    target_tile_id: Optional[int] = None
    target_lesson: Optional[str] = None

    @property
    def speeds(self):
        return [float(value) for value in self.arrays]

    @property
    def per_speed(self):
        return len(self.arrays[f'{self.speeds[0]:.1f}']['x'])

    def initial_poses(self, speed):
        entry = self.arrays[f'{float(speed):.1f}']
        return [list(pose) for pose in zip(entry['x'], entry['y'], entry['yaw'])]

    def case_ids(self, speed):
        return [f'{self.kind}:{float(speed):.1f}:{index:03d}'
                for index in range(self.per_speed)]

    def to_json(self):
        return dict(task=getattr(spec, 'TASK_ID', spec.TASK), recipe_version=spec.RECIPE_VERSION,
                    code_version=spec.CODE_VERSION,
                    checkpoint_schema=int(getattr(spec, 'CHECKPOINT_SCHEMA', spec.SCHEMA_VERSION)),
                    architecture=spec.ARCHITECTURE, protocol_id=spec.PROTOCOL_ID,
                    phase=self.phase, substage=self.substage, kind=self.kind,
                    seed=int(self.seed), table_seed=int(self.table_seed),
                    offset_source=OFFSET_SOURCE, episodes=int(self.episodes),
                    measurement_stage=self.measurement_stage,
                    measurement=self.measurement, round=self.round,
                    spawn_profile=self.spawn_profile,
                    spawn_parameters=deepcopy(self.spawn_parameters),
                    forward_distance_definition=speed.forward_distance_definition(),
                    target_tile_id=self.target_tile_id,
                    target_lesson=self.target_lesson,
                    speed_order=list(self.speeds),
                    arrays=deepcopy(self.arrays), cases=deepcopy(self.cases))


def case_table_sha(table):
    return hashlib.sha256(_canonical(table.to_json()).encode('utf-8')).hexdigest()


def _generate_cases(phase, kind, table_seed, profile, speeds, per_speed,
                    measurement_stage, tile_id=None, lesson_id=None):
    """The single case generator; ``check_table`` regenerates with it too."""
    (x_lo, x_hi), (y_lo, y_hi), (yaw_lo, yaw_hi) = (
        profile['x'], profile['y'], profile['yaw_radians'])
    rng = _new_rng(table_seed)
    arrays, cases = {}, []
    for value in speeds:
        x_values, y_values, yaw_values = [], [], []
        for index in range(per_speed):
            if kind == 'diagnostic':
                pose = [-1.0, 0.0, 0.0]
            else:
                # Consumption order is frozen: x, then y, then yaw (radians).
                pose = [float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)),
                        float(rng.uniform(yaw_lo, yaw_hi))]
            x_values.append(pose[0])
            y_values.append(pose[1])
            yaw_values.append(pose[2])
            case = dict(case_id=f'{kind}:{float(value):.1f}:{index:03d}',
                        vx=float(value), episode_index=index,
                        initial_pose=list(pose),
                        measurement_stage=measurement_stage,
                        kind=kind, phase=phase,
                        spawn_profile=profile['name'])
            if tile_id is not None:
                case['tile_id'] = int(tile_id)
                case['lesson_id'] = lesson_id
            cases.append(case)
        arrays[f'{float(value):.1f}'] = dict(x=x_values, y=y_values, yaw=yaw_values)
    return arrays, cases


def make_cases(phase, substage, kind, seed, round=None, tile_id=None):
    """Build the frozen case table for one (phase, substage, kind, seed)."""
    phase = str(phase).upper()
    if kind not in speed.CASE_KINDS:
        raise EvaluationError(f'unknown case kind: {kind!r}')
    if phase not in speed.phase_order():
        raise EvaluationError(f'unknown v0.6 phase: {phase!r}')
    if kind == 'test' and phase != 'R':
        raise EvaluationError('the frozen final test runs on the final phase R')
    measurement_stage = speed.measurement_stage(phase, kind)
    measurement = speed.measurement_kind(measurement_stage)
    table_seed = seed_for(kind, seed)
    # A final test always measures the 40-step target stair at V/R speeds.  A
    # retention set keeps its own frozen identity (SDS 6.3: the three *initial*
    # speeds, 5 cases each) in every training phase: flat = F0 (0.3/0.5/0.8,
    # standard spawn), stair = H00 (0.3/0.5/0.8, standard spawn).
    if kind == 'flat_retention':
        speeds = speed.gate_speeds('F0')
        profile_phase = 'F0'
    elif kind == 'stair_retention':
        speeds = speed.gate_speeds('H')
        profile_phase = 'H'
    else:
        speeds = (speed.gate_speeds('R') if kind == 'test'
                  else speed.gate_speeds(phase))
        profile_phase = 'R' if kind == 'test' else phase
    per_speed = speed.speed_episode_count(kind, phase)
    episodes = per_speed
    target = target_tile(phase, substage, kind, tile_id)
    target_lesson = str(spec.terrain_spec(target).lesson_id)
    profile = spawn_profile(profile_phase, kind)
    arrays, cases = _generate_cases(phase, kind, table_seed, profile, speeds,
                                    per_speed, measurement_stage, target,
                                    target_lesson)
    table = CaseTable(phase=phase, kind=kind, seed=int(seed), table_seed=int(table_seed),
                      arrays=arrays, cases=cases, episodes=int(per_speed),
                      measurement_stage=measurement_stage, measurement=measurement,
                      substage=substage, round=round,
                      spawn_profile=profile['name'], spawn_parameters=profile,
                      target_tile_id=target, target_lesson=target_lesson)
    check_table(table)
    table.sha256 = case_table_sha(table)
    return table


def check_table(table):
    if not isinstance(table, CaseTable):
        raise TypeError('check_table needs a CaseTable')
    if table.phase not in speed.phase_order() or table.kind not in speed.CASE_KINDS:
        raise EvaluationError('case table identity is unknown')
    if table.kind == 'test':
        expected_speeds = speed.gate_speeds('R')
    elif table.kind == 'flat_retention':
        expected_speeds = speed.gate_speeds('F0')
    elif table.kind == 'stair_retention':
        expected_speeds = speed.gate_speeds('H')
    else:
        expected_speeds = speed.gate_speeds(table.phase)
    if tuple(table.speeds) != tuple(expected_speeds):
        raise EvaluationError('case table does not hold the phase gate speeds')
    if table.episodes != speed.speed_episode_count(table.kind, table.phase):
        raise EvaluationError('case table holds the wrong episode count per speed')
    if table.kind == 'test':
        if table.phase != 'R':
            raise EvaluationError('the final test table must belong to R')
        # Q002-3: the final test is the full R expansion, no training ramp, L40.
        if table.spawn_profile != FULL_R_EXPANDED:
            raise EvaluationError('the final test table must declare full_R_expanded')
        expected = spawn_profile('R', 'test')
        if table.spawn_parameters.get('name') != FULL_R_EXPANDED:
            raise EvaluationError('the final test table must carry full_R_expanded parameters')
        for key in ('x', 'y', 'yaw_degrees', 'yaw_radians'):
            if tuple(table.spawn_parameters[key]) != tuple(expected[key]):
                raise EvaluationError(f'final test spawn parameter {key!r} is not the R range')
        if table.spawn_parameters.get('training_ramp') is not False:
            raise EvaluationError('the final test must not apply a training spawn ramp')
        if table.spawn_parameters.get('sampling') != 'independent_uniform':
            raise EvaluationError('the final test must use independent uniform sampling')
        if int(table.target_tile_id) != int(spec.FINAL_TILE_ID) \
                or table.target_lesson != 'L40':
            raise EvaluationError('the final test target must be the L40 stair')
    for value in table.speeds:
        entry = table.arrays[f'{value:.1f}']
        for key in ('x', 'y', 'yaw'):
            if len(entry[key]) != table.episodes:
                raise EvaluationError('incomplete case table speed block')
            if not np.isfinite(np.asarray(entry[key], dtype=np.float64)).all():
                raise EvaluationError('case table holds a non-finite initial pose')
    expected_tile = target_tile(table.phase, table.substage, table.kind,
                                table.target_tile_id)
    if table.target_tile_id is None or int(table.target_tile_id) != int(expected_tile):
        raise EvaluationError('case table target_tile_id is missing or inconsistent')
    if table.target_lesson != str(spec.terrain_spec(expected_tile).lesson_id):
        raise EvaluationError('case table target lesson does not match its tile')
    for case in table.cases:
        if int(case.get('tile_id', -1)) != int(expected_tile):
            raise EvaluationError('a case does not measure the table target tile')
    # The arrays must be exactly the declared generator's output (Q002-3): a
    # re-labelled or hand-edited table is rejected, not merely range-checked.
    profile_phase = ('F0' if table.kind == 'flat_retention'
                     else ('H' if table.kind == 'stair_retention' else table.phase))
    profile = spawn_profile(profile_phase, table.kind)
    if table.kind == 'test':
        profile = spawn_profile('R', 'test')
    regenerated, cases = _generate_cases(
        table.phase, table.kind, table.table_seed, profile,
        tuple(table.speeds), table.episodes, table.measurement_stage,
        expected_tile, table.target_lesson)
    for value in table.speeds:
        key = f'{value:.1f}'
        for axis in ('x', 'y', 'yaw'):
            if [float(item) for item in table.arrays[key][axis]] != \
                    [float(item) for item in regenerated[key][axis]]:
                raise EvaluationError(
                    'case table arrays are not the declared generator output')
    expected_ids = {case_id for value in table.speeds
                    for case_id in table.case_ids(value)}
    if {case['case_id'] for case in table.cases} != expected_ids:
        raise EvaluationError('case table ids do not match its arrays')
    if table.measurement_stage != speed.measurement_stage(table.phase, table.kind):
        raise EvaluationError('case table measurement stage is wrong')
    return True


def table_from_json(value, *, expected_sha=None):
    if not isinstance(value, dict):
        raise ValueError('missing case table')
    table = CaseTable(phase=value['phase'], kind=value['kind'], seed=int(value['seed']),
                      table_seed=int(value.get('table_seed', seed_for(
                          value['kind'], int(value['seed'])))),
                      arrays=deepcopy(value['arrays']),
                      cases=[dict(case) for case in value.get('cases', [])],
                      episodes=int(value.get('episodes', speed.case_episodes(value['kind']))),
                      measurement_stage=value.get('measurement_stage'),
                      measurement=value.get('measurement', 'stair'),
                      substage=value.get('substage'), round=value.get('round'),
                      spawn_profile=value.get('spawn_profile', 'standard'),
                      spawn_parameters=deepcopy(value.get('spawn_parameters') or {}),
                      target_tile_id=value.get('target_tile_id'),
                      target_lesson=value.get('target_lesson'))
    check_table(table)
    table.sha256 = case_table_sha(table)
    if expected_sha is not None and table.sha256 != expected_sha:
        raise EvaluationError('retained case table SHA does not match its content')
    return table


# ------------------------------------------------------------- spawn profiles
#: Q002-3: the frozen final-test spawn identity.  A table carrying this name must
#: be generated with exactly these parameters, by independent uniform sampling
#: and with no training ramp.
FULL_R_EXPANDED = 'full_R_expanded'
DIAGNOSTIC_FIXED = 'diagnostic_fixed_pose'
STANDARD_SPAWN = 'standard'


def spawn_profile(phase, kind):
    """Explicit spawn identity for a case table (Q002-3), never inferred."""
    phase = str(phase).upper()
    if kind == 'test':
        return dict(name=FULL_R_EXPANDED,
                    x=tuple(float(v) for v in spec.R_SPAWN_X),
                    y=tuple(float(v) for v in spec.R_SPAWN_Y),
                    yaw_degrees=tuple(float(v) for v in spec.R_SPAWN_YAW_DEGREES),
                    yaw_radians=(float(math.radians(spec.R_SPAWN_YAW_DEGREES[0])),
                                 float(math.radians(spec.R_SPAWN_YAW_DEGREES[1]))),
                    sampling='independent_uniform', training_ramp=False,
                    ramp_updates=0, target_lesson='L40')
    if kind == 'diagnostic':
        return dict(name=DIAGNOSTIC_FIXED, x=(-1.0, -1.0), y=(0.0, 0.0),
                    yaw_degrees=(0.0, 0.0), yaw_radians=(0.0, 0.0),
                    sampling='fixed', training_ramp=False, ramp_updates=0)
    if phase == 'R' or kind == 'perturbed':
        return dict(name=FULL_R_EXPANDED,
                    x=tuple(float(v) for v in spec.R_SPAWN_X),
                    y=tuple(float(v) for v in spec.R_SPAWN_Y),
                    yaw_degrees=tuple(float(v) for v in spec.R_SPAWN_YAW_DEGREES),
                    yaw_radians=(float(math.radians(spec.R_SPAWN_YAW_DEGREES[0])),
                                 float(math.radians(spec.R_SPAWN_YAW_DEGREES[1]))),
                    sampling='independent_uniform', training_ramp=False, ramp_updates=0)
    return dict(name=STANDARD_SPAWN,
                x=tuple(float(v) for v in spec.SPAWN_X),
                y=tuple(float(v) for v in spec.SPAWN_Y),
                yaw_degrees=tuple(float(v) for v in spec.SPAWN_YAW_DEGREES),
                yaw_radians=(float(math.radians(spec.SPAWN_YAW_DEGREES[0])),
                             float(math.radians(spec.SPAWN_YAW_DEGREES[1]))),
                sampling='independent_uniform', training_ramp=False, ramp_updates=0)


def target_tile(phase, substage, kind, tile_id=None):
    """The single terrain a case table measures (Q-fix: a gate never defaults to H00).

    F0 measures the flat tile, D/L measure their substage tile, V/R the final L40
    stair and the frozen final test always L40.  Phase H has no substage: it
    *requires* the current frontier tile, because silently falling back to H00
    would let the easiest terrain pass a harder gate.

    Retention sets are the one phase-independent exception (Q007): the flat set
    always measures the flat tile 0 with the F0 protocol and the H00 stair set
    always measures tile 1.  Before this fix the tile was derived from the
    current phase, which put the flat retention set on an H00 stair.
    """
    phase = str(phase).upper()
    final_tile = int(spec.FINAL_TILE_ID)
    if kind == 'test':
        if tile_id is not None and int(tile_id) != final_tile:
            raise EvaluationError('the final test must measure the L40 tile')
        return final_tile
    if kind in ('flat_retention', 'stair_retention'):
        frozen = (int(spec.FLAT_TILE_ID) if kind == 'flat_retention'
                  else int(spec.tile_id_for('H00')))
        if tile_id is not None and int(tile_id) != frozen:
            raise EvaluationError(
                f'{kind} must measure its frozen retention tile {frozen}, not '
                f'{int(tile_id)}; retention terrains do not follow the training phase')
        return frozen
    if tile_id is not None:
        value = int(tile_id)
        try:
            spec.terrain_spec(value)
        except (ValueError,KeyError,TypeError) as error:
            raise EvaluationError(f'target tile {value} is outside the v0.7 terrain') from error
        if phase == 'F0' and value != int(spec.FLAT_TILE_ID):
            raise EvaluationError('F0 gates measure the flat tile only')
        if phase == 'H' and value == int(spec.FLAT_TILE_ID):
            raise EvaluationError('H gates cannot measure the flat tile')
        if phase in ('D', 'L') and substage is not None \
                and value != int(spec.tile_id_for(str(substage))):
            raise EvaluationError(
                f'{phase} {substage} must measure its own substage tile, not {value}')
        if phase in ('V', 'R') and value != final_tile:
            raise EvaluationError(f'{phase} gates must measure the final L40 tile')
        return value
    if phase == 'F0':
        return int(spec.FLAT_TILE_ID)
    if phase == 'H':
        raise EvaluationError(
            'phase H case tables require the current frontier tile_id; defaulting to H00 '
            'would evaluate a harder gate on the easiest terrain')
    if phase in ('D', 'L'):
        if substage is None:
            raise EvaluationError(f'{phase} requires a substage to select its tile')
        return int(spec.tile_id_for(str(substage)))
    return final_tile


def _new_rng(seed):
    """The single case-table RNG factory (tests may spy on it)."""
    return np.random.default_rng(int(seed))


# --------------------------------------------------------------------- gates
def _flat_case_pass(record):
    """Q003: a flat case needs the 500-step net-distance evidence or a verdict."""
    has_metrics = all(record.get(name) is not None
                      for name in ('forward_distance', 'max_abs_y',
                                   'max_abs_yaw_deg', 'steps'))
    if has_metrics:
        metrics = speed.flat_case_metrics(
            speed=float(record.get('vx', 0.0)), steps=int(record['steps']),
            max_abs_y=float(record['max_abs_y']),
            max_abs_yaw_deg=float(record['max_abs_yaw_deg']),
            forward=float(record['forward_distance']),
            failure=bool(record.get('failure', False)))
        verdict = speed.flat_case_pass(metrics)
        if 'flat_pass' in record and bool(record['flat_pass']) != verdict:
            raise EvaluationError(
                'flat_pass disagrees with the Q003 net-displacement verdict')
        return verdict
    if 'flat_pass' not in record:
        raise EvaluationError(
            'a flat case has neither the Q003 net-displacement metrics nor a verdict; '
            'an incomplete evaluation is never a pass or a failure')
    return bool(record['flat_pass'])


def _case_pass(record, kind):
    if kind == 'diagnostic':
        return bool(record.get('climb_success', False)) and not bool(record.get('failure'))
    if record.get('measurement') == 'flat' or kind == 'flat_retention':
        return _flat_case_pass(record)
    return bool(record.get('climb_success', False)) and not bool(record.get('failure'))


def summarize(records, table):
    """Per-speed counts and gate verdicts from recomputed case records."""
    expected = {case['case_id'] for case in table.cases}
    if len(records) != len(expected) or {row['case_id'] for row in records} != expected:
        raise EvaluationError('missing or duplicated evaluation cases')
    required = speed.required_cases(table.kind, table.phase)
    per_speed = []
    for value in table.speeds:
        rows = [row for row in records if abs(float(row['vx']) - value) < speed.SPEED_TOLERANCE]
        if len(rows) != table.episodes:
            raise EvaluationError('incomplete speed denominator')
        count = len(rows)
        case_pass = sum(1 for row in rows if _case_pass(row, table.kind))
        per_speed.append(dict(
            c=float(value), total=count, case_pass_count=case_pass,
            case_pass_rate=case_pass / count,
            climb_success_count=sum(1 for row in rows if row.get('climb_success')),
            failure_count=sum(1 for row in rows if row.get('failure')),
            mean_vx=float(np.mean([row['mean_vx'] for row in rows])) if all(
                row.get('mean_vx') is not None for row in rows) else None,
            mean_relative_error=(float(np.mean([row['mean_relative_error'] for row in rows]))
                                 if all(row.get('mean_relative_error') is not None
                                        for row in rows) else None),
            failure_reasons={reason: sum(1 for row in rows if row.get('reason') == reason)
                             for reason in sorted({row.get('reason') for row in rows}
                                                  - {None}, key=str)},
            case_pass_required=required,
            passed=None if required is None else bool(case_pass >= required),
        ))
    gate_eligible = table.kind in ('validation', 'test')
    return dict(per_speed=per_speed, required=required, episodes=table.episodes,
                gate_pass=bool(gate_eligible and required is not None
                               and all(entry['passed'] for entry in per_speed)),
                gate_eligible=gate_eligible,
                forward_distance_definition=speed.forward_distance_definition(),
                retention=table.kind in ('flat_retention', 'stair_retention'),
                kind=table.kind, phase=table.phase, substage=table.substage,
                measurement=table.measurement, measurement_stage=table.measurement_stage,
                case_table_sha256=table.sha256, table_seed=table.table_seed,
                offset_source=OFFSET_SOURCE, speeds=list(table.speeds))


def gate_result(summary):
    """Compact, JSON-safe gate projection used by runner/recovery/telemetry."""
    if not isinstance(summary, dict) or 'per_speed' not in summary:
        raise EvaluationError('evaluation summary has no per_speed block')
    rows = [dict(c=float(entry['c']), case_pass_count=int(entry['case_pass_count']),
                 total=int(entry['total']),
                 required=(None if entry['case_pass_required'] is None
                           else int(entry['case_pass_required'])),
                 passed=(None if entry['passed'] is None else bool(entry['passed'])))
            for entry in summary['per_speed']]
    required = summary['required']
    return dict(kind=summary['kind'], phase=summary['phase'],
                substage=summary.get('substage'),
                measurement=summary['measurement'],
                measurement_stage=summary['measurement_stage'],
                case_table_sha256=summary['case_table_sha256'],
                table_seed=summary.get('table_seed'),
                required=None if required is None else int(required),
                gate_eligible=bool(summary.get('gate_eligible', False)),
                gate_pass=bool(summary.get('gate_eligible', False)
                               and all(item['passed'] for item in rows)),
                retention=bool(summary.get('retention', False)),
                speeds=rows,
                failed_cases=sum(item['total'] - item['case_pass_count'] for item in rows),
                forward_distance_definition=summary.get(
                    'forward_distance_definition', speed.forward_distance_definition()),
                statistics_unchanged=bool(summary.get('statistics_unchanged', False)),
                statistics_changed=list(summary.get('statistics_changed', [])))


def failed_cases(summary):
    return sum(int(entry['total']) - int(entry['case_pass_count'])
               for entry in summary['per_speed'])


# ------------------------------------------------- diagnostic stall collapse
# Q001-7: the inherited r2 stagnation collapse.  The consecutive-run threshold is
# the required spec constant ``DIAGNOSTIC_COLLAPSE_STREAK`` (no silent default);
# the per-case stall fraction, the minimum stalled-case count, the minimum
# first-step-absent count and the generation-update floor are the inherited r2
# numbers, read from the frozen r2 spec module so no literal lives here.
def _inherited(name):
    """Read the inherited r2 stagnation definition (never a local literal).

    The v0.6 frozen spec may re-export the inherited constants; otherwise the
    read-only r2 runner module (frozen 0.5-r2 definition) is the single source.
    """
    value = getattr(spec, name, None)
    if value is not None:
        return value
    from src.tasks.velocity.v05_r2 import runner as inherited_runner
    return getattr(inherited_runner, name)


def _collapse_constant(name):
    try:
        return _inherited(name)
    except AttributeError as error:  # pragma: no cover - frozen module is present
        raise evaluation_error('MissingConstant', name) from error


def evaluation_error(kind, name):
    return EvaluationError(f'{kind}: diagnostic collapse constant {name!r} is missing')


STALL_FRACTION = float(_collapse_constant('COLLAPSE_STALL_FRACTION'))
COLLAPSE_MIN_CASES = int(_collapse_constant('COLLAPSE_MIN_CASES'))
COLLAPSE_MIN_FIRST_STEP_ABSENT = int(_collapse_constant('COLLAPSE_MIN_FIRST_STEP_ABSENT'))
COLLAPSE_START_STAGE_UPDATE = int(_collapse_constant('COLLAPSE_START_STAGE_UPDATE'))


def diagnostic_decision(records, phase, *, generation_updates, generation=None):
    """Q001-7 inherited stall-collapse decision for one 15-case diagnostic set.

    The two counts are independent (not an intersection): >=12 cases with
    ``stall_samples/active_samples >= 0.80`` and, on stairs only, >=12 cases
    without a valid first step.  Flat has no first-step condition.  Missing
    cases, a zero denominator or a missing field is an evaluation error, never a
    non-hit and never a pass.  The consecutive-run threshold is
    :func:`speed.diagnostic_collapse_streak` (Q001-7); it is deliberately not
    used for any other diagnostic anomaly.
    """
    streak_required = speed.diagnostic_collapse_streak()
    phase = str(phase).upper()
    if phase not in speed.phase_order():
        raise EvaluationError(f'unknown v0.6 phase: {phase!r}')
    if isinstance(generation_updates, bool) or not isinstance(generation_updates, int):
        raise EvaluationError('diagnostic generation_updates must be an integer')
    expected_speeds = tuple(speed.gate_speeds(phase))
    if not isinstance(records, (list, tuple)) or len(records) != speed.DIAGNOSTIC_CASES:
        raise EvaluationError(
            f'the diagnostic must return exactly {speed.DIAGNOSTIC_CASES} cases')
    stairs = str(phase) != 'F0'
    cases = []
    for record in records:
        if not isinstance(record, dict):
            raise EvaluationError('invalid diagnostic case row')
        speed_value = record.get('vx', record.get('c'))
        if isinstance(speed_value, bool) or not isinstance(speed_value, (int, float)):
            raise EvaluationError('diagnostic case has no protocol speed')
        active = record.get('active_samples')
        stalled = record.get('stall_samples')
        if active is None or stalled is None:
            raise EvaluationError(
                'diagnostic case lacks active_samples/stall_samples (Q001-7)')
        if isinstance(active, bool) or not isinstance(active, int) or active <= 0:
            raise EvaluationError('diagnostic active_samples must be a positive integer')
        if isinstance(stalled, bool) or not isinstance(stalled, int) or stalled < 0:
            raise EvaluationError('diagnostic stall_samples must be a non-negative integer')
        if stalled > active:
            raise EvaluationError('diagnostic stall_samples exceeds active_samples')
        first_step = record.get('first_step_valid', record.get('first_step_acquired'))
        if stairs and first_step is None:
            raise EvaluationError('diagnostic case lacks first_step_valid (stairs)')
        if first_step is not None and type(first_step) is not bool:
            raise EvaluationError('diagnostic first_step_valid must be boolean')
        cases.append(dict(vx=float(speed_value), active_samples=int(active),
                          stall_samples=int(stalled),
                          stall_fraction=stalled / active,
                          first_step_valid=first_step))
    for value in expected_speeds:
        count = sum(1 for case in cases if abs(case['vx'] - value) < speed.SPEED_TOLERANCE)
        if count != speed.DIAGNOSTIC_CASES // len(expected_speeds):
            raise EvaluationError('diagnostic speed batch must have five cases per speed')
    stalled_cases = sum(1 for case in cases
                        if case['stall_fraction'] >= STALL_FRACTION - 1e-12)
    first_step_absent = (0 if not stairs else sum(1 for case in cases
                                                  if case['first_step_valid'] is False))
    eligible = int(generation_updates) >= COLLAPSE_START_STAGE_UPDATE
    hit = bool(eligible and stalled_cases >= COLLAPSE_MIN_CASES
               and (not stairs or first_step_absent >= COLLAPSE_MIN_FIRST_STEP_ABSENT))
    return dict(phase=phase, generation=generation, generation_updates=int(generation_updates),
                speeds=list(expected_speeds), stairs=stairs, eligible=eligible,
                cases=cases, stalled_cases=stalled_cases,
                first_step_absent_cases=first_step_absent,
                stall_fraction_threshold=STALL_FRACTION,
                min_stalled_cases=COLLAPSE_MIN_CASES,
                min_first_step_absent_cases=(0 if not stairs
                                             else COLLAPSE_MIN_FIRST_STEP_ABSENT),
                collapse_hit=hit, required_streak=streak_required)


def diagnostic_collapse_state(previous, decision):
    """Independent stall-collapse streak (Q001-7).

    A new generation clears the streak, a valid non-hit clears it, and only
    eligible consecutive hits accumulate to ``DIAGNOSTIC_COLLAPSE_STREAK``.  The
    retention (flat/H00) streaks and the minimum-level 4-window count live in the
    curriculum state and are never mixed into this counter.
    """
    previous = dict(previous or {})
    generation = decision.get('generation')
    streak = int(previous.get('streak', 0) or 0)
    if generation is not None and previous.get('generation') is not None \
            and int(generation) != int(previous['generation']):
        streak = 0
    if not decision.get('eligible'):
        streak = 0
    elif decision.get('collapse_hit'):
        streak += 1
    else:
        streak = 0
    required = int(decision.get('required_streak', speed.diagnostic_collapse_streak()))
    return dict(streak=streak, required_streak=required,
                generation=generation,
                generation_updates=decision.get('generation_updates'),
                update=decision.get('update', previous.get('update')),
                last_hit=bool(decision.get('collapse_hit')),
                collapse=bool(decision.get('eligible') and streak >= required))


# ----------------------------------------------------------------- isolation
def _tensor_sha(tensors):
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().to('cpu').contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('utf-8'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _scalar_token(value):
    """Deterministic scalar token that never swallows NaN/Inf or a type."""
    if isinstance(value, bool):
        return ['bool', bool(value)]
    if isinstance(value, int):
        return ['int', int(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ['float', 'nan']
        if math.isinf(value):
            return ['float', 'inf' if value > 0 else '-inf']
        return ['float', repr(value)]
    if isinstance(value, str):
        return ['str', value]
    if value is None:
        return ['none']
    if isinstance(value, (list, tuple)):
        return ['seq', [_scalar_token(item) for item in value]]
    if isinstance(value, dict):
        return ['map', {str(key): _scalar_token(value[key]) for key in sorted(value, key=str)}]
    return ['repr', f'{type(value).__name__}:{value!r}']


def _optimizer_digest(state):
    """Complete, deterministic digest of a torch optimizer state dict.

    Covers parameter-id -> per-parameter-state mapping (step/exp_avg/exp_avg_sq and
    any non-tensor structs) and the param_groups (lr/betas/eps/weight_decay/params
    membership).  Numeric values are tokenized (NaN/Inf included), tensors keep
    dtype/shape/values, and the mapping identity is preserved.  Reading never
    mutates the optimizer or leaves tensors behind.
    """
    if not isinstance(state, dict):
        raise EvaluationError('optimizer state_dict is not a mapping')
    groups = []
    for index, group in enumerate(state.get('param_groups') or []):
        if not isinstance(group, dict):
            raise EvaluationError(f'optimizer param_groups[{index}] is not a mapping')
        entry = {}
        for key in sorted(group, key=str):
            value = group[key]
            if torch.is_tensor(value):
                entry[str(key)] = dict(kind='tensor', dtype=str(value.dtype),
                                       shape=list(value.shape),
                                       sha256=_tensor_sha({str(key): value}))
            elif isinstance(value, (list, tuple)) and key != 'params':
                entry[str(key)] = _scalar_token(list(value))
            elif key == 'params':
                entry['params'] = [str(int(item)) if not torch.is_tensor(item)
                                   else _tensor_sha({'p': item}) for item in value]
            else:
                entry[str(key)] = _scalar_token(value)
        groups.append(entry)
    states = []
    raw_state = state.get('state') or {}
    if not isinstance(raw_state, dict):
        raise EvaluationError('optimizer state is not a parameter-id mapping')
    for parameter_id in sorted(raw_state, key=str):
        group = raw_state[parameter_id]
        if not isinstance(group, dict):
            raise EvaluationError(
                f'optimizer state[{parameter_id!r}] is not a per-parameter mapping')
        entry = {}
        for key in sorted(group, key=str):
            value = group[key]
            if torch.is_tensor(value):
                entry[str(key)] = dict(kind='tensor', dtype=str(value.dtype),
                                       shape=list(value.shape),
                                       sha256=_tensor_sha({str(key): value}))
            elif isinstance(value, dict):
                entry[str(key)] = _scalar_token(value)
            else:
                entry[str(key)] = _scalar_token(value)
        states.append(dict(parameter_id=str(parameter_id), state=entry))
    payload = dict(param_groups=groups, state=states,
                   state_parameter_ids=[str(key) for key in sorted(raw_state, key=str)])
    return hashlib.sha256(_canonical(jsonable(payload)).encode('utf-8')).hexdigest()


def training_state_snapshot(model, optimizer=None, counters=None, *, rng=True):
    """Serializable digest of everything evaluation must never change."""
    snapshot = dict(
        model=_tensor_sha({key: value for key, value in model.state_dict().items()}),
        std=_tensor_sha({'std': model.std.detach()}),
        normalizer=_tensor_sha(
            {f'actor.normalizer.{key}': value
             for key, value in model.actor.normalizer.state_dict().items()}),
        critic_normalizer=_tensor_sha(
            {f'critic_normalizer.{key}': value
             for key, value in model.critic_normalizer.state_dict().items()}),
        counters=dict(counters or {}),
        optimizer=None,
        rng=None,
    )
    if optimizer is not None:
        snapshot['optimizer'] = _optimizer_digest(optimizer.state_dict())
    if rng:
        snapshot['rng'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        snapshot['numpy_rng'] = hashlib.sha256(
            np.random.get_state()[1].tobytes()).hexdigest()
    return snapshot


def compare_snapshots(before, after):
    changed = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changed.append(key)
    return dict(statistics_unchanged=not changed, statistics_changed=changed,
                snapshot_sha=hashlib.sha256(_canonical(jsonable(before)).encode(
                    'utf-8')).hexdigest())


@dataclass
class EvalResult:
    table: CaseTable
    summary: dict
    records: list
    isolation: dict
    evidence_sha256: dict = field(default_factory=dict)

    @property
    def gate(self):
        return gate_result(self.summary)


def _record_fields(case, raw):
    if not isinstance(raw, dict):
        raise EvaluationError(f'case {case["case_id"]} returned no record mapping')
    record = dict(raw)
    record['case_id'] = case['case_id']
    record['vx'] = float(case['vx'])
    record.setdefault('climb_success', False)
    record.setdefault('failure', False)
    record.setdefault('flat_pass', False)
    record.setdefault('measurement', 'stair')
    if not record['measurement']:
        record['measurement'] = 'stair'
    record.setdefault('reason', None)
    return record


def batch_cases(table, size=8):
    """Split a table into same-speed batches (SDS eval env count = 8)."""
    batches = []
    for speed in table.speeds:
        rows = [case for case in table.cases
                if abs(float(case['vx']) - float(speed)) < speed_tolerance()]
        for start in range(0, len(rows), int(size)):
            batches.append(rows[start:start + int(size)])
    return batches


def speed_tolerance():
    return speed.SPEED_TOLERANCE


def evaluate_cases(model, table, case_runner: Callable, *, optimizer=None,
                   counters=None, device='cpu'):
    """Run every case with deterministic means and prove the training state is intact.

    ``case_runner(case) -> dict`` is supplied by the caller (the runner passes a
    V07Env-backed closure; tests pass a fake).  The model is switched to eval mode
    for the duration and restored afterwards.
    """
    if not isinstance(table, CaseTable):
        raise TypeError('evaluate_cases needs a CaseTable')
    if table.sha256 != case_table_sha(table):
        raise EvaluationError('case table SHA does not match its content')
    was_training = model.training
    before = training_state_snapshot(model, optimizer, counters)
    model.eval()
    records = []
    batched = bool(getattr(case_runner, 'supports_batch', False))
    try:
        with torch.no_grad():
            if batched:
                for batch in batch_cases(table):
                    rows = case_runner(batch)
                    if not isinstance(rows, (list, tuple)) or len(rows) != len(batch):
                        raise EvaluationError(
                            'a batched case runner must return exactly one record per '
                            'case in the batch')
                    if [row.get('case_id') for row in rows] != [
                            case['case_id'] for case in batch]:
                        raise EvaluationError('batched case records are out of case order')
                    records.extend(_record_fields(case, row)
                                   for case, row in zip(batch, rows))
            else:
                for case in table.cases:
                    records.append(_record_fields(case, case_runner(case)))
    finally:
        model.train(was_training)
    after = training_state_snapshot(model, optimizer, counters)
    isolation = compare_snapshots(before, after)
    if not isolation['statistics_unchanged']:
        raise EvaluationIsolationError(
            'evaluation modified training state: %s' % isolation['statistics_changed'])
    summary = summarize(records, table)
    summary.update(isolation)
    evidence = dict(case_table_sha256=table.sha256,
                    records_sha256=hashlib.sha256(_canonical(jsonable(records)).encode(
                        'utf-8')).hexdigest(),
                    summary_sha256=hashlib.sha256(_canonical(jsonable(summary)).encode(
                        'utf-8')).hexdigest())
    return EvalResult(table=table, summary=summary, records=records,
                      isolation=isolation, evidence_sha256=evidence)


def evaluate_checkpoint(model, phase, substage, kind, seed, case_runner, *,
                        optimizer=None, counters=None, round=None, device='cpu',
                        tile_id=None):
    """Convenience wrapper: build the frozen table, then evaluate it.

    ``tile_id`` is the measured terrain (H: current frontier, D/L: substage tile,
    V/R and the final test: L40).  It reaches the runner through ``case['tile_id']``.
    """
    table = make_cases(phase, substage, kind, seed, round=round, tile_id=tile_id)
    return evaluate_cases(model, table, case_runner, optimizer=optimizer,
                          counters=counters, device=device)


__all__ = [
    'DIAGNOSTIC_SEED_OFFSET', 'EvalResult', 'EvaluationError',
    'EvaluationIsolationError', 'FINAL_TEST_SEED', 'OFFSET_SOURCE',
    'PERTURBED_SEED_OFFSET', 'SEED_KIND', 'VALIDATION_SEED_OFFSET', 'CaseTable',
    'batch_cases', 'case_table_sha',
    'check_table', 'compare_snapshots', 'evaluate_cases', 'evaluate_checkpoint',
    'COLLAPSE_MIN_CASES', 'COLLAPSE_MIN_FIRST_STEP_ABSENT', 'COLLAPSE_START_STAGE_UPDATE',
    'STALL_FRACTION', 'diagnostic_collapse_state', 'diagnostic_decision', 'target_tile',
    'failed_cases', 'gate_result', 'jsonable', 'make_cases', 'seed_for', 'summarize',
    'table_from_json', 'training_state_snapshot',
]
