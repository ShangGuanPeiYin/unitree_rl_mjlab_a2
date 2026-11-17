"""v0.5-r2 evaluation: fixed case tables, gate arithmetic and lossless route evidence.

SDS 0.5-r2 revision 1.0 sections 4.8, 6 and 8.1.  This module is a derivative of
``v05/evaluation.py``; it owns exactly three things:

1. the *fixed* per-(stage, kind) case tables and their SHA (local NumPy RNG, one
   draw per x/y/yaw in speed order, reused by every later checkpoint).  The case
   *numbers*, seeds and generation rules are the SDS ones and are unchanged; the
   tables are generated in the r2 namespace and every table JSON therefore binds
   the r2 task/recipe/schema/architecture and the two r2 protocol ids, so an r2
   case SHA can never equal (or be mistaken for) an r1 case SHA (SDS 6.5, Q003-r2
   decision 3.6);
2. the pass counts and gate booleans, without any streak state -- streaks and
   the planned/result_committed/consumed ledger belong to runner/recovery;
3. a pure execution function :func:`evaluate_checkpoint` that runs the frozen
   Actor mean on the r2 environment's five-key observations, retains float32 raw
   evidence **including** ``route_error [steps,3]``/``local_y``/``yaw`` and returns
   an :class:`EvalResult` with the evidence paths and SHAs.

Gate rules are the r1 rules verbatim: F0 = 500 observed steps, ``|local_y| <= 0.20
m``, ``|yaw| <= 10 deg``, three speeds with ``case_pass_count >= 18/20`` and two
consecutive complete passes; P/S gates authorize on safe arrival only.  Nothing
here is relaxed for r2 and ``route_error`` never enters a pass criterion -- it is
retained as read-only evidence (SDS 4.8/9.3-20).

``evaluate_checkpoint`` never writes a ledger, never touches the training
environment, never consumes training RNG and never changes model statistics.  A
gate is only valid when the evidence on disk can be read back and independently
recomputed in CPU float64; :func:`verify_evaluation_evidence` does that read,
including an independent re-derivation of the retained ``route_error`` from the
retained ``local_y``/``yaw``.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Optional

import numpy as np

from . import spec as _spec
from .speed import (ROUTE_CLIP, ROUTE_LATERAL_SCALE, ROUTE_YAW_AXIS, cpu_speed_oracle,
                    route_state_contract, window_kind_for_stage)

TASK = _spec.TASK
PROTOCOL_ID = _spec.PROTOCOL_ID
FLAT_PROTOCOL_ID = _spec.FLAT_PROTOCOL_ID
CODE_VERSION = _spec.CODE_VERSION
RECIPE_VERSION = _spec.RECIPE_VERSION
SDS_REVISION = _spec.SDS_REVISION
SCHEMA_VERSION = _spec.SCHEMA_VERSION
STAGE_BY_ID = _spec.STAGE_BY_ID
STAGE_BY_TILE = _spec.STAGE_BY_TILE
STAGES = _spec.STAGES
SPEEDS = tuple(_spec.SPEEDS)
SPEED_ORDER = tuple(_spec.SPEED_ORDER)
EPISODES_PER_SPEED = _spec.EPISODES_PER_SPEED
EPISODES_PER_SPEED_FINAL = _spec.EPISODES_PER_SPEED_FINAL
#: SDS 6.3: every 50-update diagnostic is three speeds x five episodes.
DIAGNOSTIC_EPISODES = 5
#: SDS 6.3 collapse threshold for the stall-fraction count (diagnosis only).
DIAGNOSTIC_STALL_FRACTION = .80
DIAGNOSTIC_MINUTES_CASES = 12
CASE_PASS_REQUIRED = _spec.CASE_PASS_REQUIRED
CASE_PASS_REQUIRED_FINAL = _spec.CASE_PASS_REQUIRED_FINAL
VALIDATION_SEED_OFFSET = _spec.VALIDATION_SEED_OFFSET
PERTURBED_SEED_OFFSET = _spec.PERTURBED_SEED_OFFSET
DIAGNOSTIC_SEED_OFFSET = _spec.DIAGNOSTIC_SEED_OFFSET
FINAL_TEST_SEED = _spec.FINAL_TEST_SEED
F0_OBSERVE_STEPS = _spec.F0_OBSERVE_STEPS
F0_EPISODE_STEPS = _spec.F0_EPISODE_STEPS
ARCHITECTURE = getattr(_spec, 'ARCHITECTURE', None)
if ARCHITECTURE is None:
    _state = getattr(_spec, 'ROUTE_STATE', None)
    ARCHITECTURE = _state.get('architecture') if isinstance(_state, dict) else None
if ARCHITECTURE is None:
    raise RuntimeError('v0.5-r2 spec exports no architecture label; r2 evidence may '
                       'not be produced under an unversioned identity (SDS 1.1)')


def route_contract_record():
    """The frozen SDS 4.2 route contract as JSON-safe data (r2 identity).

    Prefers ``spec.route_state()`` (the full frozen table) and falls back to the
    constants bound in :mod:`v05_r2.speed`; either way the values are read, never
    re-declared here.
    """
    getter = getattr(_spec, 'route_state', None)
    if callable(getter):
        return jsonable(getter())
    return jsonable(route_state_contract())


def source_sha256():
    """SHA256 of the frozen r2 source closure (SDS 4.8/11 case-table identity).

    The per-file hash map comes from :func:`v05_r2.spec.behavior_hashes`; the
    canonical JSON of that map is hashed so one number identifies the exact source
    snapshot an evaluation report was produced from.  No local fallback value is
    allowed: evidence that cannot name its source is not evidence.
    """
    hashes = getattr(_spec, 'behavior_hashes', None)
    if hashes is None:
        raise RuntimeError('v0.5-r2 spec exports no behavior_hashes(); the evaluation '
                           'report cannot bind its source closure (SDS 1.3-3)')
    return hashlib.sha256(_spec.canonical_json(hashes()).encode('utf-8')).hexdigest()

#: Kinds accepted by :func:`make_cases` (SDS 6.2/6.3 + the flat retention test).
KINDS = ('validation', 'perturbed', 'diagnostic', 'test', 'flat_retention')
MIN_DISK_BYTES = 10 * 1024 ** 3
RAW_SCHEMA = 'v05-r2-lossless-window-evidence-1'
HEX = 64

#: Keep every case, only one batch in host memory.
FLOAT32_KEYS = ('body_vx', 'executed_c', 'local_x', 'local_y', 'yaw')
INT64_KEYS = ('step',)
#: SDS 4.8: the retained raw NPZ must carry ``route_error [steps,3]`` float32 in
#: addition to ``local_y``/``yaw``; a case without it has incomplete evidence.
ROUTE_KEYS = ('route_error',)

#: Gate kinds that authorize a stair promotion (speed never gates a stair climb).
STAIR_KINDS = ('validation', 'perturbed', 'test')

#: Evaluation execution modes.  ``probe`` is the W7 resource/lifecycle mode: it
#: measures the evaluation path but can never authorize a curriculum gate and is
#: never written into a pilot/formal ledger or checkpoint.
EVALUATION_MODES = ('pilot', 'formal', 'probe')
#: Kinds a probe may measure.  ``test`` (frozen final S5 model) and ``perturbed``
#: (second S1 gate add-on) are gate-protocol kinds and are rejected in probe mode.
PROBE_KINDS = ('validation', 'flat_retention', 'diagnostic')


def check_evaluation_mode(mode, kind):
    """Validate the (mode, kind) pair; probes are restricted to non-gate kinds."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f'evaluation mode must be one of {EVALUATION_MODES}')
    if mode == 'probe' and kind not in PROBE_KINDS:
        raise ValueError(f'probe mode measures only {PROBE_KINDS}; '
                         f'{kind!r} is a gate-protocol kind')


def apply_mode_policy(summary, mode):
    """Attach the execution-mode policy to a summary (probe never gates).

    The same function is used when a report is first written and when retained
    evidence is re-verified, so a probe can never be re-labelled as a gate.
    """
    summary = dict(summary)
    summary['mode'] = str(mode)
    summary['probe'] = bool(mode == 'probe')
    summary['counts_against_formal'] = bool(mode != 'probe')
    if mode == 'probe':
        summary['gate_eligible'] = False
        summary['gate_pass'] = False
        summary['gate_note'] = ('probe evaluation: resource/lifecycle measurement only, '
                                'never gate-eligible')
    else:
        summary.setdefault('gate_note', None)
    return summary


class DiskSpacePause(RuntimeError):
    """Requested controlled pause; it never authorizes deleting evidence."""

    reason = 'disk_space_pause'


# --------------------------------------------------------------------- misc
def jsonable(value, *, counter=None):
    """Recursively convert to JSON-safe Python data.

    Non-finite floats become ``None`` (SDS 6.1: an unavailable metric is null,
    never NaN) and are counted in ``counter['nonfinite']`` when a counter is
    supplied, so a residual non-finite value can never abort a whole evaluation
    at ``json.dumps(..., allow_nan=False)`` time.
    """
    if hasattr(value, 'detach'):
        return jsonable(value.detach().cpu().tolist(), counter=counter)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), counter=counter)
    if isinstance(value, np.generic):
        return jsonable(value.item(), counter=counter)
    if isinstance(value, float):
        if not math.isfinite(value):
            if counter is not None:
                counter['nonfinite'] = counter.get('nonfinite', 0) + 1
            return None
        return value
    if isinstance(value, dict):
        return {key: jsonable(item, counter=counter) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, counter=counter) for item in value]
    return value


def _canonical(value):
    return (_spec.canonical_json(jsonable(value)) + '\n').encode('utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    _atomic_bytes(path, _canonical(value))


def ensure_disk_space(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    usage = __import__('shutil').disk_usage(path)
    free = int(usage.free)
    if free < MIN_DISK_BYTES:
        raise DiskSpacePause(f'evaluation needs >= {MIN_DISK_BYTES} free bytes, observed {free}')
    return free


@contextmanager
def preserved_rng():
    """Protect the caller's RNG if evaluation ever runs in the training process."""
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


# ---------------------------------------------------------------- case table
@dataclass
class CaseTable:
    """One fixed (stage, kind) case table.

    ``stage`` is the gate stage the table belongs to (the flat retention test
    keeps the stage it retains for); ``measurement_stage`` is the physical stage
    the environment actually runs (``F0`` for flat retention).
    """

    stage: str
    kind: str
    seed: int
    arrays: dict
    sha256: str
    episodes: int = EPISODES_PER_SPEED
    measurement_stage: Optional[str] = None
    measurement: str = 'stair'
    round: Optional[int] = None
    cases: list = field(default_factory=list)

    @property
    def speeds(self):
        return [float(value) for value in self.arrays]

    def initial_poses(self, speed):
        entry = self.arrays[f'{float(speed):.1f}']
        return [list(pose) for pose in zip(entry['x'], entry['y'], entry['yaw'])]

    def case_ids(self, speed):
        return [f'{self.stage}:{self.kind}:{float(speed):.1f}:{index:03d}'
                for index in range(len(self.arrays[f'{float(speed):.1f}']['x']))]

    def to_json(self):
        """Case-table identity: the SDS numbers plus the r2 namespace.

        The x/y/yaw draws are identical to the r1 rule (SDS 6.5, Q003-r2 decision
        3.6) but the JSON additionally binds the r2 task/recipe/schema/architecture,
        both r2 protocol ids and the r2 route contract, so the resulting
        ``case_table_sha256`` is an r2-only identity: an r1 table can never be
        relabelled as r2 evidence.
        """
        return dict(task=TASK, sds_revision=SDS_REVISION, code_version=CODE_VERSION,
                    recipe_version=RECIPE_VERSION, checkpoint_schema=SCHEMA_VERSION,
                    architecture=ARCHITECTURE, protocol_id=self.protocol_id,
                    stair_protocol_id=PROTOCOL_ID, flat_protocol_id=FLAT_PROTOCOL_ID,
                    route_state=route_contract_record(),
                    stage=self.stage, kind=self.kind,
                    seed=int(self.seed), episodes=int(self.episodes),
                    measurement_stage=self.measurement_stage or self.stage,
                    measurement=self.measurement, speed_order=list(SPEED_ORDER),
                    arrays=deepcopy(self.arrays), cases=deepcopy(self.cases),
                    round=self.round)

    @property
    def protocol_id(self):
        return FLAT_PROTOCOL_ID if self.measurement == 'flat' else PROTOCOL_ID


def _seed_for(kind, seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError('training seed must be a positive integer')
    if kind == 'validation':
        return VALIDATION_SEED_OFFSET + seed
    if kind == 'perturbed':
        return PERTURBED_SEED_OFFSET + seed
    if kind == 'diagnostic':
        return DIAGNOSTIC_SEED_OFFSET + seed
    if kind == 'test':
        return FINAL_TEST_SEED
    if kind == 'flat_retention':
        return VALIDATION_SEED_OFFSET + seed
    raise ValueError(f'unknown case kind: {kind!r}')


def make_cases(stage, kind, seed, *, round=None):
    """Build the frozen case table for one (stage, kind).

    Speed order is ``(0.2, 0.4, 0.6)``; inside each speed the local NumPy
    generator is consumed as x, then y, then yaw.  F0/P full gates and flat
    retention use ``x~U[-1.1,-0.9], y~U[-0.05,0.05], yaw~U[-5,5]deg``; formal
    S1..S4 keep the fixed ``[-1,0,0]``; S5 and the final test use the full
    ``x+-0.2, y+-0.1, yaw+-10deg`` range; the diagnostic table uses fixed
    ``[-1,0,0]``.  The additional S1 perturbation uses ``+-0.1/+-0.05/+-5deg``.
    """
    if stage not in STAGE_BY_ID:
        raise ValueError(f'unknown v0.5 stage: {stage!r}')
    if kind not in KINDS:
        raise ValueError(f'case kind must be one of {KINDS}')
    if kind == 'test' and stage != 'S5':
        raise ValueError('the final test is S5 only')
    if kind == 'perturbed' and stage != 'S1':
        raise ValueError('the lightweight perturbation belongs to the second S1 gate')
    table_seed = _seed_for(kind, seed)
    if kind == 'test':
        episodes = EPISODES_PER_SPEED_FINAL
    elif kind == 'diagnostic':
        episodes = DIAGNOSTIC_EPISODES
    else:
        episodes = EPISODES_PER_SPEED
    measurement_stage = 'F0' if kind == 'flat_retention' else stage
    measurement = 'flat' if measurement_stage == 'F0' else 'stair'
    stage_spec = STAGE_BY_ID[measurement_stage]
    full_initial = measurement == 'flat' or stage_spec.id.startswith('P')
    final = kind == 'test' or (kind == 'validation' and stage_spec.id == 'S5')
    rng = np.random.default_rng(table_seed)
    arrays = {}
    cases = []
    for speed in SPEED_ORDER:
        x_values, y_values, yaw_values = [], [], []
        for index in range(episodes):
            if kind == 'diagnostic':
                pose = [-1., 0., 0.]
            elif kind == 'perturbed':
                pose = [-float(rng.uniform(.9, 1.1)),
                        float(rng.uniform(-.05, .05)),
                        float(rng.uniform(-math.pi / 36, math.pi / 36))]
            elif final and measurement == 'stair':
                pose = [-float(rng.uniform(.8, 1.2)),
                        float(rng.uniform(-.1, .1)),
                        float(rng.uniform(-math.pi / 18, math.pi / 18))]
            elif full_initial or kind == 'flat_retention':
                pose = [-float(rng.uniform(.9, 1.1)),
                        float(rng.uniform(-.05, .05)),
                        float(rng.uniform(-math.pi / 36, math.pi / 36))]
            else:
                pose = [-1., 0., 0.]
            x_values.append(pose[0])
            y_values.append(pose[1])
            yaw_values.append(pose[2])
            cases.append(dict(
                case_id=f'{stage}:{kind}:{float(speed):.1f}:{index:03d}',
                vx=float(speed), episode_index=index, initial_pose=list(pose),
                measurement_stage=measurement_stage))
        arrays[f'{float(speed):.1f}'] = dict(x=x_values, y=y_values, yaw=yaw_values)
    table = CaseTable(stage=stage, kind=kind, seed=int(table_seed), arrays=arrays,
                      sha256='', episodes=episodes, measurement_stage=measurement_stage,
                      measurement=measurement, round=round, cases=cases)
    table.sha256 = case_table_sha(table)
    _check_table(table)
    return table


def case_table_sha(table):
    data = table.to_json()
    return hashlib.sha256(_spec.canonical_json(jsonable(data)).encode('utf-8')).hexdigest()


def _expected_episodes(kind):
    if kind == 'test':
        return EPISODES_PER_SPEED_FINAL
    if kind == 'diagnostic':
        return DIAGNOSTIC_EPISODES
    return EPISODES_PER_SPEED


def _check_table(table):
    if table.sha256 != case_table_sha(table):
        raise ValueError('case table SHA does not match its content')
    if table.stage not in STAGE_BY_ID or table.kind not in KINDS:
        raise ValueError('case table identity is unknown')
    if int(table.episodes) != _expected_episodes(table.kind):
        raise ValueError(f'{table.kind} case tables hold {_expected_episodes(table.kind)} episodes per speed')
    if sorted(table.speeds) != [float(value) for value in SPEEDS]:
        raise ValueError('case table must contain exactly the three protocol speeds')
    for speed in table.speeds:
        entry = table.arrays[f'{speed:.1f}']
        if len(entry['x']) != table.episodes or len(entry['y']) != table.episodes or len(entry['yaw']) != table.episodes:
            raise ValueError('every case table speed must hold the frozen episode count')
        for key in ('x', 'y', 'yaw'):
            if not np.isfinite(np.asarray(entry[key], dtype=np.float64)).all():
                raise ValueError('case table holds a nonfinite initial pose')
    if table.kind == 'test' and table.stage != 'S5':
        raise ValueError('final test table must be S5')
    if table.kind == 'perturbed' and table.stage != 'S1':
        raise ValueError('the extra perturbation table must be S1')
    return True


def table_from_json(value, *, expected_sha=None):
    """Rebuild and re-validate a :class:`CaseTable` loaded from JSON.

    The SHA is always recomputed from the retained content; a stored value is
    never trusted.  ``expected_sha`` optionally binds it to the manifest SHA.
    """
    if not isinstance(value, dict):
        raise ValueError('missing case table')
    table = CaseTable(
        stage=value['stage'], kind=value['kind'], seed=int(value['seed']),
        arrays=deepcopy(value['arrays']), sha256='',
        episodes=int(value.get('episodes', EPISODES_PER_SPEED)),
        measurement_stage=value.get('measurement_stage'),
        measurement=value.get('measurement', 'stair'),
        round=value.get('round'))
    table.cases = [dict(case) for case in value.get('cases', [])]
    table.sha256 = case_table_sha(table)
    _check_table(table)
    if expected_sha is not None and table.sha256 != expected_sha:
        raise ValueError('retained case table SHA does not match its content')
    return table


# -------------------------------------------------------------------- gates
def _case_pass(kind, flat, climb_success, failure):
    """SDS 6.1: one case's pass flag for its measurement kind."""
    if kind == 'flat' or flat:
        return bool(flat)
    return bool(climb_success) and not bool(failure)


def gate_threshold(table):
    """Case-pass count required per speed, or ``None`` for a diagnostic table."""
    if table.kind == 'diagnostic':
        return None
    return CASE_PASS_REQUIRED_FINAL if table.kind == 'test' else CASE_PASS_REQUIRED


def summarize(records, table):
    """Per-speed counts and pass booleans from recomputed case records."""
    expected = {case['case_id'] for case in table.cases}
    if len(records) != len(expected) or {row['case_id'] for row in records} != expected:
        raise ValueError('missing or duplicated evaluation cases')
    per_speed = []
    for speed in table.speeds:
        rows = [row for row in records if abs(row['vx'] - speed) < 1e-9]
        if len(rows) != table.episodes:
            raise ValueError('incomplete speed denominator')
        # Filled (padded) environment slots are marked active=False and never reach
        # this list; they are already excluded from the case denominator.
        count = len(rows)
        case_pass = sum(bool(row['case_pass']) for row in rows)
        speed_pass = sum(bool(row['speed_pass']) for row in rows)
        mean_vx = [row['mean_vx'] for row in rows if row['mean_vx'] is not None and math.isfinite(row['mean_vx'])]
        per_speed.append(dict(
            c=float(speed), total=count, case_pass_count=case_pass,
            speed_pass_count=speed_pass,
            case_pass_rate=case_pass / count, speed_valid_count=sum(bool(row['speed_valid']) for row in rows),
            climb_success_count=sum(bool(row['climb_success']) for row in rows),
            failure_count=sum(bool(row['failure']) for row in rows),
            mean_vx=float(np.mean(mean_vx)) if mean_vx else None,
            mean_relative_error=(float(np.mean([row['mean_relative_error'] for row in rows
                                                if row['mean_relative_error'] is not None
                                                and math.isfinite(row['mean_relative_error'])]))
                                 if any(row['mean_relative_error'] is not None
                                        and math.isfinite(row['mean_relative_error']) for row in rows) else None),
            smooth_mae_relative=(float(np.mean([row['smooth_mae_relative'] for row in rows
                                                if row['smooth_mae_relative'] is not None]))
                                 if any(row['smooth_mae_relative'] is not None for row in rows) else None),
            longest_deviation=(max((row['longest_deviation'] for row in rows
                                    if row['longest_deviation'] is not None), default=None)),
            failure_reasons=dict(Counter(row['reason'] for row in rows)),
            case_pass_required=gate_threshold(table),
            passed=(None if gate_threshold(table) is None
                    else case_pass >= gate_threshold(table)),
            records=rows))
    required = gate_threshold(table)
    gate_eligible = table.kind in STAIR_KINDS
    gate_pass = bool(gate_eligible and required is not None
                     and all(entry['passed'] for entry in per_speed))
    return dict(per_speed=per_speed, required=required,
                episodes=table.episodes, gate_pass=gate_pass,
                gate_eligible=gate_eligible,
                retention_only=table.kind == 'flat_retention',
                kind=table.kind, stage=table.stage,
                measurement=table.measurement,
                measurement_stage=table.measurement_stage or table.stage,
                case_table_sha256=table.sha256, speeds=list(table.speeds))


def _check_json_safe(value):
    """Fail closed on any residual NaN/inf before a summary signs a gate."""
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f'evaluation summary is not JSON-safe: {error}') from error
    return True


def gate_pass_counts(result) -> dict:
    """Cheap gate projection of an :class:`EvalResult` for orchestration.

    Returns per-speed ``case_pass_count``/``total``/``speed_pass_count`` plus
    ``gate_pass`` and ``speed_gate_pass``.  No streak logic lives here.
    """
    if not isinstance(result, EvalResult):
        raise TypeError('gate_pass_counts needs an EvalResult')
    summary = result.summary
    if not isinstance(summary, dict) or 'per_speed' not in summary:
        raise ValueError('evaluation summary has no per_speed block')
    _check_json_safe(summary)
    speeds = []
    for entry in summary['per_speed']:
        speeds.append(dict(c=float(entry['c']), case_pass_count=int(entry['case_pass_count']),
                           total=int(entry['total']),
                           speed_pass_count=int(entry['speed_pass_count']),
                           required=(None if entry['case_pass_required'] is None
                                     else int(entry['case_pass_required'])),
                           passed=(None if entry['passed'] is None
                                   else bool(entry['passed']))))
    return dict(kind=summary['kind'], stage=summary['stage'],
                mode=summary.get('mode'), probe=bool(summary.get('probe', False)),
                measurement_stage=summary['measurement_stage'],
                required=(None if summary['required'] is None else int(summary['required'])),
                gate_eligible=bool(summary.get('gate_eligible', False)),
                speeds=speeds,
                gate_pass=bool(summary.get('gate_eligible', False)
                               and all(item['passed'] for item in speeds)),
                speed_gate_pass=(all(item['speed_pass_count'] >= item['required']
                                     for item in speeds)
                                 if summary['measurement'] == 'flat'
                                 and summary['required'] is not None else None),
                evidence_verified=bool(summary.get('evidence_verified', False)),
                key=result.key, checkpoint_sha=result.key[5], case_sha=result.key[6])


# ----------------------------------------------------------------- evidence
@dataclass
class EvalResult:
    """Committed evaluation result handed to runner/recovery (no ledger writes)."""

    key: tuple
    summary: dict
    raw_paths: dict
    evidence_sha256: dict


def evaluation_key(run_id, mode, stage, global_update, kind, checkpoint_sha, case_sha):
    """Unique evaluation task key (SDS 7.3)."""
    for name, value in (('run_id', run_id), ('mode', mode), ('stage', stage),
                        ('global_update', global_update), ('kind', kind)):
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise TypeError(f'{name} must be a string or integer')
    for name, value in (('checkpoint_sha', checkpoint_sha), ('case_sha', case_sha)):
        if not isinstance(value, str) or len(value) != HEX:
            raise ValueError(f'{name} must be a 64-character SHA256 string')
    return (str(run_id), str(mode), str(stage), int(global_update), str(kind),
            str(checkpoint_sha), str(case_sha))


def training_episode_limit(tile_n, tile_h, executed_c):
    """SDS 5.1 episode cap for one *slot's* real N/h/c (single source: spec).

    This is the training timeout, without any protocol margin.  The value is
    :func:`v05.spec.episode_step_limit` verbatim; evaluation must never extend or
    shorten it.
    """
    return int(_spec.episode_step_limit(int(tile_n), float(tile_h), float(executed_c)))


def evaluation_step_limit(measurement_stage, speed):
    """Evaluation observation bound: F0 window vs the real stair episode cap.

    These are deliberately different concepts:

    * ``F0`` (flat, ``N == 0``) is observed for exactly ``F0_OBSERVE_STEPS``
      (500 control steps = 10 s, SDS 6.2).  Its *episode* cap stays
      ``spec.episode_step_limit(0, 0, c) == 1500``; the 500-step observation end
      is an evaluation window, never a training timeout.
    * ``P``/``S`` stages run to their own real-tile timeout, which is
      :func:`training_episode_limit` with no extra margin.
    """
    stage = STAGE_BY_ID[measurement_stage]
    if stage.N == 0:
        return int(F0_OBSERVE_STEPS)
    return training_episode_limit(stage.N, stage.h, speed)


def _windows(table):
    return {speed: evaluation_step_limit(table.measurement_stage or table.stage, speed)
            for speed in table.speeds}


# --------------------------------------------------------------- execution
def _identity(value):
    """No-op fallback for test doubles that supply no observation validator."""
    return value


def _runtime_components():
    """The r2 environment class (imported lazily: no simulator on pure paths)."""
    from .env import V05R2Env, validate_observations
    return dict(env_class=V05R2Env, validate_observations=validate_observations)


def _route_stats(raw):
    """Independent per-case ``route_error`` evidence (SDS 4.8).

    ``route_error`` is retained read-only: it never changes ``case_pass``, the
    speed metrics or a gate count.  The first component is re-derived here from
    the retained ``local_y`` and the second/third from the retained ``yaw``, so a
    corrupted or r1-shaped (missing) route record makes the case evidence invalid
    instead of silently degrading to zeros.
    """
    route = raw.get('route_error')
    if route is None:
        raise ValueError('raw evidence is missing route_error (SDS 4.8)')
    route = np.asarray(route, dtype=np.float32)
    steps = len(raw['step'])
    if route.shape != (steps, 3):
        raise ValueError(f'raw route_error must be [steps,3], got {route.shape}')
    if steps and not np.all(np.isfinite(route)):
        raise ValueError('raw route_error holds a nonfinite value')
    if 'local_y' not in raw or 'yaw' not in raw:
        raise ValueError('raw route evidence needs local_y and yaw')
    local_y = np.asarray(raw['local_y'], dtype=np.float64)
    yaw = np.asarray(raw['yaw'], dtype=np.float64)
    if len(local_y) != steps or len(yaw) != steps:
        raise ValueError('raw route evidence has mismatched lengths')
    expected_lateral = np.clip(local_y / ROUTE_LATERAL_SCALE, -ROUTE_CLIP, ROUTE_CLIP)
    if steps and float(np.max(np.abs(expected_lateral - route[:, 0].astype(np.float64)))) > 1e-6:
        raise ValueError('retained route_error lateral component does not match local_y')
    if steps and float(np.max(np.abs(np.sin(yaw) - route[:, 1].astype(np.float64)))) > 1e-5:
        raise ValueError('retained route_error sine component does not match yaw')
    if steps and float(np.max(np.abs(np.cos(yaw) - route[:, 2].astype(np.float64)))) > 1e-5:
        raise ValueError('retained route_error cosine component does not match yaw')
    lateral = np.abs(local_y)
    heading = np.abs(yaw)
    saturated = lateral >= ROUTE_CLIP * ROUTE_LATERAL_SCALE
    return dict(
        route_samples=int(steps),
        route_abs_y_mean=float(np.mean(lateral)) if steps else None,
        route_abs_y_max=float(np.max(lateral)) if steps else None,
        route_normalized_abs_y_max=float(np.max(np.abs(route[:, 0]))) if steps else None,
        route_saturation_rate=float(np.mean(saturated)) if steps else None,
        route_heading_deg_mean=(float(np.mean(heading)) * 180. / math.pi) if steps else None,
        route_heading_deg_max=(float(np.max(heading)) * 180. / math.pi) if steps else None,
        route_nonfinite_count=0,
        route_final=([float(value) for value in route[-1]] if steps else None),
        route_yaw_axis=ROUTE_YAW_AXIS)


def _aggregate_route(records):
    """Read-only ``route/*`` aggregates over per-case route evidence.

    Field names are the bare SDS 4.8 aggregates; the report block is called
    ``route`` and telemetry prefixes it with its own ``route/`` namespace (so the
    W&B key is exactly ``route/abs_y_mean`` and never ``route/abs_y_mean`` nested
    twice).  The block is never read by ``summarize`` or a gate.
    """
    rows = [row for row in records if row.get('route_samples')]
    if not rows:
        return {'samples': 0, 'cases': 0, 'abs_y_mean': None, 'abs_y_max': None,
                'normalized_abs_y_max': None, 'saturation_rate': None,
                'heading_deg_mean': None, 'heading_deg_max': None,
                'nonfinite_count': 0}
    total = sum(int(row['route_samples']) for row in rows)
    mean_abs_y = sum(float(row['route_abs_y_mean']) * int(row['route_samples'])
                     for row in rows) / total
    saturated = sum(float(row['route_saturation_rate']) * int(row['route_samples'])
                    for row in rows)
    mean_heading = sum(float(row['route_heading_deg_mean']) * int(row['route_samples'])
                       for row in rows) / total
    return dict(
        samples=int(total), cases=int(len(rows)),
        abs_y_mean=mean_abs_y,
        abs_y_max=max(float(row['route_abs_y_max']) for row in rows),
        normalized_abs_y_max=max(float(row['route_normalized_abs_y_max']) for row in rows),
        saturation_rate=saturated / total,
        heading_deg_mean=mean_heading,
        heading_deg_max=max(float(row['route_heading_deg_max']) for row in rows),
        nonfinite_count=int(sum(int(row['route_nonfinite_count']) for row in rows)),
        yaw_axis=ROUTE_YAW_AXIS,
        lateral_scale=ROUTE_LATERAL_SCALE,
        clip=ROUTE_CLIP,
        units='radians')


def _video_writer(path):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=10)


def overlay_frame(frame, diagnostics):
    """Show executed physical values and pre-reset online previews only."""
    from PIL import Image, ImageDraw
    value = jsonable(diagnostics)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(image)
    metrics = value.get('speed_metrics') or {}
    lines = [f"stage={value.get('stage')} c={value.get('command_executed', [0])[0]:.7g} "
             f"vx={value.get('linear_velocity_body', [0, 0, 0])[0]:.7g} "
             f"step={value.get('step')}",
             f"window={metrics.get('window_kind')} samples={metrics.get('samples')} "
             f"speed_pass={metrics.get('speed_pass')}",
             f"failure={value.get('failure')} success={value.get('success')} "
             f"reason={value.get('reason')}"]
    for index, line in enumerate(lines):
        y = 7 + 17 * index
        draw.rectangle((3, y - 2, image.width - 3, y + 14), fill=(0, 0, 0))
        draw.text((5, y), line, fill=(255, 255, 255))
    return np.asarray(image)


def _record_for_case(case, raw, *, failure, climb_success, reason, terminal, metrics, identity):
    flat = identity['measurement'] == 'flat'
    if flat:
        # SDS 6.1: F0 case_pass = full 500 control steps AND no real failure AND
        # the whole-episode lateral/heading constraint AND the speed gate.
        complete = len(raw['step']) >= F0_OBSERVE_STEPS
        lateral_ok = metrics.get('lateral_heading_pass') is not False
        case_pass = bool(metrics['passed'] and not failure and lateral_ok and complete)
    else:
        case_pass = bool(climb_success) and not bool(failure)
    if flat:
        climb_success_out = False
        joint_success = None
    else:
        climb_success_out = bool(climb_success)
        joint_success = bool(case_pass and metrics['passed'])
    return dict(
        case_id=case['case_id'], vx=float(case['vx']), episode_index=int(case['episode_index']),
        initial_pose=list(case['initial_pose']), measurement_stage=identity['measurement_stage'],
        window_kind=metrics['window_kind'], samples=int(metrics['samples']),
        mean_vx=metrics['mean_vx'], mean_relative_error=metrics['mean_relative_error'],
        smooth_mae_relative=metrics['smooth_mae_relative'],
        longest_deviation=metrics['longest_deviation'],
        start_step=metrics['start_step'], end_step=metrics['end_step'],
        coverage_complete=bool(metrics['coverage_complete']),
        lateral_heading_pass=metrics.get('lateral_heading_pass'),
        speed_valid=bool(metrics['valid']), speed_pass=bool(metrics['passed']),
        invalid_reasons=list(metrics['invalid_reasons']),
        failure=bool(failure), reason=reason, climb_success=climb_success_out,
        joint_success=joint_success, case_pass=bool(case_pass),
        # SDS 4.8: route evidence is recorded, never a pass criterion.
        **_route_stats(raw),
        # SDS 6.3 per-case diagnostic evidence: an active, not-yet-arrived control
        # sample denominator and the stall_on count inside it.  A zero denominator
        # is reported as invalid evidence, never as a 0.0 stall fraction.
        active_samples=_optional_count(terminal.get('active_samples')),
        stall_samples=_optional_count(terminal.get('stall_active_samples',
                                                   terminal.get('stall_samples'))),
        first_step_acquired=(None if terminal.get('first_step_acquired') is None
                             else bool(terminal.get('first_step_acquired'))),
        stall_fraction=_stall_fraction(terminal),
        diagnostic_denominator_valid=bool(_optional_count(terminal.get('active_samples')) or 0),
        terminal=jsonable(terminal),
    )


def _optional_count(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _stall_fraction(terminal):
    active = _optional_count(terminal.get('active_samples'))
    stall = _optional_count(terminal.get('stall_active_samples',
                                         terminal.get('stall_samples')))
    if not active or stall is None:
        return None
    return float(stall) / float(active)


def _diagnostic_cases(records):
    """Per-case SDS 6.3 evidence for one diagnostic evaluation (15 cases)."""
    cases = []
    for row in sorted(records, key=lambda item: (item['vx'], item['episode_index'])):
        cases.append(dict(c=float(row['vx']), case_id=row['case_id'],
                          episode_index=int(row['episode_index']),
                          active_samples=row['active_samples'],
                          stall_samples=row['stall_samples'],
                          first_step_acquired=row['first_step_acquired'],
                          stall_fraction=row['stall_fraction'],
                          denominator_valid=bool(row['diagnostic_denominator_valid']),
                          climb_success=bool(row['climb_success']),
                          failure=bool(row['failure']), reason=row['reason'],
                          terminal_step=_terminal_step(row)))
    return cases


def _terminal_step(row):
    terminal = row.get('terminal')
    if isinstance(terminal, dict) and isinstance(terminal.get('terminal_step'), int):
        return int(terminal['terminal_step'])
    return None


def _diagnostic_aggregate(cases):
    valid = [case for case in cases if case['denominator_valid']]
    above = [case for case in valid if case['stall_fraction'] is not None
             and case['stall_fraction'] >= DIAGNOSTIC_STALL_FRACTION - 1e-12]
    first = [case for case in valid if case['first_step_acquired']]
    return dict(case_count=len(cases), valid_case_count=len(valid),
                zero_denominator_count=len(cases) - len(valid),
                stall_fraction_threshold=DIAGNOSTIC_STALL_FRACTION,
                stall_fraction_ge_threshold_count=len(above),
                first_step_acquired_count=len(first),
                no_first_step_count=len(valid) - len(first),
                collapse_case_requirement=DIAGNOSTIC_MINUTES_CASES,
                evidence_valid=bool(len(valid) == len(cases) and len(cases) == 15),
                note=('informational counts only; the collapse_hit/behavior_collapse '
                      'decision and streak belong to orchestration'))


def diagnostic_collapse_counts(result):
    """SDS 6.3 counts for orchestration; it never decides collapse itself."""
    if not isinstance(result, EvalResult):
        raise TypeError('diagnostic_collapse_counts needs an EvalResult')
    summary = result.summary
    if summary.get('kind') != 'diagnostic':
        raise ValueError('collapse counts require a diagnostic evaluation result')
    cases = summary.get('diagnostic_cases')
    if not isinstance(cases, list):
        raise ValueError('diagnostic result is missing its per-case evidence')
    aggregate = _diagnostic_aggregate(cases)
    if aggregate != summary.get('diagnostic_aggregate'):
        raise ValueError('diagnostic aggregate disagrees with its per-case evidence')
    return dict(aggregate, per_case=cases)


def evaluate_checkpoint(model, cases, *, device, num_envs, stage, kind, run_id,
                        global_update, checkpoint_sha, output_dir,
                        record_video=False, mode='formal', seed=42,
                        monitor=None):
    """Run one frozen evaluation and return a committable :class:`EvalResult`.

    Pure execution: no ledger, no training state, no training RNG, no model
    parameter update.  ``output_dir`` is created exactly once; a rerun with an
    identical key re-verifies and returns the retained evidence (idempotent),
    any other collision raises.  Padding slots of the last batch keep the
    configured physical environment count and are removed with an active mask.

    ``mode`` is one of ``EVALUATION_MODES``:

    * ``pilot``/``formal`` -- the ordinary protocol modes; gates come from the
      summary counts exactly as before.
    * ``probe`` -- the SDS 7.4 resource/lifecycle probe (W7): it runs the same
      evaluation path but is **never gate-eligible** and must never be written to
      a pilot/formal ledger or checkpoint (ledger ownership stays with
      runner/recovery).  Its key carries ``mode='probe'``, so it cannot collide
      with a pilot/formal result for the same checkpoint and cases.  Probe kinds
      are restricted to ``PROBE_KINDS`` (``validation``, ``flat_retention``,
      ``diagnostic``): ``test`` and ``perturbed`` are gate-protocol kinds and
      raise instead of being silently measured.
    """
    import torch

    if not isinstance(cases, CaseTable):
        raise TypeError('cases must be a CaseTable')
    _check_table(cases)
    check_evaluation_mode(mode, kind)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError('training seed must be a positive integer')
    if cases.stage != stage or cases.kind != kind:
        raise ValueError('case table does not belong to this stage/kind')
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    if device is None:
        raise ValueError('device is required')
    case_sha = cases.sha256
    key = evaluation_key(run_id, mode, stage, global_update, kind, checkpoint_sha, case_sha)
    out = Path(output_dir).resolve()
    if out.exists():
        retained = _load_retained(out, key)
        if retained is not None:
            return retained
        raise FileExistsError(f'evaluation output already exists with another key: {out}')
    ensure_disk_space(out.parent)
    out.mkdir(parents=True, exist_ok=False)
    identity = dict(
        task=TASK, sds_revision=SDS_REVISION, code_version=CODE_VERSION,
        recipe_version=RECIPE_VERSION, checkpoint_schema=SCHEMA_VERSION,
        protocol_id=cases.protocol_id, run_id=str(run_id), mode=str(mode),
        seed=int(seed), eval_seed=int(cases.seed), stage=stage, kind=kind,
        measurement=cases.measurement,
        measurement_stage=cases.measurement_stage or stage,
        checkpoint_sha256=str(checkpoint_sha), case_table_sha256=case_sha,
        global_update=int(global_update), num_envs=int(num_envs),
        device=str(device), record_video=bool(record_video),
        architecture=ARCHITECTURE, source_sha256=source_sha256(),
        route_state=route_contract_record(),
        route_keys=list(ROUTE_KEYS),
        speed_order=list(SPEED_ORDER), raw_schema=RAW_SCHEMA,
        window_kinds={f'{speed:.1f}': window_kind_for_stage(cases.measurement_stage or stage)
                      for speed in cases.speeds},
        protocol_step_limits=_windows(cases),
        case_table=jsonable(cases.to_json()),
    )
    atomic_json(out / 'manifest.json', identity)
    write_bytes = (out / 'manifest.json').stat().st_size
    nonfinite = {'nonfinite': 0}
    started = time.monotonic()
    records = []
    raw_paths = {'npz': [], 'episodes': str(out / 'episodes.jsonl'),
                 'manifest': str(out / 'manifest.json'),
                 'trace': str(out / 'steps_env0.jsonl')}
    component = _runtime_components()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    limits = _windows(cases)
    with preserved_rng(), (out / 'episodes.jsonl').open('w') as episodes_stream, \
            (out / 'steps_env0.jsonl').open('w') as trace_stream:
        for speed in cases.speeds:
            poses = cases.initial_poses(speed)
            ids = cases.case_ids(speed)
            for offset in range(0, len(poses), num_envs):
                batch_poses = poses[offset:offset + num_envs]
                batch_ids = ids[offset:offset + num_envs]
                active = len(batch_ids)
                padding = num_envs - active
                # Keep the configured physical cost: every slot is simulated and
                # only the active-case mask is removed from the denominators.
                padded_poses = batch_poses + [batch_poses[-1]] * padding
                env = component['env_class'](
                    num_envs=num_envs, device=device, seed=cases.seed,
                    stage=cases.measurement_stage or stage, stage_iteration=0,
                    evaluation=True, fixed_vx=float(speed),
                    initial_cases=padded_poses, render_mode=None,
                    parking_diagnostic=False)
                env.checkpoint_identity = dict(path=None, sha256=str(checkpoint_sha))
                columns = {key: [[] for _ in range(active)]
                           for key in FLOAT32_KEYS + INT64_KEYS + ROUTE_KEYS}
                terminals = {}
                try:
                    obs, _ = env.reset()
                    component.get('validate_observations', _identity)(obs)
                    maximum = limits[speed]
                    for step in range(maximum):
                        with torch.inference_mode():
                            action = model(obs)
                        obs, _, _, _, info = env.step(action)
                        component.get('validate_observations', _identity)(obs)
                        if monitor is not None:
                            monitor.poll()
                        diag = info['diagnostics']
                        for row in range(active):
                            if row in terminals:
                                continue
                            columns['body_vx'][row].append(float(diag['body_vx'][row]))
                            columns['executed_c'][row].append(float(diag['command_c'][row]))
                            columns['local_x'][row].append(float(diag['local_position'][row][0]))
                            columns['local_y'][row].append(float(diag['local_position'][row][1]))
                            columns['yaw'][row].append(float(diag['heading'][row]))
                            columns['route_error'][row].append(
                                [float(value) for value in diag['route_error'][row]])
                            columns['step'][row].append(int(diag['control_step'][row]))
                        if active and 0 not in terminals:
                            diagnostics = jsonable(env.diagnostics_snapshot(0),
                                                   counter=nonfinite)
                            entry = jsonable(dict(identity, case_id=batch_ids[0],
                                                  batch_offset=offset, step=step + 1,
                                                  diagnostics=diagnostics),
                                             counter=nonfinite)
                            trace_stream.write(json.dumps(entry, allow_nan=False) + '\n')
                        for terminal in info['episodes']:
                            row = int(terminal['env_id'])
                            if row >= active or row in terminals:
                                continue
                            if int(terminal['terminal_step']) != int(columns['step'][row][-1]):
                                raise RuntimeError('terminal evidence is not the pre-reset sample')
                            terminals[row] = dict(terminal)
                        if len(terminals) == active:
                            break
                    else:
                        if cases.measurement == 'flat':
                            for row in range(active):
                                if row in terminals:
                                    continue
                                terminals[row] = dict(
                                    env_id=row, terminal_step=maximum,
                                    duration=maximum * .02,
                                    reason='observed_500_steps',
                                    climb_success=False, climb_event=False,
                                    climb_step=None, failure=False,
                                    timeout=False, truncated=False,
                                    # The observed-500-step terminal is synthesized
                                    # here, so the SDS 6.3 diagnostic denominators must
                                    # be copied from the last pre-reset diagnostics.
                                    active_samples=_environment_counter(
                                        env, diag, 'active_samples', row),
                                    stall_active_samples=_environment_counter(
                                        env, diag, 'stall_active_samples', row),
                                    first_step_acquired=_environment_flag(
                                        env, diag, 'first_step_acquired', row),
                                    initial_pose=padded_poses[row],
                                    vx=float(speed), stage=stage,
                                    N=int(env.tile_N[row]))
                        else:
                            raise RuntimeError('missing pre-reset terminal records')
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass
                npz_path = out / f'window_batch_{len(raw_paths["npz"]):04d}.npz'
                _write_batch_npz(npz_path, columns)
                raw_paths['npz'].append(str(npz_path))
                write_bytes += npz_path.stat().st_size
                for row in range(active):
                    raw = {key: np.asarray(columns[key][row],
                                           dtype=np.int64 if key in INT64_KEYS else np.float32)
                           for key in FLOAT32_KEYS + INT64_KEYS}
                    raw['route_error'] = np.asarray(columns['route_error'][row],
                                                    dtype=np.float32).reshape(-1, 3)
                    n = int(STAGE_BY_ID[cases.measurement_stage or stage].N)
                    kind_window = ('f0_time' if cases.measurement == 'flat'
                                   else ('p_position' if STAGE_BY_ID[cases.measurement_stage or stage].id.startswith('P')
                                         else 's_oracle'))
                    metrics = cpu_speed_oracle(raw, kind_window, float(speed), n)
                    terminal = terminals[row]
                    record = _record_for_case(
                        dict(case_id=batch_ids[row], vx=float(speed), episode_index=offset + row,
                             initial_pose=batch_poses[row]),
                        raw, failure=bool(terminal.get('failure', False)),
                        climb_success=bool(terminal.get('climb_success', False)),
                        reason=str(terminal.get('reason', 'running')), terminal=terminal,
                        metrics=metrics, identity=identity)
                    record['raw_file'] = str(npz_path)
                    record['raw_row'] = row
                    records.append(record)
                    episodes_stream.write(json.dumps(jsonable(record, counter=nonfinite),
                                                     allow_nan=False) + '\n')
                episodes_stream.flush()
                write_bytes += (out / 'episodes.jsonl').stat().st_size
    after = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if set(before) != set(after) or any(not torch.equal(before[name], after[name]) for name in before):
        raise RuntimeError('evaluation changed model/statistics tensors')
    summary = apply_mode_policy(summarize(records, cases), mode)
    diagnostic_cases = None
    diagnostic_aggregate = None
    if kind == 'diagnostic':
        diagnostic_cases = _diagnostic_cases(records)
        diagnostic_aggregate = _diagnostic_aggregate(diagnostic_cases)
        if len(diagnostic_cases) != 3 * DIAGNOSTIC_EPISODES:
            raise RuntimeError('a diagnostic evaluation must retain 15 per-case records')
        path = out / 'diagnostic_cases.json'
        atomic_json(path, jsonable(dict(identity, required=None,
                                        cases=diagnostic_cases,
                                        aggregate=diagnostic_aggregate), counter=nonfinite))
        raw_paths['diagnostic_cases'] = str(path)
        write_bytes += path.stat().st_size
    video_paths = {}
    if record_video:
        video_paths = _record_videos(model, cases, component, device, out)
        for entry in video_paths.values():
            write_bytes += Path(entry['path']).stat().st_size
    report = dict(identity, per_speed=jsonable([{key: value for key, value in entry.items()
                                                 if key != 'records'}
                                                for entry in summary['per_speed']]),
                  required=summary['required'], gate_pass=summary['gate_pass'],
                  gate_eligible=summary['gate_eligible'],
                  mode=summary['mode'], probe=summary['probe'],
                  counts_against_formal=summary['counts_against_formal'],
                  gate_note=summary['gate_note'],
                  retention_only=summary['retention_only'],
                  episodes=summary['episodes'],
                  diagnostic_cases=diagnostic_cases,
                  per_case=diagnostic_cases,
                  diagnostic=({} if diagnostic_cases is None else
                              dict(per_case=diagnostic_cases, cases=diagnostic_cases,
                                   aggregate=diagnostic_aggregate,
                                   path=raw_paths.get('diagnostic_cases'))),
                  diagnostic_aggregate=diagnostic_aggregate,
                  diagnostic_cases_path=raw_paths.get('diagnostic_cases'),
                  nonfinite_fields=int(nonfinite['nonfinite']),
                  route=_aggregate_route(records), route_keys=list(ROUTE_KEYS),
                  statistics_unchanged=True, evidence_verified=False,
                  raw_paths=jsonable(raw_paths), videos=jsonable(video_paths),
                  elapsed_s=time.monotonic() - started, write_bytes=write_bytes,
                  failure_in_denominator=True,
                  note=('speed_pass does not authorize a stair climb; F0 case_pass '
                        'includes the 500-step and lateral/heading constraints'))
    atomic_json(out / 'report.json', report)
    result = EvalResult(key=key, summary=report, raw_paths=raw_paths,
                        evidence_sha256={'report.json': file_sha256(out / 'report.json')})
    verified = verify_evaluation_evidence(result, expected_key=key)
    return verified


def _environment_counter(env, diagnostics, name, row):
    """SDS 6.3 denominator from the pre-reset environment counters.

    The three counters live on :class:`V05Env` (``active_samples``,
    ``stall_active_samples``, ``first_step_acquired``); the per-step diagnostics
    copy is the fallback.  Missing evidence returns ``None`` so the case is
    marked invalid instead of being given a fabricated zero.
    """
    tensor = getattr(env, name, None)
    if tensor is not None:
        return int(tensor[row])
    value = diagnostics.get(name)
    if value is None:
        return None
    return int(value[row])


def _environment_flag(env, diagnostics, name, row):
    tensor = getattr(env, name, None)
    if tensor is not None:
        return bool(tensor[row])
    value = diagnostics.get(name)
    if value is None:
        return None
    return bool(value[row])


def _diagnostic_value(diagnostics, key, row):
    """Per-environment diagnostic scalar, or ``None`` when it is not retained.

    A missing SDS 6.3 denominator makes that case's diagnostic evidence invalid;
    it never raises and never fabricates a zero-denominator ratio.
    """
    value = diagnostics.get(key)
    if value is None:
        return None
    return int(value[row])


def _write_batch_npz(path, columns):
    offsets = np.cumsum([0] + [len(columns['step'][row]) for row in range(len(columns['step']))],
                        dtype=np.int64)
    arrays = {}
    for key in FLOAT32_KEYS:
        arrays[key] = np.asarray([value for row in columns[key] for value in row], dtype=np.float32)
    for key in INT64_KEYS:
        arrays[key] = np.asarray([value for row in columns[key] for value in row], dtype=np.int64)
    #: SDS 4.8: ``route_error`` is retained as a [steps,3] float32 array.
    for key in ROUTE_KEYS:
        rows = [[list(value) for value in row] for row in columns[key]]
        arrays[key] = np.asarray([value for row in rows for value in row],
                                 dtype=np.float32).reshape(-1, 3)
    arrays['offsets'] = offsets
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_videos(model, cases, component, device, out):
    """Optional separate one-environment playback; never part of gate evidence."""
    import torch
    videos = {}
    identity = json.loads((out / 'manifest.json').read_text())
    for speed in cases.speeds:
        poses = cases.initial_poses(speed)
        env = component['env_class'](
            num_envs=1, device=device, seed=cases.seed,
            stage=cases.measurement_stage or cases.stage, stage_iteration=0,
            evaluation=True, fixed_vx=float(speed), initial_cases=[poses[0]],
            render_mode='rgb_array', parking_diagnostic=False)
        path = out / f'playback_vx_{speed:.1f}.mp4'
        writer = _video_writer(path)
        try:
            obs, _ = env.reset()
            frames = 0
            for step in range(evaluation_step_limit(cases.measurement_stage or cases.stage, speed)):
                with torch.inference_mode():
                    action = model(obs)
                obs, _, _, _, info = env.step(action)
                if step % 5 == 0 or info['episodes']:
                    writer.append_data(overlay_frame(env.render(), env.diagnostics_snapshot(0)))
                    frames += 1
                if info['episodes']:
                    break
            if frames == 0:
                raise RuntimeError('playback produced no frames')
        finally:
            writer.close()
            env.close()
        videos[f'{float(speed):.1f}'] = dict(path=str(path), sha256=file_sha256(path),
                                            frames=frames, fps=10, num_envs=1,
                                            case_id=cases.case_ids(speed)[0],
                                            identity_sha256=identity['checkpoint_sha256'])
    return videos


def _load_retained(out, key):
    """Return the retained result for an identical key, or ``None``."""
    manifest = out / 'manifest.json'
    if not manifest.is_file():
        return None
    identity = json.loads(manifest.read_text())
    retained = (identity['run_id'], identity['mode'], identity['stage'],
                int(identity['global_update']), identity['kind'],
                identity['checkpoint_sha256'], identity['case_table_sha256'])
    if retained != tuple(key):
        return None
    report_path = out / 'report.json'
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    result = EvalResult(key=key, summary=report,
                        raw_paths=dict(report['raw_paths']),
                        evidence_sha256={'report.json': file_sha256(report_path)})
    return verify_evaluation_evidence(result, expected_key=key)


# ---------------------------------------------------------------- verify
def verify_evaluation_evidence(result, *, expected_key=None):
    """Re-read every retained raw file and recompute the whole gate in CPU float64.

    Missing/cropped/reordered evidence raises; the returned summary is trusted and
    carries ``evidence_verified=True``.  Preview flags, cached counts or a caller
    supplied pass count are never trusted.
    """
    if not isinstance(result, EvalResult):
        raise TypeError('verify_evaluation_evidence needs an EvalResult')
    out = Path(result.raw_paths['manifest']).resolve().parent
    manifest = json.loads((out / 'manifest.json').read_text())
    key = (manifest['run_id'], manifest['mode'], manifest['stage'],
           int(manifest['global_update']), manifest['kind'],
           manifest['checkpoint_sha256'], manifest['case_table_sha256'])
    if expected_key is not None and tuple(expected_key) != key:
        raise ValueError('retained evidence belongs to another evaluation key')
    table = table_from_json(manifest['case_table'],
                            expected_sha=manifest['case_table_sha256'])
    rows = [json.loads(line) for line in Path(result.raw_paths['episodes']).read_text().splitlines()
            if line.strip()]
    if len(rows) != len(table.cases):
        raise ValueError('missing or duplicated retained case evidence')
    seen = set()
    trusted = []
    for row in rows:
        case_id = row['case_id']
        if case_id in seen:
            raise ValueError('duplicated retained case evidence')
        seen.add(case_id)
        case = next((item for item in table.cases if item['case_id'] == case_id), None)
        if case is None:
            raise ValueError('retained case is not part of the frozen table')
        if abs(row['vx'] - case['vx']) > 1e-9:
            raise ValueError('retained case speed differs from the frozen table')
        if not np.allclose(row['initial_pose'], case['initial_pose'], rtol=0, atol=1e-6):
            raise ValueError('retained initial pose differs from the frozen table')
        raw = _load_raw_row(Path(row['raw_file']), int(row['raw_row']), row['case_id'], out)
        n = int(STAGE_BY_ID[row['measurement_stage']].N)
        metrics = cpu_speed_oracle(raw, row['window_kind'], float(row['vx']), n)
        for name in ('samples', 'window_kind', 'coverage_complete', 'speed_valid', 'speed_pass'):
            actual = metrics['samples'] if name == 'samples' else metrics[name]
            if name == 'samples':
                if int(row[name]) != int(actual):
                    raise ValueError('retained sample count differs from recomputation')
            elif name in ('speed_valid', 'speed_pass'):
                if bool(row[name]) != bool(actual):
                    raise ValueError(f'retained {name} differs from recomputation')
        for name in ('mean_vx', 'mean_relative_error', 'smooth_mae_relative'):
            stored, recomputed = row[name], metrics[name]
            if stored is None or recomputed is None:
                if stored is not None or recomputed is not None:
                    raise ValueError(f'retained {name} availability differs from recomputation')
            elif abs(float(stored) - float(recomputed)) > 1e-8:
                raise ValueError(f'retained {name} differs from the CPU oracle')
        rebuilt = _record_for_case(
            case, raw, failure=bool(row['terminal']['failure']),
            climb_success=bool(row['terminal']['climb_success']),
            reason=str(row['terminal']['reason']), terminal=row['terminal'],
            metrics=metrics, identity=dict(manifest, measurement_stage=row['measurement_stage']))
        for name in ('case_pass', 'joint_success', 'speed_valid', 'speed_pass', 'failure'):
            if bool(rebuilt[name]) != bool(row[name]):
                raise ValueError(f'retained {name} differs from independent recomputation')
        for name in ('active_samples', 'stall_samples', 'stall_fraction',
                     'diagnostic_denominator_valid'):
            if rebuilt[name] != row[name]:
                raise ValueError(f'retained diagnostic field differs: {name}')
        # SDS 4.8: retained route evidence must match its own independent
        # recomputation from local_y/yaw; a tampered route record invalidates the
        # case evidence even though it can never change a pass count.
        for name in ('route_samples', 'route_abs_y_mean', 'route_abs_y_max',
                     'route_normalized_abs_y_max', 'route_saturation_rate',
                     'route_heading_deg_mean', 'route_heading_deg_max',
                     'route_nonfinite_count', 'route_final', 'route_yaw_axis'):
            if rebuilt[name] != row.get(name):
                raise ValueError(f'retained route evidence differs: {name}')
        trusted.append(rebuilt)
    summary = apply_mode_policy(summarize(trusted, table), manifest['mode'])
    diagnostic_cases = None
    diagnostic_aggregate = None
    if table.kind == 'diagnostic':
        expected = _diagnostic_cases(trusted)
        diagnostic_aggregate = _diagnostic_aggregate(expected)
        path = result.raw_paths.get('diagnostic_cases')
        if not path:
            raise ValueError('diagnostic evaluation is missing its per-case evidence file')
        retained = json.loads(Path(path).read_text())
        if (retained.get('checkpoint_sha256') != manifest['checkpoint_sha256']
                or retained.get('case_table_sha256') != manifest['case_table_sha256']
                or retained.get('cases') != jsonable(expected)
                or retained.get('aggregate') != jsonable(diagnostic_aggregate)):
            raise ValueError('retained diagnostic per-case evidence differs from recomputation')
        if len(expected) != 3 * DIAGNOSTIC_EPISODES:
            raise ValueError('diagnostic evidence must hold 15 per-case records')
        diagnostic_cases = jsonable(expected)
    report_path = out / 'report.json'
    prior = json.loads(report_path.read_text()) if report_path.is_file() else {}
    report = dict(manifest, per_speed=jsonable([{key: value for key, value in entry.items()
                                                if key != 'records'} for entry in summary['per_speed']]),
                  required=summary['required'], gate_pass=summary['gate_pass'],
                  gate_eligible=summary['gate_eligible'],
                  mode=summary['mode'], probe=summary['probe'],
                  counts_against_formal=summary['counts_against_formal'],
                  gate_note=summary['gate_note'],
                  retention_only=summary['retention_only'],
                  episodes=summary['episodes'], statistics_unchanged=True,
                  evidence_verified=True, failure_in_denominator=True,
                  diagnostic_cases=diagnostic_cases,
                  per_case=diagnostic_cases,
                  diagnostic=({} if diagnostic_cases is None else
                              dict(per_case=diagnostic_cases, cases=diagnostic_cases,
                                   aggregate=diagnostic_aggregate,
                                   path=result.raw_paths.get('diagnostic_cases'))),
                  diagnostic_aggregate=diagnostic_aggregate,
                  diagnostic_cases_path=result.raw_paths.get('diagnostic_cases'),
                  nonfinite_fields=int(prior.get('nonfinite_fields', 0)),
                  route=_aggregate_route(trusted), route_keys=list(ROUTE_KEYS),
                  videos=prior.get('videos', {}),
                  raw_paths=result.raw_paths,
                  evidence_sha256={Path(path).name: file_sha256(path)
                                   for path in [result.raw_paths['episodes'],
                                                result.raw_paths['manifest'],
                                                *result.raw_paths['npz'],
                                                *([result.raw_paths['diagnostic_cases']]
                                                  if result.raw_paths.get('diagnostic_cases')
                                                  else [])]})
    atomic_json(out / 'report.json', report)
    paths = dict(result.raw_paths)
    hashes = dict(report['evidence_sha256'])
    hashes['report.json'] = file_sha256(out / 'report.json')
    return EvalResult(key=key, summary=report, raw_paths=paths, evidence_sha256=hashes)


def _load_raw_row(path, row, case_id, out):
    path = Path(path).resolve()
    if not path.is_relative_to(out) or not path.is_file():
        raise ValueError('raw evidence file is missing or foreign')
    with np.load(path, allow_pickle=False) as storage:
        files = set(storage.files)
        if not {'body_vx', 'executed_c', 'local_x', 'step'} <= files:
            raise ValueError('raw NPZ is missing required arrays')
        extra = files - set(FLOAT32_KEYS) - set(INT64_KEYS) - set(ROUTE_KEYS) - {'offsets'}
        if extra:
            raise ValueError(f'raw NPZ holds unexpected arrays: {sorted(extra)}')
        missing_route = set(ROUTE_KEYS) - files
        if missing_route:
            raise ValueError(f'raw NPZ is missing route evidence: {sorted(missing_route)}')
        offsets = storage['offsets']
        if not 0 <= row < len(offsets) - 1:
            raise ValueError('raw row index is outside the archive')
        lo, hi = int(offsets[row]), int(offsets[row + 1])
        raw = {key: storage[key][lo:hi] for key in files if key != 'offsets'}
    if len(raw['step']) < 1:
        raise ValueError(f'empty raw evidence for case {case_id}')
    if np.any(raw['step'][1:] - raw['step'][:-1] != 1):
        raise ValueError('raw evidence is not consecutive control samples')
    for key in ('local_y', 'yaw'):
        if key not in raw:
            raise ValueError(f'raw evidence is missing {key}')
    if raw['route_error'].shape != (len(raw['step']), 3):
        raise ValueError('raw route_error is not a [steps,3] array')
    return raw
