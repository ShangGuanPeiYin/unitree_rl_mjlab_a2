"""S2 task adapter and evidence export (CPU-only; no simulator construction).

The physics environment is intentionally not created here: T2 delivers the
task/state adapter, the exact S2 terrain lookup and the per-step evidence
format consumed by :mod:`s2_metrics` and :mod:`s2_map_ablation`.  The future
physics runner will call :class:`S2EvidenceRecorder` once per control step.

Every recorded step carries the fields `s2_metrics` needs, plus the extra
attribution fields the six original H cases lack: per-foot XY/contact/support,
normal force, contact pairs, PD target vs measured joint state, map validity
and sample time, and the Q024 native sample id.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import s2_spec, s2_state, s2_terrain
from .cases import canonical_json, sha256

EVIDENCE_SCHEMA = 1
OBSERVATION_SHAPES = {'proprio': (45,), 'history': (5, 45),
                      'heightmap': (3, 40, 30), 'route_error': (3,)}


class ObservationContractError(ValueError):
    """Detailed Actor observation contract failure."""


def observation_diagnostics(observation):
    diagnostics = {}
    for key, expected in OBSERVATION_SHAPES.items():
        if key not in observation:
            diagnostics[key] = dict(present=False, expected_shape=list(expected))
            continue
        value = observation[key]
        array = np.asarray(to_numpy(value))
        finite = np.isfinite(array)
        diagnostics[key] = dict(
            present=True, dtype=str(array.dtype),
            device=str(getattr(value, 'device', 'cpu')),
            shape=list(array.shape), expected_shape=list(expected),
            shape_ok=tuple(array.shape) == expected,
            min=(float(np.min(array[finite])) if finite.any() else None),
            max=(float(np.max(array[finite])) if finite.any() else None),
            nan_count=int(np.isnan(array).sum()),
            inf_count=int(np.isinf(array).sum()),
            nonfinite_indices=np.argwhere(~finite).tolist()[:20])
        if key in ('proprio', 'route_error') and array.ndim == 1:
            # One reset frame: min/max at each dimension equal its value; keep
            # both lists for a stable schema when later aggregated over frames.
            diagnostics[key]['per_dim_values'] = [
                float(v) if np.isfinite(v) else None for v in array]
            diagnostics[key]['per_dim_min'] = [
                float(v) if np.isfinite(v) else None for v in array]
            diagnostics[key]['per_dim_max'] = [
                float(v) if np.isfinite(v) else None for v in array]
    return diagnostics


def validate_policy_observation(observation):
    diagnostics = observation_diagnostics(observation)
    failures = []
    for key, expected in OBSERVATION_SHAPES.items():
        info = diagnostics[key]
        if not info.get('present'):
            failures.append(f'{key}: missing')
        elif not info['shape_ok']:
            failures.append(f"{key}: shape {info['shape']} != {list(expected)}")
        elif info['nan_count'] or info['inf_count']:
            failures.append(f"{key}: nonfinite indices {info['nonfinite_indices']}")
        elif info['dtype'] != 'float32':
            failures.append(f"{key}: dtype {info['dtype']} != float32")
    if failures:
        raise ObservationContractError(
            'Actor observation contract failed: ' + '; '.join(failures)
            + ' | diagnostics=' + canonical_json(jsonable(diagnostics)))
    return diagnostics
PITCH_SIGN_CONVENTION = dict(
    positive='nose_up', negative='nose_down', zero='level',
    formula='pitch = -asin(2(wy-zx)) = asin((R_WB e_x)_z)',
    forward_axis='+X', world_up='+Z')

TRACE_FIELDS = (
    'step', 'route_x', 'base_xy', 'feet_xy', 'feet_route_x', 'feet_z', 'pitch',
    'sim_state_sha256', 'post_reset',
    'foot_on_platform', 'foot_contact', 'g_level', 'top_arrived', 'failure',
    'stop_request', 'terminal', 'c_exec', 'c_request', 'a_request', 'a_applied',
    'velocity_route', 'pitch_rate', 'foot_normal_force', 'contact_pairs',
    'pd_target', 'joint_position', 'joint_velocity', 'map_valid',
    'map_sample_time_s', 'native_sample_id')
REQUIRED_PER_STEP = set(TRACE_FIELDS)
SCALAR_FIELDS = {'step', 'route_x', 'pitch', 'g_level', 'c_exec', 'c_request',
                 'velocity_route', 'pitch_rate', 'map_sample_time_s'}
BOOLEAN_FIELDS = {'top_arrived', 'failure', 'stop_request', 'terminal', 'post_reset'}
VECTOR3_FIELDS = {'base_xy'}
FOOT_FIELDS = {'feet_xy': (4, 2), 'feet_route_x': (4,), 'feet_z': (4,),
               'foot_on_platform': (4,), 'foot_contact': (4,),
               'foot_normal_force': (4,)}
JOINT_FIELDS = {'a_request': (12,), 'a_applied': (12,), 'pd_target': (12,),
                'joint_position': (12,), 'joint_velocity': (12,)}
CONTACT_PAIR_FIELDS = ('foot', 'geom', 'normal_force', 'duration_s',
                       'position_xy', 'surface', 'valid')


def _as_float_array(value, name, shape):
    array = np.asarray(to_numpy(value), dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f'{name} shape {array.shape} != {shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    return array


def _as_bool_array(value, name, shape):
    array = np.asarray(to_numpy(value), dtype=bool)
    if array.shape != shape:
        raise ValueError(f'{name} shape {array.shape} != {shape}')
    return array


def evidence_contract():
    return dict(
        schema_version=EVIDENCE_SCHEMA,
        sds_sha256=s2_spec.SDS_SHA256,
        required_fields=list(TRACE_FIELDS),
        scalar_fields=sorted(SCALAR_FIELDS),
        boolean_fields=sorted(BOOLEAN_FIELDS),
        vector3_fields=sorted(VECTOR3_FIELDS),
        foot_fields=dict(FOOT_FIELDS),
        joint_fields=dict(JOINT_FIELDS),
        contact_pair_fields=list(CONTACT_PAIR_FIELDS),
        native_sample_id=list(s2_plan_native_fields()),
        note='step-end samples before reset; no reset/padding frame is exported',
    )


def s2_plan_native_fields():
    from .s2_map_ablation import NATIVE_SAMPLE_ID_FIELDS
    return NATIVE_SAMPLE_ID_FIELDS


@dataclass
class S2TaskAdapter:
    """Resolve one frozen case into task config, terrain and metric frame."""

    case: dict
    terrain: s2_terrain.S2TerrainSpec
    config: s2_state.S2TaskConfig
    platform: dict | None

    @classmethod
    def from_case(cls, case):
        spec = (s2_terrain.legacy_spec_from_geometry(case['geometry'])
                if case.get('stage') == 'S1'
                else s2_terrain.spec_from_geometry(case['geometry']))
        config = s2_state.S2TaskConfig.from_case(case)
        platform = None
        if (case['task'].startswith(('stairs_', 'platform_')) or case['task'] == 'stairs'
                or case['task'] == 'legacy_stairs'):
            front = float(spec.frame_origin_world[0]) + spec.N * spec.step_depth
            platform = dict(front_x=front, length=float(spec.platform_length),
                            y_half_width=float(spec.platform_width) / 2.0,
                            top_height=float(spec.platform_height))
        return cls(case=case, terrain=spec, config=config, platform=platform)

    def state_machine(self):
        return s2_state.S2StateMachine(self.config)

    def level_gate(self):
        return s2_state.LevelGate(flat=(self.config.task == 'flat_walk'))

    def surface_height(self, points_xy):
        return s2_terrain.query_surface(points_xy, self.terrain)

    def initial_state(self):
        spawn = self.case['spawn']
        return dict(local_position=list(spawn['local_position']),
                    local_yaw_radians=float(spawn['local_yaw_radians']),
                    joint_position=list(spawn['joint_position']),
                    joint_velocity=list(spawn['joint_velocity']),
                    base_linear_velocity=list(spawn['base_linear_velocity']),
                    base_angular_velocity=list(spawn['base_angular_velocity']))

    def zero_point(self):
        return (float(self.case['spawn']['local_position_xy'][0])
                if self.config.platform_start else None)


def to_numpy(value):
    """Host numpy at every evidence boundary; preserve GPU tensors for Actor."""
    if value is None:
        return None
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'numpy') and not isinstance(value, (str, bytes)):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    if isinstance(value, (np.ndarray, np.generic, list, tuple, int, float, bool)):
        return np.asarray(value) if not isinstance(value, np.ndarray) else value
    return np.asarray(value)


def jsonable(value, _path='$'):
    """Single recursive serialization boundary for all S2 evidence/reports."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if getattr(value, 'ndim', 0) == 0:
            return value.item()
        return value.numpy().tolist()
    if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
        try:
            value = value.cpu()
            if getattr(value, 'ndim', 0) == 0:
                return value.item()
            return np.asarray(value.numpy()).tolist()
        except Exception as error:
            raise TypeError(f'cannot jsonable {_path} ({type(value).__name__}): {error}') from None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item, f'{_path}.{key}')
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, f'{_path}[{index}]')
                for index, item in enumerate(value)]
    raise TypeError(f'cannot jsonable {_path}: {type(value).__name__}')


# Backward-compatible alias for existing internal callers.
_jsonable = jsonable


RAGGED_TRACE_FIELDS = ('contact_pairs', 'native_sample_id')


class S2EvidenceRecorder:
    """Accumulate pre-reset control-step evidence for one case/branch."""

    def __init__(self, *, case, adapter, map_branch='real'):
        if map_branch not in ('real', 'plane_ablation', 'plane_diagnostic'):
            raise ValueError('map_branch must be real or plane_ablation')
        self.case = case
        self.adapter = adapter
        self.map_branch = 'plane_ablation' if map_branch == 'plane_ablation' else map_branch
        self.rows = []
        self.native_samples = []
        self.trace_path = None

    def append(self, frame):
        missing = REQUIRED_PER_STEP - set(frame)
        if missing:
            raise ValueError(f'evidence frame missing fields: {sorted(missing)}')
        row = dict(frame)
        step = int(row['step'])
        if self.rows and step <= int(self.rows[-1]['step']):
            raise ValueError('evidence step must strictly increase')
        for key in SCALAR_FIELDS:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
            row[key] = value
        row['step'] = step
        row['base_xy'] = _as_float_array(row['base_xy'], 'base_xy', (2,))
        for key, shape in FOOT_FIELDS.items():
            if key in ('foot_on_platform', 'foot_contact'):
                row[key] = _as_bool_array(row[key], key, shape)
            else:
                row[key] = _as_float_array(row[key], key, shape)
        for key, shape in JOINT_FIELDS.items():
            row[key] = _as_float_array(row[key], key, shape)
        for key in BOOLEAN_FIELDS:
            row[key] = bool(row[key])
        row['map_valid'] = np.asarray(row['map_valid'], dtype=bool)
        if row['map_valid'].ndim != 2:
            raise ValueError('map_valid must be a 2-D valid grid')
        sample = row['native_sample_id']
        if sample is not None:
            from .s2_map_ablation import validate_native_sample_id
            validate_native_sample_id(sample)
            self.native_samples.append(dict(sample, actor_step=step))
        pairs = row['contact_pairs']
        if not isinstance(pairs, (list, tuple)):
            raise ValueError('contact_pairs must be a list')
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != set(CONTACT_PAIR_FIELDS):
                raise ValueError(f'contact pair requires {CONTACT_PAIR_FIELDS}')
        self.rows.append(row)
        if self.trace_path is not None:
            with open(self.trace_path, 'a', encoding='utf-8') as handle:
                handle.write(canonical_json(_jsonable(row)) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        return row

    def attach_trace(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'refusing to overwrite trace {path}')
        path.write_text('', encoding='utf-8')
        self.trace_path = path
        return path

    def trace(self):
        if not self.rows:
            raise ValueError('no evidence rows')
        trace = {}
        for key in TRACE_FIELDS:
            values = [to_numpy(row[key]) for row in self.rows]
            if key in RAGGED_TRACE_FIELDS:
                trace[key] = np.asarray(values, dtype=object)
            else:
                trace[key] = np.asarray(values)
        return trace

    def validate(self):
        trace = self.trace()
        count = len(trace['step'])
        for key, value in trace.items():
            if key == 'contact_pairs':
                continue
            if len(value) != count:
                raise ValueError(f'{key} length differs from step')
        if np.any(np.diff(trace['step']) <= 0):
            raise ValueError('step must strictly increase')
        if np.any(trace['g_level'] < 0) or np.any(trace['g_level'] > 1):
            raise ValueError('g_level outside [0,1]')
        return dict(rows=count, fields=len(trace), valid=True)

    def write(self, directory, *, stem=None, attempt_id=None, windows=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = stem or (f'{self.case.get("suite", "S2")}_{self.case["case_id"]:02d}_'
                        f'{self.case["task"]}_{self.case["case_sha256"][:12]}_'
                        f'{self.map_branch}')
        npz = directory / f'{stem}.npz'
        if npz.exists():
            raise FileExistsError(f'refusing to overwrite evidence {npz}')
        trace = self.trace()
        payload = {}
        for key, value in trace.items():
            payload[key] = (np.asarray(value, dtype=object) if key in
                            RAGGED_TRACE_FIELDS else value)
        np.savez_compressed(npz, **payload)
        trace_json = None
        if self.trace_path is not None:
            trace_json = directory / f'{stem}_trace.jsonl'
            if trace_json.exists():
                raise FileExistsError(f'refusing to overwrite trace {trace_json}')
            text = ''.join(canonical_json(_jsonable(row)) + '\n' for row in self.rows)
            trace_json.write_text(text, encoding='utf-8')
        trace_json_sha256 = (hashlib.sha256(trace_json.read_bytes()).hexdigest()
                             if trace_json is not None else None)
        metadata = dict(schema_version=EVIDENCE_SCHEMA, sds_sha256=s2_spec.SDS_SHA256,
                        case_sha256=self.case['case_sha256'],
                        case_id=self.case['case_id'],
                        suite=self.case.get('suite'), task=self.case['task'],
                        stair_count=self.case.get('stair_count'),
                        speed=self.case['speed'], map_branch=self.map_branch,
                        geometry=self.case.get('geometry'),
                        terrain_key=self.adapter.terrain.terrain_key,
                        hard_timeout_steps=self.case['hard_timeout_steps'],
                        timeout_steps=self.adapter.config.timeout_steps,
                        attempt_id=attempt_id,
                        pitch_sign_convention=dict(PITCH_SIGN_CONVENTION),
                        windows=windows,
                        rows=len(self.rows), npz=str(npz),
                        trace_path=(str(self.trace_path) if self.trace_path else None),
                        trace_json=(str(trace_json) if trace_json else None),
                        npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                        trace_sha256=trace_json_sha256,
                        native_samples=self.native_samples,
                        contract=evidence_contract())
        meta_path = directory / f'{stem}.json'
        if meta_path.exists():
            raise FileExistsError(f'refusing to overwrite evidence metadata {meta_path}')
        meta_path.write_text(canonical_json(metadata) + '\n', encoding='utf-8')
        return dict(npz=npz, metadata=meta_path, trace_json=trace_json,
                    **{k: metadata[k] for k in (
                        'rows', 'npz_sha256', 'trace_sha256')})


def case_record(*, case, adapter, recorder, metrics=None, map_evidence=None,
                state_summary=None, failure_context=None, config_sha256=None,
                pairing_sha256=None, shared_identity_sha256=None,
                g_level_report=None, termination_reason=None, attempt_id=None):
    """Complete per-case record consumed by group reporting."""
    from .s2_map_ablation import config_identity
    config_sha256 = config_sha256 or config_identity()['config_sha256']
    return dict(
        config_sha256=config_sha256,
        pairing_sha256=pairing_sha256,
        shared_identity_sha256=shared_identity_sha256,
        schema_version=EVIDENCE_SCHEMA, sds_sha256=s2_spec.SDS_SHA256,
        case_sha256=case['case_sha256'], case_id=case['case_id'],
        task=case['task'], speed=case['speed'], stair_count=case['stair_count'],
        geometry_key=adapter.terrain.terrain_key, map_branch=recorder.map_branch,
        hard_timeout_steps=adapter.config.timeout_steps,
        evidence_rows=len(recorder.rows),
        metrics=metrics, map_evidence=map_evidence,
        state=state_summary, failure_context=failure_context,
        g_level_report=g_level_report, termination_reason=termination_reason,
        attempt_id=attempt_id,
        pitch_sign_convention=dict(PITCH_SIGN_CONVENTION),
        evidence_contract_sha256=sha256(evidence_contract()),
    )


def branch_execution_plan(manifest, run_plan):
    """Pair real/plane entries per case; they must share the whole pre-intervention prefix."""
    if run_plan.get('manifest_sha256') != manifest['manifest_sha256']:
        raise ValueError('run plan differs from the frozen manifest')
    by_case = {}
    for entry in run_plan['entries']:
        by_case.setdefault(entry['case_sha256'], []).append(entry)
    pairs = []
    for case in manifest['cases']:
        entries = by_case.get(case['case_sha256'], [])
        if {row['map_branch'] for row in entries} != {'real', 'plane_diagnostic'}:
            raise ValueError('each case requires exactly one real and one plane entry')
        shared = {}
        for key in ('case_id', 'task', 'speed', 'stair_count',
                    'platform_width', 'hard_timeout_steps'):
            values = {canonical_json(row.get(key)) for row in entries}
            if len(values) != 1:
                raise ValueError(f'branch entries differ in {key}')
            shared[key] = entries[0].get(key)
        streams = {canonical_json(row['sample_streams']) for row in entries}
        commands = {canonical_json(row['command']) for row in entries}
        if len(streams) != 1 or len(commands) != 1:
            raise ValueError('branches must share the same stream and command')
        pairs.append(dict(
            case_sha256=case['case_sha256'], case_id=case['case_id'],
            task=case['task'], shared=shared,
            sample_streams=entries[0]['sample_streams'],
            command=entries[0]['command'],
            real_operation_id=next(row['operation_id'] for row in entries
                                   if row['map_branch'] == 'real'),
            plane_operation_id=next(row['operation_id'] for row in entries
                                    if row['map_branch'] == 'plane_diagnostic'),
            intervention='plane height replacement only after the shared prefix; '
                         'full climb waits for the front-foot top-support event and '
                         'the first strict-later normal refresh'))
    if len(pairs) != len(manifest['cases']):
        raise ValueError('branch plan does not cover every frozen case')
    return dict(schema_version=EVIDENCE_SCHEMA, sds_sha256=s2_spec.SDS_SHA256,
                manifest_sha256=manifest['manifest_sha256'],
                run_plan_sha256=run_plan['plan_sha256'], pairs=pairs,
                pair_count=len(pairs),
                shared_prefix='identical case/streams/command/weight/state until '
                              'the first plane replacement is consumed')


def pre_intervention_digest(trace, *, end_index):
    """Digest of every trace field before the first plane replacement frame."""
    if end_index < 0:
        raise ValueError('end_index must be >= 0')
    import hashlib
    payload = {}
    for key, value in trace.items():
        array = np.asarray(value)
        if array.shape[0] < end_index:
            raise ValueError(f'{key} is shorter than the shared prefix')
        payload[key] = array[:end_index].tolist()
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_branches_identical_until(real_trace, plane_trace, *, end_index):
    """Executor proof: real and plane traces agree elementwise before intervention."""
    if set(real_trace) != set(plane_trace):
        raise ValueError('branch traces expose different fields')
    left = pre_intervention_digest(real_trace, end_index=end_index)
    right = pre_intervention_digest(plane_trace, end_index=end_index)
    if left != right:
        raise ValueError('branch traces differ before intervention')
    return left
