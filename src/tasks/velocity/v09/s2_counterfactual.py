"""Q031 same-frame dual-input counterfactual (offline, no physics).

Real-branch inputs are captured before each control step.  Plane inputs are
derived from those *same* arrays by the formal S2 replacement rule; no physics
state is advanced.  This module and the CLI report only the Actor's immediate
response to the input change.  Strict physical trajectory pairing remains
`not_proven`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from . import s2_map_ablation

INPUT_KEYS = dict(proprio=(45,), history=(5, 45), heightmap=(3, 40, 30),
                  route_error=(3,), c_exec=(), c_request=(), recorded_action=(12,))
STAGE_PRE = 'pre_trigger'
STAGE_ACTIVATION = 'activation_refresh'
STAGE_POST = 'post_activation'
STAGE_NOT_TRIGGERED = 'not_triggered'


class InputContractError(ValueError):
    pass


def actor_identity(actor_bundle):
    """Normalizer/inference identity for Q031 repeatability."""
    import hashlib
    model = actor_bundle['model'] if isinstance(actor_bundle, dict) else actor_bundle
    normalizer = model.actor.normalizer
    state = normalizer.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        array = np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value,
                           dtype=np.float64)
        digest.update(key.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return dict(
        weight_sha256=(actor_bundle.get('state_sha256')
                       if isinstance(actor_bundle, dict) else None),
        normalizer_version='frozen_ETH_actor_normalizer',
        normalizer_state_sha256=digest.hexdigest(),
        inference_mode='eval/no_grad/deterministic_mean',
        action_scale=0.25,
        clip_position='not_clipped')


def load_saved_inputs(npz_path, metadata_path=None):
    payload = np.load(npz_path, allow_pickle=False)
    rows = []
    arrays = {key: payload[key] for key in payload.files}
    count = int(arrays['proprio'].shape[0])
    for index in range(count):
        row = {key: (value[index] if value.shape[:1] == (count,) else value)
               for key, value in arrays.items() if key in INPUT_KEYS}
        for key in ('refresh_index', 'sample_sim_time_s',
                    'produced_at_actor_step', 'base_sample_z', 'plane_world_z',
                    'refreshed', 'front_foot_top_support', 'a_request',
                    'recorded_action', 'step'):
            if key in arrays:
                row[key] = arrays[key][index]
        rows.append(row)
    metadata = (json.loads(Path(metadata_path).read_text())
                if metadata_path is not None else {})
    return rows, metadata


def _validate_rows(rows):
    if not rows:
        raise InputContractError('no input rows')
    for index, row in enumerate(rows):
        for key, shape in INPUT_KEYS.items():
            if key not in row:
                raise InputContractError(f'row {index} missing {key}')
            value = np.asarray(row[key], dtype=np.float32)
            if value.shape != shape:
                raise InputContractError(
                    f'row {index} {key} shape {value.shape} != {shape}')
            if not np.isfinite(value).all():
                raise InputContractError(f'row {index} {key} nonfinite')
        for key in ('base_sample_z', 'plane_world_z'):
            if key not in row or not math.isfinite(float(row[key])):
                raise InputContractError(f'row {index} missing finite {key}')
    return True


def derive_plane_rows(rows, *, task, attempts=None):
    """Derive plane heightmaps from the same real rows and validate the rule."""
    _validate_rows(rows)
    plane_world_z = float(rows[0]['plane_world_z'])
    ablation = s2_map_ablation.S2PlaneAblation(
        task=task, plane_world_z=plane_world_z)
    out, events, checks = [], [], []
    first_replacement = None
    for row in rows:
        real = np.asarray(row['heightmap'], dtype=np.float32)
        valid = real[1].astype(bool)
        sample_id = None
        if row.get('refresh_index') is not None and row.get('sample_sim_time_s') is not None \
                and row.get('produced_at_actor_step') is not None:
            sample_id = dict(refresh_index=int(row['refresh_index']),
                             sample_sim_time_s=float(row['sample_sim_time_s']),
                             produced_at_actor_step=int(row['produced_at_actor_step']))
        policy, evidence = ablation.observe(
            frame_index=int(row['step']), real_map=real, valid=valid,
            base_sample_z=float(row['base_sample_z']),
            refreshed=bool(row['refreshed']),
            front_foot_top_support=bool(row['front_foot_top_support']),
            consumed=True, terminal=False, sample_id=sample_id)
        policy = np.asarray(policy, dtype=np.float32)
        out.append(policy)
        events.append(evidence)
        if evidence.get('first_consumed_replacement_frame') is not None \
                and first_replacement is None:
            first_replacement = int(evidence['first_consumed_replacement_frame'])
        active = first_replacement is not None and int(row['step']) >= first_replacement
        if not active:
            checks.append(dict(step=int(row['step']), phase='pre', exact_equal=bool(
                np.array_equal(policy, real))))
        else:
            height_only = bool(np.array_equal(policy[1:], real[1:]))
            invalid_same = bool(np.array_equal(policy[0][~valid], real[0][~valid]))
            raw_ok = (ablation.raw_cache is not None and bool(np.allclose(
                policy[0][valid], np.asarray(ablation.raw_cache)[valid],
                rtol=0.0, atol=1e-6)))
            checks.append(dict(step=int(row['step']), phase='post',
                               valid_age_equal=height_only,
                               invalid_height_equal=invalid_same,
                               replacement_value_checked=raw_ok,
                               valid_height_changed=not bool(
                                   np.array_equal(policy[0][valid], real[0][valid]))))
    trigger = ablation.trigger_frame
    activation = first_replacement
    before = [c for c in checks if c['phase'] == 'pre']
    after = [c for c in checks if c['phase'] == 'post']
    rule = dict(
        trigger_frame=trigger, activation_frame=activation,
        before_frames=len(before),
        before_exact_equal=all(c['exact_equal'] for c in before),
        after_frames=len(after),
        after_valid_age_equal=all(c.get('valid_age_equal', False) for c in after),
        after_invalid_height_equal=all(c.get('invalid_height_equal', False) for c in after),
        after_only_valid_height_changed=all(
            c.get('valid_age_equal', False) and c.get('invalid_height_equal', False)
            for c in after),
        after_replacement_value_checked=all(
            c.get('replacement_value_checked', False) for c in after),
        replacement_events=events)
    return out, rule


def _batch(rows, key):
    return torch.as_tensor(np.asarray([row[key] for row in rows], dtype=np.float32))


def run_dual_inference(rows, plane_rows, *, policy):
    """Freeze one Actor step, switch only the map, infer twice; no physics."""
    real_all, plane_all, aa_all, bb_all = [], [], [], []
    for row, plane_map in zip(rows, plane_rows):
        obs = {key: torch.as_tensor(np.asarray(row[key], dtype=np.float32)[None])
               for key in INPUT_KEYS if key != 'recorded_action'}
        plane_obs = dict(obs)
        plane_obs['heightmap'] = torch.as_tensor(
            np.asarray(plane_map, dtype=np.float32)[None])
        def run(inputs):
            value = policy(inputs)
            return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value).reshape(-1)
        real_all.append(run(obs))
        plane_all.append(run(plane_obs))
        aa_all.append(run(obs))
        bb_all.append(run(plane_obs))
    return dict(real=np.asarray(real_all), plane=np.asarray(plane_all),
                real_repeat=np.asarray(aa_all), plane_repeat=np.asarray(bb_all))


def _stage_rows(rows, activation):
    if activation is None:
        return STAGE_NOT_TRIGGERED, rows
    pre = [r for r in rows if int(r['step']) < int(activation)]
    act = [r for r in rows if int(r['step']) == int(activation)]
    post = [r for r in rows if int(r['step']) > int(activation)]
    return None, dict(pre=pre, activation=act, post=post)


def build_stage_report(rows, actions, activation):
    """Aggregate action differences by Q031 stage."""
    real, plane, repeat = actions['real'], actions['plane'], actions['real_repeat']
    plane_repeat = actions['plane_repeat']
    groups = {STAGE_PRE: [], STAGE_ACTIVATION: [], STAGE_POST: []}
    if activation is None:
        groups[STAGE_NOT_TRIGGERED] = list(range(len(rows)))
    else:
        for index, row in enumerate(rows):
            step = int(row['step'])
            if step < int(activation):
                groups[STAGE_PRE].append(index)
            elif step == int(activation):
                groups[STAGE_ACTIVATION].append(index)
            else:
                groups[STAGE_POST].append(index)
    report = {}
    for name, indices in groups.items():
        if not indices:
            report[name] = dict(status='NA', n=0)
            continue
        diff = plane[indices] - real[indices]
        noise_a = repeat[indices] - real[indices]
        noise_b = plane_repeat[indices] - plane[indices]
        report[name] = dict(
            status='ok', n=len(indices),
            action_diff_rms=float(np.sqrt(np.mean(diff ** 2))),
            action_max_abs=float(np.max(np.abs(diff))),
            a_a_noise_rms=float(np.sqrt(np.mean(noise_a ** 2))),
            a_a_noise_max_abs=float(np.max(np.abs(noise_a))),
            b_b_noise_rms=float(np.sqrt(np.mean(noise_b ** 2))),
            b_b_noise_max_abs=float(np.max(np.abs(noise_b))),
            difference_significant=bool(
                float(np.max(np.abs(diff))) > max(float(np.max(np.abs(noise_a))),
                                                  float(np.max(np.abs(noise_b))))),
            significance_rule='max|A-B| > max(max|A-A|, max|B-B|)',
            per_dim_rms=np.sqrt(np.mean(diff ** 2, axis=0)).tolist(),
            per_dim_max_abs=np.max(np.abs(diff), axis=0).tolist(),
            action_unit='policy mean, raw normalized joint-position offset; '
                        'action term scale 0.25; no tanh/clip applied',
            action_scale=0.25,
            clip_position='not_clipped')
        recorded = np.asarray([row.get('recorded_action', np.full(12, np.nan))
                               for row in [rows[i] for i in indices]], dtype=np.float32)
        if np.isfinite(recorded).all():
            report[name]['recorded_real_max_abs'] = float(
                np.max(np.abs(recorded - real[indices])))
    return report


def analyze(rows, *, policy, task, case_sha256=None, weight_sha256=None,
            identity=None):
    plane_rows, rule = derive_plane_rows(rows, task=task)
    actions = run_dual_inference(rows, plane_rows, policy=policy)
    activation = rule['activation_frame']
    stages = build_stage_report(rows, actions, activation)
    return dict(
        case_sha256=case_sha256, weight_sha256=weight_sha256, task=task,
        inputs=len(rows), rule=rule, stages=stages,
        inference_identity=identity,
        same_frame_runner=True,
        evidence_grade='pending',
        physical_trajectory='not_proven',
        statement='immediate Actor response to same-frame input replacement; '
                  'no physics trajectory is paired by this report',
        action_unit='policy mean, raw normalized joint-position offset; '
                    'action term scale 0.25; no tanh/clip applied',
        action_scale=0.25,
        clip_position='not_clipped')
