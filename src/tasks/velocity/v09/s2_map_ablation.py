"""Q016/Q023/Q024 diagnostic plane height-channel ablation (CPU-testable).

Only the height channel of *valid* map cells is replaced.  Validity, age, grid
coordinates, 10 Hz refresh cadence and the sample frame stay exactly as the
real map produced them.  Invalid cells are forced to 0/0/1.

Q023 evidence (denominators fixed by ruling):
* ``valid_fraction = Σ valid cells over Actor-consumed frames / (frames × 40 × 30)``;
* ``saturation = Σ(valid & |h_input| ≥ 0.999) / Σ(valid cells)`` over the window
  from the first replacement *seen by the Actor* to the end of the case;
* ``exact_clip_fraction = Σ(valid & |unclipped normalized height| ≥ 1)`` over the
  same denominator;
* raw numerators/denominators are retained; a zero denominator is ``None`` (NA).
  A not-exposed branch is reported separately.

Q024 same-frame semantics use Actor step indices: an event latched after the
physics step ``t`` never changes the map consumed at Actor step ``t``.
``s = first step > t with a normal fresh sample`` is the first step that may
consume the replacement; no old refresh is rewritten, back-filled or skipped.

The module changes only the policy input; physics, rewards, commands,
statistics and termination are untouched.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from . import s2_spec
from . import spec as s1_spec

MAP_SCALE = 1.5
HEIGHT_INDEX = 0
VALID_INDEX = 1
AGE_INDEX = 2
INVALID_HEIGHT = 0.0
INVALID_VALID = 0.0
INVALID_AGE = 1.0

# Q023 definitions are named so a later ruling changes only these constants.
Q023_VALID_FRACTION_DENOMINATOR = 'actor_consumed_frames_x_map_cells'
Q023_SATURATION_THRESHOLD = 0.999
Q023_SATURATION_NUMERATOR = 'valid_cells_with_abs_normalized_height_ge_threshold'
Q023_SATURATION_DENOMINATOR = 'valid_cells_over_replacement_window'
Q023_EXACT_CLIP_THRESHOLD = 1.0
Q023_ZERO_DENOMINATOR = None  # NA, never a fabricated zero

# Q024: event frame t keeps the real map; first replacement is strictly later.
Q024_TRIGGER_FRAME_POLICY = 'real'
Q024_FIRST_REPLACEMENT_STRICTLY_AFTER_TRIGGER = True
Q024_REQUIRED_PAIRING_FIELDS = (
    'case_sha256', 'stream_sha256', 'weight_sha256', 'code_sha256',
    'physics_sha256', 'reward_sha256', 'termination_sha256',
    'state_sha256', 'command', 'action', 'sample_frame')



NATIVE_SAMPLE_ID_FIELDS = ('refresh_index', 'sample_sim_time_s',
                           'produced_at_actor_step')
PAIRING_REQUIRED_FIELDS = (
    'case_sha256', 'stream_sha256', 'weight_sha256', 'code_sha256',
    'physics_sha256', 'reward_sha256', 'termination_sha256',
    'state_sha256', 'command', 'action', 'sample_frame')
CONFIG_DIFF_ALLOWED_FIELDS = ('diagnostic_map_branch', 'record_id')


def validate_native_sample_id(sample_id):
    if not isinstance(sample_id, dict) or set(sample_id) != set(NATIVE_SAMPLE_ID_FIELDS):
        raise ValueError(f'native_sample_id requires exactly {NATIVE_SAMPLE_ID_FIELDS}')
    if (isinstance(sample_id['refresh_index'], bool)
            or not isinstance(sample_id['refresh_index'], int)
            or sample_id['refresh_index'] < 0):
        raise ValueError('native_sample_id.refresh_index must be a nonnegative integer')
    if (isinstance(sample_id['produced_at_actor_step'], bool)
            or not isinstance(sample_id['produced_at_actor_step'], int)
            or sample_id['produced_at_actor_step'] < 0):
        raise ValueError('native_sample_id.produced_at_actor_step must be a nonnegative integer')
    if (isinstance(sample_id['sample_sim_time_s'], bool)
            or not isinstance(sample_id['sample_sim_time_s'], (int, float))
            or not math.isfinite(float(sample_id['sample_sim_time_s']))):
        raise ValueError('native_sample_id.sample_sim_time_s must be finite')
    return True


def validate_pairing(pairing):
    if not isinstance(pairing, dict):
        raise ValueError('pairing evidence must be a mapping')
    missing = [key for key in PAIRING_REQUIRED_FIELDS if key not in pairing]
    if missing:
        raise ValueError(f'pairing evidence missing fields: {missing}')
    diff = pairing.get('config_diff', {})
    if not isinstance(diff, dict):
        raise ValueError('config_diff must be a mapping')
    forbidden = set(diff) - set(CONFIG_DIFF_ALLOWED_FIELDS)
    if forbidden:
        raise ValueError(f'config_diff may only differ in {CONFIG_DIFF_ALLOWED_FIELDS}, got {sorted(forbidden)}')
    return True


SHARED_IDENTITY_FIELDS = ('case_sha256', 'stream_sha256', 'weight_sha256',
                          'code_sha256', 'physics_sha256', 'reward_sha256',
                          'termination_sha256')


def validate_branch_pair(real_pairing, plane_pairing):
    """Q024: both branches share case/stream/weight/code/physics/reward/termination."""
    validate_pairing(real_pairing)
    validate_pairing(plane_pairing)
    mismatched = [key for key in SHARED_IDENTITY_FIELDS
                  if real_pairing.get(key) != plane_pairing.get(key)]
    if mismatched:
        raise ValueError(f'branch pairs differ in shared identity: {mismatched}')
    allowed_diff = {}
    for key in CONFIG_DIFF_ALLOWED_FIELDS:
        left, right = real_pairing.get(key), plane_pairing.get(key)
        if left != right:
            allowed_diff[key] = [left, right]
    payload = json.dumps({key: real_pairing[key] for key in SHARED_IDENTITY_FIELDS},
                         sort_keys=True, separators=(',', ':')).encode()
    return dict(shared_identity_sha256=hashlib.sha256(payload).hexdigest(),
                allowed_diff=allowed_diff)


def verify_policy_frame(real_map, policy_map, valid):
    """Per-frame Q023 proof: valid/age identical, invalid 0/0/1, height only valid."""
    real = _map_array(real_map, 'real_map')
    policy = _map_array(policy_map, 'policy_map')
    if real.shape != policy.shape:
        raise ValueError('paired maps differ in shape')
    validity = np.asarray(valid, dtype=bool)
    if validity.shape != real.shape[1:]:
        raise ValueError('valid shape must match maps')
    valid_unchanged = np.array_equal(real[VALID_INDEX], policy[VALID_INDEX])
    age_unchanged = np.array_equal(real[AGE_INDEX], policy[AGE_INDEX])
    invalid_ok = (np.all(policy[HEIGHT_INDEX][~validity] == INVALID_HEIGHT)
                  and np.all(policy[VALID_INDEX][~validity] == INVALID_VALID)
                  and np.all(policy[AGE_INDEX][~validity] == INVALID_AGE))
    changed = real[HEIGHT_INDEX] != policy[HEIGHT_INDEX]
    only_valid_changed = bool(np.all(validity[changed])) if changed.any() else True
    passed = bool(valid_unchanged and age_unchanged and invalid_ok
                  and only_valid_changed)
    return dict(valid_channel_unchanged=bool(valid_unchanged),
                age_channel_unchanged=bool(age_unchanged),
                invalid_cells_0_0_1=bool(invalid_ok),
                only_valid_height_changed=only_valid_changed,
                passed=passed)


def assert_only_height_changed(real_map, policy_map, valid):
    return verify_policy_frame(real_map, policy_map, valid)


def _sync_valid_age(policy_map, real_map, valid):
    """Keep the current frame's valid/age channels; invalid cells are 0/0/1."""
    out = np.array(policy_map, dtype=float, copy=True)
    validity = np.asarray(valid, dtype=bool)
    out[VALID_INDEX] = real_map[VALID_INDEX]
    out[AGE_INDEX] = real_map[AGE_INDEX]
    out[HEIGHT_INDEX, ~validity] = INVALID_HEIGHT
    out[VALID_INDEX, ~validity] = INVALID_VALID
    out[AGE_INDEX, ~validity] = INVALID_AGE
    return out


def config_identity():
    payload = dict(
        module='s2_map_ablation',
        height_formula='clip((z_plane_world - z_base_sample)/1.5, -1, 1)',
        scale=MAP_SCALE,
        channels_replaced=('height',),
        channels_unchanged=('valid', 'age'),
        invalid_cells='0/0/1',
        refresh_hz=10,
        activation=('full climb: first normal refresh strictly after first real '
                    'front-foot top support; platform/flat: birth frame'),
        q023=dict(valid_fraction_denominator=Q023_VALID_FRACTION_DENOMINATOR,
                  saturation_threshold=Q023_SATURATION_THRESHOLD,
                  saturation_numerator=Q023_SATURATION_NUMERATOR,
                  saturation_denominator=Q023_SATURATION_DENOMINATOR,
                  exact_clip_threshold=Q023_EXACT_CLIP_THRESHOLD,
                  zero_denominator='NA'),
        q024=dict(trigger_frame_policy=Q024_TRIGGER_FRAME_POLICY,
                  first_replacement_strictly_after_trigger=(
                      Q024_FIRST_REPLACEMENT_STRICTLY_AFTER_TRIGGER),
                  shared_identity_fields=list(SHARED_IDENTITY_FIELDS),
                  required_pairing_fields=list(Q024_REQUIRED_PAIRING_FIELDS),
                  required_pairing_field_count=len(Q024_REQUIRED_PAIRING_FIELDS)),
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return dict(payload, config_sha256=hashlib.sha256(encoded).hexdigest())


def _map_array(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f'{name} must have shape (3, H, W)')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    return array


def plane_height_value(plane_world_z, base_sample_z, *, scale=MAP_SCALE):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError('scale must be a real number')
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('scale must be positive and finite')
    plane = np.asarray(plane_world_z, dtype=float)
    base = np.asarray(base_sample_z, dtype=float)
    if not (np.all(np.isfinite(plane)) and np.all(np.isfinite(base))):
        raise ValueError('plane/base sample heights must be finite')
    raw = (plane - base) / float(scale)
    return np.clip(raw, -1.0, 1.0), raw


def frame_evidence(valid):
    validity = np.asarray(valid, dtype=bool)
    if validity.ndim != 2:
        raise ValueError('valid must have shape (H, W)')
    cells = int(validity.size)
    valid_cells = int(validity.sum())
    return dict(frames=1, cells=cells, valid_cells=valid_cells,
                valid_fraction=(valid_cells / cells if cells else Q023_ZERO_DENOMINATOR))


def apply_plane_height_channel(real_map, valid, plane_world_z, base_sample_z,
                               *, replace_mask=None):
    """Replace only valid-cell height; keep valid/age and invalid 0/0/1."""
    real = _map_array(real_map, 'real_map')
    validity = np.asarray(valid, dtype=bool)
    if validity.shape != real.shape[1:]:
        raise ValueError('valid shape must match (H, W)')
    if replace_mask is None:
        mask = validity.copy()
    else:
        mask = np.asarray(replace_mask, dtype=bool)
        if mask.shape != validity.shape:
            raise ValueError('replace_mask shape must match (H, W)')
        mask = mask & validity
    out = real.copy()
    out[HEIGHT_INDEX, ~validity] = INVALID_HEIGHT
    out[VALID_INDEX, ~validity] = INVALID_VALID
    out[AGE_INDEX, ~validity] = INVALID_AGE
    height, raw = plane_height_value(plane_world_z, base_sample_z)
    if np.ndim(height) == 0:
        height = np.full(validity.shape, float(height))
        raw = np.full(validity.shape, float(raw))
    else:
        height = np.broadcast_to(height, validity.shape)
        raw = np.broadcast_to(raw, validity.shape)
    out[HEIGHT_INDEX, mask] = height[mask]
    target = validity & mask
    replaced = int(mask.sum())
    valid_cells = int(validity.sum())
    saturated = int((np.abs(out[HEIGHT_INDEX][target])
                     >= Q023_SATURATION_THRESHOLD).sum()) if replaced else 0
    exact_clip = int((np.abs(raw[target])
                      >= Q023_EXACT_CLIP_THRESHOLD).sum()) if replaced else 0
    info = dict(
        branch='plane_diagnostic', active=True,
        frames=1, cells=int(validity.size), valid_cells=valid_cells,
        replaced_cells=replaced,
        valid_fraction=(valid_cells / validity.size if validity.size else Q023_ZERO_DENOMINATOR),
        replaced_fraction=(replaced / validity.size if validity.size else Q023_ZERO_DENOMINATOR),
        saturation_numerator=saturated,
        saturation_denominator=valid_cells,
        saturation=(saturated / valid_cells if valid_cells else Q023_ZERO_DENOMINATOR),
        exact_clip_numerator=exact_clip,
        exact_clip_fraction=(exact_clip / valid_cells if valid_cells else Q023_ZERO_DENOMINATOR),
        raw_min=(float(np.min(raw[target])) if replaced else None),
        raw_max=(float(np.max(raw[target])) if replaced else None),
        channels_unchanged=('valid', 'age'),
        invalid_cells='0/0/1',
        q023=dict(valid_fraction_denominator=Q023_VALID_FRACTION_DENOMINATOR,
                  saturation_threshold=Q023_SATURATION_THRESHOLD,
                  saturation_numerator=Q023_SATURATION_NUMERATOR,
                  saturation_denominator=Q023_SATURATION_DENOMINATOR,
                  exact_clip_threshold=Q023_EXACT_CLIP_THRESHOLD),
    )
    return out, info


def _sha256_array(array):
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(value.tobytes()).hexdigest()


def assert_only_height_changed(real_map, policy_map, valid):
    """Paired-map proof: only valid-cell heights may differ."""
    real = _map_array(real_map, 'real_map')
    policy = _map_array(policy_map, 'policy_map')
    if real.shape != policy.shape:
        raise ValueError('paired maps differ in shape')
    validity = np.asarray(valid, dtype=bool)
    if validity.shape != real.shape[1:]:
        raise ValueError('valid shape must match maps')
    valid_unchanged = np.array_equal(real[VALID_INDEX], policy[VALID_INDEX])
    age_unchanged = np.array_equal(real[AGE_INDEX], policy[AGE_INDEX])
    invalid_equal = np.array_equal(real[HEIGHT_INDEX][~validity],
                                   policy[HEIGHT_INDEX][~validity])
    changed = real[HEIGHT_INDEX] != policy[HEIGHT_INDEX]
    only_valid_changed = bool(np.all(validity[changed])) if changed.any() else True
    return dict(valid_channel_unchanged=bool(valid_unchanged),
                age_channel_unchanged=bool(age_unchanged),
                invalid_heights_unchanged=bool(invalid_equal),
                only_valid_height_changed=only_valid_changed,
                passed=bool(valid_unchanged and age_unchanged and invalid_equal
                            and only_valid_changed))


class S2PlaneAblation:
    """Per-episode latched ablation with Q023/Q024 evidence."""

    def __init__(self, *, task, plane_world_z, shape=(3, 40, 30)):
        if task not in s2_spec.S2_TASKS:
            raise ValueError('unknown S2 task')
        if not math.isfinite(float(plane_world_z)):
            raise ValueError('plane_world_z must be finite')
        if tuple(shape) != tuple(s1_spec.MAP_SHAPE):
            raise ValueError(f'plane ablation requires the native {s1_spec.MAP_SHAPE} map')
        self.task = task
        self.plane_world_z = float(plane_world_z)
        self.shape = tuple(shape)
        self.activation = ('birth_frame' if task.startswith(('flat_', 'platform_'))
                           else 'after_front_foot_top_support')
        self.active = self.activation == 'birth_frame'
        self.latched = self.active
        self.pending_trigger = False
        self.trigger_frame = None
        self.finished = False
        self.first_replacement_frame = None
        self.first_consumed_replacement_frame = None
        self.frames_consumed = 0
        self.valid_cell_frame_sum = 0
        self.replaced_cell_sum = 0
        self.valid_cell_replaced_sum = 0
        self.saturation_numerator = 0
        self.saturation_denominator = 0
        self.exact_clip_numerator = 0
        self.policy_cache = None
        self.raw_cache = None
        self.current_replaced = 0
        self.current_valid = 0
        self.first_pairing = None
        self.trigger_pairing = None
        self.replacement_pairing = None
        self.pairing_records = []
        self.paired_proof = None
        self.frame_checks = 0
        self.frame_check_failures = 0
        self.native_sample_id = None
        self.replacement_frames_consumed = 0
        self.sample_references = []

    def reset(self):
        """Reset one vector slot for a new case; no counter crosses reset."""
        self.active = self.activation == 'birth_frame'
        self.latched = self.active
        self.pending_trigger = False
        self.trigger_frame = None
        self.finished = False
        self.first_replacement_frame = None
        self.first_consumed_replacement_frame = None
        self.frames_consumed = 0
        self.valid_cell_frame_sum = 0
        self.replaced_cell_sum = 0
        self.valid_cell_replaced_sum = 0
        self.saturation_numerator = 0
        self.saturation_denominator = 0
        self.exact_clip_numerator = 0
        self.policy_cache = None
        self.raw_cache = None
        self.current_replaced = 0
        self.current_valid = 0
        self.first_pairing = None
        self.trigger_pairing = None
        self.replacement_pairing = None
        self.pairing_records = []
        self.paired_proof = None
        self.frame_checks = 0
        self.frame_check_failures = 0
        self.native_sample_id = None
        self.replacement_frames_consumed = 0
        self.sample_references = []

    def observe(self, *, frame_index, real_map, valid, base_sample_z,
                refreshed=False, front_foot_top_support=False,
                consumed=True, terminal=False, pairing=None, sample_id=None):
        """Consume one Actor frame; returns ``(policy_map, evidence)``."""
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer')
        real = _map_array(real_map, 'real_map')
        if real.shape != self.shape:
            raise ValueError('real_map differs from the frozen map shape')
        validity = np.asarray(valid, dtype=bool)
        if validity.shape != self.shape[1:]:
            raise ValueError('valid shape must match the frozen map')
        if pairing is not None:
            validate_pairing(pairing)
        if sample_id is not None:
            validate_native_sample_id(sample_id)
        if self.finished:
            policy = real.copy() if self.policy_cache is None else self.policy_cache.copy()
            return policy, dict(branch='plane_diagnostic', active=self.active,
                                finished=True, consumed=False)
        if not consumed:
            if terminal:
                self.finished = True
            policy = real.copy() if self.policy_cache is None else self.policy_cache.copy()
            return policy, dict(branch='plane_diagnostic', active=self.active,
                                consumed=False, frames=0, terminal=bool(terminal))
        frame = frame_evidence(validity)
        self.frames_consumed += 1
        self.valid_cell_frame_sum += frame['valid_cells']
        if pairing is not None and self.first_pairing is None:
            self.first_pairing = dict(pairing)
        if self.activation == 'after_front_foot_top_support' and not self.active:
            if front_foot_top_support and self.trigger_frame is None:
                self.trigger_frame = frame_index
                self.pending_trigger = True
                if pairing is not None:
                    self.trigger_pairing = dict(pairing)
            may_activate = (
                self.pending_trigger and refreshed
                and (not Q024_FIRST_REPLACEMENT_STRICTLY_AFTER_TRIGGER
                     or frame_index > self.trigger_frame))
            if may_activate:
                self.active = True
                self.latched = True
        trigger_frame_uses_real = (
            self.trigger_frame is not None and frame_index == self.trigger_frame)
        if not self.active or trigger_frame_uses_real:
            proof = verify_policy_frame(real, real, validity)
            self.frame_checks += 1
            if not proof['passed']:
                self.frame_check_failures += 1
                raise ValueError(f'real map failed Q023 frame proof at Actor step {frame_index}')
            if terminal:
                self.finished = True
            return real.copy(), dict(
                branch='plane_diagnostic', active=self.active, consumed=True,
                frames=1, cells=frame['cells'], valid_cells=frame['valid_cells'],
                replaced_cells=0, valid_fraction=frame['valid_fraction'],
                frame_proof=proof,
                saturation_numerator=0, saturation_denominator=0,
                saturation=Q023_ZERO_DENOMINATOR,
                exact_clip_numerator=0, exact_clip_fraction=Q023_ZERO_DENOMINATOR,
                trigger_frame=self.trigger_frame,
                trigger_frame_uses_real=trigger_frame_uses_real,
                first_consumed_replacement_frame=self.first_consumed_replacement_frame,
                q023=dict(valid_fraction_denominator=Q023_VALID_FRACTION_DENOMINATOR),
                q024=dict(trigger_frame_policy=Q024_TRIGGER_FRAME_POLICY),
            )
        if refreshed or self.policy_cache is None:
            policy, applied = apply_plane_height_channel(
                real, validity, self.plane_world_z, base_sample_z)
            self.policy_cache = policy
            raw = plane_height_value(self.plane_world_z, base_sample_z)[1]
            raw = np.asarray(raw, dtype=float)
            self.raw_cache = (np.full(validity.shape, float(raw))
                              if raw.ndim == 0 else np.broadcast_to(raw, validity.shape).copy())
            self.current_replaced = applied['replaced_cells']
            self.current_valid = applied['valid_cells']
            if self.first_replacement_frame is None:
                self.first_replacement_frame = frame_index
        else:
            policy = self.policy_cache
        policy = _sync_valid_age(policy, real, validity)
        self.policy_cache = policy
        if self.first_consumed_replacement_frame is None:
            self.first_consumed_replacement_frame = frame_index
            self.replacement_pairing = dict(pairing) if pairing is not None else None
            self.native_sample_id = dict(sample_id) if sample_id is not None else None
            self.paired_proof = verify_policy_frame(real, policy, validity)
            self.paired_proof['real_map_sha256'] = _sha256_array(real)
            self.paired_proof['policy_map_sha256'] = _sha256_array(policy)
        frame_proof = verify_policy_frame(real, policy, validity)
        self.frame_checks += 1
        if not frame_proof['passed']:
            self.frame_check_failures += 1
            raise ValueError(f'policy map failed per-frame Q023 proof at Actor step {frame_index}')
        self.replacement_frames_consumed += 1
        if sample_id is not None:
            self.sample_references.append((
                frame_index, int(sample_id['refresh_index']),
                float(sample_id['sample_sim_time_s']),
                int(sample_id['produced_at_actor_step'])))
        self.saturation_numerator += int((
            np.abs(policy[HEIGHT_INDEX][validity]) >= Q023_SATURATION_THRESHOLD).sum())
        self.saturation_denominator += int(validity.sum())
        self.exact_clip_numerator += int((
            np.abs(self.raw_cache[validity]) >= Q023_EXACT_CLIP_THRESHOLD).sum())
        self.replaced_cell_sum += self.current_replaced
        self.valid_cell_replaced_sum += self.current_valid
        if pairing is not None:
            self.pairing_records.append(dict(pairing, frame_index=frame_index))
            if len(self.pairing_records) > 8:
                self.pairing_records.pop(0)
        if terminal:
            self.finished = True
        return policy.copy(), dict(
            branch='plane_diagnostic', active=True, latched=True, consumed=True,
            frames=1, cells=frame['cells'], valid_cells=frame['valid_cells'],
            replaced_cells=self.current_replaced,
            valid_fraction=frame['valid_fraction'],
            saturation_numerator=self.saturation_numerator,
            saturation_denominator=self.saturation_denominator,
            saturation=(self.saturation_numerator / self.saturation_denominator
                        if self.saturation_denominator else Q023_ZERO_DENOMINATOR),
            exact_clip_numerator=self.exact_clip_numerator,
            exact_clip_denominator=self.saturation_denominator,
            exact_clip_fraction=(self.exact_clip_numerator / self.saturation_denominator
                                 if self.saturation_denominator else Q023_ZERO_DENOMINATOR),
            zero_valid_denominator=self.saturation_denominator == 0,
            zero_valid_denominator_reason=('zero_valid_denominator'
                                           if self.saturation_denominator == 0 else None),
            exact_clip_missing_reason=None if self.saturation_denominator else (
                'zero_valid_denominator'),
            missing_reason=('event_not_reached' if self.trigger_frame is None
                            and self.activation != 'birth_frame'
                            else 'no_valid_refresh' if self.first_consumed_replacement_frame is None
                            else None),
            trigger_frame=self.trigger_frame,
            trigger_frame_uses_real=False,
            first_replacement_frame=self.first_replacement_frame,
            first_consumed_replacement_frame=self.first_consumed_replacement_frame,
            delay_steps=(None if self.trigger_frame is None
                         else self.first_consumed_replacement_frame - self.trigger_frame),
            native_sample_id=dict(self.native_sample_id) if self.native_sample_id else None,
            replacement_applied=True,
            replacement_frames_consumed=self.replacement_frames_consumed,
            sample_references=list(self.sample_references),
            frame_proof=frame_proof,
            paired_proof=dict(self.paired_proof) if self.paired_proof else None,
            q023=dict(valid_fraction_denominator=Q023_VALID_FRACTION_DENOMINATOR,
                      saturation_threshold=Q023_SATURATION_THRESHOLD),
            q024=dict(trigger_frame_policy=Q024_TRIGGER_FRAME_POLICY),
        )

    def _pairing_sha256(self):
        records = [record for record in (self.first_pairing, self.replacement_pairing)
                   if record is not None]
        encoded = json.dumps(records, sort_keys=True, separators=(',', ':'),
                             default=str).encode()
        return hashlib.sha256(encoded).hexdigest() if records else None

    def summary(self):
        frames = self.frames_consumed
        total_cells = frames * self.shape[1] * self.shape[2]
        return dict(
            task=self.task, branch='plane_diagnostic', activation=self.activation,
            active=self.active, latched=self.latched,
            exposed=self.first_consumed_replacement_frame is not None,
            trigger_frame=self.trigger_frame,
            first_replacement_frame=self.first_replacement_frame,
            first_consumed_replacement_frame=self.first_consumed_replacement_frame,
            delay_steps=(None if self.trigger_frame is None
                         or self.first_consumed_replacement_frame is None
                         else self.first_consumed_replacement_frame - self.trigger_frame),
            frames_consumed=frames,
            valid_cell_frame_sum=self.valid_cell_frame_sum,
            valid_fraction=(self.valid_cell_frame_sum / total_cells
                            if total_cells else Q023_ZERO_DENOMINATOR),
            valid_fraction_numerator=self.valid_cell_frame_sum,
            valid_fraction_denominator=total_cells,
            saturation_numerator=self.saturation_numerator,
            saturation_denominator=self.saturation_denominator,
            saturation=(self.saturation_numerator / self.saturation_denominator
                        if self.saturation_denominator else Q023_ZERO_DENOMINATOR),
            exact_clip_numerator=self.exact_clip_numerator,
            exact_clip_denominator=self.saturation_denominator,
            exact_clip_fraction=(self.exact_clip_numerator / self.saturation_denominator
                                 if self.saturation_denominator else Q023_ZERO_DENOMINATOR),
            zero_valid_denominator=self.saturation_denominator == 0,
            zero_valid_denominator_reason=('zero_valid_denominator'
                                           if self.saturation_denominator == 0 else None),
            exact_clip_missing_reason=None if self.saturation_denominator else (
                'zero_valid_denominator'),
            missing_reason=('event_not_reached' if self.trigger_frame is None
                            and self.activation != 'birth_frame'
                            else 'no_valid_refresh' if self.first_consumed_replacement_frame is None
                            else None),
            replaced_cell_sum=self.replaced_cell_sum,
            valid_cell_replaced_sum=self.valid_cell_replaced_sum,
            event_step=(0 if self.activation == 'birth_frame' else self.trigger_frame),
            first_replacement_actor_step=self.first_consumed_replacement_frame,
            s_minus_t=(0 if self.activation == 'birth_frame'
                       and self.first_consumed_replacement_frame is not None
                       else None if self.trigger_frame is None
                       or self.first_consumed_replacement_frame is None
                       else self.first_consumed_replacement_frame - self.trigger_frame),
            native_sample_id=dict(self.native_sample_id) if self.native_sample_id else None,
            replacement_applied=True,
            replacement_frames_consumed=self.replacement_frames_consumed,
            sample_reference_count=len(self.sample_references),
            sample_references_sha256=hashlib.sha256(
                json.dumps(self.sample_references, separators=(',', ':')).encode()
            ).hexdigest() if self.sample_references else None,
            first_sample_reference=(self.sample_references[0]
                                    if self.sample_references else None),
            last_sample_reference=(self.sample_references[-1]
                                   if self.sample_references else None),
            frame_checks=self.frame_checks,
            frame_check_failures=self.frame_check_failures,
            plane_world_z=self.plane_world_z,
            config_sha256=config_identity()['config_sha256'],
            pre_intervention_pairing_sha256=(
                hashlib.sha256(json.dumps(self.first_pairing, sort_keys=True,
                                          separators=(',', ':'), default=str).encode()).hexdigest()
                if self.first_pairing is not None else None),
            input_pairing_sha256=self._pairing_sha256(),
            trigger_pairing_sha256=(
                hashlib.sha256(json.dumps(
                    self.trigger_pairing or self.first_pairing, sort_keys=True,
                    separators=(',', ':'), default=str).encode()).hexdigest()
                if (self.trigger_pairing or self.first_pairing) is not None else None),
            paired_proof=dict(self.paired_proof) if self.paired_proof else None,
            note=('not_exposed_before_terminal'
                  if self.first_consumed_replacement_frame is None else None),
            q023=dict(valid_fraction_denominator=Q023_VALID_FRACTION_DENOMINATOR,
                      saturation_threshold=Q023_SATURATION_THRESHOLD,
                      saturation_numerator=Q023_SATURATION_NUMERATOR,
                      saturation_denominator=Q023_SATURATION_DENOMINATOR),
            q024=dict(trigger_frame_policy=Q024_TRIGGER_FRAME_POLICY,
                      first_replacement_strictly_after_trigger=(
                          Q024_FIRST_REPLACEMENT_STRICTLY_AFTER_TRIGGER)),
        )


class S2RealBranchEvidence:
    """Q023 real-branch window using its own trigger/refresh, no replacement."""

    def __init__(self, *, task, shape=(3, 40, 30)):
        if task not in s2_spec.S2_TASKS:
            raise ValueError('unknown S2 task')
        if tuple(shape) != tuple(s1_spec.MAP_SHAPE):
            raise ValueError(f'real-branch evidence requires the native {s1_spec.MAP_SHAPE} map')
        self.task = task
        self.shape = tuple(shape)
        self.activation = ('birth_frame' if task.startswith(('flat_', 'platform_'))
                           else 'after_front_foot_top_support')
        self.active = self.activation == 'birth_frame'
        self.trigger_frame = 0 if self.active else None
        self.window_start_frame = 0 if self.active else None
        self.pending_trigger = False
        self.finished = False
        self.frames_consumed = 0
        self.valid_cell_frame_sum = 0
        self.saturation_numerator = 0
        self.saturation_denominator = 0
        self.frame_checks = 0
        self.frame_check_failures = 0
        self.native_sample_id = None
        self.replacement_frames_consumed = 0
        self.sample_references = []

    def reset(self):
        self.active = self.activation == 'birth_frame'
        self.trigger_frame = 0 if self.active else None
        self.window_start_frame = 0 if self.active else None
        self.pending_trigger = False
        self.finished = False
        self.frames_consumed = 0
        self.valid_cell_frame_sum = 0
        self.saturation_numerator = 0
        self.saturation_denominator = 0
        self.frame_checks = 0
        self.frame_check_failures = 0
        self.native_sample_id = None
        self.replacement_frames_consumed = 0
        self.sample_references = []

    def observe(self, *, frame_index, real_map, valid,
                refreshed=False, front_foot_top_support=False,
                consumed=True, terminal=False, pairing=None, sample_id=None):
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer')
        real = _map_array(real_map, 'real_map')
        if real.shape != self.shape:
            raise ValueError('real_map differs from the frozen map shape')
        validity = np.asarray(valid, dtype=bool)
        if validity.shape != self.shape[1:]:
            raise ValueError('valid shape must match the frozen map')
        if pairing is not None:
            validate_pairing(pairing)
        if sample_id is not None:
            validate_native_sample_id(sample_id)
        if self.finished:
            return dict(branch='real', finished=True, consumed=False)
        if not consumed:
            if terminal:
                self.finished = True
            return dict(branch='real', consumed=False, frames=0,
                        terminal=bool(terminal))
        frame = frame_evidence(validity)
        self.frames_consumed += 1
        self.valid_cell_frame_sum += frame['valid_cells']
        proof = verify_policy_frame(real, real, validity)
        self.frame_checks += 1
        if not proof['passed']:
            self.frame_check_failures += 1
            raise ValueError(f'real map failed Q023 frame proof at Actor step {frame_index}')
        if self.activation == 'after_front_foot_top_support' and not self.active:
            if front_foot_top_support and self.trigger_frame is None:
                self.trigger_frame = frame_index
                self.pending_trigger = True
            if (self.pending_trigger and refreshed
                    and frame_index > self.trigger_frame):
                self.active = True
                self.window_start_frame = frame_index
                self.native_sample_id = dict(sample_id) if sample_id is not None else None
        if terminal:
            self.finished = True
        if self.active:
            self.saturation_numerator += int((
                np.abs(real[HEIGHT_INDEX][validity]) >= Q023_SATURATION_THRESHOLD).sum())
            self.saturation_denominator += int(validity.sum())
            if sample_id is not None:
                self.sample_references.append((
                    frame_index, int(sample_id['refresh_index']),
                    float(sample_id['sample_sim_time_s']),
                    int(sample_id['produced_at_actor_step'])))
        return dict(branch='real', active=self.active, consumed=True,
                    frames=1, cells=frame['cells'], valid_cells=frame['valid_cells'],
                    replaced_cells=0, valid_fraction=frame['valid_fraction'],
                    saturation_numerator=self.saturation_numerator,
                    saturation_denominator=self.saturation_denominator,
                    exact_clip_numerator=0, exact_clip_denominator=None,
                    exact_clip_fraction=None,
                    exact_clip_missing_reason='real_map_stores_only_clipped_height',
                    event_step=self.trigger_frame,
                    first_replacement_actor_step=None,
                    first_replacement_frame=None,
                    window_start_frame=self.window_start_frame,
                    s_minus_t=(None if self.trigger_frame is None
                               or self.window_start_frame is None
                               else self.window_start_frame - self.trigger_frame),
                    native_sample_id=(dict(self.native_sample_id)
                                      if self.native_sample_id else None),
                    replacement_applied=False,
                    replacement_frames_consumed=0,
                    sample_references=list(self.sample_references),
                    frame_proof=proof)

    def summary(self):
        frames = self.frames_consumed
        cells = frames * self.shape[1] * self.shape[2]
        return dict(
            task=self.task, branch='real', activation=self.activation,
            active=self.active, latched=self.active,
            exposed=self.window_start_frame is not None,
            event_step=self.trigger_frame,
            trigger_frame=self.trigger_frame,
            window_start_frame=self.window_start_frame,
            first_replacement_frame=None,
            first_replacement_actor_step=None,
            s_minus_t=(None if self.trigger_frame is None
                       or self.window_start_frame is None
                       else self.window_start_frame - self.trigger_frame),
            frames_consumed=frames,
            valid_cell_frame_sum=self.valid_cell_frame_sum,
            valid_fraction=(self.valid_cell_frame_sum / cells
                            if cells else Q023_ZERO_DENOMINATOR),
            valid_fraction_numerator=self.valid_cell_frame_sum,
            valid_fraction_denominator=cells,
            saturation_numerator=self.saturation_numerator,
            saturation_denominator=self.saturation_denominator,
            saturation=(self.saturation_numerator / self.saturation_denominator
                        if self.saturation_denominator else Q023_ZERO_DENOMINATOR),
            exact_clip_numerator=0, exact_clip_denominator=None,
            exact_clip_fraction=None,
            zero_valid_denominator=self.saturation_denominator == 0,
            zero_valid_denominator_reason=('zero_valid_denominator'
                                           if self.saturation_denominator == 0 else None),
            exact_clip_missing_reason='real_map_stores_only_clipped_height',
            missing_reason=('event_not_reached' if self.trigger_frame is None
                            and self.activation != 'birth_frame'
                            else 'no_valid_refresh' if self.window_start_frame is None
                            else None),
            native_sample_id=(dict(self.native_sample_id)
                              if self.native_sample_id else None),
            replacement_applied=False,
            replacement_frames_consumed=0,
            sample_reference_count=len(self.sample_references),
            sample_references_sha256=hashlib.sha256(
                json.dumps(self.sample_references, separators=(',', ':')).encode()
            ).hexdigest() if self.sample_references else None,
            first_sample_reference=(self.sample_references[0]
                                    if self.sample_references else None),
            last_sample_reference=(self.sample_references[-1]
                                   if self.sample_references else None),
            frame_checks=self.frame_checks,
            frame_check_failures=self.frame_check_failures,
            note=('not_exposed_before_terminal'
                  if self.window_start_frame is None else None),
            q023=dict(valid_fraction_denominator=Q023_VALID_FRACTION_DENOMINATOR,
                      saturation_threshold=Q023_SATURATION_THRESHOLD),
            q024=dict(trigger_frame_policy=Q024_TRIGGER_FRAME_POLICY),
        )


def aggregate_evidence(records):
    """Group sums of raw numerators/denominators; NA when a denominator is 0."""
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty evidence list required')
    frames = sum(int(row['frames_consumed']) for row in records)
    cells = sum(int(row['valid_fraction_denominator']) for row in records)
    valid_cells = sum(int(row['valid_cell_frame_sum']) for row in records)
    sat_num = sum(int(row['saturation_numerator']) for row in records)
    sat_den = sum(int(row['saturation_denominator']) for row in records)
    clip_available = [row for row in records
                      if row.get('exact_clip_fraction') is not None]
    clip_num = sum(int(row['exact_clip_numerator']) for row in clip_available)
    clip_den = sum(int(row['saturation_denominator']) for row in clip_available)
    return dict(
        cases=len(records), frames_consumed=frames,
        valid_fraction_numerator=valid_cells,
        valid_fraction_denominator=cells,
        valid_fraction=(valid_cells / cells if cells else Q023_ZERO_DENOMINATOR),
        saturation_numerator=sat_num, saturation_denominator=sat_den,
        saturation=(sat_num / sat_den if sat_den else Q023_ZERO_DENOMINATOR),
        exact_clip_numerator=clip_num,
        exact_clip_denominator=clip_den if clip_available else None,
        exact_clip_fraction_denominator_scope='plane_rows_only',
        exact_clip_fraction=(clip_num / clip_den if clip_den else Q023_ZERO_DENOMINATOR),
        exact_clip_available_cases=len(clip_available),
        not_exposed=sum(1 for row in records if not row['exposed']),
        zero_frames=sum(1 for row in records
                        if row['valid_fraction_denominator'] == 0),
        zero_saturation_denominator=sum(1 for row in records
                                        if row['saturation_denominator'] == 0),
        zero_valid_denominator=sum(1 for row in records
                                   if row.get('zero_valid_denominator', False)),
        zero_valid_denominator_reason_counts={'zero_valid_denominator': sum(
            1 for row in records if row.get('zero_valid_denominator_reason')
            == 'zero_valid_denominator')},
    )
