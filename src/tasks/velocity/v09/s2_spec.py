"""S2 identity frozen by SDS v1.2, kept separate from the v1.1 S1 evidence.

Historical S1 artifacts bind SDS v1.1 (:data:`src.tasks.velocity.v09.spec.SDS_SHA256`).
SDS v1.2 adds S2 diagnostics without rewriting those artifacts, so this module
pins the v1.2 SHA and the approved S1 entry weight while leaving every S1
constant untouched.  No physics, model or simulator import happens here.

The user/planning-execution ruling `S2-ENV-ORDER-1024-512` is recorded verbatim
as an execution decision, not as SDS §2.3 text: S2 probes 1024 first and falls
back to 512 only after recording concrete failure evidence.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import spec as s1_spec

PROJECT_ROOT = s1_spec.PROJECT_ROOT
SDS_PATH = PROJECT_ROOT / 'doc/sds/SDS_A2_v0.9_Sequential_Map_Walk_Stop_Robustness.md'
SDS_SHA256 = '09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe'
S1_SDS_SHA256_V11 = s1_spec.SDS_SHA256
S1_EVIDENCE_BINDING = dict(
    sds_version='v1.1',
    sds_sha256=S1_SDS_SHA256_V11,
    note='S1 checkpoints, manifests and evidence keep the v1.1 SHA; S2 never rewrites them',
)
#: Planning Q015: each context is validated against exactly one document hash.
#: There is deliberately no API that accepts both versions at once.
SDS_CONTEXT_SHA256 = {'S1': S1_SDS_SHA256_V11, 'S2': SDS_SHA256}
S2_STREAM_SALT = SDS_SHA256
IDENTITY_STATUS = 'ruled_Q015'


def validate_sds_context(context):
    """Explicit version context: S1 proves v1.1, S2 proves v1.2; never both."""
    if context not in SDS_CONTEXT_SHA256:
        raise ValueError('SDS context must be exactly S1 or S2')
    expected = SDS_CONTEXT_SHA256[context]
    actual = hashlib.sha256(SDS_PATH.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'SDS context {context} requires {expected}, file is {actual}')
    return dict(context=context, expected_sha256=expected, file_sha256=actual,
                matches=True)

CODE_VERSION = s1_spec.CODE_VERSION
RECIPE = s1_spec.RECIPE
CHECKPOINT_SCHEMA = s1_spec.CHECKPOINT_SCHEMA
STAGE = 'S2'
CONTROL_DT = s1_spec.CONTROL_DT
SEED = s1_spec.SEED
SPEEDS = s1_spec.SPEEDS
ROLLOUT_STEPS = s1_spec.ROLLOUT_STEPS

# --------------------------------------------------------- approved entry
# User-approved direct S2 entry.  `s1_least.pt` is byte-identical to the S1.2
# frozen main-course checkpoint; the original file name is retained for audit.
S1_ENTRY_WEIGHT = PROJECT_ROOT / 'logs/a2_v09/s1_seed42_r3/checkpoints/s1_least.pt'
S1_ENTRY_WEIGHT_SHA256 = 'efa9caa8ce99e2cdff967212063e198b7ae68a6415727083b1db027f615ead24'
S1_ENTRY_WEIGHT_ORIGINAL = (PROJECT_ROOT / 'logs/a2_v09/s1_seed42_r3/checkpoints/'
    's1-q010-formal-20261120-r1_latest_000120004608_00009805_f93f026e6df3444aba4bfa5833be38a2.pt')
S1_ENTRY_WEIGHT_ORIGINAL_SHA256 = S1_ENTRY_WEIGHT_SHA256
S1_ENTRY_LEARNING_SAMPLES = 100_024_320
S1_ENTRY_IDENTITY = dict(
    approved_path=str(S1_ENTRY_WEIGHT.relative_to(PROJECT_ROOT)),
    approved_sha256=S1_ENTRY_WEIGHT_SHA256,
    original_path=str(S1_ENTRY_WEIGHT_ORIGINAL.relative_to(PROJECT_ROOT)),
    original_sha256=S1_ENTRY_WEIGHT_ORIGINAL_SHA256,
    same_bytes=True,
    s1_learning_samples=S1_ENTRY_LEARNING_SAMPLES,
    s1_gate_a='not_passed (record retained; no S1_RESTORED artifact is created)',
    admission='user-approved direct S2 entry per SDS v1.2 §1.3',
)

# --------------------------------------------------------------- S2 caps
S2_TOTAL_CAP = 4_000_000_000
S2_LEARNING_CAP = 3_600_000_000
S2_NONLEARNING_CAP = 400_000_000
S2_0_NONLEARNING_SUBCAP = 50_000_000  # §4.3: 0.5 亿, deducted from the 4 亿 pool
S2_COURSE_LEARNING_CAPS = dict(S2_1=800_000_000, S2_2=1_200_000_000, S2_3=1_200_000_000)
LEDGER_NAMESPACE = 'a2-v09-s2-sequential-r2'
LEDGER_HINT = 'logs/a2_v09/s2/ledger.jsonl'

# §2.3 order is 2048→1536→1024→512; the user fixed the S2 order as 1024→512.
S2_ENV_CANDIDATES = (1024, 512)
S2_ENV_ORDER_RULING = dict(
    id='S2-ENV-ORDER-1024-512',
    status='user_fixed',
    source='execution main conversation, 2026-11-22 (user ruling)',
    text=('S2 probes 1024 first; only on documented admission failure may it '
          'fall back to 512; a 512 failure stops the stage and is reported. '
          'This overrides the SDS §2.3 candidate order for S2 only.'),
    full_admission_evidence_required=True,
    evaluation_batch_size_independent_of_training_envs=True,
)

# ---------------------------------------------------------- S2 geometry
STAIR_HEIGHT = 0.20
STAIR_DEPTH = 0.20
STAIR_WIDTH = 0.80
PLATFORM_LENGTH = 20.0
PLATFORM_WIDTHS = (0.8, 4.0)
FULL_STAIR_COUNT = 40
L_ARRIVAL = 1.2          # §4.2 complete climb arrival allowance
T_APPROACH_MAX = 1.1     # |spawn_x| upper bound of the frozen L spawn range
POST_TOP_DIAGNOSTIC_DISTANCE = 5.0   # §4.3 "登顶后五米" observation window
PLATFORM_START_OFFSET = 1.0          # §4.1 one metre behind the platform edge
GUARD_SECONDS = 10.0

FLAT_WALK_SECONDS = 20.0
FLAT_WALK_COMPLETION_STEPS = 1000
FLAT_WALK_HARD_TIMEOUT_STEPS = 1001
# §4.2 planned flat_stop completion is 28 s (instant) or 30 s (2 s ramp);
# S2.0 does not freeze stop-index timing, so only the coarse cap is declared.
FLAT_STOP_COMPLETION_SECONDS = {'instant': 28.0, 'ramp_2s': 30.0}
FLAT_STOP_HARD_TIMEOUT_STEPS = {'instant': 1401, 'ramp_2s': 1501}

S2_DIAG_SUITE = 'S2_DIAG'
S2_0_SUITE = S2_DIAG_SUITE   # Q022 formal key; course name stays S2.0
S2_0_COURSE = 'S2.0'
S2_0_SUITE_NAMING_STATUS = 'ruled_Q022'
S2_0_SUPERSEDES = dict(
    path='logs/a2_v09/s2_0/cases/S2_0.json',
    manifest_sha256='5efda5abc35ad714dcf163b1648e2ac4ceac24edc12852889ee6cd3d3739acdf',
    file_sha256='09e2fc4636ba6f0cd1d310abcf615eec751c7f8b020bdcd281e858deda6c7079',
    reason='Q022: formal suite is S2_DIAG; old S2_0 manifest kept as zero-physics superseded record',
)
S2_0_GENERATOR = 'v09-s2-sha256-u53-v1'
S2_0_CASES_PER_GROUP = 5          # Q020: fixed case_id 0..4
S2_0_MAP_BRANCHES = ('real', 'plane_diagnostic')
S2_0_PAIRS_PER_GROUP = S2_0_CASES_PER_GROUP
S2_0_NATIVE_TRAJECTORIES = 15 * S2_0_CASES_PER_GROUP * len(S2_0_MAP_BRANCHES)  # 150
# Q025: platform quality window is [spawn+1 m, spawn+5 m]; first metre is separate.
PLATFORM_QUALITY_WINDOW_START_M = 1.0
PLATFORM_QUALITY_WINDOW_END_M = 5.0
PLATFORM_NOMINAL_QUALITY_WINDOW_M = 4.0
CLIMB_QUALITY_WINDOW_OFFSET_M = 1.0


def extra_first_meter_steps(speed):
    """Sensitivity only; Q025 says the §4.2 timeout does not add these steps."""
    speed = _finite_number(speed, 'speed', positive=True)
    return int(math.ceil((2.0 * 1.0 / speed) / CONTROL_DT))


H_SERIES_TIMEOUTS = (1301, 1302, 1306, 1315, 1318, 1332)
H_SERIES_TIMEOUT_SUM = sum(H_SERIES_TIMEOUTS)            # 7874
H_SERIES_BATCH_UPPER_BOUND = 6 * max(H_SERIES_TIMEOUTS)  # 7992, same-batch padding
H_SERIES_RESERVED_SAMPLES = 9432  # Q018 explicit batch/padding reserve
CASE_KEY_FIELDS = ('stage', 'suite', 'task', 'stair_count', 'geometry',
                   'speed', 'stop_mode', 'force_family', 'case_id')
WALK_TASKS = ('flat_walk', 'stairs_walk', 'platform_walk')
STOP_TASKS = ('flat_stop', 'stairs_stop', 'platform_stop')
S2_TASKS = WALK_TASKS + STOP_TASKS


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if positive and value <= 0:
        raise ValueError(f'{name} must be > 0')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be >= 0')
    return value


def flat_schedule(task):
    """§4.2 flat completion cap; no DECEL/HOLD indices are frozen here."""
    if task == 'flat_walk':
        return dict(completion_seconds=FLAT_WALK_SECONDS,
                    completion_steps=FLAT_WALK_COMPLETION_STEPS,
                    hard_timeout_steps=FLAT_WALK_HARD_TIMEOUT_STEPS,
                    contract='flat_full_window_plus_one_guard_step')
    if task == 'flat_stop':
        # Caller must add the stop-mode specific value; S2.0 never uses this.
        raise ValueError('flat_stop needs stop_mode; use flat_stop_schedule')
    raise ValueError('not a flat task')


def flat_stop_schedule(stop_mode):
    if stop_mode not in FLAT_STOP_COMPLETION_SECONDS:
        raise ValueError('unknown stop mode')
    seconds = FLAT_STOP_COMPLETION_SECONDS[stop_mode]
    return dict(completion_seconds=seconds,
                completion_steps=int(round(seconds / CONTROL_DT)),
                hard_timeout_steps=FLAT_STOP_HARD_TIMEOUT_STEPS[stop_mode],
                contract='flat_full_window_plus_one_guard_step',
                stop_index_timing='not_frozen_in_s2_0')


def stairs_route_length(*, stair_count, step_height, step_depth, approach_length,
                        post_distance):
    """§4.2 route length inside the outer bracket (no 2x, no guard/ramp/hold)."""
    for name, value in (('stair_count', stair_count), ('approach_length', approach_length),
                        ('post_distance', post_distance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a real number')
    if int(stair_count) < 0 or float(approach_length) < 0 or float(post_distance) < 0:
        raise ValueError('route length terms must be nonnegative')
    return (float(approach_length)
            + int(stair_count) * math.hypot(float(step_height), float(step_depth))
            + L_ARRIVAL + float(post_distance))


def timeout_steps(*, task, speed, stair_count=0, step_height=0.0, step_depth=0.0,
                  approach_length=0.0, post_distance=0.0, ramp_seconds=0.0,
                  hold_seconds=0.0):
    """SDS v1.2 §4.2 T_max converted with ceil to control steps.

    ``platform_*`` tasks set L_approach=N=L_arrival=0 per §4.2 even though the
    top height derives from ``stair_count``; ``stair_count`` is therefore only
    used by ``stairs_*`` tasks.
    """
    if task not in STOP_TASKS + WALK_TASKS:
        raise ValueError('unknown S2 task')
    speed = _finite_number(speed, 'speed', positive=True)
    ramp = _finite_number(ramp_seconds, 'ramp_seconds', nonnegative=True)
    hold = _finite_number(hold_seconds, 'hold_seconds', nonnegative=True)
    post = _finite_number(post_distance, 'post_distance', nonnegative=True)
    if task.startswith('flat_'):
        if post or ramp or hold:
            raise ValueError('flat tasks use the fixed §4.2 schedule, not T_max')
        return (flat_schedule(task) if task == 'flat_walk'
                else flat_stop_schedule('instant'))['hard_timeout_steps']
    if task.startswith('platform_'):
        route = post
    else:
        route = stairs_route_length(stair_count=stair_count, step_height=step_height,
                                    step_depth=step_depth,
                                    approach_length=approach_length,
                                    post_distance=post)
    seconds = 2.0 * route / speed + ramp + hold + GUARD_SECONDS
    return int(math.ceil(seconds / CONTROL_DT))


def complete_rollouts(cap, num_envs):
    """Whole N×48 rollouts only; the remainder is not a learning sample."""
    if num_envs not in S2_ENV_CANDIDATES:
        raise ValueError('S2 environment count must be 1024 or 512')
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError('cap must be a nonnegative integer')
    per = num_envs * ROLLOUT_STEPS
    return cap // per


def validate_identity():
    """Verify the v1.2 SDS and both recorded entry-weight files."""
    import json
    result = {}
    actual = hashlib.sha256(SDS_PATH.read_bytes()).hexdigest()
    if actual != SDS_SHA256:
        raise ValueError(f'S2 SDS v1.2 SHA256 mismatch: {actual}')
    result['sds'] = dict(path=str(SDS_PATH.relative_to(PROJECT_ROOT)), sha256=actual,
                         version='v1.2')
    for label, path, expected in (
            ('entry', S1_ENTRY_WEIGHT, S1_ENTRY_WEIGHT_SHA256),
            ('entry_original', S1_ENTRY_WEIGHT_ORIGINAL, S1_ENTRY_WEIGHT_ORIGINAL_SHA256)):
        if not path.exists():
            raise ValueError(f'missing S2 {label} weight: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'S2 {label} weight SHA256 mismatch: {digest}')
        result[label] = dict(path=str(path.relative_to(PROJECT_ROOT)), sha256=digest)
    if result['entry']['sha256'] != result['entry_original']['sha256']:
        raise ValueError('s1_least.pt is not byte-identical to the approved S1.2 checkpoint')
    return json.loads(json.dumps(result))
