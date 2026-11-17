"""A2 ideal-heightmap stairs v0.5-r2 identities, stage table, frozen 0.5-r2
recipe, PPO contract, route-state observation contract and explicit
product-entry source closure.

This module is the single numeric source of the v0.5-r2 line (SDS
``SDS_A2_IdealHeightMap_Stairs40_v0.5_r2.md`` revision 1.0, sections 1, 2, 5, 7
and 8.1).  Nothing here imports an older version recipe: the reward numbers,
stage table, slot allocation, PPO contract and route definition are declared
locally so a v0.5-r2 checkpoint can never inherit r1/0.4 defaults silently.

Relative to r1 (``src/tasks/velocity/v05/spec.py``) this file changes exactly
the SDS 1.2 items that belong to ``spec.py``:

* C1 identity: ``0.5.1`` / ``0.5-r2`` / schema ``7`` /
  ``v05-route-state-v1`` / Task ``...V05R2`` / the two r2 protocol ids /
  ``logs/a2_ideal_stairs_v05_r2``;
* C2 the fifth observation key ``route_error: float32[B,3]`` and its frozen
  route-state contract (``ROUTE_STATE``);
* C7 ``MODE_BUDGETS`` / ``mode_budget`` so worker, stage machine, supervisor and
  resume read one effective update cap (pilot 2500, formal 30000) instead of
  re-summing the curriculum table;
* the identity block additionally binds this repository's SDS and both Q003-r2
  decision documents by SHA256.

Every reward weight, event amount, PPO number, stage budget, gate interval, slot
count, command range, episode-cap formula and evaluation threshold is the
unchanged r1 value (SDS 1.2 "明确不变"); ``tests/test_v05_r2_spec.py`` compares
the whole reward configuration with the r1 module and asserts that only the
``version`` identity field differs.

The source contract is an *explicit product-entry closure*, copied in spirit
from ``v04/spec.py`` and ``v05/spec.py``: it starts from the r2 product entry
list and follows static imports with ``ast``.  It never expands through
``src/tasks/__init__.py`` (the legacy ``import_packages`` scanner) and therefore
cannot grow with unrelated task directories.  ``contract()`` works while sibling
r2 modules are still being written: every ``src/tasks/velocity/v05_r2/*.py``
that already exists is included, and the not-yet-written expected entries are
reported separately as ``pending_source_entries`` instead of being silently
hashed as absent.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

# --------------------------------------------------------------- identities
CODE_VERSION = '0.5.1'
RECIPE_VERSION = '0.5-r2'
SDS_REVISION = '1.0'
SCHEMA_VERSION = 7
ARCHITECTURE = 'v05-route-state-v1'
TASK = 'Unitree-A2-IdealHeightMap-Stairs-V05R2'
PROTOCOL_ID = 'v05-r2-stair-climb-first-v1'
FLAT_PROTOCOL_ID = 'v05-r2-flat-world-route-v1'
LOG_ROOT = 'logs/a2_ideal_stairs_v05_r2'
SMOKE_CONTAINER = 'v05-r2-smoke-state-v1'
SMOKE_FORMAL_RESUMABLE = False
SMOKE_UPDATES = 2
ROOT = Path(__file__).resolve().parents[4]

# ------------------------------------------- frozen document identity (SDS 1)
# The implementation contract is bound to the frozen SDS and to both Q003-r2
# decisions by SHA256.  These are the only document hashes this module may hold:
# they are inputs to the contract, never outputs of it.
SDS_PATH = 'doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.5_r2.md'
SDS_SHA256 = '9ff356b022e6205f0a181ff4048ef0f6e773742250d3174bf9b087986c8a4abc'
DESIGN_DECISION_PATH = ('logs/a2_ideal_stairs_v05/implementation/coordination/'
                        'Q003_r2_design_decision.md')
DESIGN_DECISION_SHA256 = ('78515bb951a96a8c69b9f10b90043a6312ed39124bc7898bd08b89e046d3a0fc')
IMPLEMENTATION_DECISION_PATH = ('logs/a2_ideal_stairs_v05/implementation/coordination/'
                                'Q003_r2_implementation_decision.md')
IMPLEMENTATION_DECISION_SHA256 = ('0cd996e72c494a6cd981db067230710dbc8a5c0930d97bed1e86069a86895bf1')
FROZEN_DOCUMENTS = (
    ('sds', SDS_PATH, SDS_SHA256),
    ('design_decision', DESIGN_DECISION_PATH, DESIGN_DECISION_SHA256),
    ('implementation_decision', IMPLEMENTATION_DECISION_PATH,
     IMPLEMENTATION_DECISION_SHA256),
)

# ------------------------------------------------------- approved ETH source
INIT_PATH = '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt'
INIT_SHA256 = '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'

# ------------------------------------------------------------------- timing
CONTROL_DT = .02
PHYSICS_DT = .005
DECIMATION = 4
MAP_HZ = 10
ACTION_SCALE = .25

# -------------------------------------------------------------- observation
# SDS 2.1: five keys and three consumption levels.  ``route_error`` is the only
# r2 addition; it is a fixed scale, never a normalizer input.
ROUTE_ERROR_KEY = 'route_error'
ROUTE_ERROR_DIM = 3
ROUTE_LATERAL_SCALE = .20
ROUTE_LATERAL_CLIP = 4.
ROUTE_LATERAL_SATURATION_M = ROUTE_LATERAL_SCALE * ROUTE_LATERAL_CLIP
ROUTE_YAW_AXIS = 0.
OBSERVATION = {'proprio': [45], 'history': [5, 45], 'heightmap': [3, 48, 32],
               'privileged': [35], ROUTE_ERROR_KEY: [ROUTE_ERROR_DIM]}
OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', 'privileged', ROUTE_ERROR_KEY)
ACTOR_OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', ROUTE_ERROR_KEY)
CRITIC_OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', 'privileged',
                           ROUTE_ERROR_KEY)
ETH_NORMALIZER_DIMS = 45
CRITIC_NORMALIZER_DIMS = 80
CRITIC_HISTORY_NORMALIZED_DIMS = 45
CRITIC_CONTINUOUS_MASK_FALSE = (48, 52)
CRITIC_INPUT_DIMS = CRITIC_NORMALIZER_DIMS + 64 + 64 + ROUTE_ERROR_DIM

#: SDS 4.2/2.1/12.1: the frozen route-state contract.  ``speed.py`` reads
#: ``ROUTE_STATE['architecture']``; ``ppo.py`` cross-checks the observation key
#: and dimension; evaluation/telemetry/play consume the same literals.
ROUTE_STATE = dict(
    architecture=ARCHITECTURE,
    key=ROUTE_ERROR_KEY,
    dim=ROUTE_ERROR_DIM,
    shape=[ROUTE_ERROR_DIM],
    dtype='float32',
    order=['clip(local_y/0.20,-4.0,4.0)', 'sin(e_world)', 'cos(e_world)'],
    formula='route_error = [clip(local_y/0.20,-4.0,4.0), sin(e_world), cos(e_world)]',
    e_world='wrap_to_pi(yaw_world - yaw_axis)',
    wrap='atan2(sin(.), cos(.))',
    angle_unit='radians',
    yaw_axis=ROUTE_YAW_AXIS,
    yaw_axis_premise='current atlas tiles are translations only; no rotation field',
    lateral_scale_m=ROUTE_LATERAL_SCALE,
    lateral_clip=ROUTE_LATERAL_CLIP,
    lateral_saturation_m=ROUTE_LATERAL_SATURATION_M,
    lateral_target='|local_y| = 0.20 m -> 1.0 (the F0 gate threshold)',
    source='same-frame root pose minus the slot tile frame origin',
    per_environment_tile=True,
    batch_tile_forbidden=True,
    same_frame_as_reward=True,
    normalizer='none_fixed_scale_no_running_mean_no_running_var',
    in_eth_normalizer=False,
    in_critic_normalizer=False,
    eth_normalizer_dims=ETH_NORMALIZER_DIMS,
    critic_normalizer_dims=CRITIC_NORMALIZER_DIMS,
    noise='none',
    clamp='first component only, [-4, 4]',
    flat='N == 0 uses the same three components, never a special case or zero fill',
    missing='reject, never zero fill',
    terminal_observation='pre-reset pose',
    next_observation='post-reset pose for reset environments, current pose otherwise',
    deployment_label='ideal_route_state',
    deployment='known-tile reference in simulation; no hardware validation',
    consumer=['actor.route_fusion', 'critic input tail [208:211]'],
    not_in=['reward', 'advantage', 'normalizer', 'telemetry gate'],
)
FOOT_ORDER = ('FL', 'FR', 'RL', 'RR')
JOINT_ORDER = tuple(f'{leg}_{joint}' for leg in FOOT_ORDER
                    for joint in ('hip', 'thigh', 'calf'))

# ------------------------------------------------------------ support physics
FOOT_RADIUS = .032
SAFE_INSET = .042
SUPPORT_FORCE_GT = 5.
SUPPORT_NORMAL_COS = math.cos(math.pi / 6)
SWING_TOP_TOLERANCE = .005
SUPPORT_DEBOUNCE = 2
MAX_SWING_STEPS = 40
COMMAND_MIN = .1

# ---------------------------------------------------------------- evaluation
SPEEDS = (.2, .4, .6)
SPEED_ORDER = SPEEDS
EPISODES_PER_SPEED = 20
EPISODES_PER_SPEED_FINAL = 100
CASE_PASS_REQUIRED = 18
CASE_PASS_REQUIRED_FINAL = 95
CONSECUTIVE_GATES = 2
SPEED_WINDOW = 25
SPEED_EPSILON = 1e-6
# SDS 6.1: mean_relative_error <= 0.20, smooth_mae_relative <= 0.30, and at most
# 50 consecutive samples above the 0.50 deviation threshold.  The v0.3 precedent
# (src/tasks/velocity/v03/speed.py) applies the same order, so the two limits must
# never be swapped back.
MEAN_RELATIVE_ERROR_MAX = .20
SMOOTH_MAE_MAX = .30
LONGEST_DEVIATION_MAX = 50
DEVIATION_THRESHOLD = .50
VALIDATION_SEED_OFFSET = 200300
PERTURBED_SEED_OFFSET = 300300
DIAGNOSTIC_SEED_OFFSET = 500300
FINAL_TEST_SEED = 400342
F0_OBSERVE_STEPS = 500
F0_WINDOW_STEPS = (101, 500)
F0_EPISODE_STEPS = 1500
#: SDS 6.2/5.5 F0 lateral and heading limits (radians for the heading).
F0_LATERAL_MAX = .20
F0_HEADING_MAX = math.radians(10.)

# ---------------------------------------------------------------- geometry
GEOMETRY = dict(step_depth=.2, stair_width=.8, tread_thickness=.01, platform_length=2.,
                risers=False, side_panels=False, first_edge_local_x=0., actual_tile_metadata=True)

# ------------------------------------------------------------ slot weights
# Formal 512-environment counts (SDS 5.1).  Other environment counts use the
# same weights with an integer largest-remainder allocation whose remainder tie
# order is flat -> previous -> current.
REFERENCE_NUM_ENVS = 512
SLOT_CLASSES = ('flat', 'previous', 'current')
REFERENCE_SLOT_COUNTS = {
    'F0': (512, 0, 0),
    'P1': (102, 0, 410),
    'P2': (51, 102, 359),
    'P3': (51, 102, 359),
    'P4': (51, 102, 359),
    'S1': (51, 102, 359),
    'S2': (51, 102, 359),
    'S3': (51, 102, 359),
    'S4': (51, 102, 359),
    'S5': (51, 102, 359),
}
FLAT_TILE_ID = 0
FLAT_ALIAS = 'flat'

# ------------------------------------------------------- r2 initial state (5.3)
#: SDS 5.3/5.4 B1+B2: every slot whose real atlas tile has ``N == 0`` samples its
#: own initial state per episode.  The centre is the SDS x=-1, never the r1
#: metadata ``x=0``.  ``yaw_range`` is in radians (+-5 degrees); ``yaw_radians``
#: is the same interval under the alias name the r2 environment binding reads
#: (``v05_r2/env.py`` accepts ``yaw``/``yaw_radians``), so one table serves both.
SPAWN_FLAT = dict(x_range=(-1.1, -.9), y_range=(-.05, .05),
                  yaw_range=(-math.pi / 36., math.pi / 36.),
                  yaw_radians=(-math.pi / 36., math.pi / 36.))
SPAWN_FLAT_YAW_DEGREES = (-5., 5.)
SPAWN_FLAT_CENTER = (-1., 0., 0.)
#: SDS 5.3 sample order: x, then y, then yaw, then the episode command.
SPAWN_FLAT_SAMPLE_ORDER = 'x,y,yaw,command'
#: SDS 5.5: the S5 current-slot perturbation ramps over the first 400 updates.
S5_RAMP_UPDATES = 400


# --------------------------------------------------------------- stage table
@dataclass(frozen=True)
class Stage:
    """One curriculum stage: atlas tile, steps, riser, update budget, gate interval."""
    id: str
    tile_id: int
    N: int
    h: float
    budget: int
    gate: int


STAGES = tuple(Stage(*value) for value in (
    ('F0', 0, 0, 0., 2000, 100),
    ('P1', 1, 1, .02, 1000, 100),
    ('P2', 2, 1, .04, 1000, 100),
    ('P3', 3, 1, .06, 1000, 100),
    ('P4', 4, 1, .08, 2000, 100),
    ('S1', 5, 4, .08, 3000, 200),
    ('S2', 6, 8, .12, 4000, 200),
    ('S3', 7, 12, .16, 4000, 200),
    ('S4', 8, 20, .20, 5000, 200),
    ('S5', 9, 40, .20, 7000, 200),
))
STAGE_ORDER = tuple(stage.id for stage in STAGES)
STAGE_BY_ID = {stage.id: stage for stage in STAGES}
STAGE_BY_TILE = {stage.tile_id: stage for stage in STAGES}
TOTAL_BUDGET = sum(stage.budget for stage in STAGES)

# ------------------------------------------------------- mode budget (SDS 7.3)
# The single source of the effective update cap.  r1 pilot ``result.json`` wrote
# ``total_budget=update_cap=29500`` (a ten-stage sum); r2 must never repeat that:
# worker, stage machine, supervisor and resume all read ``mode_budget``.
MODE_BUDGETS = {
    'pilot': dict(total=2500, stages={'F0': 2000, 'P1': 500}),
    'formal': dict(total=30000, stages={'F0': 2000, 'P1': 1000, 'P2': 1000,
                                        'P3': 1000, 'P4': 2000, 'S1': 3000,
                                        'S2': 4000, 'S3': 4000, 'S4': 5000,
                                        'S5': 7000}),
}
#: W7 resource probe: the formal stage table is borrowed for configuration
#: identity only; the real learning work is strictly 5 + 50 + 1 = 56 updates
#: (Q003-r2 implementation decision section 3.8), never the formal 30000.
PROBE_WARMUP_UPDATES = 5
PROBE_MEASURE_UPDATES = 50
PROBE_RECOVERY_UPDATES = 1
PROBE_UPDATES = PROBE_WARMUP_UPDATES + PROBE_MEASURE_UPDATES + PROBE_RECOVERY_UPDATES
#: ``smoke`` is a fixed two-update engineering acceptance run and is deliberately
#: absent from ``MODE_BUDGETS`` (SDS 7.3 "smoke 固定 2 更新，不进入该表").
CONTRACT_MODES = ('pilot', 'formal', 'probe', 'smoke')
PILOT_P1_UPDATES = MODE_BUDGETS['pilot']['stages']['P1']
PILOT_MAX_UPDATES = MODE_BUDGETS['pilot']['total']
PILOT_STAGES = ('F0', 'P1')


def _format_stage_budgets(stages):
    ordered = [stage for stage in STAGE_ORDER if stage in stages]
    remaining = sorted(name for name in stages if name not in STAGE_ORDER)
    if remaining:
        raise ValueError(f'unknown stage budgets: {remaining}')
    return ','.join(f'{name}={stages[name]}' for name in ordered)


FORMAL_STAGE_BUDGETS = _format_stage_budgets(MODE_BUDGETS['formal']['stages'])
PILOT_STAGE_BUDGETS = _format_stage_budgets(MODE_BUDGETS['pilot']['stages'])


def mode_budget(mode):
    """Deep copy of the one effective budget record for ``mode`` (SDS 7.3).

    ``pilot``  -> ``total=2500`` with ``F0=2000, P1=500``;
    ``formal`` -> ``total=30000`` with the complete ten-stage table;
    ``probe``  -> the formal stage table with ``total=56`` (W7 warm-up + measure +
       one recovery update; the table is borrowed for configuration identity only);
    ``smoke``  -> the fixed two-update engineering cap, never the formal table.

    Unknown modes raise ``ValueError``; callers must never re-derive a cap.
    """
    if not isinstance(mode, str):
        raise ValueError(f'unknown mode: {mode!r}')
    if mode == 'probe':
        return dict(total=PROBE_UPDATES,
                    stages=deepcopy(MODE_BUDGETS['formal']['stages']),
                    mode='probe', source='mode_budget_probe_borrows_formal_stages',
                    formal_resumable=True)
    if mode == 'smoke':
        return dict(total=SMOKE_UPDATES, stages={}, mode='smoke',
                    source='smoke_fixed_2_updates', formal_resumable=False)
    if mode not in MODE_BUDGETS:
        raise ValueError(f'unknown mode: {mode!r}; expected one of {CONTRACT_MODES}')
    record = MODE_BUDGETS[mode]
    return dict(total=int(record['total']), stages=deepcopy(record['stages']),
                mode=mode, source='mode_budget', formal_resumable=True)


def stage_budget(mode, stage_id):
    """One stage's effective cap from the same :func:`mode_budget` record."""
    budgets = mode_budget(mode)['stages']
    if stage_id not in budgets:
        raise ValueError(f'mode {mode!r} has no budget for stage {stage_id!r}')
    return budgets[stage_id]

# ------------------------------------------------------------- reward recipe
# SDS 3.3 lists the speed rate per *second*.  The ``0.02`` printed in SDS 3.3 and
# SDS 3.4 is the control dt itself, applied exactly once per step: SDS 3.1's
# "dt multiplied once", SDS 9.1-1's "dt=0.01 halves the continuous amount" and all
# three SDS 9.3 scalar checks only reconcile with ``amount = dt * rate``.  The
# historical ``*_COEFFICIENT`` names are kept as documented aliases of that dt so
# an existing ``coefficient * rate`` composition stays numerically correct.
SPEED_COEFFICIENT = CONTROL_DT
SPEED_DT = CONTROL_DT
SPEED_EXP_SCALE = 4.
SPEED_EXP_COEFFICIENT = 1.5
SPEED_EXP_OFFSET = -.5
SPEED_OVER_DEADBAND = 1.2
SPEED_OVER_COEFFICIENT = .5
SPEED_OVER_CAP = 4.
SPEED_BACKWARD_COEFFICIENT = .5
SPEED_BACKWARD_CAP = 4.
SPEED_SIDE_COEFFICIENT = .25
SPEED_SIDE_CAP = 4.
SPEED_SIDE_SCALE = .2
# a(u=0) = 1.5*exp(-4) - 0.5 = -0.4725265417; matching speed a(u=1) = +1.0.
STILL_SPEED_RATE = SPEED_EXP_COEFFICIENT * math.exp(-SPEED_EXP_SCALE) + SPEED_EXP_OFFSET
STILL_R_SPEED = CONTROL_DT * STILL_SPEED_RATE  # per-step amount while standing
MATCH_SPEED_RATE = 1.
MATCH_R_SPEED = CONTROL_DT * MATCH_SPEED_RATE  # per-step amount at matched speed

# SDS 3.4 sustained-stall window.
STALL_HISTORY_SAMPLES = 51
STALL_INTERVAL_SAMPLES = 50
STALL_WINDOW_SECONDS = 1.
STALL_DX_FRACTION_PER_SECOND = .15
STALL_DX_THRESHOLD_SCALE = 1.
STALL_HIGHWATER_DX = .03
STALL_CONSECUTIVE_SAMPLES = 50
STALL_RATE = -.25
STALL_COEFFICIENT = CONTROL_DT
STALL_DT = CONTROL_DT
STALL_REWARD = CONTROL_DT * STALL_RATE  # per-step amount: -0.005

# SDS 3.5/3.6 stepped events.
FOOTFALL_AMOUNT = .50
SWING_AMOUNT_PER_TARGET = .15
SWING_TARGET_WINDOW_M = .60
SWING_MAX_FRACTION = 1.
SWING_BODY_HEIGHT_SCALE = .04
SWING_TARGET_HEIGHT_MARGIN = .04
SWING_DELTA_DECIMALS = 1.
SUCCESS_AMOUNT = 10.
FAILURE_AMOUNT = -10.
#: SDS 3.7 center/heading weights: unchanged r1 values, never -2.0.
CENTER_WEIGHT = -.5
HEADING_WEIGHT = -.5
CENTER_LATERAL_SCALE = .20
CENTER_SATURATION = 4.

CONTINUOUS_WEIGHTS = dict(
    track_angular_velocity=1.,
    pose=.05,
    orientation=-1.,
    body_ang_vel=-.05,
    angular_momentum=-.025,
    joint_acc=-2.5e-7,
    joint_limits=-10.,
    action_rate=-.02,
    slip=-.25,
    soft_landing=-.001,
    stand_still=-1.,
    center=CENTER_WEIGHT,
    heading=HEADING_WEIGHT,
    edge=-1.,
    nonfoot=-1.,
)

# ------------------------------------------------------------------ PPO 2.2
PPO_CONTRACT = dict(
    num_envs=512, rollout=48, epochs=5, minibatches=4, updates_adam_steps=20,
    learning_rate=5e-5, min_lr=1e-5, max_lr=1e-4, desired_kl=.01, kl_adaptation=.015,
    gamma=.995, gae_lambda=.95, clip=.2, value_coef=1., max_grad_norm=1.,
    clipped_value_loss=True, entropy=.005, initial_std=.15, std_min=.10, std_max=.30,
    optimizer='fresh_adam_original_construction', std_projection='after_each_adam_step',
    std_constrained_in=['action_distribution', 'constrain_std'],
    mean_activation='none_no_tanh_no_clip',
    control_samples_per_update=48 * 512,
)
PPO_NUM_ENVS = PPO_CONTRACT['num_envs']
PPO_ROLLOUT = PPO_CONTRACT['rollout']
PPO_EPOCHS = PPO_CONTRACT['epochs']
PPO_MINIBATCHES = PPO_CONTRACT['minibatches']
PPO_INITIAL_LR = PPO_CONTRACT['learning_rate']
PPO_MIN_LR = PPO_CONTRACT['min_lr']
PPO_MAX_LR = PPO_CONTRACT['max_lr']
PPO_DESIRED_KL = PPO_CONTRACT['desired_kl']
PPO_GAMMA = PPO_CONTRACT['gamma']
PPO_LAMBDA = PPO_CONTRACT['gae_lambda']
PPO_CLIP = PPO_CONTRACT['clip']
PPO_VALUE_COEF = PPO_CONTRACT['value_coef']
PPO_MAX_GRAD_NORM = PPO_CONTRACT['max_grad_norm']
PPO_CLIPPED_VALUE_LOSS = PPO_CONTRACT['clipped_value_loss']
PPO_ENTROPY = PPO_CONTRACT['entropy']
INITIAL_STD = PPO_CONTRACT['initial_std']
STD_MIN = PPO_CONTRACT['std_min']
STD_MAX = PPO_CONTRACT['std_max']
TRAINING_NUM_ENVS = PPO_CONTRACT['num_envs']
EVALUATION_NUM_ENVS = 8
EVALUATION_NUM_ENVS_FINAL = 8
PLAY_NUM_ENVS = 1

# ----------------------------------------------- actor/critic network (SDS 2)
ACTOR_HIDDEN = (512, 256, 128)
ACTION_DIM = 12
#: SDS 2.3: the new r2 tensor group is exactly this one bypass layer.
ROUTE_FUSION_WEIGHT_KEY = 'actor.route_fusion.weight'
ROUTE_FUSION_BIAS_KEY = 'actor.route_fusion.bias'
ROUTE_FUSION_KEYS = (ROUTE_FUSION_WEIGHT_KEY, ROUTE_FUSION_BIAS_KEY)
ROUTE_FUSION_SHAPES = {ROUTE_FUSION_WEIGHT_KEY: (ACTOR_HIDDEN[0], ROUTE_ERROR_DIM),
                       ROUTE_FUSION_BIAS_KEY: (ACTOR_HIDDEN[0],)}
FUSION_KEYS = ('actor.fusion.weight', 'actor.fusion.bias')
CRITIC_HIDDEN = (512, 256, 128)
CRITIC_FIRST_KEY = 'critic.0.weight'
CRITIC_FIRST_SHAPE = (CRITIC_HIDDEN[0], CRITIC_INPUT_DIMS)
CRITIC_FIRST_BIAS_KEY = 'critic.0.bias'
CRITIC_LAYER_KEYS = ('critic.0.weight', 'critic.0.bias', 'critic.2.weight', 'critic.2.bias',
                     'critic.4.weight', 'critic.4.bias', 'critic.6.weight', 'critic.6.bias')
ACTOR_LAST_KEY = 'actor.mlp.6.weight'
#: SDS 2.2: only the Actor 45-dim MLP and its frozen normalizer are ETH sources.
ETH_ACTOR_SOURCE_KEYS = ('std', 'obs_normalizer._mean', 'obs_normalizer._var',
                         'obs_normalizer._std', 'obs_normalizer.count',
                         'mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias',
                         'mlp.4.weight', 'mlp.4.bias', 'mlp.6.weight', 'mlp.6.bias')
#: ETH stores a ``critic_state_dict`` (first layer ``(512,72)``); it is a
#: forbidden non-source block, never migrated (SDS 2.4 / 7.5 item 1).
ETH_FORBIDDEN_SOURCE_BLOCKS = ('critic_state_dict', 'optimizer_state_dict')

# ------------------------------------------------------------ slot helpers
def _stage(stage_id):
    if stage_id not in STAGE_BY_ID:
        raise ValueError(f'unknown v0.5-r2 stage: {stage_id!r}')
    return STAGE_BY_ID[stage_id]


def _positive_count(num_envs):
    if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) < 1:
        raise ValueError('num_envs must be a positive integer')
    return int(num_envs)


def previous_stage_id(stage_id):
    """The formal predecessor in the stage table (``None`` for F0)."""
    index = STAGE_ORDER.index(_stage(stage_id).id)
    return STAGE_ORDER[index - 1] if index else None


def slot_counts(stage_id, num_envs):
    """Return ``{'flat':int,'previous':int,'current':int}`` for a stage.

    The 512-environment counts are the formal SDS 5.1 table.  Any other
    environment count uses the same weights with an integer largest-remainder
    allocation; equal remainders are resolved flat -> previous -> current.
    """
    stage = _stage(stage_id)
    count = _positive_count(num_envs)
    weights = REFERENCE_SLOT_COUNTS[stage.id]
    quotients = [divmod(count * weight, REFERENCE_NUM_ENVS) for weight in weights]
    values = [quotient for quotient, _ in quotients]
    leftover = count - sum(values)
    order = sorted(range(len(weights)), key=lambda i: (-quotients[i][1], i))
    for index in order[:leftover]:
        values[index] += 1
    return dict(zip(SLOT_CLASSES, values))


def slot_layout(stage_id, num_envs):
    """Slot class names in environment order: flat, previous, current."""
    counts = slot_counts(stage_id, num_envs)
    layout = []
    for name in SLOT_CLASSES:
        layout.extend([name] * counts[name])
    return layout


def slot_tiles(stage_id, num_envs):
    """Atlas tile id per environment slot, in the same order as ``slot_layout``."""
    stage = _stage(stage_id)
    previous = previous_stage_id(stage.id)
    previous_tile = FLAT_TILE_ID if previous is None else STAGE_BY_ID[previous].tile_id
    tiles = {'flat': FLAT_TILE_ID, 'previous': previous_tile, 'current': stage.tile_id}
    return [tiles[name] for name in slot_layout(stage_id, num_envs)]


# ------------------------------------------------------- episode step limits
# SDS 5.1: flat episodes are capped at F0_EPISODE_STEPS and stair episodes at
# ``ceil((2*(2.2+N*sqrt(h*h+0.2*0.2))/c + 10)/0.02)`` using the slot's real N/h/c.
# This is the single source for that formula; evaluation, playback and any other
# consumer must call it instead of re-implementing the expression.
def _episode_arguments(stage_n, stage_h, executed_c):
    if isinstance(stage_n, bool) or not isinstance(stage_n, Integral) or int(stage_n) < 0:
        raise ValueError('N must be a non-negative integer')
    if isinstance(stage_h, bool):
        raise ValueError('h must be a real number')
    try:
        height = float(stage_h)
    except (TypeError, ValueError):
        raise ValueError('h must be a real number') from None
    if not math.isfinite(height) or height < 0:
        raise ValueError('h must be finite and non-negative')
    if isinstance(executed_c, bool):
        raise ValueError('executed_c must be a positive real number')
    try:
        command = float(executed_c)
    except (TypeError, ValueError):
        raise ValueError('executed_c must be a positive real number') from None
    if not math.isfinite(command) or command <= 0:
        raise ValueError('executed_c must be finite and positive')
    return int(stage_n), height, command


def stair_episode_steps(stage_n, stage_h, executed_c):
    """SDS 5.1 stair episode cap for a slot's real ``N``, ``h`` and ``c``.

    ``ceil((2*(2.2 + N*sqrt(h*h+0.2*0.2))/c + 10)/0.02)``; ``N == 0`` (flat) or
    ``h == 0`` falls back to the fixed flat cap ``F0_EPISODE_STEPS`` (1500).
    """
    count, height, command = _episode_arguments(stage_n, stage_h, executed_c)
    if count == 0 or height == 0:
        return F0_EPISODE_STEPS
    inner = 2. * (2.2 + count * math.sqrt(height * height + .2 * .2)) / command + 10.
    return int(math.ceil(inner / CONTROL_DT))


def episode_step_limit(tile_n, tile_h, executed_c):
    """Per-slot episode cap: dispatch on the slot's real tile ``N``/``h``/``c``.

    Flat slots (``N == 0``, and the degenerate ``h == 0``) always get the fixed
    ``F0_EPISODE_STEPS`` cap; stairs use :func:`stair_episode_steps`.  Callers must
    pass the *slot's* tile parameters, never the supervisor stage's.
    """
    count, height, command = _episode_arguments(tile_n, tile_h, executed_c)
    if count == 0 or height == 0:
        return F0_EPISODE_STEPS
    return stair_episode_steps(count, height, command)


STAIR_TIMEOUT_REFERENCE_CASES = (
    (4, .08, .4, 1266),
    (0, 0., .4, F0_EPISODE_STEPS),
    (40, .2, .2, 7257),
    (40, .2, .6, 2753),
)


def _check_frozen_tables():
    if abs(SPEED_COEFFICIENT - CONTROL_DT) > 0. or abs(STALL_COEFFICIENT - CONTROL_DT) > 0.:
        raise RuntimeError('the 0.02 printed in SDS 3.3/3.4 must be the control dt')
    if abs(STALL_REWARD - (-.005)) > 1e-12:
        raise RuntimeError('per-step stall amount must be dt*stall_rate = -0.005')
    if abs(STILL_R_SPEED - CONTROL_DT * STILL_SPEED_RATE) > 1e-15:
        raise RuntimeError('stationary per-step speed amount must be dt*still_rate')
    if (MEAN_RELATIVE_ERROR_MAX, SMOOTH_MAE_MAX, LONGEST_DEVIATION_MAX,
            DEVIATION_THRESHOLD) != (.20, .30, 50, .50):
        raise RuntimeError('SDS 6.1 speed gates are mean<=0.20, smooth_mae<=0.30, '
                           'longest_deviation<=50 above 0.50')
    for reference_n, reference_h, reference_c, expected in STAIR_TIMEOUT_REFERENCE_CASES:
        if stair_episode_steps(reference_n, reference_h, reference_c) != expected:
            raise RuntimeError(
                f'SDS 5.1 episode cap drift for N={reference_n} h={reference_h} '
                f'c={reference_c}: expected {expected}')
    if TOTAL_BUDGET != 30000:
        raise RuntimeError('v0.5-r2 formal stage budgets must sum to 30000 updates')
    for stage in STAGES:
        if slot_counts(stage.id, REFERENCE_NUM_ENVS) != dict(zip(SLOT_CLASSES, REFERENCE_SLOT_COUNTS[stage.id])):
            raise RuntimeError(f'512-environment slot counts drift for stage {stage.id}')
    if set(REFERENCE_SLOT_COUNTS) != set(STAGE_ORDER):
        raise RuntimeError('slot table must cover exactly the stage table')
    # SDS 7.3: one budget source; formal equals the frozen stage table, pilot is
    # exactly F0=2000 + P1=500 = 2500 and never the ten-stage sum.
    formal = MODE_BUDGETS['formal']
    if formal['total'] != TOTAL_BUDGET or formal['stages'] != {stage.id: stage.budget
                                                              for stage in STAGES}:
        raise RuntimeError('SDS 7.3 formal mode budget must equal the frozen stage table')
    pilot = MODE_BUDGETS['pilot']
    if pilot['stages'] != {'F0': STAGE_BY_ID['F0'].budget, 'P1': 500}:
        raise RuntimeError('SDS 7.3 pilot stage budgets must be F0=2000 P1=500')
    if pilot['total'] != 2500 or pilot['total'] != sum(pilot['stages'].values()):
        raise RuntimeError('SDS 7.3 pilot effective cap must be 2500 = 2000 + 500')
    if MODE_BUDGETS['formal']['total'] != 30000:
        raise RuntimeError('SDS 7.3 formal effective cap must be 30000')
    if PROBE_UPDATES != 56 or PROBE_WARMUP_UPDATES + PROBE_MEASURE_UPDATES \
            + PROBE_RECOVERY_UPDATES != PROBE_UPDATES:
        raise RuntimeError('Q003-r2 probe learning work must be exactly 5+50+1=56 updates')
    if SMOKE_UPDATES != 2:
        raise RuntimeError('SDS 7.3/10.5 smoke is a fixed two-update engineering run')
    if PILOT_P1_UPDATES != 500 or PILOT_MAX_UPDATES != 2500:
        raise RuntimeError('SDS 7.1 pilot P1 is fixed at 500 and the cap at 2500')
    if PILOT_STAGE_BUDGETS != 'F0=2000,P1=500':
        raise RuntimeError('SDS 10.4 pilot CLI stage budgets must be F0=2000,P1=500')
    if FORMAL_STAGE_BUDGETS != ('F0=2000,P1=1000,P2=1000,P3=1000,P4=2000,S1=3000,'
                                'S2=4000,S3=4000,S4=5000,S5=7000'):
        raise RuntimeError('SDS 10.6 formal CLI stage budgets drifted')
    # SDS 1.2/3.7: the reward weights are unchanged; only the recipe identity is new.
    if CONTINUOUS_WEIGHTS['center'] != -.5 or CONTINUOUS_WEIGHTS['heading'] != -.5:
        raise RuntimeError('SDS 3.7 center/heading weights must stay -0.50')
    if len(CONTINUOUS_WEIGHTS) != 15:
        raise RuntimeError('SDS 3.7 continuous term count drifted')
    # SDS 2.1/4.2: the route key is the fifth observation and is 3-dimensional.
    if list(OBSERVATION[ROUTE_ERROR_KEY]) != [3] or len(OBSERVATION) != 5:
        raise RuntimeError('SDS 2.1 observation must be five keys with route_error [3]')
    # SDS 5.3/5.5: the r2 flat initial state and the S5 ramp.
    if (SPAWN_FLAT['x_range'], SPAWN_FLAT['y_range']) != ((-1.1, -.9), (-.05, .05)):
        raise RuntimeError('SDS 5.3 r2 flat spawn x/y ranges drifted')
    for alias in ('yaw_range', 'yaw_radians'):
        if (abs(SPAWN_FLAT[alias][0] + math.pi / 36.) > 1e-15
                or abs(SPAWN_FLAT[alias][1] - math.pi / 36.) > 1e-15):
            raise RuntimeError('SDS 5.3 r2 flat spawn yaw range must be +-5 degrees')
    if SPAWN_FLAT_CENTER != (-1., 0., 0.) or S5_RAMP_UPDATES != 400:
        raise RuntimeError('SDS 5.3/5.5 r2 spawn centre or S5 ramp drifted')
    if (ROUTE_ERROR_DIM, ROUTE_LATERAL_SCALE, ROUTE_LATERAL_CLIP) != (3, .20, 4.):
        raise RuntimeError('SDS 4.2 route_error constants drifted')
    if ROUTE_YAW_AXIS != 0.:
        raise RuntimeError('SDS 4.2 supports yaw_axis == 0 only (translation atlas)')
    if abs(ROUTE_LATERAL_SATURATION_M - .8) > 1e-12:
        raise RuntimeError('SDS 4.2 route saturation is |local_y| >= 0.80 m')
    if CRITIC_INPUT_DIMS != 211:
        raise RuntimeError('SDS 2.4 critic input must be 80 + 64 + 64 + 3 = 211')
    if (CODE_VERSION, RECIPE_VERSION, SDS_REVISION, SCHEMA_VERSION, ARCHITECTURE) != \
            ('0.5.1', '0.5-r2', '1.0', 7, 'v05-route-state-v1'):
        raise RuntimeError('SDS 1.1 version identity drifted')
    for name, path, digest in FROZEN_DOCUMENTS:
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f'frozen {name} SHA256 must be a 64-character digest')


_check_frozen_tables()


def route_state():
    """Deep copy of the frozen route-state contract (SDS 2.1/4.2/12.1)."""
    return deepcopy(ROUTE_STATE)


def observation_contract():
    """The five-key observation contract with its actor/critic consumption."""
    return dict(
        order=list(OBSERVATION_KEYS), shapes={key: list(OBSERVATION[key])
                                              for key in OBSERVATION_KEYS},
        actor_consumes=list(ACTOR_OBSERVATION_KEYS),
        critic_consumes=list(CRITIC_OBSERVATION_KEYS),
        dtype='float32', device='same as environment', finite=True,
        batch='identical across all present keys',
        eth_normalizer_dims=ETH_NORMALIZER_DIMS,
        critic_normalizer_dims=CRITIC_NORMALIZER_DIMS,
        route_state=route_state(),
        missing_key='ValueError naming the key and the consumer; never zero filled',
        extra_keys='ignored by the network, never required by it',
    )


def _reward_config():
    return dict(
        version=RECIPE_VERSION,
        dt=CONTROL_DT,
        composition=('dt*(speed_rate + stall_rate + sum(other_continuous_weighted_rates)) '
                     '+ footfall_amount + swing_amount + success_amount + failure_amount'),
        dt_multiplied_once=True,
        total_reward_clamped=False,
        alive_bonus=False,
        active_definition='executed_c > 0.1',
        speed=dict(
            frame='body', dt=SPEED_DT, per_step_formula='dt * speed_rate',
            coefficient=SPEED_COEFFICIENT,
            coefficient_note=('the 0.02 printed in SDS 3.3 is the control dt and is '
                              'applied once; it is not an extra weight'),
            u='body_vx/executed_c', ey='body_vy/0.2', side_scale=SPEED_SIDE_SCALE,
            exp_scale=SPEED_EXP_SCALE, exp_coefficient=SPEED_EXP_COEFFICIENT,
            exp_offset=SPEED_EXP_OFFSET,
            over_deadband=SPEED_OVER_DEADBAND, over_coefficient=SPEED_OVER_COEFFICIENT,
            over_cap=SPEED_OVER_CAP, backward_coefficient=SPEED_BACKWARD_COEFFICIENT,
            backward_cap=SPEED_BACKWARD_CAP, side_coefficient=SPEED_SIDE_COEFFICIENT,
            side_cap=SPEED_SIDE_CAP,
            positive_part_only_gated=True, includes_dt=True,
            still_rate=STILL_SPEED_RATE, still_r_speed=STILL_R_SPEED,
            match_rate=MATCH_SPEED_RATE, match_r_speed=MATCH_R_SPEED,
            truth_table_per_second_rate=[[0., -0.472527], [.25, -0.341901],
                                         [.5, 0.051819], [1., 1.], [1.2, 0.778216],
                                         [1.5, 0.006819], [2., -0.792527], [3., -2.12]],
            discount_checks=dict(
                gamma=.995, steps=1000, c=.4, per_step='dt*rate',
                stationary=-2.482749, matched=3.973384, early_failure=-6.839384,
                note='SDS 9.3 scalar anchors; reproduce them from dt*rate, not 0.02*dt*rate'),
            disabled_zero_c='zero_with_explicit_enabled_false',
            active_nonpositive_c='error',
        ),
        stall=dict(
            dt=STALL_DT, per_step_formula='dt * stall_rate',
            coefficient=STALL_COEFFICIENT, rate=STALL_RATE, per_step_amount=STALL_REWARD,
            history_samples=STALL_HISTORY_SAMPLES, intervals=STALL_INTERVAL_SAMPLES,
            window_seconds=STALL_WINDOW_SECONDS,
            dx_threshold='0.15*executed_c*1.00 (strict less-than)',
            dx_fraction=STALL_DX_FRACTION_PER_SECOND, dx_scale=STALL_DX_THRESHOLD_SCALE,
            highwater_dx=STALL_HIGHWATER_DX, highwater_comparison='strict less-than',
            consecutive_samples=STALL_CONSECUTIVE_SAMPLES,
            first_penalty_sample=99, gated=False, termination_added=False,
            climbed_disables=True, reset_writes='x0 and M0=x0 with history count 1',
        ),
        events=dict(success=SUCCESS_AMOUNT, failure=FAILURE_AMOUNT, one_time=True,
                    failure_precedence=True, timeout_amount=0.,
                    success_requires='original_safe_arrival'),
        footfall=dict(amount=FOOTFALL_AMOUNT, recompute='events.sum(-1)*0.50',
                      per='foot_per_tile_step_per_episode', debounce=SUPPORT_DEBOUNCE,
                      min_step=1, ground_pays=False, top_platform_shares_last_step=True,
                      failure_blocks=True, no_dt_multiplier=True),
        swing=dict(
            amount_per_target=SWING_AMOUNT_PER_TARGET, max_fraction=SWING_MAX_FRACTION,
            target_window_m=SWING_TARGET_WINDOW_M, foot_radius=FOOT_RADIUS,
            support_force_gt=SUPPORT_FORCE_GT, safe_inset=SAFE_INSET,
            top_tolerance=SWING_TOP_TOLERANCE, normal_deg=30., debounce=SUPPORT_DEBOUNCE,
            body_height_scale=SWING_BODY_HEIGHT_SCALE,
            target_height_margin=SWING_TARGET_HEIGHT_MARGIN, max_steps=MAX_SWING_STEPS,
            command_active_min=COMMAND_MIN, no_dt_multiplier=True,
            consume_without_gate=True, target_lock_after_window_entry=True,
            target_lock_never_moves=True, paid_fraction_scope='episode_foot_target',
            reset_scope='episode', flat_N0='no candidate, no step-1 geom query, no event',
        ),
        continuous_weights=deepcopy(CONTINUOUS_WEIGHTS),
        pose=dict(weight=.05, sigma_active=[.15, .35, .5] * 4,
                  sigma_zero_command=[.05, .1, .15] * 4,
                  baseline_subtract=1., active_only=True),
        track_angular_velocity=dict(weight=1., baseline_subtract=1., active_only=True,
                                    formula='exp(-((wz-command_wz)^2+0.05*(wx^2+wy^2))/0.5)'),
        orientation=dict(weight=-1., roll_coefficient=1.,
                         lower_deadband=math.radians(5.), lower_coefficient=4.,
                         upper_deadband=math.radians(55.)),
        nonfoot=dict(weight=-1., force_threshold=10., aggregation='any_monitored_link',
                     averaging=False),
        center=dict(weight=CENTER_WEIGHT, formula='min((local_y/0.2)^2,4)*I(active)',
                    scope='active_command_including_flat_and_approach'),
        heading=dict(weight=HEADING_WEIGHT, formula='(1-cos(local_yaw))*I(active)',
                     scope='active_command_including_flat_and_approach'),
        edge=dict(weight=-1., inset=SAFE_INSET,
                  formula='mean_4feet(I(original_support_f)*relu((0.042-distance_f)/0.042)^2)',
                  last_step_and_platform_union=True),
        slip=dict(weight=-.25, formula='sum_f(I(contact_f)*(vx_f^2+vy_f^2))*I(active)',
                  averaged_over_feet=False),
        soft_landing=dict(weight=-.001,
                          formula='sum_f(norm(force_f)*I(contact_f AND duration_f<=dt+1e-7))*I(active)',
                          dt=CONTROL_DT),
        stand_still=dict(weight=-1., formula='sum((q-q_default)^2)*I(command_norm<=0.1)',
                         is_stall_term=False),
        removed=dict(body_x_progress=dict(old_weight=2., new_contribution=0.,
                                          visibility='diagnostic_only'),
                     old_linear_velocity='diagnostic_only'),
    )


REWARD_CONFIG = _reward_config()


def reward_config():
    return deepcopy(REWARD_CONFIG)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def reward_config_sha256():
    return hashlib.sha256(canonical_json(REWARD_CONFIG).encode('utf-8')).hexdigest()


def recipe_config(recipe=RECIPE_VERSION):
    """Compact, hashable 0.5-r2 recipe identity (numbers only)."""
    if recipe != RECIPE_VERSION:
        raise ValueError(f'v0.5-r2 requires the fixed {RECIPE_VERSION} reward recipe')
    return dict(
        recipe=RECIPE_VERSION, reward_version=RECIPE_VERSION, new_rewards=True,
        dt=CONTROL_DT, composition='r_speed + r_stall + dt*sum(other_continuous_weighted_rates) '
                                    '+ r_footfall + r_swing + r_success + r_failure',
        speed_dt=SPEED_DT, speed_coefficient=SPEED_COEFFICIENT,
        speed_per_step='dt*speed_rate',
        still_speed_rate=STILL_SPEED_RATE, still_amount_per_step=STILL_R_SPEED,
        match_speed_rate=MATCH_SPEED_RATE, match_amount_per_step=MATCH_R_SPEED,
        stall_dt=STALL_DT, stall_coefficient=STALL_COEFFICIENT,
        stall_per_step='dt*stall_rate', stall_rate=STALL_RATE, stall_amount=STALL_REWARD,
        footfall_amount=FOOTFALL_AMOUNT, swing_amount=SWING_AMOUNT_PER_TARGET,
        swing_window_m=SWING_TARGET_WINDOW_M, success=SUCCESS_AMOUNT,
        failure=FAILURE_AMOUNT,
        std_min=STD_MIN, std_max=STD_MAX, initial_std=INITIAL_STD, entropy=PPO_ENTROPY,
        reward_config_sha256=reward_config_sha256(),
    )


def identity_block():
    """The schema-7 checkpoint ``identity`` block (SDS 7.5, exact five keys)."""
    return dict(task=TASK, recipe=RECIPE_VERSION, code_version=CODE_VERSION,
                protocol=PROTOCOL_ID, sds_revision=SDS_REVISION)


def document_identity():
    """The frozen document SHAs bound into the contract identity section."""
    return dict(
        sds_path=SDS_PATH, sds_sha256=SDS_SHA256,
        design_decision_path=DESIGN_DECISION_PATH,
        design_decision_sha256=DESIGN_DECISION_SHA256,
        implementation_decision_path=IMPLEMENTATION_DECISION_PATH,
        implementation_decision_sha256=IMPLEMENTATION_DECISION_SHA256,
    )

# ------------------------------------------------------- source closure (8.1)
V05R2_DIRECTORY = 'src/tasks/velocity/v05_r2'
PRODUCT_MODULES = ('__init__', 'spec', 'model', 'rewards', 'env', 'speed', 'ppo',
                   'runner', 'recovery', 'evaluation', 'orchestration', 'telemetry',
                   'connection')
SCRIPT_ENTRIES = (
    'scripts/a2_v05_r2_bootstrap.py',
    'scripts/check_a2_ideal_stairs_v05_r2_migration.py',
    'scripts/train_a2_ideal_stairs_v05_r2.py',
    'scripts/run_a2_ideal_stairs_v05_r2.py',
    'scripts/evaluate_a2_ideal_stairs_v05_r2.py',
    'scripts/play_a2_ideal_stairs_v05_r2.py',
    'scripts/smoke_a2_ideal_stairs_v05_r2.py',
    'scripts/test_a2_ideal_stairs_v05_r2.py',
    'scripts/diagnostics/analyze_a2_v05_r2_rewards.py',
)
SOURCE_ENTRIES = tuple(f'{V05R2_DIRECTORY}/{module}.py' for module in PRODUCT_MODULES) \
    + SCRIPT_ENTRIES
DYNAMIC_DEPENDENCIES = ()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def product_entries(root=ROOT):
    """Explicit v0.5-r2 entry list: existing r2 modules plus existing entries.

    Every ``src/tasks/velocity/v05_r2/*.py`` that exists is bound, including files
    another worker may add to the same package; entries from ``SOURCE_ENTRIES``
    are bound as soon as they exist.  This is a single-directory listing of the
    v0.5-r2 package, never a scan of ``src/tasks``.
    """
    root = Path(root).resolve()
    entries = []
    directory = root / V05R2_DIRECTORY
    if directory.is_dir():
        entries.extend(sorted(str(path.relative_to(root)) for path in directory.glob('*.py')))
    for relative in SOURCE_ENTRIES:
        if relative not in entries and (root / relative).is_file():
            entries.append(relative)
    return sorted(entries)


def pending_source_entries(root=ROOT):
    """Expected v0.5-r2 product entries that do not exist yet (fail-closed report)."""
    root = Path(root).resolve()
    return [relative for relative in SOURCE_ENTRIES if not (root / relative).is_file()]


def source_hashes(paths=None, *, root=ROOT):
    """Hash explicit v0.5-r2 entries and their static local import closure.

    ``paths=None`` binds :func:`product_entries` (all existing v0.5-r2 product
    files).  An explicit ``paths`` list is strict: every listed file must exist.
    Package initializers that execute are included, except the legacy
    ``src/tasks/__init__.py`` scanner which is deliberately never executed by
    this version and therefore never part of its behavior contract.
    """
    root = Path(root).resolve()
    forbidden = (root / 'src/tasks/__init__.py').resolve()
    pending, seen = [], set()

    def add(path, required=False):
        path = Path(path)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('source path escapes project root')
        if path == forbidden:
            return
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return
        if path not in seen:
            seen.add(path)
            pending.append(path)
        parent = path.parent
        while parent != root:
            initializer = parent / '__init__.py'
            if (initializer.is_file() and initializer not in seen
                    and initializer.resolve() != forbidden):
                seen.add(initializer)
                pending.append(initializer)
            parent = parent.parent

    def add_module(parts):
        if not parts:
            return
        module = root.joinpath(*parts)
        add(module.with_suffix('.py'))
        add(module / '__init__.py')

    for relative in (product_entries(root) if paths is None else paths):
        add(relative, required=True)
    for relative in DYNAMIC_DEPENDENCIES:
        add(relative, required=True)
    while pending:
        path = pending.pop()
        if path.suffix != '.py':
            continue
        parts = list(path.relative_to(root).with_suffix('').parts)
        package = parts[:-1]
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name.split('.'))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError(f'invalid relative source import: {path}')
                    base = package[:len(package) - node.level + 1]
                else:
                    base = []
                target = base + (node.module.split('.') if node.module else [])
                add_module(target)
                for alias in node.names:
                    if alias.name != '*':
                        add_module(target + alias.name.split('.'))
    return {str(path.relative_to(root)): sha256(path) for path in sorted(seen)}


def behavior_hashes():
    """Per-production-file SHA256 map used as the 0.5-r2 behavior contract."""
    return source_hashes()


# ------------------------------------------------------------------ contract
def contract(recipe=RECIPE_VERSION, mode='formal', *, seed=42, device='cuda:0'):
    """Return the frozen 0.5-r2 protocol contract written into every schema-7 file.

    ``mode`` selects the one effective budget record (``mode_budget``); the
    contract carries ``total_budget``/``update_cap``/``stage_budgets``/
    ``effective_cap_source`` so a pilot result can never report the r1
    ten-stage-sum defect again.
    """
    if recipe != RECIPE_VERSION:
        raise ValueError(f'v0.5-r2 requires the fixed {RECIPE_VERSION} reward recipe')
    if mode not in CONTRACT_MODES:
        raise ValueError(f'mode must be one of {CONTRACT_MODES}')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 1:
        raise ValueError('seed must be a positive integer')
    if not isinstance(device, str) or not device:
        raise ValueError('device must be a non-empty string')
    budget = mode_budget(mode)
    hashes = behavior_hashes()
    pending = pending_source_entries()
    documents = document_identity()
    return dict(
        code_version=CODE_VERSION, recipe_version=RECIPE_VERSION,
        sds_revision=SDS_REVISION, checkpoint_schema=SCHEMA_VERSION,
        architecture=ARCHITECTURE,
        task=TASK, protocol_id=PROTOCOL_ID, flat_protocol_id=FLAT_PROTOCOL_ID,
        log_root=LOG_ROOT, mode=mode, seed=int(seed), device=device,
        identity=identity_block(),
        # The three frozen documents bound by this contract.  They are repeated at
        # top level (``sds_sha256`` ...) and grouped here as the contract's document
        # identity block; the schema-7 checkpoint ``identity`` mapping itself stays
        # exactly the five SDS 7.5 keys, so recovery can compare it verbatim.
        document_identity=dict(documents),
        sds_path=documents['sds_path'], sds_sha256=documents['sds_sha256'],
        design_decision_path=documents['design_decision_path'],
        design_decision_sha256=documents['design_decision_sha256'],
        implementation_decision_path=documents['implementation_decision_path'],
        implementation_decision_sha256=documents['implementation_decision_sha256'],
        recipe=recipe_config(recipe), reward_version=RECIPE_VERSION,
        reward_config=deepcopy(REWARD_CONFIG), reward_config_sha256=reward_config_sha256(),
        source_eth_path=INIT_PATH, source_eth_sha256=INIT_SHA256,
        initialization='explicit_eth_actor_mlp_and_frozen_normalizer',
        critic_initialization='fresh_seeded_211', optimizer_initialization='fresh_adam',
        actor_statistics='copied_complete_and_frozen',
        route_fusion_initialization='zeros_after_construction',
        observation=deepcopy(OBSERVATION), observation_contract=observation_contract(),
        route_state=route_state(), foot_order=list(FOOT_ORDER),
        joint_order=list(JOINT_ORDER),
        architecture_detail=dict(
            actor='ELU(mlp0(eth_normalizer(proprio)) + fusion(features) + '
                  'route_fusion(route_error)) then 512-256-128-12',
            bypass='nn.Linear(3, 512) zero weight and zero bias',
            critic='211-512-256-128-1 with route_error appended at [208:211]',
            critic_normalizer_dims=CRITIC_NORMALIZER_DIMS,
            eth_normalizer_dims=ETH_NORMALIZER_DIMS,
            action_dim=ACTION_DIM, initial_std=INITIAL_STD,
            std_range=[STD_MIN, STD_MAX],
            route_fusion_keys=list(ROUTE_FUSION_KEYS)),
        physics_dt=PHYSICS_DT, control_dt=CONTROL_DT, decimation=DECIMATION,
        map_hz=MAP_HZ, action_scale=ACTION_SCALE,
        stages=[asdict(stage) for stage in STAGES], stage_order=list(STAGE_ORDER),
        total_budget=budget['total'], update_cap=budget['total'],
        stage_budgets=dict(budget['stages']),
        effective_cap_source=budget['source'],
        mode_budgets={name: dict(total=record['total'],
                                 stages=deepcopy(record['stages']))
                      for name, record in sorted(MODE_BUDGETS.items())},
        formal_stage_budgets=FORMAL_STAGE_BUDGETS, pilot_stage_budgets=PILOT_STAGE_BUDGETS,
        pilot_p1_updates=PILOT_P1_UPDATES, pilot_max_updates=PILOT_MAX_UPDATES,
        probe_updates=PROBE_UPDATES, smoke_updates=SMOKE_UPDATES,
        smoke_container=SMOKE_CONTAINER, smoke_formal_resumable=SMOKE_FORMAL_RESUMABLE,
        training_num_envs=TRAINING_NUM_ENVS, evaluation_num_envs=EVALUATION_NUM_ENVS,
        environment_count_policy='fixed_512_no_downgrade',
        slot_classes=list(SLOT_CLASSES),
        slot_reference_envs=REFERENCE_NUM_ENVS,
        slot_reference_counts={key: list(value) for key, value in REFERENCE_SLOT_COUNTS.items()},
        slot_counts_512={stage.id: slot_counts(stage.id, REFERENCE_NUM_ENVS) for stage in STAGES},
        slot_layout_8={stage.id: slot_layout(stage.id, EVALUATION_NUM_ENVS) for stage in STAGES},
        allocation='integer_largest_remainder_flat_previous_current',
        slots_fixed_within_stage=True, flat_alias='topology_metadata_only_not_a_stage',
        geometry=deepcopy(GEOMETRY),
        command=dict(vx_range=[.2, .6], vy=0., wz=0.,
                     sampling='uniform_per_episode_fixed_within_episode',
                     range_per_episode=[.2, .6], zero_command='parking_diagnostic_only'),
        timeout=dict(
            flat_steps=F0_EPISODE_STEPS,
            stairs_formula='ceil((2*(2.2+N*sqrt(h*h+0.2*0.2))/c+10)/0.02)',
            implementation='spec.episode_step_limit(tile_n, tile_h, executed_c) -> int',
            stair_implementation='spec.stair_episode_steps(stage_n, stage_h, executed_c) -> int',
            reference_cases=[list(case) for case in STAIR_TIMEOUT_REFERENCE_CASES],
            flat_exception='N == 0 or h == 0 -> F0_EPISODE_STEPS',
            bootstrap='pre_reset_terminal_observation',
            stagnation_termination=False),
        spawn=dict(
            fixed=[-1., 0., 0.], S5_ramp_updates=S5_RAMP_UPDATES,
            ramp='completed_stage_updates/400 clipped to [0,1]',
            previous_uses_own_configuration=True,
            full_x=[-1.2, -.8], full_y=[-.1, .1], full_yaw_degrees=[-10., 10.],
            flat=dict(
                distribution='uniform_per_slot_per_episode',
                x_range=list(SPAWN_FLAT['x_range']), y_range=list(SPAWN_FLAT['y_range']),
                yaw_degrees=list(SPAWN_FLAT_YAW_DEGREES),
                yaw_radians=list(SPAWN_FLAT['yaw_range']),
                center=list(SPAWN_FLAT_CENTER),
                sample_order=SPAWN_FLAT_SAMPLE_ORDER,
                subset_order='flat_subset_then_stair_subset_then_command',
                rng='the environment generator only, no global or temporary rng',
                initial_cases_precedence=True,
                initial_pose_source='r2_flat_uniform',
                applies_to='every slot whose real atlas tile has N == 0',
                b1='flat spawn x corrected from the r1 metadata x=0 to x~U[-1.1,-0.9]',
                b2='new flat y/yaw randomization, not an r1 behavior',
                terrain_metadata='read-only v02/terrain.py frame origin, spawn_x=0')),
        arrival=dict(body_x_margin=.8, body_y_abs_max=.2, foot_safe_inset=SAFE_INSET,
                     support_force_gt=SUPPORT_FORCE_GT, recent_support_samples=50,
                     same_surface_consecutive_samples=2, current_support_min_feet=2,
                     all_feet_recent=True, consecutive_samples=5,
                     failure_precedence=True, event_once_per_episode=True),
        evaluation=dict(speeds=list(SPEEDS), speed_order=list(SPEED_ORDER),
                        episodes_per_speed=EPISODES_PER_SPEED,
                        episodes_per_speed_final=EPISODES_PER_SPEED_FINAL,
                        case_pass_required=CASE_PASS_REQUIRED,
                        case_pass_required_final=CASE_PASS_REQUIRED_FINAL,
                        consecutive_gates=CONSECUTIVE_GATES,
                        validation_seed=VALIDATION_SEED_OFFSET + int(seed),
                        perturbed_seed=PERTURBED_SEED_OFFSET + int(seed),
                        diagnostic_seed=DIAGNOSTIC_SEED_OFFSET + int(seed),
                        final_test_seed=FINAL_TEST_SEED, F0_observe_steps=F0_OBSERVE_STEPS,
                        F0_window_steps=list(F0_WINDOW_STEPS), speed_window=SPEED_WINDOW,
                        mean_relative_error_max=MEAN_RELATIVE_ERROR_MAX,
                        smooth_mae_max=SMOOTH_MAE_MAX,
                        longest_deviation_max=LONGEST_DEVIATION_MAX,
                        deviation_threshold=DEVIATION_THRESHOLD,
                        epsilon=SPEED_EPSILON, speed_not_a_gate_for_stairs=True,
                        all_cases_raw_float32_npz=True, failure_in_denominator=True,
                        f0_lateral_max=F0_LATERAL_MAX, f0_heading_max_radians=F0_HEADING_MAX,
                        raw_route_evidence=['route_error', 'local_y', 'yaw']),
        ppo=deepcopy(PPO_CONTRACT),
        reward_state='new_episode_no_half_trajectory',
        resume_mode='new_episode',
        recovery=dict(completed='full48rollout_and20Adam', unknown_tail='recovery_blocked',
                      partial_adam='recovery_blocked', new_worker_dirs='monotonic_attempt',
                      same_update_latest='ledger_verified_complete_boundary',
                      rejects=['r1_schema_6', 'schema_le_6_or_unlabelled',
                               'v05_smoke_state_v1', SMOKE_CONTAINER,
                               'r1_initialization_container'],
                      model_keys='exact_state_dict_strict_true'),
        monitoring=dict(group='v05_r2', heartbeat_kind='v05_r2_heartbeat', panel_count=8,
                        heartbeat_seconds=30, unconfirmed_seconds=120,
                        pause_reason='monitoring_blocked', single_writer=True,
                        route_namespace='route/*', route_panel_group='运动/接触',
                        extra_panel_groups=0),
        behavior_sha256=hashes,
        pending_source_entries=pending,
        source_closure_complete=not pending,
    )


def contract_sha256(value=None):
    return hashlib.sha256(canonical_json(contract() if value is None else value)
                          .encode('utf-8')).hexdigest()
