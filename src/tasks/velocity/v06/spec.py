"""A2 ideal-heightmap stairs v0.6 single numeric source.

SDS: ``doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md`` revision 1.0
(SHA256 ``1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60``).
Execution contract: ``logs/a2_ideal_stairs_v06/implementation/coordination/
W0_FREEZE.md`` (frozen module interfaces, section 4.1).

This module owns **every** numerical constant of the v0.6 line: identity,
observation geometry, the 22-entry real terrain table, phase budgets and gate
speeds, the reward weight table, the PPO contract, the curriculum constants, the
D1 seed offsets, the gate thresholds and the resource ledger.  Every other v06
module must import these values; no other module may re-declare an SDS number
(W0_FREEZE section 4).  The module imports only the standard library, so
``curriculum.py``/``rewards.py``/``runner.py`` can import it without torch, and
``tests/test_v06_spec.py`` is the guard that each number still equals the SDS.

Terrain table (SDS 5.2): ``tile_id = 0`` is flat, ``T[k]`` is ``tile_id = k+1``.
``T0..T12`` are the 6-step series ``h = 0.02 + 0.13*j/12`` (``j = 0..12``),
``T13..T15`` are ``0.16/0.18/0.20``, ``T16`` is ``D25``, ``T17`` is ``D20``,
``T18..T20`` are ``L12/L20/L40``.  All stairs are 0.80 m wide, 0.01 m thin
treads with open risers/sides and a 2.0 m top platform.  The v0.6 world layout is its own explicit translation: local first-tread edge
is ``x = 0``, local ``y = 0`` is the tread centre line, local ``z = 0`` is the
ground plane, the top platform starts at local ``x = N*step_depth``, and the 22
tile frames are placed on a single row with
``frame_origin_world = (0, tile_id*8.0, 0)``.  ``valid_bounds_local`` is the
tile-cell query bound ``(-2, N*step_depth + platform_length, -2, 2)`` for stairs
and ``(-2, 22, -4, 4)`` for flat.  This is **not** a claim of numerical identity
with the old v0.2 atlas (its frame origin x is ``-10`` and its y has a whole-row
offset); only a consistent translation and the same local geometry/task are
approved (Q001-3).  These layout/bounds numbers are declared here because SDS
7.2 requires the fields but does not print their values; see
``coordination/QA_spec_rewards.md``.
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
from typing import Any, Mapping

# --------------------------------------------------------------- identities
CODE_VERSION = '0.6.0'
RECIPE_VERSION = '0.6-go2refs-r1'
SDS_REVISION = '1.0'
SCHEMA_VERSION = 8
ARCHITECTURE = 'v06-route-state-v1'
TASK = 'Unitree-A2-IdealHeightMap-Stairs-V06'
TASK_ID = TASK
CHECKPOINT_SCHEMA = SCHEMA_VERSION
PROTOCOL_ID = 'v06-go2refs-climb-first-v1'
LOG_ROOT = 'logs/a2_ideal_stairs_v06'
TRAINING_MODE = 'train'
TRAINING_SEED = 42
ROOT = Path(__file__).resolve().parents[4]

# ------------------------------------------------------- frozen documents
SDS_PATH = 'doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md'
SDS_SHA256 = '1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60'
W0_FREEZE_PATH = 'logs/a2_ideal_stairs_v06/implementation/coordination/W0_FREEZE.md'
W0_FREEZE_SHA256 = 'd1b51bc771e216e69eefd6da8ad834e6f1bb983145eb8c19d6297d5480a3c1f0'
Q001_DECISION_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                      'Q001_v06_decision.md')
Q001_DECISION_SHA256 = '5f3f2aa397fb5cf5ce286788e76384e8b19914c758e0ae29e50a67aa4d71a4b1'
Q002_DECISION_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                      'Q002_v06_decision.md')
Q002_DECISION_SHA256 = '3ebc7441f9be457cb135846366fe23926d02ec69e2232e0502d6c318ce5ccd52'
Q003_DECISION_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                      'Q003_v06_decision.md')
Q003_DECISION_SHA256 = '45129da87b905ab8bb7c2e3cb3b246be15c1ededea65c27fc5682d8a9fa7fa16'
Q004_DECISION_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                      'Q004_v06_decision.md')
Q004_DECISION_SHA256 = '768dc38a53b50effc5738785b3f8344820f0c7ef41abe2e5a08e6484458efe07'
#: Q003: the single forward-distance definition consumed by env/speed/evaluation.
FORWARD_DISTANCE_DEFINITION = 'net_displacement_pre_reset_500_steps'
W0_ADDENDUM_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                    'W0_FREEZE_ADDENDUM_1.md')
W0_ADDENDUM_SHA256 = 'b7553319abeb754b3a2838215e9110713435533b336af068ac89967f6b9c59ef'
W0_ADDENDUM2_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                     'W0_FREEZE_ADDENDUM_2.md')
W0_ADDENDUM2_SHA256 = 'c1dc2a8a092c086d876f43b41b5f6ebea7f68bdf24d93be5a4b045be18878dfc'
#: Effective contract = SDS + Q001 + Q002 decisions (W0 addendum 1/2; Q002 wins
#: over Q001, which wins over SDS on conflict).  Old hashes are kept unchanged;
#: the decisions are additive frozen inputs bound by W0_MANIFEST_v3.json.
FROZEN_DOCUMENTS = (
    ('sds', SDS_PATH, SDS_SHA256),
    ('w0_freeze', W0_FREEZE_PATH, W0_FREEZE_SHA256),
    ('q001_decision', Q001_DECISION_PATH, Q001_DECISION_SHA256),
    ('q002_decision', Q002_DECISION_PATH, Q002_DECISION_SHA256),
    ('w0_addendum_1', W0_ADDENDUM_PATH, W0_ADDENDUM_SHA256),
    ('w0_addendum_2', W0_ADDENDUM2_PATH, W0_ADDENDUM2_SHA256),
)

#: Q003/Q004 supplement the frozen contract without changing the six original
#: W0 frozen-document slots (kept stable for the W0 manifest/test contract).
SUPPLEMENTARY_DOCUMENTS = (
    ('q003_decision', Q003_DECISION_PATH, Q003_DECISION_SHA256),
    ('q004_decision', Q004_DECISION_PATH, Q004_DECISION_SHA256),
)

# ------------------------------------------------------- approved ETH source
INIT_PATH = '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt'
INIT_SHA256 = '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722'

# ------------------------------------------------------------------- timing
CONTROL_DT = .02
PHYSICS_DT = .005
DECIMATION = 4
PHYSICS_SUBSTEPS_PER_CONTROL_STEP = DECIMATION
MAP_HZ = 10
ACTION_SCALE = .25

# -------------------------------------------------------------- observation
N_JOINTS = 12
ACTION_DIM = N_JOINTS
PROPRIO_DIM = 45
HISTORY_LEN = 5
HISTORY_SHAPE = (HISTORY_LEN, PROPRIO_DIM)
HEIGHTMAP_SHAPE = (3, 48, 32)
ROUTE_DIM = 3
ROUTE_ERROR_DIM = ROUTE_DIM
ROUTE_ERROR_KEY = 'route_error'
ROUTE_LATERAL_SCALE = .20
ROUTE_LATERAL_CLIP = 4.
ROUTE_LATERAL_SATURATION_M = ROUTE_LATERAL_SCALE * ROUTE_LATERAL_CLIP
ROUTE_YAW_AXIS = 0.
CRITIC_WIDTH = 211
PRIVILEGED_DIM = 35
ETH_ACTOR_DIM = 45
CRITIC_NORMALIZER_DIMS = 80
CRITIC_HEIGHTMAP_DIMS = 64
CRITIC_PRIVILEGED_BLOCK_DIMS = 64
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
    lateral_scale_m=ROUTE_LATERAL_SCALE,
    lateral_clip=ROUTE_LATERAL_CLIP,
    lateral_saturation_m=ROUTE_LATERAL_SATURATION_M,
    normalizer='none_fixed_scale',
    noise='none',
    flat='N == 0 uses the same three components',
    missing='reject, never zero fill',
    deployment_label='ideal_route_state',
)
FOOT_ORDER = ('FL', 'FR', 'RL', 'RR')
JOINT_ORDER = tuple(f'{leg}_{joint}' for leg in FOOT_ORDER
                    for joint in ('hip', 'thigh', 'calf'))

# ------------------------------------------- existing A2 default pose/limits
# SDS 3.2 uses ``q0`` ("the existing A2 default joint angles") and the R2 soft
# joint limits.  q0 is read in FL/FR/RL/RR x (hip, thigh, calf) order from the
# existing A2 asset configuration:
# ``src/assets/robots/unitree_a2/a2_constants.py`` INIT_STATE
# (thigh 0.9, calf -1.8, R_hip 0.1, L_hip -0.1).
# Soft limits follow the existing mjlab semantics exactly (Q001-4):
#   m = (lo+hi)/2, r = (hi-lo)/2, soft = (m - 0.9*r, m + 0.9*r)
# where (lo,hi) are the XML hard ranges and the factor is the existing
# ``soft_joint_pos_limit_factor = 0.9``.  It is NOT ``hard_endpoint * 0.9``.
# ``_soft_limits`` already implements the midpoint formula and must not change.
# They are declared here so ``spec.py`` remains the single numeric source.
A2_DEFAULT_JOINT_POS = (-.1, .9, -1.8, .1, .9, -1.8,
                        -.1, .9, -1.8, .1, .9, -1.8)
A2_HARD_JOINT_LIMITS = (
    (-1.01, 1.01), (-2.34, 3.15), (-2.77, -.54),   # FL hip/thigh/calf
    (-1.01, 1.01), (-2.34, 3.15), (-2.77, -.54),   # FR
    (-1.01, 1.01), (-1.56, 3.94), (-2.77, -.54),   # RL
    (-1.01, 1.01), (-1.56, 3.94), (-2.77, -.54),   # RR
)
SOFT_JOINT_LIMIT_FACTOR = .9


def _soft_limits(hard, factor=SOFT_JOINT_LIMIT_FACTOR):
    result = []
    for low, high in hard:
        center = .5 * (low + high)
        half = .5 * (high - low)
        result.append((center - factor * half, center + factor * half))
    return tuple(result)


A2_SOFT_JOINT_LIMITS = _soft_limits(A2_HARD_JOINT_LIMITS)

# ------------------------------------------------------------ support physics
FOOT_RADIUS = .032
SAFE_INSET = .042
SUPPORT_FORCE_GT = 5.
SUPPORT_DEBOUNCE = 2
NONFOOT_FORCE_GT = 10.
NONFOOT_PERSIST_TIME = .1
ATTITUDE_LIMIT_DEGREES = 70.
ATTITUDE_LIMIT_RADIANS = math.radians(ATTITUDE_LIMIT_DEGREES)

# ---------------------------------------------------------------- geometry
FLAT_TILE_ID = 0
FLAT_ALIAS = 'flat'
STAIR_WIDTH = .80
TREAD_THICKNESS = .01
PLATFORM_LENGTH = 2.0
#: v0.6 world-layout constants used by every v0.6 consumer (see docstring).
ATLAS_TILE_SPACING = 8.0
ATLAS_ORIGIN_X = 0.0
ATLAS_ORIGIN_Z = 0.0
FLAT_QUERY_BOUNDS_LOCAL = (-2.0, 22.0, -4.0, 4.0)
STAIR_QUERY_X_MIN = -2.0
STAIR_QUERY_Y_HALF = 2.0


@dataclass(frozen=True)
class TerrainSpec:
    tile_id: int
    lesson_id: str
    N: int
    h: float
    step_depth: float
    width: float
    tread_thickness: float
    platform_length: float
    frame_origin_world: tuple[float, float, float]
    valid_bounds_local: tuple[float, float, float, float]


#: SDS 5.2 H00..H15 heights.  ``0.02 + 0.13*j/12`` for ``j = 0..12``.
STEP_HEIGHTS_H00_H15 = tuple(.02 + .13 * j / 12. for j in range(13)) + (.16, .18, .20)
STEP_DEPTH_H = .30
STEP_DEPTH_D25 = .25
STEP_DEPTH_D20 = .20
#: (lesson_id, N, h, step_depth) in T order.
TERRAIN_ROWS = tuple(
    (f'H{j:02d}', 6, STEP_HEIGHTS_H00_H15[j], STEP_DEPTH_H) for j in range(16)
) + (
    ('D25', 6, .20, STEP_DEPTH_D25),
    ('D20', 6, .20, STEP_DEPTH_D20),
    ('L12', 12, .20, STEP_DEPTH_D20),
    ('L20', 20, .20, STEP_DEPTH_D20),
    ('L40', 40, .20, STEP_DEPTH_D20),
)


def _terrain_row(tile_id, lesson_id, N, h, step_depth):
    origin = (ATLAS_ORIGIN_X, tile_id * ATLAS_TILE_SPACING, ATLAS_ORIGIN_Z)
    bounds = (STAIR_QUERY_X_MIN, N * step_depth + PLATFORM_LENGTH,
              -STAIR_QUERY_Y_HALF, STAIR_QUERY_Y_HALF)
    return TerrainSpec(tile_id=tile_id, lesson_id=lesson_id, N=N, h=h,
                       step_depth=step_depth, width=STAIR_WIDTH,
                       tread_thickness=TREAD_THICKNESS,
                       platform_length=PLATFORM_LENGTH,
                       frame_origin_world=origin, valid_bounds_local=bounds)


TERRAIN_TABLE = (
    TerrainSpec(tile_id=FLAT_TILE_ID, lesson_id='FLAT', N=0, h=0.,
                step_depth=STEP_DEPTH_D20, width=ATLAS_TILE_SPACING,
                tread_thickness=TREAD_THICKNESS, platform_length=PLATFORM_LENGTH,
                frame_origin_world=(ATLAS_ORIGIN_X, 0., ATLAS_ORIGIN_Z),
                valid_bounds_local=FLAT_QUERY_BOUNDS_LOCAL),
) + tuple(_terrain_row(index + 1, *row) for index, row in enumerate(TERRAIN_ROWS))
TERRAIN_BY_TILE = {entry.tile_id: entry for entry in TERRAIN_TABLE}
TERRAIN_BY_LESSON = {entry.lesson_id: entry for entry in TERRAIN_TABLE}
TERRAIN_TABLE_LESSONS = tuple(entry.lesson_id for entry in TERRAIN_TABLE)
T_TILE_COUNT = len(TERRAIN_ROWS)  # 21: T0..T20
FRONTIER_TILE_COUNT = T_TILE_COUNT
FINAL_TILE_ID = T_TILE_COUNT  # T20 -> tile_id 21 (L40)
FLAT_LESSON_ID = 'FLAT'
LESSON_ALIASES = {'flat': FLAT_TILE_ID, 'FLAT': FLAT_TILE_ID, 'F0': FLAT_TILE_ID}


def terrain_spec(tile_id):
    """Return the immutable :class:`TerrainSpec` for ``tile_id`` (0..21)."""
    if isinstance(tile_id, bool) or not isinstance(tile_id, Integral):
        raise ValueError(f'tile_id must be an integer, got {tile_id!r}')
    tile_id = int(tile_id)
    if tile_id not in TERRAIN_BY_TILE:
        raise ValueError(f'tile_id {tile_id} outside v0.6 terrain table [0, {T_TILE_COUNT}]')
    return TERRAIN_BY_TILE[tile_id]


def tile_id_for(lesson_id):
    """Return the tile id for a lesson name (``H00``..``L40`` and the flat alias)."""
    if not isinstance(lesson_id, str):
        raise ValueError('lesson_id must be a string')
    if lesson_id in LESSON_ALIASES:
        return LESSON_ALIASES[lesson_id]
    if lesson_id not in TERRAIN_BY_LESSON:
        raise ValueError(f'unknown v0.6 lesson_id: {lesson_id!r}')
    return TERRAIN_BY_LESSON[lesson_id].tile_id


# --------------------------------------------------------------- phases (5.1)
@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    budget: int
    substages: tuple[str, ...]
    substage_budgets: tuple[int, ...]
    gate_speeds: tuple[float, ...]
    reward_speeds: tuple[float, float]


GATE_SPEEDS_F0_H_D_L = (.3, .5, .8)
GATE_SPEEDS_V_R = (.2, .4, .6)
PHASES = (
    PhaseSpec('F0', 1000, (), (), GATE_SPEEDS_F0_H_D_L, (.3, .8)),
    PhaseSpec('H', 12000, (), (), GATE_SPEEDS_F0_H_D_L, (.3, .8)),
    PhaseSpec('D', 3000, ('D25', 'D20'), (1500, 1500), GATE_SPEEDS_F0_H_D_L, (.3, .8)),
    PhaseSpec('L', 6000, ('L12', 'L20', 'L40'), (1500, 1500, 3000),
              GATE_SPEEDS_F0_H_D_L, (.3, .8)),
    PhaseSpec('V', 4000, (), (), GATE_SPEEDS_V_R, (.2, .6)),
    PhaseSpec('R', 4000, (), (), GATE_SPEEDS_V_R, (.2, .6)),
)
PHASE_ORDER = tuple(phase.phase for phase in PHASES)
PHASE_BY_ID = {phase.phase: phase for phase in PHASES}
D_SUBSTAGES = PHASE_BY_ID['D'].substages
D_SUBSTAGE_BUDGETS = PHASE_BY_ID['D'].substage_budgets
L_SUBSTAGES = PHASE_BY_ID['L'].substages
L_SUBSTAGE_BUDGETS = PHASE_BY_ID['L'].substage_budgets
PHASE_BUDGETS = {phase.phase: phase.budget for phase in PHASES}
TOTAL_BUDGET = sum(PHASE_BUDGETS.values())  # 30000


def phase_spec(phase):
    if not isinstance(phase, str) or phase not in PHASE_BY_ID:
        raise ValueError(f'unknown v0.6 phase: {phase!r}')
    return PHASE_BY_ID[phase]


def phase_budget(phase):
    return phase_spec(phase).budget


def total_budget():
    return TOTAL_BUDGET


# --------------------------------------------------------------- reward (3.2)
REWARD_TERMS = (
    'tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z', 'orientation_roll',
    'base_height', 'action_rate', 'similar_to_default', 'dof_acc', 'dof_vel',
    'ang_vel_xy', 'feet_air_time', 'foot_slip', 'foot_clearance',
    'joint_tracking', 'center', 'heading', 'joint_limits',
)
REWARD_WEIGHTS = {
    'tracking_lin_vel': 1.5,
    'tracking_ang_vel': .8,
    'lin_vel_z': -1.0,
    'orientation_roll': -5.0,
    'base_height': -.1,
    'action_rate': -.01,
    'similar_to_default': -.05,
    'dof_acc': -2.5e-7,
    'dof_vel': -5e-4,
    'ang_vel_xy': -.05,
    'feet_air_time': .2,
    'foot_slip': -.15,
    'foot_clearance': -.5,
    'joint_tracking': -.1,
    'center': -.5,
    'heading': -.5,
    'joint_limits': -10.0,
}
FORWARD_PROGRESS_PER_M = .4
SUCCESS_REWARD = 10.0
FAILURE_REWARD = -10.0
TIMEOUT_REWARD = 0.
FLAT_CLEARANCE_TARGET = .075
BASE_HEIGHT_TARGET = .40
LIN_VEL_Z_DEADBAND = .15
AIR_TIME_TARGET = .10
TRACKING_EXP_SCALE = .25
FOOT_CLEARANCE_MIN_TARGET = .17
FOOT_CLEARANCE_HEIGHT_MARGIN = .02
CENTER_LATERAL_SCALE = .20
CENTER_SATURATION = 4.0
ROLL_WEIGHT_TERM = 'orientation_roll'
REMOVED_REWARD_TERMS = (
    'speed_piecewise', 'overspeed', 'backward', 'side_speed', 'positive_gate',
    'footfall', 'swing', 'stall', 'pose', 'angular_momentum', 'edge',
    'soft_landing', 'nonfoot',
)


def clearance_target(step_h, *, flat):
    """SDS 3.3 foot-clearance target: flat ``0.075``, stairs ``max(0.17, h+0.02)``."""
    if not isinstance(flat, bool):
        raise ValueError('flat must be a bool')
    if flat:
        return FLAT_CLEARANCE_TARGET
    if isinstance(step_h, bool):
        raise ValueError('step_h must be a real number')
    try:
        height = float(step_h)
    except (TypeError, ValueError):
        raise ValueError('step_h must be a real number') from None
    if not math.isfinite(height) or height < 0.:
        raise ValueError('step_h must be finite and non-negative')
    return max(FOOT_CLEARANCE_MIN_TARGET, height + FOOT_CLEARANCE_HEIGHT_MARGIN)


# ------------------------------------------------------------------ PPO (4)
PPO_CONFIG = dict(
    num_envs=512,
    eval_num_envs=8,
    rollout=48,
    epochs=5,
    minibatches=4,
    adam_updates_per_iteration=5 * 4,
    learning_rate=3e-4,
    min_lr=1e-5,
    max_lr=3e-4,
    desired_kl=.01,
    kl_high=.02,
    kl_low=.005,
    kl_scale=1.5,
    gamma=.99,
    gae_lambda=.95,
    clip=.2,
    entropy_coef=.003,
    value_coef=1.0,
    clipped_value_loss=True,
    max_grad_norm=1.0,
    initial_std=.30,
    std_min=.05,
    std_max=1.00,
    adam_betas=(.9, .999),
    adam_eps=1e-8,
    weight_decay=0.,
    optimizer='single_adam_each_parameter_once',
    std_projection='after_each_adam_step',
    kl_early_exit=False,
    actor_freeze='none',
    automatic_rollback=False,
    advantage_normalization=True,
    timeout_bootstrap=True,
    control_samples_per_update=512 * 48,
    physics_substeps_per_update=512 * 48 * DECIMATION,
    grad_norm_groups=('actor', 'critic', 'fusion', 'std'),
    grad_norm_does_not_change_gradient=True,
    mean_activation='none_no_tanh_no_clip',
)
PPO_NUM_ENVS = PPO_CONFIG['num_envs']
PPO_EVAL_NUM_ENVS = PPO_CONFIG['eval_num_envs']
PPO_ROLLOUT = PPO_CONFIG['rollout']
PPO_EPOCHS = PPO_CONFIG['epochs']
PPO_MINIBATCHES = PPO_CONFIG['minibatches']
PPO_INITIAL_LR = PPO_CONFIG['learning_rate']
PPO_MIN_LR = PPO_CONFIG['min_lr']
PPO_MAX_LR = PPO_CONFIG['max_lr']
PPO_DESIRED_KL = PPO_CONFIG['desired_kl']
PPO_GAMMA = PPO_CONFIG['gamma']
PPO_LAMBDA = PPO_CONFIG['gae_lambda']
PPO_CLIP = PPO_CONFIG['clip']
PPO_ENTROPY = PPO_CONFIG['entropy_coef']
PPO_VALUE_COEF = PPO_CONFIG['value_coef']
PPO_MAX_GRAD_NORM = PPO_CONFIG['max_grad_norm']
PPO_CLIPPED_VALUE_LOSS = PPO_CONFIG['clipped_value_loss']
INITIAL_STD = PPO_CONFIG['initial_std']
STD_MIN = PPO_CONFIG['std_min']
STD_MAX = PPO_CONFIG['std_max']
TRAINING_NUM_ENVS = PPO_CONFIG['num_envs']
EVALUATION_NUM_ENVS = PPO_CONFIG['eval_num_envs']
PLAY_NUM_ENVS = 1

# ------------------------------------------------------------ curriculum (5)
CURRICULUM_WINDOW = 64
CURRICULUM_WINDOWS_PER_UPDATE = 1
EMA_DECAY = .97
EMA_NEW = .03
ADVANCE_S_MIN = .65
ADVANCE_Q_MIN = .45
ADVANCE_F_MAX = .35
ADVANCE_STREAK = 5
REGRESS_F_MIN = .40
REGRESS_S_MAX = .20
REGRESS_STREAK = 2
REGRESS_TICKS = 3
LEVEL_TICK_MIN = 0
LEVEL_TICK_MAX = 100
LEVEL_TICK_SCALE = 100
FRONTIER_ROW_COUNT = 15
COOLDOWN_CHECKS = 5
H00_INITIAL_LEVEL_TICK = 0
GATE_STREAK_REQUIRED = 2

ROLE_NAMES = ('flat', 'frontier', 'near', 'easy')
ROLE_COUNTS = {'flat': 102, 'frontier': 164, 'near': 123, 'easy': 123}
ROLE_COUNT_TOTAL = sum(ROLE_COUNTS.values())  # 512
ROLE_TIE_ORDER = ROLE_NAMES
SLOT_ALLOCATION = 'integer_largest_remainder_tie_flat_frontier_near_easy'
NEAR_OFFSETS = (-2, -1)
EASY_BACKOFF = 3
V_R_FRONTIER_TILE_ID = FINAL_TILE_ID
V_R_NEAR_TILES = ('L12', 'L20')
V_R_EASY_LAST_TILE_ID = tile_id_for('D20')  # T17

V_MIN_UPDATES = 1200
V_GATE_STREAK_START = 1000
R_MIN_UPDATES = 600
R_GATE_STREAK_START = 400
SUBSTAGE_ADVANCE_STREAK = 2
SUBSTAGE_REGRESS_STREAK = 2

COMMAND_INITIAL_VX = (.30, .80)
COMMAND_FINAL_VX = (.20, .60)
COMMAND_VY = 0.
COMMAND_WZ = 0.
V_RAMP_UPDATES = 1000

SPAWN_X = (-1.10, -.90)
SPAWN_Y = (-.05, .05)
SPAWN_YAW_DEGREES = (-5., 5.)
SPAWN_YAW_RADIANS = (math.radians(-5.), math.radians(5.))
R_SPAWN_X = (-1.20, -.80)
R_SPAWN_Y = (-.10, .10)
R_SPAWN_YAW_DEGREES = (-10., 10.)
R_SPAWN_YAW_RADIANS = (math.radians(-10.), math.radians(10.))
R_SPAWN_RAMP_UPDATES = 400

CURRICULUM = dict(
    window=CURRICULUM_WINDOW,
    windows_per_update=CURRICULUM_WINDOWS_PER_UPDATE,
    window_scope='frontier_role_only',
    window_order='episode_end_control_step_then_env_id',
    unfinished_windows='retained_never_dropped',
    metrics=dict(S='safe_summit_ratio', F='failure_ratio', T='timeout_ratio',
                 sum='S+F+T=1', Q='mean_per_episode_lin_vel_raw_exp_score_no_yaw_no_weight',
                 invalid='stop_on_nonfinite_or_out_of_range'),
    ema=dict(decay=EMA_DECAY, new=EMA_NEW, first_batch='ema=batch_metric',
             formula='ema=0.97*ema+0.03*batch_metric'),
    advance=dict(s_min=ADVANCE_S_MIN, q_min=ADVANCE_Q_MIN, f_max=ADVANCE_F_MAX,
                 streak=ADVANCE_STREAK, cooldown_required=True,
                 level_tick_delta=+1),
    regress=dict(f_min=REGRESS_F_MIN, s_max=REGRESS_S_MAX, streak=REGRESS_STREAK,
                 level_tick_delta=-REGRESS_TICKS, priority_over_advance=True,
                 cooldown_blocks=False),
    cooldown_checks=COOLDOWN_CHECKS,
    level_tick_range=(LEVEL_TICK_MIN, LEVEL_TICK_MAX),
    level_scale=LEVEL_TICK_SCALE,
    frontier_rows=FRONTIER_ROW_COUNT,
    level_formula='level=level_tick/100',
    frontier_row_formula='frontier_row=floor(15*level_tick/100)',
    initial_level_tick=H00_INITIAL_LEVEL_TICK,
    h00_initial_gate_streak=GATE_STREAK_REQUIRED,
    h15_final_gate_streak=GATE_STREAK_REQUIRED,
    h15_entry='level_tick=100 and H15 two complete gates',
    generation=dict(reset_on_row_change=True,
                    clears=('ema', 'fifo', 'ready_streak', 'hard_streak'),
                    cooldown=COOLDOWN_CHECKS, old_generation='log_only_not_counted'),
    terrain_commit='full_update_boundary_only',
    reset_on_commit=('env', 'contacts', 'map_history', 'reward_diff_state'),
    roles=dict(counts=dict(ROLE_COUNTS), total=ROLE_COUNT_TOTAL,
               tie_order=ROLE_TIE_ORDER, allocation=SLOT_ALLOCATION),
    sampling=dict(episode_reset_only=True, rng='independent_resumable_training_rng',
                  regress_truncates=True),
    near=dict(offsets=NEAR_OFFSETS, k_below_2='T[k]', equal_probability=True,
              v_r_tiles=V_R_NEAR_TILES),
    easy=dict(range='T[0]..T[max(k-3,0)]', backoff=EASY_BACKOFF,
              equal_probability=True, v_r_last_tile_id=V_R_EASY_LAST_TILE_ID),
    v_r_frontier_tile_id=V_R_FRONTIER_TILE_ID,
    phase_budgets=dict(PHASE_BUDGETS),
    substages=dict(
        D=dict(names=D_SUBSTAGES, budgets=D_SUBSTAGE_BUDGETS),
        L=dict(names=L_SUBSTAGES, budgets=L_SUBSTAGE_BUDGETS),
        advance_streak=SUBSTAGE_ADVANCE_STREAK,
        regress_streak=SUBSTAGE_REGRESS_STREAK,
        hard_condition='same_as_H',
        no_cross_phase_regress=True, lowest_substage_holds=True,
        regress_new_generation=True, old_pass_invalid=True,
        budgets_cumulative=True, regress_keeps_spent=True,
        exhausted_return_stops=True, no_borrowing=True,
    ),
    v=dict(min_updates=V_MIN_UPDATES, gate_streak_start=V_GATE_STREAK_START,
           gate_speeds=GATE_SPEEDS_V_R, advance_streak=GATE_STREAK_REQUIRED,
           gates_use_final_speeds_from_first_day=True),
    r=dict(min_updates=R_MIN_UPDATES, gate_streak_start=R_GATE_STREAK_START,
           gate_speeds=GATE_SPEEDS_V_R, advance_streak=GATE_STREAK_REQUIRED,
           gates_use_final_spawn_from_first_day=True,
           flat_keeps_small_spawn=True),
    command=dict(fixed_within_episode=True, resample_seconds=None,
                 initial_vx=COMMAND_INITIAL_VX, final_vx=COMMAND_FINAL_VX,
                 vy=COMMAND_VY, wz=COMMAND_WZ,
                 v_ramp_updates=V_RAMP_UPDATES,
                 v_ramp='p=min(stage_iteration/1000,1);vx_min=0.3-0.1p;vx_max=0.8-0.2p',
                 zero_command_environments=False),
    spawn=dict(x=SPAWN_X, y=SPAWN_Y, yaw_degrees=SPAWN_YAW_DEGREES,
               yaw_radians=SPAWN_YAW_RADIANS,
               r_x=R_SPAWN_X, r_y=R_SPAWN_Y, r_yaw_degrees=R_SPAWN_YAW_DEGREES,
               r_yaw_radians=R_SPAWN_YAW_RADIANS,
               r_ramp_updates=R_SPAWN_RAMP_UPDATES, r_ramp='linear',
               r_flat_keeps_small=True),
    final_update='execute_due_gates_then_judge_no_extra_budget',
)

# -------------------------------------------------------------- gates (6)
GATE_INTERVAL = 100
DIAGNOSTIC_INTERVAL = 50
SAVE_INTERVAL = 100
RETENTION_INTERVAL = 200
GATE_CASES = 20
GATE_PASS_MIN = 18
FINAL_CASES = 100
FINAL_PASS_MIN = 95
GATE_EVALUATION='deterministic_mean_action'
DIAGNOSTIC_CASES = 15
RETENTION_CASES = 15
RETENTION_CASES_PER_SPEED = 5
RETENTION_FAIL_STREAK = 2
RETENTION_FAIL_MIN = 12
MIN_LEVEL_FAILURE_STREAK = 4
MIN_LEVEL_FAILURE_F_MIN = .80
MIN_LEVEL_PHASE_UPDATES = 200
#: Q001-7: inherited-stall collapse needs two consecutive qualifying 15-case
#: diagnostics.  Scope is exactly the inherited stall collapse; it must not be
#: reused for any other diagnostic anomaly.  The flat/H00 retention regression
#: counters and the minimum-level 4-window counter stay independent.
DIAGNOSTIC_COLLAPSE_STREAK = 2

# ---------------------------------------------------- inherited stall diagnostic
#: Q001-7: inherited r2 stall diagnostic definition.  This is a *diagnostic only*
#: state machine: it never pays a reward and never terminates an episode.  v06
#: closes the old r2 stall payment, keeps the recognition/monitoring identical,
#: and ``env.py`` must read these constants from ``spec`` instead of hardcoding
#: them (SDS 7.1 single numeric source).  Names and values are inherited from
#: ``src/tasks/velocity/v05_r2/spec.py`` and must stay verbatim.
STALL_HISTORY_SAMPLES = 51
STALL_DX_FRACTION_PER_SECOND = .15
STALL_DX_THRESHOLD_SCALE = 1.0
STALL_HIGHWATER_DX = .03
STALL_CONSECUTIVE_SAMPLES = 50
STALL_ACTIVE_COMMAND = .1

F0_OBSERVE_STEPS = 500
F0_LATERAL_MAX = .20
F0_HEADING_MAX_DEGREES = 10.
F0_HEADING_MAX_RADIANS = math.radians(F0_HEADING_MAX_DEGREES)
F0_PROGRESS_SECONDS = 10.
F0_PROGRESS_FRACTION = .5
F0_EPISODE_STEPS = 1500

# success (6.2): real top-platform support
SUCCESS_RECENT_SUPPORT_STEPS = 50
SUCCESS_SINGLE_FOOT_FRAMES = 2
SUCCESS_MIN_SUPPORT_FEET = 2
SUCCESS_BODY_PLATFORM_MARGIN = .8
SUCCESS_LATERAL_MAX = .20
SUCCESS_FOOT_SAFE_INSET = SAFE_INSET
SUCCESS_CONSECUTIVE_FRAMES = 5

# timeout (6.2) and failure masks
STAIR_TIMEOUT_BASE_M = 2.2
STAIR_TIMEOUT_SECONDS = 10.
TIMEOUT_FORMULA = 'ceil((2*(2.2+N*sqrt(h^2+d^2))/c+10)/0.02)'
FAILURE_BODY_CONTACT = True
FAILURE_NONFOOT_PERSIST_TIME = NONFOOT_PERSIST_TIME
FAILURE_ATTITUDE_LIMIT_DEGREES = ATTITUDE_LIMIT_DEGREES
FAILURE_LATERAL_BOUNDS = True
FAILURE_FALL = True
FAILURE_CONTACT_OVERFLOW = True

#: Q002-2: fall-through threshold (metres) against the **expected tread**, never
#: against the latest foot contact, the four-foot average support or the spawn
#: height.  With ``d = tile.step_depth``, ``h = tile.h`` and ``local_x/local_z``
#: in the current tile frame::
#:
#:     region        = (N > 0) & (local_x >= 0)
#:     z_expected    = min(max(floor(local_x / d) + 1, 0), N) * h
#:     below_support = region & (local_z < z_expected - FALL_DROP)   # strict <
#:
#: On the top platform ``z_expected = N*h``.  Flat tiles (N == 0) and the
#: approach region ``local_x < 0`` disable this check; equality never triggers.
#: ``events.py``/``env.py`` must read this constant directly (no getattr default).
FALL_DROP = 0.50

# ------------------------------------------------------- stop/lifecycle states
#: Q002-4: the eight training stop states, in frozen order.
STOP_STATES = (
    'behavior_regression',
    'behavior_collapse',
    'curriculum_stalled',
    'budget_exhausted',
    'gate_failed',
    'user_requested_stop',
    'monitoring_blocked',
    'resource_insufficient',
)
#: Q002-4: lifecycle states; ``complete`` only means the final 300-case test
#: passed (95 per speed), never curriculum/worker completion.
LIFECYCLE_STATES = ('running', 'ready_for_final_test', 'complete')
ALL_TRAINING_STATES = LIFECYCLE_STATES + STOP_STATES
COMPLETE_MEANING = 'final_300_case_test_passed_only'
PILOT_NOT_APPLICABLE_IS_TERMINAL = False
N_ONLY_CURRENT_STEP_DEPTH = True

# ------------------------------------------------------------- D1 seeds (8.3)
VALIDATION_SEED_OFFSET = 200300
DIAGNOSTIC_SEED_OFFSET = 500300
PERTURBED_SEED_OFFSET = 300300
FINAL_TEST_SEED = 400342
SEED_OFFSET_KINDS = ('validation', 'diagnostic', 'perturbed', 'final')
SEED_OFFSETS = {
    'validation': VALIDATION_SEED_OFFSET,
    'diagnostic': DIAGNOSTIC_SEED_OFFSET,
    'perturbed': PERTURBED_SEED_OFFSET,
    'final': FINAL_TEST_SEED,
}

# ------------------------------------------------------------ ledger (8.1/8.5)
CONTROL_SAMPLES_PER_UPDATE = PPO_CONFIG['control_samples_per_update']  # 24576
PHYSICS_SUBSTEPS_PER_UPDATE = PPO_CONFIG['physics_substeps_per_update']  # 98304
ADAM_UPDATES_PER_ITERATION = PPO_CONFIG['adam_updates_per_iteration']  # 20
TOTAL_CONTROL_SAMPLES = TOTAL_BUDGET * CONTROL_SAMPLES_PER_UPDATE
TOTAL_PHYSICS_SUBSTEPS = TOTAL_BUDGET * PHYSICS_SUBSTEPS_PER_UPDATE
TOTAL_ADAM_UPDATES = TOTAL_BUDGET * ADAM_UPDATES_PER_ITERATION
RESOURCE_LIMITS = dict(gpu_mib=7168, process_mib=5120, host_free_mib=4096,
                       disk_free_gib=10, single_gpu_worker=True)
SMOKE_SEED = 42
SMOKE_ENVS = 8
SMOKE_TILE = 'H00'
SMOKE_UPDATES = 2
SMOKE_CONTROL_SAMPLES = SMOKE_ENVS * PPO_ROLLOUT * SMOKE_UPDATES
SMOKE_PHYSICS_SUBSTEPS = SMOKE_CONTROL_SAMPLES * DECIMATION
SMOKE_ADAM_UPDATES = SMOKE_UPDATES * ADAM_UPDATES_PER_ITERATION
PROBE_ENVS = 512
PROBE_EVAL_ENVS = 8
PROBE_WARMUP_UPDATES = 5
PROBE_MEASURE_UPDATES = 50
PROBE_RECOVERY_UPDATES = 1
PROBE_UPDATES = PROBE_WARMUP_UPDATES + PROBE_MEASURE_UPDATES + PROBE_RECOVERY_UPDATES
PROBE_CONTROL_SAMPLES = (PROBE_WARMUP_UPDATES + PROBE_MEASURE_UPDATES
                         + PROBE_RECOVERY_UPDATES) * CONTROL_SAMPLES_PER_UPDATE
SMOKE_CONTAINER = 'v06-smoke-state-v1'
PROBE_CONTAINER = 'v06-probe-state-v1'
SMOKE_FORMAL_RESUMABLE = False
PROBE_FORMAL_RESUMABLE = False

# ------------------------------------------------------ recipe CLI (11.2)
CLI_DEFAULTS = dict(
    mode='train',
    execute=True,
    recipe=RECIPE_VERSION,
    seed=TRAINING_SEED,
    device='cuda:0',
    num_envs=512,
    eval_num_envs=8,
    rollout=48,
    epochs=5,
    minibatches=4,
    learning_rate=.0003,
    gamma=.99,
    entropy_coef=.003,
    initial_std=.30,
    std_min=.05,
    std_max=1.00,
    initial_vx_min=.30,
    initial_vx_max=.80,
    final_vx_min=.20,
    final_vx_max=.60,
    vy=.0,
    wz=.0,
    max_updates=30000,
    phase_budgets='F0=1000,H=12000,D=3000,L=6000,V=4000,R=4000',
    save_interval=100,
    diagnostic_interval=50,
    gate_interval=100,
    init_checkpoint=INIT_PATH,
    init_sha256=INIT_SHA256,
    acceptance_file='logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md',
    output_dir='logs/a2_ideal_stairs_v06/train_seed42_trial01',
)


def _parse_phase_budgets(value):
    if isinstance(value, str):
        result = {}
        for item in value.split(','):
            name, _, budget = item.partition('=')
            result[name.strip()] = int(budget)
        return result
    if isinstance(value, Mapping):
        return {str(name): int(budget) for name, budget in value.items()}
    raise ValueError(f'phase_budgets must be a string or mapping, got {value!r}')


def _assert_cli_value(name, value, expected):
    if name == 'execute':
        if not isinstance(value, bool):
            raise ValueError('execute must be a bool')
        return value
    if name == 'phase_budgets':
        parsed = _parse_phase_budgets(value)
        expected = _parse_phase_budgets(expected)
        if parsed != expected:
            raise ValueError(f'recipe flag phase_budgets must be {expected!r}, got {parsed!r}')
        return parsed
    if isinstance(expected, bool):
        if value is not expected:
            raise ValueError(f'recipe flag {name} must be {expected!r}, got {value!r}')
        return value
    if isinstance(expected, str):
        if not isinstance(value, str) or value != expected:
            raise ValueError(f'recipe flag {name} must be {expected!r}, got {value!r}')
        return value
    if isinstance(expected, int):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) != expected:
            raise ValueError(f'recipe flag {name} must be {expected!r}, got {value!r}')
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'recipe flag {name} must be {expected!r}, got {value!r}') from None
    if not math.isfinite(number) or abs(number - expected) > 1e-12:
        raise ValueError(f'recipe flag {name} must be {expected!r}, got {value!r}')
    return number


def recipe_config(**cli):
    """Assert the SDS 11.2 CLI flags and return the resolved recipe record.

    Only assertion, never a way to override the SDS: every supplied flag must
    equal the frozen value, an unknown flag is an error, and the returned record
    is the canonical expected configuration.  ``execute`` is the one flag whose
    two values are both legal (``False`` is the SDS preflight mode); it is
    type-checked but not forced to ``True``.
    """
    resolved = deepcopy(CLI_DEFAULTS)
    for key, value in cli.items():
        name = str(key).replace('-', '_')
        if name not in CLI_DEFAULTS:
            raise ValueError(f'unknown SDS 11.2 recipe flag: {key!r}')
        resolved[name] = _assert_cli_value(name, value, CLI_DEFAULTS[name])
    resolved['phase_budgets'] = _parse_phase_budgets(resolved['phase_budgets'])
    return resolved


# ------------------------------------------------------- curriculum helpers
def curriculum_level(level_tick):
    """``level = level_tick / 100`` with an integer tick in ``[0,100]``."""
    _check_tick(level_tick)
    return int(level_tick) / LEVEL_TICK_SCALE


def curriculum_frontier_row(level_tick):
    """``frontier_row = floor(15 * level_tick / 100)`` (0..15)."""
    _check_tick(level_tick)
    return (FRONTIER_ROW_COUNT * int(level_tick)) // LEVEL_TICK_SCALE


def _check_tick(level_tick):
    if (isinstance(level_tick, bool) or not isinstance(level_tick, Integral)
            or not LEVEL_TICK_MIN <= int(level_tick) <= LEVEL_TICK_MAX):
        raise ValueError(f'level_tick must be an integer in '
                         f'[{LEVEL_TICK_MIN},{LEVEL_TICK_MAX}], got {level_tick!r}')
    return int(level_tick)


def curriculum_ema(previous, batch_metric):
    """First batch initialises EMA; afterwards ``0.97*EMA + 0.03*batch``."""
    try:
        value = float(batch_metric)
    except (TypeError, ValueError):
        raise ValueError('batch_metric must be a real number') from None
    if not math.isfinite(value):
        raise ValueError('batch_metric must be finite')
    if previous is None:
        return value
    try:
        old = float(previous)
    except (TypeError, ValueError):
        raise ValueError('previous EMA must be a real number') from None
    if not math.isfinite(old):
        raise ValueError('previous EMA must be finite')
    return EMA_DECAY * old + EMA_NEW * value


def curriculum_transition(level_tick, s_ema, q_ema, f_ema, ready_streak=0,
                          hard_streak=0, cooldown=0):
    """One SDS 5.5 check: regress first, then advance, then cooldown tick.

    Returns a dict with ``event`` in ``{'regress','advance','hold'}``, the new
    ``level_tick``/``ready_streak``/``hard_streak``/``cooldown`` and the
    ``level_changed`` flag.  A level change sets ``cooldown=5``; a check without
    a level change decrements the cooldown to a floor of zero.  Regress wins over
    advance in the same check and is never blocked by cooldown (SDS 5.5).
    """
    tick = _check_tick(level_tick)
    for name, value in (('s_ema', s_ema), ('q_ema', q_ema), ('f_ema', f_ema)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be a real number') from None
        if not math.isfinite(number):
            raise ValueError(f'{name} must be finite')
    for name, value in (('ready_streak', ready_streak), ('hard_streak', hard_streak),
                        ('cooldown', cooldown)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    s_ema, q_ema, f_ema = float(s_ema), float(q_ema), float(f_ema)
    hard = (f_ema >= REGRESS_F_MIN) or (s_ema <= REGRESS_S_MAX)
    hard_streak = int(hard_streak) + 1 if hard else 0
    ready = (s_ema >= ADVANCE_S_MIN and q_ema >= ADVANCE_Q_MIN
             and f_ema <= ADVANCE_F_MAX)
    ready_streak = int(ready_streak) + 1 if ready else 0
    event, changed, new_tick = 'hold', False, tick
    if hard_streak >= REGRESS_STREAK:
        new_tick = max(LEVEL_TICK_MIN, tick - REGRESS_TICKS)
        changed = new_tick != tick
        event = 'regress'
        hard_streak = 0
    elif ready_streak >= ADVANCE_STREAK and int(cooldown) == 0:
        new_tick = min(LEVEL_TICK_MAX, tick + 1)
        changed = new_tick != tick
        event = 'advance'
        ready_streak = 0
    new_cooldown = COOLDOWN_CHECKS if changed else max(0, int(cooldown) - 1)
    if changed:
        ready_streak = 0
        hard_streak = 0
    return dict(event=event, level_tick=new_tick, level_changed=changed,
                ready_streak=ready_streak, hard_streak=hard_streak,
                cooldown=new_cooldown,
                frontier_row=curriculum_frontier_row(new_tick))


def fifo_append(fifo, outcome, *, window=CURRICULUM_WINDOW):
    """Append one episode outcome, sorted by ``(end_step, env_id)``.

    ``outcome`` is a mapping with ``step`` and ``env_id``; the returned
    ``(fifo, ready)`` keeps the window FIFO sorted and ``ready`` is true once at
    least ``window`` outcomes wait.  Callers pop exactly one window per update.
    """
    if not isinstance(outcome, Mapping) or 'step' not in outcome or 'env_id' not in outcome:
        raise ValueError('outcome must be a mapping with step and env_id')
    if isinstance(window, bool) or not isinstance(window, Integral) or int(window) < 1:
        raise ValueError('window must be a positive integer')
    entries = list(fifo) + [dict(outcome)]
    entries.sort(key=lambda item: (int(item['step']), int(item['env_id'])))
    return entries, len(entries) >= int(window)


def fifo_pop_window(fifo, *, window=CURRICULUM_WINDOW):
    """Return ``(window_entries, remaining)``; never drops a partial remainder."""
    entries = list(fifo)
    size = int(window)
    return entries[:size], entries[size:]


def generation_starts(previous_row, row):
    """A frontier-row change starts a new generation (SDS 5.5)."""
    if isinstance(previous_row, bool) or isinstance(row, bool):
        raise ValueError('rows must be integers')
    return int(previous_row) != int(row)


def evaluation_seed(kind, seed):
    """Apply exactly one D1 offset to the raw training seed (SDS 8.3)."""
    if kind not in SEED_OFFSETS:
        raise ValueError(f'unknown seed kind: {kind!r}; expected one of {SEED_OFFSET_KINDS}')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    offset = SEED_OFFSETS[kind]
    if kind == 'final':
        return FINAL_TEST_SEED
    return offset + int(seed)


def evaluation_seeds(seed=TRAINING_SEED):
    return {kind: evaluation_seed(kind, seed) for kind in SEED_OFFSET_KINDS}


def episode_timeout_steps(N, h, step_depth, executed_c):
    """SDS 6.2 stair timeout ``ceil((2*(2.2+N*sqrt(h^2+d^2))/c+10)/0.02)``."""
    for name, value in (('N', N), ('h', h), ('step_depth', step_depth), ('executed_c', executed_c)):
        if isinstance(value, bool):
            raise ValueError(f'{name} must be a real number')
    count, height, depth, command = int(N), float(h), float(step_depth), float(executed_c)
    if count < 0 or height < 0 or depth <= 0 or command <= 0:
        raise ValueError('N/h/step_depth/executed_c outside valid range')
    if count == 0 or height == 0:
        return F0_EPISODE_STEPS
    inner = 2. * (STAIR_TIMEOUT_BASE_M + count * math.sqrt(height * height + depth * depth))
    return int(math.ceil((inner / command + STAIR_TIMEOUT_SECONDS) / CONTROL_DT))


def episode_step_limit(tile_id, executed_c):
    """Per-tile episode cap reading the one :func:`terrain_spec` entry."""
    spec = terrain_spec(tile_id)
    if spec.N == 0 or spec.h == 0:
        return F0_EPISODE_STEPS
    return episode_timeout_steps(spec.N, spec.h, spec.step_depth, executed_c)


# ------------------------------------------------------------- reward config
def _reward_config():
    return dict(
        version=RECIPE_VERSION,
        dt=CONTROL_DT,
        terms=list(REWARD_TERMS),
        weights=deepcopy(REWARD_WEIGHTS),
        composition=('dt*sum_i(weight_i*rate_i) + 0.4*(x_t-x_prev) '
                     '+ 10*I_success - 10*I_failure'),
        dt_multiplied_once=True,
        terminal_not_scaled_by_dt=True,
        total_reward_clamped=False,
        failure_precedence=True,
        timeout_reward=TIMEOUT_REWARD,
        forward_progress=dict(per_m=FORWARD_PROGRESS_PER_M,
                              formula='0.4*(local_x_t-local_x_prev)',
                              no_abs=True, no_highwater=True, reset_init='spawn_x'),
        success=SUCCESS_REWARD, failure=FAILURE_REWARD,
        base_height_target=BASE_HEIGHT_TARGET,
        clearance=dict(flat=FLAT_CLEARANCE_TARGET,
                       stair='max(0.17,h_step+0.02)',
                       min_stair_target=FOOT_CLEARANCE_MIN_TARGET,
                       height_margin=FOOT_CLEARANCE_HEIGHT_MARGIN,
                       foot_radius=FOOT_RADIUS,
                       surface='current foot x/y surface height, no smoothing, no next step'),
        tracking=dict(exp_scale=TRACKING_EXP_SCALE,
                      lin='exp(-((vx-c)^2+vy^2)/0.25)',
                      yaw='exp(-wz^2/0.25)'),
        lin_vel_z=dict(deadband=LIN_VEL_Z_DEADBAND, formula='max(abs(vz)-0.15,0)^2'),
        orientation=dict(roll_only=True, formula='g_y^2'),
        air_time=dict(target=AIR_TIME_TARGET,
                      formula='first_contact*(air_time_before_reset-0.10)',
                      settle_before_clear=True, first_contact_initializes_only=True,
                      standing_pays=False, partial_reset_scoped=True),
        slip=dict(formula='sum_f(C_f*||u_f||^2)', contact='filtered'),
        clearance_term=dict(formula='sum_f((1-C_f)*(h_f-h_target)^2*||u_f||)'),
        joint_tracking=dict(formula='sum((q_target_executed-q)^2)'),
        center=dict(lateral_scale=CENTER_LATERAL_SCALE, saturation=CENTER_SATURATION,
                    formula='min((local_y/0.20)^2,4)'),
        heading=dict(formula='1-cos(e_world)', yaw_axis=ROUTE_YAW_AXIS),
        joint_limits=dict(formula='sum(max(q_min_soft-q,0)+max(q-q_max_soft,0))',
                          soft_factor=SOFT_JOINT_LIMIT_FACTOR),
        default_joint_pos=list(A2_DEFAULT_JOINT_POS),
        soft_joint_limits=[list(pair) for pair in A2_SOFT_JOINT_LIMITS],
        removed_zero=dict.fromkeys(REMOVED_REWARD_TERMS, 0.),
        fall_drop=dict(
            drop=FALL_DROP,
            formula='z_expected=min(max(floor(local_x/d)+1,0),N)*h; '
                    'below=(N>0)&(local_x>=0)&(local_z<z_expected-FALL_DROP)',
            platform='z_expected=N*h',
            flat='disabled',
            approach='local_x<0 disabled',
            comparison='strict_less_than',
            reference='expected_tread_not_foot_contact_not_average_support',
        ),
        diagnostics=dict(
            collapse_streak=DIAGNOSTIC_COLLAPSE_STREAK,
            scope='inherited_stall_collapse_only',
            applies_to='inherited_r2_stall_definition_not_any_diagnostic_anomaly',
            independent_counters=('flat_retention_collapse',
                                  'h00_retention_collapse',
                                  'minimum_level_failure_streak'),
            qualification='current_generation_at_least_100_updates',
            missing_or_zero_denominator='evaluation_error_never_a_pass',
            stall=dict(
                definition='inherited_r2_stall_diagnostic_only_no_payment',
                history_samples=STALL_HISTORY_SAMPLES,
                dx_fraction_per_second=STALL_DX_FRACTION_PER_SECOND,
                dx_threshold_scale=STALL_DX_THRESHOLD_SCALE,
                highwater_dx=STALL_HIGHWATER_DX,
                consecutive_samples=STALL_CONSECUTIVE_SAMPLES,
                active_command=STALL_ACTIVE_COMMAND,
                reward_payment=False,
                termination=False,
            ),
        ),
        discount_reference=dict(gamma=PPO_GAMMA, note='terminal +10/-10 not dt-scaled'),
    )


REWARD_CONFIG = _reward_config()
REWARD_CONFIG['forward_distance_definition'] = FORWARD_DISTANCE_DEFINITION


def reward_config():
    return deepcopy(REWARD_CONFIG)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def reward_config_sha256():
    return hashlib.sha256(canonical_json(REWARD_CONFIG).encode('utf-8')).hexdigest()


# ------------------------------------------------------- source closure (7.3)
V06_DIRECTORY = 'src/tasks/velocity/v06'
PRODUCT_MODULES = ('__init__', 'spec', 'terrain', 'map', 'contacts', 'events',
                   'env_cfg', 'env', 'rewards', 'curriculum', 'model', 'ppo',
                   'runner', 'recovery', 'evaluation', 'speed', 'orchestration',
                   'ledger', 'telemetry', 'connection')
SCRIPT_ENTRIES = (
    'scripts/a2_v06_bootstrap.py',
    'scripts/train_a2_ideal_stairs_v06.py',
    'scripts/run_a2_ideal_stairs_v06.py',
    'scripts/evaluate_a2_ideal_stairs_v06.py',
    'scripts/play_a2_ideal_stairs_v06.py',
    'scripts/smoke_a2_ideal_stairs_v06.py',
    'scripts/test_a2_ideal_stairs_v06.py',
    'scripts/check_a2_ideal_stairs_v06_migration.py',
    'scripts/diagnostics/analyze_a2_v06_rewards.py',
)
SOURCE_ENTRIES = tuple(f'{V06_DIRECTORY}/{module}.py' for module in PRODUCT_MODULES) \
    + SCRIPT_ENTRIES


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def product_entries(root=ROOT):
    """Explicit v0.6 entry list: every existing v06 module plus existing entries."""
    root = Path(root).resolve()
    entries = []
    directory = root / V06_DIRECTORY
    if directory.is_dir():
        entries.extend(sorted(str(path.relative_to(root)) for path in directory.glob('*.py')))
    for relative in SOURCE_ENTRIES:
        if relative not in entries and (root / relative).is_file():
            entries.append(relative)
    return sorted(entries)


def pending_source_entries(root=ROOT):
    """Expected v0.6 product entries that do not exist yet (fail-closed report)."""
    root = Path(root).resolve()
    return [relative for relative in SOURCE_ENTRIES if not (root / relative).is_file()]


def source_hashes(root=ROOT, paths=None):
    """Hash explicit v0.6 entries and their static local import closure."""
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


def behavior_hashes(root=ROOT):
    """Per-production-file SHA256 map used as the v0.6 behavior contract."""
    return source_hashes(root=root)


def identity_block():
    return dict(code_version=CODE_VERSION, recipe_version=RECIPE_VERSION,
                schema_version=SCHEMA_VERSION, architecture=ARCHITECTURE,
                task=TASK, protocol=PROTOCOL_ID, log_root=LOG_ROOT,
                training_mode=TRAINING_MODE, training_seed=TRAINING_SEED)


def document_identity():
    return dict(sds_path=SDS_PATH, sds_sha256=SDS_SHA256,
                w0_freeze_path=W0_FREEZE_PATH, w0_freeze_sha256=W0_FREEZE_SHA256,
                q001_decision_path=Q001_DECISION_PATH,
                q001_decision_sha256=Q001_DECISION_SHA256,
                q002_decision_path=Q002_DECISION_PATH,
                q002_decision_sha256=Q002_DECISION_SHA256,
                w0_addendum_path=W0_ADDENDUM_PATH,
                w0_addendum_sha256=W0_ADDENDUM_SHA256,
                w0_addendum_2_path=W0_ADDENDUM2_PATH,
                w0_addendum_2_sha256=W0_ADDENDUM2_SHA256)


# ------------------------------------------------------------------ contract
def contract(**cli):
    """Frozen v0.6 protocol contract (asserts the SDS 11.2 CLI, changes nothing)."""
    resolved = recipe_config(**cli)
    hashes = behavior_hashes()
    pending = pending_source_entries()
    documents = document_identity()
    return dict(
        identity=identity_block(),
        identity_flat=dict(code_version=CODE_VERSION, recipe_version=RECIPE_VERSION,
                           checkpoint_schema=SCHEMA_VERSION, architecture=ARCHITECTURE,
                           task=TASK, protocol_id=PROTOCOL_ID, log_root=LOG_ROOT),
        document_identity=documents,
        sds_path=SDS_PATH, sds_sha256=SDS_SHA256,
        w0_freeze_path=W0_FREEZE_PATH, w0_freeze_sha256=W0_FREEZE_SHA256,
        q001_decision_path=Q001_DECISION_PATH,
        q001_decision_sha256=Q001_DECISION_SHA256,
        q002_decision_path=Q002_DECISION_PATH,
        q002_decision_sha256=Q002_DECISION_SHA256,
        w0_addendum_path=W0_ADDENDUM_PATH,
        w0_addendum_sha256=W0_ADDENDUM_SHA256,
        w0_addendum_2_path=W0_ADDENDUM2_PATH,
        w0_addendum_2_sha256=W0_ADDENDUM2_SHA256,
        diagnostics=dict(
            collapse_streak=DIAGNOSTIC_COLLAPSE_STREAK,
            scope='inherited_stall_collapse_only',
            stall=dict(
                definition='inherited_r2_stall_diagnostic_only_no_payment',
                history_samples=STALL_HISTORY_SAMPLES,
                dx_fraction_per_second=STALL_DX_FRACTION_PER_SECOND,
                dx_threshold_scale=STALL_DX_THRESHOLD_SCALE,
                highwater_dx=STALL_HIGHWATER_DX,
                consecutive_samples=STALL_CONSECUTIVE_SAMPLES,
                active_command=STALL_ACTIVE_COMMAND,
                reward_payment=False,
                termination=False,
            ),
        ),
        fall_drop=dict(
            drop=FALL_DROP,
            formula='z_expected=min(max(floor(local_x/d)+1,0),N)*h; '
                    'below=(N>0)&(local_x>=0)&(local_z<z_expected-FALL_DROP)',
            platform='z_expected=N*h', flat='disabled',
            approach='local_x<0 disabled', comparison='strict_less_than',
            reference='expected_tread_not_foot_contact_not_average_support'),
        states=dict(
            stop=list(STOP_STATES), lifecycle=list(LIFECYCLE_STATES),
            all=list(ALL_TRAINING_STATES), complete_meaning=COMPLETE_MEANING,
            pilot_not_applicable_is_terminal=PILOT_NOT_APPLICABLE_IS_TERMINAL),
        terrain=[asdict(entry) for entry in TERRAIN_TABLE],
        terrain_rows=[list(row) for row in TERRAIN_ROWS],
        phases=[asdict(phase) for phase in PHASES],
        phase_order=list(PHASE_ORDER),
        phase_budgets=dict(PHASE_BUDGETS),
        total_budget=TOTAL_BUDGET,
        ppo=deepcopy(PPO_CONFIG),
        curriculum=deepcopy(CURRICULUM),
        reward=reward_config(), reward_config_sha256=reward_config_sha256(),
        observation=dict(proprio=[PROPRIO_DIM], history=list(HISTORY_SHAPE),
                         heightmap=list(HEIGHTMAP_SHAPE), route_error=[ROUTE_ERROR_DIM],
                         privileged=[PRIVILEGED_DIM]),
        route_state=deepcopy(ROUTE_STATE),
        critic_width=CRITIC_WIDTH, privileged_dim=PRIVILEGED_DIM,
        foot_order=list(FOOT_ORDER), joint_order=list(JOINT_ORDER),
        timing=dict(control_dt=CONTROL_DT, physics_dt=PHYSICS_DT,
                    decimation=DECIMATION, map_hz=MAP_HZ, action_scale=ACTION_SCALE),
        init=dict(path=INIT_PATH, sha256=INIT_SHA256),
        gates=dict(interval=GATE_INTERVAL, cases=GATE_CASES, pass_min=GATE_PASS_MIN,
                   final_cases=FINAL_CASES, final_pass_min=FINAL_PASS_MIN,
                   f0_h_d_l_speeds=list(GATE_SPEEDS_F0_H_D_L),
                   v_r_speeds=list(GATE_SPEEDS_V_R),
                   diagnostic_interval=DIAGNOSTIC_INTERVAL,
                   save_interval=SAVE_INTERVAL, retention_interval=RETENTION_INTERVAL),
        seed_offsets=evaluation_seeds(TRAINING_SEED),
        ledger=dict(control_samples_per_update=CONTROL_SAMPLES_PER_UPDATE,
                    physics_substeps_per_update=PHYSICS_SUBSTEPS_PER_UPDATE,
                    adam_updates_per_iteration=ADAM_UPDATES_PER_ITERATION,
                    total_control_samples=TOTAL_CONTROL_SAMPLES,
                    total_physics_substeps=TOTAL_PHYSICS_SUBSTEPS,
                    total_adam_updates=TOTAL_ADAM_UPDATES,
                    smoke=dict(envs=SMOKE_ENVS, updates=SMOKE_UPDATES,
                               control_samples=SMOKE_CONTROL_SAMPLES,
                               physics_substeps=SMOKE_PHYSICS_SUBSTEPS,
                               adam_updates=SMOKE_ADAM_UPDATES),
                    probe=dict(envs=PROBE_ENVS, eval_envs=PROBE_EVAL_ENVS,
                               updates=PROBE_UPDATES,
                               control_samples=PROBE_CONTROL_SAMPLES)),
        resource_limits=deepcopy(RESOURCE_LIMITS),
        smoke_container=SMOKE_CONTAINER, probe_container=PROBE_CONTAINER,
        smoke_formal_resumable=SMOKE_FORMAL_RESUMABLE,
        probe_formal_resumable=PROBE_FORMAL_RESUMABLE,
        recipe=deepcopy(resolved),
        recipe_version=RECIPE_VERSION,
        behavior_sha256=hashes,
        pending_source_entries=pending,
        source_closure_complete=not pending,
    )


def contract_sha256(value=None):
    return hashlib.sha256(canonical_json(contract() if value is None else value)
                          .encode('utf-8')).hexdigest()


# ----------------------------------------------------------- frozen checks
def _check_frozen_tables():
    if (CODE_VERSION, RECIPE_VERSION, SDS_REVISION, SCHEMA_VERSION, ARCHITECTURE,
            TASK, PROTOCOL_ID) != ('0.6.0', '0.6-go2refs-r1', '1.0', 8,
                                   'v06-route-state-v1',
                                   'Unitree-A2-IdealHeightMap-Stairs-V06',
                                   'v06-go2refs-climb-first-v1'):
        raise RuntimeError('SDS 2.1 version identity drifted')
    if LOG_ROOT != 'logs/a2_ideal_stairs_v06' or TRAINING_SEED != 42:
        raise RuntimeError('SDS 2.1 log root / seed drifted')
    if (CONTROL_DT, PHYSICS_DT, DECIMATION, ACTION_SCALE) != (.02, .005, 4, .25):
        raise RuntimeError('SDS 2.3 control timing drifted')
    if (N_JOINTS, PROPRIO_DIM, HISTORY_LEN, HEIGHTMAP_SHAPE, ROUTE_DIM,
            PRIVILEGED_DIM, CRITIC_WIDTH) != (12, 45, 5, (3, 48, 32), 3, 35, 211):
        raise RuntimeError('SDS 2.3 observation geometry drifted')
    if CRITIC_NORMALIZER_DIMS + CRITIC_HEIGHTMAP_DIMS + CRITIC_PRIVILEGED_BLOCK_DIMS \
            + ROUTE_DIM != CRITIC_WIDTH:
        raise RuntimeError('critic input must be 80 + 64 + 64 + 3 = 211')
    if tuple(entry.tile_id for entry in TERRAIN_TABLE) != tuple(range(T_TILE_COUNT + 1)):
        raise RuntimeError('SDS 5.2 terrain tile ids must be 0..21')
    if TERRAIN_TABLE[0].lesson_id != FLAT_LESSON_ID or TERRAIN_TABLE[0].N != 0:
        raise RuntimeError('SDS 5.2 tile 0 must be flat')
    for index, row in enumerate(TERRAIN_ROWS):
        expected_id = index + 1
        spec = TERRAIN_BY_TILE[expected_id]
        if (spec.lesson_id, spec.N, spec.h, spec.step_depth) != row:
            raise RuntimeError(f'SDS 5.2 terrain row drift at T{index}')
        if (spec.width, spec.tread_thickness, spec.platform_length) != (.80, .01, 2.0):
            raise RuntimeError(f'SDS 5.2 stair geometry drift at {spec.lesson_id}')
    expected_heights = (.02, .030833333333333334, .04166666666666667,
                        .052500000000000005, .06333333333333334,
                        .07416666666666667, .085, .09583333333333334,
                        .10666666666666667, .1175, .12833333333333333,
                        .1391666666666667, .15, .16, .18, .20)
    if any(abs(a - b) > 1e-12 for a, b in zip(STEP_HEIGHTS_H00_H15, expected_heights)):
        raise RuntimeError('SDS 5.2 H-series step heights drifted')
    if [phase.phase for phase in PHASES] != ['F0', 'H', 'D', 'L', 'V', 'R']:
        raise RuntimeError('SDS 5.1 phase order drifted')
    if PHASE_BUDGETS != {'F0': 1000, 'H': 12000, 'D': 3000, 'L': 6000,
                         'V': 4000, 'R': 4000} or TOTAL_BUDGET != 30000:
        raise RuntimeError('SDS 5.1 phase budgets must sum to 30000')
    if D_SUBSTAGE_BUDGETS != (1500, 1500) or L_SUBSTAGE_BUDGETS != (1500, 1500, 3000):
        raise RuntimeError('SDS 5.6 substage budgets drifted')
    if tuple(REWARD_TERMS) != ('tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z',
                               'orientation_roll', 'base_height', 'action_rate',
                               'similar_to_default', 'dof_acc', 'dof_vel',
                               'ang_vel_xy', 'feet_air_time', 'foot_slip',
                               'foot_clearance', 'joint_tracking', 'center',
                               'heading', 'joint_limits'):
        raise RuntimeError('SDS 3.2 reward term order drifted')
    expected_weights = dict(tracking_lin_vel=1.5, tracking_ang_vel=.8, lin_vel_z=-1.0,
                            orientation_roll=-5.0, base_height=-.1, action_rate=-.01,
                            similar_to_default=-.05, dof_acc=-2.5e-7, dof_vel=-5e-4,
                            ang_vel_xy=-.05, feet_air_time=.2, foot_slip=-.15,
                            foot_clearance=-.5, joint_tracking=-.1, center=-.5,
                            heading=-.5, joint_limits=-10.0)
    if REWARD_WEIGHTS != expected_weights:
        raise RuntimeError('SDS 3.2 reward weights drifted')
    if (FORWARD_PROGRESS_PER_M, SUCCESS_REWARD, FAILURE_REWARD, TIMEOUT_REWARD) != \
            (.4, 10.0, -10.0, 0.):
        raise RuntimeError('SDS 3.1 forward/terminal amounts drifted')
    if abs(clearance_target(.02, flat=False) - .17) > 1e-15 \
            or abs(clearance_target(.15, flat=False) - .17) > 1e-15 \
            or abs(clearance_target(.18, flat=False) - .20) > 1e-15 \
            or abs(clearance_target(.20, flat=False) - .22) > 1e-15 \
            or clearance_target(0., flat=True) != .075:
        raise RuntimeError('SDS 3.3 clearance targets drifted')
    if (BASE_HEIGHT_TARGET, LIN_VEL_Z_DEADBAND, AIR_TIME_TARGET) != (.40, .15, .10):
        raise RuntimeError('SDS 3.2/3.3 scalar constants drifted')
    expected_ppo = dict(num_envs=512, eval_num_envs=8, rollout=48, epochs=5,
                        minibatches=4, learning_rate=3e-4, min_lr=1e-5, max_lr=3e-4,
                        desired_kl=.01, gamma=.99, gae_lambda=.95, clip=.2,
                        entropy_coef=.003, value_coef=1.0, clipped_value_loss=True,
                        max_grad_norm=1.0, initial_std=.30, std_min=.05, std_max=1.00)
    for name, value in expected_ppo.items():
        if PPO_CONFIG[name] != value:
            raise RuntimeError(f'SDS 4 PPO field {name} drifted')
    if PPO_CONFIG['adam_updates_per_iteration'] != 20 \
            or PPO_CONFIG['control_samples_per_update'] != 24576 \
            or PPO_CONFIG['physics_substeps_per_update'] != 98304:
        raise RuntimeError('SDS 8.1 ledger per-update values drifted')
    if (CONTROL_SAMPLES_PER_UPDATE, PHYSICS_SUBSTEPS_PER_UPDATE,
            ADAM_UPDATES_PER_ITERATION) != (24576, 98304, 20):
        raise RuntimeError('SDS 8.1 ledger drifted')
    if (TOTAL_CONTROL_SAMPLES, TOTAL_PHYSICS_SUBSTEPS, TOTAL_ADAM_UPDATES) != \
            (737_280_000, 2_949_120_000, 600_000):
        raise RuntimeError('SDS 8.1 30000-update ledger drifted')
    if (CURRICULUM_WINDOW, CURRICULUM_WINDOWS_PER_UPDATE, EMA_DECAY, EMA_NEW) != \
            (64, 1, .97, .03):
        raise RuntimeError('SDS 5.5 curriculum window/EMA drifted')
    if (ADVANCE_S_MIN, ADVANCE_Q_MIN, ADVANCE_F_MAX, ADVANCE_STREAK) != (.65, .45, .35, 5):
        raise RuntimeError('SDS 5.5 advance thresholds drifted')
    if (REGRESS_F_MIN, REGRESS_S_MAX, REGRESS_STREAK, REGRESS_TICKS) != (.40, .20, 2, 3):
        raise RuntimeError('SDS 5.5 regress thresholds drifted')
    if (LEVEL_TICK_MIN, LEVEL_TICK_MAX, FRONTIER_ROW_COUNT, COOLDOWN_CHECKS) != (0, 100, 15, 5):
        raise RuntimeError('SDS 5.5 level tick / cooldown drifted')
    if ROLE_COUNTS != {'flat': 102, 'frontier': 164, 'near': 123, 'easy': 123} \
            or ROLE_COUNT_TOTAL != 512:
        raise RuntimeError('SDS 5.3 role counts must be 102/164/123/123 = 512')
    if (NEAR_OFFSETS, EASY_BACKOFF) != ((-2, -1), 3):
        raise RuntimeError('SDS 5.3 near/easy offsets drifted')
    if (V_R_FRONTIER_TILE_ID, V_R_NEAR_TILES, V_R_EASY_LAST_TILE_ID) != (21, ('L12', 'L20'), 18):
        raise RuntimeError('SDS 5.3 V/R sampling anchors drifted')
    if (V_MIN_UPDATES, V_GATE_STREAK_START, R_MIN_UPDATES, R_GATE_STREAK_START) != \
            (1200, 1000, 600, 400):
        raise RuntimeError('SDS 5.6 V/R budgets drifted')
    if (COMMAND_INITIAL_VX, COMMAND_FINAL_VX, V_RAMP_UPDATES, COMMAND_VY,
            COMMAND_WZ) != ((.30, .80), (.20, .60), 1000, 0., 0.):
        raise RuntimeError('SDS 5.4 command range drifted')
    if (SPAWN_X, SPAWN_Y, R_SPAWN_X, R_SPAWN_Y, R_SPAWN_RAMP_UPDATES) != \
            ((-1.10, -.90), (-.05, .05), (-1.20, -.80), (-.10, .10), 400):
        raise RuntimeError('SDS 5.4 spawn range drifted')
    if (SPAWN_YAW_DEGREES, R_SPAWN_YAW_DEGREES) != ((-5., 5.), (-10., 10.)):
        raise RuntimeError('SDS 5.4 spawn yaw drifted')
    if any(abs(a - math.radians(e)) > 1e-15 for a, e in
           zip(SPAWN_YAW_RADIANS, SPAWN_YAW_DEGREES)) or \
            any(abs(a - math.radians(e)) > 1e-15 for a, e in
                zip(R_SPAWN_YAW_RADIANS, R_SPAWN_YAW_DEGREES)):
        raise RuntimeError('SDS 5.4 spawn yaw radians drifted')
    if (GATE_INTERVAL, DIAGNOSTIC_INTERVAL, SAVE_INTERVAL, RETENTION_INTERVAL) != \
            (100, 50, 100, 200):
        raise RuntimeError('SDS 6.1/8.5 intervals drifted')
    if (GATE_CASES, GATE_PASS_MIN, FINAL_CASES, FINAL_PASS_MIN) != (20, 18, 100, 95):
        raise RuntimeError('SDS 6.1/6.4 gate case counts drifted')
    if GATE_SPEEDS_F0_H_D_L != (.3, .5, .8) or GATE_SPEEDS_V_R != (.2, .4, .6):
        raise RuntimeError('SDS 6.1 gate speeds drifted')
    if (F0_OBSERVE_STEPS, F0_LATERAL_MAX, F0_HEADING_MAX_DEGREES,
            F0_PROGRESS_FRACTION, F0_EPISODE_STEPS) != (500, .20, 10., .5, 1500):
        raise RuntimeError('SDS 6.1 F0 gate constants drifted')
    if (SUCCESS_RECENT_SUPPORT_STEPS, SUCCESS_SINGLE_FOOT_FRAMES,
            SUCCESS_MIN_SUPPORT_FEET, SUCCESS_BODY_PLATFORM_MARGIN,
            SUCCESS_LATERAL_MAX, SUCCESS_CONSECUTIVE_FRAMES) != (50, 2, 2, .8, .20, 5):
        raise RuntimeError('SDS 6.2 success definition drifted')
    if abs(SUCCESS_FOOT_SAFE_INSET - .042) > 1e-15:
        raise RuntimeError('SDS 6.2 safe inset must be 0.042 m')
    if (FAILURE_NONFOOT_PERSIST_TIME, FAILURE_ATTITUDE_LIMIT_DEGREES) != (.1, 70.):
        raise RuntimeError('SDS 6.2 failure constants drifted')
    if DIAGNOSTIC_COLLAPSE_STREAK != 2:
        raise RuntimeError('Q001-7 diagnostic collapse streak must be 2')
    if FALL_DROP != .50:
        raise RuntimeError('Q002-2 FALL_DROP must be 0.50 m')
    if REWARD_CONFIG['fall_drop']['drop'] != FALL_DROP:
        raise RuntimeError('Q002-2 FALL_DROP must be part of REWARD_CONFIG')
    if STOP_STATES != ('behavior_regression', 'behavior_collapse',
                       'curriculum_stalled', 'budget_exhausted', 'gate_failed',
                       'user_requested_stop', 'monitoring_blocked',
                       'resource_insufficient'):
        raise RuntimeError('Q002-4 stop-state names/order drifted')
    if LIFECYCLE_STATES != ('running', 'ready_for_final_test', 'complete'):
        raise RuntimeError('Q002-4 lifecycle-state names/order drifted')
    if len(STOP_STATES) != 8 or len(LIFECYCLE_STATES) != 3:
        raise RuntimeError('Q002-4 must freeze eight stop and three lifecycle states')
    if 'pilot_not_applicable' in ALL_TRAINING_STATES:
        raise RuntimeError('Q002-4 pilot_not_applicable is not a v06 terminal state')
    if REWARD_CONFIG['diagnostics']['collapse_streak'] != DIAGNOSTIC_COLLAPSE_STREAK:
        raise RuntimeError('Q001-7 collapse streak must be part of REWARD_CONFIG')
    if (STALL_HISTORY_SAMPLES, STALL_DX_FRACTION_PER_SECOND,
            STALL_DX_THRESHOLD_SCALE, STALL_HIGHWATER_DX,
            STALL_CONSECUTIVE_SAMPLES, STALL_ACTIVE_COMMAND) != \
            (51, .15, 1.0, .03, 50, .1):
        raise RuntimeError('inherited r2 stall diagnostic constants drifted')
    for container in (REWARD_CONFIG['diagnostics']['stall'],):
        if (container['history_samples'], container['dx_fraction_per_second'],
                container['dx_threshold_scale'], container['highwater_dx'],
                container['consecutive_samples'], container['active_command']) != \
                (51, .15, 1.0, .03, 50, .1):
            raise RuntimeError('stall diagnostic constants must be in the contract/config')
        if container['reward_payment'] or container['termination']:
            raise RuntimeError('inherited stall diagnostic must not pay or terminate')
    if (VALIDATION_SEED_OFFSET, DIAGNOSTIC_SEED_OFFSET, PERTURBED_SEED_OFFSET,
            FINAL_TEST_SEED) != (200300, 500300, 300300, 400342):
        raise RuntimeError('SDS 8.3 D1 seed offsets drifted')
    if (SMOKE_SEED, SMOKE_ENVS, SMOKE_TILE, SMOKE_UPDATES) != (42, 8, 'H00', 2):
        raise RuntimeError('SDS 9.3 smoke contract drifted')
    if (SMOKE_CONTROL_SAMPLES, SMOKE_PHYSICS_SUBSTEPS, SMOKE_ADAM_UPDATES) != \
            (768, 3072, 40):
        raise RuntimeError('SDS 9.3 smoke ledger drifted')
    if (PROBE_UPDATES, PROBE_CONTROL_SAMPLES) != (56, 1_376_256):
        raise RuntimeError('SDS 9.3 probe ledger drifted')
    if RESOURCE_LIMITS != dict(gpu_mib=7168, process_mib=5120, host_free_mib=4096,
                               disk_free_gib=10, single_gpu_worker=True):
        raise RuntimeError('SDS 8.5 resource limits drifted')
    if (episode_timeout_steps(40, .20, .20, .2),
            episode_timeout_steps(40, .20, .20, .6)) != (7257, 2753):
        raise RuntimeError('SDS 6.2 timeout formula drifted')
    if curriculum_frontier_row(0) != 0 or curriculum_frontier_row(100) != 15 \
            or curriculum_frontier_row(7) != 1 or curriculum_frontier_row(1) != 0:
        raise RuntimeError('SDS 5.5 frontier row formula drifted')
    for name, path, digest in FROZEN_DOCUMENTS:
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f'frozen {name} SHA256 must be a 64-character digest')


_check_frozen_tables()

# C1 is opt-in and resolved during spec import, before consumers cache constants.
# The unchanged base frozen checks above always run first.
import os as _os
_continuation = _os.environ.get('A2_V06_CONTINUATION')
if _continuation:
    from .c1_contract import CONTINUATION_ID as _c1_id, apply as _apply_c1
    if _continuation != _c1_id:
        raise RuntimeError('unknown v06 continuation contract')
    _apply_c1(globals())
