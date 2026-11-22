"""S2 reward specification §5 (CPU-only, pure functions).

The continuous terms are evaluated once with ``0.02`` and never per sub-term.
``I_cruise`` requires a positive request, a non-DECEL/HOLD phase and an executed
command that has reached the request.  A same-frame failure/timeout pays -10 and
suppresses the one-time top/success rewards.  Stopping zeroes forward progress
and air-time reward and clears the air-time accumulator.
"""
from __future__ import annotations

import math

import numpy as np

DT = 0.02
FLAT_CLEARANCE = 0.075
STAIR_CLEARANCE_MIN = 0.17
STAIR_CLEARANCE_MARGIN = 0.02
SUCCESS_REWARD = 8.0
TOP_REWARD = 2.0
FAILURE_REWARD = -10.0
FORWARD_PROGRESS_PER_M = 0.4

# (cruise, decel) columns of §5.3.
WEIGHTS = {
    'tracking_lin_vel': (1.5, 1.5),
    'tracking_ang_vel': (0.8, 0.8),
    'lin_vel_z': (-1.0, -1.0),
    'orientation_roll': (-5.0, -5.0),
    'orientation_pitch_level': (-5.0, -5.0),
    'base_height': (-0.1, -0.1),
    'action_rate': (-0.01, -0.01),
    'similar_to_default': (-0.05, -0.05),
    'dof_acc': (-2.5e-7, -2.5e-7),
    'dof_vel': (-5e-4, -5e-4),
    'ang_vel_xy': (-0.05, -0.20),
    'feet_air_time': (0.2, 0.0),
    'foot_slip': (-0.15, -0.15),
    'foot_clearance': (-0.5, -0.5),
    'joint_tracking': (-0.1, -0.1),
    'center': (-0.5, -0.5),
    'heading': (-0.5, -0.5),
    'joint_limits': (-10.0, -10.0),
    'horizontal_overspeed': (-2.0, -2.0),
    'stable_stand': (0.0, 1.0),
}


def _arr(value, size=None):
    array = np.asarray(value, dtype=np.float64)
    if size is not None and array.size != size:
        raise ValueError(f'expected {size} values, got {array.size}')
    return array


def is_cruise(*, c_request, c_exec, phase):
    return bool(c_request > 0 and phase not in ('DECEL', 'HOLD')
                and abs(float(c_exec) - float(c_request)) <= 1e-9)


def clearance_target(surface, step_height):
    if surface in ('ground', 'flat', 'platform'):
        return FLAT_CLEARANCE
    if surface in ('tread', 'stairs', 'stair'):
        return max(STAIR_CLEARANCE_MIN, float(step_height) + STAIR_CLEARANCE_MARGIN)
    raise ValueError(f'unknown foot surface: {surface}')


def reward_terms(frame):
    """Unweighted 20 terms and flags; all arrays are per-env numpy."""
    c_request = float(frame.get('c_request', 0.0))
    c_exec = float(frame.get('c_exec', 0.0))
    phase = frame.get('phase', 'POST_WALK')
    cruise = is_cruise(c_request=c_request, c_exec=c_exec, phase=phase)
    decel = phase in ('DECEL', 'HOLD')
    column = 0 if not decel else 1
    g = _arr(frame.get('gravity_b', [0.0, 0.0, -1.0]), 3)
    v_b = _arr(frame.get('velocity_b', [0.0, 0.0, 0.0]), 3)
    omega = _arr(frame.get('omega_b', [0.0, 0.0, 0.0]), 3)
    q = _arr(frame.get('joint_position', np.zeros(12)), 12)
    q0 = _arr(frame.get('default_joint_position', np.zeros(12)), 12)
    qdot = _arr(frame.get('joint_velocity', np.zeros(12)), 12)
    qdot_prev = _arr(frame.get('joint_velocity_prev', qdot), 12)
    action = _arr(frame.get('action', np.zeros(12)), 12)
    action_prev = _arr(frame.get('previous_action', action), 12)
    q_target = _arr(frame.get('q_target_applied', q), 12)
    contact = _arr(frame.get('contact_filtered', np.zeros(4)), 4)
    foot_v = _arr(frame.get('foot_velocity_xy', np.zeros((4, 2))), 8).reshape(4, 2)
    foot_h = _arr(frame.get('foot_height', np.zeros(4)), 4)
    surfaces = frame.get('foot_surfaces', ['ground'] * 4)
    step_heights = frame.get('foot_step_heights', [0.0] * 4)
    g_level = float(frame.get('g_level', 0.0))
    if not 0.0 <= g_level <= 1.0:
        raise ValueError('g_level outside [0,1]')
    targets = np.array([clearance_target(surfaces[i], step_heights[i])
                        for i in range(4)])
    terms = {}
    terms['tracking_lin_vel'] = math.exp(-(((v_b[0] - c_exec) ** 2 + v_b[1] ** 2) / 0.25))
    terms['tracking_ang_vel'] = math.exp(-(omega[2] ** 2) / 0.25)
    terms['lin_vel_z'] = max(abs(v_b[2]) - 0.15, 0.0) ** 2
    terms['orientation_roll'] = float(g[1] ** 2)
    terms['orientation_pitch_level'] = g_level * float(g[0] ** 2)
    terms['base_height'] = float((float(frame.get('base_height', 0.4))
                                  - float(frame.get('surface_height', 0.0))
                                  - 0.40) ** 2)
    terms['action_rate'] = float(np.sum((action - action_prev) ** 2))
    terms['similar_to_default'] = float(np.sum(np.abs(q - q0)))
    terms['dof_acc'] = float(np.sum(((qdot - qdot_prev) / DT) ** 2))
    terms['dof_vel'] = float(np.sum(qdot ** 2))
    terms['ang_vel_xy'] = float(omega[0] ** 2 + omega[1] ** 2)
    terms['feet_air_time'] = float(frame.get('air_time_reward', 0.0))
    terms['foot_slip'] = float(np.sum(contact * np.sum(foot_v ** 2, axis=1)))
    terms['foot_clearance'] = float(np.sum(
        (1.0 - contact) * (foot_h - targets) ** 2
        * np.linalg.norm(foot_v, axis=1)))
    terms['joint_tracking'] = float(np.sum((q_target - q) ** 2))
    terms['center'] = min((float(frame.get('local_y', 0.0)) / 0.20) ** 2, 4.0)
    terms['heading'] = 1.0 - math.cos(float(frame.get('yaw_error', 0.0)))
    terms['joint_limits'] = float(frame.get('joint_limit_violation', 0.0))
    terms['horizontal_overspeed'] = g_level * max(
        float(frame.get('v_heading', v_b[0])) - c_exec - 0.15, 0.0) ** 2
    # §5.3: I_stable × I_abs(c_exec)≤1e-7; partial DECEL commands never pay it.
    terms['stable_stand'] = 1.0 if (decel and abs(c_exec) <= 1e-7
                                     and frame.get('stable_stand', False)) else 0.0
    if decel:
        # Stop request: forward progress and air-time reward are closed.
        air_time = 0.0
    else:
        air_time = terms['feet_air_time']
    terms['feet_air_time'] = air_time
    return dict(terms=terms, weights=dict(WEIGHTS), column=column,
                cruise=cruise, decel=decel,
                target_clearance=targets.tolist(),
                clear_air_accumulator=bool(decel))


def compute_reward(frame):
    """One control-step reward with the §5.1 one-time terms and precedence."""
    result = reward_terms(frame)
    cruise = result['cruise']
    column = result['column']
    continuous = 0.02 * sum(WEIGHTS[name][column] * value
                            for name, value in result['terms'].items())
    dx = 0.0 if result['decel'] else float(frame.get('dx', 0.0))
    forward = FORWARD_PROGRESS_PER_M * dx if cruise else 0.0
    failure = bool(frame.get('failure', False) or frame.get('timeout', False))
    top = bool(frame.get('top_once', False)) and not failure
    success = bool(frame.get('success_once', False)) and not failure
    total = continuous + forward
    if top:
        total += TOP_REWARD
    if success:
        total += SUCCESS_REWARD
    if failure:
        total += FAILURE_REWARD
    return dict(raw_terms=result['terms'], weighted_terms={
        name: WEIGHTS[name][column] * value for name, value in result['terms'].items()},
        continuous=continuous, forward_progress=forward,
        top_reward=TOP_REWARD if top else 0.0,
        success_reward=SUCCESS_REWARD if success else 0.0,
        failure_reward=FAILURE_REWARD if failure else 0.0,
        total=total, cruise=cruise, decel=result['decel'],
        clear_air_accumulator=result['clear_air_accumulator'],
        target_clearance=result['target_clearance'])
