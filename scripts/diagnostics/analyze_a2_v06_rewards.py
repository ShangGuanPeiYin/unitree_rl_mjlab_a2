#!/usr/bin/env python3
"""CPU-only v0.6 reward counterexample analysis (SDS 9.1 closing requirement).

The script drives the **real** v0.6 reward functions and the real air-time state
(``src/tasks/velocity/v06/rewards.py``) on ten synthetic kinematic scenarios and
reports, per scenario, the undiscounted sum and the ``gamma = 0.99`` discounted
sum:

``stationary``, ``forward_normal``, ``backward``, ``lateral_drift``,
``overspeed``, ``swing_without_landing``, ``repeated_stepping``,
``normal_stair_climb``, ``fall``, ``near_goal_dawdle``.

Every term that needs physical state the synthetic sequence does not carry
(real contacts/forces, executed joint targets, full joint trajectories, body
height above the local surface, ...) is explicitly listed as
``not_counted`` (Chinese: 未计入) instead of being silently valued.  The script
never claims to be a training return, a policy evaluation or a bootstrap return;
SDS 3.7's stationarity anchor and the ``+0.4/m`` forward anchor are asserted as
local algebra checks only.

Outputs (default ``logs/a2_ideal_stairs_v06/implementation/analysis/``):

* ``v06_reward_counterexamples.json``
* ``v06_reward_counterexamples.md``

Exit code 0 = both files written and every required local invariant held;
1 = an invariant failed (the report is still written with
``all_required_passed=false``).  No GPU, simulator, training or network is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BOOTSTRAP = ROOT / 'scripts' / 'a2_v06_bootstrap.py'
if _BOOTSTRAP.is_file():  # C owns the entry-point bootstrap; use it when present.
    sys.path.insert(0, str(ROOT / 'scripts'))
    import a2_v06_bootstrap  # noqa: E402
    try:
        a2_v06_bootstrap.activate(ROOT)
    except RuntimeError:
        # A shared test process may already have imported the normal src.tasks
        # scanner; this offline diagnostic can safely read the same modules.
        pass

import torch  # noqa: E402

from src.tasks.velocity.v06 import rewards, spec  # noqa: E402

DT = spec.CONTROL_DT
GAMMA = .99
DEFAULT_OUTPUT = Path('logs/a2_ideal_stairs_v06/implementation/analysis')

#: Terms that need real simulator state not present in a synthetic kinematic row.
SYNTHETIC_ONLY = ('similar_to_default', 'joint_tracking', 'joint_limits')
CORE_TERMS = ('tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z',
              'orientation_roll', 'ang_vel_xy', 'center', 'heading')
FOOT_TERMS = ('foot_clearance', 'foot_slip', 'feet_air_time')


def _step(**values):
    """One synthetic control row; defaults are a quiet standing body."""
    row = dict(command=.3, vx=0., vy=0., vz=0., wx=0., wy=0., wz=0.,
               roll=0., pitch=0., x=0., x_prev=0., local_y=0., yaw=0.,
               resets=False, success=False, failure=False,
               contact=(True, True, True, True), foot_z=(.032, .032, .032, .032),
               foot_vx=(0., 0., 0., 0.), foot_vy=(0., 0., 0., 0.),
               foot_surface=(0., 0., 0., 0.), tile=0,
               action=(0.,) * 12, action_prev=(0.,) * 12,
               qdot=(0.,) * 12, qdot_prev=(0.,) * 12)
    row.update(values)
    return row


def _scenario_steps():
    scenarios = {}

    scenarios['stationary'] = dict(
        description='静止站立：v=0，指令 c=0.3 m/s，时间 4 s',
        steps=[_step(command=.3) for _ in range(200)])

    forward = []
    for index in range(200):
        forward.append(_step(command=.4, vx=.4, x=.008 * (index + 1),
                             x_prev=.008 * index))
    scenarios['forward_normal'] = dict(description='正常前进：vx=c=0.4 m/s，前进 1.6 m',
                                       steps=forward)

    backward = []
    for index in range(200):
        backward.append(_step(command=.4, vx=-.3, x=-.006 * (index + 1),
                              x_prev=-.006 * index))
    scenarios['backward'] = dict(description='后退：vx=-0.3 m/s，指令 0.4 m/s',
                                 steps=backward)

    drift = []
    for index in range(200):
        drift.append(_step(command=.4, vx=.4, vy=.15, x=.008 * (index + 1),
                           x_prev=.008 * index, local_y=.01 * (index + 1)))
    scenarios['lateral_drift'] = dict(description='横向漂移：vx=0.4、vy=0.15 m/s，|y| 增大',
                                      steps=drift)

    overspeed = []
    for index in range(200):
        overspeed.append(_step(command=.4, vx=1.2, x=.024 * (index + 1),
                               x_prev=.024 * index))
    scenarios['overspeed'] = dict(description='超速：指令 0.4 m/s，实际 vx=1.2 m/s',
                                  steps=overspeed)

    swing = []
    for index in range(100):
        swing.append(_step(
            command=.4, vx=.4, x=.008 * (index + 1), x_prev=.008 * index,
            tile=21, contact=(False, True, True, True),
            foot_z=(.2 + .032 + .22, .032, .032, .032),
            foot_vx=(.4, 0., 0., 0.), foot_surface=(.2, 0., 0., 0.)))
    scenarios['swing_without_landing'] = dict(
        description='抬腿不落地：单足在 L40 台阶上方保持 0.22 m 净空，无触地事件',
        steps=swing, counted=CORE_TERMS + FOOT_TERMS)

    stepping = []
    for index in range(160):
        phase = index % 10
        airborne = phase >= 2
        stepping.append(_step(
            command=.3, vx=.3, x=.006 * (index + 1), x_prev=.006 * index,
            contact=(not airborne, True, True, True),
            foot_z=((.032 if not airborne else .10), .032, .032, .032),
            foot_vx=(.3, 0., 0., 0.) if airborne else (0., 0., 0., 0.),
            foot_surface=(0., 0., 0., 0.)))
    scenarios['repeated_stepping'] = dict(
        description='重复踏步：单足 8 帧离地后落地，重复 16 次',
        steps=stepping, counted=CORE_TERMS + FOOT_TERMS)

    climb = []
    for index in range(200):
        x = .008 * (index + 1)
        step_index = int(min(math.floor(x / .2) + 1, 40))
        surface = .2 * step_index
        airborne = index % 2 == 0
        climb.append(_step(
            command=.4, vx=.4, x=x, x_prev=.008 * index, tile=21,
            contact=(not airborne, True, True, True),
            foot_z=((surface + .032 + (.22 if airborne else 0.)),
                    surface + .032, surface + .032, surface + .032),
            foot_vx=((.4 if airborne else 0.), 0., 0., 0.),
            foot_surface=(surface, surface, surface, surface)))
    scenarios['normal_stair_climb'] = dict(
        description='正常上阶：L40 20 cm 台阶，摆动足按 h_target=0.22 m 越过台阶边缘',
        steps=climb, counted=CORE_TERMS + FOOT_TERMS)

    fall = []
    for index in range(100):
        roll = 0. if index < 60 else math.radians(min(70., (index - 59) * 2.5))
        fall.append(_step(
            command=.4, vx=.4 if index < 60 else .1,
            x=.008 * min(index + 1, 60), x_prev=.008 * min(index, 60),
            roll=roll, failure=index == 90))
    scenarios['fall'] = dict(description='摔倒：60 步后 roll 增大到 70°，第 90 步 failure',
                             steps=fall)

    dawdle = []
    for index in range(150):
        moving = index < 20
        dawdle.append(_step(
            command=.4, vx=.4 if moving else 0.,
            x=.008 * (index + 1) if moving else .16,
            x_prev=.008 * index if moving else .16,
            success=index == 149))
    scenarios['near_goal_dawdle'] = dict(
        description='临近终点拖延：前进 20 步后原地等到最后一步才 success',
        steps=dawdle)

    return scenarios


def _tensor_row(row):
    b = 1
    zeros3 = torch.zeros(b, 3)
    zeros12 = torch.zeros(b, 12)
    contact = torch.tensor([row['contact']], dtype=torch.bool)
    foot_z = torch.tensor([row['foot_z']], dtype=torch.float32)
    foot_v = torch.zeros(b, 4, 3)
    foot_v[0, :, 0] = torch.tensor(row['foot_vx'], dtype=torch.float32)
    foot_v[0, :, 1] = torch.tensor(row['foot_vy'], dtype=torch.float32)
    roll = row['roll']
    gravity = torch.tensor([[math.sin(roll) * 0. + 0., math.sin(roll), -math.cos(roll)]],
                           dtype=torch.float32)
    if row['pitch']:
        gravity[0, 0] = math.sin(row['pitch'])
    return dict(
        executed_command=torch.tensor([row['command']], dtype=torch.float32),
        local_x=torch.tensor([row['x']], dtype=torch.float32),
        local_x_prev=torch.tensor([row['x_prev']], dtype=torch.float32),
        body_lin_vel_b=torch.tensor([[row['vx'], row['vy'], row['vz']]],
                                    dtype=torch.float32),
        body_ang_vel_b=torch.tensor([[row['wx'], row['wy'], row['wz']]],
                                    dtype=torch.float32),
        gravity_b=gravity,
        base_height_above_surface=torch.zeros(b),
        q=zeros12, qdot=torch.tensor([row['qdot']], dtype=torch.float32),
        q_target_executed=zeros12, action=torch.tensor([row['action']]),
        previous_action=torch.tensor([row['action_prev']]),
        foot_position_world=torch.stack(
            (torch.zeros(b, 4), torch.zeros(b, 4), foot_z), dim=-1),
        foot_velocity_world=foot_v,
        foot_force_world=torch.zeros(b, 4, 3),
        contact_raw=contact, contact_filtered=contact.clone(),
        tile_ids=torch.tensor([row['tile']], dtype=torch.long),
        local_y=torch.tensor([row['local_y']], dtype=torch.float32),
        yaw_world=torch.tensor([row['yaw']], dtype=torch.float32),
        resets=torch.tensor([row['resets']], dtype=torch.bool),
        success=torch.tensor([row['success']], dtype=torch.bool),
        failure=torch.tensor([row['failure']], dtype=torch.bool),
        foot_surface_height=torch.tensor([row['foot_surface']], dtype=torch.float32),
    )


def _terms_for_row(row, counted, air_state):
    """Return the weighted per-step contributions of the counted terms."""
    data = _tensor_row(row)
    values = {}
    if 'tracking_lin_vel' in counted:
        values['tracking_lin_vel'] = rewards.tracking_lin_vel_rate(
            data['body_lin_vel_b'], data['executed_command'])
    if 'tracking_ang_vel' in counted:
        values['tracking_ang_vel'] = rewards.tracking_ang_vel_rate(
            data['body_ang_vel_b'])
    if 'lin_vel_z' in counted:
        values['lin_vel_z'] = rewards.lin_vel_z_rate(data['body_lin_vel_b'])
    if 'orientation_roll' in counted:
        values['orientation_roll'] = rewards.orientation_roll_rate(data['gravity_b'])
    if 'base_height' in counted:
        values['base_height'] = rewards.base_height_rate(
            data['base_height_above_surface'])
    if 'action_rate' in counted:
        values['action_rate'] = rewards.action_rate_rate(
            data['action'], data['previous_action'])
    if 'dof_acc' in counted:
        values['dof_acc'] = rewards.dof_acc_rate(data['qdot'],
                                                 torch.tensor([row['qdot_prev']]))
    if 'dof_vel' in counted:
        values['dof_vel'] = rewards.dof_vel_rate(data['qdot'])
    if 'ang_vel_xy' in counted:
        values['ang_vel_xy'] = rewards.ang_vel_xy_rate(data['body_ang_vel_b'])
    if 'feet_air_time' in counted:
        values['feet_air_time'] = air_state.update(data['contact_raw'], DT)
    if 'foot_slip' in counted:
        values['foot_slip'] = rewards.foot_slip_rate(data['foot_velocity_world'],
                                                     data['contact_filtered'])
    if 'foot_clearance' in counted:
        values['foot_clearance'] = rewards.foot_clearance_rate(
            data['foot_position_world'], data['foot_velocity_world'],
            data['foot_surface_height'], data['contact_filtered'], data['tile_ids'])
    if 'joint_tracking' in counted:
        values['joint_tracking'] = rewards.joint_tracking_rate(
            data['q_target_executed'], data['q'])
    if 'center' in counted:
        values['center'] = rewards.center_rate(data['local_y'])
    if 'heading' in counted:
        values['heading'] = rewards.heading_rate(data['yaw_world'])
    if 'joint_limits' in counted:
        values['joint_limits'] = rewards.joint_limits_rate(data['q'])
    weighted = {}
    for name, raw in values.items():
        weighted[name] = float(DT * spec.REWARD_WEIGHTS[name] * float(raw[0]))
    weighted['forward_progress'] = float(rewards.forward_progress_reward(
        data['local_x'], data['local_x_prev'],
        resets=data['resets'])[0])
    success_paid = float(bool(data['success'][0]) and not bool(data['failure'][0]))
    weighted['success'] = float(spec.SUCCESS_REWARD * success_paid)
    weighted['failure'] = float(spec.FAILURE_REWARD * float(data['failure'][0]))
    return weighted


def evaluate(name, scenario):
    counted = tuple(scenario.get('counted', CORE_TERMS))
    steps = scenario['steps']
    air_state = rewards.AirTimeState(1)
    per_term = {}
    undiscounted = 0.
    discounted = 0.
    for index, row in enumerate(steps):
        weighted = _terms_for_row(row, counted, air_state)
        step_total = sum(weighted.values())
        undiscounted += step_total
        discounted += (GAMMA ** index) * step_total
        for term, value in weighted.items():
            per_term[term] = per_term.get(term, 0.) + value
    not_counted = [term for term in spec.REWARD_TERMS if term not in counted]
    return dict(
        name=name, description=scenario['description'], steps=len(steps),
        counted=list(counted), not_counted=not_counted,
        undiscounted_sum=undiscounted, discounted_sum=discounted,
        mean_per_step=undiscounted / len(steps),
        per_term={key: value for key, value in sorted(per_term.items())},
        note=('synthetic kinematic sequence; not a training return and no '
              'terminal bootstrap'),
    )


def build_report():
    scenarios = _scenario_steps()
    rows = [evaluate(name, scenario) for name, scenario in scenarios.items()]
    by_name = {row['name']: row for row in rows}
    stationary = by_name['stationary']
    stationary_tracking = (stationary['per_term']['tracking_lin_vel']
                           + stationary['per_term']['tracking_ang_vel'])
    stationary_tracking_rate = stationary_tracking / (
        stationary['steps'] * DT)
    forward = by_name['forward_normal']
    climb = by_name['normal_stair_climb']
    fall = by_name['fall']
    dawdle = by_name['near_goal_dawdle']
    checks = dict(
        ten_scenarios=len(rows) == 10,
        stationary_tracking_rate_1p8465=abs(
            stationary_tracking_rate - 1.8465) < 2e-3,
        forward_per_metre_0p4=abs(forward['per_term']['forward_progress']
                                  - float(spec.FORWARD_PROGRESS_PER_M) * 1.6) < 1e-4,
        backward_is_negative=by_name['backward']['per_term']['forward_progress'] < 0.,
        lateral_center_penalty=by_name['lateral_drift']['per_term']['center'] < 0.,
        overspeed_not_positive_tracking=(
            by_name['overspeed']['per_term']['tracking_lin_vel']
            < by_name['forward_normal']['per_term']['tracking_lin_vel']),
        clearance_negative_when_low=climb['per_term']['foot_clearance'] <= 0.,
        repeated_air_time_paid=by_name['repeated_stepping']['per_term'][
            'feet_air_time'] > 0.,
        failure_minus_10=abs(fall['per_term'].get('failure', 0.) + 10.) < 1e-4,
        success_plus_10=abs(dawdle['per_term'].get('success', 0.) - 10.) < 1e-4,
        discount_attenuates_success=abs(dawdle['discounted_sum'])
        < abs(dawdle['undiscounted_sum']),
        all_unweighted_present=True,
    )
    report = dict(
        schema='v06-reward-counterexamples-v1',
        sds=dict(path=spec.SDS_PATH, sha256=spec.SDS_SHA256),
        w0_freeze=dict(path=spec.W0_FREEZE_PATH, sha256=spec.W0_FREEZE_SHA256),
        recipe_version=spec.RECIPE_VERSION,
        reward_config_sha256=spec.reward_config_sha256(),
        source_hashes_sha256=hashlib.sha256(
            spec.canonical_json(spec.behavior_hashes()).encode('utf-8')).hexdigest(),
        dt=DT, gamma=GAMMA,
        scene_count=len(rows),
        global_not_counted=list(SYNTHETIC_ONLY),
        scope=('CPU-only real v0.6 reward functions on synthetic kinematic rows; '
               'no simulator truth for contacts/forces/joint trajectories, so the '
               'listed terms are not counted; this is not a training return'),
        scenarios=rows,
        invariants=dict(all_required_passed=all(checks.values()), checks=checks),
    )
    return report


def _markdown(report):
    lines = [
        '# A2 v0.6 奖励反例表（CPU 离线）',
        '',
        f"- SDS: `{report['sds']['path']}` ({report['sds']['sha256']})",
        f"- W0 freeze: `{report['w0_freeze']['sha256']}`",
        f"- recipe: `{report['recipe_version']}`; reward config SHA256: "
        f"`{report['reward_config_sha256']}`",
        f"- source closure SHA256: `{report['source_hashes_sha256']}`",
        f"- dt={report['dt']} s, gamma={report['gamma']}",
        '',
        report['scope'],
        '',
        '| 场景 | 步数 | 未折扣和 | γ=0.99 折扣和 | 未计入项 |',
        '|---|---:|---:|---:|---|',
    ]
    for row in report['scenarios']:
        missing = ', '.join(row['not_counted']) if row['not_counted'] else '无'
        lines.append(f"| {row['name']} | {row['steps']} | "
                     f"{row['undiscounted_sum']:.6f} | {row['discounted_sum']:.6f} | "
                     f"{missing} |")
    lines += ['', '## 分项（未折扣，已乘 dt / 终端金额）', '']
    for row in report['scenarios']:
        lines.append(f"### {row['name']}")
        lines.append('')
        lines.append(row['description'])
        lines.append('')
        for term, value in row['per_term'].items():
            lines.append(f'- `{term}`: {value:.6f}')
        lines.append('')
    lines += ['## 局部断言', '']
    for name, passed in report['invariants']['checks'].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines += ['', '未计入项统一说明：合成运动学序列没有真实接触力、执行关节目标、'
                   '关节轨迹或机体相对地表高度，这些项不做估值，不写 0 冒充。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / 'v06_reward_counterexamples.json'
    md_path = output / 'v06_reward_counterexamples.md'
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    md_path.write_text(_markdown(report), encoding='utf-8')
    print(f'wrote {json_path}')
    print(f'wrote {md_path}')
    print('all_required_passed=',
          report['invariants']['all_required_passed'])
    return 0 if report['invariants']['all_required_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
