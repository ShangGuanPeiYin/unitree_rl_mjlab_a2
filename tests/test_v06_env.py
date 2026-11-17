"""SDS 9.1 items 11/12/16: real v0.6 environment lifecycle on CPU.

A four-environment CPU scene is built once; the tests exercise the observation
contract, schedule commits (partial reset), SDS 5.4 spawn/command ranges, the
agent-A reward computer wiring and the pre-reset terminal snapshots.
"""
import json
import math

import numpy as np
import pytest
import torch

from src.tasks.velocity.v06 import curriculum, events, spec, terrain
from src.tasks.velocity.v06.events import euler_heading
from src.tasks.velocity.v06.curriculum import ScheduleAssignment
from src.tasks.velocity.v06.env import V06Env


@pytest.fixture(scope='module')
def env():
    environment = V06Env(num_envs=4, device='cpu', seed=42, phase='H')
    yield environment
    del environment


def _schedule(env, *, phase, substage, frontier, tiles):
    roles = tuple(env.roles)
    return ScheduleAssignment(phase=phase, substage=substage,
                              frontier_tile_id=frontier, roles=roles,
                              tile_ids=tuple(tiles), generation=env.generation)


def test_observation_contract_and_route_formula(env):
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    obs = env._observations(update=False)
    assert set(obs) == {'proprio', 'history', 'heightmap', 'privileged', 'route_error'}
    assert tuple(obs['proprio'].shape) == (4, 45)
    assert tuple(obs['history'].shape) == (4, 5, 45)
    assert tuple(obs['heightmap'].shape) == (4, 3, 48, 32)
    assert tuple(obs['privileged'].shape) == (4, 35)
    assert tuple(obs['route_error'].shape) == (4, 3)
    for value in obs.values():
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
    data = env.robot.data
    local = data.root_link_pos_w - env.origins[env.tile_ids]
    _, _, yaw = euler_heading(data.root_link_quat_w)
    expected = torch.stack((torch.clamp(local[:, 1] / 0.20, -4, 4),
                            torch.sin(yaw), torch.cos(yaw)), -1)
    torch.testing.assert_close(obs['route_error'], expected)
    # Every assigned tile comes from the v0.6 table (no v0.2 atlas).
    assert bool(((env.tile_ids >= 0) & (env.tile_ids < 22)).all())
    assert int(env.roles.count('flat')) == 1
    assert set(env.roles) == {'flat', 'frontier', 'near', 'easy'}


def test_step_reward_is_agent_a_computer_and_outcomes_are_per_env(env):
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 12))
    assert reward.shape == (4,) and torch.isfinite(reward).all()
    tensors = info['step_reward_tensors']
    weighted = {key: value for key, value in tensors.items()
                if key.startswith('reward_weighted/')}
    total = sum(weighted.values())
    torch.testing.assert_close(total, reward, atol=1e-5, rtol=1e-5)
    assert set(info['failure_masks']) == set(events.FAILURE_NAMES)
    assert info['roles'] == env.roles
    assert info['generation'] == env.generation
    assert float(info['tracking_score'].min()) >= 0.0
    assert 'episodes' in info and 'terminal_observation' in info
    for key, value in tensors.items():
        assert torch.isfinite(value).all(), key
    # No state is shared between two environments' reward accumulation.
    assert set(weighted) <= set(env.episode_reward_sums)


def test_timeout_terminates_with_pre_reset_record_and_partial_reset(env):
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    env.episode_length_buf[:] = 10 ** 6
    pre_reset = env.episode_length_buf.clone()
    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 12))
    assert bool(truncated.all()) and not bool(terminated.any())
    records = info['episodes']
    assert len(records) == 4
    for record in records:
        assert record['timeout'] is True
        assert record['failure'] is False
        assert record['terminal_step'] == int(pre_reset[record['env_id']]) + 1
        assert record['reason'] == 'timeout'
        assert record['tracking_score'] == pytest.approx(
            float(info['tracking_score'][record['env_id']]), abs=1e-5)
        assert record['role'] == env.roles[record['env_id']]
        assert record['generation'] == env.generation
        assert record['tile_id'] == int(env.tile_ids[record['env_id']])
    # All environments were reset: the next episode starts from zero steps.
    assert bool((env.episode_length_buf == 0).all())
    assert bool((env.robot.data.root_link_pos_w[:, 2] < 1.0).all())


def test_set_schedule_resets_only_affected_environments(env):
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    # A first L40 commit changes every environment (phase change).
    tiles = [int(v) for v in env.tile_ids]
    for index, role in enumerate(env.roles):
        if role == 'frontier':
            tiles[index] = spec.tile_id_for('L40')
        elif role == 'near':
            tiles[index] = spec.tile_id_for('L12')
        elif role == 'easy':
            tiles[index] = spec.tile_id_for('H02')
    schedule = _schedule(env, phase='L', substage='L40',
                         frontier=spec.tile_id_for('L40'), tiles=tiles)
    env.set_schedule(schedule, 0)
    assert env.phase == 'L' and env.substage == 'L40'
    assert bool((env.episode_length_buf == 0).all())
    for _ in range(3):
        env.step(torch.zeros(4, 12))
    before = env.episode_length_buf.clone()
    assert int(before.min()) >= 2
    # Same phase/substage/roles, one tile change -> exactly that env resets.
    changed = list(tiles)
    target = next(index for index, role in enumerate(env.roles) if role == 'easy')
    other = next(index for index, role in enumerate(env.roles) if role == 'near')
    changed[target] = spec.tile_id_for('H03')
    schedule = _schedule(env, phase='L', substage='L40',
                         frontier=spec.tile_id_for('L40'), tiles=changed)
    env.set_schedule(schedule, 0)
    assert int(env.episode_length_buf[target]) == 0
    for index in range(4):
        if index != target:
            assert int(env.episode_length_buf[index]) == int(before[index])
    # A role-only change also resets exactly that environment.
    roles = list(env.roles)
    near_index = next(index for index, role in enumerate(roles) if role == 'near')
    roles[near_index] = 'easy'
    schedule = ScheduleAssignment(phase='L', substage='L40',
                                  frontier_tile_id=spec.tile_id_for('L40'),
                                  roles=tuple(roles),
                                  tile_ids=tuple(changed), generation=env.generation)
    env.set_schedule(schedule, 0)
    assert int(env.episode_length_buf[near_index]) == 0


def test_set_schedule_rejects_invalid_assignments(env):
    tiles = [int(v) for v in env.tile_ids]
    with pytest.raises(ValueError):
        env.set_schedule(_schedule(env, phase='L', substage='L40',
                                   frontier=spec.tile_id_for('D20'), tiles=tiles))
    with pytest.raises(ValueError):
        env.set_schedule(_schedule(env, phase='H', substage=None,
                                   frontier=spec.tile_id_for('H00'),
                                   tiles=[99] * 4))
    with pytest.raises(ValueError):
        env.set_schedule(_schedule(env, phase='H', substage=None,
                                   frontier=spec.tile_id_for('H00'),
                                   tiles=tiles[:3]))
    with pytest.raises(ValueError):
        env.set_schedule(_schedule(env, phase='D', substage='L12',
                                   frontier=spec.tile_id_for('L12'), tiles=tiles))


def test_spawn_and_command_ranges_follow_sds_5_4(env):
    env.evaluation = False
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    assert bool((env.episode_initial_pose[:, 0] >= -1.10 - 1e-6).all())
    assert bool((env.episode_initial_pose[:, 0] <= -0.90 + 1e-6).all())
    assert bool((env.episode_initial_pose[:, 1].abs() <= 0.05 + 1e-6).all())
    assert bool((env.episode_initial_pose[:, 2].abs() <= math.radians(5) + 1e-6).all())
    assert bool((env.episode_command >= 0.30 - 1e-6).all())
    assert bool((env.episode_command <= 0.80 + 1e-6).all())
    for phase in ('F0', 'H', 'D', 'L'):
        env.phase = phase
        env.stage_iteration = 0
        ranges = env._spawn_ranges()
        assert ranges['vx'] == (0.30, 0.80), phase
        assert ranges['x'] == (-1.10, -0.90) and ranges['yaw'] == pytest.approx(
            (-math.radians(5), math.radians(5)))
    env.phase = 'V'
    for iteration, low, high in ((0, 0.30, 0.80), (500, 0.25, 0.70),
                                 (1000, 0.20, 0.60), (2000, 0.20, 0.60)):
        env.stage_iteration = iteration
        ranges = env._spawn_ranges()
        assert ranges['vx'] == pytest.approx((low, high)), iteration
    env.phase = 'R'
    env.stage_iteration = 0
    base = env._spawn_ranges()
    assert base['x'] == pytest.approx((-1.10, -0.90))
    assert base['y'] == pytest.approx((-0.05, 0.05))
    assert base['yaw'] == pytest.approx((-math.radians(5), math.radians(5)))
    env.stage_iteration = 200
    mid = env._spawn_ranges()
    assert mid['x'] == pytest.approx((-1.15, -0.85))
    env.stage_iteration = 400
    final = env._spawn_ranges()
    assert final['x'] == pytest.approx((-1.20, -0.80))
    assert final['y'] == pytest.approx((-0.10, 0.10))
    assert final['yaw'] == pytest.approx((-math.radians(10), math.radians(10)))
    assert final['vx'] == pytest.approx((0.20, 0.60))
    # R flat environments keep the smaller spawn range.
    env.reset()
    flat = env.roles.index('flat')
    assert -1.10 - 1e-6 <= float(env.episode_initial_pose[flat, 0]) <= -0.90 + 1e-6
    assert abs(float(env.episode_initial_pose[flat, 1])) <= 0.05 + 1e-6
    assert abs(float(env.episode_initial_pose[flat, 2])) <= math.radians(5) + 1e-6
    env.phase, env.stage_iteration = 'H', 0


def test_evaluation_cases_and_fixed_speed(env):
    old_phase, old_eval, old_cases, old_command = (
        env.phase, env.evaluation, env.initial_cases, env.initial_command)
    try:
        env.initial_cases = torch.tensor([[-0.5, 0.01, 0.02]]).expand(4, -1).clone()
        env.initial_command = torch.full((4,), 0.6)
        env.fixed_vx = 0.6
        env.evaluation = True
        env.phase = 'R'
        env.stage_iteration = 0
        env.reset()
        torch.testing.assert_close(env.episode_initial_pose[:, :2],
                                   torch.tensor([[-0.5, 0.01]]).expand(4, -1))
        assert bool((env.episode_command == 0.6).all())
        env.step(torch.zeros(4, 12))
        assert env.initial_cases is not None
    finally:
        env.phase, env.evaluation = old_phase, old_eval
        env.initial_cases, env.initial_command = old_cases, old_command
        env.fixed_vx = None
        env.reset()


def test_evaluation_case_dicts_broadcast_to_the_batch(env):
    from src.tasks.velocity.v06.env import _case_tensor
    poses, command = _case_tensor(
        [dict(initial_pose=[-0.7, 0.03, -0.02], vx=0.4)], 8, 'cpu')
    assert tuple(poses.shape) == (8, 3)
    assert tuple(command.shape) == (8,)
    torch.testing.assert_close(poses[0], torch.tensor([-0.7, 0.03, -0.02]))
    torch.testing.assert_close(command, torch.full((8,), 0.4))
    poses, command = _case_tensor(None, 8, 'cpu')
    assert poses is None and command is None
    poses, command = _case_tensor(
        [dict(initial_pose=[-0.7, 0.03, -0.02])], 2, 'cpu')
    assert command is None and tuple(poses.shape) == (2, 3)


def test_env_fall_uses_the_current_tile_local_frame(env):
    env.phase, env.stage_iteration = 'H', 0
    env.evaluation = False
    env.reset()
    index = 1
    tile_id = spec.tile_id_for('D20')
    tile = spec.terrain_spec(tile_id)
    env.tile_ids[index] = tile_id
    n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
    local_x = n * d + 0.5
    local_z = n * h - float(spec.FALL_DROP) - 0.01     # strictly below the limit
    state = env.robot.data.default_root_state[index:index + 1].clone()
    state[0, 0] = tile.frame_origin_world[0] + local_x
    state[0, 1] = tile.frame_origin_world[1]
    state[0, 2] = local_z
    state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[0, 7:] = 0
    env.robot.write_root_state_to_sim(state, torch.tensor([index]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    _, _, _, _, info = env.step(torch.zeros(4, 12))
    assert bool(info['failure_masks']['fall'][index])
    # The criterion ran in the tile frame: local_z = world_z - frame origin.
    diagnostics = info['diagnostics']
    local_written = torch.tensor([local_x, 0.0, local_z])
    assert torch.allclose(diagnostics['local_position'][index], local_written,
                          atol=0.05)
    # (Strict equality / nextafter is covered by the pure events test: a physics
    # step would move the body by gravity, so it is not asserted here.)


def test_stall_diagnostics_are_counted_without_payment(env):
    from src.tasks.velocity.v06.env import StallDiagnostics
    # (a) stationary after the first step: stall samples accumulate.
    stall = StallDiagnostics(1, 'cpu')
    stall.reset(torch.tensor([0]), torch.tensor([-1.0]))
    command = torch.tensor([0.5])
    acquired = torch.tensor([True])
    local = torch.full((1,), -1.0)
    enabled = torch.tensor([True])
    for _ in range(110):
        stall.update(local, command, torch.tensor([False]), acquired, enabled)
    active = int(stall.active_samples[0])
    stalled = int(stall.stall_samples[0])
    assert active == 110
    assert 0 < stalled <= active
    assert int(stall.consecutive_count[0]) >= 50
    # (b) normal forward progress: no stall sample at all.
    stall = StallDiagnostics(1, 'cpu')
    stall.reset(torch.tensor([0]), torch.tensor([-1.0]))
    for step in range(110):
        stall.update(torch.tensor([-1.0 + 0.02 * step]), command,
                     torch.tensor([False]), acquired, enabled)
    assert int(stall.stall_samples[0]) == 0
    assert int(stall.active_samples[0]) == 110
    # (c) before the first step is acquired the same stationary robot never
    # produces a stall sample; after the flag it does.
    stall = StallDiagnostics(1, 'cpu')
    stall.reset(torch.tensor([0]), torch.tensor([-1.0]))
    for _ in range(110):
        stall.update(local, command, torch.tensor([False]),
                     torch.tensor([False]), enabled)
    assert int(stall.stall_samples[0]) == 0
    assert int(stall.active_samples[0]) == 110
    for _ in range(110):
        stall.update(local, command, torch.tensor([False]), acquired, enabled)
    assert int(stall.stall_samples[0]) > 0
    # (d) the environment counts the same way for a stationary robot and the
    # reward never pays for it (no stall term in the agent-A frame/result).
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    env.first_step_acquired[:] = True
    for _ in range(110):
        obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 12))
    index = 1
    assert int(info['active_samples'][index]) > 0
    assert 0 < int(info['stall_samples'][index]) <= int(info['active_samples'][index])
    assert int(env.stall.stall_samples[index]) <= int(env.stall.active_samples[index])
    assert not any('stall' in key.lower()
                   for key in info['step_reward_tensors'])
    weighted = {key: value for key, value in info['step_reward_tensors'].items()
                if key.startswith('reward_weighted/')}
    torch.testing.assert_close(sum(weighted.values()), reward, atol=1e-5, rtol=1e-5)
    # The diagnostic counters are per-environment and exposed for the case
    # runner both as scalars (env 0) and as a per-case list.
    diagnostics = env.diagnostics
    assert tuple(diagnostics['active_samples'].shape) == (4,)
    assert tuple(diagnostics['stall_samples'].shape) == (4,)
    assert tuple(diagnostics['first_step_valid'].shape) == (4,)
    assert tuple(diagnostics['active_samples_by_env'].shape) == (4,)
    assert tuple(diagnostics['stall_samples_by_env'].shape) == (4,)
    cases = env.diagnostics_cases()
    assert len(cases) == 4
    sample = cases[0]
    assert set(sample) >= {'active_samples', 'stall_samples', 'first_step_valid',
                           'vx', 'tile_id'}
    assert isinstance(sample['active_samples'], int)
    assert isinstance(sample['stall_samples'], int)
    assert isinstance(sample['first_step_valid'], bool)
    json.dumps(cases)
    # (e) F0 gate case metrics are available for the deterministic-mean runner.
    env.phase, env.stage_iteration = 'F0', 0
    env.reset()
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 12))
        if bool(terminated.any()) or bool(truncated.any()):
            break
    assert tuple(info['max_abs_y'].shape) == (4,)
    assert tuple(info['max_abs_yaw_deg'].shape) == (4,)
    assert tuple(info['forward_distance'].shape) == (4,)
    assert tuple(info['flat_pass'].shape) == (4,)
    assert tuple(info['active_samples'].shape) == (4,)
    assert tuple(info['stall_samples'].shape) == (4,)
    assert tuple(info['first_step_valid'].shape) == (4,)
    assert info['stall_samples'].dtype == torch.long


def test_evaluation_frontier_uses_current_tick_and_explicit_tile():
    # H00 is forbidden for an H evaluation environment unless stated explicitly.
    with pytest.raises(ValueError):
        V06Env(num_envs=1, device='cpu', seed=42, phase='H', evaluation=True)
    # level_tick=47 -> frontier_row=floor(15*47/100)=7 -> T7 -> tile id 8.
    derived = V06Env(num_envs=2, device='cpu', seed=42, phase='H',
                     evaluation=True, level_tick=47)
    try:
        assert derived.level_tick == 47
        assert derived.frontier_tile_id == 8
        assert derived.tile_ids.tolist() == [8, 8]
        derived.reset()
        assert derived.tile_ids.tolist() == [8, 8], 'eval resets keep the single tile'
        _, _, _, _, info = derived.step(torch.zeros(2, 12))
        assert info['frontier_tile_id'] == 8
        assert int(info['diagnostics']['frontier_tile_id']) == 8
        assert derived.diagnostics_cases()[0]['frontier_tile_id'] == 8
        assert derived.diagnostics_cases()[0]['tile_id'] == 8
    finally:
        del derived
    explicit = V06Env(num_envs=1, device='cpu', seed=42, phase='H',
                      evaluation=True, frontier_tile_id=spec.tile_id_for('H15'))
    try:
        assert explicit.frontier_tile_id == spec.tile_id_for('H15')
        assert explicit.tile_ids.tolist() == [spec.tile_id_for('H15')]
        explicit.reset()
        assert explicit.tile_ids.tolist() == [spec.tile_id_for('H15')]
    finally:
        del explicit
    # A training environment keeps the default H00 frontier (level_tick=0) and
    # the usual mixed roles.
    training = V06Env(num_envs=4, device='cpu', seed=42, phase='H')
    try:
        assert training.frontier_tile_id == spec.tile_id_for('H00')
        frontier = [int(training.tile_ids[i]) for i, role in enumerate(training.roles)
                    if role == 'frontier']
        assert frontier == [spec.tile_id_for('H00')]
    finally:
        del training
    # An explicitly pinned tile must be a legal frontier for its phase.
    with pytest.raises(ValueError):
        V06Env(num_envs=1, device='cpu', seed=42, phase='H', evaluation=True,
               frontier_tile_id=spec.tile_id_for('L40'))


def test_flat_forward_distance_is_signed_net_displacement():
    from src.tasks.velocity.v06.env import flat_case_verdict
    from src.tasks.velocity.v06 import speed

    # Q003 section 3 table: c=.3, threshold 1.5 m.
    rows = [
        # (max_x, x_end, expected_pass, expected_d_net)
        (0.5, 0.5, True, 1.5),        # exactly enough
        (1.0, 0.0, False, 1.0),       # far peak, current position is the truth
        (-0.5, -1.0, False, 0.0),     # back-and-forth, positive path > 1.5
        (0.6, 0.6, True, 1.6),        # retreat then real forward progress
        (-1.0, -1.2, False, -0.2),    # net backwards, field keeps the sign
        (1.0, 0.499999, False, 1.499999),
    ]
    for max_x, end, expected_pass, expected_net in rows:
        result = flat_case_verdict(x_start=-1.0, x_end=end, command=0.3,
                                   steps=500, max_abs_y=0.0,
                                   max_abs_yaw_deg=0.0)
        assert result['d_net'] == pytest.approx(expected_net, abs=1e-12)
        assert result['passed'] is expected_pass, (max_x, end)
        assert result['required'] == pytest.approx(1.5)
        # The same case through C's independent verdict must agree per case.
        metrics = speed.flat_case_metrics(
            speed=0.3, steps=500, max_abs_y=0.0, max_abs_yaw_deg=0.0,
            forward=result['d_net'], failure=False)
        assert speed.flat_case_pass(metrics) is expected_pass
        # A peak-based (old max_body_x) rule would have wrongly passed rows 2/3.
        if max_x - (-1.0) >= 1.5:
            assert max_x - (-1.0) >= 1.5
    # Signed value is exposed unclamped: a net backwards case keeps -0.2.
    backwards = flat_case_verdict(x_start=-1.0, x_end=-1.2, command=0.3, steps=500)
    assert backwards['d_net'] == pytest.approx(-0.2) and not backwards['passed']
    # Thresholds are the 5*c formula: c=.5 -> 2.5 m, c=.8 -> 4.0 m.
    for command, required in ((0.5, 2.5), (0.8, 4.0)):
        exact = flat_case_verdict(x_start=-1.0, x_end=-1.0 + required,
                                  command=command, steps=500)
        assert exact['required'] == pytest.approx(required)
        assert exact['passed']
        short = flat_case_verdict(x_start=-1.0, x_end=-1.0 + required - 1e-6,
                                  command=command, steps=500)
        assert not short['passed']
    # Same net from different starts, and world translation: identical verdict.
    for start in (-0.9, -1.1):
        for end, expected in ((-0.9 + 1.5, True), (-0.9 + 1.0, False)):
            assert flat_case_verdict(x_start=start, x_end=start + (end - (-0.9)),
                                     command=0.3, steps=500)['passed'] is expected
    tile = spec.terrain_spec(spec.tile_id_for('H00'))
    world_start = tile.frame_origin_world[0] - 1.0
    world_end = world_start + 1.5
    shifted = flat_case_verdict(
        x_start=world_start - 1000.0, x_end=world_end - 1000.0,
        command=0.3, steps=500)
    assert shifted['passed'] and shifted['d_net'] == pytest.approx(1.5)
    # Other F0 conditions still veto a sufficient distance.
    assert not flat_case_verdict(x_start=-1.0, x_end=0.6, command=0.3,
                                 steps=500, failure=True)['passed']
    assert not flat_case_verdict(x_start=-1.0, x_end=0.6, command=0.3,
                                 steps=500, max_abs_y=0.21)['passed']
    assert not flat_case_verdict(x_start=-1.0, x_end=0.6, command=0.3,
                                 steps=500, max_abs_yaw_deg=10.1)['passed']
    assert not flat_case_verdict(x_start=-1.0, x_end=0.6, command=0.3,
                                 steps=499)['passed']
    # Recompute from an x0/x500 trace.
    trace = [-1.0 + 1.6 * index / 499 for index in range(500)]
    traced = flat_case_verdict(x_start=trace[0], x_end=trace[-1], command=0.3,
                               steps=len(trace))
    assert traced['d_net'] == pytest.approx(1.6) and traced['passed']


def test_forward_distance_terminal_snapshot_and_partial_reset(env):
    env.phase, env.stage_iteration = 'F0', 0
    env.evaluation = False
    env.reset()
    index, other = 0, 1
    start = float(env.episode_start_local_x[index])
    # The start is the actual post-sense position, not a command constant.
    actual = float(env.robot.data.root_link_pos_w[index, 0]
                   - env.origins[env.tile_ids[index], 0])
    assert env.episode_start_local_x[index] == pytest.approx(actual, abs=1e-5)
    # Step 500 with a real net displacement: a passing flat case.
    env.episode_length_buf[:] = 499
    env.command[:] = 0
    env.command[:, 0] = 0.3
    env.start_vx[:] = 0.3
    env.episode_command[:] = 0.3
    target = start + 1.6
    state = env.robot.data.default_root_state[index:index + 1].clone()
    state[0, 0] = env.origins[env.tile_ids[index], 0] + target
    state[0, 1] = env.origins[env.tile_ids[index], 1]
    state[0, 2] = 0.4
    state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[0, 7:] = 0
    env.robot.write_root_state_to_sim(state, torch.tensor([index]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    _, _, terminated, truncated, info = env.step(torch.zeros(4, 12))
    assert not bool(terminated.any()) and not bool(truncated.any())
    net = float(info['forward_distance'][index])
    assert net == pytest.approx(1.6, abs=0.02)
    assert info['forward_distance_start'][index] == pytest.approx(start, abs=1e-5)
    assert bool(info['flat_pass'][index]) is True
    assert env.episode_length_buf[index].item() == 500
    # Force an automatic reset on the next step and confirm the terminal
    # snapshot is frozen before the reset and the other env is untouched.
    other_start = float(env.episode_start_local_x[other])
    other_frozen_before = float(env.frozen_forward_distance[other])
    env.episode_length_buf[index] = 1499
    class_state = env.robot.data.default_root_state[index:index + 1].clone()
    class_state[0, 0] = env.origins[env.tile_ids[index], 0] + start + 1.7
    class_state[0, 1] = env.origins[env.tile_ids[index], 1]
    class_state[0, 2] = 0.4
    class_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    class_state[0, 7:] = 0
    env.robot.write_root_state_to_sim(class_state, torch.tensor([index]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    _, _, _, trunc, info = env.step(torch.zeros(4, 12))
    assert bool(trunc[index])
    terminal_net = float(info['forward_distance'][index])
    assert terminal_net == pytest.approx(start + 1.7 - start, abs=0.02)
    assert float(info['forward_distance_terminal'][index]) == pytest.approx(
        terminal_net, abs=1e-6)
    record = next(row for row in info['episodes'] if row['env_id'] == index)
    assert record['forward_distance'] == pytest.approx(terminal_net, abs=1e-6)
    assert record['forward_distance_start'] == pytest.approx(start, abs=1e-5)
    assert record['forward_distance_end'] == pytest.approx(
        float(info['forward_distance_end'][index]), abs=1e-6)
    assert record['forward_distance_end'] == pytest.approx(
        float(info['diagnostics']['local_position'][index, 0]), abs=1e-6)
    assert record['terminal_step'] == 1500
    # After the automatic reset the frozen case result is not overwritten by
    # the new episode's 0 start, and env 1 keeps its own start/counter.
    assert env.episode_start_local_x[index] != pytest.approx(start)
    assert float(env.frozen_forward_distance[index]) == pytest.approx(
        terminal_net, abs=1e-6)
    assert float(env.episode_start_local_x[other]) == pytest.approx(other_start)
    if math.isnan(other_frozen_before):
        assert math.isnan(float(env.frozen_forward_distance[other]))
    else:
        assert float(env.frozen_forward_distance[other]) == pytest.approx(
            other_frozen_before)
    # A 499-step early failure uses its own real terminal snapshot and never
    # appends to the next episode.
    env.reset()
    env.episode_length_buf[other] = 498
    other_start = float(env.episode_start_local_x[other])
    fail_state = env.robot.data.default_root_state[other:other + 1].clone()
    fail_state[0, 0] = (env.origins[env.tile_ids[other], 0]
                        + other_start - 0.2)                    # net backwards
    fail_state[0, 1] = env.origins[env.tile_ids[other], 1] + 10.0   # out of bounds
    fail_state[0, 2] = 0.4
    fail_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    fail_state[0, 7:] = 0
    env.robot.write_root_state_to_sim(fail_state, torch.tensor([other]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    _, _, term, _, info = env.step(torch.zeros(4, 12))
    assert bool(term[other])
    record = next(row for row in info['episodes'] if row['env_id'] == other)
    assert record['terminal_step'] == 499
    assert record['forward_distance'] == pytest.approx(
        float(info['forward_distance'][other]), abs=1e-6)
    assert record['forward_distance'] < 0.0   # net backwards keeps its sign
    assert not bool(record['flat_pass'])
    # The next episode starts from its own new start (0 progress), never 499.
    _, _, _, _, info_next = env.step(torch.zeros(4, 12))
    assert abs(float(info_next['forward_distance'][other])) < 0.05


def test_forward_distance_end_is_the_observed_same_frame_position(env):
    env.phase, env.stage_iteration = 'F0', 0
    env.evaluation = False
    env.reset()
    index = 0
    trace = []
    info = None
    for _ in range(500):
        _, _, terminated, truncated, info = env.step(torch.zeros(4, 12))
        local_x = float(env.robot.data.root_link_pos_w[index, 0]
                        - env.origins[env.tile_ids[index], 0])
        trace.append(local_x)
        if bool(terminated.any()) or bool(truncated.any()):
            break
    # Independent per-frame observation of the final local x, never rebuilt from
    # start + distance.
    end = float(info['forward_distance_end'][index])
    start = float(info['forward_distance_start'][index])
    assert end == pytest.approx(trace[-1], abs=1e-6)
    assert end == pytest.approx(float(info['diagnostics']['local_position'][index, 0]),
                                abs=1e-6)
    assert float(info['forward_distance'][index]) == pytest.approx(
        end - start, abs=1e-6)
    assert float(info['diagnostics']['forward_distance'][index]) == pytest.approx(
        end - start, abs=1e-6)
    cases = env.diagnostics_cases()
    assert cases[index]['forward_distance_end'] == pytest.approx(end, abs=1e-6)
    assert cases[index]['forward_distance'] == pytest.approx(end - start, abs=1e-6)
    assert cases[index]['forward_distance'] == pytest.approx(
        cases[index]['forward_distance_end'] - cases[index]['forward_distance_start'],
        abs=1e-6)


def test_forward_distance_per_env_endpoints_and_backwards_case():
    environment = V06Env(num_envs=8, device='cpu', seed=7, phase='F0')
    try:
        environment.reset()
        _, _, _, _, info = environment.step(torch.zeros(8, 12))
        origins = environment.origins[environment.tile_ids, 0]
        observed = (environment.robot.data.root_link_pos_w[:, 0] - origins)
        start = info['forward_distance_start']
        end = info['forward_distance_end']
        torch.testing.assert_close(end, observed, atol=1e-6, rtol=0)
        torch.testing.assert_close(info['forward_distance'], end - start,
                                   atol=1e-6, rtol=0)
        diag = info['diagnostics']
        torch.testing.assert_close(diag['forward_distance_end'], observed,
                                   atol=1e-6, rtol=0)
        torch.testing.assert_close(diag['forward_distance'],
                                   diag['forward_distance_end']
                                   - diag['forward_distance_start'],
                                   atol=1e-6, rtol=0)
        # All eight environments keep their own start/end pair.
        assert len({round(float(value), 6) for value in start}) == 8
        assert len({round(float(value), 6) for value in end}) == 8
        # Net backwards: write a genuinely earlier end and check sign handling.
        index = 0
        target = float(start[index]) - 0.2
        state = environment.robot.data.default_root_state[index:index + 1].clone()
        state[0, 0] = float(origins[index]) + target
        state[0, 1] = float(environment.origins[environment.tile_ids[index], 1])
        state[0, 2] = 0.4
        state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        state[0, 7:] = 0
        environment.robot.write_root_state_to_sim(state, torch.tensor([index]))
        environment.scene.write_data_to_sim()
        environment.sim.forward()
        environment.sim.sense()
        _, _, _, _, info = environment.step(torch.zeros(8, 12))
        assert float(info['forward_distance_start'][index]) == pytest.approx(
            float(start[index]), abs=1e-6)
        assert float(info['forward_distance_end'][index]) < float(
            info['forward_distance_start'][index])
        assert float(info['forward_distance'][index]) == pytest.approx(
            float(info['forward_distance_end'][index])
            - float(info['forward_distance_start'][index]), abs=1e-6)
        assert float(info['forward_distance'][index]) < 0.0
    finally:
        del environment


def test_rng_state_is_resumable_and_sampler_is_json_safe(env):
    sampler = env.sampler_rng_state()
    json.dumps(sampler)
    first = env.sampler_rng_state()
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    second = env.sampler_rng_state()
    assert first != second
    env.load_sampler_rng_state(first)
    assert env.sampler_rng_state() == first
    torch.manual_seed(0)
    state = env.rng_state()
    assert 'torch' in state and torch.is_tensor(state['torch'])
    env.load_rng_state(state)
    value = torch.rand(3, generator=env.rng)
    env.load_rng_state(state)
    torch.testing.assert_close(value, torch.rand(3, generator=env.rng))


def test_scene_geometry_is_the_v06_atlas_only(env):
    index = terrain.index_geometry(env.sim.mj_model, env.specs)
    assert len(index.geom_ids) == sum(
        1 if int(tile.N) == 0 else int(tile.N) + 2 for tile in env.specs)
    assert int(index.tile.max()) == 21
    for tile in env.specs:
        geoms = [g for g in range(env.sim.mj_model.ngeom)
                 if int(index.tile[g]) == int(tile.tile_id)]
        expected = terrain.tile_boxes(tile)
        assert len(geoms) == len(expected)
    assert tuple(env.tile_h.shape) == (22,) and tuple(env.tile_d.shape) == (22,)
    reference = torch.tensor([float(tile.step_depth) for tile in spec.TERRAIN_TABLE])
    torch.testing.assert_close(env.tile_d, reference)
    assert float(env.tile_d[spec.tile_id_for('H00')]) == pytest.approx(0.30)
    assert float(env.tile_d[spec.tile_id_for('D20')]) == pytest.approx(0.20)


def test_curriculum_outcome_roundtrip_from_env_records(env):
    env.phase, env.stage_iteration = 'H', 0
    env.reset()
    env.episode_length_buf[:] = 10 ** 6
    _, _, _, _, info = env.step(torch.zeros(4, 12))
    outcomes = [curriculum.EpisodeOutcome.from_record(record)
                for record in info['episodes']]
    manager = curriculum.CurriculumManager(num_envs=4, seed=42)
    manager.observe(outcomes)
    # The window needs 64 frontier episodes; a four-environment batch keeps them
    # pending and never drops them.
    assert len(manager.state.episode_fifo) == sum(
        1 for record in info['episodes'] if record['role'] == 'frontier')
    assert manager.state.episode_fifo[0]['role'] == 'frontier'
    assert manager.state.episode_fifo[0]['generation'] == env.generation
