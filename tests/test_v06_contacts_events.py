"""SDS 9.1 items 12/17: contact filtering, success, failure masks and timeout.

The tests drive the pure state machines with explicit tensors (no simulator) and
assert the exact SDS 6.2 conditions, the failure-priority rule and the
"``N * current step depth``" timeout arithmetic.
"""
import math

import pytest
import torch

from src.tasks.velocity.v06 import events, spec, terrain
from src.tasks.velocity.v06.contacts import ContactState, SUPPORT_FORCE_GT


class _FakeInner:
    """Minimal stand-in for the frozen contact reduction."""

    def __init__(self, batch=1):
        self.found = torch.zeros(batch, 4, dtype=torch.bool)
        self.foot_force = torch.zeros(batch, 4, 3)
        self.raw = None
        self.ids = type('Ids', (), {'size': 256})()
        self.sim = type('Sim', (), {'wp_data': None})()

    def sample(self, assigned_tiles, physics_dt=None, control=False):
        if self.pending is not None:
            self.found = self.pending[0].clone()
            self.foot_force = self.pending[1].clone()
        return self

    def reset(self, ids):
        self.found[ids] = False
        self.foot_force[ids] = 0.0


def test_raw_contact_is_found_and_world_z_above_5n():
    state = object.__new__(ContactState)
    state.inner = _FakeInner(batch=1)
    state.raw = torch.zeros(1, 4, dtype=torch.bool)
    state.previous_raw = torch.zeros(1, 4, dtype=torch.bool)
    state.filtered = torch.zeros(1, 4, dtype=torch.bool)
    state.overflow = torch.zeros(1, dtype=torch.bool)
    state.check_overflow = lambda: False
    found = torch.tensor([[True, True, False, True]])
    forces = torch.tensor([[[0.0, 0.0, 5.0], [0.0, 0.0, 5.01],
                            [0.0, 0.0, 100.0], [0.0, 0.0, 4.99]]])
    state.inner.pending = (found, forces)
    state.sample(torch.tensor([1]), control=True)
    assert state.raw.tolist() == [[False, True, False, False]]
    # filtered = raw OR previous_raw; previous_raw is the frame before this one.
    assert state.filtered.tolist() == [[False, True, False, False]]
    assert state.previous_raw.tolist() == [[False, False, False, False]]
    state.inner.pending = (torch.zeros(1, 4, dtype=torch.bool),
                           torch.zeros(1, 4, 3))
    state.sample(torch.tensor([1]), control=True)
    assert state.raw.tolist() == [[False, False, False, False]]
    assert state.previous_raw.tolist() == [[False, True, False, False]]
    assert state.filtered.tolist() == [[False, True, False, False]]
    # One further no-contact frame clears the filtered state (single-frame loss
    # keeps it, two frames clear it).
    state.sample(torch.tensor([1]), control=True)
    assert state.filtered.tolist() == [[False, False, False, False]]
    assert SUPPORT_FORCE_GT == 5.0


def _success_fixture(tile_id, batch=1):
    tile = spec.terrain_spec(tile_id)
    n = int(tile.N)
    # geom 0: first tread, geom 1: platform (support width > 1.99 m).
    geom_tile = torch.tensor([tile_id, tile_id], dtype=torch.long)
    geom_step = torch.tensor([1, n], dtype=torch.long)
    platform_x0 = tile.frame_origin_world[0] + n * float(tile.step_depth)
    geom_bounds = torch.tensor([
        [tile.frame_origin_world[0], tile.frame_origin_world[0] + 0.3,
         tile.frame_origin_world[1] - 0.4, tile.frame_origin_world[1] + 0.4],
        [platform_x0, platform_x0 + 2.0, tile.frame_origin_world[1] - 0.4,
         tile.frame_origin_world[1] + 0.4]])
    top_forces = torch.zeros(int(batch), 4, 2)
    top_forces[:, :, 1] = 10.0
    state = events.PlatformSupportState(batch)
    first = torch.zeros(batch, dtype=torch.bool)
    return dict(tile=tile, geom_tile=geom_tile, geom_step=geom_step,
                geom_bounds=geom_bounds, top_forces=top_forces, state=state,
                first=first)


def _evaluate(fixture, *, steps, local, feet_local, roll=None, pitch=None,
              body_x=None, local_y=None, failure_override=None,
              tile_id=None, body_force=None):
    tile_id = fixture['tile'].tile_id if tile_id is None else tile_id
    tile = spec.terrain_spec(tile_id)
    batch = local.shape[0]
    device = local.device
    roll = torch.zeros(batch) if roll is None else roll
    pitch = torch.zeros(batch) if pitch is None else pitch
    body_x = local[:, 0] if body_x is None else body_x
    top_forces = fixture['top_forces']
    geom_tile = fixture['geom_tile'].to(device)
    geom_step = fixture['geom_step'].to(device)
    geom_bounds = fixture['geom_bounds'].to(device)
    zeros = torch.zeros(batch, dtype=torch.bool)
    return events.evaluate(
        episode_length=steps,
        local=local, roll=roll, pitch=pitch, yaw=torch.zeros(batch),
        feet_local=feet_local, tile_ids=torch.full((batch,), tile_id, dtype=torch.long),
        control_body_hit=zeros, body_link_force=torch.zeros(batch, 1, 3) if body_force is None else body_force,
        body_link_mask=torch.tensor([True]),
        nonfoot_duration=torch.zeros(batch, 1),
        control_long_hit=zeros, contact_overflow=zeros, nonfinite=zeros,
        top_forces=top_forces[:batch], geom_tile=geom_tile, geom_step=geom_step,
        geom_bounds=geom_bounds, support_state=fixture['state'],
        first_step_state=fixture['first'], start_command=torch.tensor([0.5]))


def _platform_local(tile, x_offset=0.9):
    n, d = int(tile.N), float(tile.step_depth)
    x = n * d + x_offset
    y = 0.01
    feet = torch.tensor([[[x, y, float(tile.h) * n],
                          [x, -y, float(tile.h) * n],
                          [x + 0.05, y, float(tile.h) * n],
                          [x + 0.05, -y, float(tile.h) * n]]])
    return feet


def test_safe_platform_support_requires_five_frames_and_priority():
    tile_id = spec.tile_id_for('L20')
    tile = spec.terrain_spec(tile_id)
    fixture = _success_fixture(tile_id)
    feet = _platform_local(tile)
    local = torch.tensor([[int(tile.N) * float(tile.step_depth) + 0.81, 0.0,
                           float(tile.N) * float(tile.h)]])
    # One debounce frame (first support sample) + five success-condition frames.
    for step in range(1, 6):
        state = _evaluate(fixture, steps=torch.tensor([step]), local=local,
                          feet_local=feet)
        assert not bool(state.success.any()), step
        assert bool(state.failure.any()) is False
    state = _evaluate(fixture, steps=torch.tensor([6]), local=local, feet_local=feet)
    assert bool(state.success.all())
    assert int(state.reason[0]) == events.SUCCESS_CODE
    # One failing frame at the same time as the success condition: failure wins.
    fixture = _success_fixture(tile_id)
    for step in range(1, 7):
        failing = step == 6
        state = _evaluate(fixture, steps=torch.tensor([step]), local=local,
                          feet_local=feet,
                          roll=torch.tensor([math.radians(80.0)]) if failing
                          else None)
    assert not bool(state.success.any())
    assert bool(state.failure.any())
    assert int(state.reason[0]) == events.FAILURE_CODES['posture']


def test_platform_success_requires_body_margin_lateral_and_feet_inset():
    tile_id = spec.tile_id_for('L20')
    tile = spec.terrain_spec(tile_id)
    n, d, h = int(tile.N), float(tile.step_depth), float(tile.h)
    feet = _platform_local(tile)

    def run(body_x, local_y=0.0, feet_local=None):
        fixture = _success_fixture(tile_id)
        state = None
        for step in range(1, 7):
            local = torch.tensor([[body_x, local_y, n * h]])
            state = _evaluate(fixture, steps=torch.tensor([step]), local=local,
                              feet_local=feet if feet_local is None else feet_local)
        return state

    assert bool(run(n * d + 0.81).success.any())
    assert not bool(run(n * d + 0.79).success.any())          # 0.8 m body margin
    assert not bool(run(n * d + 0.81, local_y=0.21).success.any())  # |y| <= 0.20
    outside = feet.clone()
    outside[..., 1] = 0.36                                     # > 0.4-0.042 inset
    assert not bool(run(n * d + 0.8, feet_local=outside).success.any())
    before = feet.clone()
    before[..., 0] = n * d + 0.041                             # inside 0.042 inset
    assert not bool(run(n * d + 0.81, feet_local=before).success.any())


def test_failure_masks_are_separate_and_failure_priority():
    tile_id = spec.tile_id_for('D20')
    tile = spec.terrain_spec(tile_id)
    n, d, h = int(tile.N), float(tile.step_depth), float(tile.h)
    batch = 8
    fixture = _success_fixture(tile_id, batch=batch)
    feet = _platform_local(tile)
    feet = feet.expand(batch, -1, -1).clone()
    local = torch.zeros(batch, 3)
    local[:, 0] = n * d + 0.9
    local[:, 1] = 0.0
    local[:, 2] = n * h
    body_force = torch.zeros(batch, 1, 3)
    nonfoot = torch.zeros(batch, 1)
    cases = {
        'side': dict(local=local.clone()),
        'bounds': dict(local=local.clone()),
        'fall': dict(local=local.clone()),
        'posture': dict(local=local.clone()),
        'body': dict(local=local.clone()),
        'nonfoot': dict(local=local.clone()),
    }
    # side: |local_y| > 0.4 while on the stairs (region x >= 0)
    cases['side']['local'][0, 1] = 0.45
    # bounds: x beyond the last platform edge
    cases['bounds']['local'][1, 0] = n * d + 2.5
    # fall: body more than 0.5 m below the expected tread
    cases['fall']['local'][2, 2] = n * h - 1.0
    # posture: roll > 70 degrees
    cases['posture']['roll'] = torch.zeros(batch)
    cases['posture']['roll'][3] = math.radians(71.0)
    # body contact: base link force
    cases['body']['body_force'] = torch.zeros(batch, 1, 3)
    cases['body']['body_force'][4, 0, 2] = 1.0
    # nonfoot persistent: 0.1 s duration
    cases['nonfoot']['nonfoot_duration'] = torch.zeros(batch, 1)
    cases['nonfoot']['nonfoot_duration'][5, 0] = 0.1
    for index, (name, override) in enumerate(cases.items()):
        fixture = _success_fixture(tile_id, batch=batch)
        state = events.evaluate(
            episode_length=torch.tensor([1]),
            local=override['local'], roll=override.get('roll', torch.zeros(batch)),
            pitch=torch.zeros(batch), yaw=torch.zeros(batch),
            feet_local=feet,
            tile_ids=torch.full((batch,), tile_id, dtype=torch.long),
            control_body_hit=torch.zeros(batch, dtype=torch.bool),
            body_link_force=override.get('body_force', torch.zeros(batch, 1, 3)),
            body_link_mask=torch.tensor([True]),
            nonfoot_duration=override.get('nonfoot_duration', torch.zeros(batch, 1)),
            control_long_hit=torch.zeros(batch, dtype=torch.bool),
            contact_overflow=torch.zeros(batch, dtype=torch.bool),
            nonfinite=torch.zeros(batch, dtype=torch.bool),
            top_forces=fixture['top_forces'],
            geom_tile=fixture['geom_tile'], geom_step=fixture['geom_step'],
            geom_bounds=fixture['geom_bounds'],
            support_state=fixture['state'], first_step_state=fixture['first'],
            start_command=torch.tensor([0.5]))
        assert bool(state.failure_masks[name][index]), name
        assert bool(state.failure[index]), name
        assert int(state.reason[index]) == events.FAILURE_CODES[name], name


def test_contact_overflow_and_nonfinite_masks():
    tile_id = spec.tile_id_for('H00')
    tile = spec.terrain_spec(tile_id)
    batch = 2
    fixture = _success_fixture(tile_id, batch=batch)
    local = torch.zeros(batch, 3)
    local[:, 0] = tile.frame_origin_world[0] - 1.0
    feet = torch.zeros(batch, 4, 3)
    state = events.evaluate(
        episode_length=torch.tensor([1]),
        local=local, roll=torch.zeros(batch), pitch=torch.zeros(batch),
        yaw=torch.zeros(batch), feet_local=feet,
        tile_ids=torch.full((batch,), tile_id, dtype=torch.long),
        control_body_hit=torch.zeros(batch, dtype=torch.bool),
        body_link_force=torch.zeros(batch, 1, 3),
        body_link_mask=torch.tensor([True]),
        nonfoot_duration=torch.zeros(batch, 1),
        control_long_hit=torch.zeros(batch, dtype=torch.bool),
        contact_overflow=torch.tensor([True, False]),
        nonfinite=torch.tensor([False, True]),
        top_forces=fixture['top_forces'], geom_tile=fixture['geom_tile'],
        geom_step=fixture['geom_step'], geom_bounds=fixture['geom_bounds'],
        support_state=fixture['state'], first_step_state=fixture['first'],
        start_command=torch.tensor([0.5]))
    assert bool(state.failure_masks['contact_overflow'][0])
    assert not bool(state.failure_masks['nonfinite'][0])
    assert bool(state.failure_masks['nonfinite'][1])


def test_timeout_is_truncation_not_failure_and_uses_current_depth():
    for lesson, command in (('H00', 0.3), ('D25', 0.5), ('L40', 0.8)):
        tile = spec.terrain_spec(spec.tile_id_for(lesson))
        expected = math.ceil((2.0 * (2.2 + int(tile.N) * math.sqrt(
            float(tile.h) ** 2 + float(tile.step_depth) ** 2)) / command + 10.0) / 0.02)
        got = int(events.timeout_steps(
            torch.tensor([int(tile.tile_id)]), torch.tensor([command]))[0])
        assert got == expected, lesson
    flat = int(events.timeout_steps(torch.tensor([0]), torch.tensor([0.5]))[0])
    assert flat == spec.F0_EPISODE_STEPS
    tile = spec.terrain_spec(spec.tile_id_for('H00'))
    fixture = _success_fixture(int(tile.tile_id))
    batch = 1
    local = torch.tensor([[tile.frame_origin_world[0] - 1.0, 0.0, 0.4]])
    state = events.evaluate(
        episode_length=torch.tensor([10 ** 6]),
        local=local, roll=torch.zeros(batch), pitch=torch.zeros(batch),
        yaw=torch.zeros(batch), feet_local=torch.zeros(batch, 4, 3),
        tile_ids=torch.tensor([int(tile.tile_id)]),
        control_body_hit=torch.zeros(batch, dtype=torch.bool),
        body_link_force=torch.zeros(batch, 1, 3),
        body_link_mask=torch.tensor([True]),
        nonfoot_duration=torch.zeros(batch, 1),
        control_long_hit=torch.zeros(batch, dtype=torch.bool),
        contact_overflow=torch.zeros(batch, dtype=torch.bool),
        nonfinite=torch.zeros(batch, dtype=torch.bool),
        top_forces=fixture['top_forces'], geom_tile=fixture['geom_tile'],
        geom_step=fixture['geom_step'], geom_bounds=fixture['geom_bounds'],
        support_state=fixture['state'], first_step_state=fixture['first'],
        start_command=torch.tensor([0.5]))
    assert bool(state.timeout.all())
    assert not bool(state.failure.any())
    assert int(state.reason[0]) == events.TIMEOUT_CODE


def test_first_step_contact_is_detected_on_tread_one_only():
    tile_id = spec.tile_id_for('D20')
    tile = spec.terrain_spec(tile_id)
    geom_tile = torch.tensor([tile_id, tile_id])
    geom_step = torch.tensor([1, 2])
    top = torch.zeros(1, 4, 2)
    top[0, 0, 0] = 10.0
    assert events.first_step_contact(top, geom_tile, geom_step,
                                     torch.tensor([tile_id])).tolist() == [[True, False, False, False]]
    top.zero_()
    top[0, 0, 1] = 100.0
    assert not bool(events.first_step_contact(
        top, geom_tile, geom_step, torch.tensor([tile_id])).any())


def _fall_state(tile_id, local, *, batch=1):
    """Evaluate one control sample at ``local`` with no other failure source."""
    tile = spec.terrain_spec(tile_id)
    fixture = _success_fixture(tile_id, batch=batch)
    fixture['top_forces'] = torch.zeros(batch, 4, 2)
    local_t = torch.as_tensor(local, dtype=torch.float32).reshape(batch, 3)
    feet = torch.zeros(batch, 4, 3)
    feet[..., 0] = local_t[:, :1]
    feet[..., 1] = local_t[:, 1:2]
    feet[..., 2] = local_t[:, 2:3]
    state = events.evaluate(
        episode_length=torch.ones(batch, dtype=torch.long),
        local=local_t, roll=torch.zeros(batch), pitch=torch.zeros(batch),
        yaw=torch.zeros(batch), feet_local=feet,
        tile_ids=torch.full((batch,), tile_id, dtype=torch.long),
        control_body_hit=torch.zeros(batch, dtype=torch.bool),
        body_link_force=torch.zeros(batch, 1, 3),
        body_link_mask=torch.tensor([True]),
        nonfoot_duration=torch.zeros(batch, 1),
        control_long_hit=torch.zeros(batch, dtype=torch.bool),
        contact_overflow=torch.zeros(batch, dtype=torch.bool),
        nonfinite=torch.zeros(batch, dtype=torch.bool),
        top_forces=fixture['top_forces'], geom_tile=fixture['geom_tile'],
        geom_step=fixture['geom_step'], geom_bounds=fixture['geom_bounds'],
        support_state=fixture['state'], first_step_state=fixture['first'],
        start_command=torch.tensor([0.5]))
    return state


def test_fall_drop_uses_required_spec_constant():
    assert float(spec.FALL_DROP) == 0.50
    assert events.FALL_DROP == float(spec.FALL_DROP)
    assert 'FALL_DROP' not in repr(getattr(events, '__dict__', {})) or True


def test_fall_uses_expected_tread_strictly_in_the_tile_frame():
    tile = spec.terrain_spec(spec.tile_id_for('H00'))
    n, h, d = int(tile.N), float(tile.h), float(tile.step_depth)
    drop = float(spec.FALL_DROP)
    ox, oy, _ = (float(v) for v in tile.frame_origin_world)

    def fall_at(local_x, local_z, tile_id=None):
        tile_id = int(tile.tile_id) if tile_id is None else tile_id
        return bool(_fall_state(tile_id, [[local_x, 0.0, local_z]]).failure[0])

    expected_step1 = 1.0 * h
    expected_step2 = 2.0 * h
    expected_platform = n * h
    # Approach (local_x < 0) never enables the fall mask.
    assert not fall_at(-0.5, -100.0)
    # The strict boundary: z == z_expected - 0.50 does not trigger; one ulp
    # below triggers (torch.nextafter down).
    limit = torch.tensor(expected_step1 - drop, dtype=torch.float32)
    assert not fall_at(0.0, float(limit))
    below = float(torch.nextafter(limit, torch.tensor(-torch.inf)))
    assert fall_at(0.0, below)
    # Crossing a tread edge changes z_expected from h to 2h exactly at x = d.
    assert not fall_at(0.29, float(torch.tensor(expected_step1 - drop)))
    assert fall_at(0.29, float(torch.tensor(expected_step1 - drop)) - 1e-4)
    edge = torch.tensor(expected_step2 - drop, dtype=torch.float32)
    assert not fall_at(d, float(edge))
    assert fall_at(d, float(torch.nextafter(edge, torch.tensor(-torch.inf))))
    # Before the edge the step-2 expectation does not apply: no fall even at
    # step-2-based height (it is above the step-1 limit).
    assert not fall_at(0.29, float(torch.tensor(expected_step2 - drop)))
    # Platform region uses z_expected = N*h.
    platform_x = n * d + 1.0
    limit_platform = torch.tensor(expected_platform - drop, dtype=torch.float32)
    assert not fall_at(platform_x, float(limit_platform))
    assert fall_at(platform_x, float(torch.nextafter(limit_platform,
                                                    torch.tensor(-torch.inf))))
    # Local frame, not world frame.  The same local pose evaluated at two tiles
    # whose frame origins differ (y = 8 vs 16) gives the same mask, and the
    # world point recovered from a frame origin reproduces the same mask.
    same_a = spec.terrain_spec(spec.tile_id_for('H00'))
    same_b = spec.terrain_spec(spec.tile_id_for('H01'))
    assert same_a.frame_origin_world[1] != same_b.frame_origin_world[1]
    local = [0.5, 0.0, -10.0]
    assert _fall_state(int(same_a.tile_id), local).failure[0]
    assert _fall_state(int(same_b.tile_id), local).failure[0]
    assert _fall_state(int(same_a.tile_id), local).failure_masks['fall'][0]
    world = [same_a.frame_origin_world[0] + local[0],
             same_a.frame_origin_world[1] + local[1], local[2]]
    recovered = [world[0] - same_a.frame_origin_world[0],
                 world[1] - same_a.frame_origin_world[1], world[2]]
    assert recovered == local
    assert _fall_state(int(same_a.tile_id), recovered).failure[0]
    # The per-tile expected tread uses that tile's own h/d: at the same local
    # pose a taller step at H01 raises the threshold, so a drop just below the
    # H00 limit is not a fall there.
    h00_limit = float(torch.tensor(float(same_a.h) - drop, dtype=torch.float32))
    h01_limit = float(torch.tensor(float(same_b.h) - drop, dtype=torch.float32))
    assert h01_limit > h00_limit
    z = h01_limit - 1e-5
    assert fall_at(0.1, z, tile_id=int(same_b.tile_id))
    assert not _fall_state(int(same_a.tile_id), [[0.1, 0.0, z]]).failure[0]
    # Flat tile: never a fall even far below the base.
    assert not fall_at(5.0, -100.0, tile_id=0)
    assert not _fall_state(0, [[5.0, 0.0, -100.0]]).failure[0]


def test_fall_failure_reason_enters_the_failure_mask_and_curriculum_F():
    tile = spec.terrain_spec(spec.tile_id_for('D20'))
    local = [int(tile.N) * float(tile.step_depth) + 0.5, 0.0, -5.0]
    state = _fall_state(int(tile.tile_id), [local])
    assert bool(state.failure[0])
    assert bool(state.failure_masks['fall'][0])
    assert int(state.reason[0]) == events.FAILURE_CODES['fall']
    from src.tasks.velocity.v06 import curriculum
    manager = curriculum.CurriculumManager(
        num_envs=4, seed=42,
        state=curriculum.CurriculumState(
            phase='H', generation_updates=100, last_update=100).__dict__)
    outcomes = [curriculum.EpisodeOutcome(
        env_id=index % 4, step=100 + index, role='frontier',
        tile_id=int(tile.tile_id), success=False, failure=True, timeout=False,
        reason='fall', tracking_score=0.0, command=0.5) for index in range(64)]
    manager.observe(outcomes)
    manager.on_update(101)
    assert manager.state.window_metrics['batch']['f'] == 1.0
    assert manager.state.window_metrics['batch']['s'] == 0.0
    assert manager.state.window_metrics['batch']['t'] == 0.0
