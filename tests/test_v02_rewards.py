"""SDS revision 1.1 counterexamples; CPU only, no simulator or learning claim."""
import unittest

import torch

from src.tasks.velocity.v02.rewards import (
    FirstFootfallState, ParkingState, ProgressState, baseline_adjustments,
    step_candidates,
)


def event_fixture(batch=1):
    # ground, step 1, step 2, coplanar last platform, neighboring-tile step 1.
    tile = torch.tensor([1, 1, 1, 1, 2])
    step = torch.tensor([0, 1, 2, 2, 1])
    bounds = torch.tensor([[-2., 0., -.4, .4], [0., .2, -.4, .4],
                           [.2, 2.4, -.4, .4], [.2, 2.4, -.4, .4], [0., .2, -.4, .4]])
    feet = torch.zeros(batch, 4, 2)
    feet[..., 0] = .1
    forces = torch.zeros(batch, 4, 5)
    tiles = torch.ones(batch, dtype=torch.long)
    failure = torch.zeros(batch, dtype=torch.bool)
    state = FirstFootfallState(batch, 2)

    def sample():
        return state.update(tiles, feet, forces, tile, step, bounds, failure)
    return state, feet, forces, tiles, failure, sample, (tile, step, bounds)


def test_first_single_foot_once_without_dt_or_repeat():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 1] = 6
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25
    assert sample().reward.item() == 0
    force.zero_(); sample()
    force[0, 0, 1] = 6
    assert sample().reward.item() == 0 and sample().reward.item() == 0
    assert state.foot_step_rewarded[0, 0].tolist() == [True, False]


def test_ground_to_step_cannot_inherit_one_sample():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 0] = 100
    for _ in range(5):
        assert sample().reward.item() == 0
    assert state.qualified_count[0, 0] == 0
    force.zero_(); force[0, 0, 1] = 6
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25


def test_step_change_cannot_inherit_and_skip_is_not_backfilled():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 1] = 6
    sample()
    feet[0, 0, 0] = .3
    force.zero_(); force[0, 0, 2] = 6
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25
    assert state.foot_step_rewarded[0, 0].tolist() == [False, True]


def test_tile_change_and_different_foot_never_inherit_count():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 1] = 6; sample()
    force.zero_(); force[0, 1, 1] = 6
    assert sample().reward.item() == 0
    tiles[:] = 2
    force.zero_(); force[0, 1, 4] = 6
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25


def test_coplanar_force_sum_and_seam_preserve_key_once():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    feet[0, 0, 0] = .4  # The tread/platform seam is not an outside edge.
    force[0, 0, 2] = 3; force[0, 0, 3] = 3
    assert sample().reward.item() == 0
    force[0, 0, 2] = 0; force[0, 0, 3] = 6
    assert sample().reward.item() == .25
    assert state.last_qualified_step[0, 0] == 2
    assert sample().reward.item() == 0


def test_margin_or_interruption_clears_key_and_restarts_two_samples():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 1] = 6; sample()
    feet[0, 0, 0] = .041
    assert sample().reward.item() == 0
    assert state.last_qualified_step[0, 0] == -1 and state.qualified_count[0, 0] == 0
    feet[0, 0, 0] = .1
    assert sample().reward.item() == 0
    force.zero_(); sample()
    force[0, 0, 1] = 6
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25


def test_strict_force_threshold_neighbor_exclusion_and_safe_margin():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 1] = 5; force[0, 0, 4] = 100
    sample(); assert sample().reward.item() == 0
    force[0, 0, 1] = 6; feet[0, 0, 1] = .359
    sample(); assert sample().reward.item() == 0
    feet[0, 0, 1] = 0; feet[0, 0, 0] = .042
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25


def test_multicandidate_maxforce_then_lower_step_independent_of_geom_order():
    tiles = torch.tensor([8]); feet = torch.zeros(1, 4, 2)
    tile = torch.tensor([8, 8, 8]); step = torch.tensor([2, 1, 1])
    bounds = torch.tensor([[-1., 1., -1., 1.]] * 3)
    force = torch.tensor([[[7., 3., 4.]] * 4])
    chosen, _ = step_candidates(tiles, feet, force, tile, step, bounds, 2)
    assert chosen.tolist() == [[1] * 4]
    force[..., 0] = 8
    chosen, _ = step_candidates(tiles, feet, force, tile, step, bounds, 2)
    assert chosen.tolist() == [[2] * 4]


def test_failure_suppresses_event_and_does_not_mark_as_rewarded():
    state, feet, force, tiles, failure, sample, _ = event_fixture()
    force[0, 0, 1] = 6; sample(); failure[:] = True
    assert sample().reward.item() == 0
    assert not state.foot_step_rewarded.any()
    # A terminal sample updates debounce before reward suppression; automatic
    # episode reset must clear it before another episode's first sample.
    assert state.qualified_count[0, 0] == 2
    state.reset(torch.tensor([0])); failure[:] = False
    assert sample().reward.item() == 0
    assert sample().reward.item() == .25


def test_event_reset_per_environment_and_ab_path_is_inactive():
    state, feet, force, tiles, failure, sample, metadata = event_fixture(2)
    force[:, 0, 1] = 6; sample(); sample()
    state.reset(torch.tensor([0]))
    assert not state.foot_step_rewarded[0].any() and state.foot_step_rewarded[1, 0, 0]
    assert state.last_qualified_tile[0].eq(-1).all() and state.qualified_count[0].eq(0).all()
    before = [t.clone() for t in (force, *metadata, state.qualified_count)]
    result = state.update(tiles, feet, force, *metadata, failure, enabled=False)
    assert result.reward.eq(0).all()
    for old, new in zip(before, (force, *metadata, state.qualified_count)):
        torch.testing.assert_close(old, new)
    assert sample().reward.sum() == 0
    assert sample().reward.tolist() == [.25, 0.]


def test_actual_contact_filter_rejects_side_underside_calf_and_bad_normal():
    from src.tasks.velocity.mdp.ideal_stairs_contacts import reduce_contacts
    # Actual contact extraction is shared and unmodified. Only contact #0 is a
    # qualified football top; calf remains a nonfoot collision.
    geom = torch.tensor([[0, 1]] * 4 + [[0, 2]])
    point = torch.tensor([[0., 0., .08], [0., 0., .07], [0., 0., .08], [0., 0., .08], [0., 0., .08]])
    normal = torch.tensor([[0., 0., 1.], [0., 0., 1.], [1., 0., 0.], [0., 0., .85], [0., 0., 1.]])
    _, tops, links = reduce_contacts(geom, torch.zeros(5, dtype=torch.long), point, normal,
        torch.tensor([[0., 0., 6.]] * 5), torch.ones(5, dtype=torch.bool),
        torch.tensor([-1, 0, -1]), torch.tensor([-1, -1, 0]), torch.tensor([1, -1, -1]),
        torch.tensor([.08, 0., 0.]), torch.tensor([1]), 1)
    assert tops.sum().item() == 6 and links.sum().item() == 6


def test_generic_support_unchanged_by_event_debounce():
    from src.tasks.velocity.mdp.ideal_stairs_contacts import support_from_top_forces
    state, feet, force, tiles, failure, sample, metadata = event_fixture()
    force[0, 0, 0] = 6
    before = support_from_top_forces(force, torch.tensor([0., .1, .2, .2, .1]), metadata[0])
    sample(); force[0, 0, 0] = 0; force[0, 0, 1] = 6
    expected = support_from_top_forces(force, torch.tensor([0., .1, .2, .2, .1]), metadata[0])
    assert sample().reward.item() == 0  # generic support could already count 2.
    after = support_from_top_forces(force, torch.tensor([0., .1, .2, .2, .1]), metadata[0])
    for a, b in zip(expected, after): torch.testing.assert_close(a, b)
    assert before[0][0, 0] == after[0][0, 0] == 6


def progress_fixture(batch=1):
    state = ProgressState(batch); x = torch.zeros(batch); state.reset(torch.arange(batch), x)
    args = dict(body_x=x, x_stop=torch.ones(batch), vx_cmd=torch.full((batch,), .4),
                support=torch.ones(batch, 4, dtype=torch.bool), active=torch.ones(batch, dtype=torch.bool),
                stairs=torch.ones(batch, dtype=torch.bool), stop_latched=torch.zeros(batch, dtype=torch.bool),
                failure=torch.zeros(batch, dtype=torch.bool))
    return state, args


def test_progress_cap_no_catchup_no_double_dt_and_backtracking():
    state, args = progress_fixture()
    args['body_x'][:] = .5
    torch.testing.assert_close(state.update(**args), torch.tensor([.016]))
    assert state.update(**args).item() == 0
    args['body_x'][:] = .2; assert state.update(**args).item() == 0
    args['body_x'][:] = .5; assert state.update(**args).item() == 0
    args['body_x'][:] = .504
    torch.testing.assert_close(state.update(**args), torch.tensor([.008]), atol=1e-7, rtol=0)
    args['body_x'][:] = 5; state.update(**args)
    assert state.x_max.item() == 1 and state.update(**args).item() == 0


def test_progress_ineligible_motion_still_consumes_highwater():
    for key, value in [('support', False), ('active', False), ('stairs', False),
                       ('stop_latched', True), ('failure', True)]:
        state, args = progress_fixture(); original = args[key].clone()
        args[key][:] = value; args['body_x'][:] = .2
        assert state.update(**args).item() == 0
        args[key].copy_(original)
        assert state.update(**args).item() == 0
        args['body_x'][:] = .204
        assert state.update(**args).item() > 0


def test_progress_ground_support_negative_cmd_disabled_reset_and_isolation():
    state, args = progress_fixture(2)
    # Generic ground support qualifies, exactly two feet suffice.
    args['support'][:, 2:] = False; args['body_x'][:] = .004
    torch.testing.assert_close(state.update(**args), torch.tensor([.008, .008]))
    args['vx_cmd'][:] = -.4; args['body_x'][:] = .1
    assert state.update(**args).eq(0).all()
    state.reset(torch.tensor([0]), torch.tensor([-.5]))
    assert state.x_max.tolist() == [-.5, float(args['body_x'][1])]
    args['body_x'][:] = .2
    assert state.update(**args, enabled=False).eq(0).all()
    assert state.x_max.eq(.2).all()


def parking_fixture(batch=1):
    state = ParkingState(batch)
    args = dict(stairs=torch.ones(batch, dtype=torch.bool), body_x=torch.full((batch,), 1.),
                x_stop=torch.ones(batch), all_feet_safe=torch.ones(batch, dtype=torch.bool),
                all_platform_support_safe=torch.ones(batch, dtype=torch.bool), body_y=torch.zeros(batch),
                speed=torch.zeros(batch), failure=torch.zeros(batch, dtype=torch.bool))
    return state, args


def test_parking_requires_latch_and_current_stop_position_all_recipes():
    for recipe in 'ABCD':
        state, args = parking_fixture(); args['body_x'][:] = .99999
        for _ in range(151): result = state.update(**args)
        assert not result.stop_latched.item() and not result.success.item(), recipe
        # No latch means the caller continues using its original nonzero command.
        assert result.parking_steps.item() == 0


def test_parking_retreat_keeps_zero_command_and_requires_new_150_cycles():
    state, args = parking_fixture()
    for _ in range(149): assert not state.update(**args).success.item()
    args['body_x'][:] = .99999
    result = state.update(**args)
    assert result.stop_latched.item() and result.parking_steps.item() == 0 and not result.success.item()
    args['body_x'][:] = 1
    for _ in range(149): assert not state.update(**args).success.item()
    assert state.update(**args).success.item()
    args['failure'][:] = True
    result = state.update(**args)
    assert result.parking_steps.item() == 0 and not result.success.item()


def test_parking_rejects_frontfeet_only_body_only_short_contact_speed_y_and_flat():
    for key, value in [('all_feet_safe', False), ('all_platform_support_safe', False),
                       ('speed', .05), ('body_y', .20001), ('stairs', False), ('failure', True)]:
        state, args = parking_fixture(); args[key][:] = value
        for _ in range(151): assert not state.update(**args).success.item(), key
    state, args = parking_fixture()
    state.update(**args); args['all_platform_support_safe'][:] = False
    assert state.update(**args).parking_steps.item() == 0


def test_parking_reset_is_per_environment_and_latch_projection_needs_no_contact():
    state, args = parking_fixture(2)
    args['all_platform_support_safe'][:] = False
    result = state.update(**args)
    assert result.stop_latched.all() and result.parking_steps.eq(0).all()
    args['all_platform_support_safe'][:] = True
    state.update(**args); state.reset(torch.tensor([0]))
    assert state.stop_latched.tolist() == [False, True]
    assert state.parking_steps.tolist() == [0, 1]


def test_baseline_stationary_active_and_parking_original_terms_ab_identical():
    command = torch.tensor([[.4, 0., 0.], [0., 0., 0.]])
    linear = torch.exp(-command[:, :2].square().sum(-1) / .25)
    angular = torch.ones(2); posture = torch.ones(2)
    result = baseline_adjustments(linear, angular, posture, command)
    assert all(term[0] == 0 for term in result['adjusted'].values())
    assert all(term[1] == 1 for term in result['adjusted'].values())
    for recipe in 'AB':
        result = baseline_adjustments(linear, angular, posture, command, enabled=False)
        for key in result['raw']:
            torch.testing.assert_close(result['raw'][key], result['adjusted'][key])
            assert result['base'][key].eq(0).all(), recipe


class TestV02Rewards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)


for _name, _test in list(globals().items()):
    if _name.startswith('test_') and callable(_test):
        setattr(TestV02Rewards, _name, lambda self, fn=_test: fn())

del _name, _test

if __name__ == '__main__':
    unittest.main()
