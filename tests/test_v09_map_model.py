"""CPU algebra/input/optimizer checks; these do not certify S1 behavior."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap  # noqa: E402
a2_v07_bootstrap.activate(ROOT)

from src.tasks.velocity.v09 import map as maps, model as target  # noqa: E402


@pytest.fixture(scope='module')
def migrated():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = target.V09ActorCritic.migrate_6400_checkpoint()
    source = torch.load(target.SOURCE_CHECKPOINT, map_location='cpu', weights_only=False)['model']
    yield model, source
    torch.set_num_threads(old_threads)


def test_grid_centers_rotation_and_invalid_channels():
    def surface(points, tile_ids):
        z = points[..., 0] + 2 * points[..., 1]
        valid = points[..., 0] < 0
        return torch.where(valid, z, float('nan')), valid

    cache = maps.HeightMap(2, surface_query=surface)
    assert cache.grid.shape == (40, 30, 2)
    torch.testing.assert_close(cache.grid[0, 0], torch.tensor([-.775, -.725]))
    torch.testing.assert_close(cache.grid[-1, -1], torch.tensor([1.175, .725]))
    pos = torch.tensor([[0., 0., .4], [0., 0., .4]])
    yaw = torch.tensor([0., torch.pi / 2])
    cache.reset(pos, yaw, torch.tensor([0, 0]))
    torch.testing.assert_close(cache.sample_points_world[0], cache.grid)
    torch.testing.assert_close(cache.sample_points_world[1, ..., 0], -cache.grid[..., 1])
    torch.testing.assert_close(cache.sample_points_world[1, ..., 1], cache.grid[..., 0])
    expected = (cache.sample_points_world[..., 0] + 2 * cache.sample_points_world[..., 1] - .4) / 1.5
    valid = cache.map[:, 1].bool()
    torch.testing.assert_close(cache.map[:, 0][valid], expected.clamp(-1, 1)[valid])
    assert torch.equal(cache.map[:, 0][~valid], torch.zeros_like(cache.map[:, 0][~valid]))
    assert bool((cache.map[:, 2][valid] == 0).all())
    assert bool((cache.map[:, 2][~valid] == 1).all())
    assert torch.isfinite(cache.map).all()


def test_refresh_age_sample_frame_hold_and_subset_reset():
    calls = []

    def surface(points, tile_ids):
        calls.append(points.clone())
        return points[..., 0], points[..., 1] > 0

    cache = maps.HeightMap(2, surface_query=surface)
    pos = torch.zeros(2, 3)
    yaw, tiles = torch.zeros(2), torch.zeros(2, dtype=torch.long)
    cache.reset(pos, yaw, tiles)
    old_height, old_points = cache.map[:, 0].clone(), cache.sample_points_world.clone()
    pos[:, 0] += 1
    for step in range(1, 5):
        cache.update(pos, yaw, tiles)
        assert len(calls) == 1
        assert torch.equal(cache.map[:, 0], old_height)
        assert torch.equal(cache.sample_points_world, old_points)
        assert cache.map[0, 2, 0, -1].item() == pytest.approx(step * .02 / .5)
        assert cache.map[0, 2, 0, 0].item() == 1
    cache.update(pos, yaw, tiles)
    assert len(calls) == 2
    assert cache.steps.tolist() == [0, 0]
    assert cache.map[0, 2, 0, -1] == 0
    preserved = cache.map[1].clone()
    pos[0, 0] = 2
    cache.reset(pos, yaw, tiles, [0])
    assert len(calls) == 3 and calls[-1].shape[0] == 1
    assert torch.equal(cache.map[1], preserved)
    cache.update(pos, yaw, tiles, [])
    assert len(calls) == 3


def test_paired_grids_use_two_real_queries_and_one_executed_history():
    calls = []

    def surface(points, tile_ids):
        calls.append((points.clone(), tile_ids.clone()))
        return points[..., 0].square() + points[..., 1], torch.ones_like(points[..., 0], dtype=torch.bool)

    pair = maps.PairedHeightMaps(1, surface_query=surface)
    pos, yaw, tiles = torch.tensor([[.2, .3, .4]]), torch.tensor([.5]), torch.tensor([7])
    teacher_map, student_map = pair.reset(pos, yaw, tiles)
    assert teacher_map.shape == (1, 3, 48, 32)
    assert student_map.shape == (1, 3, 40, 30)
    assert len(calls) == 2
    assert calls[0][0].shape == (1, 48, 32, 2)
    assert calls[1][0].shape == (1, 40, 30, 2)
    assert torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(pair.teacher.sample_position, pair.student.sample_position)
    assert torch.equal(pair.teacher.sample_yaw, pair.student.sample_yaw)
    shared = target.synthetic_observations(batch=1)
    teacher, student = pair.observations(shared)
    for key in ('proprio', 'history', 'route_error', 'privileged'):
        assert teacher[key] is student[key] is shared[key]
    assert teacher['heightmap'] is pair.teacher.map
    assert student['heightmap'] is pair.student.map


def test_fc_interval_overlap_and_constant_feature_weight_sum():
    expected = torch.tensor([[1., .2, 0, 0, 0, 0], [0, .8, .4, 0, 0, 0],
                             [0, 0, .6, .6, 0, 0], [0, 0, 0, .4, .8, 0],
                             [0, 0, 0, 0, .2, 1.]])
    torch.testing.assert_close(target.overlap_matrix(), expected)
    source = torch.randn(64, 768, generator=torch.Generator().manual_seed(42))
    changed = target.migrate_map_fc(source)
    # A constant map feature along the length axis preserves the weight sum.
    torch.testing.assert_close(changed.reshape(64, 32, 5, 4).sum(2),
                               source.reshape(64, 32, 6, 4).sum(2), atol=1e-6, rtol=1e-6)
    with pytest.raises(target.MigrationError):
        target.migrate_map_fc(torch.zeros(64, 640))


def test_real_6400_all_53_entries_are_accounted_for(migrated):
    model, source = migrated
    metadata, state = model.transfer_metadata, model.state_dict()
    assert len(metadata['entries']) == len(source) == len(state) == 53
    assert metadata['copied_entries'] == 51 and metadata['transformed_entries'] == 2
    assert metadata['source_sha256'] == '30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2'
    assert metadata['sds_sha256'] == '500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976'
    assert metadata['old_training_state_loaded'] is False
    assert {row['key'] for row in metadata['entries']} == set(source)
    for key, value in source.items():
        if key in target.FC_KEYS:
            assert state[key].shape == (64, 640)
            torch.testing.assert_close(state[key], target.migrate_map_fc(value), rtol=0, atol=0)
        else:
            assert torch.equal(state[key], value), key
    assert tuple(model.critic[0].weight.shape) == (512, 211)


def test_native_convolution_output_and_strict_full_observation_contract(migrated):
    model, _ = migrated
    obs = target.synthetic_observations(batch=2)
    assert model.actor.map_encoder[:6](obs['heightmap']).shape == (2, 32, 5, 4)
    assert model.critic_map_encoder[:6](obs['heightmap']).shape == (2, 32, 5, 4)
    assert model(obs).shape == (2, 12)
    assert model.evaluate(obs).shape == (2, 1)
    for key in ('proprio', 'history', 'heightmap', 'route_error'):
        bad = dict(obs)
        del bad[key]
        with pytest.raises(target.ObservationError):
            model(bad)
    with pytest.raises(target.ObservationError):
        model(dict(obs, heightmap=torch.zeros(2, 3, 48, 32)))
    with pytest.raises(target.ObservationError):
        model(dict(obs, proprio=obs['proprio'].double()))
    bad = dict(obs)
    del bad['privileged']
    with pytest.raises(target.ObservationError):
        model.evaluate(bad)


def test_s1_freeze_membership_and_momentum_survive_unfreeze(migrated):
    model = target.V09ActorCritic()
    model.load_state_dict(migrated[0].state_dict())
    model.set_s1_course('S1.1')
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert names and all(name.startswith(target.S1_ADAPT_PREFIXES) for name in names)
    assert not model.std.requires_grad
    optimizer = torch.optim.Adam(model.optimizer_parameters(), lr=3e-5)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert set(ids) == {id(p) for p in model.parameters()}
    obs = target.synthetic_observations(batch=2)
    (model(obs).square().mean() + model.evaluate(obs).square().mean()).backward()
    optimizer.step()
    parameter = model.actor.map_encoder[7].weight
    momentum = optimizer.state[parameter]['exp_avg']
    momentum_before = momentum.clone()
    assert bool(momentum.abs().sum() > 0)
    model.set_s1_course('S1.2')
    assert all(p.requires_grad == (name != 'std') for name, p in model.named_parameters())
    assert optimizer.state[parameter]['exp_avg'] is momentum
    assert torch.equal(momentum, momentum_before)
    optimizer.zero_grad(set_to_none=True)
    (model(obs).square().mean() + model.evaluate(obs).square().mean()).backward()
    assert model.actor.mlp[0].weight.grad is not None
    assert model.std.grad is None
    optimizer.step()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 44
    assert all(p.grad is not None and 'exp_avg' in optimizer.state[p]
               and 'exp_avg_sq' in optimizer.state[p] for p in trainable.values())
    assert model.std not in optimizer.state


def test_normalizers_stay_exact_under_train_and_direct_update(migrated):
    model, _ = migrated
    before = {k: v.clone() for k, v in model.state_dict().items() if 'normalizer' in k}
    obs = target.synthetic_observations(batch=2)
    model.train()
    model.update_critic_statistics(obs)
    model.critic_normalizer.update(torch.randn(100, 80))
    model(obs)
    model.evaluate(obs)
    assert model.actor.normalizer.training is False
    assert model.critic_normalizer.training is False
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    stats = model.normalizer_saturation(obs)
    assert len(stats) == 7 and all(value is None or 0 <= value <= 1 for value in stats.values())


def test_input_saturation_reports_valid_map_and_route_boundaries(migrated):
    model, _ = migrated
    obs = target.synthetic_observations(batch=2)
    obs['heightmap'][:] = 0
    obs['heightmap'][0, 1, 0, :2] = 1
    obs['heightmap'][0, 0, 0, 0] = -1
    obs['heightmap'][0, 0, 0, 1] = .5
    obs['heightmap'][1, 0] = 1  # invalid points must not enter saturation denominator
    obs['route_error'][:, 0] = torch.tensor([-4., 3.])
    stats = model.normalizer_saturation(obs)
    assert stats['heightmap_valid_fraction'] == pytest.approx(2 / (2 * 40 * 30))
    assert stats['heightmap_valid_saturation'] == pytest.approx(.5)
    assert stats['route_lateral_saturation'] == pytest.approx(.5)
    obs['heightmap'][:, 1] = 0
    stats = model.normalizer_saturation(obs)
    assert stats['heightmap_valid_fraction'] == 0
    assert stats['heightmap_valid_saturation'] is None


def test_teacher_is_original_frozen_and_labels_keep_value_shape():
    teacher = target.Frozen6400Teacher.from_6400_checkpoint()
    teacher.train(True)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    obs = target._source_observations(batch=2)
    before = target.state_sha256(teacher.state_dict())
    mu, value = teacher.labels(obs)
    teacher.update_critic_statistics(obs)
    teacher.critic_normalizer.update(torch.randn(10, 80))
    assert mu.shape == (2, 12) and value.shape == (2, 1)
    assert not mu.requires_grad and not value.requires_grad
    assert before == target.state_sha256(teacher.state_dict())


def test_rejects_source_replacement_and_incomplete_tensor_transfer(migrated, tmp_path):
    with pytest.raises(target.MigrationError):
        target.V09ActorCritic.migrate_6400_checkpoint(expected_sha='0' * 64)
    wrong = tmp_path / 'wrong.pt'
    wrong.write_bytes(b'not the designated 6400')
    with pytest.raises(target.MigrationError):
        target.V09ActorCritic.migrate_6400_checkpoint(path=wrong)
    source = dict(migrated[1])
    del source['std']
    with pytest.raises(target.MigrationError):
        target.migrate_model_state(source)
