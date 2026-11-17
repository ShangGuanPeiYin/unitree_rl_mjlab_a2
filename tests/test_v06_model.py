"""CPU-only v0.6 model contract (SDS 2.2/4, 9.2): ETH migration, zero bypass,
fresh 211 Critic, std 0.30 in [0.05, 1.00] and the single-Adam membership."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v06_bootstrap  # noqa: E402

a2_v06_bootstrap.activate(ROOT)

import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from src.tasks.velocity.v06 import model as target  # noqa: E402
from src.tasks.velocity.v06 import ppo as ppo_module  # noqa: E402
from src.tasks.velocity.v06 import spec  # noqa: E402


@pytest.fixture(scope='module')
def migrated():
    model = target.V06ActorCritic.migrate_eth_checkpoint(seed=42, device='cpu')
    source, state, forbidden = target.load_validated_eth(target.INIT_CHECKPOINT,
                                                        target.INIT_SHA256)
    return model, source, state, forbidden


def test_identity_matches_frozen_spec():
    assert target.ARCHITECTURE == spec.ARCHITECTURE == 'v06-route-state-v1'
    assert target.TASK_ID == 'Unitree-A2-IdealHeightMap-Stairs-V06'
    assert target.SCHEMA_VERSION == 8
    assert target.INIT_SHA256 == spec.INIT_SHA256
    assert target.CRITIC_INPUT_DIMS == 211
    assert target.STD_MIN == 0.05 and target.STD_MAX == 1.00
    assert target.INITIAL_STD == 0.30


def test_initial_std_is_exactly_030(migrated):
    model, _source, _state, _forbidden = migrated
    expected = torch.full((target.N_JOINTS,), 0.30)
    assert torch.equal(model.std.detach(), expected)
    assert model.std_min == 0.05 and model.std_max == 1.00


def test_bypass_is_zero_and_trainable(migrated):
    model = migrated[0]
    parameters = dict(model.named_parameters())
    for key in target.BYPASS_KEYS:
        assert key in parameters, key
        assert parameters[key].requires_grad
        assert float(parameters[key].abs().sum()) == 0.0, key
        assert any(parameter is parameters[key]
                   for parameter in model.optimizer_parameters()), key
    route = model.route_state_dict()
    assert route['all_zero'] is True
    assert route['keys'] == list(target.BYPASS_KEYS)
    assert len(route['sha256']) == 64
    # The map/history encoders are fresh but trainable and enter the optimizer.
    for key in ('actor.map_encoder.0.weight', 'actor.history_encoder.1.0.weight'):
        assert parameters[key].requires_grad
        assert any(parameter is parameters[key] for parameter in model.optimizer_parameters())


def test_critic_is_fresh_211(migrated):
    model, source, _state, forbidden = migrated
    assert tuple(model.critic[0].weight.shape) == (512, 211)
    # The ETH critic block exists in the source and is a forbidden non-source block.
    assert 'critic_state_dict' in forbidden
    assert forbidden['critic_state_dict'] is True
    assert set(source['critic_state_dict'])


def test_normalizer_is_frozen_copy(migrated):
    model, _source, state, _forbidden = migrated
    assert not list(model.actor.normalizer.parameters())
    for key, value in state.items():
        if not key.startswith('obs_normalizer.'):
            continue
        name = key[len('obs_normalizer.'):]
        assert torch.equal(model.actor.normalizer.state_dict()[name], value), key
    # The normalizer buffers are not attached to the optimizer.
    ids = {id(parameter) for parameter in model.optimizer_parameters()}
    for buffer in model.actor.normalizer.buffers():
        assert id(buffer) not in ids


def test_action_mean_matches_eth_within_1e6(migrated):
    model, _source, state, _forbidden = migrated
    equivalence = target.action_mean_equivalence(model, state, seed=42)
    assert equivalence['max_abs_error'] <= 1e-6
    assert equivalence['method'] == 'deterministic_mean'
    assert equivalence['batch'] >= 32


def test_pose_route_observation_matches_sds_formula():
    local_y = torch.tensor([0.0, 0.1, -0.5])
    yaw = torch.tensor([0.0, math.pi / 2, math.pi])
    route = target.route_error_from_pose(local_y, yaw)
    assert tuple(route.shape) == (3, 3)
    assert torch.allclose(route[:, 0], torch.tensor([0.0, 0.5, -2.5]))
    assert torch.allclose(route[:, 1], torch.sin(yaw))
    assert torch.allclose(route[:, 2], torch.cos(yaw))


def test_migration_rejects_wrong_sha(tmp_path):
    with pytest.raises(target.EthMigrationError):
        target.V06ActorCritic.migrate_eth_checkpoint(
            target.INIT_CHECKPOINT, '0' * 64, seed=42, device='cpu')
    other = tmp_path / 'model.pt'
    other.write_bytes(b'not the eth checkpoint')
    with pytest.raises(target.EthMigrationError):
        target.V06ActorCritic.migrate_eth_checkpoint(other, target.INIT_SHA256)


def test_observation_validation_is_strict(migrated):
    model = migrated[0]
    obs = target.synthetic_observations(batch=4, seed=1)
    assert model(obs).shape == (4, target.N_JOINTS)
    missing = dict(obs)
    del missing['route_error']
    with pytest.raises(target.ObservationError):
        model(missing)
    bad = dict(obs)
    bad['proprio'] = torch.zeros(4, 44)
    with pytest.raises(target.ObservationError):
        model(bad)
    nonfinite = dict(obs)
    nonfinite['history'][0, 0, 0] = float('nan')
    with pytest.raises(target.ObservationError):
        model(nonfinite)


def test_parameter_groups_cover_every_parameter(migrated):
    model = migrated[0]
    groups = model.parameter_groups()
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    covered = [name for values in groups.values() for name in values]
    assert sorted(covered) == sorted(names)
    assert len(covered) == len(set(covered))
    assert any(name.startswith('actor.route_fusion') for name in groups['fusion'])
    assert groups['std'] == ['std']
    assert all(name.startswith('critic') for name in groups['critic'])


def test_single_adam_membership(migrated):
    model = migrated[0]
    optimizer = ppo_module.build_optimizer(model)
    membership = ppo_module.optimizer_membership(optimizer, model)
    assert membership['exactly_once'] is True
    assert membership['groups'] == 1
    assert tuple(membership['betas']) == (0.9, 0.999)
    assert membership['eps'] == 1e-8
    assert membership['weight_decay'] == 0.0
    assert membership['default_lr'] == 3e-4


def test_std_projection_after_optimizer_step(migrated):
    model = migrated[0]
    with torch.no_grad():
        model.std.fill_(5.0)
    model.constrain_std()
    assert float(model.std.max()) <= 1.00 + 1e-12
    with torch.no_grad():
        model.std.fill_(0.0)
    model.constrain_std()
    assert float(model.std.min()) >= 0.05 - 1e-12
    distribution = model.action_distribution(target.synthetic_observations(batch=2))
    assert float(distribution.stddev.min()) >= 0.05 - 1e-12


def test_initialization_block_validation(migrated):
    model = migrated[0]
    metadata = model.transfer_metadata
    block = dict(contract={'task': target.TASK_ID}, model={'x': 1},
                 source_path=metadata['source_path'], source_sha256=target.INIT_SHA256,
                 migrated_model_sha256='a' * 64, transfer_metadata=metadata,
                 resume_mode='new_episode', resume=False)
    assert target.validate_initialization(block) is True
    broken = dict(block, source_sha256='b' * 64)
    with pytest.raises(target.EthMigrationError):
        target.validate_initialization(broken)
    broken = dict(block, resume_mode='resume')
    with pytest.raises(ValueError):
        target.validate_initialization(broken)
    metadata_broken = dict(metadata, route_fusion='random')
    with pytest.raises(ValueError):
        target.validate_initialization(dict(block, transfer_metadata=metadata_broken))
