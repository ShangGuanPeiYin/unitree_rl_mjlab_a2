"""Behavioral regression for ETH transfer, fusion and privileged isolation."""
import importlib.util
from pathlib import Path
import torch
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src/tasks/velocity/rl' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_class():
    path = ROOT / 'src/tasks/velocity/rl/ideal_heightmap_actor_critic.py'
    assert path.exists(), 'W3 trainable actor critic is not implemented'
    return load_file('ideal_actor', path.name).IdealHeightmapActorCritic


def observations(n=8):
    return dict(proprio=torch.randn(n,45), history=torch.randn(n,5,45),
                heightmap=torch.randn(n,3,48,32), privileged=torch.randn(n,35))


def test_eth_transfer_thousand_means_and_hash(tmp_path):
    cls = model_class()
    checkpoint = Path('/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt')
    model = cls.from_eth(checkpoint)
    reference = load_file('eth_actor_reference', 'eth_actor.py').EthActor.load(checkpoint)
    torch.manual_seed(42)
    obs = observations(1000)
    with torch.no_grad():
        assert (model(obs) - reference(obs['proprio'])).abs().max().item() <= 1e-5
    assert all(p.requires_grad for p in model.actor.parameters())
    assert model.transfer_metadata['source_iteration'] == 2000
    bad = tmp_path / 'bad.pt'
    bad.write_bytes(b'bad checkpoint')
    with unittest.TestCase().assertRaisesRegex(ValueError, "SHA256"):
        cls.from_eth(bad)


def test_fusion_receives_gradients_then_encoders_learn():
    model = model_class()()
    obs = observations()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(obs).square().mean().backward()
    assert model.actor.fusion.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    model(obs).square().mean().backward()
    for encoder in (model.actor.map_encoder, model.actor.history_encoder):
        assert sum(p.grad.abs().sum() for p in encoder.parameters()) > 0


def test_privileged_isolation_statistics_and_distribution():
    model = model_class()()
    obs = observations()
    before = model(obs).detach()
    original_stats = {k:v.clone() for k,v in model.actor.normalizer.state_dict().items()}
    model.update_critic_statistics(obs)
    changed = dict(obs, privileged=obs['privileged'] + 100)
    assert torch.equal(before, model(changed))
    assert not torch.equal(model.evaluate(obs), model.evaluate(changed))
    for key, value in model.actor.normalizer.state_dict().items():
        assert torch.equal(value, original_stats[key])
    assert model.critic_normalizer.count > 0
    assert model.critic_normalizer.mean[48:52].eq(0).all()  # contact flags unnormalized
    with torch.no_grad():
        model.std.fill_(100)
    assert model.action_distribution(obs).scale.max() <= .30
    with torch.no_grad():
        model.std.fill_(-100)
    dist = model.action_distribution(obs)
    assert dist.scale.min() >= .05
    model.constrain_std()
    assert model.std.min() >= .05 and model.std.max() <= .30
    assert torch.isfinite(dist.log_prob(dist.sample())).all()
    model.eval()
    count = model.critic_normalizer.count.clone()
    model.update_critic_statistics(changed)
    assert torch.equal(count, model.critic_normalizer.count)
    assert model.evaluate(obs).shape == (8,1)
    assert model.act(obs, deterministic=True).shape == (8,12)
    obs['history'] = torch.randn(8,4,45)
    with unittest.TestCase().assertRaises(ValueError):
        model(obs)
