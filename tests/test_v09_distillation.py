"""SDS S1 logical minibatch reductions and optimizer preservation."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
from src.tasks.velocity.v09.distillation import distill_minibatch


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Linear(3, 12)
        self.value = torch.nn.Linear(3, 1)

    def forward(self, obs):
        return self.mean(obs['proprio'])

    def evaluate(self, obs):
        return self.value(obs['proprio'])


def test_uneven_microbatch_matches_full_batch():
    torch.manual_seed(42)
    full, micro = Toy(), Toy()
    micro.load_state_dict(full.state_dict())
    obs = {'proprio': torch.randn(11, 3)}
    labels = torch.randn(11, 12)
    values = torch.randn(11, 1)
    left = torch.optim.Adam(full.parameters(), lr=3e-5)
    right = torch.optim.Adam(micro.parameters(), lr=3e-5)
    a = distill_minibatch(full, left, obs, labels, values, value_scale=2., microbatch=512)
    b = distill_minibatch(micro, right, obs, labels, values, value_scale=2., microbatch=4)
    assert abs(a['loss'] - b['loss']) < 1e-6
    for p, q in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)


def test_complete_rollout_executes_twenty_logical_updates():
    from src.tasks.velocity.v09.distillation import train_iteration
    model = Toy()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    obs = {'proprio': torch.randn(48, 4, 3)}
    labels = torch.randn(48, 4, 12)
    values = torch.randn(48, 4, 1)
    stats = train_iteration(model, optimizer, obs, labels, values, value_scale=1.,
                            generator=torch.Generator().manual_seed(42), microbatch=7)
    assert stats['samples'] == 192
    assert stats['adam_steps'] == 20
    assert all(int(state['step']) == 20 for state in optimizer.state.values())
