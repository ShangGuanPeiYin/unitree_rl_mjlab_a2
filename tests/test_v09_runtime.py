"""First-terminal capture retains done-slot physical waiting costs."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
from src.tasks.velocity.v09.runtime import evaluate_batch


class Policy:
    def __call__(self, observations):
        return torch.zeros(2, 12)


class Batch:
    num_envs = 2
    def __init__(self):
        self.t = 0
    def observation_pair(self):
        return {}, {}
    def step(self, actions):
        self.t += 1
        episodes = [dict(env_id=0, terminal_step=self.t)]
        if self.t == 3:
            episodes.append(dict(env_id=1, terminal_step=3))
        return {}, None, None, None, dict(episodes=episodes)


def test_first_terminal_and_padding_cost():
    result = evaluate_batch(Batch(), Policy(), policy_map='student', max_steps=5)
    assert [record['terminal_step'] for record in result['records']] == [1, 3]
    assert result['control_steps'] == 3
    assert result['physical_control_samples'] == 6


def test_calibration_moments_equal_population_std_across_uneven_batches():
    from src.tasks.velocity.v09.runtime import ValueMoments
    stats = ValueMoments()
    stats.update(torch.tensor([1., 2., 3.]))
    stats.update(torch.tensor([7.]))
    expected = torch.tensor([1., 2., 3., 7.], dtype=torch.float64).std(correction=0)
    assert abs(stats.value_scale - float(expected)) < 1e-12
    assert stats.count == 4
