"""CPU lifecycle includes tile22 without changing observations/commands/roles."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec, curriculum
from src.tasks.velocity.v07.env import V07Env


@pytest.fixture(scope='module')
def env():
    value = V07Env(num_envs=4,device='cpu',seed=42,phase='D',substage='D22')
    yield value
    del value


def test_native_d22_scene_and_commands_use_the_appended_tile(env):
    assert env.origins.shape == (23,3)
    assert env.tile_N.shape == env.tile_h.shape == env.tile_d.shape == (23,)
    assert int(env.frontier_tile_id) == 22
    assert float(env.tile_d[22]) == pytest.approx(.22)
    assert int(env.geom_tile.max()) == 22
    assert len(env.geom_ids) == sum(1 if t.N == 0 else t.N+2 for t in spec.TERRAIN_TABLE)
    obs = env._observations(update=False)
    assert {key:tuple(value.shape) for key,value in obs.items()} == {
        'proprio':(4,45),'history':(4,5,45),'heightmap':(4,3,48,32),
        'privileged':(4,35),'route_error':(4,3)}
    assert all(torch.isfinite(value).all() for value in obs.values())
    for _ in range(4):
        env.reset()
        assert bool(((env.command[:,0]>=.3)&(env.command[:,0]<=.6)).all())
        assert bool((env.command[:,1:]==0.).all())
        local = env.robot.data.root_link_pos_w-env.origins[env.tile_ids]
        assert bool(((local[:,0]>=-1.1)&(local[:,0]<=-.9)).all())
        assert bool((local[:,1].abs()<=.05).all())
    _,reward,_,_,info = env.step(torch.zeros(4,12))
    assert torch.isfinite(reward).all()
    assert info['generation'] == env.generation


def test_schedule_frontier_can_be_above_final_id_and_invalid_tile_is_rejected(env):
    schedule = curriculum.ScheduleAssignment(phase='D',substage='D22',
        frontier_tile_id=22,roles=tuple(env.roles),tile_ids=(22,22,22,22),generation=env.generation)
    env.set_schedule(schedule,stage_iteration=0)
    assert env.tile_ids.tolist() == [22]*4
    bad = curriculum.ScheduleAssignment(phase='D',substage='D22',
        frontier_tile_id=22,roles=tuple(env.roles),tile_ids=(22,22,22,23),generation=env.generation)
    with pytest.raises(ValueError):
        env.set_schedule(bad,stage_iteration=0)
