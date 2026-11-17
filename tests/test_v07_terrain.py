"""D22 geometry against measured collision and height/timeout consumers."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import spec, terrain, events
from src.tasks.velocity.v07.map import HeightMap


@pytest.fixture(scope='module')
def build():
    return terrain.build_terrain(seed=42)


def test_d22_measured_collision_height_and_safe_platform_are_one_spec(build):
    tile = spec.terrain_spec(22)
    assert len(build.specs) == 23
    assert len(build.tile_geoms(22)) == 8
    model = build.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    for x,z in [(0.11,.2),(.33,.4),(.55,.6),(.77,.8),(.99,1.),(1.21,1.2),(1.42,1.2)]:
        gid = np.array([-1], dtype=np.int32)
        ray = mujoco.mj_ray(model,data,np.array([x,176.,4.]),np.array([0.,0.,-1.]),None,True,-1,gid)
        assert 4.-ray == pytest.approx(z, abs=1e-6)
        assert build.index.tile[gid[0]] == 22
        height, valid = terrain.query_surface(torch.tensor([[x,176.,0.]]),torch.tensor([22]))
        assert bool(valid[0])
        assert float(height[0]) == pytest.approx(z,abs=1e-6)
    geoms = build.tile_geoms(22)
    first = next(g for g in geoms if build.index.step[g] == 1)
    assert model.geom_size[first].tolist() == pytest.approx([.11,.4,.005])
    assert terrain.platform_start(22) == pytest.approx(1.32)
    assert terrain.platform_bounds(22) == pytest.approx((1.362,3.278,-.358,.358))
    assert tile.N * tile.step_depth + spec.SUCCESS_BODY_PLATFORM_MARGIN == pytest.approx(2.12)


def test_all_23_tiles_work_in_height_map_geometry_and_contact_routing(build):
    from src.tasks.velocity.v07.events import first_step_contact
    tiles = torch.arange(23)
    base = torch.tensor([s.frame_origin_world for s in build.specs])
    base[:,0] += .1
    base[:,2] += .4
    cache = HeightMap(23,'cpu')
    mapped = cache.reset(base,torch.zeros(23),tiles)
    assert mapped.shape == (23,3,48,32)
    assert torch.isfinite(mapped).all()
    assert cache.sample_tile.tolist() == list(range(23))
    values = events.tile_parameters(tiles)
    assert [v.shape[0] for v in values] == [23,23,23,23]
    tensors = terrain.tensorize_geometry(build.index,'cpu')
    forces = torch.zeros(23,4,len(build.index.tile))
    first = next(g for g in build.tile_geoms(22) if build.index.step[g] == 1)
    forces[22,:,first] = 6.
    contact = first_step_contact(forces,tensors['tile'],tensors['step'],tiles)
    assert bool(contact[22].all())
    assert not bool(contact[:22].any())
    with pytest.raises(ValueError):
        terrain.query_surface(torch.zeros(1,3),torch.tensor([23]))


@pytest.mark.parametrize('speed,want',[(.3,1828),(.5,1297),(.6,1164)])
def test_d22_timeouts_agree_in_scalar_and_tensor_paths(speed,want):
    assert spec.episode_step_limit(22,speed) == want
    assert int(terrain.timeout_steps(22,torch.tensor([speed]))[0]) == want
    assert int(events.timeout_steps(torch.tensor([22]),torch.tensor([speed]))[0]) == want


def test_d22_safe_summit_body_threshold_and_debounce_remain_unchanged():
    state = events.PlatformSupportState(2,'cpu')
    safe = torch.ones(2,4,dtype=torch.bool)
    feet = torch.tensor([[[1.40,0.,1.2]]*4, [[1.40,0.,1.2]]*4])
    assert bool(events.feet_safe_platform(feet,torch.tensor([22,22]))[1].all())
    for step in range(1,7):
        result = state.update(torch.tensor([step,step]),torch.tensor([True,True]),
            torch.tensor([22,22]),torch.zeros(2,4,dtype=torch.long), safe,safe,
            torch.tensor([2.119,2.12],dtype=torch.float64),
            torch.tensor([1.32,1.32],dtype=torch.float64),torch.zeros(2),
            torch.zeros(2,dtype=torch.bool))
    assert result.climbed.tolist() == [False,True]
