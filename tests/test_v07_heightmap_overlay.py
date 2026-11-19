"""Overlay must show cached samples without mutating the policy observation."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
import numpy as np
import torch
import mujoco
from src.tasks.velocity.v07.map import HeightMap
from play_a2_v07_extended_platform import draw_heightmap


def test_cached_scan_overlay():
    cache = HeightMap(1, 'cpu')
    cache.reset(torch.tensor([[9., 168., 8.4]]), torch.tensor([0.]), torch.tensor([21]))
    before = cache.map.clone()
    scene = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=4096)
    env = SimpleNamespace(map_cache=cache)
    draw_heightmap(scene, env)
    assert scene.ngeom == 1536 + 96 + 5
    # First sample is on the low ground beside the elevated platform.
    np.testing.assert_allclose(scene.geoms[0].pos, [8.225, 167.225, .015], atol=2e-5)
    np.testing.assert_allclose(scene.geoms[0].rgba, [1., .45, .05, 1.])
    # Forward footprint edge spans x=10.6 at +/-0.8m in y.
    np.testing.assert_allclose(scene.geoms[1632].pos, [10.6, 167.2, 8.75], atol=2e-5)
    assert torch.equal(before, cache.map)
    draw_heightmap(scene, env)
    assert scene.ngeom == 1637  # no accumulating render-only geoms
