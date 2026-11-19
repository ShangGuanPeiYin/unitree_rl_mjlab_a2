"""Run in a fresh process: playback geometry must agree with map and contacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
import torch
from src.tasks.velocity.v07 import terrain, events


def _check_extended_platform():
    original = terrain.build_terrain_specs()
    assert original[21].platform_length == 2.0
    terrain.configure_playback_platform(tile_id=21, length=20.0)
    table = terrain.build_terrain_specs()
    assert table[21].valid_bounds_local == (-2.0, 28.0, -2.0, 2.0)
    assert all(a == b for i, (a, b) in enumerate(zip(original, table)) if i != 21)
    built = terrain.build_terrain()
    platform = terrain.tile_boxes(table[21])[-1]
    assert platform.bounds == (8.0, 28.0, 167.6, 168.4)
    points = torch.tensor([[10.5, 168., 0.], [27.9, 168., 0.], [28.1, 168., 0.], [20., 169., 0.]])
    height, valid = terrain.query_surface(points, torch.tensor(21))
    assert valid.tolist() == [True, True, False, True]
    assert height[[0, 1, 3]].tolist() == [8., 8., 0.]
    feet = torch.tensor([[[20., 0., 8.]] * 4])
    assert events.feet_safe_platform(feet, torch.tensor([21]))[1].item()
    assert built.index.bounds[:, 1].max() == 28.0


def test_extended_platform():
    # The playback configuration is deliberately process-local.
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True)


if __name__ == '__main__':
    _check_extended_platform()
    print('PASS: compiled collision geometry, map, bounds and safe platform agree')
