"""Run in a fresh process: playback geometry must agree with map and contacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
import torch
from src.tasks.velocity.v07 import terrain, events


def _check_extended_platform(wide=False):
    original = terrain.build_terrain_specs()
    assert original[21].platform_length == 2.0
    terrain.configure_playback_platform(tile_id=21, length=20.0, width=4.0 if wide else 0.8)
    table = terrain.build_terrain_specs()
    assert table[21].valid_bounds_local == (-2.0, 28.0, -2.0, 2.0)
    assert all(a == b for i, (a, b) in enumerate(zip(original, table)) if i != 21)
    built = terrain.build_terrain()
    platform = terrain.tile_boxes(table[21])[-1]
    assert platform.bounds == ((8.0, 28.0, 166.0, 170.0) if wide else (8.0, 28.0, 167.6, 168.4))
    assert terrain.tile_boxes(table[21])[-2].half[1] == 0.4
    points = torch.tensor([[10.5, 168., 0.], [27.9, 168., 0.], [28.1, 168., 0.], [20., 169., 0.]])
    height, valid = terrain.query_surface(points, torch.tensor(21))
    assert valid.tolist() == [True, True, False, True]
    assert height[[0, 1, 3]].tolist() == ([8., 8., 8.] if wide else [8., 8., 0.])
    feet = torch.tensor([[[20., 0., 8.]] * 4])
    assert events.feet_safe_platform(feet, torch.tensor([21]))[1].item()
    assert built.index.bounds[:, 1].max() == 28.0
    if wide:
        from src.tasks.velocity.v07.map import HeightMap
        cache = HeightMap(1, 'cpu')
        for yaw in (0., 0.5, 1.1, 1.57, 2.5, 3.14):
            cache.reset(torch.tensor([[12., 168.2, 8.4]]), torch.tensor([yaw]), torch.tensor([21]))
            assert cache.map[0, 1].all()
            torch.testing.assert_close(cache.map[0, 0], torch.full((48, 32), -0.4/1.5))
        # Widening must not fill the open sides beside the last stair tread.
        h, _ = terrain.query_surface(torch.tensor([[7.9, 169., 0.], [12., 170.1, 0.]]), torch.tensor(21))
        assert h[0].item() == 0. and torch.isnan(h[1])


def test_extended_platform():
    # The playback configuration is deliberately process-local.
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True)


def test_wide_platform():
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--wide'], check=True)


if __name__ == '__main__':
    _check_extended_platform(wide='--wide' in sys.argv)
    print('PASS: compiled collision geometry, map, bounds and safe platform agree')
