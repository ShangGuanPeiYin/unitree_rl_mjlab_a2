"""Isolated playback override must leave the frozen L40 training table intact."""
import sys
from pathlib import Path

def test_ten_step_playback():
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True)

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from a2_v07_bootstrap import activate
    activate()
    import torch
    from src.tasks.velocity.v07 import terrain, spec
    terrain.configure_playback_platform(tile_id=21, length=20., width=4., steps=10)
    table = terrain.build_terrain_specs()
    assert spec.TERRAIN_TABLE[21].N == 40
    assert table[21].N == 10
    assert all(a == b for i, (a, b) in enumerate(zip(table, spec.TERRAIN_TABLE)) if i != 21)
    built = terrain.build_terrain()
    boxes = terrain.tile_boxes(table[21])
    assert len(boxes) == 12
    assert boxes[-1].bounds == (2., 22., 166., 170.)
    assert boxes[-1].top == 2.
    height, valid = terrain.query_surface(torch.tensor([[1.9,168.,0.], [5.,169.,0.]]), torch.tensor(21))
    assert valid.all() and height.tolist() == [2.,2.]
    assert len(built.tile_geoms(21)) == 12
    print('PASS: 10 real treads, 20x4 landing at z=2m, original L40 preserved')
