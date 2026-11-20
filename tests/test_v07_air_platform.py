import sys
from pathlib import Path

def test_air_platform():
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True)

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from a2_v07_bootstrap import activate
    activate()
    import torch
    from src.tasks.velocity.v07 import terrain, spec
    terrain.configure_air_platform(length=20., width=.8, height=2.)
    table=terrain.build_terrain_specs()
    assert table[0].frame_origin_world == (0.,0.,2.)
    assert spec.TERRAIN_TABLE[0].frame_origin_world == (0.,0.,0.)
    assert table[1:] == spec.TERRAIN_TABLE[1:]
    built=terrain.build_terrain()
    box=terrain.tile_boxes(table[0])
    assert len(box)==1 and box[0].bounds == (0.,20.,-.4,.4)
    assert box[0].top==2.
    h,v=terrain.query_surface(torch.tensor([[1.,0.,0.],[19.,.3,0.],[1.,.5,0.]]),torch.tensor(0))
    assert v.tolist()==[True,True,False]
    assert h[:2].tolist()==[2.,2.] and torch.isnan(h[2])
    assert len(built.tile_geoms(0))==1
    print('PASS: isolated 0.8x20m slab at z=2m, outside has no surface')
