import torch
from src.tasks.velocity.v08 import spec,terrain
from src.tasks.velocity.v08.map import HeightMap

def test_immutable_physical_table_and_rear_map_descends_without_flipping():
    assert [(name,spec.tile_id_for(name)) for name in ('H00','D20','L40')]==[('H00',1),('D20',18),('L40',21)]
    cache=HeightMap(1);pos=torch.tensor([spec.terrain_spec(21).frame_origin_world],dtype=torch.float32)
    pos[:,0]+=4.;pos[:,2]+=4.5
    image=cache.reset(pos,torch.zeros(1),torch.tensor([21]))
    gx=cache.grid[:,16,0];rear=gx<0
    assert rear.any()
    heights=image[0,0,:,16]*1.5+pos[0,2]
    assert (heights[rear][1:]>=heights[rear][:-1]).all()
    assert heights[rear][-1]>heights[rear][0]
    truth=terrain.query_surface_height(cache.sample_points_world,torch.tensor([21]))
    torch.testing.assert_close(image[:,0],((truth-pos[:,None,None,2])/1.5).clamp(-1,1))

def test_negative_cruise_timeout_is_identical_to_positive():
    torch.testing.assert_close(terrain.timeout_steps(21,torch.tensor([-.5])),terrain.timeout_steps(21,torch.tensor([.5])))
