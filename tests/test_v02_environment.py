"""SDS 0.2 independent geometry, contact and environment contracts."""
import math
import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import TerrainGenerator


def test_all_stage_geometry_metadata_and_map_are_consistent():
    from src.tasks.velocity.v02.terrain import atlas_generator_cfg, atlas_metadata, query_top_surface, index_geometry
    spec = mujoco.MjSpec()
    generator = TerrainGenerator(atlas_generator_cfg(seed=42))
    generator.compile(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    metadata = atlas_metadata()
    assert len(metadata) == 10
    indexed = index_geometry(model)
    for meta in metadata:
        np.testing.assert_allclose(generator.terrain_origins[0,meta.tile_id],meta.terrain_origin_world)
        assert (indexed.tile == meta.tile_id).sum() == (1 if meta.N == 0 else meta.N+2)
        assert meta.tread_thickness < meta.h if meta.N else True
        samples = [(-1.,0.),(.1,.7),(meta.N*.2+1.,0.)]
        samples += [((i-.5)*.2,0.) for i in range(1,meta.N+1)]
        for x,y in samples:
            xy = np.array([x,y])+np.array(meta.frame_origin_world[:2])
            z,valid = query_top_surface(torch.tensor(xy),torch.tensor(meta.tile_id))
            gid = np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(model,data,np.r_[xy,12.],np.array([0.,0.,-1.]),None,True,-1,gid)
            assert valid and distance >= 0
            assert abs(z.item()-(12.-distance)) < 1e-5
            assert indexed.tile[gid[0]] == meta.tile_id
            if x >= 0 and abs(y)<.4 and meta.N:
                assert indexed.step[gid[0]] == min(math.floor(x/.2)+1,meta.N)
        if meta.N:
            # At half first-tread height no solid riser or sidewall can be hit.
            xy=np.array(meta.frame_origin_world)
            gid=np.array([-1],dtype=np.int32)
            assert mujoco.mj_ray(model,data,xy+[-.1,0.,meta.h/2-.002],np.array([1.,0.,0.]),None,True,-1,gid) == -1
            last=np.flatnonzero((indexed.tile==meta.tile_id)&(indexed.step==meta.N))
            assert len(last)==2
            np.testing.assert_allclose(indexed.support_bounds[last[0]],indexed.support_bounds[last[1]])


def test_non_power_of_two_slot_allocation_and_first_formal_stage():
    from src.tasks.velocity.v02.terrain import allocate_stage_slots
    for stage in ('P1','P4','S5'):
        slots=allocate_stage_slots(258,stage,mode='comparison')
        assert len(slots)==258 and (slots==0).sum()==26
    slots=allocate_stage_slots(258,'P4',mode='formal',previous_stage=None)
    assert set(slots)=={0,4}
    slots=allocate_stage_slots(258,'S5',mode='formal',previous_stage='S4')
    assert [(slots==i).sum() for i in (0,8,9)]==[26,52,180]


def test_cached_map_hold_isolation_and_subset_reset():
    from src.tasks.velocity.v02.map import IdealHeightMap, ProprioHistory
    from src.tasks.velocity.v02.terrain import atlas_metadata, query_top_surface
    metadata=atlas_metadata()
    positions=torch.tensor([metadata[1].frame_origin_world,metadata[9].frame_origin_world])+torch.tensor([-.1,0.,.4])
    tiles=torch.tensor([1,9]); yaw=torch.zeros(2)
    cache=IdealHeightMap(2)
    old=cache.reset(positions,yaw,tiles).clone()
    for i in range(1,5):
        new=cache.update(positions+torch.tensor([.012*i,0.,0.]),yaw,tiles)
        torch.testing.assert_close(new[:,:2],old[:,:2])
        torch.testing.assert_close(new[:,2][new[:,1].bool()],torch.full_like(new[:,2][new[:,1].bool()],i*.04))
    untouched=cache.map[1].clone()
    cache.reset(positions,yaw,tiles,[0])
    torch.testing.assert_close(cache.map[1],untouched)
    point=torch.tensor(metadata[9].frame_origin_world[:2])
    _,valid=query_top_surface(point,torch.tensor(1))
    assert not valid
    history=ProprioHistory(2)
    history.update(torch.ones(2,45)); history.update(torch.ones(2,45)*2)
    history.reset(torch.ones(2,45)*3,[0])
    assert history.history[0].eq(3).all() and history.history[1,-1].eq(2).all()


def test_contact_top_rules_tile_isolation_and_common_support():
    from src.tasks.velocity.v02.contacts import reduce_contacts, support_from_top_forces
    geom=torch.tensor([[0,1],[1,0],[0,2],[3,1]])
    foot,top,link=reduce_contacts(geom,torch.zeros(4,dtype=torch.long),
        torch.tensor([[0.,0.,.2],[0.,0.,.1],[0.,0.,.2],[0.,0.,.2]]),
        torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,0.,1.],[0.,0.,1.]]),
        torch.tensor([[0.,0.,8.],[-9.,0.,0.],[0.,0.,12.],[0.,0.,90.]]),torch.ones(4,dtype=torch.bool),
        torch.tensor([-1,0,-1,-1]),torch.tensor([-1,-1,0,-1]),torch.tensor([1,-1,-1,2]),
        torch.tensor([.2,0.,0.,.2]),torch.tensor([1]),1)
    torch.testing.assert_close(foot[0,0],torch.tensor([9.,0.,8.]))
    assert top.sum()==8 and link[0,0,2]==12
    upward,gid=support_from_top_forces(torch.tensor([[[3.,3.,4.]]]),torch.tensor([.2,.2,.4]),torch.tensor([1,1,1]))
    assert upward.item()==6 and gid.item() in (0,1)


def fake_task_state():
    from types import SimpleNamespace as S
    from src.tasks.velocity.v02.env import V02Env
    from src.tasks.velocity.v02.rewards import ParkingState
    data=S(root_link_pos_w=torch.tensor([[.9,0.,.5]]),root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]),
           site_pos_w=torch.tensor([[[.5,.1,.112],[.5,-.1,.112],[1.3,.1,.112],[1.3,-.1,.112]]]),
           root_link_lin_vel_b=torch.zeros(1,3),joint_pos=torch.zeros(1,12),joint_vel=torch.zeros(1,12))
    env=S(robot=S(data=data),foot_ids=[0,1,2,3],origins=torch.zeros(10,3),tile_ids=torch.tensor([4]),
          command=torch.tensor([[.4,0.,0.]]),geom_top=torch.tensor([.08]),geom_step=torch.tensor([1]),
          contacts=S(support_geom=torch.zeros(1,4,dtype=torch.long),support=torch.ones(1,4,dtype=torch.bool),
                     link_force=torch.zeros(1,1,3),body_link_mask=torch.tensor([True]),nonfoot_duration=torch.zeros(1,1)),
          control_body_hit=torch.tensor([False]),control_long_hit=torch.tensor([False]),parking=ParkingState(1),
          episode_length_buf=torch.tensor([1]),start_vx=torch.tensor([.4]),num_envs=1,device='cpu')
    env._parameters=lambda:(torch.tensor([1.]),torch.tensor([.08]))
    return env,V02Env._task_state


def test_environment_requires_stop_line_latch_and_recounts_after_retreat():
    env,call=fake_task_state()
    for _ in range(160): assert not call(env)[5]
    assert env.command[0,0]==.4 and not env.stop.any()
    env.robot.data.root_link_pos_w[0,0]=1.
    for _ in range(149): assert not call(env)[5]
    assert call(env)[5] and env.command.eq(0).all()
    env.robot.data.root_link_pos_w[0,0]=.99
    assert not call(env)[5] and env.stop_steps.eq(0).all() and env.stop.all()
    env.robot.data.root_link_pos_w[0,0]=1.
    for _ in range(149): assert not call(env)[5]
    env.control_body_hit[:]=True
    result=call(env)
    assert result[6] and not result[5] and env.stop_steps.eq(0).all()


def test_environment_timeout_is_truncation_and_partial_support_never_success():
    env,call=fake_task_state()
    env.robot.data.root_link_pos_w[0,0]=1.
    env.contacts.support[0,2:]=False
    for _ in range(160): assert not call(env)[5]
    env.episode_length_buf[:]=100000
    result=call(env)
    assert result[7] and not result[6]


def test_stage_iteration_does_not_reset_and_stage_change_does():
    from types import SimpleNamespace as S
    from src.tasks.velocity.v02.env import V02Env
    calls=[]
    e=S(stage='S2',stage_iteration=0,previous_stage='S1',num_envs=258,evaluation=False,
        mode='formal',tile_ids=torch.zeros(258,dtype=torch.long),device='cpu',reset=lambda:calls.append('reset'))
    V02Env.set_stage(e,'S2',123)
    assert calls==[] and e.stage_iteration==123 and e.previous_stage=='S1'
    V02Env.set_stage(e,'S3',0,previous_stage='S2')
    assert calls==['reset'] and set(e.tile_ids.tolist())=={0,6,7}


def test_flat_never_succeeds_and_s5_disturbance_is_explicit_metadata():
    from src.tasks.velocity.v02.terrain import atlas_metadata
    e,call=fake_task_state()
    e.tile_ids[:]=0;e._parameters=lambda:(torch.tensor([0.]),torch.tensor([0.]))
    e.robot.data.root_link_pos_w[0,0]=1.
    for _ in range(160): assert not call(e)[5]
    assert not e.stop.any() and e.command[0,0]==.4
    m=atlas_metadata()[-1]
    assert m.initial_distance_range==(.8,1.2) and m.initial_y_range==(-.1,.1)
    assert m.perturb_ramp_updates==400
    assert abs(m.initial_yaw_range[1]-math.radians(10))<1e-12


import unittest
class V02EnvironmentTests(unittest.TestCase):
    pass
for _name,_test in list(globals().items()):
    if _name.startswith('test_') and callable(_test):
        setattr(V02EnvironmentTests,_name,lambda self,fn=_test:fn())
del _name,_test
if __name__=='__main__': unittest.main()
