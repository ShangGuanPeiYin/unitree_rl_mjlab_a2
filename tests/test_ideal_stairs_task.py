"""Contract tests independent of training results."""
import math
import torch
from src.tasks.velocity.mdp.ideal_stairs_contacts import reduce_contacts
from src.tasks.velocity.mdp.ideal_stairs_env import euler_heading,event_reward,timeout_steps


def test_nose_up_sign():
    angle=math.radians(30)
    q=torch.tensor([[math.cos(angle/2),0.,-math.sin(angle/2),0.]])
    roll,pitch,yaw=euler_heading(q)
    torch.testing.assert_close(pitch,torch.tensor([angle]))
    assert abs(roll.item())<1e-6 and abs(yaw.item())<1e-6


def test_events_not_dt_scaled_or_timeout_penalized():
    assert event_reward(torch.tensor([True,False,False]),torch.tensor([False,True,False])).tolist()==[10.,-4.,0.]


def test_timeout_full_stairs():
    assert timeout_steps(torch.tensor([40.]),torch.tensor([.2]),torch.tensor([.2])).item()==7257


def test_contact_pair_direction_top_and_calf_separation():
    # geom 0 terrain tile 0, geom 1 FL football, geom 2 calf, geom 3 neighboring tile.
    geom=torch.tensor([[0,1],[1,0],[0,2],[3,1]])
    world=torch.zeros(4,dtype=torch.long)
    point=torch.tensor([[0.,0.,.2],[0.,0.,.1],[0.,0.,.2],[0.,0.,.2]])
    normal=torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,0.,1.],[0.,0.,1.]])
    force=torch.tensor([[0.,0.,8.],[-9.,0.,0.],[0.,0.,12.],[0.,0.,90.]])
    foot,top,link=reduce_contacts(geom,world,point,normal,force,torch.ones(4,dtype=torch.bool),
        torch.tensor([-1,0,-1,-1]),torch.tensor([-1,-1,0,-1]),torch.tensor([0,-1,-1,1]),
        torch.tensor([.2,0.,0.,.2]),torch.tensor([0]),1)
    torch.testing.assert_close(foot[0,0],torch.tensor([9.,0.,8.]))
    assert top[0,0,0]==8 and top.sum()==8
    assert link[0,0,2]==12


def fake_task_state():
    from types import SimpleNamespace as S
    from src.tasks.velocity.mdp.ideal_stairs_env import IdealStairsEnv
    data=S(root_link_pos_w=torch.tensor([[1.6,0.,.72]]),
           root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]),
           site_pos_w=torch.tensor([[[1.3,.1,.352],[1.3,-.1,.352],[1.8,.1,.352],[1.8,-.1,.352]]]),
           root_link_lin_vel_b=torch.zeros(1,3),joint_pos=torch.zeros(1,12),joint_vel=torch.zeros(1,12))
    e=S(robot=S(data=data),foot_ids=[0,1,2,3],origins=torch.zeros(2,3),tile_ids=torch.tensor([1]),
        stop=torch.tensor([False]),command=torch.tensor([[.4,0.,0.]]),stop_steps=torch.tensor([0]),
        geom_top=torch.tensor([.32]),contacts=S(support_geom=torch.zeros(1,4,dtype=torch.long),support=torch.ones(1,4,dtype=torch.bool),
        link_force=torch.zeros(1,1,3),body_link_mask=torch.tensor([True]),nonfoot_duration=torch.zeros(1,1)),
        control_body_hit=torch.tensor([False]),control_long_hit=torch.tensor([False]),
        episode_length_buf=torch.tensor([1]),start_vx=torch.tensor([.4]),num_envs=1,device='cpu')
    e._parameters=lambda:(torch.tensor([4.]),torch.tensor([.08]))
    return e,IdealStairsEnv._task_state


def test_parking_150_samples_and_failure_precedence():
    e,call=fake_task_state()
    for _ in range(149): result=call(e);assert not result[5].item()
    assert call(e)[5].item()
    assert e.command[0,0]==0
    e.control_body_hit[:]=True
    result=call(e)
    assert result[6].item() and not result[5].item() and result[-1].item()==8


def test_support_needed_for_parking_side_fall_and_timeout():
    e,call=fake_task_state()
    e.contacts.support[:]=False
    for _ in range(151): assert not call(e)[5].item()
    e.robot.data.root_link_pos_w[0,1]=.41
    assert call(e)[6].item()
    e.robot.data.root_link_pos_w[0,1]=0
    e.episode_length_buf[:]=10000
    result=call(e)
    assert result[7].item() and not result[6].item()


def test_foot_height_rejects_overhead_tread_and_missing_defaults():
    from types import SimpleNamespace as S
    from src.tasks.velocity.mdp.ideal_stairs_env import IdealStairsEnv
    e=S(robot=S(data=S(site_pos_w=torch.tensor([[[.1,0.,.132],[.1,0.,.332],[2.,0.,.5],[.1,0.,.232]]]))),
        foot_ids=[0,1,2,3],bounds=torch.tensor([[-1.,1.,-1.,1.],[0.,.2,-.4,.4]]),
        geom_tile=torch.tensor([0,0]),tile_ids=torch.tensor([0]),geom_top=torch.tensor([0.,.2]),
        contacts=S(support=torch.tensor([[False,False,False,True]]),support_geom=torch.tensor([[-1,-1,-1,1]])))
    height,missing=IdealStairsEnv._foot_heights(e)
    torch.testing.assert_close(height,torch.tensor([[.1,.1,1.5,0.]]),atol=1e-6,rtol=0)
    assert missing.tolist()==[[False,False,True,False]]


def test_coplanar_force_sum_excludes_other_heights():
    from src.tasks.velocity.mdp.ideal_stairs_contacts import support_from_top_forces
    force=torch.tensor([[[3.,3.,4.]]])
    upward,gid=support_from_top_forces(force,torch.tensor([.2,.2,.4]),torch.tensor([1,1,1]))
    assert upward.item()==6 and gid.item() in (0,1)


import unittest
class TestIdealStairsContracts(unittest.TestCase):
    pass
for _name,_test in list(globals().items()):
    if _name.startswith('test_') and callable(_test):
        setattr(TestIdealStairsContracts,_name,lambda self,fn=_test:fn())
del _name, _test
if __name__=='__main__':
    unittest.main()
