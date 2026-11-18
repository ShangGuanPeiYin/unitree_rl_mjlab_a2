import torch
from src.tasks.velocity.v08.rewards import parking_adjustments

def test_signed_progress_parking_pitch_and_zero_crossing():
    out=parking_adjustments(dx=torch.tensor([-.1,.1,.1,.1]),executed=torch.tensor([-.5,-.5,0.,0.]),transition=torch.tensor([False,False,True,False]),parking=torch.tensor([False,False,False,True]),level=torch.tensor([True,True,True,False]),gravity_x=torch.ones(4)*.5,angular_xy_sq=torch.ones(4),stand=torch.ones(4,dtype=torch.bool))
    torch.testing.assert_close(out['forward_progress'],torch.tensor([.04,-.04,0.,0.]))
    torch.testing.assert_close(out['standing'],torch.tensor([0.,0.,0.,.02]))
    assert out['parking_pitch'].tolist()==[0.,0.,0.,0.]
    assert out['parking_angular'][3].item()<0

def _frame():
    from src.tasks.velocity.v08.rewards import RewardFrame
    z=torch.zeros;b=2
    return RewardFrame(executed_command=torch.tensor([.5,-.5]),local_x=torch.tensor([.1,-.1]),local_x_prev=z(b),body_lin_vel_b=torch.tensor([[.5,0.,0.],[-.5,0.,0.]]),body_ang_vel_b=z(b,3),gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(b,1),base_height_above_surface=torch.ones(b)*.35,q=z(b,12),qdot=z(b,12),q_target_executed=z(b,12),action=z(b,12),previous_action=z(b,12),foot_position_world=z(b,4,3),foot_velocity_world=z(b,4,3),foot_force_world=z(b,4,3),contact_raw=torch.ones(b,4,dtype=torch.bool),contact_filtered=torch.ones(b,4,dtype=torch.bool),tile_ids=torch.zeros(b,dtype=torch.long),local_y=z(b),yaw_world=z(b),resets=torch.zeros(b,dtype=torch.bool),success=torch.zeros(b,dtype=torch.bool),failure=torch.zeros(b,dtype=torch.bool),foot_surface_height=z(b,4))

def test_original_continuous_terms_unchanged_in_cruise_and_air_time_cannot_burst():
    from copy import deepcopy
    from src.tasks.velocity.v07.rewards import RewardComputer as Old
    from src.tasks.velocity.v08.rewards import RewardComputer
    f=_frame();old=Old();new=RewardComputer();false=torch.zeros(2,dtype=torch.bool)
    result=new(f,transition=false,parking=false,level=~false,stand=false,arrival=false,success=false,failure=false);baseline=old(deepcopy(f))
    for name in baseline.weighted_terms:
        if name!='forward_progress':torch.testing.assert_close(result.weighted_terms[name],baseline.weighted_terms[name],atol=0,rtol=0)
    torch.testing.assert_close(result.weighted_terms['forward_progress'],torch.tensor([.04,.04]))
    f.contact_raw[:]=False;f.contact_filtered[:]=False
    for _ in range(80):result=new(f,transition=false,parking=~false,level=~false,stand=false,arrival=false,success=false,failure=false)
    assert result.weighted_terms['feet_air_time'].abs().sum()==0
    f.contact_raw[:]=True;f.contact_filtered[:]=True
    result=new(f,transition=~false,parking=false,level=~false,stand=false,arrival=false,success=false,failure=false)
    assert result.weighted_terms['feet_air_time'].abs().sum()==0
    assert new.state.air_time.has_contacted_once.all()

def test_level_only_pitch_parking_angular_and_failure_suppresses_events():
    from src.tasks.velocity.v08.rewards import RewardComputer
    f=_frame();f.executed_command[:]=0;f.gravity_b[:,0]=.5;f.body_ang_vel_b[:,0]=1.
    yes=torch.ones(2,dtype=torch.bool);no=~yes
    r=RewardComputer()(f,transition=no,parking=yes,level=torch.tensor([True,False]),stand=yes,arrival=yes,success=yes,failure=yes)
    torch.testing.assert_close(r.weighted_terms['parking_pitch'],torch.tensor([-.025,0.]))
    torch.testing.assert_close(r.weighted_terms['ang_vel_xy']+r.weighted_terms['parking_angular'],torch.tensor([-.004,-.004]))
    assert r.weighted_terms['arrival'].sum()==0 and r.weighted_terms['success'].sum()==0
    assert r.weighted_terms['failure'].tolist()==[-10.,-10.]
