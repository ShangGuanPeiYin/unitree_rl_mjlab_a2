import torch
from src.tasks.velocity.v08.contacts import reduce_contacts

def test_valid_support_requires_assigned_actual_surface():
    # The same physical foot against another atlas tile is never valid support.
    args=dict(geom=torch.tensor([[0,1]]),world=torch.tensor([0]),point=torch.tensor([[0.,0.,.2]]),normal=torch.tensor([[0.,0.,-1.]]),force=torch.tensor([[0.,0.,-10.]]),valid=torch.tensor([True]),foot_lookup=torch.tensor([0,-1]),nonfoot_lookup=torch.tensor([-1,-1]),terrain_tile=torch.tensor([-1,18]),geom_top=torch.tensor([0.,.2]),assigned_tiles=torch.tensor([21]),num_links=1)
    force,tops,links=reduce_contacts(**args)
    assert force.abs().sum()==0 and tops.abs().sum()==0
    args['assigned_tiles']=torch.tensor([18]);force,tops,links=reduce_contacts(**args)
    assert force[0,0,2]>5 and tops[0,0,1]>5

def test_bottom_arrival_requires_recent_four_actual_supports_current_two_and_five_frames():
    from src.tasks.velocity.v08.events import bottom_arrival_update
    step=torch.zeros(1,dtype=torch.long);last=torch.full((1,4),-51,dtype=torch.long);count=step.clone();local=torch.tensor([[-.8,0.,.4]]);feet=torch.zeros(1,4,3);feet[:,:,0]=-1.
    for k,flags in enumerate(([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1]),1):
        last,count,success=bottom_arrival_update(last,count,torch.tensor([k]),local,feet,torch.tensor([flags],dtype=torch.bool))
        assert bool(success[0])==(k==8)
    last,count,success=bottom_arrival_update(last,count,torch.tensor([53]),local,feet,torch.tensor([[0,0,1,1]],dtype=torch.bool))
    assert not success.any() and count[0]==0
    local[0,0]=-.79
    for k in range(5):last,count,success=bottom_arrival_update(last,count,torch.tensor([54+k]),local,feet,torch.ones(1,4,dtype=torch.bool))
    assert not success.any()

def test_final_review_overflow_latches_across_later_samples_until_next_control():
    from types import SimpleNamespace
    from src.tasks.velocity.v08.contacts import ContactState
    contacts=object.__new__(ContactState);contacts.overflow=torch.zeros(1,dtype=torch.bool)
    contacts.raw=torch.zeros(1,4,dtype=torch.bool);contacts.device='cpu'
    contacts.inner=SimpleNamespace(sample=lambda *args,**kwargs:None,found=contacts.raw,foot_force=torch.zeros(1,4,3))
    checks=iter([True,False,False]);contacts.check_overflow=lambda:next(checks)
    contacts.sample(torch.tensor([21]),physics_dt=.005)
    contacts.sample(torch.tensor([21]),physics_dt=.005)
    contacts.sample(torch.tensor([21]),control=True)
    assert contacts.overflow.all()
    contacts.begin_control_step()
    assert not contacts.overflow.any()
