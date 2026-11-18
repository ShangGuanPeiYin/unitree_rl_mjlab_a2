import torch
from src.tasks.velocity.v08.standing import StandWindow

def test_full_window_required_and_drift_rejected():
    s=StandWindow(1); ids=torch.tensor([0]);s.reset(ids)
    def add(x=0.,pitch=0.,mid=False):
        return s.update(torch.tensor([[x,0.]]),torch.zeros(1),torch.zeros(1),torch.tensor([pitch]),torch.zeros(1),torch.zeros(1),torch.ones(1)*.4,torch.zeros(1),torch.ones(1,dtype=torch.bool),torch.ones(1,dtype=torch.bool),torch.tensor([mid]))
    for _ in range(499): assert not add().all()
    assert add().all()
    assert not add(.11).all()

def test_mid_median_uses_four_recent_real_supports_and_allows_pitch():
    s=StandWindow(1);heights=torch.tensor([[0.,.2,.4,.6]])
    median,known=s.support_height(heights,torch.tensor([[True,True,True,False]]))
    assert not known.any()
    median,known=s.support_height(heights,torch.ones(1,4,dtype=torch.bool))
    torch.testing.assert_close(median,torch.tensor([.3]));assert known.all()
    kwargs=dict(xy=torch.zeros(1,2),speed=torch.zeros(1),roll=torch.zeros(1),pitch=torch.tensor([.6]),yaw=torch.zeros(1),omega=torch.zeros(1),height=torch.tensor([.3]),foot_speed=torch.zeros(1),four_support=torch.ones(1,dtype=torch.bool),surface_ok=known,mid=torch.ones(1,dtype=torch.bool))
    for _ in range(150):valid=s.update(**kwargs)
    assert valid.all()
    kwargs['mid'][:]=False
    for _ in range(500):valid=s.update(**kwargs)
    assert not valid.any()
