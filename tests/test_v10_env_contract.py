import math
import torch
from src.tasks.velocity.v10.geometry import Terrain, primitives, query_local, safe_top, route_height, corridor
from src.tasks.velocity.v10.tasks import TaskState, allocate_roles, combinations
from src.tasks.velocity.v10.contacts import ContactFilter


def test_geometry_real_g_u_boundaries_and_union():
    for mode in ('G', 'U'):
        t = Terrain('platform', H=8., W=.8, mode=mode)
        z, valid = query_local(torch.tensor([[1.,0.],[1.,.5],[14.,0.],[14.01,0.]]), t)
        assert z[0] == 8 and valid[0] and valid[2] and not valid[3]
        assert valid[1] == (mode == 'G')
        if mode == 'G': assert z[1] == 0
    t = Terrain('stairs',N=10,W=.8,mode='U')
    z, valid = query_local(torch.tensor([[1.,.5],[2.1,.5],[2.,0.]]),t)
    assert valid.tolist() == [True,False,True]
    assert z[2] == 2
    # Coplanar last tread and platform form one connected support surface.
    assert safe_top(torch.tensor([[2.,0.],[2.,.37]]), torch.tensor([2.,2.]),t).tolist() == [True,False]


def test_route_height_does_not_follow_fall_and_width_not_old_boundary():
    for h in (2.,4.,8.):
        t=Terrain('platform',H=h,W=4.,mode='G')
        assert route_height(torch.tensor([1.]),t).item()==h
        assert corridor(torch.tensor([[1.,1.9]]),t).item()
        assert not corridor(torch.tensor([[1.,2.01]]),t).item()


def test_contact_filter_is_not_safety():
    f=ContactFilter(1,'cpu')
    force=torch.zeros(1,4,3);force[0,0,0]=6
    a=f.update(force,torch.zeros(1,4),torch.ones(1,4,dtype=torch.bool))
    assert a['contact_raw'][0,0] and a['reward_contact'][0,0]
    assert not a['safe_support'].any()
    a=f.update(torch.zeros_like(force),torch.zeros(1,4),torch.ones(1,4,dtype=torch.bool))
    assert not a['contact_raw'].any() and a['reward_contact'][0,0]
    assert not f.update(torch.zeros_like(force),torch.zeros(1,4),torch.ones(1,4,dtype=torch.bool))['reward_contact'].any()


def test_roles_and_combinations():
    assert list(allocate_roles(1024).values()) == [205,102,154,154,102,307]
    assert list(allocate_roles(512).values()) == [102,51,77,77,51,154]
    assert sum(len(combinations(r,'D1')) for r in allocate_roles(1024)) ==154
    assert sum(len(combinations(r,'D2')) for r in allocate_roles(1024)) ==226
    assert sum(len(combinations(r,'D3')) for r in allocate_roles(1024)) ==334


def test_hold_and_ramp_exact_and_reward_phase_latched():
    for mode in ('instant','ramp'):
        t=TaskState(dict(task='F_STOP',speed=.5,stop_mode=mode), 'D3',1.)
        commands=[];phases=[]
        for i in range(1800):
            commands.append(t.command);phases.append(t.phase)
            out=t.advance(1.,False,False)
            if out['finished']:break
        assert len(commands)==(1650 if mode=='instant' else 1750)
        assert sum(p=='HOLD' for p in phases)==1000
        assert sum(p=='DECEL' for p in phases)==(0 if mode=='instant' else 100)
        assert phases[399]=='WALK' and phases[400]==('HOLD' if mode=='instant' else 'DECEL')
        assert commands[400]==(.5 if mode=='ramp' else 0)
        assert commands[500]==0 if mode=='ramp' else True
        assert phases[-1]=='RESTART'


def test_top_second_frame_switch_next_and_failure_precedence():
    t=TaskState(dict(task='S_ROUTE',speed=.5,N=10,route='walk'), 'D3',-1.)
    assert t.teacher=='stairs'
    assert not t.advance(2.5,True,False)['top']
    assert t.teacher=='stairs'
    out=t.advance(2.6,True,False)
    assert out['top'] and t.teacher=='blind' and t.g_level==0 and t.top_x==2.6
    t=TaskState(dict(task='S_ROUTE',speed=.5,N=10,route='top'), 'D1',-1.)
    t.advance(2.5,True,False)
    out=t.advance(2.6,True,True)
    assert out['failure'] and not out['top'] and not out['finished']
