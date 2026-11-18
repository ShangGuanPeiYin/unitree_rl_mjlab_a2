import torch
from src.tasks.velocity.v08.tasks import TaskMachine

def test_flat_does_not_arrive_early_and_timeout_beats_arrival():
    t=TaskMachine([dict(task='flat',speed=.5)],'cpu');t.reset([0],torch.tensor([0.]))
    for i in range(399):
        r=t.advance(torch.tensor([i*.01]),torch.tensor([False]),torch.tensor([False]),torch.tensor([False]),torch.tensor([False]),torch.tensor([0.]),torch.tensor([False]))
    assert not r['arrival'].any()
    r=t.advance(torch.tensor([4.]),torch.tensor([False]),torch.tensor([False]),torch.tensor([False]),torch.tensor([False]),torch.tensor([0.]),torch.tensor([True]))
    assert r['failure'].all() and not r['arrival'].any()

def test_reversal_crossing_must_be_directional():
    t=TaskMachine([dict(task='E1',speed=.5,trigger_p=.5,tile_id=21)],'cpu');t.reset([0],torch.tensor([-1.]))
    args=(torch.tensor([False]),)*4
    r=t.advance(torch.tensor([4.1]),*args,torch.tensor([.5]),torch.tensor([False]))
    assert r['request'][0]==-.5

def _tick(t,x=0.,top=False,bottom=False,stable=False,failed=False,c=.5,transition=False,target=None,timeout=False):
    b=len(t.cases)
    f=lambda v,dtype=None:torch.full((b,),v,dtype=dtype)
    return t.advance(f(x),f(top,torch.bool),f(bottom,torch.bool),f(stable,torch.bool),f(failed,torch.bool),f(c),f(timeout,torch.bool),transition=f(transition,torch.bool),requested=f(c if target is None else target))

def test_flat_exact_400_100_1000_and_failure_priority():
    t=TaskMachine([dict(task='flat_stop',speed=.5)]);t.reset([0],torch.tensor([0.]))
    for k in range(1,401):r=_tick(t,x=k*.01)
    assert r['arrival'].all() and r['request'][0]==0 and not r['success'].any()
    assert sum(e['steps'] for e in t.budget_events[0])==1500
    for k in range(1,101):
        r=_tick(t,x=4.,c=.5*(1-k/100),transition=True,target=0.)
        assert not r['success'].any()
    for k in range(999):
        r=_tick(t,x=4.,c=0.,stable=True)
        assert not r['success'].any()
    r=_tick(t,x=4.,c=0.,stable=True,failed=True)
    assert r['failure'].all() and not r['success'].any() and not r['arrival'].any()
    assert not _tick(t,x=4.,c=0.,stable=True)['failure'].any()

def test_direct_reversal_zero_is_not_a_stop_and_new_cruise_is_causal():
    t=TaskMachine([dict(task='E1',speed=.5,trigger_p=.5,tile_id=21)]);t.reset([0],torch.tensor([3.9]))
    assert _tick(t,x=4.01)['request'][0]==-.5
    for k in range(1,101):
        r=_tick(t,x=4.01,c=.5-k*.01,transition=True,target=-.5)
        assert not r['arrival'].any() and not r['success'].any()
    # At step102 the first cruise frame moves .61 from the transition end.
    _tick(t,x=3.4,c=-.5)
    assert len(t.direction_checks[0])==1
    check=t.direction_checks[0][0]
    assert check['request_step']==1 and check['cruise_start_step']==102
    assert check['net_progress']>=.6 and check['elapsed_steps']==101

def test_stopping_cannot_fake_direction_and_timeout_at_arrival_wins():
    t=TaskMachine([dict(task='E1',speed=.5,trigger_p=.5,tile_id=21)]);t.reset([0],torch.tensor([3.9]));_tick(t,x=4.01)
    for _ in range(399):r=_tick(t,x=4.01,c=0.)
    assert not r['failure'].any()
    assert _tick(t,x=4.01,c=0.)['failure'].all()
    t=TaskMachine([dict(task='up_stop',speed=.5,tile_id=21)])
    t.leg_steps[0]=t._limit(0)-1
    r=_tick(t,x=9.,top=True)
    assert r['failure'].all() and not r['arrival'].any()

def test_e3_midstop_and_e4_intermediate_top_have_no_arrival_reward():
    for task in ('E3','E4'):
        t=TaskMachine([dict(task=task,speed=.5,tile_id=21,trigger_p=.5,script='up_reverse')]);t.reset([0],torch.tensor([3.9]))
        r=_tick(t,x=4.01,top=task=='E4')
        assert r['request'][0]==0 and not r['arrival'].any()
        for k in range(100):r=_tick(t,x=4.01,c=0.,transition=True,target=0.)
        needed=300 if task=='E3' else 1000
        for _ in range(needed):r=_tick(t,x=4.01,c=0.,stable=True)
        assert r['request'][0]==-.5 and not r['arrival'].any() and not r['success'].any()
        assert t.direction_pending[0]
        budget=t.budget_events[0];assert len([e for e in budget if e['event']=='travel'])==2
        assert [e['steps'] for e in budget if e['event'].endswith('hold')]==[needed]
        assert [e['steps'] for e in budget if 'transition' in e['event']]==[100]
        for _ in range(100):_tick(t,x=4.01,c=-.2,transition=True,target=-.5)
        _tick(t,x=3.3,c=-.5)
        assert t.direction_checks[0][0]['net_progress']>.6
        r=_tick(t,x=-1.,c=-.5,bottom=True)
        assert r['arrival'].all()

def test_top_startup_only_grants_one_transition_and_logical_recovery_preserves_clocks():
    t=TaskMachine([dict(task='down_stop',tile_id=18,speed=.5)])
    assert [e['event'] for e in t.budget_events[0]]==['travel','startup_transition']
    t.leg_steps[0]=t._limit(0)+99
    r=_tick(t,x=-1.,bottom=True,c=-.5)
    assert r['failure'].all() and not r['arrival'].any()
    t=TaskMachine([dict(task='E1',tile_id=21,speed=.5,trigger_p=.5)]);t.reset([0],torch.tensor([3.9]));_tick(t,x=4.01)
    state=t.state_dict();other=TaskMachine([dict(task='E1',tile_id=21,speed=.5)]);other.load_state_dict(state)
    assert other.budget_events==t.budget_events and other.points==t.points
    torch.testing.assert_close(other.steps,t.steps);torch.testing.assert_close(other.request_step,t.request_step)

def test_safe_top_arrival_is_only_event_and_success_is_once_after_full_hold():
    t=TaskMachine([dict(task='up_stop',tile_id=21,speed=.5)]);t.reset([0],torch.tensor([-1.]))
    r=_tick(t,x=9.,top=True)
    assert r['arrival'].all() and not r['success'].any() and t.mode[0]==t.DECEL
    for _ in range(100):_tick(t,x=9.,c=0.,transition=True,target=0.)
    for _ in range(999):assert not _tick(t,x=9.,c=0.,stable=True)['success'].any()
    r=_tick(t,x=9.,c=0.,stable=True)
    assert r['success'].all() and not r['arrival'].any()
    assert not _tick(t,x=9.,c=0.,stable=True)['success'].any()

def test_e2_downward_crossing_reverses_up_and_each_leg_has_single_travel_cap():
    t=TaskMachine([dict(task='E2',tile_id=21,speed=.5,trigger_p=.4)]);t.reset([0],torch.tensor([9.]))
    r=_tick(t,x=3.19,c=-.5)
    assert r['request'][0]==.5 and not r['arrival'].any()
    for k in range(1,101):_tick(t,x=3.19,c=-.5+k*.01,transition=True,target=.5)
    _tick(t,x=3.8,c=.5)
    assert t.direction_checks[0][0]['direction']==1 and t.direction_checks[0][0]['net_progress']>.6
    assert _tick(t,x=9.,top=True,c=.5)['arrival'].all()
    events=t.budget_events[0]
    assert len([e for e in events if e['event']=='travel'])==2
    assert [e['steps'] for e in events if 'transition' in e['event']]==[100,100]
