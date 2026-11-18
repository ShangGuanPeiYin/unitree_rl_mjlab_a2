import pytest
import torch
from src.tasks.velocity.v08.commands import CommandState

def test_exact_boundaries_and_causal_latch():
    c=CommandState(2); c.reset([0,1],torch.tensor([.5,.3])); c.request([0],-.5)
    assert c.executed.tolist()==pytest.approx([.5,.3])
    c.advance(); assert c.current.tolist()==pytest.approx([.49,.3])
    for _ in range(49): c.advance()
    assert c.current[0].item()==pytest.approx(0,abs=1e-7)
    c.latch(); assert c.executed[0].item()==pytest.approx(0,abs=1e-7)
    for _ in range(50): c.advance()
    assert c.current[0].item()==pytest.approx(-.5)

def test_interrupt_repeat_and_batch():
    c=CommandState(2);c.reset([0,1],torch.tensor([.5,.6]));c.request([0,1],0.)
    for _ in range(30): c.advance()
    c.request([0],0.); assert c.k.tolist()==[30,30]
    c.latch();c.request([0],-.5);c.advance()
    assert c.current.tolist()==pytest.approx([.35+(-.5-.35)/100,.414])
    with pytest.raises(ValueError): c.request([0],float('nan'))

def test_exactly_100_executed_transition_samples_and_interruption_uses_latched():
    c=CommandState(1);c.reset([0],.5);c.request([0],-.5)
    for k in range(1,101):
        c.advance();c.latch()
        assert c.executed.item()==pytest.approx(.5-k/100,abs=1e-7)
        assert c.executed_transition.all()
    c.advance();c.latch();assert not c.executed_transition.any()
    c.request([0],.5);c.advance();c.latch();last=c.executed.item();c.advance()
    c.request([0],0.);c.advance()
    assert c.current.item()==pytest.approx(last*(1-1/100),abs=1e-7)

def test_stall_uses_signed_cruise_and_fresh_window_after_zero_or_transition():
    from src.tasks.velocity.v08.speed import StallDiagnostics
    s=StallDiagnostics(2);yes=torch.ones(2,dtype=torch.bool);no=~yes;c=torch.tensor([.5,-.5])
    for k in range(110):s.update(torch.tensor([k*.01,-k*.01]),c,no,no,yes,no)
    assert not s.penalty_on.any();assert (s.dx_1s>.49).all()
    for _ in range(120):s.update(torch.tensor([1.1,-1.1]),c,no,no,yes,no)
    assert s.penalty_on.all()
    s.update(torch.tensor([1.1,-1.1]),torch.zeros(2),no,yes,yes,no)
    assert not s.penalty_on.any() and not s.window_ready.any()
    s.update(torch.tensor([1.1,-1.1]),c,yes,no,yes,no)
    assert not s.penalty_on.any()
    s.update(torch.tensor([1.1,-1.1]),c,no,no,yes,no)
    assert s.count.tolist()==[1,1]
