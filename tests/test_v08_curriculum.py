from collections import Counter
import pytest

def module():
    from src.tasks.velocity.v08 import curriculum
    return curriculum

def test_formal_allocations_and_all_e3_combinations():
    c=module()
    expected={"A":{"L40:up_stop":320,"FLAT:flat_stop":128,"H00:up_stop":64},"B":{"L40:up_stop":320,"FLAT:flat_stop":128,"H00:up_stop":64},"C":{"D20:down_stop":256,"L40:up_stop":128,"FLAT:flat_stop":96,"H00:down_stop":32},"D":{"L40:down_stop":256,"L40:up_stop":128,"FLAT:flat_stop":96,"D20:down_stop":32},"E":{"L40:E1":64,"L40:E2":64,"L40:E3":64,"L40:E4":64,"L40:up_stop":96,"L40:down_stop":64,"FLAT:flat_stop":64,"H00:up_stop":32}}
    for phase,counts in expected.items():
        m=c.CurriculumManager();m.state.phase=phase
        a=m.schedule();assert len(a.cases)==512
        assert Counter(x["terrain_name"]+":"+x["task"] for x in a.cases)==counts
        if phase in ("A","B"):assert all(x["task"] in ("up_stop","flat_stop") for x in a.cases)
        if phase=="E":assert Counter(x["script"] for x in a.cases if x["task"]=="E3")=={"up_same":16,"up_reverse":16,"down_same":16,"down_reverse":16}

def test_b_ramp_eval_full_and_sampler_resume():
    c=module();m=c.CurriculumManager();m.state.phase="B"
    assert c.spawn_bounds("B",0)["y"]==(-.05,.05)
    assert c.spawn_bounds("B",400)["y"]==(-.1,.1)
    assert c.spawn_bounds("B",0,evaluation=True)["y"]==(-.1,.1)
    m.schedule();saved=m.state_dict();expected=m.schedule().as_dict();r=c.CurriculumManager(state=saved)
    assert r.schedule().as_dict()==expected

def test_streak_reset_caps_and_no_budget_transfer():
    c=module();m=c.CurriculumManager()
    for _ in range(100):m.on_update()
    assert not m.record_gate({"gate_pass":True})
    for _ in range(100):m.on_update()
    assert not m.record_gate({"gate_pass":False})
    for _ in range(100):m.on_update()
    assert not m.record_gate({"gate_pass":True})
    for _ in range(100):m.on_update()
    assert m.record_gate({"gate_pass":True})
    assert m.state.phase=="B" and m.state.spent_by_phase["A"]==400 and m.state.generation_updates==0
    for _ in range(2000):m.on_update()
    assert m.state.finished and m.state.stop_status=="budget_exhausted"
    assert m.state.last_update==8800
    assert sum(c.PHASE_BUDGETS.values())==18000

def test_h00_report_only_cannot_reset_or_advance_gate():
    c=module();m=c.CurriculumManager();m.state.gate_streak=1
    assert not m.record_gate({"kind":"h00","gate_pass":False})
    assert m.state.gate_streak==1 and not m.state.finished

def test_cap_boundary_gate_can_promote_without_extra_update_and_duplicate_is_idempotent():
    c=module();m=c.CurriculumManager()
    for _ in range(3900):m.on_update()
    assert not m.record_gate({'gate_pass':True})
    assert not m.record_gate({'gate_pass':True})
    assert m.state.gate_streak==1
    for _ in range(100):m.on_update()
    assert m.finished and m.stop_status=='budget_exhausted'
    assert m.record_gate({'gate_pass':True}) and m.phase=='B' and not m.finished
    assert m.state.spent_by_phase['A']==4000 and m.state.spent_by_phase['B']==0

def test_nonconsecutive_gates_cannot_satisfy_streak_and_invalid_resume_rejected():
    c=module();m=c.CurriculumManager()
    for _ in range(100):m.on_update()
    m.record_gate({'gate_pass':True})
    for _ in range(200):m.on_update()
    assert not m.record_gate({'gate_pass':True}) and m.state.gate_streak==1
    state=m.state_dict();state['state']['last_update']+=1
    with pytest.raises(ValueError):c.CurriculumManager(state=state)
