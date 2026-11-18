from collections import Counter

def test_frozen_case_tables_baseline_stage_final_and_independent_streams():
    from src.tasks.velocity.v08 import evaluation as e
    baseline=e.make_cases(kind="baseline");final=e.make_cases(kind="final")
    assert len(baseline.cases)==60 and len(final.cases)==2100
    assert baseline.sha256==e.make_cases(kind="baseline").sha256
    assert final.sha256==e.case_table_sha(final)
    assert len({x["case_id"] for x in final.cases})==2100
    assert not set(x["seed"] for x in baseline.cases)&set(x["seed"] for x in final.cases)
    for phase,n in (("A",120),("B",120),("C",180),("D",180),("E",420)):
        assert len(e.make_cases(phase=phase).cases)==n
        h=e.make_cases(phase=phase,kind="h00")
        assert len(h.cases)==15
        assert {x["task"] for x in h.cases}==({"down_stop"} if phase in ("C","D") else {"up_stop"})
    for speed in (.3,.5,.6):
        rows=[x for x in final.cases if x["task"]=="E3" and x["speed"]==speed]
        assert Counter(x["script"] for x in rows)=={"up_same":25,"up_reverse":25,"down_same":25,"down_reverse":25}

def test_trigger_positions_cycle_and_top_initialization_is_physical():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(phase="E")
    for task,values in (("E1",(.45,.6,.75)),("E2",(.25,.4,.55)),("E3",(.3,.5,.7))):
        rows=[x for x in table.cases if x["task"]==task and x["speed"]==.3]
        assert all(abs(row["trigger_p"]-values[i%3])<1e-12 for i,row in enumerate(rows))
    down=next(x for x in table.cases if x["task"]=="down_stop")
    assert down["x"]==9.0 and down["command"]==0 and down["initial_direction"]==-1
    assert all(x["spawn_mode"]=="bottom" for x in table.cases if x["task"]=="E4")
