import hashlib
from src.tasks.velocity.v10 import spec, cases


def test_frozen_identity_and_allocations():
    assert hashlib.sha256(spec.SDS_PATH.read_bytes()).hexdigest() == spec.SDS_SHA256
    assert spec.role_counts(1024) == (205,102,154,154,102,307)
    assert spec.role_counts(512) == (102,51,77,77,51,154)


def test_combinations_and_distinct_initial_states():
    for course, count in [('D1',154),('D2',226),('D3',334)]:
        assert len(cases.groups(course)) == count
    a=cases.suite('D3','A',20)
    b=cases.suite('D3','B',20)
    assert len(a)==6680
    assert set(c['case_sha256'] for c in a).isdisjoint(c['case_sha256'] for c in b)
    assert a[0]['initial_joint_offsets'] != a[1]['initial_joint_offsets']
    assert a == cases.suite('D3','A',20)
    assert all(c['max_steps'] > 0 for c in a)


def test_learning_budget_and_gate_boundaries():
    assert sum(spec.BUDGETS.values()) == 196608000
    assert spec.gate_rounds('D1',1024) == (200,400,500)
    assert spec.gate_rounds('PPO',512) == (500,1000,1500,2000)


def test_evaluation_missing_windows_and_pair_coverage():
    from src.tasks.velocity.v10.evaluation import grade_case, grade_group
    row=cases.suite('D1','A',1)[0]
    assert not grade_case(row, [], completed=True)['passed']
    candidate=[dict(passed=True,windows={'walk':{'rmse':.05,'pitch_rms':2.}},case_sha256=str(i)) for i in range(20)]
    reference=[dict(passed=True,windows={'walk':{'rmse':.04,'pitch_rms':1.5}},case_sha256=str(i)) for i in range(20)]
    assert grade_group(candidate,reference,final=False)['passed']
    candidate[0]['windows']={};candidate[1]['windows']={};candidate[2]['windows']={}
    assert not grade_group(candidate,reference,final=False)['passed']


def test_repeated_candidate_cannot_fake_denominator():
    from src.tasks.velocity.v10.evaluation import grade_group
    reference=[dict(passed=True,windows={'walk':{'rmse':.04,'pitch_rms':1.}},case_sha256=str(i)) for i in range(20)]
    assert not grade_group([reference[0]]*20,reference,final=False)['passed']


def test_missing_reference_window_cannot_bypass_coverage():
    from src.tasks.velocity.v10.evaluation import grade_group
    candidate=[dict(passed=True,windows={'walk':{'rmse':.04,'pitch_rms':1.}},case_sha256=str(i)) for i in range(20)]
    reference=[dict(passed=True,windows={},case_sha256=str(i)) for i in range(20)]
    assert not grade_group(candidate,reference,final=False)['passed']


def _trace_row(**kw):
    return dict(phase='WALK',command=.3,position=[0.,0.],v_body=[.3,0.],v_world=[.3,0.],roll_deg=0.,pitch_deg=0.,clearance=.4,safe_support=[True]*4,stable=False,**kw)


def test_completed_flag_does_not_substitute_for_task_duration():
    from src.tasks.velocity.v10.evaluation import grade_case
    row=cases.suite('D1','A',1)[0]
    assert not grade_case(row,[_trace_row() for _ in range(150)],completed=True)['passed']


def test_nonfinite_measurement_stops_evaluation():
    import pytest
    from src.tasks.velocity.v10.evaluation import grade_case
    row=cases.suite('D1','A',1)[0]
    trace=[_trace_row() for _ in range(1000)]
    trace[500]['pitch_deg']=float('nan')
    with pytest.raises(FloatingPointError): grade_case(row,trace,completed=True)


def test_evaluation_padding_first_terminal_and_equal_upper_bound():
    from src.tasks.velocity.v10.eval_worker import BatchAccounting
    b=BatchAccounting([10,10])
    b.step([0]);b.step([0]);b.step([1])
    assert b.total==6 and b.effective==4 and b.padding==2
    assert b.first_terminals == {0:1,1:3}
    assert b.upper_bound == 20
    import pytest
    with pytest.raises(ValueError): BatchAccounting([10,11])


def test_flat_stop_restart_is_five_seconds_not_five_meters():
    from src.tasks.velocity.v10.evaluation import grade_case
    row=next(c for c in cases.suite('D1','A',1) if c['task']=='F_STOP' and c['stop_mode']=='instant' and c['speed']==.3)
    trace=[];x=row['spawn_x']
    for phase,count in [('WALK',400),('HOLD',1000),('RESTART',250)]:
        for _ in range(count):
            r=_trace_row();r['phase']=phase
            if phase=='HOLD':r.update(command=0.,v_body=[0.,0.],v_world=[0.,0.],stable=True)
            x+=r['command']*.02;r['position']=[x,0.];trace.append(r)
    assert grade_case(row,trace,completed=True)['passed']


def test_stand_drift_uses_start_of_last_ten_seconds():
    from src.tasks.velocity.v10.evaluation import grade_case
    row=next(c for c in cases.suite('D1','A',1) if c['task']=='F_STAND')
    trace=[]
    for i in range(1000):
        r=_trace_row();r.update(phase='STAND',command=0.,v_body=[0.,0.],v_world=[0.,0.],stable=True,position=[.11 if i>=500 else 0.,0.])
        trace.append(r)
    assert not grade_case(row,trace,completed=True)['passed']


def test_gpu_execution_lock_survives_parent_restart_boundary(tmp_path):
    from src.tasks.velocity.v10.gpu_lock import gpu_execution_lock
    import subprocess,sys,time
    lock=tmp_path/'gpu.lock';ready=tmp_path/'ready'
    child=subprocess.Popen([sys.executable,'-c',
        'from src.tasks.velocity.v10.gpu_lock import gpu_execution_lock; from pathlib import Path; import sys\n'
        'with gpu_execution_lock(sys.argv[1]):\n Path(sys.argv[2]).write_text("ready"); sys.stdin.readline()',str(lock),str(ready)],stdin=subprocess.PIPE)
    try:
        deadline=time.monotonic()+20
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert ready.exists()
        import pytest
        with pytest.raises(BlockingIOError):
            with gpu_execution_lock(lock,blocking=False):pass
    finally:
        child.communicate(b'\n',timeout=20)
    with gpu_execution_lock(lock,blocking=False):pass


def test_teacher_admission_stops_below_eighteen_without_partial_pass():
    from src.tasks.velocity.v10.eval_worker import teacher_group_failed
    assert teacher_group_failed([{'passed':True}]*17+[{'passed':False}]*3)
    assert not teacher_group_failed([{'passed':True}]*18+[{'passed':False}]*2)


def test_checkpoint_refuses_changed_source_identity(tmp_path):
    import torch,pytest
    from src.tasks.velocity.v10.checkpoint import checkpoint_save,checkpoint_load
    p=tmp_path/'state.pt'
    checkpoint_save(p,{'num_envs':512,'physical_sha256':'physics'})
    state=torch.load(p,weights_only=False)
    assert len(state.get('code_sha256',''))==64
    state['code_sha256']='0'*64;torch.save(state,p)
    with pytest.raises(ValueError,match='identity'):
        checkpoint_load(p,num_envs=512,physical_sha256='physics',restore_random=False)
