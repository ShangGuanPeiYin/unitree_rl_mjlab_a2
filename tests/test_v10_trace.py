import numpy as np
import pytest


def frame(n=2):
    return dict(yaw=np.arange(n)*.2,foot_position_local=np.zeros((n,4,3)),
                contact_raw=np.ones((n,4),bool),top_upward_force=np.ones((n,4))*6,
                failure_masks={k:np.zeros(n,bool) for k in
                               ('body_hit','long_hit','attitude','route_fall','bounds','timeout')})


def test_trace_freezes_first_terminal_and_preserves_rows20(tmp_path):
    from src.tasks.velocity.v10.trace import TraceBatchV2
    b=TraceBatchV2(2,[1,2])
    action=dict(requested_command=np.array([[.3,0,0],[.5,0,0]]),
                teacher_command=np.array([[.3,0,-.1],[.5,0,.1]]),
                action_yaw=np.array([.1,.2]),action_route_error=np.zeros((2,3)),
                teacher_command_valid=np.ones(2,bool),teacher_is_stairs=np.array([False,True]))
    f=frame();f['failure_masks']['bounds'][0]=True
    b.append(action,f,{0,1})
    old=np.arange(20,dtype=float)[None,:]
    b.save(0,tmp_path/'a.npz',old)
    f['yaw'][:]=9;action['teacher_command'][:]=9
    b.append(action,f,{1})
    b.save(1,tmp_path/'b.npz',np.zeros((2,20)))
    with np.load(tmp_path/'a.npz',allow_pickle=False) as a:
        assert a['schema_version'].item()==2
        np.testing.assert_array_equal(a['rows'],old)
        assert a['yaw_next'].shape==(1,)
        assert a['yaw_next'][0]==0 and a['action_yaw'][0]==.1
        assert a['teacher_command'][0,2]==-.1
        assert a['failure_bounds'].tolist()==[True]
        assert a['foot_position_local'].shape==(1,4,3)
    with np.load(tmp_path/'b.npz',allow_pickle=False) as a:
        assert a['yaw_next'].tolist()==[.2,9]
    with pytest.raises(ValueError,match='frozen'):
        b.append(action,f,{0})


@pytest.mark.parametrize('bad', ['shape','finite','mask'])
def test_trace_invalid_data_fails_engineering(bad):
    from src.tasks.velocity.v10.trace import TraceBatchV2
    a=dict(requested_command=np.zeros((2,3)),teacher_command=np.zeros((2,3)),
           action_yaw=np.zeros(2),action_route_error=np.zeros((2,3)),
           teacher_command_valid=np.ones(2,bool),teacher_is_stairs=np.zeros(2,bool))
    f=frame()
    if bad=='shape': f['foot_position_local']=np.zeros((2,3,3))
    if bad=='finite': f['top_upward_force'][0,0]=np.nan
    if bad=='mask': f['failure_masks']['bounds']=np.array([0.,.5])
    with pytest.raises(ValueError):TraceBatchV2(2,[1,1]).append(a,f,{0,1})


def test_trace_compact_fixed_buffers_enforce_full_boundary_and_skip_padding(tmp_path):
    from src.tasks.velocity.v10.trace import TraceBatchV2
    b=TraceBatchV2(2,[2,1])
    a=dict(requested_command=np.zeros((2,3),np.float32),teacher_command=np.zeros((2,3),np.float32),
           action_yaw=np.zeros(2,np.float32),action_route_error=np.zeros((2,3),np.float32),
           teacher_command_valid=np.ones(2,bool),teacher_is_stairs=np.zeros(2,bool))
    f=frame();b.append(a,f,{0,1});b.save(1,tmp_path/'early.npz',np.zeros((1,20)))
    b.append(a,f,{0})
    assert b.counts.tolist()==[2,1]
    assert isinstance(b.buffers,dict) and all(isinstance(v,np.ndarray) for v in b.buffers.values())
    assert b.buffers['requested_command'].shape==(2,2,3)
    assert b.buffers['requested_command'].dtype==np.float32
    before=b.buffers['yaw_next'][1,0].copy()
    f['yaw'][1]=100;b.append(a,f,set())
    assert b.buffers['yaw_next'][1,0]==before
    with pytest.raises(ValueError,match='capacity'):b.append(a,f,{0})
    b.save(0,tmp_path/'full.npz',np.zeros((2,20)))
    with np.load(tmp_path/'full.npz') as saved:assert saved['rows'].shape==(2,20)


def test_diagnostic_manifest_guard_preserves_formal_denominator():
    from src.tasks.velocity.v10.eval_worker import validate_trace_mode,teacher_group_failed
    assert validate_trace_mode(None,'teachers','A',20,True) is False
    assert validate_trace_mode([{}]*4,'teachers','A',4,True) is True
    assert teacher_group_failed([dict(passed=True)]*18+[dict(passed=False)]*2) is False
    with pytest.raises(ValueError):validate_trace_mode([{}]*4,'candidate','A',4,True)
    with pytest.raises(ValueError):validate_trace_mode([{}]*3,'teachers','A',4,True)


def test_action_snapshot_uses_same_adapter_and_never_mutates_student():
    import torch
    from src.tasks.velocity.v10.trace import action_snapshot
    from src.tasks.velocity.v10.teachers import blind_yaw_command
    prop=torch.zeros((2,45));prop[:,6:9]=torch.tensor([.3,.1,.2])
    route=torch.tensor([[1.,0.,1.],[2.,0.,1.]])
    before=prop.clone();obs=dict(proprio=prop,route_error=route)
    result=action_snapshot(obs,[False,True])
    np.testing.assert_allclose(result['teacher_command'][0],blind_yaw_command(prop,route)[0])
    np.testing.assert_allclose(result['teacher_command'][1],prop[1,6:9])
    assert torch.equal(prop,before)
    result['requested_command'][:]=0
    assert torch.equal(prop,before)
    candidate=action_snapshot(obs)
    assert not candidate['teacher_command_valid'].any()


@pytest.mark.parametrize("mode", ["early_stop", "complete", "step_error"])
def test_diagnostic_real_loop_freezes_each_case_and_charges_padding(monkeypatch,tmp_path,mode):
    import torch
    import weakref
    from src.tasks.velocity.v10 import eval_worker,env,model,teachers,runtime
    refs=[];closed=[]
    batch=[dict(case_sha256=str(i),group_id='g'+str(i//4),max_steps=4,passed=(mode!='early_stop' or i!=3)) for i in range(8)]
    class FakeModel:
        @classmethod
        def from_stairs(cls,**kwargs):return cls()
        def eval(self):return self
        def requires_grad_(self,*args):return self
    class FakeTeacher:
        def __init__(self,*args):pass
        def to(self,*args):return self
        def label(self,obs,mask):return torch.zeros((len(mask),12))
    class FakeMetrics:
        def __init__(self,c):self.case=c;self.count=0;self.rows=np.zeros((6,20))
        def update(self,row):self.rows[self.count,1]=row['command'];self.count+=1
        def finish(self,**kwargs):return dict(case_sha256=self.case['case_sha256'],passed=self.case['passed'])
    class FakeEnv:
        def __init__(self,**kwargs):
            assert all(r() is None for r in refs), 'previous group still resident'
            self.n=kwargs['num_envs'];self.tick=0;self.closed=False
            self.callback=self.observe  # Same bound-method cycle as HeightMap.
            refs.append(weakref.ref(self))
        def observe(self):
            prop=torch.zeros((self.n,45));prop[:,6]=.3
            route=torch.tensor([[0.,np.sin(self.tick*.1),np.cos(self.tick*.1)]]*self.n)
            return dict(proprio=prop,route_error=route)
        def metadata(self):return dict(teacher=['blind']*self.n)
        def step(self,action):
            if mode=='step_error':raise RuntimeError('injected step failure')
            self.tick+=1;n=self.n
            f={k:torch.as_tensor(v) if k!='failure_masks' else {kk:torch.as_tensor(vv) for kk,vv in v.items()} for k,v in frame(n).items()}
            f.update(yaw=torch.full((n,),self.tick*.1),command=torch.full((n,),.3),position=torch.zeros((n,3)),
                     v_body=torch.zeros((n,3)),v_world=torch.zeros((n,3)),roll_deg=torch.zeros(n),pitch_deg=torch.zeros(n),
                     clearance=torch.full((n,),.4),safe_support=torch.ones((n,4),dtype=torch.bool),stable=torch.zeros(n,dtype=torch.bool),
                     top_event=torch.zeros(n,dtype=torch.bool),failure=torch.zeros(n,dtype=torch.bool),timeout=torch.zeros(n,dtype=torch.bool),phase=['WALK']*n)
            return self.observe(),None,None,dict(frame=f,episodes=[dict(env_id=self.tick-1)])
        def close(self):self.closed=True;closed.append(True)
    monkeypatch.setattr(env,'V10Env',FakeEnv);monkeypatch.setattr(model,'V10ActorCritic',FakeModel)
    monkeypatch.setattr(teachers,'DualTeacher',FakeTeacher)
    monkeypatch.setattr(teachers,'blind_yaw_command',lambda p,r: p[:,6:9]+torch.tensor([0.,0.,-.05]),raising=False)
    monkeypatch.setattr(runtime,'physical_contract',lambda e:dict(sha256='cpu-fake'))
    monkeypatch.setattr(eval_worker,'EpisodeMetrics',FakeMetrics)
    def run():return eval_worker.run_suite(course='D1',suite_name='A',replicas=4,model_path=None,reference='teachers',
         output_dir=tmp_path/'diag',ledger_path=tmp_path/'cost.jsonl',eval_limit=32,device='cpu',
         stop_on_failed_group=True,diagnostic_manifest=batch)
    if mode=='step_error':
        with pytest.raises(RuntimeError,match='injected step failure'):run()
        # Exception traceback retains step(self) until unwound; the worker exits.
        import gc
        gc.collect()
        assert closed==[True] and all(r() is None for r in refs)
        return
    result=run()
    assert all(r() is None for r in refs), 'finished group still resident'
    import json
    stages=[json.loads(line) for line in (tmp_path/'diag'/'memory_groups.jsonl').read_text().splitlines()]
    assert len(stages)==4*len(refs)
    assert [r['stage'] for r in stages[:4]]==['before_environment','environment_ready','before_collection','after_collection']
    assert all(r['process_rss_bytes']>0 and r['host_available_bytes']>0 for r in stages)
    if mode=='complete':
        assert result['complete'] and result['tested_case_count']==8
        assert result['physical_samples']==32
    else:
        assert result['diagnostic_only'] and not result['complete'] and result['tested_case_count']==4
        assert result['physical_samples']==16 and result['effective_samples']==10 and result['padding_samples']==6
    assert not (tmp_path/'cache').exists()
    for i in range(4):
        with np.load(tmp_path/'diag'/f'trace_{i}.npz') as a:
            assert a['rows'].shape==(i+1,20)
            np.testing.assert_allclose(a['action_yaw'],np.arange(i+1)*.1)
            np.testing.assert_allclose(a['yaw_next'],np.arange(1,i+2)*.1)
            np.testing.assert_allclose(a['teacher_command'][:,2],-.05)
