import torch
from src.tasks.velocity.v08.env import V08Env

def test_real_cpu_scene_signed_command_causal_and_schedule_no_update_reset():
    e=V08Env(num_envs=2,device='cpu',seed=42,initial_cases=[dict(task='up_stop',tile_id=21,speed=.5,x=-1.,y=0.,yaw=0.),dict(task='down_stop',tile_id=18,speed=.3,x=2.2,y=0.,yaw=0.)])
    try:
        obs,_=e.reset();assert obs['proprio'].shape==(2,45)
        torch.testing.assert_close(e.command[:,0],torch.tensor([.5,-.003]))
        e.request_speed([0],-.5);assert abs(e.command[0,0].item()-.49)<1e-6
        _,r,_,_,info=e.step(torch.zeros(2,12));assert abs(info['command_executed'][0,0].item()-.49)<1e-6
        assert torch.isfinite(r).all();length=e.episode_length_buf.clone()
        e.set_schedule(dict(phase='A',cases=e.case_templates,generation=1),stage_iteration=1)
        assert torch.equal(e.episode_length_buf,length)
        state=e.state_dict();e.load_state_dict(state,restore_physics=False)
        assert e.recovery_mode=='fresh_physical_episode';assert (e.episode_length_buf==0).all()
    finally:e.close()

def test_arbitrary_boundary_observation_reads_do_not_advance_and_reward_is_latched():
    from src.tasks.velocity.v08 import spec
    e=V08Env(num_envs=1,device='cpu',seed=42,initial_cases=[dict(task='flat_stop',tile_id=0,speed=.5,x=-1.,y=0.,yaw=0.)])
    try:
        e.request_speed([0],0.);k=e.command_state.k.clone();history=e.history_cache.history.clone()
        for _ in range(3):e.get_observations()
        assert torch.equal(e.command_state.k,k) and torch.equal(history,e.history_cache.history)
        assert e.executed_c.item()==.5 and abs(e.command[0,0]-.495)<1e-6
        _,_,_,_,info=e.step(torch.zeros(1,12))
        assert abs(info['command_executed'][0,0]-.495)<1e-6
        expected=spec.CONTROL_DT*spec.REWARD_WEIGHTS['tracking_lin_vel']*torch.exp(-((e.robot.data.root_link_lin_vel_b[:,0]-.495).square()+e.robot.data.root_link_lin_vel_b[:,1].square())/spec.TRACKING_EXP_SCALE)
        torch.testing.assert_close(info['step_reward_tensors']['reward_weighted/tracking_lin_vel'],expected)
        assert e.command[0,0]<info['command_executed'][0,0]
        assert not info['success'].any()
    finally:e.close()


def test_external_zero_classification_uses_actual_position_and_terminal_hook_precedes_reset():
    from types import SimpleNamespace
    from src.tasks.velocity.v08 import spec
    dummy=SimpleNamespace(device='cpu',robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[4.,0.,4.4],[9.,0.,8.4],[-1.,0.,.4]]))),origins=torch.zeros(23,3),tile_ids=torch.tensor([21,21,21]),tile_N=torch.tensor([x.N for x in spec.TERRAIN_TABLE]),tile_d=torch.tensor([x.step_depth for x in spec.TERRAIN_TABLE]),parking_location=torch.full((3,),-1))
    V08Env._set_parking_location(dummy,torch.arange(3));assert dummy.parking_location.tolist()==[3,1,2]
    seen=[]
    e=V08Env(num_envs=1,device='cpu',seed=42,initial_cases=[dict(task='flat_stop',tile_id=0,speed=.5,x=-1.,y=0.,yaw=0.)],pre_reset_callback=lambda env,info:seen.append((int(env.episode_length_buf[0]),info['episodes'][0]['env_id'])))
    try:
        e.tasks.steps[0]=399
        _,_,done,_,info=e.step(torch.zeros(1,12))
        assert done.all() and seen==[(1,0)] and e.episode_length_buf[0]==0
        assert info['episodes'][0]['reason']=='flat_progress'
    finally:e.close()

def test_evaluation_trace_contains_real_continuous_action_boundaries():
    e=V08Env(num_envs=1,device='cpu',seed=42,evaluation=True,initial_cases=[dict(task='flat_stop',tile_id=0,speed=.5,x=-1.,y=0.,yaw=0.)])
    try:
        start=float(e.robot.data.root_link_pos_w[0,0]-e.origins[0,0]);e.request_speed([0],-.5)
        for _ in range(3):e.step(torch.zeros(1,12))
        rows=e.trace[0];assert [r['control_step'] for r in rows]==[0,1,2,3]
        assert rows[0]['local_x']==start
        assert abs(rows[1]['executed_c']-.49)<1e-6 and rows[1]['transition']
        assert abs(rows[-1]['local_x']-float(e.robot.data.root_link_pos_w[0,0]-e.origins[0,0]))<1e-7
    finally:e.close()

def test_fresh_new_sampler_discards_constructor_draws_without_resetting_physics():
    e=V08Env(num_envs=1,device='cpu',seed=42)
    try:
        e.step(torch.zeros(1,12));length=e.episode_length_buf.clone();qpos=e.sim.data.qpos.clone()
        e.reset_task_sampler(seed=42);first=e._sample_case(e.case_templates[0])
        e.reset_task_sampler(seed=42);second=e._sample_case(e.case_templates[0])
        assert first==second and torch.equal(e.episode_length_buf,length) and torch.equal(e.sim.data.qpos,qpos)
        state=e.state_dict();calls=[];original=e._reset_ids
        def counted(ids,**kw):calls.append(list(ids));return original(ids,**kw)
        e._reset_ids=counted;e.load_state_dict(state,restore_physics=False)
        assert len(calls)==1 and e.recovered_logical_state['tasks']['tensors']['steps'][0]==1
        assert e.tasks.steps[0]==0
    finally:e.close()

def test_actual_top_support_event_keeps_simulating_same_physical_episode():
    e=V08Env(num_envs=1,device='cpu',seed=42,initial_cases=[dict(task='down_stop',tile_id=18,speed=.3,spawn_x=2.2,spawn_y=0.,spawn_yaw=0.)])
    try:
        for _ in range(24):
            _,_,done,_,_=e.step(torch.zeros(1,12))
            assert not done.any()
        assert e.climb.climbed.all() and e.climb.consecutive_steps[0]>=5
        assert e.episode_length_buf[0]==24 and e.command_state.k[0]==25
    finally:e.close()

def test_fix1_real_producer_first_zero_hold_is_not_labelled_deceleration():
    from src.tasks.velocity.v08.tasks import TaskMachine
    e=V08Env(num_envs=1,device='cpu',seed=42,evaluation=True,initial_cases=[dict(task='flat_stop',tile_id=0,speed=.3,x=-1.,y=0.,yaw=0.)])
    try:
        e.tasks.mode[0]=TaskMachine.DECEL;e.request_speed([0],0.)
        for _ in range(101):e.step(torch.zeros(1,12))
        assert e.trace[0][100]['transition'] and e.trace[0][100]['segment'].endswith('_deceleration')
        assert not e.trace[0][101]['transition'] and e.trace[0][101]['executed_c']==0.
        assert e.trace[0][101]['segment'].endswith('_hold')
        assert e.tasks.hold_steps[0]==1
    finally:e.close()


def test_fix1_actual_terminal_masks_producer_to_consumer_keep_failure_categories():
    from dataclasses import replace
    from src.tasks.velocity.v08 import evaluation,events
    cases=[c for c in evaluation.make_cases(phase='A').cases if c['task']=='up_stop'][:4]
    e=V08Env(num_envs=4,device='cpu',seed=42,evaluation=True,initial_cases=cases)
    try:
        actual=e._task_state
        def terminal_masks():
            state=actual();masks={name:value.clone() for name,value in state.failure_masks.items()}
            masks['body'][0]=True;masks['fall'][1]=True
            return replace(state,failure=torch.tensor([True,True,False,False]),failure_masks=masks,reason=torch.tensor([events.FAILURE_CODES['body'],events.FAILURE_CODES['fall'],0,0]))
        e._task_state=terminal_masks
        e.tasks.leg_steps[2]=e.tasks._limit(2)-1
        e.tasks.direction_pending[3]=True;e.tasks.direction_steps[3]=399
        _,_,done,_,info=e.step(torch.zeros(4,12))
        assert done.all();rows=info['episodes']
        for case,raw,reason,physical in zip(cases,rows,('body','fall','timeout','cannot_reverse'),(True,True,False,False)):
            row=evaluation.normalize_record(raw,case)
            assert raw['reason']==reason and raw.get('physical_failure') is physical
            assert raw['failure_kind']==('physical' if physical else 'task')
            assert row['status']=='failure' and row['reason']==reason and row['physical_failure'] is physical
            if physical:assert raw['failure_masks'][reason]
    finally:e.close()

def test_fix1_frozen_baseline_gate_and_e4_cases_construct_real_env_with_optional_trigger():
    from src.tasks.velocity.v08 import evaluation
    baseline=evaluation.make_cases(kind='baseline').cases
    gate=evaluation.make_cases(phase='A').cases
    stage_e=evaluation.make_cases(phase='E').cases
    cases=[next(c for c in baseline if c['task']=='flat_stop'),next(c for c in baseline if c['task']=='up_stop'),
           next(c for c in baseline if c['task']=='down_stop' and c['terrain_name']=='D20'),next(c for c in baseline if c['task']=='down_stop' and c['terrain_name']=='L40'),
           next(c for c in gate if c['task']=='flat_stop'),next(c for c in gate if c['task']=='up_stop'),
           next(c for c in stage_e if c['task']=='E4'),next(c for c in stage_e if c['task']=='E3' and c['script']=='up_reverse')]
    e=V08Env(num_envs=8,device='cpu',seed=42,evaluation=True,initial_cases=cases)
    try:
        assert all(e.tasks.points[i]==0. for i in range(7))
        assert e.tasks.points[7]==cases[7]['trigger_p']*8.
        _,reward,_,_,info=e.step(torch.zeros(8,12))
        assert torch.isfinite(reward).all() and all(len(rows)==2 for rows in e.trace)
        assert [c['trigger_p'] for c in cases[:7]]==[None]*7
        template=dict(cases[7],trigger_p=None,training=True,trigger_range=(.3,.7))
        sampled=[e._sample_case(template)['trigger_p'] for _ in range(10)]
        assert all(.3<=p<=.7 for p in sampled) and any(abs(p-.5)>1e-6 for p in sampled)
    finally:e.close()

def test_fix1_e3_reverse_and_e4_producer_raw_are_accepted_by_unchanged_consumer():
    """Deterministic kinematic fixture, not a trained-policy physics result.

    Drive production CommandState/TaskMachine/StandWindow and the actual env
    trace/window/terminal exporters. The separate first-hold test runs physics.
    """
    from types import SimpleNamespace
    from src.tasks.velocity.v08 import evaluation,spec
    from src.tasks.velocity.v08.tasks import TaskMachine
    frozen=evaluation.make_cases(phase='E').cases
    cases=[next(c for c in frozen if c['task']=='E3' and c['script']==script) for script in ('up_reverse','down_reverse')]
    cases.append(next(c for c in frozen if c['task']=='E4'))
    for case in cases:
        e=V08Env(num_envs=1,device='cpu',seed=42,evaluation=True,initial_cases=[case])
        try:
            x=torch.tensor([e.trace[0][0]['local_x']]);y=torch.tensor([e.trace[0][0]['local_y']]);yaw=torch.tensor([e.trace[0][0]['yaw']]);zero=torch.zeros(1);yes=torch.ones(1,dtype=torch.bool);no=~yes
            accepted=None
            for _ in range(12000):
                e.command_state.latch()
                if e.tasks.mode[0]!=TaskMachine.DONE:x=x+e.executed_c*spec.CONTROL_DT
                mid=e.tasks.stop_kind==1
                stable=e.standing.update(torch.stack((x,y),-1),e.executed_c.abs(),zero,zero,yaw,zero,torch.ones(1)*.4,e.executed_c.abs(),yes,yes,mid)
                old_mode=e.tasks.mode.clone();old_hold=e.tasks.hold_steps.clone();old_kind=e.tasks.stop_kind.clone()
                result=e.tasks.advance(x,x>=8.8,x<=-.8,stable,no,e.executed_c,no,transition=e.command_state.executed_transition,requested=e.command_state.executed_requested)
                e._record_segments=list(e.tasks.action_segments);e.episode_length_buf.copy_(e.tasks.steps)
                restarting=(old_mode!=TaskMachine.WALK)&(e.tasks.mode==TaskMachine.WALK)
                if restarting[0]:
                    window=e._export_window(0);window['count']=int(old_hold[0])+1;window['kind']='mid' if old_kind[0]==1 else 'top';e.stop_windows[0].append(window)
                request=result['request'];ids=torch.isfinite(request).nonzero().flatten();e.command_state.request(ids,request[ids])
                if bool((request==0).any() or restarting.any()):e.standing.reset(torch.tensor([0]))
                e.command[:,0]=e.command_state.advance()
                state=SimpleNamespace(local=torch.stack((x,y,torch.ones(1)*.4),-1),yaw=yaw,failure_masks={name:no.clone() for name in ('side','bounds','body','nonfoot','posture','fall','contact_overflow','nonfinite')})
                e.max_stair_abs_yaw=torch.maximum(e.max_stair_abs_yaw,torch.where((x>=0)&(x<=8.),yaw.abs(),zero))
                e._record_action_trace(state,request)
                done=result['success']|result['failure']
                if done[0]:
                    raw=e._records(state,result,done)[0];accepted=evaluation.normalize_record(raw,case)
                    stops=[f['control_step'] for f in raw['trace'] if f['request_event']==0.]
                    for stop in stops:
                        assert raw['trace'][stop+100]['transition']
                        assert raw['trace'][stop+101]['segment'].endswith('_hold')
                    assert accepted['status']=='success',accepted
                    assert [w['kind'] for w in raw['stop_windows']]==(['mid','bottom' if case['resume_direction']<0 else 'top'] if case['task']=='E3' else ['top','bottom'])
                    break
            assert accepted is not None
        finally:e.close()
