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
