"""Real frozen safety hooks and bounded pre-reset sampling, CPU fixtures only."""
import importlib
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace as S, MethodType
import numpy as np
import torch


def env_class():
    try:return importlib.import_module('src.tasks.velocity.v03.env').V03Env
    except ModuleNotFoundError:raise AssertionError('V03 environment implementation is missing') from None


def task_fixture(parking=False,batch=1):
    from src.tasks.velocity.v02_formal.success import ClimbState,ParkingDiagnosticState
    from src.tasks.velocity.v03.speed import SegmentSpeedState
    cls=env_class();e=object.__new__(cls)
    d=S(root_link_pos_w=torch.tensor([[1.6,0.,.8]]*batch),root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]*batch),
        site_pos_w=torch.tensor([[[1.7,.1,.352],[1.7,-.1,.352],[1.2,.1,.352],[1.2,-.1,.352]]]*batch),
        root_link_lin_vel_b=torch.tensor([[.4,0.,0.]]*batch),root_link_lin_vel_w=torch.tensor([[.35,0.,0.]]*batch),root_link_ang_vel_b=torch.zeros(batch,3),root_link_ang_vel_w=torch.zeros(batch,3),joint_pos=torch.zeros(batch,12),joint_vel=torch.zeros(batch,12))
    e.robot=S(data=d);e.foot_ids=list(range(4));e.origins=torch.zeros(10,3);e.tile_ids=torch.full((batch,),5,dtype=torch.long)
    e.command=torch.tensor([[.4,0.,0.]]*batch);e.geom_top=torch.tensor([.32,.32,.32]);e.geom_step=torch.tensor([4,4,4]);e.geom_tile=torch.tensor([5,5,6])
    e.bounds=torch.tensor([[.6,.8,-.4,.4],[.8,2.8,-.4,.4],[.8,2.8,-.4,.4]])
    tops=torch.zeros(batch,4,3);tops[:,:,1]=6.
    e.contacts=S(top_forces=tops,link_force=torch.zeros(batch,1,3),body_link_mask=torch.tensor([True]),nonfoot_duration=torch.zeros(batch,1))
    e.control_body_hit=torch.zeros(batch,dtype=torch.bool);e.control_long_hit=e.control_body_hit.clone();e.climb=ClimbState(batch)
    e.parking_diagnostics=ParkingDiagnosticState(batch);e.parking_diagnostic=parking;e.stop_steps=torch.zeros(batch,dtype=torch.long)
    e.episode_length_buf=torch.zeros(batch,dtype=torch.long);e.start_vx=torch.full((batch,),.4);e.scene=S(num_envs=batch)
    e.speed_state=SegmentSpeedState(batch);e.initial_pose=torch.tensor([[-1.,0.,0.]]*batch);e.slot_classes=torch.ones(batch,dtype=torch.long)
    e.evaluation=False;e.run_seed=42;e.checkpoint_identity='fixture';e.stage='S1';e._executed_command=None
    e.sim=S(device='cpu',data=S(qpos=torch.zeros(batch,19),qvel=torch.zeros(batch,18)))
    def step():
        e._executed_command=e.command.clone();e.episode_length_buf+=1
        return e._task_state()
    return e,step

def complete_fixture(batch=2):
    """Fake only simulator I/O; use real frozen reset/step/reward/safety methods."""
    from src.tasks.velocity.v02.rewards import ParkingState,FirstFootfallState,ProgressState
    from src.tasks.velocity.v02.terrain import atlas_metadata
    from src.tasks.velocity.v03.spec import recipe_config
    e,_=task_fixture(batch=batch);d=e.robot.data;b=batch
    class Scene(dict):
        num_envs=b
        def reset(self,ids):pass
        def write_data_to_sim(self):pass
        def update(self,dt):pass
    e.scene=Scene({'robot/root_angmom':S(data=torch.zeros(b,3))})
    e.cfg=S(sim=S(mujoco=S(timestep=.005)),decimation=4)
    e.sim.reset=lambda ids:None;e.sim.step=lambda:None;e.sim.forward=lambda:None;e.sim.sense=lambda:None
    d.default_root_state=torch.zeros(b,13);d.default_root_state[:,2]=.5;d.default_root_state[:,3]=1
    d.default_joint_pos=torch.zeros(b,12);d.joint_acc=torch.zeros(b,12);d.joint_vel=torch.zeros(b,12)
    d.root_link_ang_vel_b=torch.zeros(b,3);d.root_link_ang_vel_w=torch.zeros(b,3);d.projected_gravity_b=torch.tensor([[0.,0.,-1.]]*b)
    d.site_lin_vel_w=torch.zeros(b,4,3);d.soft_joint_pos_limits=torch.tensor([[[-1.,1.]]*12]*b)
    e.robot.indexing=S(joint_v_adr=list(range(12)));e.sim.data.qfrc_actuator=torch.zeros(b,12)
    e.robot.write_root_state_to_sim=lambda state,ids:(d.root_link_pos_w.__setitem__(ids,state[:,:3]),d.root_link_quat_w.__setitem__(ids,state[:,3:7]),d.root_link_lin_vel_b.__setitem__(ids,state[:,7:10]))
    e.robot.write_joint_state_to_sim=lambda pos,vel,env_ids:(d.joint_pos.__setitem__(env_ids,pos),d.joint_vel.__setitem__(env_ids,vel))
    e.joint_ids=list(range(12));e.action_manager=S(action=torch.zeros(b,12))
    e.action_manager.process_action=lambda a:setattr(e.action_manager,'action',a.clone())
    e.action_manager.apply_action=lambda:None;e.action_manager.reset=lambda ids:e.action_manager.action.__setitem__(ids,0.)
    e.contacts.sample=lambda *args,**kwargs:None;e.contacts.reset=lambda ids:None
    e.contacts.support_geom=torch.ones(b,4,dtype=torch.long);e.contacts.found=torch.ones(b,4,dtype=torch.bool);e.contacts.support=torch.ones(b,4,dtype=torch.bool)
    e.contacts.foot_force=torch.zeros(b,4,3);e.contacts.foot_force[:,:,2]=6.;e.contacts.contact_duration=torch.ones(b,4);e.contacts.link_names=['body']
    e.geom_step_bounds=e.bounds.clone();e.recipe_options=recipe_config();e.recipe='C';e.mode='formal'
    e.parking=ParkingState(b);e.footfall=FirstFootfallState(b,40);e.progress=ProgressState(b)
    e.initial_cases=None;e.fixed_vx=None;e.rng=torch.Generator().manual_seed(42);e.stage_iteration=0;e.previous_stage=None
    meta=atlas_metadata();e.spawn_x=torch.tensor([m.terrain_origin_world[0]-m.frame_origin_world[0] for m in meta])
    e.initial_noise=torch.tensor([[(m.initial_distance_range[1]-m.initial_distance_range[0])/2,(m.initial_y_range[1]-m.initial_y_range[0])/2,(m.initial_yaw_range[1]-m.initial_yaw_range[0])/2] for m in meta])
    e.perturb_ramp=torch.tensor([m.perturb_ramp_updates for m in meta]);e.history_cache=S(reset=lambda obs,ids:None);e.map_cache=S(reset=lambda *args:None)
    e.stop=torch.zeros(b,dtype=torch.bool);e.last_contact=torch.zeros(b,4,dtype=torch.bool);e.episode_reward_sums={}
    for name in ('sum_vx','sum_speed','max_lateral','highest','slip_sum','nonfoot_sum','limit_sum','peak_torque','height_defaults','first_step_support_time','first_nonfoot_time','max_body_x','stagnation_total','stagnation_current','stagnation_longest','max_heading','sum_vx_world','sum_horizontal_speed_world','moving_sum_vx','moving_sum_speed'):
        setattr(e,name,torch.zeros(b))
    e.moving_count=torch.zeros(b,dtype=torch.long);e.first_nonfoot_link=torch.full((b,),-1,dtype=torch.long);e.first_nonfoot_force=torch.zeros(b,3)
    e.first_foot_times=torch.full((b,4),-1.);e.foot_step_times=torch.full((b,4,40),-1.);e.before_first_foot_height=torch.zeros(b,2)
    e.body_crossed_first=torch.zeros(b,dtype=torch.bool);e.all_feet_on_platform=torch.zeros(b,dtype=torch.bool)
    e._sim_step_counter=0;e.common_step_counter=0;e._terminal_speed={}
    e._observations=lambda update=True:{'proprio':d.root_link_pos_w.clone(),'privileged':torch.zeros(b,35)}
    e._foot_heights=lambda:(torch.zeros(b,4),torch.zeros(b,4,dtype=torch.bool))
    return e

def inherited_observation_fixture():
    """Exercise the real privileged/proprio observation construction, no finite stub."""
    from src.tasks.velocity.v02_formal.env import FormalEnv
    e=complete_fixture();e.contacts.top_forces.zero_();e.robot.data.root_link_pos_w[:,0]=.2
    e.history_cache.update=lambda *args:None;e.map_cache.update=lambda *args:None
    e.history_cache.history=torch.zeros(2,5,45);e.map_cache.map=torch.zeros(2,3,48,32)
    e._observations=MethodType(FormalEnv._observations,e)
    return e


class EnvironmentTests(unittest.TestCase):
    def test_inherited_observations_reject_nonfinite_vz_and_qvel_before_state_update(self):
        from src.tasks.velocity.v02_formal.env import NumericalFailure
        for bad in (float('nan'),float('inf')):
            with self.subTest(bad=bad):
                e=inherited_observation_fixture();e.step(torch.zeros(2,12))
                original={k:v.clone() for k,v in vars(e.speed_state).items() if isinstance(v,torch.Tensor)}
                d=e.robot.data;d.root_link_lin_vel_b[0,2]=bad;d.root_link_lin_vel_w[0,2]=bad;e.sim.data.qvel[0,2]=bad
                resets=[];reset=e._reset_ids
                e._reset_ids=lambda ids:(resets.append(ids.clone()),reset(ids))
                with self.assertRaises(NumericalFailure) as caught:e.step(torch.zeros(2,12))
                self.assertEqual(resets,[]);self.assertEqual(e.episode_length_buf.tolist(),[2,2])
                for key,value in original.items():torch.testing.assert_close(getattr(e.speed_state,key),value,rtol=0,atol=0)
                fault=caught.exception.diagnostics
                for key in ('root_link_lin_vel_b','root_link_lin_vel_w','root_link_ang_vel_b','root_link_ang_vel_w','qpos','qvel','command_executed','step'):
                    self.assertIn(key,fault)
                self.assertEqual(fault['step'].dtype,torch.int64);self.assertEqual(fault['step'].tolist(),[2,2])
                self.assertFalse(torch.isfinite(fault['root_link_lin_vel_b'][0,2]));self.assertFalse(torch.isfinite(fault['qvel'][0,2]))
                d.root_link_lin_vel_b.zero_();e.sim.data.qvel.zero_()
                self.assertFalse(torch.isfinite(fault['root_link_lin_vel_b'][0,2]));self.assertFalse(torch.isfinite(fault['qvel'][0,2]))
                self.assertTrue(torch.isfinite(e.sum_speed).all())

    def test_all_root_command_and_physics_components_fail_before_speed_mutation(self):
        from src.tasks.velocity.v02_formal.env import NumericalFailure
        names=('root_link_lin_vel_b','root_link_lin_vel_w','root_link_ang_vel_b','root_link_ang_vel_w')
        cases=[(name,axis) for name in names for axis in range(3)]
        cases += [('command',axis) for axis in range(3)]
        cases += [('qpos',axis) for axis in (0,18)]+[('qvel',axis) for axis in (0,17)]
        for name,axis in cases:
            for bad in (float('nan'),float('inf')):
                with self.subTest(name=name,axis=axis,bad=bad):
                    e=inherited_observation_fixture();e.step(torch.zeros(2,12))
                    before={k:v.clone() for k,v in vars(e.speed_state).items() if isinstance(v,torch.Tensor)}
                    owner=e if name=='command' else e.sim.data if name in ('qpos','qvel') else e.robot.data
                    tensor=getattr(owner,name);tensor[0,axis]=bad
                    with self.assertRaises(NumericalFailure) as caught:e.step(torch.zeros(2,12))
                    for key,value in before.items():torch.testing.assert_close(getattr(e.speed_state,key),value,rtol=0,atol=0)
                    key='command_executed' if name=='command' else name
                    self.assertFalse(torch.isfinite(caught.exception.diagnostics[key][0,axis]))
                    self.assertEqual(e.episode_length_buf.tolist(),[2,2]);self.assertTrue(torch.isfinite(e.sum_speed).all())

    def test_terminal_integer_time_identity_for_parking_postclimb_failure(self):
        e=complete_fixture();e.parking_diagnostic=True
        for _ in range(6):
            result=e.step(torch.zeros(2,12));self.assertFalse(result[2].any())
        self.assertTrue(e.climb.climbed.all());self.assertTrue(e.command.eq(0).all())
        e.contacts.link_force[0,0,0]=1.
        _,_,terminated,_,info=e.step(torch.zeros(2,12))
        self.assertEqual(terminated.tolist(),[True,False])
        record=info['episodes'][0]
        self.assertEqual(record['terminal_step'],7);self.assertIs(type(record['terminal_step']),int)
        self.assertEqual(record['climb_step'],6);self.assertIs(type(record['climb_step']),int)
        self.assertIs(record['parking_diagnostic'],True)
        self.assertTrue(record['climb_success']);self.assertTrue(record['failure']);self.assertTrue(record['post_climb_failure'])
        self.assertFalse(record['joint_success']);self.assertEqual(e.episode_length_buf[0].item(),0)

    def test_float32_fixed_command_endpoint_and_reset_failure_restore_classes(self):
        from src.tasks.velocity.v02_formal.env import FormalEnv as FrozenEnv
        cls=env_class()
        with patch.object(FrozenEnv,'__init__',return_value=None):
            e=cls(num_envs=1,device='cpu',fixed_vx=float(np.float32(.6)))
            self.assertIsNotNone(e.speed_state)
        e=complete_fixture();e.slot_classes[:]=torch.tensor([0,1]);original=e.slot_classes
        def broken_reset(instance,ids):
            self.assertEqual(instance.slot_classes.tolist(),[0,3]);raise RuntimeError('physics reset unavailable')
        with patch.object(FrozenEnv,'_reset_ids',broken_reset):
            with self.assertRaisesRegex(RuntimeError,'physics reset unavailable'):e._reset_ids(torch.tensor([1]))
        self.assertIs(e.slot_classes,original);self.assertEqual(e.slot_classes.tolist(),[0,1])

    def test_formal_zero_or_negative_command_is_fault_not_parking(self):
        from src.tasks.velocity.v02_formal.env import NumericalFailure
        for value in (0.,-.4):
            e,step=task_fixture();e.command[:,0]=value
            with self.assertRaises(NumericalFailure):step()

    def test_nonfinite_lateral_speed_is_wrapped_with_sample_evidence(self):
        from src.tasks.velocity.v02_formal.env import NumericalFailure
        e=complete_fixture();e.robot.data.root_link_lin_vel_b[0,1]=float('nan')
        with self.assertRaises(NumericalFailure) as fault:e.step(torch.zeros(2,12))
        for name in ('executed_c','body_vx','local_x','step','qpos','qvel'):self.assertIn(name,fault.exception.diagnostics)
        self.assertEqual(e.episode_length_buf.tolist(),[1,1])

    def test_full_step_terminal_samples_bootstrap_and_async_reset(self):
        e=complete_fixture();d=e.robot.data
        e.contacts.top_forces.zero_();d.root_link_pos_w[:,0]=torch.tensor([.2,.2])
        from src.tasks.velocity.v02_formal.env import timeout_steps
        e.episode_length_buf[0]=timeout_steps(torch.tensor([4.]),torch.tensor([.08]),e.start_vx[:1])[0]-1
        d.root_link_lin_vel_b[:,0]=torch.tensor([.7,.3]);oldc=e.command[:,0].clone()
        obs,reward,terminated,truncated,info=e.step(torch.zeros(2,12))
        self.assertEqual(terminated.tolist(),[False,False]);self.assertEqual(truncated.tolist(),[True,False])
        self.assertEqual(info['terminal_observation']['proprio'][:,0].tolist(),[float(torch.tensor(.2)),float(torch.tensor(.2))])
        self.assertEqual(obs['proprio'][0,0].item(),-1.)
        self.assertEqual(info['segment_samples']['local_x'][0].item(),float(torch.tensor(.2)))
        torch.testing.assert_close(info['segment_samples']['executed_c'],oldc,atol=0,rtol=0)
        self.assertEqual(info['speed_snapshot']['sample_count'].tolist(),[1,1])
        self.assertEqual(e.speed_state.sample_count.tolist(),[0,1]);self.assertEqual(e.speed_state.phase_count[1,1].item(),1)
        ep=info['episodes'][0];self.assertEqual(ep['env_id'],0);self.assertFalse(ep['failure']);self.assertFalse(ep['joint_success'])
        self.assertEqual(ep['slot_class'],'current_stairs');self.assertEqual(ep['protocol_id'],'v03-stair-body-speed-v1')
        self.assertEqual(ep['speed_source'],'training_preview');self.assertEqual(ep['reason'],'timeout')
        self.assertEqual(ep['terminal_step'],int(info['segment_samples']['step'][0]));self.assertIsNone(ep['climb_step']);self.assertIs(ep['parking_diagnostic'],False)
        self.assertEqual(info['diagnostics']['joint_status'],'failed')
        self.assertFalse(any('flat' in k for k in info['motion_terms']['groups']))
        saved={k:v.clone() for k,v in info['segment_samples'].items()}
        e.step(torch.zeros(2,12))
        for k,v in saved.items():torch.testing.assert_close(info['segment_samples'][k],v,atol=0,rtol=0)
        self.assertEqual(e.speed_state.phase_count[1,1].item(),2)

    def test_pending_joint_even_after_speed_pass_and_arrival_does_not_wait(self):
        e=complete_fixture();d=e.robot.data;e.contacts.top_forces.zero_()
        for j in range(25):
            d.root_link_pos_w[:,0]=.8 if j==24 else j*.03
            _,_,_,_,info=e.step(torch.zeros(2,12))
        self.assertTrue(info['speed_snapshot']['speed_pass'].all());self.assertEqual(info['diagnostics']['joint_status'],'pending')
        d.root_link_pos_w[:,0]=1.6;e.contacts.top_forces[:,:,1]=6
        for _ in range(5):self.assertFalse(e.step(torch.zeros(2,12))[2].any())
        _,_,terminated,_,info=e.step(torch.zeros(2,12))
        self.assertTrue(terminated.all());self.assertTrue(all(ep['joint_success'] for ep in info['episodes']))
        # An equally safe arrival with too-short speed data terminates immediately.
        e=complete_fixture()
        for _ in range(6):result=e.step(torch.zeros(2,12))
        self.assertTrue(result[2].all());self.assertTrue(all(ep['climb_success'] and not ep['speed_valid'] and not ep['joint_success'] for ep in result[4]['episodes']))

    def test_reset_fixed_spawn_s5_own_slot_ramp_and_local_rng(self):
        e=complete_fixture();e.stage='S5';e.tile_ids[:]=torch.tensor([8,9]);e.slot_classes[:]=torch.tensor([0,1])
        before=torch.random.get_rng_state().clone()
        for iteration,scale in ((0,0.),(200,.5),(400,1.)):
            e.stage_iteration=iteration;g=e.rng.get_state().clone();e._reset_ids(torch.arange(2))
            pose=e.initial_pose.clone();cmd=e.command.clone();self.assertEqual(e.slot_classes.tolist(),[0,1])
            self.assertEqual(pose[0].tolist(),[-1.,0.,0.]);self.assertTrue(torch.all(cmd[:,0]>=.2));self.assertTrue(torch.all(cmd[:,0]<=.6));self.assertTrue(cmd[:,1:].eq(0).all())
            self.assertLessEqual(abs(float(pose[1,0])+1.),.2*scale+1e-7);self.assertLessEqual(abs(float(pose[1,1])),.1*scale+1e-7)
            self.assertLessEqual(abs(float(pose[1,2])),math.pi/18*scale+1e-7)
            e.rng.set_state(g);e._reset_ids(torch.arange(2));torch.testing.assert_close(e.initial_pose,pose,atol=0,rtol=0);torch.testing.assert_close(e.command,cmd,atol=0,rtol=0)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        # Both evaluation and the previous S4 slot retain their own spawn rules.
        e.evaluation=True;e.stage_iteration=0;e._reset_ids(torch.arange(2));self.assertEqual(e.initial_pose[0].tolist(),[-1.,0.,0.])
        self.assertGreater(abs(float(e.initial_pose[1,1])),0.)

    def test_tile_parameters_and_no_flat_stage_slots(self):
        cls=env_class()
        for n in (258,512,1024):
            e=object.__new__(cls);e.stage=None;e.previous_stage=None;e.evaluation=False;e.scene=S(num_envs=n);e.sim=S(device='cpu')
            e.tile_ids=torch.zeros(n,dtype=torch.long);e.slot_classes=torch.zeros(n,dtype=torch.long);e.command=torch.zeros(n,3)
            resets=[];e.reset=lambda:resets.append(True)
            for stage,tile,N,h in [('S1',5,4,.08),('S2',6,8,.12),('S3',7,12,.16),('S4',8,20,.2),('S5',9,40,.2)]:
                e.set_stage(stage,0);ns,hs=e._parameters()
                self.assertTrue((e.tile_ids>=5).all());self.assertTrue((e.slot_classes<=1).all())
                cur=e.slot_classes==1;self.assertTrue((e.tile_ids[cur]==tile).all());self.assertTrue((ns[cur]==N).all())
                torch.testing.assert_close(hs[cur],torch.full_like(hs[cur],h),rtol=0,atol=0)
                if stage=='S1':self.assertTrue(cur.all())
            before=len(resets);e.set_stage('S5',400);self.assertEqual(len(resets),before)
            for invalid in ('flat','R0'):
                with self.assertRaises(ValueError):e.set_stage(invalid)

    def test_real_safe_arrival_and_invalid_contact_cases(self):
        for mutation in ('air','front_only','last_tread','foreign','unsafe','body_only','expired'):
            e,step=task_fixture()
            if mutation=='air':e.contacts.top_forces.zero_()
            if mutation=='front_only':e.contacts.top_forces[:,2:,1]=0
            if mutation=='last_tread':e.contacts.top_forces.zero_();e.contacts.top_forces[:,:,0]=100
            if mutation=='foreign':e.contacts.top_forces.zero_();e.contacts.top_forces[:,:,2]=100
            if mutation=='unsafe':e.robot.data.site_pos_w[:,2,0]=.83
            if mutation=='body_only':e.contacts.top_forces.zero_()
            if mutation=='expired':
                step();step();e.robot.data.root_link_pos_w[:,0]=1.5;e.contacts.top_forces[:,2:,1]=0
                for _ in range(51):step()
                e.robot.data.root_link_pos_w[:,0]=1.6
            for _ in range(8):self.assertFalse(step()[5].any(),mutation)
        e,step=task_fixture()
        for _ in range(5):self.assertFalse(step()[5].any())
        self.assertTrue(step()[5].all());self.assertFalse(step()[5].any())

    def test_failure_priority_and_timeout_are_unchanged(self):
        from src.tasks.velocity.v02_formal.env import timeout_steps
        e,step=task_fixture()
        for _ in range(5):step()
        e.control_body_hit[:]=True;s=step();self.assertTrue(s[6].all());self.assertFalse(s[5].any())
        e,step=task_fixture();e.contacts.top_forces.zero_();e.episode_length_buf[:]=timeout_steps(torch.tensor([4.]),torch.tensor([.08]),e.start_vx)
        s=step();self.assertTrue(s[7].all());self.assertFalse(s[5].any());self.assertFalse(s[6].any())

    def test_executed_command_and_parking_never_enter_body_after_end(self):
        e,step=task_fixture(parking=True)
        for _ in range(6):step()
        samples=e._segment_samples
        self.assertAlmostEqual(float(samples['executed_c'][0]),.4,places=6);self.assertEqual(float(e.command[0,0]),0.)
        count=e.speed_state.sample_count.clone();step()
        self.assertEqual(int(e._segment_samples['sample_phase'][0]),-1)
        self.assertEqual(float(e._segment_samples['executed_c'][0]),0.)
        torch.testing.assert_close(count,e.speed_state.sample_count,rtol=0,atol=0)
        self.assertEqual(e._segment_samples['body_vx'].dtype,torch.float32);self.assertEqual(e._segment_samples['step'].dtype,torch.int64)
        self.assertIn('pitch',samples);self.assertIn('yaw',samples);self.assertIn('world_vx',samples);self.assertIn('local_displacement',samples)

    def test_nonfinite_speed_has_reset_prevention_and_physics_evidence(self):
        from src.tasks.velocity.v02_formal.env import NumericalFailure
        for field in ('root_link_lin_vel_b','root_link_pos_w'):
            e,step=task_fixture();getattr(e.robot.data,field)[0,0]=float('nan')
            with self.assertRaises(NumericalFailure) as caught:step()
            for key in ('body_vx','executed_c','local_x','step','qpos','qvel'):self.assertIn(key,caught.exception.diagnostics)
            self.assertFalse(e.speed_state.bound.any())

    def test_partial_reset_preserves_other_environment_window(self):
        e,step=task_fixture(batch=2);e.robot.data.root_link_pos_w[:,0]=.1
        for _ in range(26):step()
        before=e.speed_state.snapshot();e.speed_state.reset(torch.tensor([0]))
        self.assertEqual(int(e.speed_state.sample_count[0]),0);self.assertEqual(int(e.speed_state.sample_count[1]),26)
        self.assertEqual(int(before['sample_count'][0]),26)


if __name__=='__main__':unittest.main()
