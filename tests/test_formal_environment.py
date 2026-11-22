"""SDS Formal 1.2 CPU counterexamples; no simulated learning claim."""
import unittest
from types import SimpleNamespace as S
import torch
import numpy as np
from src.tasks.velocity.v02_formal.success import ClimbState, ParkingDiagnosticState, allocate_stage_slots
from src.tasks.velocity.v02_formal.spec import STAGES, contract
from src.tasks.velocity.v02_formal.env import FormalEnv, timeout_steps


def fixture(parking=False):
    d=S(root_link_pos_w=torch.tensor([[1.,0.,.5]]),root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]),
        site_pos_w=torch.tensor([[[1.1,.1,.112],[1.1,-.1,.112],[.6,.1,.112],[.6,-.1,.112]]]),
        root_link_lin_vel_b=torch.tensor([[.2,0.,0.]]),joint_pos=torch.zeros(1,12),joint_vel=torch.zeros(1,12))
    # geom0=last tread (same top); geom1=actual landing; geom2=foreign landing.
    tops=torch.zeros(1,4,3);tops[:,:,1]=6.
    e=S(robot=S(data=d),foot_ids=[0,1,2,3],origins=torch.zeros(10,3),tile_ids=torch.tensor([4]),
        command=torch.tensor([[.4,0.,0.]]),geom_top=torch.tensor([.08,.08,.08]),geom_step=torch.tensor([1,1,1]),geom_tile=torch.tensor([4,4,5]),
        bounds=torch.tensor([[0.,.2,-.4,.4],[.2,2.2,-.4,.4],[.2,2.2,-.4,.4]]),
        contacts=S(top_forces=tops,link_force=torch.zeros(1,1,3),body_link_mask=torch.tensor([True]),nonfoot_duration=torch.zeros(1,1)),
        control_body_hit=torch.tensor([False]),control_long_hit=torch.tensor([False]),climb=ClimbState(1),
        parking_diagnostics=ParkingDiagnosticState(1),parking_diagnostic=parking,stop_steps=torch.zeros(1,dtype=torch.long),
        episode_length_buf=torch.tensor([0]),start_vx=torch.tensor([.4]),num_envs=1,device='cpu')
    e._parameters=lambda:(torch.tensor([1.]),torch.tensor([.08]))
    def step():
        e.episode_length_buf+=1
        return FormalEnv._task_state(e)
    return e,step


class FormalEnvironmentTests(unittest.TestCase):
    def test_version_and_budget(self):
        c=contract();self.assertEqual(c['checkpoint_schema'],3);self.assertEqual(c['protocol_id'],'v02-formal-climb-v1')
        self.assertEqual(sum(s.budget for s in STAGES),26000)

    def test_four_explicit_classes_512_and_258(self):
        for n,expected_r0,expected_s in [(512,[205,51,0,256],[128,26,102,256]),(258,[103,26,0,129],[64,13,52,129])]:
            for stage,expected in [('R0',expected_r0),('S1',expected_s)]:
                tiles,classes,counts=allocate_stage_slots(n,stage)
                np.testing.assert_array_equal(counts,expected)
                self.assertEqual(len(tiles),n)
                self.assertTrue((tiles[classes==1]==0).all())
                self.assertTrue((tiles[classes==0]==0).all())
                if stage=='S1':self.assertTrue((tiles[classes==2]==4).all())
        # Equal remainders follow class-table order (walk before stand).
        self.assertEqual(allocate_stage_slots(2,'S1')[2].tolist(),[1,0,0,1])

    def test_air_and_front_only_and_unsafe_projection_fail(self):
        e,step=fixture();e.contacts.top_forces.zero_()
        for _ in range(10):self.assertFalse(step()[5])
        e.contacts.top_forces[:,:2,1]=6
        for _ in range(10):self.assertFalse(step()[5])
        e.contacts.top_forces[:,:,1]=6;e.robot.data.site_pos_w[0,2,0]=.23
        for _ in range(10):self.assertFalse(step()[5])
        self.assertEqual(float(e.command[0,0]),float(torch.tensor(.4)));self.assertFalse(e.stop.any())

    def test_final_tread_and_foreign_tile_are_not_landing(self):
        e,step=fixture();e.contacts.top_forces.zero_();e.contacts.top_forces[:,:,0]=100
        for _ in range(8):self.assertFalse(step()[5])
        e.contacts.top_forces.zero_();e.contacts.top_forces[:,:,2]=100
        for _ in range(8):self.assertFalse(step()[5])
        self.assertFalse(e.climb.current_support.any())

    def test_dynamic_gait_recent_support_and_five_current_cycles(self):
        e,step=fixture()
        # Each foot acquired genuine two-sample support, then only two remain.
        self.assertFalse(step()[5]);self.assertFalse(step()[5])
        e.contacts.top_forces[:,2:,1]=0
        for _ in range(3):self.assertFalse(step()[5])
        self.assertTrue(step()[5]);self.assertTrue(e.climb.climbed[0])
        self.assertFalse(step()[5]) # event exactly once even if not auto reset here
        self.assertFalse(e.stop.any())

    def test_history_expires_and_reset_forgets_all(self):
        e,step=fixture();step();step()
        e.robot.data.root_link_pos_w[0,0]=.9;e.contacts.top_forces[:,2:,1]=0
        for _ in range(51):self.assertFalse(step()[5])
        e.robot.data.root_link_pos_w[0,0]=1.
        for _ in range(8):self.assertFalse(step()[5])
        self.assertFalse(((e.episode_length_buf[:,None]-e.climb.last_support_step)<=50)[0,2:].any())
        e.climb.reset([0]);self.assertTrue((e.climb.last_support_step<0).all());self.assertFalse(e.climb.climbed.any())

    def test_same_surface_or_tile_switch_restarts_two_samples(self):
        c=ClimbState(1);step=torch.tensor([1]);stairs=torch.tensor([True]);tile=torch.tensor([4]);surface=torch.ones(1,4,dtype=torch.long)
        safe=torch.ones(1,4,dtype=torch.bool);bad=torch.tensor([False])
        def run():return c.update(step,stairs,tile,surface,safe,safe,torch.tensor([1.]),torch.tensor([.2]),torch.tensor([0.]),bad)
        self.assertFalse(run().current_support.any())
        surface[:]=2;step+=1;self.assertFalse(run().current_support.any())
        step+=1;self.assertTrue(run().current_support.all())
        tile[:]=5;step+=1;self.assertFalse(run().current_support.any())

    def test_failure_has_priority_on_fifth_cycle(self):
        e,step=fixture()
        for _ in range(5):self.assertFalse(step()[5])
        e.control_body_hit[:]=True
        state=step();self.assertTrue(state[6]);self.assertFalse(state[5]);self.assertFalse(e.climb.climbed.any())

    def test_parking_retains_climb_no_x_line_and_postclimb_failure(self):
        e,step=fixture(parking=True)
        for _ in range(6):self.assertFalse(step()[5])
        self.assertTrue(e.climb.climbed[0]);self.assertTrue(e.command.eq(0).all())
        e.robot.data.root_link_pos_w[0,0]=.9;e.robot.data.root_link_lin_vel_b.zero_()
        for _ in range(149):self.assertFalse(step()[5])
        self.assertTrue(step()[5]);self.assertTrue(e.climb.climbed[0]);self.assertEqual(int(e.climb.climb_step[0]),6)
        e.control_body_hit[:]=True
        self.assertTrue(step()[6]);self.assertTrue(e.climb.climbed[0])

    def test_parking_ten_seconds_and_timeout_bootstrap(self):
        e,step=fixture(parking=True)
        for _ in range(6):step()
        for _ in range(499):self.assertFalse(step()[7])
        state=step();self.assertTrue(state[7]);self.assertFalse(state[5]);self.assertFalse(state[6]);self.assertTrue(e.climb.climbed[0])
        e,step=fixture();e.contacts.top_forces.zero_();e.episode_length_buf[:]=timeout_steps(torch.tensor([1.]),torch.tensor([.08]),e.start_vx)
        state=step();self.assertTrue(state[7]);self.assertFalse(state[6]);self.assertFalse(state[5])

    def test_flat_timeout_no_climb_stop_or_events(self):
        e,step=fixture();e.tile_ids[:]=0;e._parameters=lambda:(torch.tensor([0.]),torch.tensor([0.]))
        e.episode_length_buf[:]=1499
        state=step();self.assertTrue(state[7]);self.assertFalse(state[5]);self.assertFalse(e.climb_event.any());self.assertFalse(e.stop.any())

    def test_stage_transition_resets_but_iteration_does_not(self):
        calls=[]
        e=S(stage='R0',previous_stage=None,stage_iteration=1,evaluation=False,num_envs=258,device='cpu',
            tile_ids=torch.zeros(258,dtype=torch.long),slot_classes=torch.zeros(258,dtype=torch.long),reset=lambda:calls.append('reset'))
        FormalEnv.set_stage(e,'R0',123);self.assertEqual(calls,[])
        FormalEnv.set_stage(e,'S1',0,previous_stage='R0');self.assertEqual(calls,['reset'])
        self.assertEqual(set(e.tile_ids.tolist()),{0,4,5});self.assertEqual(set(e.slot_classes.tolist()),{0,1,2,3})

    def test_unchanged_C_reward_core_matches_legacy_cpu(self):
        from src.tasks.velocity.v02.env import V02Env
        from src.tasks.velocity.v02.rewards import FirstFootfallState, ProgressState
        def make():
            d=S(root_link_lin_vel_b=torch.tensor([[.3,.01,0.]]),root_link_ang_vel_b=torch.tensor([[.01,.02,.03]]),
                root_link_ang_vel_w=torch.tensor([[.01,.02,.03]]),joint_pos=torch.zeros(1,12),default_joint_pos=torch.zeros(1,12),
                joint_acc=torch.ones(1,12),soft_joint_pos_limits=torch.tensor([[[-1.,1.]]*12]),
                site_lin_vel_w=torch.ones(1,4,3)*.01,site_pos_w=torch.tensor([[[.1,.1,.112],[.1,-.1,.112],[.1,.1,.112],[.1,-.1,.112]]]))
            e=S(robot=S(data=d),joint_ids=list(range(12)),foot_ids=list(range(4)),command=torch.tensor([[.4,0.,0.]]),
                contacts=S(found=torch.ones(1,4,dtype=torch.bool),support=torch.ones(1,4,dtype=torch.bool),support_geom=torch.zeros(1,4,dtype=torch.long),
                    foot_force=torch.tensor([[[0.,0.,6.]]*4]),contact_duration=torch.ones(1,4)*.04,link_force=torch.zeros(1,1,3),top_forces=torch.ones(1,4,1)*6),
                bounds=torch.tensor([[0.,.2,-.4,.4]]),tile_ids=torch.tensor([4]),geom_top=torch.tensor([.08]),geom_tile=torch.tensor([4]),geom_step=torch.tensor([1]),
                geom_step_bounds=torch.tensor([[0.,2.2,-.4,.4]]),origins=torch.zeros(10,3),action_manager=S(action=torch.zeros(1,12)),
                scene={'robot/root_angmom':S(data=torch.zeros(1,3))},recipe_options={'new_rewards':True},footfall=FirstFootfallState(1,40),progress=ProgressState(1),
                stop=torch.zeros(1,dtype=torch.bool),step_dt=.02,episode_reward_sums={},slip_sum=torch.zeros(1),nonfoot_sum=torch.zeros(1),limit_sum=torch.zeros(1),climb_event=torch.zeros(1,dtype=torch.bool))
            e.progress.reset([0],torch.tensor([-.1]));e._parameters=lambda:(torch.tensor([1.]),torch.tensor([.08]))
            return e
        old,new=make(),make()
        for frame in range(8):
            success=torch.tensor([frame==6]);failure=torch.tensor([frame==7]);new.climb_event=success
            state=(torch.tensor([[-.05+frame*.005,0.,.5]]),torch.tensor([.01]),torch.tensor([.02]),torch.tensor([.03]),torch.tensor([False]),success,failure,None,None)
            a=V02Env._reward(old,state,torch.ones(1,12)*.01)
            b=FormalEnv._reward(new,state,torch.ones(1,12)*.01)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(set(old.step_reward_tensors),set(new.step_reward_tensors))
            for key in old.step_reward_tensors:torch.testing.assert_close(old.step_reward_tensors[key],new.step_reward_tensors[key],rtol=0,atol=0)
        self.assertEqual(float(new.episode_reward_sums['reward_event/footfall']),1.)
        self.assertEqual(float(new.episode_reward_sums['reward_event/success']),10.)
        self.assertEqual(float(new.episode_reward_sums['reward_event/failure']),-4.)

    def test_monitor_arrival_frame_and_next_parking_sample_share_boundary(self):
        e,step=fixture(parking=True)
        e.slot_classes=torch.tensor([3]);e.joint_ids=list(range(12))
        e.sim=S(data=S(qfrc_actuator=torch.ones(1,12)))
        e.robot.indexing=S(joint_v_adr=list(range(12)))
        d=e.robot.data;d.root_link_lin_vel_w=torch.tensor([[.2,0.,0.]])
        d.site_lin_vel_w=torch.zeros(1,4,3)
        e.contacts.found=torch.ones(1,4,dtype=torch.bool);e.contacts.foot_force=torch.zeros(1,4,3)
        e._motion_parking_mask=lambda:FormalEnv._motion_parking_mask(e)
        for _ in range(6):state=step()
        e.last_termination=(state[5],state[6],state[7],state[8])
        self.assertTrue(e.stop[0]);self.assertEqual(float(e.command[0,0]),0.)
        arrival=FormalEnv.motion_terms(e)
        self.assertFalse(e._motion_parking_mask()[0]) # episode moving_count uses this same mask
        self.assertEqual(arrival['groups']['current_stairs/moving']['sample_count'],1)
        self.assertNotIn('current_stairs/parking',arrival['groups'])
        self.assertAlmostEqual(arrival['command_vx'],.4,places=6)
        self.assertEqual(arrival['command_vx_post_decision'],0.)
        self.assertEqual(arrival['parking'],0.)
        state=step();e.last_termination=(state[5],state[6],state[7],state[8])
        after=FormalEnv.motion_terms(e)
        self.assertTrue(e._motion_parking_mask()[0])
        self.assertEqual(after['groups']['current_stairs/parking']['sample_count'],1)
        self.assertNotIn('current_stairs/moving',after['groups'])
        self.assertEqual(after['command_vx'],0.);self.assertEqual(after['parking'],1.)

    def test_motion_maximums_are_true_across_envs_and_group_samples(self):
        count=8;angles=torch.arange(count,dtype=torch.float32)*.1
        q=torch.zeros(count,4);q[:,0]=torch.cos(angles/2);q[:,3]=torch.sin(angles/2)
        pos=torch.zeros(count,3);pos[:,1]=torch.arange(count,dtype=torch.float32)*.1
        torque=torch.zeros(count,12);torque[:,0]=torch.arange(1,count+1,dtype=torch.float32)
        d=S(root_link_pos_w=pos,root_link_quat_w=q,root_link_lin_vel_b=torch.zeros(count,3),root_link_lin_vel_w=torch.zeros(count,3),site_lin_vel_w=torch.zeros(count,4,3))
        e=S(robot=S(data=d,indexing=S(joint_v_adr=list(range(12)))),sim=S(data=S(qfrc_actuator=torque)),
            origins=torch.zeros(10,3),tile_ids=torch.zeros(count,dtype=torch.long),joint_ids=list(range(12)),foot_ids=list(range(4)),
            command=torch.zeros(count,3),sample_command=torch.zeros(count,3),num_envs=count,slot_classes=torch.tensor([0,0,1,1,2,2,3,3]),
            climb=S(climbed=torch.zeros(count,dtype=torch.bool),climb_step=torch.full((count,),-1)),episode_length_buf=torch.ones(count,dtype=torch.long),parking_diagnostic=False,
            contacts=S(link_force=torch.zeros(count,1,3),found=torch.zeros(count,4,dtype=torch.bool),foot_force=torch.zeros(count,4,3)),
            last_termination=(None,None,None,torch.zeros(count,dtype=torch.long)))
        e._motion_parking_mask=lambda:FormalEnv._motion_parking_mask(e)
        result=FormalEnv.motion_terms(e)
        self.assertEqual(result['sample_count'],8)
        self.assertEqual(result['peak_torque'],4.5);self.assertEqual(result['peak_torque_max'],8.)
        self.assertAlmostEqual(result['lateral_max'],.7,places=6);self.assertAlmostEqual(result['heading_max'],.7,places=6)
        for class_id,name in enumerate(('flat_walk','flat_stand','previous_stairs','current_stairs')):
            group=result['groups'][name+'/moving'];self.assertEqual(group['sample_count'],2)
            self.assertEqual(group['peak_torque'],2*class_id+1.5)
            self.assertEqual(group['peak_torque_max'],2*class_id+2.)
            self.assertAlmostEqual(group['lateral_max'],(2*class_id+1)*.1,places=6)
            self.assertAlmostEqual(group['heading_max'],(2*class_id+1)*.1,places=6)

    def test_local_generator_does_not_consume_global_rng(self):
        before=torch.random.get_rng_state().clone();g=torch.Generator().manual_seed(42)
        state=g.get_state().clone();expected=torch.rand(10,generator=g)
        g.set_state(state);actual=torch.rand(10,generator=g)
        torch.testing.assert_close(actual,expected);self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

if __name__=='__main__':unittest.main()
