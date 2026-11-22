"""Independent reward scalars and frozen-term CPU comparisons; no learning claim."""
import importlib
import math
import unittest
import numpy as np
import torch
from types import SimpleNamespace as S


def api():
    try:
        return importlib.import_module('src.tasks.velocity.v03.rewards')
    except ModuleNotFoundError:
        raise AssertionError('V03 reward implementation is missing') from None


def reward_fixture(cls):
    from src.tasks.velocity.v02.rewards import FirstFootfallState, ProgressState
    from src.tasks.velocity.v03.spec import recipe_config
    e = object.__new__(cls)
    b=3
    d=S(root_link_lin_vel_b=torch.tensor([[0.,.03,0.],[.4,-.02,0.],[1.55,.1,0.]]),
        root_link_ang_vel_b=torch.tensor([[.1,.2,.3]]*b), root_link_ang_vel_w=torch.tensor([[.2,.1,0.]]*b),
        joint_pos=torch.full((b,12),.03), default_joint_pos=torch.zeros(b,12), joint_acc=torch.full((b,12),.5),
        site_pos_w=torch.tensor([[[.1,.1,.112],[.1,-.1,.112],[.1,.1,.112],[.1,-.1,.112]]]*b),
        site_lin_vel_w=torch.full((b,4,3),.02),soft_joint_pos_limits=torch.tensor([[[-1.,1.]]*12]*b))
    e.robot=S(data=d);e.scene={'robot/root_angmom':S(data=torch.full((b,3),.1))}
    e.joint_ids=list(range(12));e.foot_ids=list(range(4));e.command=torch.tensor([[.2,0.,0.],[.4,0.,0.],[.6,0.,0.]])
    e.sample_command=e.command.clone();e.action_manager=S(action=torch.full((b,12),.02))
    e.contacts=S(found=torch.ones(b,4,dtype=torch.bool),support=torch.ones(b,4,dtype=torch.bool),
        support_geom=torch.zeros(b,4,dtype=torch.long),foot_force=torch.full((b,4,3),6.),
        contact_duration=torch.full((b,4),.02),link_force=torch.zeros(b,1,3),top_forces=torch.full((b,4,1),6.))
    e.bounds=torch.tensor([[0.,.2,-.4,.4]]);e.geom_top=torch.tensor([.08]);e.geom_tile=torch.tensor([5]);e.geom_step=torch.tensor([1]);e.geom_step_bounds=e.bounds.clone()
    e.tile_ids=torch.full((b,),5,dtype=torch.long);e.origins=torch.zeros(10,3)
    e.footfall=FirstFootfallState(b,40);e.progress=ProgressState(b);e.progress.reset(torch.arange(b),torch.zeros(b))
    e.stop=torch.zeros(b,dtype=torch.bool);e.climb_event=torch.tensor([True,False,False]);e.recipe_options=recipe_config()
    e.cfg=S(sim=S(mujoco=S(timestep=.005)),decimation=4);e.episode_reward_sums={};e.slip_sum=torch.zeros(b);e.nonfoot_sum=torch.zeros(b);e.limit_sum=torch.zeros(b)
    e._parameters=lambda:(torch.full((b,),4.),torch.full((b,),.08))
    state=(torch.tensor([[.1,.01,.5]]*b),torch.full((b,),.02),torch.full((b,),.03),torch.full((b,),.04),
           torch.ones(b,dtype=torch.bool),e.climb_event.clone(),torch.tensor([False,False,True]),torch.zeros(b,dtype=torch.bool),torch.zeros(b,dtype=torch.long))
    return e,state,torch.zeros(b,12)

def controlled_trajectories():
    """Hand-controlled 1D samples; other=-.01/control, no physical-policy claim."""
    result={};c=float(np.float32(.4));dt=.02;distance=1.6
    policies={
        'accurate_complete':lambda k:c,
        'continue_overspeed_complete':lambda k:1.55,
        'delay_complete':lambda k:0. if k<100 else c,
        'alternating_stop_go_complete':lambda k:.8 if (k//25)%2==0 else 0.,
        'early_real_failure':lambda k:c,
    }
    for name,policy in policies.items():
        velocities=[];positions=[];x=0.;target=.4 if name=='early_real_failure' else distance
        for k in range(10000):
            v=min(policy(k),max(0.,(target-x)/dt));x+=v*dt
            velocities.append(v);positions.append(x)
            if x>=target-1e-12:break
        vx=np.asarray(velocities,dtype=np.float32);cmd=np.full(len(vx),c,dtype=np.float32)
        speed=api().speed_reward(torch.from_numpy(vx),torch.zeros(len(vx)),torch.from_numpy(cmd)).numpy()
        event=np.zeros(len(vx));event[-1]=-4. if name=='early_real_failure' else 1.
        other=np.full(len(vx),-.01);discount=np.power(.99,np.arange(len(vx),dtype=np.float64))
        oldspeed=.02*(np.exp(-(vx.astype(np.float64)-cmd.astype(np.float64))**2/.25)-np.exp(-cmd.astype(np.float64)**2/.25))
        oldevent=event.copy()
        if name!='early_real_failure':oldevent[-1]=10.
        result[name]=dict(body_vx=vx,executed_c=cmd,local_x=np.asarray(positions,dtype=np.float32),
            step=np.arange(1,len(vx)+1,dtype=np.int64),speed_reward=speed,event_reward=event,other_fixed=other,
            old_speed_reward=oldspeed,old_event_reward=oldevent,discount=discount)
    return result


class RewardTests(unittest.TestCase):
    def test_controlled_discounted_trajectories_match_independent_scalar_sums(self):
        trajectories=controlled_trajectories()
        for name,raw in trajectories.items():
            c=raw['executed_c'].astype(np.float64);v=raw['body_vx'].astype(np.float64);ex=(v-c)/c
            expected=.02*(2*np.exp(-4*ex**2)-2*ex**2)
            np.testing.assert_allclose(raw['speed_reward'],expected,atol=3e-7,rtol=2e-7)
            self.assertEqual(raw['event_reward'][-1],-4. if name=='early_real_failure' else 1.)
            self.assertEqual(np.count_nonzero(raw['event_reward']),1)
            self.assertAlmostEqual(float(np.sum(v)*.02),.4 if name=='early_real_failure' else 1.6,places=6)
            discounted=sum((raw[k]*raw['discount']).sum() for k in ('speed_reward','event_reward','other_fixed'))
            independent=sum(.99**j*(expected[j]+raw['event_reward'][j]-.01) for j in range(len(v)))
            self.assertAlmostEqual(discounted,independent,delta=2e-5)
        self.assertEqual(len(trajectories['accurate_complete']['step']),200)
        self.assertEqual(len(trajectories['delay_complete']['step']),300)
        self.assertEqual(len(trajectories['continue_overspeed_complete']['step']),52)

    def test_independent_scalar_oracle_all_commands_and_lateral(self):
        r=api()
        for c in (.2,.4,.6):
            for vx,vy in ((c,0.),(0.,0.),(1.55,.1),(-.2,-.3)):
                actual=float(r.speed_reward(torch.tensor([vx]),torch.tensor([vy]),torch.tensor([c]),dt=.02))
                cf=float(np.float32(c));vf=float(np.float32(vx));yf=float(np.float32(vy))
                ex=(vf-cf)/cf;ey=yf/.2
                expected=.02*(2*math.exp(-4*ex*ex-ey*ey)-2*ex*ex-ey*ey)
                self.assertAlmostEqual(actual,expected,delta=max(2e-6,abs(expected)*2e-7))
        self.assertAlmostEqual(float(r.speed_reward(torch.tensor([0.]),torch.tensor([0.]),torch.tensor([.4])))/.02,-1.9633687222,places=6)
        self.assertAlmostEqual(float(r.speed_reward(torch.tensor([.4]),torch.tensor([0.]),torch.tensor([.4])))/.02,2.,places=6)

    def test_parking_skip_and_failure_event_priority(self):
        r=api()
        self.assertEqual(r.speed_reward(torch.tensor([.1]),torch.tensor([0.]),torch.tensor([0.]),enabled=torch.tensor([False])).item(),0.)
        with self.assertRaises(ValueError): r.speed_reward(torch.tensor([0.]),torch.tensor([0.]),torch.tensor([0.]))
        self.assertEqual(r.event_reward(torch.tensor([True,True,False]),torch.tensor([False,True,True])).tolist(),[1.,-4.,-4.])

    def test_other_terms_exactly_equal_frozen_reward_and_sum_no_old_linear(self):
        from src.tasks.velocity.v02_formal.env import FormalEnv
        try: from src.tasks.velocity.v03.env import V03Env
        except ModuleNotFoundError: self.fail('V03 environment implementation is missing')
        old,state,previous=reward_fixture(FormalEnv);new,state2,previous2=reward_fixture(V03Env)
        FormalEnv._reward(old,state,previous);V03Env._reward(new,state2,previous2)
        for key,value in old.step_reward_tensors.items():
            if key in ('reward_total','reward_event/success') or key.endswith('/track_linear_velocity'): continue
            torch.testing.assert_close(new.step_reward_tensors[key],value,rtol=0,atol=0)
        self.assertNotIn('reward_baseline/track_linear_velocity',new.step_reward_tensors)
        self.assertEqual(new.step_reward_tensors['reward_event/success'].tolist(),[1.,0.,0.])
        weighted=[v for k,v in new.step_reward_tensors.items() if k.startswith('reward_weighted/')]
        want=sum(weighted)+new.step_reward_tensors['reward_event/success']+new.step_reward_tensors['reward_event/failure']+new.step_reward_tensors['reward_event/footfall']+new.step_reward_tensors['reward_progress']
        torch.testing.assert_close(new.step_reward_tensors['reward_total'],want,rtol=0,atol=0)
        for key,value in new.episode_reward_sums.items(): torch.testing.assert_close(value,new.step_reward_tensors[key],rtol=0,atol=0)
        self.assertFalse(any(k.startswith('diagnostic/') for k in new.episode_reward_sums))
        # A second actual contact rewards each foot once; subsequent calls cannot reclaim it.
        new.climb_event.zero_();V03Env._reward(new,state2,previous2)
        self.assertEqual(new.step_reward_tensors['reward_event/footfall'].tolist(),[1.,1.,0.])
        V03Env._reward(new,state2,previous2)
        self.assertEqual(new.step_reward_tensors['reward_event/footfall'].tolist(),[0.,0.,0.])

    def test_reward_uses_executed_command_when_postarrival_command_is_zero(self):
        try: from src.tasks.velocity.v03.env import V03Env
        except ModuleNotFoundError: self.fail('V03 environment implementation is missing')
        e,state,previous=reward_fixture(V03Env);e.command.zero_()
        V03Env._reward(e,state,previous)
        want=api().speed_reward(e.robot.data.root_link_lin_vel_b[:,0],e.robot.data.root_link_lin_vel_b[:,1],e.sample_command[:,0])
        torch.testing.assert_close(e.step_reward_tensors['reward_weighted/track_linear_velocity'],want,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
