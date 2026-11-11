"""Small bounded inference checks; explicitly opt in to GPU physics."""
import os
import unittest
import torch
from src.tasks.pointcloud.env_cfg import pointcloud_env_cfg

class ConfigTests(unittest.TestCase):
    def test_independent_reward_replacements(self):
        from src.tasks.velocity.config.a2.eth_env_cfg import unitree_a2_eth_flat_env_cfg
        old=unitree_a2_eth_flat_env_cfg(); new=pointcloud_env_cfg()
        self.assertIn('is_terminated',old.rewards)
        self.assertNotIn('is_terminated',new.rewards)
        self.assertEqual(new.decimation,4); self.assertEqual(new.sim.mujoco.timestep,.005)
        for k in new.rewards: self.assertEqual(new.rewards[k].weight,old.rewards[k].weight)

@unittest.skipUnless(os.environ.get('POINTCLOUD_PHYSICS_TEST')=='1','opt-in bounded GPU physics')
class PhysicsTests(unittest.TestCase):
    def test_two_worlds_two_steps_reset(self):
        from src.tasks.pointcloud.environment import PointCloudEnv
        e=PointCloudEnv(num_envs=2,device='cuda:0',stage='P2a',evaluation=True,command_speed=.2)
        try:
            o=e.reset()
            self.assertEqual(o['privileged'].shape,(2,63))
            self.assertTrue(torch.all(e.origin[:,0]==0))
            self.assertTrue(torch.all((e.robot.data.root_link_pos_w[:,0]+1).abs()<=.2))
            for _ in range(2):
                o,r,f,s,t,info=e.step(torch.zeros(2,12,device='cuda:0'))
                self.assertTrue(torch.isfinite(r).all())
                self.assertFalse((f&s|s&t|f&t).any())
                self.assertTrue(torch.isfinite(info['terminal_state']['foot_slip']).all())
                self.assertEqual(info['terminal_state']['failure_reasons'].shape,(2,5))
            before=e.obs_history[1].clone()
            e._reset(torch.tensor([0],device='cuda:0'))
            self.assertTrue(torch.equal(before,e.obs_history[1]))
            self.assertFalse(e.history.read()[1][0].any())
            self.assertEqual(e.history.scan_group[0].item(),0)
        finally: e.close()

    def test_p5_strength_reset_isolation_and_nominal_restore(self):
        from src.tasks.pointcloud.environment import PointCloudEnv
        e=PointCloudEnv(num_envs=2,device='cuda:0',stage='P5')
        try:
            e.reset()
            self.assertTrue(((e.strength>=.9)&(e.strength<=1.1)).all())
            untouched={name:getattr(e.env.sim.model,name)[1].clone() for name in e._strength_fields}
            for _ in range(3):
                e._reset(torch.tensor([0],device='cuda:0'))
                for name in e._strength_fields:
                    actual=getattr(e.env.sim.model,name)
                    self.assertTrue(torch.equal(actual[1],untouched[name]))
                    self.assertTrue(torch.allclose(actual[0],e._nominal_actuator[name]*e.strength[0]))
            _,reward,*_=e.step(torch.zeros(2,12,device='cuda:0'))
            self.assertTrue(torch.isfinite(reward).all())
            for stage,evaluation,degraded in [('P1',False,False),('P4',False,False),('P5',True,False),('P5',True,True)]:
                e.evaluation=evaluation; e.perception_degraded=degraded; e.set_stage(stage); e.reset()
                self.assertTrue(torch.equal(e.strength,torch.ones_like(e.strength)))
                for name in e._strength_fields:
                    self.assertTrue(torch.equal(getattr(e.env.sim.model,name),e._nominal_actuator[name][None].expand(2,-1,-1)))
        finally: e.close()
