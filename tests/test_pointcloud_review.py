"""Independent SDS regression tests; synthetic data only, no training."""
import unittest
import math
import torch
from src.tasks.pointcloud.geometry import fit_normal
from src.tasks.pointcloud.rewards import SwingTargets
from src.tasks.pointcloud.evaluation import EpisodeSpeed
from src.tasks.pointcloud.terminations import EpisodeEnds


class IndependentReviewTests(unittest.TestCase):
    def test_normal_valid_at_actual_curriculum_translation(self):
        x,y=torch.meshgrid(torch.linspace(-.6,.6,13),torch.linspace(-.3,.3,7),indexing='ij')
        points=torch.stack((x.flatten(),y.flatten(),.2*x.flatten()),-1)[None]
        valid=torch.ones(1,91,dtype=torch.bool)
        reference,ok=fit_normal(points,valid)
        self.assertTrue(ok.all())
        for offset in (40.,160.,280.):
            normal,ok=fit_normal(points+torch.tensor([0.,offset,0.]),valid)
            self.assertTrue(ok.all(),f'Full rank local grid rejected at terrain y={offset}')
            torch.testing.assert_close(normal,reference,atol=1e-5,rtol=1e-5)

    def test_swing_reference_does_not_jump_on_early_touch(self):
        swing=SwingTargets(1)
        yes=torch.ones(1,4,dtype=torch.bool); no=~yes
        zero=torch.zeros(1,4); target=torch.full((1,4),.2)
        swing.update(yes,zero,yes,no,target,yes)
        h0,h1,valid=swing.update(no,zero,yes,yes,target,yes)
        frozen=h0.clone()
        h0,h1,valid=swing.update(yes,target,yes,yes,target,yes)
        torch.testing.assert_close(h0,frozen)

    def test_inherited_tilt_failure_uses_body_axis_not_euler_max(self):
        ends=EpisodeEnds(1)
        body=torch.tensor([[0.,0.,.4]])
        feet=torch.zeros(1,4,3)
        angles=torch.tensor([[math.pi/3,math.pi/3]])
        failure,_,_=ends.update(body,feet,angles,torch.zeros(1,4),torch.tensor([False]),torch.tensor([True]),torch.tensor([1.]))
        self.assertTrue(failure.item(),'60deg roll plus 60deg pitch tilts body-up by 75.52deg')

    def test_speed_interval_stops_at_first_top_entry(self):
        metric=EpisodeSpeed(.3)
        for t,x,v in ((1.,0.,.3),(2.,8.1,.3),(3.,7.9,-.2)):
            metric.update(t,x,8.,[v,0,0],0.,[x,0])
        self.assertEqual(metric.velocities,[.3])


class RunnerBoundaryReviewTests(unittest.TestCase):
    def test_p5_block_saves_completed_update_before_stage_switch(self):
        import tempfile, random
        import numpy as np
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.tasks.pointcloud.runner import PointCloudRunner
        from src.tasks.pointcloud.curriculum import Curriculum
        events=[]
        class FakeEnv:
            def set_stage(self,stage):
                events.append(('stage',stage))
                raise RuntimeError('P5 blocked')
        with tempfile.TemporaryDirectory() as root:
            runner=PointCloudRunner.__new__(PointCloudRunner)
            runner.settings=SimpleNamespace(output_dir=str(Path(root)/'run'),max_iterations=4000,save_interval=25)
            runner.config={}; runner.env=FakeEnv(); runner.model=object()
            runner.iteration=199; runner.curriculum=Curriculum(index=7,iteration=199,consecutive=1)
            runner.best_score={}
            def fake_update():
                runner.iteration+=1
                return {'iteration':runner.iteration}
            def fake_save(out):
                events.append(('save',runner.iteration))
                path=Path(out)/'checkpoint_000200.pt'; path.write_text('mock only')
                return path
            runner.update=fake_update; runner.save=fake_save
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            expected=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            def validate(*args):
                random.random(); np.random.rand(); torch.rand(1)
                return True,{.2:{'fraction':1.,'speed':{'passed':True,'mae':.01}}}
            with patch('src.tasks.pointcloud.runner.subprocess.check_output',return_value='mock'):
                with self.assertRaisesRegex(RuntimeError,'P5 blocked'):
                    runner.learn(validate)
            self.assertEqual(events,[('save',200),('stage','P5')])
            self.assertTrue((Path(runner.settings.output_dir)/'best_P4.json').exists())
            self.assertTrue((Path(runner.settings.output_dir)/'blocked.json').exists())
            self.assertEqual(random.getstate(),expected[0])
            self.assertTrue(np.array_equal(np.random.get_state()[1],expected[1][1]))
            self.assertTrue(torch.equal(torch.get_rng_state(),expected[2]))

    def test_restore_rebuilds_history_after_state_restore(self):
        import tempfile
        from pathlib import Path
        from src.tasks.pointcloud.checkpoint import save_checkpoint, restore_checkpoint
        from src.tasks.pointcloud.perception import PointCloudHistory
        model=torch.nn.Linear(2,1); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        history=PointCloudHistory(2)
        history.advance(torch.ones(2,70,3),torch.ones(2,70,dtype=torch.bool))
        expected={k:v.clone() for k,v in model.state_dict().items()}
        torch.manual_seed(321); expected_rng=torch.get_rng_state().clone()
        called=[]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.pt'
            save_checkpoint(path,model,optimizer,{'index':0},25,{},update_complete=True)
            with torch.no_grad(): model.weight.add_(1)
            torch.rand(3)
            def reset_all():
                self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
                for key,value in model.state_dict().items(): torch.testing.assert_close(value,expected[key])
                history.reset(torch.arange(2),delay_steps=torch.tensor([2,3]))
                called.append(True)
            state=restore_checkpoint(path,model,optimizer,reset_all)
        self.assertEqual(called,[True]); self.assertEqual(state['event'],'resume_reset')
        self.assertFalse(history.read()[1].any()); self.assertFalse(history.scan_group.any())
        self.assertTrue(torch.equal(history.delay,torch.tensor([2,3])))

    def test_reward_diagnostics_do_not_replace_critic_values(self):
        from types import SimpleNamespace
        from src.tasks.pointcloud.runner import PointCloudRunner
        observation={'actor':{'obs_history':torch.zeros(1,5,45),'mask':torch.ones(1,5,70,dtype=torch.bool)},'privileged':torch.tensor([[7.]])}
        class Model:
            actor=SimpleNamespace(eth=lambda obs:torch.zeros(1,12))
            def eval(self): pass
            def distribution(self,inputs): return torch.distributions.Normal(torch.zeros(1,12),torch.ones(1,12))
            def critic(self,privileged): return privileged
        class Env:
            def step(self,action):
                zero=torch.tensor([False])
                info={'terminal_observation':observation,'reward_terms':{'last_reward':torch.tensor([99.])},'contact_capacity_hits':0,'constraint_capacity_hits':0,'diagnostics':{'support_missing':.25,'normal_missing':.5,'swing_target_missing':.75}}
                return observation,torch.ones(1),zero,zero,zero,info
        runner=PointCloudRunner.__new__(PointCloudRunner)
        runner.obs=observation; runner.model=Model(); runner.env=Env(); runner.settings=SimpleNamespace(rollout=1)
        batch,metrics=runner.collect()
        torch.testing.assert_close(batch['old_value'],torch.tensor([7.]))
        self.assertEqual(metrics['support_missing'],.25)
        self.assertEqual(metrics['normal_missing'],.5)
        self.assertEqual(metrics['swing_target_missing'],.75)

class VideoCleanupReviewTests(unittest.TestCase):
    def test_writer_closed_when_environment_step_fails(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from src.tasks.pointcloud.validation import episode
        writer=Mock()
        env=SimpleNamespace(reset=lambda:{'actor':{}},step=Mock(side_effect=RuntimeError('synthetic step failure')))
        model=SimpleNamespace(eval=lambda:None,actor=lambda **inputs:torch.zeros(1,12))
        with patch('imageio.v2.get_writer',return_value=writer):
            with self.assertRaisesRegex(RuntimeError,'synthetic step failure'):
                episode(model,env,.2,False,video_path='mock.mp4')
        writer.close.assert_called_once()

@unittest.skipUnless(__import__('os').environ.get('POINTCLOUD_PHYSICS_TEST')=='1','opt-in bounded GPU physics')
class P5SaturationReviewTests(unittest.TestCase):
    def test_actual_limits_not_nominal_under_p5_strength(self):
        from src.tasks.pointcloud.environment import PointCloudEnv
        e=PointCloudEnv(num_envs=2,device='cuda:0',stage='P5')
        try:
            e.reset()
            e.strength[:]=torch.tensor([.9,1.1],device='cuda:0')
            for name in e._strength_fields:
                getattr(e.env.sim.model,name)[:]=e._nominal_actuator[name][None]*e.strength[:,None,None]
            _,_,_,_,_,info=e.step(torch.full((2,12),100.,device='cuda:0'))
            force=e.robot.data.actuator_force
            limits=e.env.sim.model.actuator_forcerange[:,e.robot.indexing.ctrl_ids]
            expected=((force<=limits[...,0]*.999)|(force>=limits[...,1]*.999)).float().mean(-1)
            torch.testing.assert_close(info['terminal_state']['torque_saturation'],expected)
            self.assertGreater(expected[0].item(),0.,'Low-strength force limit must actually be reached')
            nominal=e._nominal_actuator['actuator_forcerange'][e.robot.indexing.ctrl_ids]
            incorrect=((force<=nominal[None,...,0]*.999)|(force>=nominal[None,...,1]*.999)).float().mean(-1)
            self.assertGreater(expected[0].item(),incorrect[0].item(),'Counterexample must expose the old nominal threshold')
        finally: e.close()

if __name__=='__main__': unittest.main()
