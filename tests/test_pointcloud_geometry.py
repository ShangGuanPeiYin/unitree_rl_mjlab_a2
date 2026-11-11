import unittest
import torch
from src.tasks.pointcloud.geometry import query_top, stair_surfaces, transform, fit_normal
from src.tasks.pointcloud.rewards import alignment, SwingTargets
from src.tasks.pointcloud.terminations import EpisodeEnds
from src.tasks.pointcloud.curriculum import Curriculum
from src.tasks.pointcloud.evaluation import speed_gate, FirstTermination

class GeometryTests(unittest.TestCase):
    def test_local_support(self):
        p = torch.tensor([[[.1,0,.25],[4.1,0,4.25],[4.1,.5,4.25],[.1,0,.195],[-1.,0,.2],[8.5,0,8.1]]])
        z, valid, _ = query_top(p, stair_surfaces())
        self.assertEqual(valid.tolist(), [[True,True,False,True,True,True]])
        self.assertAlmostEqual(z[0,1].item(),4.2,places=5)
        _, below, _ = query_top(torch.tensor([[[4.1,0,4.18]]]), stair_surfaces())
        self.assertFalse(below.item())

    def test_transform_and_alignment(self):
        p = torch.rand(2,4,3)
        origin = torch.tensor([[10.,-4.,2.],[-3.,8.,1.]])
        yaw = torch.tensor([.7,-1.2])
        self.assertTrue(torch.allclose(transform(transform(p,origin,yaw),origin,yaw,True),p,atol=1e-6))
        xx, yy = torch.meshgrid(torch.linspace(-.6,.6,13),torch.linspace(-.3,.3,7),indexing='ij')
        pts = torch.stack((xx.flatten(),yy.flatten(),xx.flatten()*.5),-1)[None]
        n, good = fit_normal(pts,torch.ones(1,91,dtype=torch.bool))
        self.assertTrue(good.item())
        self.assertGreater(alignment(n,n,good).item(),alignment(torch.tensor([[0.,0.,1.]]),n,good).item())

    def test_top_swing_and_flat(self):
        e = EpisodeEnds(1)
        b = torch.tensor([[8.5,0.,8.4]])
        f = torch.tensor([[[8.4,.1,8.],[8.6,-.1,8.2],[8.3,-.1,8.],[8.7,.1,8.2]]])
        args = (b,f,torch.zeros(1,2),torch.tensor([[6.,0,6.,0]]),torch.tensor([False]),torch.tensor([False]),torch.tensor([90.]))
        for _ in range(50): fail,success,timeout = e.update(*args)
        self.assertTrue(success.item()); self.assertFalse(fail.item()); self.assertFalse(timeout.item())
        e.reset([0]); b[:] = torch.tensor([[20.,4.,.4]])
        fail,success,timeout = e.update(b,f,torch.zeros(1,2),torch.zeros(1,4),torch.tensor([False]),torch.tensor([True]),torch.tensor([30.]))
        self.assertFalse(fail.item()); self.assertFalse(success.item()); self.assertTrue(timeout.item())

    def test_frozen_swing(self):
        s = SwingTargets(1)
        t = torch.ones(1,4,dtype=torch.bool); z = torch.zeros(1,4)
        s.update(t,z,t,~t,z,t)
        s.update(~t,z,t,t,z+.2,t)
        _,h1,_ = s.update(~t,z,t,t,z+.4,t)
        self.assertTrue(torch.allclose(h1,z+.2))
        s.reset([0]); self.assertFalse(s.known.any())

    def test_budget_and_actual_speed(self):
        c = Curriculum(index=1)
        for _ in range(100): c.advance_iteration()
        c.validate(False)
        for _ in range(100): c.advance_iteration()
        c.validate(True)
        self.assertTrue(c.stopped)
        self.assertFalse(speed_gate([[0.,0.]],.6)['passed'])
        self.assertFalse(speed_gate([[]],.6)['passed'])
        r=FirstTermination(); r.capture(True,False,False,{'x':1}); r.capture(False,True,False,{'x':8})
        self.assertEqual(r.record['state']['x'],1)

if __name__ == '__main__': unittest.main()

class AdditionalFailureTests(unittest.TestCase):
    def test_failure_precedence_and_async_reset(self):
        e=EpisodeEnds(2)
        b=torch.tensor([[8.5,0.,8.4],[4.,.6,4.5]])
        f=torch.tensor([[[8.5,0.,8.]]*4,[[4.,0.,4.]]*4])
        for _ in range(50):
            fail,success,timeout=e.update(b,f,torch.zeros(2,2),torch.ones(2,4)*6,torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool),torch.ones(2)*90)
        self.assertEqual(fail.tolist(),[True,True]); self.assertFalse(success.any()); self.assertFalse(timeout.any())
        e.reset([0]); self.assertEqual(e.side.tolist(),[0,50])

    def test_surface_per_environment_and_normal_rank(self):
        surfaces=stair_surfaces().repeat(2,1,1)
        surfaces[1,:,4] += 10
        _,valid,_=query_top(torch.tensor([[[.1,0.,.3]],[[.1,0.,.3]]]),surfaces)
        self.assertEqual(valid.tolist(),[[True],[False]])
        _,good=fit_normal(torch.zeros(1,6,3),torch.ones(1,6,dtype=torch.bool))
        self.assertFalse(good.item())

    def test_first_failure_speed_not_spliced(self):
        from src.tasks.pointcloud.evaluation import EpisodeSpeed,stage_gate
        ep=EpisodeSpeed(.6)
        ep.update(1,-.1,8,[.6,0,0],0,[0,0],failure=True,state={'x':-.1})
        ep.update(2,1,8,[.6,0,0],0,[1,0],success=True)
        passed,_=stage_gate({.6:[ep.result()]})
        self.assertFalse(passed); self.assertEqual(ep.velocities,[])

    def test_drop_persistence_and_mixed_geometry(self):
        e=EpisodeEnds(2)
        b=torch.tensor([[.1,0.,.34],[.1,0.,.5]])
        for i in range(5):
            fail,_,_=e.update(b,torch.zeros(2,4,3),torch.zeros(2,2),torch.zeros(2,4),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),torch.zeros(2),stair_end=torch.tensor([.8,8.]),stair_height=torch.tensor([.4,8.]),step_height=torch.tensor([.2,.2]))
            self.assertEqual(fail.tolist(),[i==4,False])

class RegressionTests(unittest.TestCase):
    def test_touch_during_swing_does_not_change_reference(self):
        s=SwingTargets(1); yes=torch.ones(1,4,dtype=torch.bool); z=torch.zeros(1,4)
        s.update(yes,z,yes,~yes,z,yes)
        s.update(~yes,z,yes,yes,z+.2,yes)
        h0,h1,_=s.update(yes,z+.2,yes,yes,z+.4,yes)
        self.assertTrue(torch.equal(h0,z)); self.assertTrue(torch.allclose(h1,z+.2))
        self.assertTrue(torch.allclose(s.last_support,z+.2))

    def test_top_entry_latches_speed_interval(self):
        from src.tasks.pointcloud.evaluation import EpisodeSpeed
        e=EpisodeSpeed(.3)
        e.update(1,1,8,[.3,0,0],0,[1,0])
        e.update(2,8.1,8,[.3,0,0],0,[8.1,0])
        e.update(3,7.9,8,[-.3,0,0],0,[7.9,0],failure=True)
        self.assertEqual(e.velocities,[.3])

class TranslationRegression(unittest.TestCase):
    def test_plane_normal_large_world_offset(self):
        xx,yy=torch.meshgrid(torch.linspace(-.6,.6,13),torch.linspace(-.3,.3,7),indexing='ij')
        p=torch.stack((xx.flatten(),yy.flatten(),xx.flatten()*.5),-1)[None]
        valid=torch.ones(1,91,dtype=torch.bool)
        expected,_=fit_normal(p,valid)
        for offset in (40.,160.,280.):
            n,good=fit_normal(p+torch.tensor([0.,offset,8.]),valid)
            self.assertTrue(good.item()); self.assertTrue(torch.allclose(n,expected,atol=1e-5))

class TiltRegression(unittest.TestCase):
    def test_combined_roll_pitch(self):
        e=EpisodeEnds(1)
        fail,_,_=e.update(torch.tensor([[0.,0.,.4]]),torch.zeros(1,4,3),torch.tensor([[1.04719755,1.04719755]]),torch.zeros(1,4),torch.tensor([False]),torch.tensor([True]),torch.zeros(1))
        self.assertTrue(fail.item())

class CurriculumValidationRegression(unittest.TestCase):
    def test_cannot_duplicate_or_skip_validation(self):
        c=Curriculum()
        for _ in range(100): c.advance_iteration()
        with self.assertRaises(RuntimeError): c.advance_iteration()
        c.validate(True)
        with self.assertRaises(ValueError): c.validate(True)
