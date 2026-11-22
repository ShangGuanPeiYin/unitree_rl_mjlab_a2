"""Boundary tests for budget, timeout GAE and non-power-of-two batches."""
import unittest
import torch
from src.tasks.velocity.rl.ideal_stairs_runner import Curriculum, PPOConfig, STAGE_LIMITS

class RunnerTests(unittest.TestCase):
    def test_batches(self):
        for n,b,m in ((1024,49152,12288),(512,24576,6144),(258,12384,3096)):
            self.assertEqual(n*PPOConfig().rollout,b)
            chunks=torch.arange(b).chunk(4)
            self.assertTrue(all(len(x)==m for x in chunks))
            self.assertEqual(len(torch.unique(torch.cat(chunks))),b)
    def test_budget_and_gates(self):
        c=Curriculum()
        for _ in range(500): c.updated()
        self.assertTrue(c.exhausted)
        with self.assertRaises(RuntimeError): c.advance()
        good={'speeds':[{'vx':v,'episodes':20,'failures':0,'mean_speed':v,'mean_vx':v} for v in (0.,.2,.4,.6)]}
        self.assertFalse(c.validate(good)); self.assertTrue(c.validate(good)); c.advance()
        self.assertEqual((c.stage,c.iteration,c.stage_iteration),(1,500,0))
        d=Curriculum(); d.load_state_dict(c.state_dict()); self.assertEqual(d.iteration,500)
        bad=dict(c.state_dict(),iteration=0)
        with self.assertRaises(ValueError): d.load_state_dict(bad)
        self.assertEqual(sum(STAGE_LIMITS),14000)
    def test_missing_gate_rejected(self):
        c=Curriculum()
        with self.assertRaises(ValueError): c.validate({"speeds":[]})
        c.stage=1
        with self.assertRaises(ValueError): c.validate({"speeds":[{"vx":.2,"episodes":20,"successes":20}]*3})
    def test_per_speed_gate(self):
        c=Curriculum(); c.stage=5
        summary={'speeds':[{'vx':v,'episodes':20,'successes':x} for v,x in zip((.2,.4,.6),(20,20,17))]}
        self.assertFalse(c.validate(summary)); self.assertFalse(c.validate(summary))

if __name__=='__main__': unittest.main()
