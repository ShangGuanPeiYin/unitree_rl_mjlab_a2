"""Synthetic checks only: no simulator rollout or learning loop."""
import unittest
import torch
from src.tasks.pointcloud.runner import tree_map, stack
from src.tasks.pointcloud.algorithm import compute_gae
from src.tasks.pointcloud.config import TrainSettings

class RunnerTests(unittest.TestCase):
    def test_timeout_bootstraps_terminal_not_reset(self):
        reward=torch.tensor([[1.,1.,1.]])
        values=torch.zeros_like(reward)
        last=torch.tensor([[10.,20.,30.]])
        failure=torch.tensor([[True,False,False]])
        success=torch.tensor([[False,True,False]])
        timeout=torch.tensor([[False,False,True]])
        adv,_=compute_gae(reward,values,last,failure,success,timeout)
        torch.testing.assert_close(adv,torch.tensor([[1.,1.,30.7]]))
    def test_rollout_boundary_is_not_done(self):
        zero=torch.zeros(1,1,dtype=torch.bool)
        adv,_=compute_gae(torch.ones(1,1),torch.zeros(1,1),torch.full((1,1),2.),zero,zero,zero)
        self.assertAlmostEqual(adv.item(),2.98,places=5)
    def test_nested_storage_keeps_actor_separate(self):
        records=[{'actor':{'points':torch.zeros(2,5,70,3)},'privileged':torch.ones(2,63)} for _ in range(3)]
        batch=tree_map(lambda x:x.flatten(0,1),stack(records))
        self.assertEqual(batch['actor']['points'].shape,(6,5,70,3))
        self.assertNotIn('privileged',batch['actor'])
    def test_recipe_rejects_silent_env_rounding(self):
        with self.assertRaisesRegex(ValueError,'8096'):
            TrainSettings('missing','out',num_envs=8192).validate()

if __name__=='__main__': unittest.main()
