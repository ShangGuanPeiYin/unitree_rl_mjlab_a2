import unittest
from src.tasks.pointcloud.config import TrainSettings
from src.tasks.pointcloud.curriculum import STAGES

class ConfigTests(unittest.TestCase):
    def test_budget_exact(self):
        self.assertEqual(sum(s[3] for s in STAGES),4000)
        self.assertEqual(8096*24,194304)
        self.assertEqual(8096*24//64,3036)
    def test_fixed_recipe(self):
        for key,value in [('rollout',25),('epochs',4),('mini_batches',32),('vx_max',.7),('max_iterations',4001)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                TrainSettings('missing','unused',**{key:value}).validate()
