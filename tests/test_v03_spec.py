"""Version isolation, source closure and no-flat slot allocation contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class SpecTests(unittest.TestCase):
    def spec(self):
        # A missing package is an assertion failure during the initial red run.
        self.assertIsNotNone(importlib.util.find_spec('src.tasks.velocity.v03.spec'),
                             'v03 specification is not implemented')
        from src.tasks.velocity.v03 import spec
        return spec

    def test_contract_binds_v03_source3000_and_fixed_training_recipe(self):
        s = self.spec()
        c = s.contract()
        self.assertEqual((c['code_version'], c['checkpoint_schema'], c['task'], c['protocol_id']),
                         ('0.3.0', 4, 'Unitree-A2-IdealHeightMap-Stairs-V03', 'v03-stair-body-speed-v1'))
        self.assertEqual(c['source_c_sha256'], '2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81')
        self.assertEqual(s.sha256(s.INIT_PATH), c['source_c_sha256'])
        self.assertEqual(c['source_completed_updates'], 3000)
        self.assertEqual(c['total_budget'], 23000)
        self.assertEqual((c['rollout'], c['epochs'], c['minibatches'], c['save_interval'], c['formal_eval_interval']), (48,5,4,100,200))
        self.assertEqual((c['vx_range'], c['vy'], c['wz']), ([.2,.6],0.,0.))
        self.assertEqual((c['initial_lr'],c['gamma'],c['gae_lambda'],c['recipe']['entropy']), (1e-4,.99,.95,.005))
        self.assertEqual(c['speed']['window_samples'], 25)
        self.assertEqual((c['speed']['epsilon'],c['speed']['max_deviation_samples']), (1e-6,50))
        self.assertEqual(c['evaluation']['seeds'], {'validation':200342,'perturbed':300342,'test':400342})
        self.assertEqual(c['rewards']['arrival'], 1.)
        self.assertEqual(c['rewards']['failure'], -4.)
        self.assertEqual(c['resources']['candidate_num_envs'], [1024,512,258])
        self.assertEqual(json.loads(json.dumps(c)), c)
        for recipe,mode in [('A','formal'),('C','comparison')]:
            with self.assertRaises(ValueError): s.contract(recipe,mode)

    def test_only_five_real_tile_stages_are_trainable(self):
        s = self.spec()
        self.assertEqual([(x.id,x.tile_id,x.N,x.h,x.budget) for x in s.STAGES],
                         [('S1',5,4,.08,3000),('S2',6,8,.12,4000),('S3',7,12,.16,4000),('S4',8,20,.20,5000),('S5',9,40,.20,7000)])
        self.assertNotIn('flat',s.STAGE_BY_ID)
        self.assertNotIn('R0',s.STAGE_BY_ID)
        self.assertEqual(set(s.STAGE_BY_TILE), {5,6,7,8,9})

    def test_slot_allocation_retains_exact_largest_remainders_and_order(self):
        s = self.spec()
        for n,expected in [(1024,[205,819]),(512,[102,410]),(258,[52,206]),(1,[0,1]),(2,[0,2]),(3,[1,2]),(4,[1,3])]:
            for stage,previous,current_tile in [('S2','S1',6),('S5','S4',9)]:
                tiles,classes,counts=s.allocate_stage_slots(n,stage,previous)
                self.assertEqual(counts.tolist(),expected)
                self.assertEqual(classes.tolist(),[0]*expected[0]+[1]*expected[1])
                self.assertEqual(tiles.tolist(),[current_tile-1]*expected[0]+[current_tile]*expected[1])
                self.assertEqual(tiles.dtype,np.int64)
            tiles,classes,counts=s.allocate_stage_slots(n,'S1')
            self.assertEqual(counts.tolist(),[0,n])
            self.assertTrue(np.all(tiles==5))
            self.assertTrue(np.all(classes==1))
        for args in [(0,'S1'),(-1,'S1'),(1.5,'S1'),(True,'S1'),(10,'flat'),(10,'R0'),(10,'S3','S1'),(10,'S1','S5')]:
            with self.assertRaises(ValueError,msg=str(args)): s.allocate_stage_slots(*args)

    def test_behavior_hashes_include_existing_new_and_transitive_sources(self):
        s=self.spec()
        hashes=s.behavior_hashes()
        for relative in ('src/tasks/velocity/v03/__init__.py','src/tasks/velocity/v03/spec.py',
                         'src/tasks/velocity/v02_formal/env.py','src/tasks/velocity/v02/rewards.py',
                         'src/tasks/velocity/rl/ideal_heightmap_actor_critic.py','src/terrains/open_stairs.py'):
            self.assertIn(relative,hashes)
        self.assertTrue(all((s.ROOT/p).is_file() and s.sha256(s.ROOT/p)==digest for p,digest in hashes.items()))
        # Real temporary files verify source changes and a transitive import are bound.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,contents in {'src/pkg/__init__.py':'','src/pkg/entry.py':'from .helper import value\n',
                                  'src/pkg/helper.py':'value=1\n'}.items():
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(contents)
            first=s.source_hashes(['src/pkg/entry.py'],root=root)
            self.assertIn('src/pkg/helper.py',first)
            self.assertIn('src/pkg/__init__.py',first)
            (root/'src/pkg/helper.py').write_text('value=2\n')
            second=s.source_hashes(['src/pkg/entry.py'],root=root)
            self.assertNotEqual(first['src/pkg/helper.py'],second['src/pkg/helper.py'])
            with self.assertRaises(FileNotFoundError): s.source_hashes(['src/pkg/missing.py'],root=root)


    def test_dynamic_task_scan_targets_and_their_static_imports_are_source_bound(self):
        s = self.spec()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                'src/tasks/__init__.py': 'from mjlab.utils.lab_api.tasks.importer import import_packages\nimport_packages(__name__, ["utils", ".mdp"])\n',
                'src/tasks/velocity/v03/entry.py': 'value=0\n',
                'src/tasks/velocity/config/as2/env_cfgs.py': 'from src.shared.config import value\n',
                'src/shared/config.py': 'value=1\n',
            }
            for name, contents in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents)
            first = s.source_hashes(['src/tasks/velocity/v03/entry.py'], root=root)
            target = root / 'src/tasks/velocity/config/as2/env_cfgs.py'
            target.write_text('from src.shared.config import value\nnew_setting=2\n')
            second = s.source_hashes(['src/tasks/velocity/v03/entry.py'], root=root)
            # This fails before fix1: both dictionaries ignore the real edited file.
            self.assertNotEqual(first, second)
            self.assertIn('src/tasks/velocity/config/as2/env_cfgs.py', first)
            self.assertIn('src/shared/config.py', first)
            (root / 'src/shared/config.py').write_text('value=3\n')
            third = s.source_hashes(['src/tasks/velocity/v03/entry.py'], root=root)
            self.assertNotEqual(second['src/shared/config.py'], third['src/shared/config.py'])
        actual = s.contract()['behavior_sha256']
        self.assertIn('src/tasks/velocity/config/as2/env_cfgs.py', actual)
        self.assertIn('src/tasks/velocity/config/g1/env_cfgs.py', actual)

if __name__=='__main__': unittest.main()
