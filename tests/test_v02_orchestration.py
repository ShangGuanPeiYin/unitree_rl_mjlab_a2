"""Case isolation and durable child lifecycle checks without GPU training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scripts.evaluate_a2_ideal_stairs_v02 import make_cases
from scripts.run_a2_ideal_stairs_v02 import child

class CaseTableTests(unittest.TestCase):
    def test_fixed_perturb_and_test_cases_are_frozen_and_distinct(self):
        fixed=make_cases('P4','fixed',20)
        self.assertTrue(all(c['initial_pose']==[-1.,0.,0.] for c in fixed['cases']))
        perturb=make_cases('P4','perturbed',20)
        self.assertEqual(perturb,make_cases('P4','perturbed',20))
        self.assertEqual(len(perturb['cases']),60)
        values=np.asarray([c['initial_pose'] for c in perturb['cases']])
        self.assertTrue(np.all((values[:,0]>=-1.1)&(values[:,0]<=-.9)))
        self.assertTrue(np.all(np.abs(values[:,1])<=.05))
        self.assertTrue(np.all(np.abs(values[:,2])<=np.pi/36))
        validation=make_cases('S5','validation',20);test=make_cases('S5','test',100)
        self.assertEqual(len(test['cases']),300)
        self.assertNotEqual(test['eval_seed'],validation['eval_seed'])
        self.assertNotEqual(test['cases'][0]['initial_pose'],validation['cases'][0]['initial_pose'])
        self.assertEqual(len(make_cases('flat','flat',20)['cases']),80)

    def test_finished_child_is_reused_without_duplicate_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'child'
            source='from pathlib import Path; import json; p=Path('+repr(str(target))+'); p.mkdir(); (p/"result.json").write_text(json.dumps({"status":"synthetic_complete"}))'
            command=[sys.executable,'-c',source]
            first=child(command,target)
            record=target.parent/(target.name+'_process.json');before=record.read_text()
            again=child(command,target)
            self.assertEqual(first,again);self.assertEqual(before,record.read_text())
            state=json.loads(before);self.assertEqual(state['returncode'],0)
            self.assertGreaterEqual(state['cuda_pid_absent_at'],state['ended'])
            with self.assertRaises(RuntimeError):child(command+['changed'],target)

if __name__=='__main__':unittest.main()
