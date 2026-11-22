"""Preallocation failures cannot refund paid updates; orphan adoption is strict."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from src.tasks.velocity.v02_formal.runner import Curriculum,PPO
from src.tasks.velocity.v02_formal.model import FormalActorCritic
from src.tasks.velocity.v02_formal.spec import INIT_PATH,INIT_SHA256
from src.tasks.velocity.v02_formal.recovery import recovered_budget,preload_checkpoint,preload_curriculum,recover_child_record,continuation_plan,option
from src.tasks.velocity.rl.ideal_stairs_runner import atomic_json
from test_formal_model_runner import FakeEnv

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def checkpoint(self,count=37):
        c=Curriculum()
        for _ in range(count):c.updated(512)
        path=self.root/'parent.pt';runner=PPO(FormalActorCritic.from_c(INIT_PATH,INIT_SHA256),FakeEnv());digest=runner.save(path,c,{'seed':42})
        return path,digest,c
    def test_preallocation_preserves_parent_and_rejects_refund(self):
        path,digest,c=self.checkpoint();loaded=preload_checkpoint(path,digest)
        self.assertEqual(preload_curriculum(loaded).iteration,37)
        directory=self.root/'child';directory.mkdir();atomic_json(directory/'budget.json',dict(curriculum=c.state_dict(),phase='preallocation'))
        restored,audit=recovered_budget(directory,c.state_dict());self.assertEqual(restored['iteration'],37)
        bad=Curriculum().state_dict();atomic_json(directory/'result.json',dict(curriculum=bad))
        atomic_json(directory/'budget.json',dict(curriculum=bad,phase='preallocation'))
        with self.assertRaises(ValueError):recovered_budget(directory,c.state_dict())
    def test_optimizer_inflight_charged_once_actual_258_env_steps_collect_not_charged(self):
        _,_,c=self.checkpoint();directory=self.root/'worker';directory.mkdir()
        atomic_json(directory/'budget.json',dict(curriculum=c.state_dict()))
        atomic_json(directory/'update_inflight.json',dict(phase='collect',before=c.state_dict(),attempt=38,num_envs=258))
        state,_=recovered_budget(directory,c.state_dict());self.assertEqual(state['iteration'],37)
        atomic_json(directory/'update_inflight.json',dict(phase='optimizer_started',before=c.state_dict(),attempt=38,num_envs=258))
        state,audit=recovered_budget(directory,c.state_dict());self.assertEqual(state['iteration'],38);self.assertEqual(state['env_steps'],c.env_steps+258*48);self.assertEqual(audit['conservative_charge'],1)
        again,_=recovered_budget(directory,state);self.assertEqual(again,state)
    def test_orphan_requires_pid_and_cuda_release_and_probe_lifecycle(self):
        path,digest,c=self.checkpoint();directory=self.root/'worker';directory.mkdir()
        atomic_json(directory/'result.json',dict(status='chunk_complete',checkpoint=str(path),sha256=digest,curriculum=c.state_dict()))
        record=dict(pid=123,command=['python','worker','--execution','train'])
        with self.assertRaises(RuntimeError):recover_child_record(record,directory,pid_alive=lambda _:True,gpu_pids=lambda:set())
        with self.assertRaises(RuntimeError):recover_child_record(record,directory,pid_alive=lambda _:False,gpu_pids=lambda:{123})
        restored,result=recover_child_record(record,directory,pid_alive=lambda _:False,gpu_pids=lambda:set());self.assertEqual(restored['returncode'],0)
        record['command'][-1]='probe'
        with self.assertRaises(RuntimeError):recover_child_record(record,directory,pid_alive=lambda _:False,gpu_pids=lambda:set())
    def test_worker_constructor_resource_failure_reports_parent_consumed_budget(self):
        import sys
        from scripts import train_a2_ideal_stairs_v02_formal as worker
        from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
        path,digest,c=self.checkpoint();out=self.root/'oom'
        args=['worker','--num-envs','258','--seed','42','--device','cuda:0','--output-dir',str(out),'--max-iterations','1','--resume',str(path),'--resume-sha256',digest,'--allow-num-env-change']
        with patch.object(sys,'argv',args),patch.object(worker,'ResourceMonitor',side_effect=ResourceLimit('injected constructor guard')):
            with self.assertRaises(ResourceLimit):worker.main()
        result=json.loads((out/'result.json').read_text());self.assertEqual(result['status'],'resource_failure');self.assertEqual(result['curriculum'],c.state_dict())
        self.assertEqual(preload_curriculum(preload_checkpoint(path,digest),out/'result.json').iteration,37)
    def test_worker_pause_during_optimizer_finishes_one_update_and_saves_before_stopping(self):
        import sys
        from scripts import train_a2_ideal_stairs_v02_formal as worker
        from src.tasks.velocity.v02_formal.spec import sha256
        control=self.root/'control.json';out=self.root/'paused';counts={'collect':0,'update':0}
        class Monitor:
            def start(self):return self
            def poll(self):pass
            def close(self):pass
            def summary(self):return {}
        class Environment(FakeEnv):
            def __init__(self,**kwargs):super().__init__()
            def close(self):pass
        class Runner:
            def __init__(self,model,env):self.model=model;self.env=env
            def reset(self):self.env.reset()
            def save(self,path,curriculum,metadata):
                path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);torch.save(curriculum.state_dict(),path);return sha256(path)
            def collect(self):counts['collect']+=1;return []
            def update(self):
                counts['update']+=1;atomic_json(control,dict(action='pause',reason='monitoring_blocked'));return {'loss':1.}
        args=['worker','--num-envs','512','--device','cpu','--seed','42','--output-dir',str(out),'--max-iterations','5','--init-checkpoint',INIT_PATH,'--init-checkpoint-sha256',INIT_SHA256,'--control-file',str(control)]
        with patch.object(sys,'argv',args),patch.object(worker,'ResourceMonitor',return_value=Monitor()),patch('src.tasks.velocity.v02_formal.env.FormalEnv',Environment),patch('src.tasks.velocity.v02_formal.runner.PPO',Runner):worker.main()
        result=json.loads((out/'result.json').read_text());self.assertEqual(result['status'],'monitoring_blocked');self.assertEqual(counts,{'collect':1,'update':1})
        self.assertEqual(result['curriculum']['iteration'],1);self.assertEqual(result['curriculum']['env_steps'],512*48)
        self.assertTrue(Path(result['checkpoint']).is_file());self.assertEqual(len((out/'updates.jsonl').read_text().splitlines()),1)

    def test_motion_aggregation_weights_samples_and_preserves_actual_maxima(self):
        from scripts.train_a2_ideal_stairs_v02_formal import MotionAccumulator
        motion=MotionAccumulator()
        motion.add(dict(sample_count=2,actual_vx_body=.2,peak_torque=5.,peak_torque_max=9.,lateral_max=.3,groups={'current_stairs/moving':dict(sample_count=2,actual_vx_body=.2,peak_torque_max=9.)}))
        motion.add(dict(sample_count=1,actual_vx_body=.5,peak_torque=8.,peak_torque_max=8.,lateral_max=.4,groups={'current_stairs/moving':dict(sample_count=1,actual_vx_body=.5,peak_torque_max=8.),'flat_stand/moving':dict(sample_count=0,actual_vx_body=0.)}))
        result=motion.result();self.assertEqual(result['sample_count'],3);self.assertAlmostEqual(result['actual_vx_body'],.3)
        self.assertEqual(result['peak_torque_max'],9.);self.assertEqual(result['lateral_max'],.4);self.assertEqual(result['mean_env_peak_torque'],6.)
        self.assertEqual(result['groups']['current_stairs/moving']['sample_count'],3);self.assertAlmostEqual(result['groups']['current_stairs/moving']['actual_vx_body'],.3)
        self.assertNotIn('flat_stand/moving',result['groups'])

    def test_monitor_close_resource_failure_cannot_hide_primary_numerical_failure(self):
        import sys
        from scripts import train_a2_ideal_stairs_v02_formal as worker
        from src.tasks.velocity.v02_formal.spec import sha256
        from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
        out=self.root/'primary_failure'
        class Monitor:
            def start(self):return self
            def poll(self):pass
            def close(self):raise ResourceLimit('secondary close margin')
            def summary(self):return {}
        class Environment(FakeEnv):
            def __init__(self,**kwargs):super().__init__()
            def close(self):pass
        class Runner:
            def __init__(self,model,env):self.model=model;self.env=env
            def reset(self):self.env.reset()
            def save(self,path,curriculum,metadata):
                path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);torch.save(curriculum.state_dict(),path);return sha256(path)
            def collect(self):raise FloatingPointError('primary numerical failure')
        args=['worker','--num-envs','512','--device','cpu','--seed','42','--output-dir',str(out),'--max-iterations','1','--init-checkpoint',INIT_PATH,'--init-checkpoint-sha256',INIT_SHA256]
        with patch.object(sys,'argv',args),patch.object(worker,'ResourceMonitor',return_value=Monitor()),patch('src.tasks.velocity.v02_formal.env.FormalEnv',Environment),patch('src.tasks.velocity.v02_formal.runner.PPO',Runner):
            with self.assertRaisesRegex(FloatingPointError,'primary numerical failure'):worker.main()
        result=json.loads((out/'result.json').read_text());self.assertEqual(result['status'],'implementation_failure')
        self.assertIn('primary numerical failure',result['reason']);self.assertIn('secondary close margin',result['monitor_close_error'])
        self.assertEqual(result['curriculum']['iteration'],0)

    def test_boundary_pause_only_recognizes_explicit_monitoring_block(self):
        from scripts.train_a2_ideal_stairs_v02_formal import boundary_pause
        path=self.root/'control.json';self.assertIsNone(boundary_pause(path))
        atomic_json(path,dict(action='pause',reason='monitoring_blocked'));self.assertEqual(boundary_pause(path)['reason'],'monitoring_blocked')
        atomic_json(path,dict(action='ignore',reason='x'))
        with self.assertRaises(ValueError):boundary_pause(path)

if __name__=='__main__':unittest.main()
