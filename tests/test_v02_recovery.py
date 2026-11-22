"""CPU fault injection for preallocation accounting and orphaned child recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.tasks.velocity.v02.runner import Curriculum
from src.tasks.velocity.v02.spec import contract, S0_SHA256
from src.tasks.velocity.v02.recovery import (preload_checkpoint, preload_curriculum,
    recovered_budget, recover_child_record)
from src.tasks.velocity.rl.ideal_stairs_runner import atomic_json
import torch

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
    def checkpoint(self,iteration=37):
        c=Curriculum('D','formal')
        for _ in range(iteration):c.updated()
        state=dict(contract=contract('D','formal'),curriculum=c.state_dict(),metadata={'seed':42},
            num_envs=1024,resume_mode='new_episode')
        path=self.root/f'checkpoint{iteration}.pt';torch.save(state,path)
        from src.tasks.velocity.v02.spec import sha256
        return path,sha256(path),c
    def test_parent_budget_loaded_before_any_gpu_constructor_and_sha_checked(self):
        path,digest,c=self.checkpoint()
        loaded=preload_checkpoint(path,digest,'D','formal')
        restored=preload_curriculum('D','formal',loaded)
        self.assertEqual(restored.iteration,37)
        with self.assertRaises(ValueError):preload_checkpoint(path,'0'*64,'D','formal')
        with self.assertRaises(ValueError):preload_checkpoint(path,digest,'A','formal')
        # Constructor-OOM result must use this preloaded curriculum, never a zero default.
        self.assertEqual(restored.state_dict()['consumed']['P1'],37)
    def test_recovery_never_refunds_and_uncertain_optimizer_attempt_charged_once(self):
        path,digest,c=self.checkpoint()
        directory=self.root/'interrupted';directory.mkdir()
        atomic_json(directory/'budget.json',dict(curriculum=c.state_dict(),last_complete_update=37))
        atomic_json(directory/'update_inflight.json',dict(phase='optimizer_started',before=c.state_dict(),attempt=38))
        state,audit=recovered_budget(directory,'D','formal',c.state_dict())
        self.assertEqual(state['iteration'],38);self.assertEqual(audit['conservative_charge'],1)
        self.assertEqual(recovered_budget(directory,'D','formal',state)[0]['iteration'],38)
        old=copy.deepcopy(c.state_dict());old['iteration']=0;old['consumed']['P1']=0;old['stage_iteration']=0
        atomic_json(directory/'budget.json',dict(curriculum=old))
        with self.assertRaises(ValueError):recovered_budget(directory,'D','formal',c.state_dict())
    def test_collect_interruption_does_not_charge_and_completed_update_not_double_charged(self):
        _,_,c=self.checkpoint();directory=self.root/'collect';directory.mkdir()
        atomic_json(directory/'budget.json',dict(curriculum=c.state_dict()))
        atomic_json(directory/'update_inflight.json',dict(phase='collect',before=c.state_dict(),attempt=38))
        state,audit=recovered_budget(directory,'D','formal',c.state_dict())
        self.assertEqual(state['iteration'],37);self.assertEqual(audit['conservative_charge'],0)
        c.updated();atomic_json(directory/'budget.json',dict(curriculum=c.state_dict()))
        marker=json.loads((directory/'update_inflight.json').read_text());marker['phase']='optimizer_started'
        atomic_json(directory/'update_inflight.json',marker)
        state,audit=recovered_budget(directory,'D','formal',c.state_dict())
        self.assertEqual(state['iteration'],38);self.assertEqual(audit['conservative_charge'],0)
    def test_orphan_complete_record_rebuilt_only_after_dead_pid_and_gpu_release(self):
        path,digest,c=self.checkpoint();directory=self.root/'worker';directory.mkdir()
        atomic_json(directory/'result.json',dict(status='chunk_complete',checkpoint=str(path),sha256=digest,curriculum=c.state_dict()))
        command=['python','train_a2_ideal_stairs_v02.py','--recipe','D','--mode','formal','--output-dir',str(directory)]
        record=dict(pid=123,command=command)
        with self.assertRaises(RuntimeError):recover_child_record(record,directory,pid_alive=lambda pid:True,gpu_pids=lambda:{123})
        with self.assertRaises(RuntimeError):recover_child_record(record,directory,pid_alive=lambda pid:False,gpu_pids=lambda:{123})
        restored,result=recover_child_record(record,directory,pid_alive=lambda pid:False,gpu_pids=lambda:set())
        self.assertEqual(restored['returncode'],0);self.assertEqual(result['curriculum']['iteration'],37)
        self.assertTrue(restored['returncode_inferred'])
    def test_pending_gate_survives_checkpoint_and_recovery_ledger(self):
        path,digest,c=self.checkpoint(200)
        loaded=preload_checkpoint(path,digest,'D','formal')
        resumed=preload_curriculum('D','formal',loaded)
        self.assertTrue(resumed.validation_due)
        with self.assertRaises(RuntimeError):resumed.updated()

    def test_worker_constructor_oom_preserves_nonzero_parent_in_result(self):
        import sys
        from scripts import train_a2_ideal_stairs_v02 as worker
        from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
        path,digest,c=self.checkpoint()
        for failure in ('monitor','environment'):
            out=self.root/failure
            arguments=['worker','--recipe','D','--mode','formal','--num-envs','512','--seed','42',
                '--device','cuda:0','--output-dir',str(out),'--max-iterations','1',
                '--resume',str(path),'--resume-sha256',digest,'--allow-num-env-change']
            class Monitor:
                def start(self):return self
                def close(self):pass
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys,'argv',arguments))
                stack.enter_context(patch.object(worker.subprocess,'check_output',return_value='synthetic'))
                if failure=='monitor':
                    stack.enter_context(patch.object(worker,'ResourceMonitor',side_effect=ResourceLimit('injected preallocation OOM')))
                else:
                    stack.enter_context(patch.object(worker,'ResourceMonitor',return_value=Monitor()))
                    stack.enter_context(patch('src.tasks.velocity.v02.env.V02Env',side_effect=torch.cuda.OutOfMemoryError('injected constructor OOM')))
                with self.assertRaises((ResourceLimit,torch.cuda.OutOfMemoryError)):worker.main()
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'],'resource_failure')
            self.assertEqual(result['curriculum'],c.state_dict())
            # The next downgrade can consume the same ledger without a refund error.
            restored=preload_curriculum('D','formal',preload_checkpoint(path,digest,'D','formal'),out/'result.json')
            self.assertEqual(restored.iteration,37)

    def test_periodic_checkpoint_plus_budget_creates_same_count_continuation(self):
        from src.tasks.velocity.v02.recovery import continuation_plan,option
        path,digest,c=self.checkpoint();directory=self.root/'interrupted_worker';directory.mkdir()
        for _ in range(8):c.updated()
        atomic_json(directory/'budget.json',dict(curriculum=c.state_dict()))
        atomic_json(directory/'update_inflight.json',dict(phase='optimizer_started',before=c.state_dict(),attempt=46))
        command=['python','train_a2_ideal_stairs_v02.py','--recipe','D','--mode','formal',
            '--execution','train','--num-envs','1024','--resume',str(path),'--resume-sha256',digest,'--output-dir',str(directory)]
        plan=continuation_plan(command,directory)
        self.assertEqual(option(plan['command'],'--num-envs'),'1024')
        self.assertNotEqual(plan['directory'],str(directory))
        restored=preload_curriculum('D','formal',preload_checkpoint(path,digest,'D','formal'),option(plan['command'],'--recovery-state'))
        self.assertEqual(restored.iteration,46)
        self.assertTrue((directory/'continuation.json').is_file())
        self.assertTrue((directory/'budget.json').is_file())

    def test_policy_hash_changes_only_actor_or_std_and_callback_observes_all_48_steps(self):
        from src.tasks.velocity.v02.runner import PPO,policy_state_sha256
        from src.tasks.velocity.v02.model import V02ActorCritic
        model=V02ActorCritic('B');before=policy_state_sha256(model)
        with torch.no_grad():model.critic[0].bias.add_(1)
        self.assertEqual(policy_state_sha256(model),before)
        with torch.no_grad():model.std[0].sub_(.01)
        self.assertNotEqual(policy_state_sha256(model),before)
        # Real inherited rollout/GAE, cheap policy only to avoid CPU convolution cost.
        from test_v02_model_runner import TinyModel,TerminalEnv
        env=TerminalEnv();runner=PPO(TinyModel(),env);original=env.step;records=[]
        runner.diagnostics_callback=lambda info:records.append(info)
        runner.collect()
        self.assertEqual(len(records),48)
        self.assertEqual(env.step,original)

    def test_replay_resource_guard_starts_before_cuda_model_and_failure_is_durable(self):
        from argparse import Namespace
        from scripts import play_a2_ideal_stairs_v02 as replay
        from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
        path,digest,_=self.checkpoint();out=self.root/'replay'
        arguments=Namespace(output_dir=str(out),checkpoint=str(path),checkpoint_sha256=digest,
            recipe='D',mode='formal',device='cuda:0')
        with patch.object(replay,'ResourceMonitor',side_effect=ResourceLimit('injected replay margin')):
            with self.assertRaises(ResourceLimit):replay.run(arguments)
        result=json.loads((out/'result.json').read_text())
        self.assertEqual(result['status'],'resource_failure')
        self.assertEqual(result['steps'],0)

    def test_supervisor_adopts_orphan_completed_child_without_launching_process(self):
        from scripts import run_a2_ideal_stairs_v02 as supervisor
        path,digest,c=self.checkpoint();directory=self.root/'orphan';directory.mkdir()
        command=['python','train_a2_ideal_stairs_v02.py','--recipe','D','--mode','formal','--output-dir',str(directory)]
        record=directory.parent/(directory.name+'_process.json')
        atomic_json(record,dict(pid=99999999,command=command))
        atomic_json(directory/'result.json',dict(status='chunk_complete',checkpoint=str(path),sha256=digest,curriculum=c.state_dict()))
        with patch.object(supervisor,'gpu_pids',return_value=set()),patch.object(supervisor.subprocess,'Popen',side_effect=AssertionError('duplicate process')):
            result=supervisor.child(command,directory)
        self.assertEqual(result['returncode'],0)
        self.assertEqual(result['curriculum']['iteration'],37)
        rebuilt=json.loads(record.read_text())
        self.assertTrue(rebuilt['returncode_inferred']);self.assertIn('cuda_pid_absent_at',rebuilt)

    def test_pending_validation_consumed_in_preflight_survives_allocation_failure(self):
        from scripts import train_a2_ideal_stairs_v02 as worker
        from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
        from src.tasks.velocity.v02.spec import TASK
        import sys
        path,digest,c=self.checkpoint(200);validation=self.root/'gate.json'
        atomic_json(validation,dict(task=TASK,code_version='0.2.0',recipe_version='0.2',checkpoint_schema=2,
            recipe='D',mode='formal',stage='P1',seed=42,eval_seed=200042,source_s0_sha256=S0_SHA256,
            training_num_envs=1024,checkpoint_sha256=digest,kind='validation',iteration=200,stage_iteration=200,
            speeds=[dict(vx=v,episodes=20,successes=18,failures=0) for v in (.2,.4,.6)]))
        out=self.root/'gate_oom';arguments=['worker','--recipe','D','--mode','formal','--num-envs','512','--seed','42',
            '--device','cuda:0','--output-dir',str(out),'--resume',str(path),'--resume-sha256',digest,
            '--validation',str(validation),'--allow-num-env-change']
        with patch.object(sys,'argv',arguments),patch.object(worker.subprocess,'check_output',return_value='synthetic'),patch.object(worker,'ResourceMonitor',side_effect=ResourceLimit('gate boundary OOM')):
            with self.assertRaises(ResourceLimit):worker.main()
        restored=preload_curriculum('D','formal',preload_checkpoint(path,digest,'D','formal'),out/'result.json')
        self.assertEqual(restored.iteration,200)
        self.assertEqual(len(restored.history),1)
        self.assertFalse(restored.validation_due)
        with self.assertRaises(ValueError):restored.validate(json.loads(validation.read_text()),expected_checkpoint_sha=digest)

    def test_probe_completion_without_monitor_close_is_not_adopted(self):
        path,digest,c=self.checkpoint();directory=self.root/'probe';directory.mkdir()
        command=['python','train_a2_ideal_stairs_v02.py','--recipe','D','--mode','formal','--execution','probe','--output-dir',str(directory)]
        atomic_json(directory/'result.json',dict(status='probe_updates_passed',checkpoint=str(path),sha256=digest,curriculum=c.state_dict()))
        with self.assertRaises(RuntimeError):
            recover_child_record(dict(pid=123,command=command),directory,pid_alive=lambda pid:False,gpu_pids=lambda:set())

    def test_resolved_eval_artifact_and_fixed2000_source_use_recovery_result(self):
        from src.tasks.velocity.v02.recovery import resolved_artifact,validated_checkpoint_source
        path,digest,_=self.checkpoint();original=self.root/'original';resolved=self.root/'original_recovery_001'
        resolved.mkdir();atomic_json(resolved/'summary.json',dict(checkpoint_sha256=digest))
        result=dict(resolved_directory=str(resolved),checkpoint=str(path),sha256=digest)
        self.assertEqual(resolved_artifact(result,original,'summary.json'),resolved/'summary.json')
        selected=validated_checkpoint_source(result,json.loads((resolved/'summary.json').read_text()))
        self.assertEqual(selected,dict(checkpoint=str(path),sha256=digest))
        self.assertFalse(original.exists())

if __name__=='__main__':unittest.main()
