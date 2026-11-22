"""Worker transaction, CLI and real checkpoint/ledger CPU evidence."""
import importlib.util
import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

def worker():
    path=ROOT/'scripts/train_a2_ideal_stairs_v03.py'
    if not path.exists():raise AssertionError('Missing V5s actual worker entrypoint')
    spec=importlib.util.spec_from_file_location('v03_worker_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class WorkerContracts(unittest.TestCase):
    def setUp(self):self.w=worker()
    def test_fixed_global_and_internal_chunk_are_separate(self):
        a=self.w.parse_args(['--num-envs','258','--output-dir','fresh','--init-checkpoint','init.pt',
            '--init-checkpoint-sha256','2'*64,'--ledger-dir','ledger','--ledger-run-id','run42','--review-report','review.json'])
        self.assertEqual(a.max_iterations,23000);self.assertEqual(a.worker_chunk_iterations,200)
        for flags in (['--max-iterations','200'],['--worker-chunk-iterations','201'],['--resume-sha256','1'*64]):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):self.w.parse_args(['--num-envs','258','--output-dir','fresh','--init-checkpoint','init.pt',
                '--init-checkpoint-sha256','2'*64,'--ledger-dir','ledger','--ledger-run-id','run42','--review-report','review.json']+flags)
    def test_complete_transaction_persists_actual_controls_adam_and_budget_anchor(self):
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=Curriculum();ledger=AttemptLedger(root/'ledger',run_id='r')
            class Runner:
                learning_state={'optimizer_initialized':False,'completed_ppo_updates':0,'optimizer_steps':0}
                def collect(self):
                    for step in range(1,49):ledger.control_step(dict(step_index=step,num_envs=258,physics_samples_increment=258))
                def update(self):
                    ledger.update_event('optimizer_started',dict(learning_state=self.learning_state.copy(),expected_optimizer_steps=20))
                    for step in range(1,21):
                        self.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=0,optimizer_steps=step)
                        ledger.update_event('optimizer_step_completed',dict(learning_state=self.learning_state.copy()))
                    self.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20)
                    ledger.update_event('update_completed',dict(learning_state=self.learning_state.copy(),optimizer_steps_in_update=20));return {'loss':1.}
            self.w.perform_update(Runner(),c,ledger,258,root/'budget.json')
            document=json.loads((root/'budget.json').read_text())
            self.assertEqual(c.iteration,1);self.assertEqual(document['ledger_anchor'],ledger.anchor())
            self.assertEqual(document['ledger_path'],str(ledger.path.resolve()))
            self.assertEqual((ledger.completed_ppo_updates,ledger.physics_samples_observed),(1,12384))
            self.assertEqual(ledger.events[-1]['event'],'complete_attempt')
    def test_monitoring_reason_is_enum_and_pause_other_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'control.json';p.write_text(json.dumps({'action':'pause','reason':'monitoring_blocked'}))
            self.assertEqual(self.w.boundary_pause(p)['reason'],'monitoring_blocked')
            p.write_text(json.dumps({'action':'pause','reason':'network slow'}))
            with self.assertRaises(ValueError):self.w.boundary_pause(p)
    def test_gate_commit_precedes_boundary_save_and_environment_side_effect(self):
        calls=[]
        class Ledger:
            def record_curriculum(self,c):calls.append('record')
        class Runner:
            def save(self,*a,**k):calls.append('save');return '1'*64
        self.w.commit_curriculum(Runner(),object(),Ledger(),Path('boundary.pt'),{'seed':42})
        self.assertEqual(calls,['record','save'])

    def test_actual_core_48_control_20_adam_transaction_uses_real_ledger(self):
        import torch
        from test_v03_runner import TinyEnv,TinyModel
        from src.tasks.velocity.v03.runner import PPO,Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=AttemptLedger(root/'ledger',run_id='core');c=Curriculum()
            runner=PPO(TinyModel(),TinyEnv(),ledger=ledger)
            metrics=self.w.perform_update(runner,c,ledger,258,root/'budget.json')
            self.assertEqual(runner.learning_state,dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20))
            self.assertEqual(ledger.completed_ppo_updates,1);self.assertEqual(ledger.physics_samples_observed,12384)
            self.assertEqual(sum(e['event']=='optimizer_step_completed' for e in ledger.events),20)
            self.assertIn('clip_fraction',metrics)
    def initialization(self,root,fail_close=False):
        import torch
        from test_v03_runner import Env
        from src.tasks.velocity.v03.model import V03ActorCritic
        from src.tasks.velocity.v03.spec import INIT_PATH,INIT_SHA256
        torch.set_num_threads(2)
        a=self.w.parse_args(['--execution','initialize','--num-envs','258','--device','cpu','--output-dir',str(root/'initialize'),
            '--init-checkpoint',INIT_PATH,'--init-checkpoint-sha256',INIT_SHA256,'--ledger-dir',str(root/'ledger'),
            '--ledger-run-id','cpu_test_only','--review-report','CPU_FAKE_APPROVAL'])
        closed=[]
        class Environment(Env):
            def __init__(self,**kwargs):super().__init__()
            def close(self):closed.append('env')
        class Monitor:
            def __init__(self,path):self.path=path
            def start(self):return self
            def poll(self):pass
            def summary(self):return {}
            def close(self):
                closed.append('monitor');self.path.write_text(json.dumps({'error':'guard' if fail_close else None}))
                if fail_close:
                    from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
                    raise ResourceLimit('CPU injected close guard')
        with patch.object(self.w,'verify_worker_authorization'):
            result=self.w.run_worker(a,env_factory=Environment,monitor_factory=Monitor,
                model_factory=lambda:V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256))
        return result,closed
    def test_real_actor_zero_checkpoint_manifest_budget_and_closed_result(self):
        from src.tasks.velocity.v03.recovery import preload_checkpoint,AttemptLedger,audit_recovery
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,closed=self.initialization(root)
            self.assertEqual(result['status'],'initialized');self.assertEqual(closed,['env','monitor'])
            self.assertGreater(result['io']['write_bytes'],0);self.assertGreaterEqual(result['io']['checkpoint_save_elapsed_s'],0)
            self.assertGreaterEqual(result['disk_free_bytes'],10*1024**3)
            state=preload_checkpoint(result['checkpoint'],result['sha256'])
            self.assertEqual(state['learning_state']['completed_ppo_updates'],0);self.assertFalse(state['optimizer']['state'])
            manifest=json.loads((root/'initialize/manifest.json').read_text())
            self.assertEqual(manifest['execution'],'initialize');self.assertEqual(manifest['ledger_path'],str(root/'ledger/ledger.jsonl'))
            ledger=AttemptLedger(root/'ledger',run_id='cpu_test_only')
            self.assertEqual(audit_recovery(state,ledger)['status'],'eligible')
            self.assertEqual(json.loads((root/'initialize/result.json').read_text())['status'],'initialized')
    def test_close_guard_changes_returned_and_durable_success_to_resource_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,closed=self.initialization(root,fail_close=True)
            self.assertEqual(result['status'],'resource_failure')
            self.assertEqual(json.loads((root/'initialize/result.json').read_text())['status'],'resource_failure')
            self.assertEqual(closed,['env','monitor'])

    def test_dead_or_reused_supervisor_pid_requests_monitoring_boundary_pause(self):
        import argparse,os,psutil
        self.assertTrue(hasattr(self.w,'supervisor_pause'),'Missing owned-supervisor liveness boundary')
        a=argparse.Namespace(execution='train',supervisor_pid=os.getpid(),supervisor_start_time=psutil.Process().create_time())
        self.assertIsNone(self.w.supervisor_pause(a))
        a.supervisor_start_time-=1
        self.assertEqual(self.w.supervisor_pause(a)['reason'],'monitoring_blocked')
        a.supervisor_pid=2**30
        self.assertEqual(self.w.supervisor_pause(a)['reason'],'monitoring_blocked')

    def test_metadata_changed_gate_resume_requires_exact_same_policy_and_learning(self):
        import torch
        self.assertTrue(hasattr(self.w,'require_same_evaluation_policy'),'Missing pre/post gate policy proof')
        reference={'model':{'actor':torch.tensor([1.]),'critic':torch.tensor([2.])},'learning_state':{'completed_ppo_updates':400}}
        parent={'model':{k:v.clone() for k,v in reference['model'].items()},'learning_state':reference['learning_state'].copy()}
        self.w.require_same_evaluation_policy(parent,reference)
        parent['model']['actor']+=1
        with self.assertRaises(ValueError):self.w.require_same_evaluation_policy(parent,reference)
        parent['model']['actor']-=1;parent['learning_state']['completed_ppo_updates']+=1
        with self.assertRaises(ValueError):self.w.require_same_evaluation_policy(parent,reference)

    def test_invalid_resume_blocks_before_monitor_environment_or_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bad=root/'corrupt.pt';bad.write_text('not a checkpoint')
            a=self.w.parse_args(['--execution','train','--num-envs','258','--device','cpu','--output-dir',str(root/'attempt'),
                '--resume',str(bad),'--resume-sha256','a'*64,'--ledger-dir',str(root/'ledger'),'--ledger-run-id','r','--review-report','CPU_FAKE'])
            def forbidden(*a,**k):self.fail('Invalid resume allocated runtime components')
            with patch.object(self.w,'verify_worker_authorization'):
                result=self.w.run_worker(a,env_factory=forbidden,monitor_factory=forbidden,model_factory=forbidden)
            self.assertEqual(result['status'],'recovery_blocked')
            self.assertEqual(json.loads((root/'attempt/result.json').read_text())['status'],'recovery_blocked')

    def test_real_raw_gate_record_then_save_crash_blocks_previous_same_global_state(self):
        from test_v03_recovery import complete,checkpoint
        from test_v03_evaluation import make_gate_fixture
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger,audit_recovery,RecoveryBlocked
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=Curriculum();ledger=AttemptLedger(root/'ledger',run_id='gate-crash')
            # Synthetic accounting prefix only; all real gate raw verification
            # and the actual record/save interruption below keep real fsync.
            with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):
                for _ in range(200):complete(ledger,c)
            prior=checkpoint(c,ledger)
            summary,case_sha=make_gate_fixture(root/'raw',checkpoint_sha='1'*64)
            c.validate(summary,'1'*64,case_sha)
            class InterruptedSave:
                def save(self,*args,**kwargs):raise OSError('CPU injected interruption after durable curriculum commit')
            with self.assertRaises(OSError):self.w.commit_curriculum(InterruptedSave(),c,ledger,root/'gate.pt',{'seed':42})
            with self.assertRaisesRegex(RecoveryBlocked,'newer durable curriculum'):
                audit_recovery(prior,AttemptLedger(root/'ledger',run_id='gate-crash'))

    def test_round1_r1_direct_worker_gpu_is_blocked_before_any_runtime_entry(self):
        import argparse,fcntl
        from src.tasks.velocity.v03 import orchestration as o
        self.assertTrue(hasattr(o,'gpu_queue_guard'),'R1 worker lacks shared queue guard')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=root/'logs/a2_ideal_stairs_v03/.gpu_queue.lock';queue.parent.mkdir(parents=True)
            with queue.open('a') as owner,patch.object(o,'ROOT',root):
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);calls=[]
                with patch.object(self.w,'_run_worker',create=True,side_effect=lambda *a,**k:calls.append(a)):
                    with self.assertRaises(BlockingIOError):self.w.run_worker(argparse.Namespace(device='cuda:0'))
                    self.assertEqual(calls,[])
                    self.w.run_worker(argparse.Namespace(device='cpu'));self.assertEqual(len(calls),1)
    def test_round1_r2_probe_continue_has_explicit_remaining_and_target_contract(self):
        self.assertTrue(hasattr(self.w,'probe_remaining'),'R2 missing exact remainder validator')
        a=self.w.parse_args(['--execution','probe-continue','--num-envs','258','--output-dir','next_attempt','--resume','original.pt',
            '--resume-sha256','a'*64,'--ledger-dir','same_ledger','--ledger-run-id','same_run','--review-report','review.json',
            '--probe-remaining-updates','38','--probe-target-updates','55'])
        self.assertEqual(a.probe_remaining_updates,38);self.assertEqual(a.probe_target_updates,55)
        self.assertTrue(hasattr(self.w,'probe_remaining'),'Missing exact remainder validator')
        self.assertEqual(self.w.probe_remaining(a,17),38)
        with self.assertRaises(ValueError):self.w.probe_remaining(a,16)

    def test_fix1_all_external_gate_flags_reject_cpu_fixture_before_runtime_allocation(self):
        from src.tasks.velocity.v03.runner import Curriculum
        parent=dict(curriculum=Curriculum().state_dict(),metadata={'seed':42},num_envs=258)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for flag in ('validation','perturbed-validation','final-validation'):
                with self.subTest(flag=flag):
                    summary=root/(flag+'.json');summary.write_text(json.dumps({'evidence_mode':'cpu_fixture'}))
                    a=self.w.parse_args(['--execution','train','--num-envs','258','--device','cpu','--output-dir',str(root/flag),
                        '--resume','CPU_fake_parent.pt','--resume-sha256','a'*64,'--ledger-dir',str(root/(flag+'_ledger')),
                        '--ledger-run-id','r','--review-report','CPU_FAKE','--'+flag,str(summary),'--'+flag+'-cases-sha256','b'*64])
                    allocated=[]
                    def forbidden(*args,**kwargs):allocated.append(True);raise AssertionError('Runtime allocation before mode admission')
                    with patch.object(self.w,'verify_worker_authorization'),patch('src.tasks.velocity.v03.recovery.preload_checkpoint',return_value=parent),patch('src.tasks.velocity.v03.recovery.audit_recovery',return_value={'status':'eligible'}):
                        result=self.w.run_worker(a,env_factory=forbidden,monitor_factory=forbidden,model_factory=forbidden)
                    self.assertEqual(allocated,[]);self.assertIn('runtime',result['reason'])
                    self.assertEqual(result['curriculum']['history'],[])

if __name__=='__main__':unittest.main()
