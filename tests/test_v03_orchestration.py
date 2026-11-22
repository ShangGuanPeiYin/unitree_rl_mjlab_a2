"""CPU supervisor contract tests; no GPU or W&B session is created."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

class SupervisorContracts(unittest.TestCase):
    def setUp(self):
        try:
            from src.tasks.velocity.v03 import orchestration as o
        except ImportError as exc:
            self.fail('Missing V5s supervisor behavior: '+str(exc))
        self.o=o;self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def test_only_actual_resource_failure_authorizes_candidate_fallback(self):
        tried=[]
        def attempt(n):
            tried.append(n);return {'status':'resource_failure' if n==1024 else 'passed'}
        self.assertEqual(self.o.first_resource_fit(attempt),512);self.assertEqual(tried,[1024,512])
        for status in ('numerical_failure','implementation_failure','recovery_blocked','learning_failure'):
            tried=[]
            with self.assertRaises(RuntimeError):
                self.o.first_resource_fit(lambda n:(tried.append(n) or {'status':status}))
            self.assertEqual(tried,[1024])
    def test_worker_global_budget_is_not_chunk_budget_and_no_legacy_bypass_flags(self):
        a=self.o.parser().parse_args(['--output-dir',str(self.root/'run'),'--review-report','review.json',
            '--init-checkpoint','source.pt','--init-checkpoint-sha256','2'*64])
        self.assertEqual(a.max_iterations,23000);self.assertEqual(a.worker_chunk_iterations,200)
        self.assertEqual(tuple(a.num_envs_candidates),(1024,512,258))
        flags={option for action in self.o.parser()._actions for option in action.option_strings}
        self.assertFalse(any('monitor-patch' in x or 'preflight-restart' in x or 'flat' in x for x in flags))
        a.worker_chunk_iterations=201
        with self.assertRaises(ValueError):self.o.validate_arguments(a,check_files=False)
    def test_approval_binds_source_identity_and_independent_model(self):
        path=self.root/'approval.json';base=dict(approved=True,model='gpt-6.1-sol',reasoning_effort='high',
            task=self.o.TASK,protocol_id=self.o.PROTOCOL_ID,source_hashes={'source.py':'a'*64})
        path.write_text(json.dumps(base))
        with patch.object(self.o,'behavior_hashes',return_value=base['source_hashes']):
            self.assertEqual(self.o.verify_engineering_approval(path),base)
            for change in ({'approved':False},{'model':'other'},{'source_hashes':{}},{'task':'old'}):
                path.write_text(json.dumps(base|change))
                with self.assertRaises(ValueError):self.o.verify_engineering_approval(path)
    def test_runtime_approval_binds_same_run_zero_checkpoint_and_artifact_bytes(self):
        artifact=self.root/'migration.json';artifact.write_text('{}')
        engineering=self.root/'engineering.json'
        eng=dict(approved=True,model='gpt-6.1-sol',reasoning_effort='high',task=self.o.TASK,protocol_id=self.o.PROTOCOL_ID,source_hashes={'x':'b'*64})
        engineering.write_text(json.dumps(eng));zero=self.root/'zero.pt';zero.write_text('zero')
        state=dict(run_id='run42',wandb_run_id='remote42',num_envs=258,checkpoint=str(zero),sha256=self.o.sha256(zero),iteration=0,env_steps=0,
            phase='ready_for_formal',preflight_artifacts={k:{'path':str(artifact),'sha256':self.o.sha256(artifact)} for k in ('resources','baseline','online','migration')})
        document=eng|dict(code_version=self.o.CODE_VERSION,checkpoint_schema=self.o.SCHEMA_VERSION,seed=42,num_envs=258,
            run_id='run42',wandb_run_id='remote42',zero_checkpoint={'path':str(zero),'sha256':self.o.sha256(zero)},
            engineering_approval={'path':str(engineering),'sha256':self.o.sha256(engineering)},artifacts=state['preflight_artifacts'])
        p=self.root/'runtime.json';p.write_text(json.dumps(document))
        with patch.object(self.o,'behavior_hashes',return_value={'x':'b'*64}):
            self.assertEqual(self.o.verify_runtime_approval(p,state),document)
            for key,value in (('wandb_run_id','other'),('num_envs',512),('seed',43)):
                p.write_text(json.dumps(document|{key:value}))
                with self.assertRaises(ValueError):self.o.verify_runtime_approval(p,state)
            p.write_text(json.dumps(document));artifact.write_text('{"changed":true}')
            with self.assertRaises(ValueError):self.o.verify_runtime_approval(p,state)
    def test_child_release_refuses_still_owned_gpu_pid(self):
        ticks=iter([0.,16.])
        with self.assertRaises(RuntimeError):
            self.o.wait_cuda_release(7,query=lambda:{7},clock=lambda:next(ticks),sleep=lambda _:None)
    def test_resource_classification_does_not_hide_numerics(self):
        self.assertFalse(self.o.resource_failure(RuntimeError('nonfinite model')))
        self.assertFalse(self.o.resource_failure({'status':'recovery_blocked'}))
        self.assertTrue(self.o.resource_failure(RuntimeError('CUDA out of memory. requested allocation')))

    def test_runtime_controls_and_explicit_resume_do_not_change_recipe_configuration(self):
        a=self.o.parser().parse_args(['--output-dir','run','--review-report','old.json','--init-checkpoint','init.pt','--init-checkpoint-sha256','2'*64])
        supervisor=object.__new__(self.o.Supervisor);supervisor.a=a;first=supervisor.configuration()
        a.resume_run=True;a.stop_after_preflight=True;a.runtime_approval='runtime.json';a.review_report='new.json'
        a.init_checkpoint=None;a.init_checkpoint_sha256=None;a.resume='checkpoint.pt';a.resume_sha256='a'*64
        self.assertEqual(first,supervisor.configuration())
        a.vx_min=.3;self.assertNotEqual(first,supervisor.configuration())
    def test_real_child_exit_is_closed_and_restart_does_not_launch_duplicate(self):
        import sys
        target=self.root/'child';proof=self.root/'launched.txt'
        script="from pathlib import Path; import json; p=Path("+repr(str(target))+ ");p.mkdir();Path("+repr(str(proof))+ ").write_text('once');(p/'result.json').write_text(json.dumps({'status':'replay_complete'}));(p/'resources.json').write_text(json.dumps({'error':None}))"
        command=[sys.executable,'-c',script]
        a=self.o.child(command,target,gpu_query=lambda:set())
        b=self.o.child(command,target,gpu_query=lambda:set())
        self.assertEqual(a['status'],'replay_complete');self.assertEqual(a['returncode'],0)
        self.assertEqual(b['status'],'replay_complete');self.assertEqual(proof.read_text(),'once')
    def test_real_checkpoint_recovery_uses_shared_ledger_and_latest_anchor(self):
        import torch
        from test_v03_runner import Env
        from src.tasks.velocity.v03.runner import Curriculum,PPO
        from src.tasks.velocity.v03.model import V03ActorCritic
        from src.tasks.velocity.v03.recovery import AttemptLedger
        from src.tasks.velocity.v03.spec import INIT_PATH,INIT_SHA256
        torch.set_num_threads(2)
        ledger=AttemptLedger(self.root/'ledger',run_id='recovery');c=Curriculum()
        runner=PPO(V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256),Env(),ledger=ledger)
        older=self.root/'old.pt';runner.save(older,c,{'seed':42},ledger=ledger)
        newest=self.root/'new.pt';digest=runner.save(newest,c,{'seed':42},ledger=ledger)
        directory=self.root/'worker';directory.mkdir()
        (directory/'manifest.json').write_text(json.dumps(dict(execution='train',ledger_path=str(ledger.path),ledger_run_id='recovery')))
        result=self.o.recover_worker_boundary(directory)
        self.assertEqual(result['status'],'recovered_boundary');self.assertEqual(result['sha256'],digest)
        ledger.begin_attempt(c,258)
        ledger.control_step(dict(step_index=1,num_envs=258,physics_samples_increment=258))
        result=self.o.recover_worker_boundary(directory)
        self.assertEqual(result['status'],'recovery_blocked')
        self.assertTrue((directory/'recovery_blocked.json').exists())
    def test_probe_lifecycle_is_serial_55_evaluation_resume1_and_first_fit(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.state={};supervisor.save=lambda:None
        calls=[]
        def train(target,n,**kwargs):
            calls.append((kwargs['execution'],n))
            if n==1024:return {'status':'resource_failure'}
            count=56 if kwargs['execution']=='probe-resume' else 55
            from src.tasks.velocity.v03.runner import ProbeCurriculum
            c=ProbeCurriculum()
            for _ in range(count):c.updated(n)
            return dict(status='probe_updates_passed',lifecycle_closed=True,checkpoint='probe.pt',sha256='a'*64,curriculum=c.state_dict(),actual_num_envs=n,
                probe_timings=[{'iteration':i,'elapsed_s':1.} for i in range(1,count+1)])
        supervisor.train=train
        supervisor.evaluate=lambda *a,**k:(calls.append(('evaluate',512)) or 'summary.json')
        self.assertEqual(supervisor.probes(),512)
        self.assertEqual(calls,[('probe',1024),('probe',512),('evaluate',512),('probe-resume',512)])
        self.assertEqual(supervisor.probes(),512);self.assertEqual(len(calls),4)
    def test_ready_closes_nonterminal_session_and_never_starts_formal(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root
        supervisor.state=dict(run_id='run',wandb_run_id='remote',iteration=0,env_steps=0,num_envs=258,probes=[],checkpoint='zero.pt',sha256='a'*64,curriculum={'stage':'S1'},baseline=str(self.root/'baseline.json'))
        (self.root/'zero.pt').write_text('zero');supervisor.state['checkpoint']=str(self.root/'zero.pt')
        for name in ('baseline.json','wandb_manifest.json'):(self.root/name).write_text('{}')
        supervisor.save=lambda:None
        class Monitor:
            def close_session(self,summary):return dict(status='remote_verified',sdk_finish='completed',terminal=False)
        supervisor.telemetry=Monitor()
        result=supervisor.ready()
        self.assertEqual(result['status'],'ready_for_formal');self.assertEqual(result['formal_updates'],0)
        self.assertEqual(supervisor.state['phase'],'ready_for_formal');self.assertNotIn('final',supervisor.state)

    def test_full_preflight_with_fakebackend_records_baseline_failure_and_stops_zero(self):
        from src.tasks.velocity.v03.runner import Curriculum
        from test_v03_telemetry import Backend
        a=self.o.parser().parse_args(['--output-dir',str(self.root/'run'),'--review-report','fake_cpu_only',
            '--init-checkpoint','source.pt','--init-checkpoint-sha256','2'*64,'--stop-after-preflight'])
        calls=[];backend=Backend()
        class Harness(self.o.Supervisor):
            def probes(self):self.state['probes']=[{'num_envs':258,'status':'passed'}];return 258
            def train(self,target,n,**kwargs):
                calls.append(kwargs['execution']);p=self.out/'zero.pt';p.write_text('CPU synthetic checkpoint')
                return dict(status='initialized',lifecycle_closed=True,checkpoint=str(p),sha256=self.artifact(p)['sha256'],
                    curriculum=Curriculum().state_dict(),worker_directory=str(target),actual_num_envs=n)
            def evaluate(self,*args,**kwargs):
                p=self.out/'baseline.json';p.write_text(json.dumps({'passed':False,'kind':'baseline'}));return str(p)
        with patch.object(self.o,'ROOT',self.root):
            supervisor=Harness(a,backend_factory=lambda *args:backend)
            try:
                result=supervisor.run();self.assertEqual(result['status'],'ready_for_formal')
                self.assertEqual(calls,['initialize']);self.assertEqual(supervisor.state['iteration'],0)
                self.assertFalse(json.loads(Path(supervisor.state['baseline']).read_text())['passed'])
                self.assertEqual(backend.starts,[supervisor.state['wandb_run_id']])
                self.assertTrue(any(e['kind']=='session_pause' for e in backend.sent))
                self.assertFalse(any(e['kind']=='stop' for e in backend.sent))
                self.assertFalse((supervisor.out/'final.json').exists())
                self.assertEqual(supervisor.state['wandb_target']['project'],'p')
                frozen=(supervisor.out/'preflight_online.json').read_bytes();supervisor.connect()
                self.assertEqual((supervisor.out/'preflight_online.json').read_bytes(),frozen)
                a.wandb_project='conflicting'
                with self.assertRaises(ValueError):supervisor.connect()
                a.wandb_project=None
                other=argparse.Namespace(**vars(a));other.output_dir=str(self.root/'other')
                with self.assertRaises(BlockingIOError):Harness(other,backend_factory=lambda *args:backend)
            finally:supervisor.lock.close();supervisor.queue_lock.close()
    def test_eval_disk_pause_is_not_resource_fallback(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.telemetry=None
        supervisor.a=argparse.Namespace(eval_num_envs=8,device='cuda:0');supervisor.state={'iteration':0,'env_steps':0,'num_envs':258}
        calls=[]
        supervisor.save=lambda:None
        supervisor.child_runner=lambda *a,**k:(calls.append(a) or {'status':'disk_space_pause','returncode':1})
        from src.tasks.velocity.v03.evaluation import DiskSpacePause
        with self.assertRaises(DiskSpacePause):supervisor.evaluate('checkpoint.pt','a'*64,'S1','baseline',self.root/'eval')
        self.assertEqual(len(calls),1)
        with self.assertRaises(DiskSpacePause):supervisor.evaluate('checkpoint.pt','a'*64,'S1','baseline',self.root/'eval')
        self.assertNotEqual(str(calls[0][1]),str(calls[1][1]))

    def test_queue_lock_survives_parent_descriptor_close_while_child_is_alive(self):
        import fcntl,sys,threading,time
        queue=self.root/'queue.lock';handle=queue.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=self.root/'owned';ready=self.root/'ready';release=self.root/'release';outcome=[]
        script="from pathlib import Path; import json,time; p=Path("+repr(str(target))+ ");p.mkdir();Path("+repr(str(ready))+ ").write_text('ready');r=Path("+repr(str(release))+ ");exec('while not r.exists(): time.sleep(.01)');(p/'result.json').write_text(json.dumps({'status':'replay_complete'}));(p/'resources.json').write_text(json.dumps({'error':None}))"
        def execute():
            try:outcome.append(self.o.child([sys.executable,'-c',script],target,gpu_query=lambda:set()))
            except Exception as e:outcome.append(e)
        with patch.object(self.o,'QUEUE_LOCK_HANDLE',handle,create=True):
            thread=threading.Thread(target=execute);thread.start()
            try:
                deadline=time.monotonic()+10
                while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(ready.exists());handle.close()
                with queue.open('a') as contender:
                    with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:release.write_text('finish');thread.join(15);handle.close()
        self.assertFalse(thread.is_alive());self.assertIsInstance(outcome[0],dict)
        with queue.open('a') as contender:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_extra_and_final_use_original_second_gate_checkpoint_after_metadata_save(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.save=lambda:None
        source={'checkpoint':'second_regular.pt','sha256':'a'*64,'stage':'S1'}
        supervisor.state=dict(iteration=400,env_steps=1,checkpoint='post_gate.pt',sha256='b'*64,
            curriculum={'stage':'S1'},pending_pass=source,pending_action={'status':'needs_perturbed','stage_changed':False})
        from test_v03_telemetry import Backend
        from src.tasks.velocity.v03.telemetry import Telemetry
        backend=Backend();monitor=Telemetry(self.root/'gate_monitor',backend,{'CPU_synthetic_scheduler_only':True});monitor.start()
        calls=[]
        def evaluate(*args,**kwargs):
            calls.append((args,kwargs))
            summary=dict(stage=args[2],kind=args[3],iteration=400,stage_iteration=400,checkpoint_sha256=args[1],
                case_table_sha256=('c' if args[3]=='perturbed' else 'd' if args[3]=='test' else 'e')*64,
                parking_diagnostic=kwargs.get('parking',False),passed=True,speeds=[dict(vx=v,episodes=20,joint_successes=20) for v in (.2,.4,.6)])
            monitor.evaluation(summary);return 'CPU_synthetic_summary.json'
        supervisor.evaluate=evaluate;supervisor.replay=lambda *args:calls.append((args,{}))
        supervisor.diagnostics()
        self.assertEqual(calls[0][0][:4],('second_regular.pt','a'*64,'S1','perturbed'))
        supervisor.state['curriculum']={'stage':'S5','frozen_candidate_sha':'a'*64};source['stage']='S5'
        supervisor.state['pending_action']={'status':'qualified','stage_changed':False}
        calls.clear();supervisor.diagnostics()
        self.assertEqual(calls[-1][0][:4],('second_regular.pt','a'*64,'S5','test'))
        self.assertEqual(supervisor.state['candidate']['sha256'],'a'*64)
        events=[e for e in backend.sent if e['kind']=='evaluation']
        self.assertEqual(len(events),3)
        self.assertTrue(all(e['metrics']['eval/checkpoint_sha256']=='a'*64 for e in events))
        self.assertFalse(any(e['metrics'].get('eval/checkpoint_sha256')=='b'*64 for e in events))
    def test_terminal_reports_unsaved_complete_learning_without_relabeling_checkpoint(self):
        from test_v03_recovery import complete
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger
        ledger=AttemptLedger(self.root/'formal_ledger',run_id='r');c=Curriculum();complete(ledger,c)
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.telemetry=None;supervisor.save=lambda:None
        supervisor.state=dict(run_id='r',iteration=0,env_steps=0,checkpoint='zero.pt',sha256='a'*64,curriculum=Curriculum().state_dict())
        final=supervisor.stop('recovery_blocked',{'reason':'unsaved complete update'})
        self.assertEqual(final['checkpoint_iteration'],0)
        self.assertEqual(final['accounting']['completed_ppo_updates'],1)
        self.assertEqual(final['accounting']['physics_samples_observed'],12384)
        self.assertEqual(supervisor.state['iteration'],0)

    def test_play_wrapper_uses_independent_one_case_table_and_explicit_limit(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/play_a2_ideal_stairs_v03.py'
        spec=importlib.util.spec_from_file_location('play_v03_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        calls=[]
        with patch.object(module,'evaluation_main',side_effect=lambda argv:calls.append(argv)):
            module.main(['--checkpoint','frozen.pt','--checkpoint-sha256','a'*64,'--stage','S5','--vx','0.4',
                '--device','cpu','--output-dir',str(self.root/'replay'),'--max-steps','10000','--video'])
        args=calls[0];case_path=Path(args[args.index('--cases')+1]);table=json.loads(case_path.read_text())
        self.assertEqual(table['kind'],'replay');self.assertEqual(len(table['cases']),1)
        self.assertEqual(table['cases'][0]['vx'],.4);self.assertIn('--max-steps',args)
        self.assertEqual(args[args.index('--episodes')+1],'1')

    def test_new_supervisor_reuses_original_owned_child_command_only_for_owner_fields(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.control=None;supervisor.telemetry=None
        supervisor.state={'iteration':0,'env_steps':0,'num_envs':258};supervisor.save=lambda:None
        target=self.root/'owned';old=['python','train.py','--execution','train','--supervisor-pid','11','--supervisor-start-time','1.0']
        new=['python','train.py','--execution','train','--supervisor-pid','22','--supervisor-start-time','2.0']
        record=target.parent/(target.name+'_process.json');record.write_text(json.dumps({'command':old,'source_hashes':self.o.behavior_hashes()}))
        supervisor.worker_command=lambda *args,**kwargs:new
        seen=[];supervisor.child_runner=lambda command,*args,**kwargs:(seen.append(command) or {'status':'resource_failure'})
        supervisor.train(target,258)
        self.assertEqual(seen,[old])
        new.extend(['--vx-min','0.3'])
        with self.assertRaises(ValueError):supervisor.train(target,258)

    def test_round1_r1_shared_guard_blocks_independent_gpu_entrypoints_but_not_cpu(self):
        import fcntl,importlib.util
        self.assertTrue(hasattr(self.o,'gpu_queue_guard'),'R1: missing common GPU queue guard')
        queue=self.root/'logs/a2_ideal_stairs_v03/.gpu_queue.lock';queue.parent.mkdir(parents=True)
        with queue.open('a') as owner,patch.object(self.o,'ROOT',self.root):
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with self.o.gpu_queue_guard('cuda:0'):self.fail('Busy independent GPU queue entered')
            with self.o.gpu_queue_guard('cpu') as lease:self.assertIsNone(lease)
            with self.o.gpu_queue_guard('cuda:0',inherited_fd=owner.fileno()) as fd:
                self.assertEqual(fd,owner.fileno())
            for filename,extra in (('evaluate_a2_ideal_stairs_v03.py',['--kind','baseline','--cases','cases.json']),
                                   ('play_a2_ideal_stairs_v03.py',['--vx','0.4'])):
                path=Path(__file__).resolve().parents[1]/'scripts'/filename
                spec=importlib.util.spec_from_file_location('guard_'+filename[:-3],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                from src.tasks.velocity.v03 import evaluation as core
                called=[]
                def forbidden_monitor(*args):called.append(args);raise AssertionError('GPU runtime while queue held')
                # Keep the real core main: the sole CLI guard belongs there.
                with patch.object(core,'_resource_monitor',side_effect=forbidden_monitor):
                    with self.assertRaises(BlockingIOError):module.main(['--checkpoint','f.pt','--checkpoint-sha256','a'*64,'--stage','S1',
                        '--device','cuda:0','--output-dir',str(self.root/filename)]+extra)
                self.assertEqual(called,[])
    def test_round1_r2_recovered55_and_recovered56_continue_lifecycle_once(self):
        from src.tasks.velocity.v03.runner import ProbeCurriculum
        def boundary(count):
            c=ProbeCurriculum()
            for _ in range(count):c.updated(1024)
            return dict(status='recovered_boundary',checkpoint='original.pt',sha256='a'*64,curriculum=c.state_dict(),actual_num_envs=1024,
                recovery_audit={'status':'eligible'},lifecycle_closed=True,
                probe_timings=[{'iteration':i,'elapsed_s':1.+i/100} for i in range(1,count+1)])
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.state={};supervisor.save=lambda:None
        calls=[]
        def train(target,n,**kw):calls.append((kw['execution'],kw));return boundary(56 if kw['execution']=='probe-resume' else 55)
        supervisor.train=train;supervisor.evaluate=lambda *a,**k:(calls.append(('eval',{})) or 'frozen_summary.json')
        self.assertEqual(supervisor.probes(),1024)
        self.assertEqual([x[0] for x in calls],['probe','eval','probe-resume'])
        self.assertEqual(supervisor.probes(),1024);self.assertEqual(len(calls),3)
    def test_round1_r2_recovered0_and17_resume_original_remaining_without_refund(self):
        from src.tasks.velocity.v03.runner import ProbeCurriculum
        for count in (0,17):
            with self.subTest(count=count):
                def boundary(n):
                    c=ProbeCurriculum()
                    for _ in range(n):c.updated(1024)
                    return dict(status='recovered_boundary',checkpoint=f'original_{n}.pt',sha256='a'*64,curriculum=c.state_dict(),actual_num_envs=1024,
                        recovery_audit={'status':'eligible'},lifecycle_closed=True,
                        probe_timings=[{'iteration':i,'elapsed_s':1.+i/100} for i in range(1,n+1)])
                supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.state={};supervisor.save=lambda:None;calls=[]
                def train(target,n,**kw):
                    calls.append(kw)
                    return boundary(count if len(calls)==1 else 56 if kw['execution']=='probe-resume' else 55)
                supervisor.train=train;supervisor.evaluate=lambda *a,**k:'frozen_cases_summary.json'
                self.assertEqual(supervisor.probes(),1024)
                self.assertEqual(calls[1]['execution'],'probe-continue');self.assertEqual(calls[1]['checkpoint'],f'original_{count}.pt')
                self.assertEqual(calls[1]['probe_remaining'],55-count)
                self.assertEqual(calls[2]['execution'],'probe-resume')
                stats=supervisor.state['probes'][0]['statistics']
                self.assertEqual((stats['warmup'],stats['measured']),(5,50))
    def test_round1_r3_preflight_fatal_preserves_classification_and_final_without_formal_model(self):
        for phase,status in (('probe','numerical_failure'),('probe','implementation_failure'),('initialize','numerical_failure'),('initialize','implementation_failure')):
            with self.subTest(phase=phase,status=status):
                root=self.root/(phase+status);root.mkdir();supervisor=object.__new__(self.o.Supervisor)
                supervisor.out=root;supervisor.state=dict(phase=phase,run_id='r',iteration=0,env_steps=0,num_envs=1024,checkpoint=None,curriculum=None,probes=[])
                supervisor.save=lambda:None;supervisor.telemetry=None
                supervisor.connect=lambda:None
                calls=[]
                supervisor.train=lambda *a,**k:(calls.append((a,k)) or dict(status=status,returncode=1,worker_directory=str(root/'failed_task'),reason='injected CPU structured fatal',checkpoint='probe_only.pt'))
                supervisor.replay=lambda *a:self.fail('Probe model must not be replayed as formal')
                result=supervisor.run()
                self.assertEqual(result['status'],status);self.assertEqual(supervisor.state['phase'],'terminal')
                self.assertEqual(len(calls),1);self.assertTrue((root/'final.json').exists())
                self.assertIn('failed_task',json.dumps(result))

    def test_fix1_runtime_consumer_rejects_explicit_cpu_fixture_before_gate_consumption(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.telemetry=None;supervisor.save=lambda:None
        supervisor.a=argparse.Namespace(eval_num_envs=8,device='cuda:0');supervisor.state={'iteration':0,'env_steps':0,'num_envs':258}
        def fake_child(command,target,**kwargs):
            target.mkdir();cases=Path(command[command.index('--cases')+1]);(target/'summary.json').write_text(json.dumps(dict(
                checkpoint_sha256='a'*64,case_table_sha256=self.o.sha256(cases),evidence_mode='cpu_fixture')))
            return {'status':'evaluation_complete','returncode':0,'worker_directory':str(target)}
        supervisor.child_runner=fake_child
        try:supervisor.evaluate('checkpoint.pt','a'*64,'S1','validation',self.root/'eval')
        except Exception as error:
            self.assertIsInstance(error,self.o.ChildTaskFailure);self.assertIn('runtime',str(error))
        else:self.fail('Production accepted explicitly synthetic CPU fixture')

    def test_fix1_recovery_admission_requires_real_closed_lifecycle_and_eligible_audit(self):
        from src.tasks.velocity.v03.runner import ProbeCurriculum,Curriculum
        supervisor=object.__new__(self.o.Supervisor);supervisor.state={'num_envs':258}
        base=dict(status='recovered_boundary',curriculum=ProbeCurriculum().state_dict(),actual_num_envs=258,
            recovery_audit={'status':'eligible'},lifecycle_closed=True,probe_timings=[])
        self.assertEqual(supervisor._probe_boundary(base,258)[0],0)
        for change in ({'lifecycle_closed':False},{'recovery_audit':{'status':'blocked'}},{'status':'recovery_blocked'}):
            with self.subTest(change=change),self.assertRaises(self.o.ChildTaskFailure):supervisor._probe_boundary(base|change,258)
        zero=dict(status='recovered_boundary',curriculum=Curriculum().state_dict(),actual_num_envs=258,
            recovery_audit={'status':'eligible'},lifecycle_closed=True,returncode=137)
        supervisor._initialized_boundary(zero)
        with self.assertRaises(self.o.ChildTaskFailure):supervisor._initialized_boundary(zero|{'lifecycle_closed':False})
    def test_fix1_preflight_fatal_finishes_session_and_startup_refusal_remains_refusal(self):
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.save=lambda:None
        supervisor.state=dict(phase='initialize',run_id='r',iteration=0,env_steps=0,num_envs=258,checkpoint=None,curriculum=None,probes=[])
        finished=[]
        class Monitor:
            def finish(self,summary):finished.append(summary);return {'status':'remote_verified'}
        supervisor.telemetry=Monitor();supervisor.connect=lambda:None
        supervisor.train=lambda *a,**k:dict(status='implementation_failure',worker_directory='failed_initialize',reason='CPU fatal')
        result=supervisor.run();self.assertEqual(result['status'],'implementation_failure');self.assertEqual(len(finished),1)
        self.assertEqual(finished[0]['result/status'],'implementation_failure')
        other=self.root/'startup';other.mkdir();supervisor.out=other;supervisor.state['phase']='ready_for_formal'
        supervisor.a=argparse.Namespace(stop_after_preflight=False,runtime_approval=str(other/'missing_approval.json'))
        with self.assertRaises(OSError):supervisor.run()
        self.assertFalse((other/'final.json').exists());self.assertEqual(len(finished),1)

    def test_source_dependent_fix1_real_checkpoint0_55_56_closed_probe_handoff(self):
        """Synthetic55 accounting/Adam counters; real model/save/load/audit/close.

        This does not execute55 PPO updates or any real GPU resource probe.
        Only the55-event fixture prefix suppresses fsync; all boundary saves,
        recovery checks and the56th accounting transaction keep real fsync.
        """
        import torch
        from test_v03_runner import Env
        from test_v03_recovery import complete
        from src.tasks.velocity.v03.runner import PPO,ProbeCurriculum,Curriculum
        from src.tasks.velocity.v03.model import V03ActorCritic
        from src.tasks.velocity.v03.recovery import AttemptLedger,preload_checkpoint
        from src.tasks.velocity.v03.spec import INIT_PATH,INIT_SHA256
        torch.set_num_threads(1)
        ledger=AttemptLedger(self.root/'probe_ledger',run_id='strict-probe',execution='probe');c=ProbeCurriculum();env=Env();env.num_envs=1024
        runner=PPO(V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256),env,ledger=ledger)
        def recovered(name,execution):
            directory=self.root/name;directory.mkdir()
            (directory/'manifest.json').write_text(json.dumps(dict(execution=execution,ledger_path=str(ledger.path),ledger_run_id='strict-probe')))
            (directory/'resources.json').write_text(json.dumps({'error':None}))
            record={'pid':2**30,'command':['python','--execution',execution]}
            _,result=self.o._recover_exited(record,directory,lambda:set());result['worker_directory']=str(directory)
            self.assertTrue(result['lifecycle_closed']);self.assertEqual(result['recovery_audit']['status'],'eligible')
            return result
        runner.save(self.root/'probe0.pt',c,{'seed':42,'probe_timings':[]},ledger=ledger)
        zero=recovered('probe0_task','probe');self.assertEqual(zero['curriculum']['iteration'],0)
        # Genuine Adam tensors for every trainable parameter; counters below are
        # deliberately synthetic to isolate checkpoint lifecycle/state-machine.
        sum(p.sum() for p in runner.model.parameters() if p.requires_grad).backward();runner.optimizer.step();runner.model.constrain_std();runner.optimizer.zero_grad()
        with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):
            for _ in range(55):complete(ledger,c,n=1024)
        for state in runner.optimizer.state.values():state['step'].fill_(1100)
        runner.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=55,optimizer_steps=1100)
        rows=[dict(iteration=i,elapsed_s=1.) for i in range(1,56)]
        digest=runner.save(self.root/'probe55.pt',c,{'seed':42,'probe_timings':rows},ledger=ledger)
        self.assertEqual(preload_checkpoint(self.root/'probe55.pt',digest,execution='probe')['curriculum']['iteration'],55)
        at55=recovered('probe55_task','probe');calls=[]
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=self.root;supervisor.state={};supervisor.save=lambda:None
        def train(target,n,**kwargs):
            calls.append(kwargs['execution'])
            if kwargs['execution']=='probe':return at55
            self.assertEqual(kwargs['execution'],'probe-resume');self.assertEqual(kwargs['probe_remaining'],1)
            complete(ledger,c,n=1024)
            for state in runner.optimizer.state.values():state['step'].fill_(1120)
            runner.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=56,optimizer_steps=1120)
            runner.save(self.root/'probe56.pt',c,{'seed':42,'probe_timings':rows+[dict(iteration=56,elapsed_s=1.)]},ledger=ledger)
            return recovered('probe56_task','probe-resume')
        supervisor.train=train;supervisor.evaluate=lambda *a,**k:(calls.append('eval60_fake_boundary') or 'CPU_fake_evaluation_task')
        self.assertEqual(supervisor.probes(),1024);self.assertEqual(calls,['probe','eval60_fake_boundary','probe-resume'])
        self.assertEqual(supervisor.probes(),1024);self.assertEqual(len(calls),3);self.assertEqual(ledger.completed_ppo_updates,56)
        # A true formal0 checkpoint follows the same resource-close adoption path.
        formal=AttemptLedger(self.root/'formal_zero',run_id='formal-zero');formal_env=Env();formal_env.num_envs=1024
        init=PPO(V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256),formal_env,ledger=formal)
        init.save(self.root/'formal0.pt',Curriculum(),{'seed':42},ledger=formal)
        directory=self.root/'init_task';directory.mkdir();(directory/'manifest.json').write_text(json.dumps(dict(execution='initialize',ledger_path=str(formal.path),ledger_run_id='formal-zero')))
        (directory/'resources.json').write_text(json.dumps({'error':None}))
        _,result=self.o._recover_exited({'pid':2**30,'command':['python','--execution','initialize']},directory,lambda:set())
        supervisor.state['num_envs']=1024;supervisor._initialized_boundary(result)
        (directory/'resources.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'resource close'):
            self.o._recover_exited({'pid':2**30,'command':['python','--execution','initialize']},directory,lambda:set())

    def _fix2_executed_eval_without_resource_close(self,phase):
        """CPU child boundary, real _recover_exited/evaluate/run/stop, no GPU."""
        from src.tasks.velocity.v03.runner import Curriculum,ProbeCurriculum
        root=self.root/phase;root.mkdir()
        supervisor=object.__new__(self.o.Supervisor);supervisor.out=root;supervisor.save=lambda:None
        supervisor.a=argparse.Namespace(eval_num_envs=8,device='cuda:0')
        supervisor.state=dict(phase=phase,run_id='cpu-r3-close',iteration=0,env_steps=0,num_envs=258,
            checkpoint=None,sha256='a'*64,curriculum=Curriculum().state_dict(),probes=[])
        finished=[];calls=[];probe_calls=[];originals=[]
        class Monitor:
            def finish(self,summary):finished.append(summary);return {'status':'remote_verified'}
        supervisor.telemetry=Monitor();supervisor.connect=lambda:None
        supervisor.replay=lambda *args:self.fail('No formal checkpoint may trigger replay')
        def probe_boundary(target,n,**kwargs):
            probe_calls.append(n);c=ProbeCurriculum()
            for _ in range(55):c.updated(n)
            return dict(status='probe_updates_passed',lifecycle_closed=True,actual_num_envs=n,
                checkpoint='CPU_probe_only.pt',sha256='a'*64,curriculum=c.state_dict(),
                probe_timings=[dict(iteration=i,elapsed_s=1.) for i in range(1,56)])
        supervisor.train=probe_boundary
        def exited_child(command,target,**kwargs):
            calls.append((command,target));target.mkdir()
            original=dict(status='evaluation_complete',checkpoint=command[command.index('--checkpoint')+1],
                sha256='a'*64,summary=str(target/'summary.json'),marker='preserve-original-result')
            originals.append(original);(target/'result.json').write_text(json.dumps(original))
            # No resources file: the real lifecycle verifier must reject this.
            _,result=self.o._recover_exited({'pid':2**30,'command':command},target,lambda:set())
            return result
        supervisor.child_runner=exited_child
        result=supervisor.run()
        self.assertEqual(result['status'],'implementation_failure');self.assertEqual(supervisor.state['phase'],'terminal')
        self.assertTrue((root/'final.json').exists());self.assertEqual(len(finished),1);self.assertEqual(len(calls),1)
        failure=result['result'];self.assertEqual(failure['worker_directory'],str(calls[0][1]))
        self.assertEqual(failure['original_result'],originals[0]);self.assertIn('resource close',failure['reason'])
        self.assertIsNot(failure.get('lifecycle_closed'),True);self.assertFalse(self.o.resource_failure(failure))
        self.assertTrue((calls[0][1]/'result.json').exists());self.assertIsNone(result['last_checkpoint'])
        self.assertEqual(finished[0]['result/status'],'implementation_failure')
        self.assertEqual(probe_calls,[1024] if phase=='probe' else [])
        supervisor.run();self.assertEqual(len(finished),1);self.assertEqual(len(calls),1)
    def test_fix2_baseline_exited_eval_missing_close_is_terminal_and_finishes_once(self):
        self._fix2_executed_eval_without_resource_close('baseline')
    def test_fix2_probe_exited_eval_missing_close_is_terminal_and_finishes_once(self):
        self._fix2_executed_eval_without_resource_close('probe')

    def _v6_r1_load_wrapper(self,filename,tag):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts'/filename
        spec=importlib.util.spec_from_file_location('v6_wrapper_'+tag,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    def _v6_r1_wrapper_args(self,filename,root,device):
        from src.tasks.velocity.v03 import evaluation as core
        args=['--task',self.o.TASK,'--checkpoint','CPU_not_loaded.pt','--checkpoint-sha256','a'*64,
            '--stage','S1','--num-envs','1','--seed','42','--device',device,'--vy','0.0','--wz','0.0','--output-dir',str(root/'output')]
        if filename.startswith('evaluate'):
            cases=root/'input_cases.json';core.atomic_json(cases,core.make_cases('S1','baseline',20))
            return args+['--kind','baseline','--episodes','20','--cases',str(cases)]
        return args+['--vx','0.4','--max-steps','10000','--video']
    def test_v6_r1_wrappers_delegate_guard_once_and_preserve_fd_bridge(self):
        """Stage-A bridge RED; future core is a real same-guard CPU sentinel.

        This is not the joint lock acceptance test; that separate test keeps
        actual evaluation.main and replaces only its resource constructor.
        """
        import os
        from src.tasks.velocity.v03 import evaluation as core
        for filename in ('evaluate_a2_ideal_stairs_v03.py','play_a2_ideal_stairs_v03.py'):
            for inherited in (False,True):
                with self.subTest(wrapper=filename,inherited=inherited):
                    root=self.root/(filename+str(inherited));root.mkdir();start=len(self.o._OWNED_GPU_QUEUE_LEASES);owner=None
                    real_guard=self.o.gpu_queue_guard;entries=[];received=[];runtime=[];marker=object()
                    def counted_guard(device,**kwargs):
                        raw=kwargs.get('inherited_fd')
                        entries.append((device,raw if raw is not None else os.environ.get('V03_GPU_QUEUE_LOCK_FD')))
                        return real_guard(device,**kwargs)
                    def future_core_guard_sentinel(argv):
                        args=core.parser().parse_args(argv);raw=os.environ.pop('V03_GPU_QUEUE_LOCK_FD',None);received.append(raw)
                        with self.o.gpu_queue_guard(args.device,inherited_fd=raw):runtime.append('CPU_before_GPU');return marker
                    try:
                        with patch.object(self.o,'ROOT',root),patch.object(self.o,'QUEUE_LOCK_HANDLE',None),patch.dict(os.environ):
                            os.environ.pop('V03_GPU_QUEUE_LOCK_FD',None)
                            if inherited:
                                owner=self.o._new_gpu_queue_lease();os.set_inheritable(owner.fileno(),True)
                                os.environ['V03_GPU_QUEUE_LOCK_FD']=str(owner.fileno())
                            expected_fd=str(owner.fileno()) if owner is not None else None
                            with patch.object(self.o,'gpu_queue_guard',side_effect=counted_guard):
                                module=self._v6_r1_load_wrapper(filename,'bridge_'+filename+str(inherited));error=None;result=None
                                with patch.object(module,'evaluation_main',side_effect=future_core_guard_sentinel):
                                    try:result=module.main(self._v6_r1_wrapper_args(filename,root,'cuda:0'))
                                    except Exception as caught:error=caught
                            self.assertIsNone(error,'Wrapper acquired a lease before the sole future core guard: '+repr(error))
                            self.assertIs(result,marker);self.assertEqual(received,[expected_fd]);self.assertEqual(runtime,['CPU_before_GPU'])
                            self.assertEqual(len(entries),1,'Wrapper and core must not both acquire/validate queue leases')
                    finally:
                        for lease in self.o._OWNED_GPU_QUEUE_LEASES[start:]:lease.close()
                        del self.o._OWNED_GPU_QUEUE_LEASES[start:]
                        if owner is not None:owner.close()
    def test_v6_r1_wrappers_real_core_main_lock_to_resource_sentinel(self):
        """Joint GREEN must run real core main, no replacement of its guard."""
        import os
        from src.tasks.velocity.v03 import evaluation as core
        class CPUResourceSentinel(RuntimeError):pass
        for filename in ('evaluate_a2_ideal_stairs_v03.py','play_a2_ideal_stairs_v03.py'):
            for mode in ('independent','inherited','busy','cpu'):
                with self.subTest(wrapper=filename,mode=mode):
                    root=self.root/(filename+mode);root.mkdir();start=len(self.o._OWNED_GPU_QUEUE_LEASES);owner=None
                    real_guard=self.o.gpu_queue_guard;entries=[];runtime=[]
                    def counted_guard(device,**kwargs):
                        raw=kwargs.get('inherited_fd')
                        entries.append((device,raw if raw is not None else os.environ.get('V03_GPU_QUEUE_LOCK_FD')))
                        return real_guard(device,**kwargs)
                    def resource_sentinel(path,device):runtime.append((str(path),device));raise CPUResourceSentinel('CPU_STOP_BEFORE_NVML_CUDA')
                    try:
                        with patch.object(self.o,'ROOT',root),patch.object(self.o,'QUEUE_LOCK_HANDLE',None),patch.dict(os.environ):
                            os.environ.pop('V03_GPU_QUEUE_LOCK_FD',None)
                            if mode in ('inherited','busy','cpu'):owner=self.o._new_gpu_queue_lease()
                            if mode=='inherited':
                                os.set_inheritable(owner.fileno(),True);os.environ['V03_GPU_QUEUE_LOCK_FD']=str(owner.fileno())
                            with patch.object(self.o,'gpu_queue_guard',side_effect=counted_guard):
                                module=self._v6_r1_load_wrapper(filename,'real_'+filename+mode);error=None
                                with patch.object(core,'_resource_monitor',side_effect=resource_sentinel):
                                    try:module.main(self._v6_r1_wrapper_args(filename,root,'cpu' if mode=='cpu' else 'cuda:0'))
                                    except Exception as caught:error=caught
                            if mode=='busy':self.assertIsInstance(error,BlockingIOError);self.assertEqual(runtime,[])
                            else:self.assertIsInstance(error,CPUResourceSentinel);self.assertEqual(len(runtime),1)
                            self.assertEqual(len(entries),1,'Only the actual core CLI may consume the queue guard')
                            if mode=='inherited':
                                self.assertEqual(str(entries[0][1]),str(owner.fileno()));self.assertFalse(os.get_inheritable(owner.fileno()))
                                self.assertNotIn('V03_GPU_QUEUE_LOCK_FD',os.environ)
                    finally:
                        for lease in self.o._OWNED_GPU_QUEUE_LEASES[start:]:lease.close()
                        del self.o._OWNED_GPU_QUEUE_LEASES[start:]
                        if owner is not None:owner.close()

if __name__=='__main__':unittest.main()
