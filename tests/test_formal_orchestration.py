"""Case identity, statistical denominators and serial resource lifecycle contracts."""
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.tasks.velocity.v02_formal.evaluation import make_cases,validate_record,summarize_records,wilson
from src.tasks.velocity.v02_formal.orchestration import first_resource_fit,resource_failure,child,TRAIN_COUNTS

class CaseTests(unittest.TestCase):
    def test_local_rng_does_not_consume_python_or_numpy_global_state(self):
        random.seed(42);np.random.seed(42);before=(random.getstate(),np.random.get_state())
        a=make_cases('S5','validation');b=make_cases('S5','test',100)
        self.assertEqual(random.getstate(),before[0]);np.testing.assert_array_equal(np.random.get_state()[1],before[1][1])
        self.assertEqual(a,make_cases('S5','validation'));self.assertEqual(len(a['cases']),60);self.assertEqual(len(b['cases']),300)
        self.assertEqual(a['eval_seed'],200142);self.assertEqual(b['eval_seed'],400142)
        self.assertNotEqual(a['cases'][0]['initial_pose'],b['cases'][0]['initial_pose'])
    def test_perturb_bounds_parking_reuses_cases_and_flat80(self):
        table=make_cases('R0','perturbed');self.assertEqual(table['eval_seed'],300142)
        poses=np.array([case['initial_pose'] for case in table['cases']])
        self.assertTrue(np.all((-1.1<=poses[:,0])&(poses[:,0]<=-.9)));self.assertTrue(np.all(np.abs(poses[:,1])<=.05));self.assertTrue(np.all(np.abs(poses[:,2])<=math.pi/36))
        self.assertEqual(make_cases('S5','validation',parking=True),make_cases('S5','validation'))
        self.assertEqual(len(make_cases('flat','flat')['cases']),80)
        for stage,kind,count in [('R0','test',100),('S1','perturbed',20),('R0','validation',100),('flat','validation',20)]:
            with self.assertRaises(ValueError):make_cases(stage,kind,count)
    def records(self,stage='R0',kind='validation',episodes=20):
        return [dict(case_id=c['case_id'],env_id=0,initial_pose=c['initial_pose'],vx=c['vx'],climb_success=True,reason='success',duration=30.,mean_vx=c['vx'],mean_speed=c['vx'],max_lateral=.01,max_heading=.01,steps=1500) for c in make_cases(stage,kind,episodes)['cases']]
    def test_record_requires_prereset_identity_and_finite_stats(self):
        case=make_cases('R0','validation')['cases'][0];record=self.records()[0]
        result=validate_record(record,case,{'checkpoint_sha256':'a'*64});self.assertEqual(result['case_id'],case['case_id'])
        record['initial_pose']=[-.5,0,0]
        with self.assertRaises(ValueError):validate_record(record,case,{})
        record=self.records()[0];del record['climb_success']
        with self.assertRaises(ValueError):validate_record(record,case,{})
        record=self.records()[0];record['mean_vx']=float('nan')
        with self.assertRaises(ValueError):validate_record(record,case,{})
    def test_parking_not_arrived_is_not_failure_and_climb_history_retained(self):
        rows=self.records()
        for row in rows:row.update(parking_entered=False,parking_success=None,climb_success=False)
        for speed in (.2,.4,.6):
            selected=[r for r in rows if r['vx']==speed]
            selected[0].update(parking_entered=True,parking_success=True,climb_success=True)
            selected[1].update(parking_entered=True,parking_success=False,climb_success=True,post_climb_failure=True,reason='body_contact')
        for s in summarize_records(rows,'R0',20,parking=True):
            self.assertEqual(s['arrivals'],2);self.assertEqual(s['not_entered'],18);self.assertEqual(s['parking_failures'],1)
            self.assertEqual(s['successes'],2);self.assertEqual(s['conditional_success_rate'],.5);self.assertEqual(s['total_success_rate'],.05);self.assertEqual(s['post_arrival_failures'],1)
        for row in rows:row.update(parking_entered=False,parking_success=None)
        self.assertIsNone(summarize_records(rows,'R0',20,parking=True)[0]['conditional_success_rate'])
    def test_flat_gate_ignores_stairs_success_requires_30s_and_true_failure(self):
        rows=self.records('flat','flat')
        for row in rows:row['climb_success']=False;row['reason']='timeout'
        self.assertTrue(all(s['passed'] for s in summarize_records(rows,'flat',20)))
        rows[0]['reason']='orientation';self.assertFalse(summarize_records(rows,'flat',20)[0]['passed'])
        rows[0]['reason']='timeout';rows[0]['duration']=29.
        with self.assertRaises(ValueError):summarize_records(rows,'flat',20)
    def test_missing_duplicates_refused_and_wilson_not_probability_guarantee(self):
        rows=self.records()
        with self.assertRaises(ValueError):summarize_records(rows[:-1],'R0',20)
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):summarize_records(rows,'R0',20)
        low,high=wilson(95,100);self.assertLess(low,.95);self.assertGreater(high,.95);self.assertIsNone(wilson(0,0))

class ResourceTests(unittest.TestCase):
    def test_first_passing_lifecycle_stops_search(self):
        for failures,expected,tried in [(0,1024,[1024]),(1,512,[1024,512]),(2,258,[1024,512,258])]:
            calls=[]
            def attempt(n):
                calls.append(n);return {'status':'resource_failure' if len(calls)<=failures else 'passed'}
            self.assertEqual(first_resource_fit(attempt),expected);self.assertEqual(calls,tried)
    def test_interface_numeric_learning_failure_cannot_downgrade(self):
        for status in ('implementation_failure','numeric_failure','budget_exhausted','learning_failure'):
            calls=[]
            def attempt(n):calls.append(n);return {'status':status}
            with self.assertRaises(RuntimeError):first_resource_fit(attempt)
            self.assertEqual(calls,[1024])
        self.assertFalse(resource_failure(RuntimeError('shape mismatch OOM maybe')))
        self.assertFalse(resource_failure(ValueError('CUDA out of memory.')))
        self.assertTrue(resource_failure(RuntimeError('CUDA out of memory. allocated 2GiB')))
    def test_persistent_child_runs_once_and_waits_cuda_release(self):
        with tempfile.TemporaryDirectory() as path:
            target=Path(path)/'worker';source='import json; from pathlib import Path; p=Path('+repr(str(target))+');p.mkdir();(p/"result.json").write_text(json.dumps({"status":"synthetic_complete"}))'
            command=[sys.executable,'-c',source];queries=[]
            def no_gpu():queries.append(True);return set()
            first=child(command,target,gpu_query=no_gpu);again=child(command,target,gpu_query=no_gpu)
            self.assertEqual(first,again);self.assertGreaterEqual(len(queries),1)
            process=json.loads((target.parent/(target.name+'_process.json')).read_text())
            self.assertGreaterEqual(process['cuda_pid_absent_at'],process['started'])
            with self.assertRaises(RuntimeError):child(command+['changed'],target,gpu_query=no_gpu)


class SupervisorTransactionTests(unittest.TestCase):
    def test_full_serial_schedule_freezes_gate_candidate_and_recovers_diagnostics(self):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor,parser
        stages=[('R0',200,'needs_validation'),('R0',400,'needs_validation'),('R0',400,'needs_perturbed'),
                ('S1',600,'needs_validation'),('S1',800,'needs_validation'),('S2',1000,'needs_validation'),('S2',1200,'needs_validation'),
                ('S3',1400,'needs_validation'),('S3',1600,'needs_validation'),('S4',1800,'needs_validation'),('S4',2000,'needs_validation'),
                ('S5',2200,'needs_validation'),('S5',2400,'needs_validation'),('S5',2400,'qualified'),('S5',2400,'final_passed')]
        calls=[];evaluations=[];videos=[];crash=[True]
        class FakeSupervisor(Supervisor):
            def probes(self):return 512
            def connect(self):pass
            def train(self,target,n,**kwargs):
                initialize=kwargs.get('execution')=='initialize'
                stage,iteration,status=('R0',0,'initialized') if initialize else stages[self.state['chunk']]
                if not initialize:calls.append(self.state['chunk'])
                target=Path(target);target.mkdir(parents=True,exist_ok=True);checkpoint=target/'model.pt';checkpoint.write_text(str(target))
                digest=__import__('hashlib').sha256(str(target).encode()).hexdigest()
                c=dict(stage=stage,iteration=iteration,stage_iteration=iteration%400 or (400 if iteration else 0),consecutive_passes=0,r0_extra_passed=stage!='R0')
                if status=='qualified':c['frozen_candidate_sha']=self.state['pending_pass']['sha256']
                return dict(status=status,checkpoint=str(checkpoint),sha256=digest,curriculum=c,env_steps=iteration*512*48,worker_directory=str(target),returncode=0)
            def evaluate(self,checkpoint,digest,stage,kind,target,**kwargs):
                target=Path(target);target.mkdir(parents=True,exist_ok=True)
                path=target/'summary.json';path.write_text(json.dumps(dict(case_table_sha256='c'*64,checkpoint_sha256=digest)))
                evaluations.append((checkpoint,digest,stage,kind,kwargs.get('parking',False)))
                if crash[0] and target.name=='validation_000200':crash[0]=False;raise RuntimeError('simulated supervisor crash during diagnostics')
                return str(path)
            def replay(self,checkpoint,digest,stage,target):videos.append((stage,digest))
        with tempfile.TemporaryDirectory() as path:
            args=parser().parse_args(['--output-dir',str(Path(path)/'run'),'--review-report','unused','--wandb-run-name','test'])
            first=FakeSupervisor(args)
            with self.assertRaisesRegex(RuntimeError,'simulated supervisor crash'):first.run()
            saved=json.loads(first.state_path.read_text());self.assertEqual(saved['phase'],'diagnostics');self.assertEqual(saved['chunk'],1)
            first.lock.close();args.resume_run=True;resumed=FakeSupervisor(args);resumed.run()
            final=json.loads((resumed.out/'final.json').read_text());self.assertEqual(final['status'],'final_passed')
            self.assertEqual(calls,list(range(len(stages))))
            tests=[r for r in evaluations if r[3]=='test'];self.assertEqual(len(tests),1)
            candidate=final['selected_checkpoint'];self.assertEqual(tests[0][1],candidate['sha256'])
            self.assertNotEqual(candidate['sha256'],final['last_sha256'])
            r0_joint=next(r for r in evaluations if r[2:4]==('R0','validation') and 'chunk_0001_' in r[0])
            r0_perturb=next(r for r in evaluations if r[3]=='perturbed');self.assertEqual(r0_joint[1],r0_perturb[1])
            self.assertEqual([s for s,_ in videos],['R0','S1','S2','S3','S4','S5','S5'])
            resumed.lock.close()

class LiveResourceTests(unittest.TestCase):
    def test_live_gpu_memory_matches_worker_not_supervisor_and_missing_stays_missing(self):
        from src.tasks.velocity.v02_formal.orchestration import resource_snapshot,resource_close_metrics,safe_proxy_manifest
        def query(args):return '3200, 90\n' if '--query-gpu=memory.used,utilization.gpu' in args else '100, 400\n200, 1200\n'
        with tempfile.TemporaryDirectory() as path:
            row=resource_snapshot(200,0,path,query=query)
            self.assertEqual(row['system/process_gpu_mib'],1200.)
            self.assertEqual(row['system/gpu_used_mib'],3200.);self.assertEqual(row['system/gpu_utilization_percent'],90.)
            row=resource_snapshot(300,0,path,query=query);self.assertNotIn('system/process_gpu_mib',row)
            (Path(path)/'resources.json').write_text(json.dumps(dict(process_budget_mib=5120,sampling_interval_s=.02,peak_definition='NVML observed 20ms; spikes not guaranteed',error=None,
                samples=[dict(gpu_used_mib=2000,process_gpu_mib=1000,host_available_mib=5000),dict(gpu_used_mib=6000,process_gpu_mib=4000,host_available_mib=4500)])))
            peaks=resource_close_metrics(path);self.assertEqual(peaks['system/observed_peak_gpu_used_mib'],6000)
            self.assertEqual(peaks['system/minimum_host_available_mib'],4500);self.assertEqual(peaks['system/resource_sampling_interval_s'],.02)
        with patch.dict('os.environ',{'HTTP_PROXY':'http://user:secret@127.0.0.1:7897'}):self.assertEqual(safe_proxy_manifest()['HTTP_PROXY'],'http://127.0.0.1:7897')

class FormalDowngradeTests(unittest.TestCase):
    def test_supervisor_downgrades_only_after_each_real_resource_failure_and_preserves_paid_steps(self):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor,parser
        for failure_count,expected in [(0,[1024]),(1,[1024,512]),(2,[1024,512,258]),(3,[1024,512,258])]:
            calls=[];steps=[0];iteration=[0]
            class Fake(Supervisor):
                def connect(self):pass
                def probes(self):return 1024
                def train(self,target,n,**kwargs):
                    initializing=kwargs.get('execution')=='initialize'
                    if not initializing:
                        calls.append((n,kwargs.get('resource_change',False)))
                        iteration[0]+=10;steps[0]+=10*48*n
                    target=Path(target);target.mkdir(parents=True,exist_ok=True);checkpoint=target/'model.pt';checkpoint.write_text('mock')
                    status='initialized' if initializing else ('resource_failure' if len(calls)<=failure_count else 'budget_exhausted')
                    return dict(status=status,returncode=int(status=='resource_failure'),checkpoint=str(checkpoint),sha256=str(n).zfill(64),
                        worker_directory=str(target),curriculum=dict(stage='R0',iteration=iteration[0],stage_iteration=iteration[0],consumed={'R0':iteration[0]}),env_steps=steps[0])
                def evaluate(self,checkpoint,digest,stage,kind,target,**kwargs):
                    path=Path(target);path.mkdir(parents=True,exist_ok=True);(path/'summary.json').write_text('{}');return str(path/'summary.json')
                def resource_recovery(self,result):
                    self.state['recovery_state']='verified_mock_recovery_ledger';return result
                def replay(self,*args):pass
            with tempfile.TemporaryDirectory() as path:
                args=parser().parse_args(['--output-dir',str(Path(path)/'run'),'--review-report','unused','--wandb-run-name','test'])
                supervisor=Fake(args);supervisor.run()
                self.assertEqual([n for n,_ in calls],expected)
                self.assertEqual([changed for _,changed in calls],[False]+[True]*(len(calls)-1))
                self.assertEqual(supervisor.state['env_steps'],sum(10*48*n for n in expected))
                self.assertEqual(supervisor.state['iteration'],10*len(expected))
                self.assertEqual(supervisor.state['status'],'resources_exhausted' if failure_count==3 else 'budget_exhausted')
                supervisor.lock.close()


class TrustedCaseShaTests(unittest.TestCase):
    def test_worker_gate_sha_comes_from_frozen_table_and_tampering_rejected(self):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor,parser
        from src.tasks.velocity.v02_formal.spec import sha256
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(directory)
            supervisor.a=parser().parse_args(['--output-dir',directory,'--review-report','unused'])
            supervisor.state={'curriculum':{'stage':'R0'},'validation':str(Path(directory)/'summary.json')}
            table=Path(directory)/'cases'/'R0_validation_20.json';atomic_json(table,make_cases('R0','validation'))
            atomic_json(supervisor.state['validation'],{'case_table_sha256':sha256(table)})
            command=supervisor.worker_command(Path(directory)/'worker',512,checkpoint='parent.pt',digest='a'*64)
            self.assertEqual(command[command.index('--validation-cases-sha256')+1],sha256(table))
            atomic_json(supervisor.state['validation'],{'case_table_sha256':'b'*64})
            with self.assertRaisesRegex(ValueError,'Summary case SHA'):supervisor.worker_command(Path(directory)/'worker',512,checkpoint='parent.pt',digest='a'*64)
            atomic_json(supervisor.state['validation'],{'case_table_sha256':sha256(table)})
            changed=make_cases('R0','validation');changed['cases'][0]['initial_pose'][0]=-.5;atomic_json(table,changed)
            with self.assertRaisesRegex(ValueError,'Trusted frozen case table'):supervisor.worker_command(Path(directory)/'worker',512,checkpoint='parent.pt',digest='a'*64)


class PipelineFailureFix1Tests(unittest.TestCase):
    def orphan(self,directory,command,result=None,log=''):
        from src.tasks.velocity.v02_formal.spec import behavior_hashes
        directory=Path(directory);directory.mkdir()
        if result is not None:(directory/'result.json').write_text(json.dumps(result))
        (directory.parent/(directory.name+'.log')).write_text(log)
        (directory.parent/(directory.name+'_process.json')).write_text(json.dumps(dict(pid=999999,command=command,started=0.,source_hashes=behavior_hashes())))
    def test_t3_structured_primary_numeric_failure_never_becomes_resource_downgrade(self):
        with tempfile.TemporaryDirectory() as path:
            target=Path(path)/'orphan';command=[sys.executable,'worker','--execution','train']
            self.orphan(target,command,dict(status='implementation_failure',primary_error_type='FloatingPointError'),
                'FloatingPointError: invalid PPO\ncleanup\nsrc.tasks.velocity.rl.ideal_stairs_resources.ResourceLimit: secondary guard\n')
            calls=[]
            def attempt(n):calls.append(n);return child(command,target,gpu_query=lambda:set())
            with self.assertRaises(RuntimeError):first_resource_fit(attempt)
            self.assertEqual(calls,[1024]);self.assertEqual(child(command,target,gpu_query=lambda:set())['status'],'implementation_failure')
    def test_t4_no_continuation_or_adoption_before_old_cuda_release(self):
        with tempfile.TemporaryDirectory() as path:
            target=Path(path)/'orphan';command=[sys.executable,'worker','--execution','train'];self.orphan(target,command)
            queries=[]
            def old_cuda():queries.append(True);return {999999}
            with patch('src.tasks.velocity.v02_formal.recovery.continuation_plan') as continuation:
                with self.assertRaisesRegex(RuntimeError,'Old child CUDA'):child(command,target,gpu_query=old_cuda)
                continuation.assert_not_called()
            self.assertTrue(queries)
    def test_t4_live_pid_identity_mismatch_is_not_adopted(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as path:
            target=Path(path)/'orphan';command=[sys.executable,'worker','--execution','train'];self.orphan(target,command)
            record_path=target.parent/(target.name+'_process.json');record=json.loads(record_path.read_text());record['pid_start_time']=1.;record_path.write_text(json.dumps(record))
            process=SimpleNamespace(create_time=lambda:2.,cmdline=lambda:command)
            with patch('src.tasks.velocity.v02_formal.orchestration.pid_alive',return_value=True),patch('psutil.Process',return_value=process):
                with self.assertRaisesRegex(RuntimeError,'PID identity differs'):child(command,target,gpu_query=lambda:set())
    def test_t4_legal_schema3_probe_requires_resource_close_and_no_guard_error(self):
        import torch
        from src.tasks.velocity.v02_formal.model import FormalActorCritic
        from src.tasks.velocity.v02_formal.runner import PPO,Curriculum
        from src.tasks.velocity.v02_formal.spec import INIT_PATH,INIT_SHA256
        from test_formal_model_runner import FakeEnv
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as path:
            target=Path(path)/'probe';command=[sys.executable,'worker','--execution','probe']
            target.mkdir();model=FormalActorCritic.from_c(INIT_PATH,INIT_SHA256);c=Curriculum()
            checkpoint=Path(path)/'probe.pt';digest=PPO(model,FakeEnv()).save(checkpoint,c,{'seed':42})
            target.rmdir();self.orphan(target,command,dict(status='probe_updates_passed',checkpoint=str(checkpoint),sha256=digest,curriculum=c.state_dict()))
            with self.assertRaisesRegex(RuntimeError,'lacks resource monitor close'):child(command,target,gpu_query=lambda:set())
            (target/'resources.json').write_text(json.dumps({'error':'ResourceLimit at close'}))
            with self.assertRaisesRegex(RuntimeError,'guard error'):child(command,target,gpu_query=lambda:set())
            (target/'resources.json').write_text(json.dumps({'error':None,'samples':[{'gpu_used_mib':1000}]}))
            result=child(command,target,gpu_query=lambda:set());self.assertEqual(result['status'],'probe_updates_passed');self.assertEqual(result['returncode'],0)
    def test_t4_play_guard_precedes_model_cuda_constructor(self):
        from scripts import play_a2_ideal_stairs_v02_formal as play
        from src.tasks.velocity.rl.ideal_stairs_resources import ResourceLimit
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as path:
            args=['play','--checkpoint','unused','--checkpoint-sha256','a'*64,'--stage','R0','--vx','0.2','--output-dir',str(Path(path)/'blocked')]
            model=Mock()
            with patch.object(sys,'argv',args),patch('src.tasks.velocity.v02_formal.recovery.preload_checkpoint',return_value={}),patch('src.tasks.velocity.rl.ideal_stairs_resources.ResourceMonitor',side_effect=ResourceLimit('guard before any CUDA')),patch('src.tasks.velocity.v02_formal.model.FormalActorCritic',model):
                with self.assertRaises(ResourceLimit):play.main()
                model.assert_not_called()
            self.assertEqual(json.loads((Path(path)/'blocked'/'failure.json').read_text())['status'],'resource_failure')

class TerminalFix1Tests(unittest.TestCase):
    def supervisor(self,path):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(path);supervisor.state_path=Path(path)/'supervisor_state.json';supervisor.telemetry=None
        supervisor.state=dict(status='running',phase='train',checkpoint='last.pt',sha256='a'*64,iteration=3000,env_steps=48*258*3000,num_envs=258,
            curriculum=dict(stage='R0',stage_iteration=3000),consumption_ledger=[{'iteration':3000,'num_envs':258}],candidate={'checkpoint':'frozen.pt','sha256':'b'*64})
        return supervisor
    def test_t5_resource_exhaustion_persists_final_and_skips_gpu_replay(self):
        with tempfile.TemporaryDirectory() as path:
            supervisor=self.supervisor(path);supervisor.replay=unittest.mock.Mock(side_effect=RuntimeError('must not start GPU'))
            final=supervisor.stop('resources_exhausted',{'status':'resource_failure'})
            supervisor.replay.assert_not_called();self.assertEqual(final['status'],'resources_exhausted')
            self.assertEqual(final['diagnostics']['replay']['status'],'skipped_resource_guard');self.assertEqual(final['env_steps'],48*258*3000)
            self.assertEqual(json.loads((Path(path)/'final.json').read_text())['last_sha256'],'a'*64)
    def test_t5_encoder_and_upload_failures_do_not_override_final_failed(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as path:
            supervisor=self.supervisor(path)
            def replay(*args):
                self.assertEqual(json.loads((Path(path)/'final.json').read_text())['status'],'final_failed')
                raise ValueError('encoder unavailable')
            supervisor.replay=replay;finish=unittest.mock.Mock(side_effect=ConnectionError('upload unavailable'));supervisor.telemetry=SimpleNamespace(finish=finish)
            final=supervisor.stop('final_failed',{'status':'final_failed'})
            self.assertEqual(final['status'],'final_failed');self.assertEqual(final['diagnostics']['replay']['error_type'],'ValueError')
            self.assertEqual(final['diagnostics']['wandb_finish']['error_type'],'ConnectionError');finish.assert_called_once()
    def test_t5_crash_after_final_persist_resumes_without_learning_tests_or_replay(self):
        with tempfile.TemporaryDirectory() as path:
            supervisor=self.supervisor(path);supervisor.replay=unittest.mock.Mock(side_effect=KeyboardInterrupt('crash during optional diagnostic'))
            with self.assertRaises(KeyboardInterrupt):supervisor.stop('final_failed',{'status':'final_failed'})
            self.assertTrue((Path(path)/'final.json').exists())
            supervisor.connect=unittest.mock.Mock(side_effect=AssertionError('no remote/GPU work on terminal resume'))
            supervisor.train=unittest.mock.Mock(side_effect=AssertionError('no learning on terminal resume'))
            result=supervisor.run();self.assertEqual(result['status'],'final_failed')
            self.assertEqual(result['diagnostics']['replay']['status'],'interrupted_not_retried')
            self.assertEqual(supervisor.replay.call_count,1);supervisor.connect.assert_not_called();supervisor.train.assert_not_called()


class RestartMonitoringFix1Tests(unittest.TestCase):
    def test_online_recovery_failure_pauses_owned_live_worker_before_resume(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        from src.tasks.velocity.v02_formal.telemetry import MonitoringBlocked
        from src.tasks.velocity.v02_formal.spec import behavior_hashes
        with tempfile.TemporaryDirectory() as path:
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(path);supervisor.state_path=Path(path)/'supervisor_state.json';supervisor.control=None
            supervisor.a=SimpleNamespace(wandb_project=None,wandb_entity=None,wandb_run_name='test')
            supervisor.state={'config':{},'source_hashes':behavior_hashes(),'iteration':200,'env_steps':200*512*48}
            directory=Path(path)/'formal_seed42';directory.mkdir();control=directory/'worker_control.json'
            (directory/'worker_process.json').write_text(json.dumps({'source_hashes':behavior_hashes(),'pid':999999,'command':['worker','--control-file',str(control)]}))
            def unavailable():raise MonitoringBlocked('SDK queue remote delivery unknown')
            monitor=SimpleNamespace(restore_pending=lambda:None,start=unavailable)
            with patch('src.tasks.velocity.v02_formal.orchestration.WandbBackend'),patch('src.tasks.velocity.v02_formal.orchestration.Telemetry',return_value=monitor),patch('src.tasks.velocity.v02_formal.orchestration._validate_live_identity',return_value=True):
                with self.assertRaises(MonitoringBlocked):supervisor.connect()
            self.assertEqual(json.loads(control.read_text())['reason'],'monitoring_blocked')
            self.assertEqual(supervisor.state['iteration'],200);self.assertEqual(supervisor.state['env_steps'],200*512*48)


class VerifiedResumeFix1Tests(unittest.TestCase):
    def test_old_boundary_pause_can_continue_after_same_run_remote_reverification(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        with tempfile.TemporaryDirectory() as path:
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(path);supervisor.state_path=Path(path)/'supervisor_state.json'
            supervisor.state={'phase':'diagnostics','curriculum':{'stage':'R0'},'pending_action':{'status':'monitoring_blocked','stage_changed':False}}
            supervisor.telemetry=SimpleNamespace(state={'online_status':'remote_verified'})
            self.assertTrue(supervisor.diagnostics());self.assertEqual(supervisor.state['phase'],'train')
            self.assertNotIn('pending_action',supervisor.state)


class DiagnosticHeartbeatIsolationTests(unittest.TestCase):
    def test_diagnostic_child_keeps_heartbeat_resources_but_never_reads_training_jsonl(self):
        from types import SimpleNamespace
        calls=[]
        def read(_):raise AssertionError('diagnostic records must never enter training ingestion')
        spy=SimpleNamespace(pending_updates=[],read_worker=read,tick=lambda progress,**kw:calls.append(progress),emit_once=lambda *a,**kw:None)
        with tempfile.TemporaryDirectory() as path:
            target=Path(path)/'diagnostic';release=Path(path)/'sampled';source='import json,time;from pathlib import Path;p=Path('+repr(str(target))+');p.mkdir();deadline=time.monotonic()+10\nwhile not Path('+repr(str(release))+').exists():\n if time.monotonic()>deadline:raise RuntimeError("supervisor never sampled live diagnostic")\n time.sleep(.01)\n(p/"result.json").write_text(json.dumps({"status":"synthetic_complete"}))'
            def sample(*args):release.touch();return {'system/gpu_used_mib':123.}
            with patch('src.tasks.velocity.v02_formal.orchestration.resource_snapshot',side_effect=sample):
                result=child([sys.executable,'-c',source],target,telemetry=spy,gpu_query=lambda:set())
            self.assertEqual(result['returncode'],0);self.assertTrue(calls);self.assertTrue(any(row.get('system/gpu_used_mib')==123. for row in calls))
    def test_manifest_records_defined_groups_and_remote_count_verification(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        from src.tasks.velocity.v02_formal.telemetry import PANELS
        from test_formal_telemetry import Backend
        class ReportsBackend(Backend):
            def ensure_panels(self,existing=None):
                self.panel_verification={'remote_verified':True,'remote_group_count':8,'report_url':'https://example.invalid/report'}
                return 'https://example.invalid/report'
        with tempfile.TemporaryDirectory() as path:
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(path);supervisor.a=SimpleNamespace(wandb_project=None,wandb_entity=None,wandb_run_name='test');supervisor.state={'config':{'seed':42}}
            with patch('src.tasks.velocity.v02_formal.orchestration.WandbBackend',return_value=ReportsBackend()):supervisor.connect()
            manifest=json.loads((Path(path)/'wandb_manifest.json').read_text())
            self.assertEqual(manifest['panel_groups'],PANELS);self.assertTrue(manifest['panel_verification']['remote_verified']);self.assertEqual(manifest['panel_verification']['remote_group_count'],8)


class PreflightRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from src.tasks.velocity.v02_formal.model import FormalActorCritic
        from src.tasks.velocity.v02_formal.runner import PPO,Curriculum
        from src.tasks.velocity.v02_formal.spec import INIT_PATH,INIT_SHA256
        from test_formal_model_runner import FakeEnv
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'template.pt';PPO(FormalActorCritic.from_c(INIT_PATH,INIT_SHA256),FakeEnv()).save(path,Curriculum(),{'seed':42})
            cls.template=torch.load(path,map_location='cpu',weights_only=False)
    def fixture(self,directory):
        import copy,torch
        from src.tasks.velocity.v02_formal.orchestration import Supervisor,parser,PREFLIGHT_MONITOR_PATCH_SOURCES
        from src.tasks.velocity.v02_formal.spec import TASK,PROTOCOL_ID,behavior_hashes,contract,sha256
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        root=Path(directory);parent=root/'parent';parent.mkdir();new=root/'new';new.mkdir()
        old_sources=behavior_hashes()
        for index,key in enumerate(sorted(PREFLIGHT_MONITOR_PATCH_SOURCES)):old_sources[key]=str(index)*64
        old_contract=contract();old_contract['behavior_sha256']=old_sources
        approval=lambda sources:dict(approved=True,model='gpt-6.1-sol',reasoning_effort='high',task=TASK,protocol_id=PROTOCOL_ID,source_hashes=sources)
        old_approval=root/'old_approval.json';new_approval=root/'new_approval.json';atomic_json(old_approval,approval(old_sources));atomic_json(new_approval,approval(behavior_hashes()))
        args=parser().parse_args(['--output-dir',str(new),'--review-report',str(new_approval),'--wandb-run-name','test',
            '--preflight-restart-from',str(parent),'--preflight-parent-approval',str(old_approval),'--preflight-parent-checkpoint-sha256','a'*64,'--preflight-parent-wandb-run-id','same-run'])
        fake=Supervisor.__new__(Supervisor);fake.a=args;cfg=fake.configuration();old_cfg=copy.deepcopy(cfg);old_cfg.update(output_dir=str(parent),review_report=str(old_approval),contract=old_contract)
        initial=copy.deepcopy(self.template);initial['contract']=old_contract
        initial_dir=parent/'initialize';initial_dir.mkdir();initial_path=initial_dir/'initial.pt';torch.save(initial,initial_path)
        paused=parent/'paused.pt';torch.save(initial,paused);args.preflight_parent_checkpoint_sha256=sha256(paused)
        atomic_json(initial_dir/'result.json',dict(checkpoint=str(initial_path),sha256=sha256(initial_path)))
        state=dict(phase='train',status='monitoring_blocked',iteration=0,env_steps=0,num_envs=512,checkpoint=str(paused),sha256=sha256(paused),
            curriculum=copy.deepcopy(initial['curriculum']),consumption_ledger=[dict(iteration=0,env_steps=0,num_envs=512)],source_hashes=old_sources,config=old_cfg,
            probes=[dict(num_envs=1024,status='resource_failure'),dict(num_envs=512,status='passed')])
        atomic_json(parent/'supervisor_state.json',state);atomic_json(parent/'active_supervisor.json',{'pid':999999});(parent/'supervisor.lock').touch()
        tdir=parent/'wandb_telemetry';tdir.mkdir();events=[dict(id='same-run:000000000001',sequence=1,kind='initialize',created=0.,metrics={'global_iteration':0},source_ids=[]),dict(id='same-run:000000000002',sequence=2,kind='evaluation',created=0.,metrics={'global_iteration':0},source_ids=['evaluation:old'])]
        (tdir/'events.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
        diagnostic=dict(kind='validation',case_id='old-case',slot_class='current_stairs',climb_success=True,reward_sums={'reward_total':100.},duration=1.,_telemetry_source_id='old-eval/episodes.jsonl:0')
        tstate=dict(delivery_schema=2,run_id='same-run',sequence=2,acked=[event['id'] for event in events],cursors={'old-eval/episodes.jsonl':123},
            config=old_cfg,identity=dict(run_id='same-run',project='p',entity='e',url='https://example.invalid/same-run'),transport={event['id']:{'step':event['sequence']-1,'state':'confirmed','session':'old'} for event in events},next_transport_step=2,
            pending_updates=[],pending_episodes=[diagnostic])
        atomic_json(tdir/'state.json',tstate)
        return args,cfg,state,initial,tstate
    def parent_bytes(self,parent):return {str(path.relative_to(parent)):path.read_bytes() for path in Path(parent).rglob('*') if path.is_file()}
    def test_zero_header_restart_preserves_model_adam_rng_run_history_and_512(self):
        from src.tasks.velocity.v02_formal.orchestration import prepare_preflight_restart,_tree_equal
        from src.tasks.velocity.v02_formal.recovery import preload_checkpoint
        from src.tasks.velocity.v02_formal.spec import sha256
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,initial,tstate=self.fixture(directory);before=self.parent_bytes(args.preflight_restart_from)
            state=prepare_preflight_restart(args,args.output_dir,cfg,gpu_query=lambda:set())
            self.assertEqual(state['phase'],'preflight_revalidate');self.assertEqual(state['num_envs'],512);self.assertEqual(state['iteration'],0);self.assertEqual(state['env_steps'],0)
            migrated=preload_checkpoint(state['checkpoint'],state['sha256'])
            for key in ('model','optimizer','rng','learning_state','curriculum','ppo','transfer_metadata','num_envs'):self.assertTrue(_tree_equal(migrated[key],initial[key]),key)
            with self.assertRaises(ValueError):preload_checkpoint(old['checkpoint'],old['sha256'])
            new_tdir=Path(args.output_dir)/'wandb_telemetry';copied=json.loads((new_tdir/'state.json').read_text())
            self.assertEqual(copied['run_id'],'same-run');self.assertEqual(copied['cursors'],tstate['cursors']);self.assertEqual(copied['acked'],tstate['acked']);self.assertEqual(copied['transport'],tstate['transport'])
            self.assertEqual(copied['pending_episodes'],[]);self.assertEqual(len((new_tdir/'quarantine.jsonl').read_text().splitlines()),1)
            self.assertEqual((new_tdir/'events.jsonl').read_bytes(),before['wandb_telemetry/events.jsonl']);self.assertEqual(len(copied['config_history']),2)
            self.assertEqual(self.parent_bytes(args.preflight_restart_from),before)
            self.assertEqual(copied['pending_source_revision']['receipt_sha256'],sha256(Path(args.output_dir)/'preflight_restart.json'))
    def test_rejects_paid_budget_extra_source_live_pid_wrong_run_adam_rng_and_c_sha(self):
        import torch
        from src.tasks.velocity.v02_formal.orchestration import prepare_preflight_restart
        from src.tasks.velocity.v02_formal.spec import sha256
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        for mutation in ('budget','extra_source','live_pid','wrong_run','adam','rng','actor','c_sha','checkpoint_sha','hyperparameter'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                args,cfg,old,initial,tstate=self.fixture(directory);parent=Path(args.preflight_restart_from)
                if mutation=='budget':old['iteration']=1;atomic_json(parent/'supervisor_state.json',old)
                if mutation=='extra_source':
                    approval=json.loads(Path(args.preflight_parent_approval).read_text());key='src/tasks/velocity/v02_formal/env.py';approval['source_hashes'][key]='f'*64;old['source_hashes'][key]='f'*64
                    atomic_json(args.preflight_parent_approval,approval);atomic_json(parent/'supervisor_state.json',old)
                if mutation=='wrong_run':args.preflight_parent_wandb_run_id='another-run'
                if mutation=='checkpoint_sha':args.preflight_parent_checkpoint_sha256='f'*64
                if mutation=='hyperparameter':cfg['vx_max']=.5
                if mutation in ('adam','rng','actor','c_sha'):
                    bad=torch.load(old['checkpoint'],map_location='cpu',weights_only=False)
                    if mutation=='adam':bad['optimizer']['state']={0:{}}
                    if mutation=='rng':bad['rng']['torch'][0]^=1
                    if mutation=='actor':bad['model']['std'][0]+=.01
                    if mutation=='c_sha':bad['transfer_metadata']['sha256']='f'*64
                    torch.save(bad,old['checkpoint']);old['sha256']=sha256(old['checkpoint']);args.preflight_parent_checkpoint_sha256=old['sha256'];atomic_json(parent/'supervisor_state.json',old)
                before=self.parent_bytes(parent)
                with patch('src.tasks.velocity.v02_formal.orchestration.pid_alive',return_value=mutation=='live_pid'):
                    with self.assertRaises((ValueError,RuntimeError)):prepare_preflight_restart(args,args.output_dir,cfg,gpu_query=lambda:set())
                self.assertEqual(self.parent_bytes(parent),before)
    def test_new_directory_regular_resume_is_not_another_preflight_migration(self):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor,prepare_preflight_restart
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,initial,tstate=self.fixture(directory);state=prepare_preflight_restart(args,args.output_dir,cfg,gpu_query=lambda:set())
            atomic_json(Path(args.output_dir)/'supervisor_state.json',state);args.preflight_restart_from=None;args.resume_run=True
            args.preflight_parent_approval=args.preflight_parent_checkpoint_sha256=args.preflight_parent_wandb_run_id=None
            with patch('src.tasks.velocity.v02_formal.orchestration.prepare_preflight_restart',side_effect=AssertionError('must not migrate twice')):
                resumed=Supervisor(args)
            self.assertEqual(resumed.state['checkpoint'],state['checkpoint']);self.assertEqual(resumed.state['phase'],'preflight_revalidate');resumed.lock.close()
    def test_real_cli_help_and_incomplete_restart_flags_reject_before_launch(self):
        import contextlib,io
        from src.tasks.velocity.v02_formal.orchestration import main
        text=io.StringIO()
        with patch.object(sys,'argv',['run','--help']),contextlib.redirect_stdout(text):
            with self.assertRaises(SystemExit) as exit:main()
        self.assertEqual(exit.exception.code,0)
        for flag in ('--preflight-restart-from','--preflight-parent-approval','--preflight-parent-checkpoint-sha256','--preflight-parent-wandb-run-id'):self.assertIn(flag,text.getvalue())
        with patch.object(sys,'argv',['run','--output-dir','new','--review-report','unused','--preflight-restart-from','parent']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:main()
        self.assertEqual(exit.exception.code,2)

class RevalidationPhaseTests(unittest.TestCase):
    def test_selected512_revalidation_is_one_shot_and_never_replaces_zero_model(self):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        with tempfile.TemporaryDirectory() as path:
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(path);supervisor.state_path=Path(path)/'supervisor_state.json';supervisor.telemetry=None
            supervisor.state=dict(phase='preflight_revalidate',num_envs=512,checkpoint='migrated_zero.pt',sha256='a'*64,iteration=0,env_steps=0)
            calls=[]
            def probes(candidates,**kwargs):calls.append((candidates,kwargs));return 512
            supervisor.probes=probes;supervisor.connect=unittest.mock.Mock(side_effect=InterruptedError('stop after resource phase'))
            for _ in range(2):
                with self.assertRaises(InterruptedError):supervisor.run()
            self.assertEqual(calls,[((512,258),{'prefix':'revalidate','ledger':'resource_revalidation'})]);self.assertEqual(supervisor.state['checkpoint'],'migrated_zero.pt')
            self.assertEqual(supervisor.state['phase'],'baseline');self.assertTrue(supervisor.state['resource_revalidation_completed'])
    def test_revalidation512_failure_only_then258_and_separate_probe_history(self):
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        with tempfile.TemporaryDirectory() as path:
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(path);supervisor.state_path=Path(path)/'supervisor_state.json';supervisor.telemetry=None
            supervisor.state={'phase':'preflight_revalidate','num_envs':512,'checkpoint':'migrated_zero.pt','iteration':0,'env_steps':0,'probes':[{'num_envs':512,'status':'passed','parent':True}]}
            calls=[]
            def train(target,n,**kwargs):
                calls.append((n,kwargs['execution']))
                return {'status':'resource_failure','returncode':1} if n==512 else {'status':'probe_updates_passed','returncode':0,'checkpoint':'probe_not_formal.pt','sha256':'b'*64}
            supervisor.train=train;supervisor.evaluate=lambda *args,**kwargs:'probe_eval.json'
            self.assertEqual(supervisor.probes((512,258),prefix='revalidate',ledger='resource_revalidation'),258)
            self.assertEqual(calls,[(512,'probe'),(258,'probe'),(258,'probe-resume')]);self.assertEqual(supervisor.state['checkpoint'],'migrated_zero.pt')
            self.assertEqual(supervisor.state['probes'],[{'num_envs':512,'status':'passed','parent':True}]);self.assertEqual(len(supervisor.state['resource_revalidation']),2)


class ConfigRevisionProofTests(unittest.TestCase):
    fixture=PreflightRestartTests.fixture
    @classmethod
    def setUpClass(cls):PreflightRestartTests.setUpClass.__func__(cls)
    def test_revision_config_update_and_event_require_verified_receipt_and_remote_readback(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.orchestration import prepare_preflight_restart
        from src.tasks.velocity.v02_formal.telemetry import Telemetry,WandbBackend,MonitoringBlocked
        from test_formal_telemetry import Backend
        class Config(dict):
            def update(self,data,allow_val_change=False):
                if not allow_val_change:raise AssertionError('Revision must be explicit')
                super().update(data);self.explicit=True
        class RevisionBackend(Backend):
            def __init__(self):super().__init__();self.updated=False;self.verified=False
            def apply_config_revision(self,config,revision):
                actual=WandbBackend.__new__(WandbBackend);actual.run=SimpleNamespace(id='same-run',config=self.config)
                actual.apply_config_revision(config,revision);self.updated=True
            def verify_config_revision(self,revision):return self.verified
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,initial,tstate=self.fixture(directory);prepare_preflight_restart(args,args.output_dir,cfg,gpu_query=lambda:set())
            tdir=Path(args.output_dir)/'wandb_telemetry';backend=RevisionBackend();backend.config=Config()
            backend.remote=set(tstate['acked']);monitor=Telemetry(tdir,backend,cfg);monitor.restore_pending()
            with self.assertRaises(MonitoringBlocked):monitor.start()
            self.assertTrue(backend.updated);self.assertEqual(backend.config['contract'],cfg['contract'])
            self.assertIn('pending_source_revision',monitor.state)
            revisions=[event for event in monitor.events if event['kind']=='source_revision'];self.assertEqual(len(revisions),1)
            self.assertEqual(len([event for event in monitor.events if event['kind']=='initialize']),1)
            backend.verified=True;identity=monitor.start();self.assertEqual(identity['run_id'],'same-run')
            self.assertNotIn('pending_source_revision',monitor.state);self.assertEqual(len([e for e in monitor.events if e['kind']=='source_revision']),1)
            self.assertEqual((tdir/'events.jsonl').read_text().splitlines()[:2],(Path(args.preflight_restart_from)/'wandb_telemetry'/'events.jsonl').read_text().splitlines())
    def test_backend_will_not_update_config_with_forged_or_changed_receipt(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.orchestration import prepare_preflight_restart
        from src.tasks.velocity.v02_formal.telemetry import WandbBackend
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,initial,tstate=self.fixture(directory);prepare_preflight_restart(args,args.output_dir,cfg,gpu_query=lambda:set())
            state=json.loads((Path(args.output_dir)/'wandb_telemetry'/'state.json').read_text());revision=state['pending_source_revision']
            backend=WandbBackend.__new__(WandbBackend);config=unittest.mock.Mock();backend.run=SimpleNamespace(id='same-run',config=config)
            Path(revision['receipt_path']).write_text('{}')
            with self.assertRaises(ValueError):backend.apply_config_revision(cfg,revision)
            config.update.assert_not_called()


class LearnedMonitorRevisionTests(unittest.TestCase):
    fixture_zero=PreflightRestartTests.fixture
    parent_bytes=PreflightRestartTests.parent_bytes
    @classmethod
    def setUpClass(cls):
        import torch
        from src.tasks.velocity.v02_formal.model import FormalActorCritic
        from src.tasks.velocity.v02_formal.runner import PPO,Curriculum
        from src.tasks.velocity.v02_formal.spec import INIT_PATH,INIT_SHA256
        from test_formal_model_runner import FakeEnv
        PreflightRestartTests.setUpClass.__func__(cls)
        runner=PPO(FormalActorCritic.from_c(INIT_PATH,INIT_SHA256),FakeEnv())
        # Valid synthetic learned boundary: actual Adam tensors initialized by
        # 20 CPU steps, then explicit synthetic counters, not physical training.
        for _ in range(20):
            sum(p.sum() for p in runner.model.parameters()).backward();runner.optimizer.step();runner.model.constrain_std();runner.optimizer.zero_grad()
        for state in runner.optimizer.state.values():state['step'].fill_(1000)
        runner.learning_state={'optimizer_initialized':True,'completed_ppo_updates':50,'optimizer_steps':1000};c=Curriculum()
        for _ in range(50):c.updated(512)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'learned.pt';runner.save(path,c,{'seed':42});cls.learned=torch.load(path,map_location='cpu',weights_only=False)
    def fixture(self,directory):
        import copy,torch,shutil
        from src.tasks.velocity.v02_formal.spec import sha256
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        args,cfg,old,initial,tstate=self.fixture_zero(directory);parent=Path(args.preflight_restart_from)
        args.monitor_patch_resume_from=args.preflight_restart_from;args.monitor_parent_approval=args.preflight_parent_approval;args.monitor_parent_wandb_run_id=args.preflight_parent_wandb_run_id
        args.preflight_restart_from=args.preflight_parent_approval=args.preflight_parent_checkpoint_sha256=args.preflight_parent_wandb_run_id=None
        state=copy.deepcopy(self.learned);state['contract']=old['config']['contract'];worker=parent/'formal_seed42'/'chunk_0000_n512';(worker/'checkpoints').mkdir(parents=True)
        path=worker/'checkpoints'/'monitoring_blocked.pt';torch.save(state,path);digest=sha256(path);args.monitor_parent_checkpoint_sha256=digest
        result=dict(status='monitoring_blocked',checkpoint=str(path),sha256=digest,curriculum=state['curriculum'],global_iteration=50,env_steps=1228800)
        atomic_json(worker/'result.json',result)
        old.update(checkpoint=str(path),sha256=digest,curriculum=copy.deepcopy(state['curriculum']),iteration=50,env_steps=1228800,parent_run=str(worker))
        old['consumption_ledger'].append({'iteration':50,'env_steps':1228800,'num_envs':512,'checkpoint':str(path),'sha256':digest})
        old['resource_revalidation']=[{'num_envs':512,'status':'passed','from_parent':True}];old['resource_revalidation_completed']=True
        atomic_json(parent/'supervisor_state.json',old);shutil.rmtree(parent/'initialize')
        events=[]
        for sequence in range(1,89):
            kind='initialize' if sequence==1 else 'heartbeat';iteration=0
            if sequence in (54,76,85):kind='train';iteration={54:10,76:40,85:50}[sequence]
            if sequence==87:kind='resource_close'
            if sequence==88:kind='checkpoint';iteration=50
            events.append(dict(id=f'same-run:{sequence:012d}',sequence=sequence,kind=kind,created=float(sequence),metrics={'global_iteration':iteration,'stage_iteration':iteration,'env_steps':iteration*512*48},source_ids=[]))
        tdir=parent/'wandb_telemetry';(tdir/'events.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
        tstate.update(sequence=88,acked=[event['id'] for event in events],transport={event['id']:{'step':event['sequence']-1,'state':'confirmed','session':'old'} for event in events},next_transport_step=88,pending_updates=[],pending_episodes=[],
            source_revisions=[{'receipt_sha256':'old-receipt','receipt_path':'old-receipt.json','formal_iteration':0}],sdk_spools={'old-sdk':{'run_id':'same-run','sdk_owner_pid':999998}})
        atomic_json(tdir/'state.json',tstate);return args,cfg,old,state,tstate
    def test_learned50_preserves_all_trees_ledger88_history_and_not_zero_route(self):
        from src.tasks.velocity.v02_formal.orchestration import prepare_monitor_patch_resume,prepare_preflight_restart,_tree_equal,Supervisor
        from src.tasks.velocity.v02_formal.recovery import preload_checkpoint
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,previous,tstate=self.fixture(directory);before=self.parent_bytes(args.monitor_patch_resume_from)
            state=prepare_monitor_patch_resume(args,args.output_dir,cfg,gpu_query=lambda:set())
            self.assertEqual(state['phase'],'monitor_revalidate');self.assertEqual(state['iteration'],50);self.assertEqual(state['env_steps'],1228800);self.assertEqual(state['consumption_ledger'],old['consumption_ledger'])
            current=preload_checkpoint(state['checkpoint'],state['sha256'])
            for key in ('model','optimizer','rng','learning_state','curriculum','ppo','transfer_metadata','num_envs'):self.assertTrue(_tree_equal(previous[key],current[key]),key)
            self.assertEqual(state['resource_revalidation'],[]);self.assertNotIn('resource_revalidation_completed',state)
            self.assertEqual(state['parent_resource_revalidation_history'][-1]['records'],old['resource_revalidation'])
            tdir=Path(args.output_dir)/'wandb_telemetry';copied=json.loads((tdir/'state.json').read_text())
            for key in ('acked','transport','cursors','sdk_spools'):self.assertEqual(copied[key],tstate[key])
            self.assertEqual((tdir/'events.jsonl').read_bytes(),before['wandb_telemetry/events.jsonl']);self.assertEqual(copied['pending_source_revision']['formal_iteration'],50)
            self.assertEqual(self.parent_bytes(args.monitor_patch_resume_from),before)
            with self.assertRaises(ValueError):preload_checkpoint(old['checkpoint'],old['sha256'])
            atomic_json(Path(args.output_dir)/'supervisor_state.json',state)
            args.monitor_patch_resume_from=None;args.monitor_parent_approval=args.monitor_parent_checkpoint_sha256=args.monitor_parent_wandb_run_id=None;args.resume_run=True
            with patch('src.tasks.velocity.v02_formal.orchestration.prepare_monitor_patch_resume',side_effect=AssertionError('one-shot must not repeat')):resumed=Supervisor(args)
            self.assertEqual(resumed.state['iteration'],50);resumed.lock.close()
    def test_refuses_unacked_or_changed_adam_rng_result_budget_source_or_live_parent(self):
        import torch
        from src.tasks.velocity.v02_formal.orchestration import prepare_monitor_patch_resume
        from src.tasks.velocity.v02_formal.spec import sha256
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        for mutation in ('unacked','budget','adam','result','rng_shape','extra_source','live','wrong_run'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                args,cfg,old,previous,tstate=self.fixture(directory);parent=Path(args.monitor_patch_resume_from)
                if mutation=='unacked':tstate['acked'].pop();atomic_json(parent/'wandb_telemetry/state.json',tstate)
                if mutation=='budget':old['iteration']=49;atomic_json(parent/'supervisor_state.json',old)
                if mutation=='result':
                    path=Path(old['checkpoint']).parent.parent/'result.json';record=json.loads(path.read_text());record['global_iteration']=49;atomic_json(path,record)
                if mutation in ('adam','rng_shape'):
                    bad=torch.load(old['checkpoint'],map_location='cpu',weights_only=False)
                    if mutation=='adam':bad['optimizer']['state'][next(iter(bad['optimizer']['state']))]['step'].fill_(999)
                    if mutation=='rng_shape':bad['rng'].pop('torch')
                    torch.save(bad,old['checkpoint']);old['sha256']=sha256(old['checkpoint']);args.monitor_parent_checkpoint_sha256=old['sha256'];atomic_json(parent/'supervisor_state.json',old)
                    result_path=Path(old['checkpoint']).parent.parent/'result.json';record=json.loads(result_path.read_text());record['sha256']=old['sha256'];atomic_json(result_path,record)
                if mutation=='extra_source':
                    approval=json.loads(Path(args.monitor_parent_approval).read_text());approval['source_hashes']['src/tasks/velocity/v02_formal/env.py']='f'*64;atomic_json(args.monitor_parent_approval,approval);old['source_hashes']=approval['source_hashes'];atomic_json(parent/'supervisor_state.json',old)
                if mutation=='wrong_run':args.monitor_parent_wandb_run_id='different-run'
                before=self.parent_bytes(parent)
                with patch('src.tasks.velocity.v02_formal.orchestration.pid_alive',return_value=mutation=='live'):
                    with self.assertRaises((ValueError,RuntimeError)):prepare_monitor_patch_resume(args,args.output_dir,cfg,gpu_query=lambda:set())
                self.assertEqual(self.parent_bytes(parent),before)
    def test_learned_source_revision_event_is50_and_next_phase_train_no_baseline(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.orchestration import prepare_monitor_patch_resume,Supervisor
        from src.tasks.velocity.v02_formal.telemetry import Telemetry,WandbBackend
        from test_formal_telemetry import Backend
        class Config(dict):
            def update(self,values,allow_val_change=False):
                if not allow_val_change:raise AssertionError('explicit revision required')
                super().update(values)
        class RevisionBackend(Backend):
            def apply_config_revision(self,config,revision):
                backend=WandbBackend.__new__(WandbBackend);backend.run=SimpleNamespace(id='same-run',config=Config());backend.apply_config_revision(config,revision)
            def verify_config_revision(self,revision):return True
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,previous,tstate=self.fixture(directory);state=prepare_monitor_patch_resume(args,args.output_dir,cfg,gpu_query=lambda:set())
            backend=RevisionBackend();backend.remote=set(tstate['acked']);t=Telemetry(Path(args.output_dir)/'wandb_telemetry',backend,cfg);t.start()
            event=next(event for event in backend.sent if event['kind']=='source_revision');self.assertEqual(event['metrics']['global_iteration'],50);self.assertEqual(event['metrics']['env_steps'],1228800)
            supervisor=Supervisor.__new__(Supervisor);supervisor.out=Path(args.output_dir);supervisor.state_path=supervisor.out/'supervisor_state.json';supervisor.state=state;supervisor.telemetry=None
            supervisor.probes=unittest.mock.Mock(return_value=512);supervisor.connect=unittest.mock.Mock(side_effect=InterruptedError('after full revalidation'))
            with self.assertRaises(InterruptedError):supervisor.run()
            self.assertEqual(supervisor.state['phase'],'train');self.assertEqual(supervisor.state['iteration'],50);self.assertEqual(supervisor.state['checkpoint'],state['checkpoint']);supervisor.probes.assert_called_once_with((512,258),prefix='revalidate',ledger='resource_revalidation')
            with self.assertRaises(InterruptedError):supervisor.run()
            self.assertEqual(supervisor.probes.call_count,1)
    def test_learned_cli_help_and_zero_entry_cannot_admit_consumed_state(self):
        import contextlib,io
        from src.tasks.velocity.v02_formal.orchestration import main,prepare_preflight_restart
        output=io.StringIO()
        with patch.object(sys,'argv',['run','--help']),contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:main()
        self.assertEqual(result.exception.code,0)
        for flag in ('--monitor-patch-resume-from','--monitor-parent-approval','--monitor-parent-checkpoint-sha256','--monitor-parent-wandb-run-id'):self.assertIn(flag,output.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            args,cfg,old,previous,tstate=self.fixture(directory)
            args.preflight_restart_from=args.monitor_patch_resume_from;args.preflight_parent_approval=args.monitor_parent_approval;args.preflight_parent_checkpoint_sha256=args.monitor_parent_checkpoint_sha256;args.preflight_parent_wandb_run_id=args.monitor_parent_wandb_run_id
            with self.assertRaisesRegex(ValueError,'zero'):prepare_preflight_restart(args,args.output_dir,cfg,gpu_query=lambda:set())


class ChildProgressClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):LearnedMonitorRevisionTests.setUpClass.__func__(cls)
    def worker(self,path):
        import torch,copy
        from src.tasks.velocity.v02_formal.spec import contract,sha256
        from test_formal_telemetry import train_manifest
        target=Path(path)/'worker';(target/'checkpoints').mkdir(parents=True);train_manifest(target)
        state=copy.deepcopy(self.learned);state['contract']=contract();checkpoint=target/'checkpoints'/'monitoring_blocked.pt';torch.save(state,checkpoint)
        result={'status':'monitoring_blocked','checkpoint':str(checkpoint),'sha256':sha256(checkpoint),'curriculum':state['curriculum'],'global_iteration':50,'env_steps':1228800}
        (target/'budget.json').write_text(json.dumps({'curriculum':state['curriculum']}));(target/'updates.jsonl').write_text(json.dumps({'global_iteration':50,'stage_iteration':50,'stage':'R0','env_steps':1228800,'num_envs':512})+'\n')
        (target/'resources.json').write_text(json.dumps({'error':None,'sampling_interval_s':.02,'samples':[{'gpu_used_mib':1000}]}))
        return target,result
    def test_live_adopt_restart_force_and_close_keep50_after_queue_cleared(self):
        import subprocess,time,psutil
        from src.tasks.velocity.v02_formal.spec import behavior_hashes
        from src.tasks.velocity.v02_formal.telemetry import Telemetry
        from src.tasks.velocity.v02_formal.orchestration import Supervisor
        from test_formal_telemetry import Clock,Backend
        with tempfile.TemporaryDirectory() as path:
            target,result=self.worker(path);release=Path(path)/'sampled';source='import time,json;from pathlib import Path;deadline=time.monotonic()+10\nwhile not Path('+repr(str(release))+').exists():\n if time.monotonic()>deadline:raise RuntimeError("supervisor never adopted live worker")\n time.sleep(.01)\nPath('+repr(str(target/'result.json'))+').write_text('+repr(json.dumps(result))+')'
            command=[sys.executable,'-c',source,'--execution','train'];process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            record={'pid':process.pid,'pid_start_time':psutil.Process(process.pid).create_time(),'command':command,'started':time.time(),'source_hashes':behavior_hashes()}
            (target.parent/(target.name+'_process.json')).write_text(json.dumps(record));(target.parent/(target.name+'.log')).write_text('')
            clock=Clock();backend=Backend();monitor=Telemetry(Path(path)/'telemetry',backend,{},clock=clock,wall_clock=clock);monitor.start();clock.advance(31)
            start={'global_iteration':0,'stage_iteration':0,'env_steps':0,'num_envs':512,'curriculum/stage':'R0'}
            try:
                def sample(*args):release.touch();return {'system/gpu_used_mib':1000}
                with patch('src.tasks.velocity.v02_formal.orchestration.resource_snapshot',side_effect=sample):r=child(command,target,telemetry=monitor,progress=start,gpu_query=lambda:set())
            finally:process.wait(timeout=10)
            relevant=[event for event in monitor.events if event['kind'] in ('train','heartbeat','resource_close')]
            self.assertTrue(any(event['kind']=='train' for event in relevant))
            self.assertTrue(any(event['kind']=='resource_close' for event in relevant))
            for event in relevant:
                self.assertEqual([event['metrics'][key] for key in ('global_iteration','env_steps','stage_iteration')],[50,1228800,50])
            supervisor=Supervisor.__new__(Supervisor);supervisor.state={'num_envs':512,'consumption_ledger':[]};supervisor.update_result(r);monitor.emit_once('checkpoint',r['sha256'],supervisor.progress())
            self.assertEqual(monitor.events[-1]['metrics']['global_iteration'],50)
            resumed=Telemetry(Path(path)/'telemetry',backend,{},clock=clock,wall_clock=clock);resumed.restore_pending();clock.advance(31)
            child(command,target,telemetry=resumed,progress=start,gpu_query=lambda:set())
            self.assertEqual(resumed.events[-1]['kind'],'heartbeat');self.assertEqual(resumed.events[-1]['metrics']['global_iteration'],50);self.assertEqual(resumed.worker_progress(target,start)['env_steps'],1228800)

if __name__=='__main__':unittest.main()
