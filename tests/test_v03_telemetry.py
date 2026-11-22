"""CPU-only v03 telemetry and frozen delivery/clock regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.tasks.velocity.v03.telemetry import Telemetry, MonitoringBlocked, aggregate_updates, evaluation_metrics, finite_metrics, WandbBackend, PANELS


def train_manifest(directory):
    from src.tasks.velocity.v03.spec import TASK,PROTOCOL_ID,CODE_VERSION,INIT_SHA256,contract
    directory=Path(directory)
    (directory/'manifest.json').write_text(json.dumps(dict(task=TASK,protocol_id=PROTOCOL_ID,code_version=CODE_VERSION,source_c_sha256=INIT_SHA256,seed=42,execution='train',num_envs=512,contract=contract(),arguments={'execution':'train','output_dir':str(directory.resolve()),'num_envs':512})))

def train_episode(directory,**values):
    from src.tasks.velocity.v03.spec import TASK,PROTOCOL_ID,CODE_VERSION,INIT_SHA256
    return dict(task=TASK,protocol_id=PROTOCOL_ID,code_version=CODE_VERSION,source_c_sha256=INIT_SHA256,seed=42,worker_run=str(Path(directory).resolve()),**values)

class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def advance(self,value):self.value+=value

class Backend:
    def __init__(self):self.remote=set();self.sent=[];self.down=False;self.auth_failure=False;self.hidden=False;self.starts=[]
    def start(self,run_id,config):
        if self.auth_failure:raise PermissionError('denied')
        self.starts.append(run_id);return dict(project='p',entity='e',run_id=run_id,url='https://example.invalid/run/'+run_id)
    def remote_ids(self):
        if self.down:raise ConnectionError('offline')
        return set() if self.hidden else self.remote.copy()
    def send(self,event):
        if self.down:raise ConnectionError('offline')
        self.sent.append(event);self.remote.add(event['id'])
    def verify_start(self,run_id,config):return True
    def ensure_panels(self,existing_url=None):
        self.panel_verification=dict(remote_verified=True,remote_group_count=8)
        return 'https://example.invalid/report'
    def finish(self,summary):pass

class TelemetryTests(unittest.TestCase):
    def setUp(self):
        sleeper=patch("time.sleep", return_value=None);sleeper.start();self.addCleanup(sleeper.stop)
    def setup_monitor(self,path):
        clock=Clock();backend=Backend();pauses=[]
        monitor=Telemetry(path,backend,{'seed':42},clock=clock,pause=pauses.append);monitor.start()
        return monitor,backend,clock,pauses
    def test_auth_failure_prevents_initialization(self):
        with tempfile.TemporaryDirectory() as path:
            backend=Backend();backend.auth_failure=True
            monitor=Telemetry(path,backend,{})
            with self.assertRaises(PermissionError):monitor.start()
            self.assertEqual(backend.sent,[])
    def test_sdk_success_without_remote_readback_blocks_start(self):
        with tempfile.TemporaryDirectory() as path:
            backend=Backend();backend.hidden=True
            with self.assertRaises(MonitoringBlocked):Telemetry(path,backend,{}).start()
    def test_120_seconds_outage_requests_boundary_pause_and_recovery_same_id(self):
        with tempfile.TemporaryDirectory() as path:
            m,b,c,pauses=self.setup_monitor(path);run_id=m.state['run_id'];b.down=True
            m.emit('train',{'global_iteration':10});c.advance(119);m.tick({'global_iteration':10})
            self.assertEqual(pauses,[])
            c.advance(1);m.tick({'global_iteration':10});self.assertEqual(pauses,['monitoring_blocked'])
            self.assertTrue(m.state['monitoring_blocked']);b.down=False
            resumed=Telemetry(path,b,{'seed':42},clock=c);identity=resumed.start()
            self.assertEqual(identity['run_id'],run_id);self.assertNotIn('monitoring_blocked',resumed.state)
            self.assertEqual(len({event['id'] for event in b.sent}),len(b.sent))
    def test_resume_reconciles_uncertain_send_no_duplicate(self):
        with tempfile.TemporaryDirectory() as path:
            m,b,c,_=self.setup_monitor(path)
            event=m.emit('train',{'global_iteration':10},flush=False);b.send(event)
            resumed=Telemetry(path,b,{'seed':42},clock=c);resumed.start()
            self.assertEqual(sum(e['id']==event['id'] for e in b.sent),1)
            self.assertIn(event['id'],resumed.state['acked'])
    def test_live_tail_ten_update_and_30_second_cadence_and_evaluation_immediate(self):
        with tempfile.TemporaryDirectory() as path:
            m,b,c,_=self.setup_monitor(Path(path)/'telemetry');worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            with (worker/'updates.jsonl').open('w') as f:
                for i in range(1,10):f.write(json.dumps(dict(global_iteration=i,stage_iteration=i,num_envs=512,env_steps=i*512*48,policy_loss=float(i)))+'\n')
            m.read_worker(worker);m.tick({'global_iteration':0});self.assertEqual(len(b.sent),1)
            with (worker/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(global_iteration=10,stage_iteration=10,num_envs=512,env_steps=10*512*48,policy_loss=10.))+'\n')
            m.read_worker(worker);m.tick({'global_iteration':0});self.assertEqual(b.sent[-1]['metrics']['global_iteration'],10)
            self.assertEqual(b.sent[-1]['metrics']['ppo/policy_loss'],5.5)
            with (worker/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(global_iteration=11,stage_iteration=11,num_envs=512,env_steps=11*512*48))+'\n')
            m.read_worker(worker);c.advance(30);m.tick({'global_iteration':11})
            self.assertTrue(any(e['kind']=='train' and e['metrics']['global_iteration']==11 for e in b.sent))
            before=len(b.sent);m.evaluation(dict(stage='S1',kind='validation',iteration=11,stage_iteration=11,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[dict(vx=.2,episodes=20,successes=18)],passed=True))
            self.assertEqual(len(b.sent),before+1);self.assertEqual(b.sent[-1]['kind'],'evaluation')
    def test_pending_cursor_is_durable_and_resumed_once(self):
        with tempfile.TemporaryDirectory() as path:
            m,b,c,_=self.setup_monitor(Path(path)/'monitor');worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            (worker/'updates.jsonl').write_text(json.dumps({'global_iteration':1,'num_envs':512,'env_steps':24576})+'\n'+ '{"incomplete":')
            m.read_worker(worker)
            resumed=Telemetry(Path(path)/'monitor',b,{'seed':42},clock=c);resumed.restore_pending();resumed.read_worker(worker);resumed.tick({},force=True)
            self.assertEqual(b.sent[-1]['metrics']['train/window_updates'],1)
            again=Telemetry(Path(path)/'monitor',b,{'seed':42},clock=c);again.restore_pending();again.read_worker(worker)
            self.assertEqual(again.pending_updates,[])
    def test_actual_steps_and_weighting_after_environment_change_missing_not_zero(self):
        rows=[dict(global_iteration=1,num_envs=512,env_steps=24576,policy_loss=1.,motion_terms={'groups':{'current_stairs/moving':{'sample_count':10,'actual_vx_body':.2}}}),
              dict(global_iteration=2,num_envs=258,env_steps=36960,policy_loss=3.,motion_terms={'groups':{'current_stairs/moving':{'sample_count':30,'actual_vx_body':.6}}})]
        metrics=aggregate_updates(rows,[])
        self.assertEqual(metrics['env_steps'],36960)
        self.assertAlmostEqual(metrics['ppo/policy_loss'],(512+3*258)/770)
        self.assertAlmostEqual(metrics['motion/groups/current_stairs/moving/actual_vx_body'],.5)
        self.assertFalse(any('climb_success_rate' in key for key in metrics))
        self.assertEqual(finite_metrics({'missing':None,'nan':float('nan'),'true_zero':0}),{'true_zero':0})
    def test_reward_episode_and_control_sample_denominators(self):
        rows=[dict(slot_class='current_stairs',climb_success=True,duration=1.,reward_sums={'reward_total':10.}),
              dict(slot_class='current_stairs',climb_success=False,duration=3.,reward_sums={'reward_total':6.})]
        metrics=aggregate_updates([],rows)
        self.assertEqual(metrics['episode/current_stairs/climb_success_rate'],.5)
        self.assertEqual(metrics['reward/episode_mean/current_stairs/reward_total'],8.)
        self.assertEqual(metrics['reward/per_control_sample/current_stairs/reward_total'],16/200)

class StatisticsRegressionTests(unittest.TestCase):
    def test_maxima_counts_event_sentinels_and_actual_episode_return(self):
        rows=[dict(num_envs=512,global_iteration=1,env_steps=24576,motion_terms=dict(sample_count=100,peak_torque_max=12.,groups={'current_stairs/moving':dict(sample_count=3,lateral_max=.2,mean_env_peak_torque=5.)})),
              dict(num_envs=258,global_iteration=2,env_steps=36960,motion_terms=dict(sample_count=50,peak_torque_max=30.,groups={'current_stairs/moving':dict(sample_count=1,lateral_max=.5,mean_env_peak_torque=9.)}))]
        episodes=[dict(slot_class='current_stairs',climb_success=False,duration=1.,first_step_support_time=-1.,reward_sums={'reward_total':4.}),
                  dict(slot_class='current_stairs',climb_success=True,duration=2.,first_step_support_time=.5,reward_sums={'reward_total':8.})]
        result=aggregate_updates(rows,episodes)
        self.assertEqual(result['motion/sample_count'],150)
        self.assertEqual(result['motion/peak_torque_max'],30.)
        self.assertEqual(result['motion/groups/current_stairs/moving/sample_count'],4)
        self.assertEqual(result['motion/groups/current_stairs/moving/lateral_max'],.5)
        self.assertEqual(result['motion/groups/current_stairs/moving/mean_env_peak_torque'],6.)
        self.assertEqual(result['episode/current_stairs/return'],6.)
        self.assertEqual(result['episode/current_stairs/first_step_support_time'],.5)
        self.assertEqual(result['episode/current_stairs/first_step_support_time_event_episodes'],1)
    def test_repeated_evaluation_identity_emits_once(self):
        with tempfile.TemporaryDirectory() as path:
            backend=Backend();monitor=Telemetry(path,backend,{});monitor.start()
            summary=dict(stage='S1',kind='validation',iteration=200,stage_iteration=200,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[dict(vx=.2,episodes=20,successes=18)])
            monitor.evaluation(summary);monitor.evaluation(summary)
            self.assertEqual(len([e for e in backend.sent if e['kind']=='evaluation']),1)

class ExtremumTests(unittest.TestCase):
    def test_std_min_and_max_are_window_extrema_and_event_counts_actual_totals(self):
        result=aggregate_updates([dict(num_envs=512,std_min=.08,std_max=.20,reward_sample_means={'reward_event/footfall_count':.25}),
                                  dict(num_envs=258,std_min=.05,std_max=.30,reward_sample_means={'reward_event/footfall_count':.50})],[])
        self.assertEqual(result['ppo/std_min'],.05);self.assertEqual(result['ppo/std_max'],.30)
        self.assertEqual(result['reward/window_total/reward_event/footfall_count'],.25*512*48+.5*258*48)

class SDKQueue:
    """Local sender callback with real monotonic history/commit constraints."""
    def __init__(self):self.calls=[];self.records=[];self.remote=[];self.hold=set();self.sender_step=0
    def callback(self,data,step,commit):
        self.calls.append((step,commit))
        if step<self.sender_step:return
        if commit:
            self.records.append(dict(data));self.sender_step=step+1
            if step not in self.hold:self.remote.append(dict(data))
    def deliver(self):
        for row in self.records:
            if not any(previous['event_id']==row['event_id'] for previous in self.remote):self.remote.append(dict(row))
        self.hold.clear()

class SDKLocalRun:
    """Invoke installed wandb.Run._log; no init/network/service is created."""
    def __init__(self,queue,step=0):
        import os
        self.queue=queue;self._local_step=step;self._init_pid=os.getpid();self._is_attached=False
    @property
    def step(self):return self._local_step
    def _partial_history_callback(self,data,step,commit):self.queue.callback(data,step,commit)
    def log(self,data,step=None,commit=None):
        from wandb.sdk.wandb_run import Run
        return Run._log(self,data,step,commit)

class SDKBackend(Backend):
    def __init__(self,queue,step=0,spool=None):
        from types import SimpleNamespace
        from src.tasks.velocity.v03.telemetry import WandbBackend
        super().__init__();self.queue=queue;self.sdk=WandbBackend.__new__(WandbBackend)
        self.sdk.run=SDKLocalRun(queue,step);self.sdk.wandb=SimpleNamespace();self.crash=False;self.spool=spool
    def start(self,run_id,config):
        result=super().start(run_id,config)
        if self.spool:result.update(sdk_spool_directory=str(self.spool),sdk_spool_files=[str(self.spool/'run.wandb')],sdk_owner_pid=999999,sdk_owner_start_time=0.)
        return result
    def remote_ids(self):return {row['event_id'] for row in self.queue.remote}
    def next_transport_step(self):return self.sdk.next_transport_step()
    def send(self,event):
        self.sdk.send(event);self.sent.append(event)
        if self.crash:self.crash=False;raise KeyboardInterrupt('simulated crash after SDK enqueue before local acknowledgement')

class DeliveryFix1Tests(unittest.TestCase):
    def setUp(self):
        sleeper=patch("time.sleep", return_value=None);sleeper.start();self.addCleanup(sleeper.stop)
    def test_t1_installed_sdk_commits_initialize_and_immediate_eval(self):
        with tempfile.TemporaryDirectory() as path:
            queue=SDKQueue();backend=SDKBackend(queue,step=7);monitor=Telemetry(path,backend,{})
            monitor.start();self.assertEqual(queue.calls,[(7,True)]);self.assertEqual(backend.next_transport_step(),8)
            monitor.emit('evaluation',{'global_iteration':200,'eval/climb/successes':18})
            self.assertEqual(queue.calls[-1],(8,True));self.assertEqual(len(queue.remote),2)
            self.assertEqual(queue.remote[-1]['event_sequence'],2)
            self.assertEqual(queue.remote[-1]['transport_step'],8)
    def test_t2_uncertain_old_event_is_barrier_same_sdk_and_same_run_restart(self):
        with tempfile.TemporaryDirectory() as path:
            queue=SDKQueue();backend=SDKBackend(queue);monitor=Telemetry(path,backend,{})
            monitor.start();run_id=monitor.state['run_id'];queue.hold.add(1)
            old=monitor.emit('train',{'global_iteration':10});later=monitor.emit('evaluation',{'global_iteration':10})
            for _ in range(6):self.assertFalse(monitor.flush())
            self.assertEqual(queue.calls,[(0,True),(1,True)])
            self.assertNotIn(later['id'],backend.remote_ids())
            resumed_backend=SDKBackend(queue,step=2);resumed=Telemetry(path,resumed_backend,{})
            with self.assertRaises(MonitoringBlocked):resumed.start()
            self.assertEqual(resumed.state['run_id'],run_id);self.assertTrue(resumed.state['queue_recovery_required'])
            self.assertEqual(queue.calls,[(0,True),(1,True)])
            queue.deliver();resumed.start()
            self.assertEqual(queue.calls,[(0,True),(1,True),(2,True)])
            self.assertEqual([row['event_id'] for row in queue.remote],[monitor.events[0]['id'],old['id'],later['id']])
            self.assertFalse(set(e['id'] for e in resumed.events)-set(resumed.state['acked']))
    def test_t2_crash_after_send_intent_uses_original_spool_sync_and_readback(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as path:
            spool=Path(path)/'sdk_run';spool.mkdir();(spool/'run.wandb').write_bytes(b'synthetic durable SDK stream')
            queue=SDKQueue();backend=SDKBackend(queue,spool=spool);monitor=Telemetry(Path(path)/'telemetry',backend,{})
            monitor.start();queue.hold.add(1);backend.crash=True
            with self.assertRaises(KeyboardInterrupt):monitor.emit('train',{'global_iteration':10})
            pending=monitor.events[-1];self.assertEqual(monitor.state['transport'][pending['id']]['state'],'intent')
            self.assertEqual(backend.next_transport_step(),2)
            resumed_backend=SDKBackend(queue,step=2);resumed=Telemetry(Path(path)/'telemetry',resumed_backend,{})
            with self.assertRaises(MonitoringBlocked):resumed.start()
            captured=[]
            def sync(command,**kwargs):
                captured.append(command);queue.deliver();return SimpleNamespace(returncode=0)
            self.assertTrue(resumed.sync_spool(spool,execute=sync))
            self.assertEqual(captured[0][-1],str(spool.resolve()));self.assertIn('--include-online',captured[0])
            self.assertEqual(captured[0][captured[0].index('--id')+1],resumed.state['run_id'])
            self.assertIn(pending['id'],resumed.state['acked']);self.assertEqual(len(queue.remote),2)
            self.assertEqual(queue.calls,[(0,True),(1,True)])
            with self.assertRaises(ValueError):resumed.sync_spool(Path(path)/'unowned',execute=sync)
    def test_t2_sync_exit_success_without_remote_evidence_does_not_clear_barrier(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as path:
            spool=Path(path)/'sdk';spool.mkdir();(spool/'run.wandb').write_bytes(b'stream')
            queue=SDKQueue();backend=SDKBackend(queue,spool=spool);monitor=Telemetry(Path(path)/'telemetry',backend,{})
            monitor.start();queue.hold.add(1);monitor.emit('train',{'global_iteration':10})
            self.assertFalse(monitor.sync_spool(spool,execute=lambda *args,**kwargs:SimpleNamespace(returncode=0)))
            self.assertIn('delivery_barrier',monitor.state);self.assertEqual(len(queue.remote),1)

class BarrierDeadlineFix1Tests(unittest.TestCase):
    def test_unconfirmed_sdk_delivery_obeys_120_second_pause_without_overtaking(self):
        with tempfile.TemporaryDirectory() as path:
            queue=SDKQueue();backend=SDKBackend(queue);clock=Clock();pauses=[]
            monitor=Telemetry(path,backend,{},clock=clock,pause=pauses.append);monitor.start()
            queue.hold.add(1);monitor.emit('train',{'global_iteration':10});monitor.emit('evaluation',{'global_iteration':10})
            clock.advance(120);monitor.tick({'global_iteration':10})
            self.assertEqual(pauses,['monitoring_blocked']);self.assertEqual(queue.calls,[(0,True),(1,True)])
            queue.deliver();self.assertTrue(monitor.flush());self.assertNotIn('monitoring_blocked',monitor.state)
            self.assertEqual([step for step,_ in queue.calls],list(range(len(queue.calls))))
            self.assertEqual(len(queue.remote),len({row['event_id'] for row in queue.remote}))
    def test_sync_cannot_compete_with_live_original_sdk_owner(self):
        import os
        import psutil
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as path:
            spool=Path(path)/'sdk';spool.mkdir();(spool/'run.wandb').write_bytes(b'stream')
            queue=SDKQueue();backend=SDKBackend(queue,spool=spool);monitor=Telemetry(Path(path)/'telemetry',backend,{});monitor.start()
            record=monitor.state['sdk_spools'][str(spool)];record['sdk_owner_pid']=os.getpid();record['sdk_owner_start_time']=psutil.Process().create_time()
            execute=Mock()
            with self.assertRaisesRegex(MonitoringBlocked,'owner is alive'):monitor.sync_spool(spool,execute=execute)
            execute.assert_not_called()

class InitializationReadbackFix1Tests(unittest.TestCase):
    def setUp(self):
        sleeper=patch("time.sleep", return_value=None);sleeper.start();self.addCleanup(sleeper.stop)
    def test_all_acked_local_ledger_is_not_online_proof_on_restart(self):
        with tempfile.TemporaryDirectory() as path:
            backend=Backend();monitor=Telemetry(path,backend,{});monitor.start();sent=len(backend.sent)
            backend.hidden=True;resumed=Telemetry(path,backend,{})
            with self.assertRaises(MonitoringBlocked):resumed.start()
            self.assertEqual(resumed.state['online_status'],'unconfirmed');self.assertEqual(len(backend.sent),sent)

class OnlineIsolationTests(unittest.TestCase):
    def test_eval_flat_parking_probe_initialize_do_not_enter_training_window(self):
        with tempfile.TemporaryDirectory() as path:
            monitor,backend,clock,_=TelemetryTests().setup_monitor(Path(path)/'telemetry')
            for kind in ('validation','flat','parking','probe','initialize'):
                directory=Path(path)/kind;directory.mkdir()
                (directory/'manifest.json').write_text(json.dumps({'execution':kind,'kind':kind}))
                record=dict(kind=kind,case_id='diagnostic:'+kind,slot_class='current_stairs',climb_success=True,duration=2.,reward_sums={'reward_total':100.},parking_diagnostic=kind=='parking')
                (directory/'episodes.jsonl').write_text(json.dumps(record)+'\n')
                (directory/'updates.jsonl').write_text(json.dumps({'probe':kind=='probe','global_iteration':1,'num_envs':512})+'\n')
                monitor.read_worker(directory)
            self.assertEqual(monitor.pending_episodes,[]);self.assertEqual(monitor.pending_updates,[])
            worker=Path(path)/'train';worker.mkdir();train_manifest(worker)
            episode=train_episode(worker,slot_class='current_stairs',climb_success=False,duration=2.,reward_sums={'reward_total':0.})
            (worker/'episodes.jsonl').write_text(json.dumps(episode)+'\n');monitor.read_worker(worker);monitor.tick({},force=True)
            self.assertEqual(len(monitor.pending_episodes),1);self.assertFalse(any(event['kind']=='train' for event in backend.sent))
            (worker/'updates.jsonl').write_text(json.dumps(dict(global_iteration=1,num_envs=512,env_steps=24576))+'\n');monitor.read_worker(worker);monitor.tick({},force=True)
            event=next(e for e in backend.sent if e['kind']=='train');metrics=event['metrics']
            self.assertEqual(metrics['episode/current_stairs/denominator'],1);self.assertEqual(metrics['episode/current_stairs/climb_success_rate'],0.)
            self.assertEqual(metrics['episode/current_stairs/return'],0.)
            self.assertTrue(all(str(worker) in source for source in event['source_ids']))
            audit=[json.loads(line) for line in (Path(path)/'telemetry'/'quarantine.jsonl').read_text().splitlines()]
            self.assertEqual(len(audit),10);self.assertEqual({row['record'].get('kind') for row in audit if row['record'].get('case_id')},{'validation','flat','parking','probe','initialize'})
    def test_valid_train_manifest_cannot_authorize_a_diagnostic_record(self):
        with tempfile.TemporaryDirectory() as path:
            monitor,backend,clock,_=TelemetryTests().setup_monitor(Path(path)/'telemetry');worker=Path(path)/'train';worker.mkdir();train_manifest(worker)
            evaluation=train_episode(worker,kind='validation',case_id='frozen-case',slot_class='current_stairs',climb_success=True,duration=2.,reward_sums={'reward_total':100.})
            (worker/'episodes.jsonl').write_text(json.dumps(evaluation)+'\n');monitor.read_worker(worker)
            self.assertEqual(monitor.pending_episodes,[]);self.assertEqual(len(monitor.state['quarantine_ids']),1)
    def test_restore_quarantines_old_pending_but_keeps_true_train_and_audits_source_ids(self):
        with tempfile.TemporaryDirectory() as path:
            monitor,backend,clock,_=TelemetryTests().setup_monitor(path)
            evaluation=dict(kind='flat',case_id='old-eval',slot_class='flat_walk',climb_success=True,duration=2.,reward_sums={'reward_total':100.},_telemetry_source_id='old_eval/episodes.jsonl:0')
            training=dict(worker_run='old_train',slot_class='current_stairs',climb_success=False,duration=2.,reward_sums={'reward_total':0.},_telemetry_source_id='old_train/episodes.jsonl:0')
            monitor.state['pending_episodes']=[evaluation,training];monitor.state['pending_updates']=[dict(global_iteration=1,num_envs=512,env_steps=24576,_telemetry_source_id='old_train/updates.jsonl:0')];monitor._save()
            resumed=Telemetry(path,backend,{'seed':42},clock=clock);resumed.restore_pending();resumed.restore_pending();resumed.tick({},force=True)
            event=next(e for e in backend.sent if e['kind']=='train')
            self.assertEqual(event['metrics']['episode/current_stairs/denominator'],1);self.assertNotIn('episode/flat_walk/denominator',event['metrics'])
            self.assertNotIn(evaluation['_telemetry_source_id'],event['source_ids']);self.assertIn(training['_telemetry_source_id'],event['source_ids'])
            audit=[json.loads(line) for line in (Path(path)/'quarantine.jsonl').read_text().splitlines()]
            self.assertEqual(len(audit),1);self.assertEqual(audit[0]['record'],evaluation)
    def test_pure_aggregate_defensively_excludes_all_diagnostic_identities(self):
        training=dict(slot_class='current_stairs',climb_success=False,duration=2.,reward_sums={'reward_total':0.})
        rejected=[dict(training,kind='validation',climb_success=True),dict(training,parking_diagnostic=True),dict(training,probe=True),dict(training,execution='initialize'),dict(training,case_id='case')]
        metrics=aggregate_updates([dict(num_envs=512,global_iteration=1),dict(num_envs=1024,global_iteration=0,probe=True)],[training]+rejected)
        self.assertEqual(metrics['episode/current_stairs/denominator'],1);self.assertEqual(metrics['episode/current_stairs/climb_success_rate'],0.)
        self.assertEqual(metrics['train/window_updates'],1);self.assertEqual(metrics['train/window_samples'],512*48)

class MonitoringClockTests(unittest.TestCase):
    def monitor(self,path):
        clock=Clock();backend=Backend();pauses=[]
        t=Telemetry(path,backend,{'seed':42},clock=clock,wall_clock=clock,pause=pauses.append);t.start();return t,backend,clock,pauses
    def progress(self,i,stage='S1',stage_i=None):return dict(global_iteration=i,stage_iteration=i if stage_i is None else stage_i,env_steps=i*512*48,num_envs=512,**{'curriculum/stage':stage})
    def test_train40_empty_pending_heartbeat_and_restart_use_trusted40_then41(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,_=self.monitor(Path(path)/'telemetry');worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            (worker/'updates.jsonl').write_text(json.dumps(dict(self.progress(40),stage='S1'))+'\n');t.read_worker(worker)
            t.tick(t.worker_progress(worker,self.progress(0)),force=True);self.assertEqual(t.pending_updates,[])
            c.advance(31);t.tick(t.worker_progress(worker,self.progress(0)))
            heartbeat=next(event for event in b.sent if event['kind']=='heartbeat');self.assertEqual(heartbeat['metrics']['global_iteration'],40);self.assertEqual(heartbeat['metrics']['env_steps'],983040)
            resumed=Telemetry(Path(path)/'telemetry',b,{'seed':42},clock=c,wall_clock=c);resumed.restore_pending()
            self.assertEqual(resumed.worker_progress(worker,self.progress(0))['global_iteration'],40)
            # Rebuild from durable train event even if the optional cache was lost.
            resumed.state.pop('worker_progress',None);resumed._save();rebuilt=resumed.worker_progress(worker,self.progress(0));self.assertEqual(rebuilt['global_iteration'],40)
            with (worker/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(self.progress(41),stage='S1'))+'\n')
            resumed.read_worker(worker);resumed.tick(resumed.worker_progress(worker,self.progress(0)),force=True)
            self.assertEqual(b.sent[-1]['metrics']['global_iteration'],41)
    
    def test_unconfirmed_network_does_not_turn_50_updates_into_1_update_windows(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,pauses=self.monitor(Path(path)/'telemetry');b.down=True;worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            for i in range(1,51):
                c.advance(3)
                with (worker/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(self.progress(i),stage='S1'))+'\n')
                with (worker/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(train_episode(worker,slot_class='current_stairs',climb_success=False,duration=1.,reward_sums={'reward_total':float(i)}))+'\n')
                t.read_worker(worker);t.tick(t.worker_progress(worker,self.progress(0)))
            trains=[event for event in t.events if event['kind']=='train'];self.assertEqual([event['metrics']['train/window_updates'] for event in trains],[10]*5)
            self.assertEqual(sum(event['metrics']['episode/current_stairs/denominator'] for event in trains),50)
            ids=[source for event in trains for source in event['source_ids']];self.assertEqual(len(ids),100);self.assertEqual(len(set(ids)),100)
            self.assertFalse(any(event['kind']=='heartbeat' for event in t.events));self.assertEqual(pauses,['monitoring_blocked']);self.assertEqual(t.state['online_status'],'unconfirmed')
            self.assertEqual(t.state['last_aggregate_wall'],150);self.assertIsNotNone(t.state.get('failure_since_wall'))
    def test_window_clock_persists_and_idle_eval_heartbeat_and_immediate_eval_remain(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,_=self.monitor(Path(path)/'telemetry');worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            for i in range(1,4):
                c.advance(3)
                with (worker/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(self.progress(i),stage='S1'))+'\n')
                t.read_worker(worker);t.tick(t.worker_progress(worker,self.progress(0)))
            self.assertEqual(t.state['window_started_wall'],3.)
            c.advance(20);resumed=Telemetry(Path(path)/'telemetry',b,{'seed':42},clock=c,wall_clock=c);resumed.restore_pending();resumed.tick(resumed.worker_progress(worker,self.progress(0)))
            self.assertFalse(any(event['kind']=='train' for event in resumed.events))
            c.advance(1);resumed.tick(resumed.worker_progress(worker,self.progress(0)));self.assertEqual(b.sent[-1]['kind'],'train');self.assertEqual(b.sent[-1]['metrics']['train/window_updates'],3)
            c.advance(30);resumed.tick(self.progress(3));self.assertEqual(b.sent[-1]['kind'],'heartbeat')
            resumed.evaluation(dict(stage='S1',kind='validation',iteration=3,stage_iteration=3,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[]))
            self.assertEqual(b.sent[-1]['kind'],'evaluation')
    def test_slow_updates_use_30_second_local_window_instead_of_a_redundant_heartbeat(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,_=self.monitor(path);b.down=True
            for i in range(1,10):
                c.advance(3.229);t.pending_updates.append(self.progress(i));t.tick(self.progress(i))
            self.assertFalse(any(e['kind']=='train' for e in t.events));c.advance(30-c())
            t.tick(self.progress(9));trains=[e for e in t.events if e['kind']=='train']
            self.assertEqual(len(trains),1);self.assertEqual(trains[0]['metrics']['train/window_updates'],9)
            self.assertFalse(any(e['kind']=='heartbeat' for e in t.events))
    def test_partial_ack_and_local_emit_do_not_clear_120_second_outage(self):
        with tempfile.TemporaryDirectory() as path:
            queue=SDKQueue();backend=SDKBackend(queue);clock=Clock();pauses=[]
            t=Telemetry(path,backend,{},clock=clock,wall_clock=clock,pause=pauses.append);t.start();clock.advance(30);queue.hold.add(1)
            t.emit('train',self.progress(10));t.emit('train',self.progress(20));queue.deliver();queue.hold.add(2)
            self.assertFalse(t.flush());self.assertEqual(len(t.state['acked']),2)
            clock.advance(120);t.tick(self.progress(20));self.assertEqual(pauses,['monitoring_blocked']);self.assertEqual(t.state['online_status'],'unconfirmed');self.assertIn('delivery_barrier',t.state)

class V03AdaptationTests(unittest.TestCase):
    def setUp(self):
        sleeper=patch('time.sleep',return_value=None);sleeper.start();self.addCleanup(sleeper.stop)
    def test_closed_preflight_session_is_nonterminal_durable_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            clock=Clock();backend=Backend();closed=[]
            backend.finish=lambda summary:closed.append(dict(summary))
            t=Telemetry(directory,backend,{'seed':42},clock=clock,wall_clock=clock);t.start()
            self.assertTrue(hasattr(t,'close_session'),'v03 requires a nonterminal ACK-verified SDK close')
            receipt=t.close_session({'global_iteration':0,'system/preflight_ready':1})
            self.assertEqual(receipt['status'],'remote_verified');self.assertEqual(receipt['sdk_finish'],'completed')
            self.assertEqual(receipt['run_id'],t.state['run_id']);self.assertIn(receipt['event_id'],t.state['acked'])
            self.assertFalse(any(e['kind']=='stop' for e in t.events));self.assertEqual(len(closed),1)
            self.assertEqual(t.close_session({'global_iteration':0,'system/preflight_ready':1}),receipt)
            self.assertEqual(len(closed),1)
            persisted=json.loads((Path(directory)/'state.json').read_text())
            self.assertEqual(persisted['session_receipts'][-1],receipt)
            resumed=Telemetry(directory,backend,{'seed':42},clock=clock,wall_clock=clock);resumed.start()
            self.assertEqual(resumed.state['run_id'],receipt['run_id']);self.assertEqual(len([e for e in resumed.events if e['kind']=='initialize']),1)
    def test_restarting_same_object_closes_each_sdk_session_once(self):
        with tempfile.TemporaryDirectory() as directory:
            t,b,clock,_=TelemetryTests().setup_monitor(directory);closed=[];b.finish=lambda summary:closed.append(summary)
            first=t.close_session({'global_iteration':0});t.start();second=t.close_session({'global_iteration':0})
            self.assertEqual(first['run_id'],second['run_id']);self.assertNotEqual(first['event_id'],second['event_id'])
            self.assertEqual(len(closed),2);self.assertEqual(len(t.state['session_receipts']),2)
    def test_close_session_does_not_finish_unconfirmed_queue_or_fabricate_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            t,b,clock,_=TelemetryTests().setup_monitor(directory);closed=[];b.finish=lambda summary:closed.append(summary)
            self.assertTrue(hasattr(t,'close_session'),'v03 requires nonterminal close')
            b.hidden=True;receipt=t.close_session({'global_iteration':0})
            self.assertEqual(receipt['status'],'unconfirmed');self.assertEqual(receipt['sdk_finish'],'not_requested');self.assertEqual(closed,[])
            self.assertIn('delivery_barrier',t.state);b.hidden=False
            receipt=t.close_session({'global_iteration':0});self.assertEqual(receipt['status'],'remote_verified');self.assertEqual(len(closed),1)
    def test_sdk_close_failure_stays_unconfirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            t,b,clock,_=TelemetryTests().setup_monitor(directory)
            self.assertTrue(hasattr(t,'close_session'),'v03 requires nonterminal close')
            def fail(summary):raise ConnectionError('SDK close unconfirmed')
            b.finish=fail;receipt=t.close_session({'global_iteration':0})
            self.assertEqual(receipt['status'],'unconfirmed');self.assertEqual(receipt['sdk_finish'],'failed')
            self.assertIn(receipt['event_id'],t.state['acked'])
    def test_segment_metrics_have_episode_and_phase_sample_denominators(self):
        rows=[dict(slot_class='current_stairs',climb_success=True,speed_valid=True,speed_pass=True,joint_success=True,
                   speed_source='training_preview',duration=1.,mean_vx=.2,mean_relative_error=.1,smooth_mae_relative=.2,
                   longest_deviation_seconds=.4,phase_sample_counts=[10,25,5],phase_mean_vx=[.1,.2,.3]),
              dict(slot_class='current_stairs',climb_success=True,speed_valid=False,speed_pass=False,joint_success=False,
                   speed_source='training_preview',duration=1.,mean_vx=.6,mean_relative_error=.5,smooth_mae_relative=None,
                   longest_deviation_seconds=None,phase_sample_counts=[30,5,0],phase_mean_vx=[.5,.6,None])]
        result=aggregate_updates([],rows);root='stair_segment/current_stairs'
        self.assertEqual(result['episode/current_stairs/climb_success_rate'],1.)
        self.assertEqual(result['episode/current_stairs/speed_pass_rate'],.5)
        self.assertEqual(result['episode/current_stairs/joint_success_rate'],.5)
        self.assertEqual(result[root+'/denominator'],2);self.assertEqual(result[root+'/valid_count'],1)
        self.assertEqual(result[root+'/valid_ratio'],.5);self.assertEqual(result[root+'/source'],'training_preview')
        self.assertAlmostEqual(result[root+'/mean_vx'],.4);self.assertEqual(result[root+'/mean_vx_episodes'],2)
        self.assertEqual(result[root+'/smooth_mae_relative'],.2);self.assertEqual(result[root+'/smooth_mae_relative_episodes'],1)
        self.assertAlmostEqual(result[root+'/approach/mean_vx'],.4);self.assertEqual(result[root+'/approach/sample_count'],40)
        self.assertAlmostEqual(result[root+'/body/mean_vx'],8/30);self.assertEqual(result[root+'/top/sample_count'],5)
        self.assertFalse(any('joint_success_rate' in k for k in aggregate_updates([],[])))
    def test_old_raw_linear_is_diagnostic_and_not_added_to_total(self):
        result=aggregate_updates([dict(num_envs=258,sample_count=12384,reward_sample_means={'reward_total':2.,'reward/linear_old_raw':100.})],[])
        self.assertEqual(result['reward/reward_total'],2.)
        self.assertEqual(result['diagnostic/reward/linear_old_raw'],100.)
        self.assertNotIn('reward/reward/linear_old_raw',result)
        self.assertEqual(result['reward/window_total/reward_total'],24768.)
    def test_old_stage_and_flat_slot_records_cannot_enter_v03_training(self):
        result=aggregate_updates([dict(stage='S1',num_envs=258),dict(stage='R0',num_envs=258)],
            [dict(slot_class='flat_walk',duration=1.,climb_success=True),dict(slot_class='current_stairs',duration=1.,climb_success=False)])
        self.assertEqual(result['train/window_updates'],1);self.assertNotIn('episode/flat_walk/denominator',result)
        with self.assertRaises(ValueError):evaluation_metrics(dict(stage='flat',kind='validation',iteration=0,stage_iteration=0,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[]))
    def test_evaluation_keeps_counts_identity_and_joint_namespace(self):
        result=evaluation_metrics(dict(stage='S1',kind='validation',iteration=200,stage_iteration=200,env_steps=2476800,
            num_envs=258,checkpoint_sha256='a'*64,case_table_sha256='b'*64,protocol_id='v03-stair-body-speed-v1',
            speeds=[dict(vx=.2,episodes=20,climb_successes=20,speed_valid_count=19,speed_pass_count=18,joint_successes=17)],passed=False))
        root='eval/joint/validation/S1/vx_0.2'
        self.assertEqual(result[root+'/joint_successes'],17);self.assertEqual(result[root+'/episodes'],20)
        self.assertEqual(result['env_steps'],2476800);self.assertEqual(result['eval/protocol_id'],'v03-stair-body-speed-v1')
    def test_backend_init_sends_real_group_and_verifies_remote_config(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            config={'seed':42,'contract':{'protocol_id':'v03-stair-body-speed-v1'}}
            run=SimpleNamespace(settings=SimpleNamespace(mode='online',base_url='https://example.invalid'),project='p',entity='e',id='run',
                url='https://example.invalid/run',dir=str(Path(directory)/'files'),step=0,summary={},define_metric=Mock())
            remote=SimpleNamespace(config=config|{'telemetry_run_id':'run'},group='v03')
            backend=WandbBackend.__new__(WandbBackend);backend.name='v03_seed42_time';backend.project='p';backend.entity='e'
            backend.wandb=SimpleNamespace(init=Mock(return_value=run),Settings=lambda **kwargs:None,Api=lambda **kwargs:SimpleNamespace(run=lambda path:remote))
            backend.start('run',config)
            self.assertEqual(backend.wandb.init.call_args.kwargs.get('group'),'v03')
            self.assertTrue(backend.verify_start('run',config));remote.group='v02';self.assertFalse(backend.verify_start('run',config))
            remote.group='v03';remote.config['seed']=43;self.assertFalse(backend.verify_start('run',config))
    def test_start_requires_config_and_eight_panel_readback(self):
        for failure in ('config','panels'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                backend=Backend();backend.verify_start=lambda *args:failure!='config'
                backend.ensure_panels=lambda existing_url=None:'https://example.invalid/report'
                backend.panel_verification=dict(remote_verified=True,remote_group_count=7 if failure=='panels' else 8)
                with self.assertRaises(MonitoringBlocked):Telemetry(directory,backend,{}).start()
    def test_worker_source_contract_or_directory_mismatch_is_quarantined(self):
        import copy
        for mutation in ('contract','directory','environments'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                t,b,clock,_=TelemetryTests().setup_monitor(Path(directory)/'monitor');w=Path(directory)/'worker';w.mkdir();train_manifest(w)
                manifest=json.loads((w/'manifest.json').read_text())
                if mutation=='contract':manifest['contract']['seed']=43
                elif mutation=='directory':manifest['arguments']['output_dir']=str(w.parent/'foreign')
                else:manifest['arguments']['num_envs']=258
                (w/'manifest.json').write_text(json.dumps(manifest))
                (w/'updates.jsonl').write_text(json.dumps(dict(global_iteration=1,stage_iteration=1,stage='S1',num_envs=512,env_steps=24576))+'\n')
                t.read_worker(w);self.assertEqual(t.pending_updates,[]);self.assertEqual(len(t.state['quarantine_ids']),1)
    def test_zero_evaluation_denominator_omits_rates(self):
        result=evaluation_metrics(dict(stage='S1',kind='validation',iteration=0,stage_iteration=0,checkpoint_sha256='a'*64,
            case_table_sha256='b'*64,speeds=[dict(vx=.2,episodes=0,climb_successes=0,speed_pass_count=0,joint_successes=0,joint_success_rate=0.)]))
        self.assertFalse(any(key.endswith('_rate') for key in result))
    def test_panel_definitions_route_joint_and_segment_without_flat_gate(self):
        patterns=[p for ps in PANELS.values() for p in ps]
        self.assertIn('stair_segment/*',patterns);self.assertIn('eval/joint/*',patterns)
        self.assertNotIn('eval/flat/*',patterns);self.assertEqual(len(PANELS),8)

class V03CurriculumAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
    def setUp(self):
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger
        from src.tasks.velocity.v03.spec import contract
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.t,self.backend,self.clock,_=TelemetryTests().setup_monitor(self.root/'telemetry')
        self.w=self.root/'worker';self.w.mkdir();train_manifest(self.w)
        self.c=Curriculum();self.ledger=AttemptLedger(self.root/'attempts',run_id='v03-seed42',execution='formal')
        self.manifest=json.loads((self.w/'manifest.json').read_text())
        self.manifest.update(num_envs=258,contract=contract(),ledger_path=str(self.ledger.path.resolve()),ledger_run_id='v03-seed42')
        self.manifest['arguments'].update(output_dir=str(self.w.resolve()),num_envs=258)
        (self.w/'manifest.json').write_text(json.dumps(self.manifest))
    def paid(self,count):
        from test_v03_recovery import complete
        # Keep real JSONL writes/hash chains and state transitions; avoid many test fsync syscalls.
        with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):
            for _ in range(count):complete(self.ledger,self.c,258)
        row=self.progress();row['stage']=self.c.stage
        with (self.w/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        self.t.read_worker(self.w)
    def progress(self):
        return dict(global_iteration=self.c.iteration,stage_iteration=self.c.stage_iteration,env_steps=self.c.env_steps,num_envs=258,
            **{'curriculum/stage':self.c.stage,'curriculum/consecutive_passes':99,'curriculum/s1_extra_passed':99})
    def budget(self):
        self.ledger.record_curriculum(self.c)
        document=dict(curriculum=self.c.state_dict(),last_complete_update=self.c.iteration,
                      ledger_path=str(self.ledger.path.resolve()),ledger_anchor=self.ledger.anchor())
        (self.w/'budget.json').write_text(json.dumps(document));return document
    def gate(self,*,kind='validation',passed=True):
        from test_v03_evaluation import make_gate_fixture
        summary,case=make_gate_fixture(self.root/f'raw_{self.c.iteration}_{kind}_{passed}',stage=self.c.stage,kind=kind,
            iteration=self.c.iteration,stage_iteration=self.c.stage_iteration,checkpoint_sha='1'*64,num_envs=258,passed=passed)
        method=self.c.validate if kind=='validation' else self.c.validate_perturbed
        method(summary,'1'*64,case)
        self.ledger.record_curriculum(self.c)
    def test_verified_exit_uses_formal_checkpoint_with_train_manifest(self):
        from src.tasks.velocity.v03.runner import PPO
        from src.tasks.velocity.v03.model import V03ActorCritic
        from src.tasks.velocity.v03.spec import INIT_PATH,INIT_SHA256
        from test_v03_runner import Env
        model=V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256);runner=PPO(model,Env(),ledger=self.ledger)
        path=self.w/'checkpoints'/'initialized.pt';path.parent.mkdir()
        sha=runner.save(path,self.c,{'seed':42},ledger=self.ledger)
        result=dict(curriculum=self.c.state_dict(),checkpoint=str(path),sha256=sha,actual_num_envs=258)
        p=self.t.worker_progress(self.w,self.progress(),result=result)
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        self.assertEqual(p['curriculum/snapshot_source'],'verified_child_result:'+sha)
        self.assertEqual(p['accounting/completed_ppo_updates'],0)
    def test_unverified_exited_result_never_advances_trusted_worker_cursor(self):
        import copy
        self.paid(1);self.budget();start=self.progress();self.t.worker_progress(self.w,start)
        fake=copy.deepcopy(self.c);fake.updated(258)
        result=dict(curriculum=fake.state_dict(),checkpoint=str(self.w/'checkpoints'/'missing.pt'),sha256='0'*64,actual_num_envs=258)
        p=self.t.worker_progress(self.w,start,result=result)
        self.assertEqual(p['global_iteration'],1);self.assertEqual(p['env_steps'],12384)
        self.assertEqual(self.t.state['worker_progress'][str(self.w.resolve())]['progress']['global_iteration'],1)
    def test_complete_attempt_budget_does_not_require_redundant_full_curriculum_commit(self):
        self.paid(2)
        document=dict(curriculum=self.c.state_dict(),last_complete_update=2,ledger_path=str(self.ledger.path.resolve()),ledger_anchor=self.ledger.anchor())
        (self.w/'budget.json').write_text(json.dumps(document))
        p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        self.assertEqual(p['accounting/completed_ppo_updates'],2)
        self.assertIsNone(self.ledger.curriculum_state)
    def test_committed_prefix_excludes_concurrent_inflight_tail(self):
        self.paid(2);document=self.budget();p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        self.assertEqual(p['accounting/completed_ppo_updates'],2)
        self.assertEqual(p['accounting/physics_samples_observed'],24768)
        self.assertEqual(p['accounting/discarded_samples'],0)
        self.assertEqual(p['accounting/update_attempts'],2)
        self.assertEqual(p['curriculum/snapshot_ledger_seq'],document['ledger_anchor']['seq'])
        self.ledger.begin_attempt(self.c,258);self.ledger.control_step(dict(step_index=1,num_envs=258,physics_samples_increment=258))
        p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot');self.assertEqual(p['global_iteration'],2)
        self.assertEqual(p['curriculum/snapshot_ledger_seq'],document['ledger_anchor']['seq'])
        document['ledger_anchor']=self.ledger.anchor();(self.w/'budget.json').write_text(json.dumps(document))
        p=self.t.worker_progress(self.w,self.progress());self.assertEqual(p['curriculum/snapshot_status'],'last_verified_snapshot')
    def test_warm_same_history_uses_incremental_paid_updates_but_restart_is_cold(self):
        from src.tasks.velocity.v03.runner import Curriculum
        self.paid(1);document=self.budget();self.t.worker_progress(self.w,self.progress())
        calls=[];original=Curriculum.load_state_dict
        def checked(obj,state):calls.append(state['iteration']);return original(obj,state)
        with patch.object(Curriculum,'load_state_dict',new=checked):
            self.paid(1);document=self.budget();calls.clear()
            p=self.t.worker_progress(self.w,self.progress());self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
            self.assertEqual(calls,[])
            resumed=Telemetry(self.t.directory,self.backend,{'seed':42},clock=self.clock)
            resumed.worker_progress(self.w,self.progress())
            self.assertEqual(resumed._ledger_cache[str(self.ledger.path.resolve())]['ledger'].completed_ppo_updates,2)
    def test_rewritten_prefix_and_replaced_file_cannot_use_verified_cache(self):
        self.paid(1);document=self.budget();self.t.worker_progress(self.w,self.progress())
        original=self.ledger.path.read_bytes();self.ledger.path.write_bytes(original.replace(b'v03-seed42',b'v03-seed43',1))
        with self.assertRaises(ValueError):self.t._budget_ledger(self.w,self.manifest,document)
        self.ledger.path.write_bytes(original)
        replacement=self.ledger.path.with_suffix('.replacement');replacement.write_bytes(original);replacement.replace(self.ledger.path)
        with self.assertRaisesRegex(ValueError,'replaced'):self.t._budget_ledger(self.w,self.manifest,document)
        resumed=Telemetry(self.t.directory,self.backend,{'seed':42},clock=self.clock)
        p=resumed.worker_progress(self.w,self.progress());self.assertEqual(p['curriculum/snapshot_status'],'last_verified_snapshot')
        self.assertEqual(resumed._ledger_cache,{})
    def test_missing_wrong_source_or_uncommitted_budget_never_invents_counts(self):
        import copy
        p=self.t.worker_progress(self.w,self.progress())
        self.assertNotIn('curriculum/consecutive_passes',p);self.assertNotIn('curriculum/s1_extra_passed',p)
        self.paid(1);good=self.budget()
        bad=copy.deepcopy(good);bad['ledger_anchor']['execution']='train'
        (self.w/'budget.json').write_text(json.dumps(bad));p=self.t.worker_progress(self.w,self.progress())
        self.assertNotIn('curriculum/consecutive_passes',p)
        (self.w/'budget.json').write_text(json.dumps(good));wrong=copy.deepcopy(self.manifest);wrong['ledger_run_id']='other'
        (self.w/'manifest.json').write_text(json.dumps(wrong));p=self.t.worker_progress(self.w,self.progress())
        self.assertNotIn('curriculum/consecutive_passes',p)
        (self.w/'manifest.json').write_text(json.dumps(self.manifest));p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/consecutive_passes'],0);self.assertEqual(p['curriculum/s1_extra_passed'],0)
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
    def test_real_raw_gate_streak_zero_one_zero_and_stale_budget(self):
        from src.tasks.velocity.v03.telemetry import atomic_json
        self.paid(200);before=self.budget();p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/consecutive_passes'],0)
        self.gate();self.budget();p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/consecutive_passes'],1);self.assertEqual(p['curriculum/snapshot_gate_iteration'],200)
        atomic_json(self.w/'budget.json',before);p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/consecutive_passes'],1);self.assertEqual(p['curriculum/snapshot_status'],'last_verified_snapshot')
        self.paid(200);self.gate(passed=False);self.budget();p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/consecutive_passes'],0);self.assertEqual(p['curriculum/snapshot_gate_iteration'],400)
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
    def test_valid_but_substituted_old_gate_cannot_append_over_verified_history(self):
        import copy
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import canonical,digest
        from test_v03_evaluation import make_gate_fixture
        self.paid(200);self.gate();document=self.budget();self.t.worker_progress(self.w,self.progress())
        summary,case=make_gate_fixture(self.root/'substituted_failure',stage='S1',iteration=200,stage_iteration=200,
            checkpoint_sha='1'*64,num_envs=258,passed=False)
        alternate=Curriculum()
        for _ in range(200):alternate.updated(258)
        alternate.validate(summary,'1'*64,case)
        # A self-consistent hash-chain append still cannot rewrite an accepted gate.
        anchor=document['ledger_anchor'];state=alternate.state_dict()
        event=dict(seq=anchor['seq']+1,previous=anchor['digest'],event='curriculum_committed',
            payload=dict(curriculum=state,curriculum_digest=digest(state)));event['digest']=digest(event)
        with self.ledger.path.open('ab') as stream:stream.write((canonical(event)+'\n').encode())
        bad=copy.deepcopy(document);bad['curriculum']=state;bad['ledger_anchor'].update(seq=event['seq'],digest=event['digest'])
        with self.assertRaisesRegex(ValueError,'history rewritten'):self.t._budget_ledger(self.w,self.manifest,bad)
    def test_s1_extra_failure_resets_two_passes_without_stage_change(self):
        self.paid(200);self.gate();self.paid(200);self.gate();self.budget()
        self.assertEqual(self.t.worker_progress(self.w,self.progress())['curriculum/consecutive_passes'],2)
        self.gate(kind='perturbed',passed=False);self.budget();p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual((p['curriculum/consecutive_passes'],p['curriculum/s1_extra_passed'],p['curriculum/stage']),(0,0,'S1'))
    def test_s1_extra_and_next_stage_reset_preserve_global_samples(self):
        self.paid(200);self.gate();self.paid(200);self.gate();self.budget()
        start=self.progress();self.t.worker_progress(self.w,start)
        self.gate(kind='perturbed');self.ledger.record_curriculum(self.c);self.c.advance();self.budget()
        p=self.t.worker_progress(self.w,start)
        self.assertEqual((p['global_iteration'],p['stage_iteration'],p['curriculum/stage']),(400,0,'S2'))
        self.assertEqual((p['curriculum/consecutive_passes'],p['curriculum/s1_extra_passed']),(0,1))
        self.assertEqual(p['env_steps'],4953600);self.assertEqual(p['curriculum/stage_budget'],4000)
        self.paid(1);self.budget();row=self.progress();row['stage']=self.c.stage
        (self.w/'updates.jsonl').write_text(json.dumps(row)+'\n');self.t.read_worker(self.w)
        self.assertEqual(self.t.worker_progress(self.w,start)['stage_iteration'],1)

class ConstructorTargetTests(unittest.TestCase):
    """Exercise the real constructor and the SDK's pure settings merge boundary."""
    def resolved_target(self,root,*,environment=None,project=None,entity=None):
        import os
        import wandb
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import Mock
        safe_env=dict(environment or {});captured={}
        def initialize(**kwargs):
            captured.update(kwargs)
            # The external init boundary alone is replaced. Target resolution
            # uses the installed SDK's real pure system/workspace/env merger.
            # Default run names below are synthetic fixture values, not claims
            # about W&B's actual default project/account or an online session.
            settings=wandb.Settings(root_dir=str(root),settings_system=str(root/'system_settings'),silent=True)
            settings.update_from_system_settings();settings.update_from_env_vars(safe_env)
            settings.update_from_dict(dict(project=kwargs['project'],entity=kwargs['entity']))
            return SimpleNamespace(settings=SimpleNamespace(mode='online',base_url='https://example.invalid'),
                project=settings.project or 'fixture_sdk_default_project',entity=settings.entity or 'fixture_sdk_default_entity',
                id='run',url='https://example.invalid/run',dir=str(root/'sdk'/'files'),step=0,summary={},define_metric=Mock())
        # Only temporary project/entity settings and safe env keys are parsed;
        # no real credential/settings file is read, and services are blocked.
        with ExitStack() as stack:
            for call in ('setup','Api'):
                stack.enter_context(patch.object(wandb,call,side_effect=AssertionError('No SDK service/setup/API permitted')))
            stack.enter_context(patch.object(wandb,'init',side_effect=initialize))
            stack.enter_context(patch.dict(os.environ,safe_env))
            backend=WandbBackend(project=project,entity=entity)
            identity=backend.start('run',{'seed':42})
        self.assertEqual((identity['project'],identity['entity']),(backend.project,backend.entity))
        return (backend.project,backend.entity),(captured['project'],captured['entity'])
    def write_config(self,root,*,system=True,workspace=True):
        if system:(root/'system_settings').write_text('[default]\nproject = configured_system_project\nentity = configured_system_entity\n')
        if workspace:
            (root/'wandb').mkdir(exist_ok=True)
            (root/'wandb'/'settings').write_text('[default]\nproject = configured_workspace_project\nentity = configured_workspace_entity\n')
    def test_existing_environment_target_is_not_overridden_by_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_config(root)
            resolved,_=self.resolved_target(root,environment={'WANDB_PROJECT':'configured_env_project','WANDB_ENTITY':'configured_env_entity'})
            self.assertEqual(resolved,('configured_env_project','configured_env_entity'))
    def test_existing_system_and_workspace_config_targets_are_preserved(self):
        for workspace,expected in ((False,('configured_system_project','configured_system_entity')),
                                   (True,('configured_workspace_project','configured_workspace_entity'))):
            with self.subTest(workspace=workspace), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.write_config(root,workspace=workspace)
                resolved,_=self.resolved_target(root)
                self.assertEqual(resolved,expected)
    def test_explicit_project_and_entity_override_existing_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_config(root)
            resolved,passed=self.resolved_target(root,environment={'WANDB_PROJECT':'configured_env_project','WANDB_ENTITY':'configured_env_entity'},
                project='explicit_project',entity='explicit_entity')
            self.assertEqual(resolved,('explicit_project','explicit_entity'));self.assertEqual(passed,('explicit_project','explicit_entity'))
            resolved,_=self.resolved_target(root,environment={'WANDB_PROJECT':'configured_env_project','WANDB_ENTITY':'configured_env_entity'},project='explicit_project')
            self.assertEqual(resolved,('explicit_project','configured_env_entity'))
    def test_truly_unconfigured_target_is_left_for_sdk_default_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved,passed=self.resolved_target(Path(directory))
            self.assertEqual(passed,(None,None));self.assertEqual(resolved,('fixture_sdk_default_project','fixture_sdk_default_entity'))

class SharedLedgerHandoffTests(unittest.TestCase):
    """One real formal hash-chain shared by sequential legal worker directories."""
    setUpClass=classmethod(V03CurriculumAuditTests.setUpClass.__func__)
    paid=V03CurriculumAuditTests.paid
    progress=V03CurriculumAuditTests.progress
    budget=V03CurriculumAuditTests.budget
    gate=V03CurriculumAuditTests.gate
    def setUp(self):
        V03CurriculumAuditTests.setUp(self)
        # Both workers retain the same actual captured source contract; these
        # logic-only cases do not ask a raw verifier to accept developing source.
        self.t=Telemetry(self.root/'bound_telemetry',self.backend,{'seed':42,'contract':self.manifest['contract']},
            clock=self.clock,wall_clock=self.clock);self.t.start()
    def warm_first(self):
        self.paid(1);document=self.budget();first=(self.w,self.manifest,self.progress(),document)
        p=self.t.worker_progress(self.w,self.progress());self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        return first
    def next_worker(self,name='worker_2'):
        import copy
        worker=self.root/name;worker.mkdir();manifest=copy.deepcopy(self.manifest)
        manifest['arguments']['output_dir']=str(worker.resolve())
        (worker/'manifest.json').write_text(json.dumps(manifest))
        return worker,manifest
    def adopt(self,worker,manifest):
        from src.tasks.velocity.v03.recovery import AttemptLedger
        self.w=worker;self.manifest=manifest
        self.ledger=AttemptLedger(self.root/'attempts',run_id='v03-seed42',execution='formal')
    def test_two_legal_chunks_share_one_ledger_and_bind_current_worker_as_of(self):
        first=self.warm_first();worker,manifest=self.next_worker();self.adopt(worker,manifest)
        self.budget();p=self.t.worker_progress(worker,self.progress())
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        self.assertEqual(p['curriculum/snapshot_worker'],str(worker.resolve()))
        self.paid(1);self.budget();p=self.t.worker_progress(worker,self.progress())
        self.assertEqual((p['global_iteration'],p['env_steps'],p['accounting/completed_ppo_updates']),(2,24768,2))
        self.assertEqual(p['curriculum/snapshot_as_of_iteration'],2);self.assertEqual(len(self.t._ledger_cache),1)
    def test_handoff_budget_ahead_one_keeps_main_cursor_and_explicit_as_of(self):
        from test_v03_recovery import complete
        first=self.warm_first();worker,manifest=self.next_worker();self.adopt(worker,manifest)
        self.budget();self.t.worker_progress(worker,self.progress());before=self.progress()
        with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):complete(self.ledger,self.c,258)
        self.budget();p=self.t.worker_progress(worker,before)
        self.assertEqual((p['global_iteration'],p['curriculum/snapshot_as_of_iteration'],p['curriculum/snapshot_document_iteration']),(1,1,2))
        self.assertEqual(p['accounting/as_of_iteration'],2)
        row=self.progress();row['stage']=self.c.stage
        (worker/'updates.jsonl').write_text(json.dumps(row)+'\n');self.t.read_worker(worker)
        self.assertEqual(self.t.worker_progress(worker,before)['global_iteration'],2)
    def test_cold_new_worker_still_checks_persisted_prefix_bytes(self):
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger
        from test_v03_recovery import complete
        first=self.warm_first();worker,manifest=self.next_worker()
        resumed=Telemetry(self.t.directory,self.backend,self.t.state['config'],clock=self.clock,wall_clock=self.clock)
        alternate=AttemptLedger(self.root/'alternate_cold',run_id='v03-seed42',execution='formal');c=Curriculum();alternate.record_curriculum(c)
        with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):complete(alternate,c,258)
        alternate.record_curriculum(c);self.ledger.path.write_bytes(alternate.path.read_bytes())
        document=dict(curriculum=c.state_dict(),last_complete_update=1,ledger_path=str(self.ledger.path.resolve()),ledger_anchor=alternate.anchor())
        with self.assertRaises(ValueError):resumed._budget_ledger(worker,manifest,document)
        self.assertEqual(resumed._ledger_cache,{})
    def test_same_worker_manifest_rewrite_is_rejected(self):
        import copy
        worker,manifest,progress,document=self.warm_first();bad=copy.deepcopy(manifest);bad['worker_note']='rewritten'
        with self.assertRaises(ValueError):self.t._budget_ledger(worker,bad,document)
    def test_foreign_run_or_source_cannot_handoff(self):
        import copy
        first=self.warm_first();worker,manifest=self.next_worker()
        for mutation in ('run','source'):
            with self.subTest(mutation=mutation):
                bad=copy.deepcopy(manifest);document=copy.deepcopy(first[3])
                if mutation=='run':bad['ledger_run_id']='foreign';document['ledger_anchor']['run_id']='foreign'
                else:bad['contract']['protocol_id']='foreign-source'
                with self.assertRaises(ValueError):self.t._budget_ledger(worker,bad,document)
        self.assertEqual(self.t.worker_progress(first[0],first[2])['global_iteration'],1)
    def test_inode_replacement_blocks_even_a_new_worker(self):
        first=self.warm_first();worker,manifest=self.next_worker();raw=self.ledger.path.read_bytes()
        replacement=self.ledger.path.with_suffix('.replacement');replacement.write_bytes(raw);replacement.replace(self.ledger.path)
        with self.assertRaises(ValueError):self.t._budget_ledger(worker,manifest,first[3])
    def test_cold_known_worker_inode_replacement_is_not_new_confirmation(self):
        first=self.warm_first();raw=self.ledger.path.read_bytes()
        replacement=self.ledger.path.with_suffix('.cold_replacement');replacement.write_bytes(raw);replacement.replace(self.ledger.path)
        resumed=Telemetry(self.t.directory,self.backend,self.t.state['config'],clock=self.clock,wall_clock=self.clock)
        with self.assertRaises(ValueError):resumed._budget_ledger(first[0],first[1],first[3])
        self.assertEqual(resumed._ledger_cache,{})
    def test_cold_new_worker_inode_replacement_is_rejected(self):
        first=self.warm_first();worker,manifest=self.next_worker();raw=self.ledger.path.read_bytes()
        replacement=self.ledger.path.with_suffix('.cold_handoff_replacement');replacement.write_bytes(raw);replacement.replace(self.ledger.path)
        resumed=Telemetry(self.t.directory,self.backend,self.t.state['config'],clock=self.clock,wall_clock=self.clock)
        with self.assertRaises(ValueError):resumed._budget_ledger(worker,manifest,first[3])
        self.assertEqual(resumed._ledger_cache,{})
    def test_self_consistent_new_manifest_and_rewritten_prefix_cannot_handoff(self):
        from src.tasks.velocity.v03.runner import Curriculum
        from src.tasks.velocity.v03.recovery import AttemptLedger
        from test_v03_recovery import complete
        first=self.warm_first();worker,manifest=self.next_worker()
        alternate=AttemptLedger(self.root/'alternate',run_id='v03-seed42',execution='formal');c=Curriculum()
        alternate.record_curriculum(c)
        with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):complete(alternate,c,258)
        alternate.record_curriculum(c)
        # Same run/source/counts and valid hash-chain, but rewrites an accepted
        # prefix on the same inode. Dropping the old cache would accept it.
        self.ledger.path.write_bytes(alternate.path.read_bytes())
        document=dict(curriculum=c.state_dict(),last_complete_update=c.iteration,ledger_path=str(self.ledger.path.resolve()),ledger_anchor=alternate.anchor())
        with self.assertRaises(ValueError):self.t._budget_ledger(worker,manifest,document)
    def test_retired_worker_cannot_reactivate_or_publish_stale_global_progress(self):
        first=self.warm_first();worker,manifest=self.next_worker();self.adopt(worker,manifest)
        self.budget();self.t.worker_progress(worker,self.progress());self.paid(1);self.budget()
        current=self.t.worker_progress(worker,self.progress());self.assertEqual(current['global_iteration'],2)
        with self.assertRaises(ValueError):self.t.worker_progress(first[0],first[2])
        # Even a retired worker presenting the newest budget cannot take over.
        with self.assertRaises(ValueError):self.t._budget_ledger(first[0],first[1],self.budget())
        self.assertEqual(self.t.worker_progress(worker,self.progress())['global_iteration'],2)
    def test_restart_retains_worker_binding_and_verified_prefix_provenance(self):
        first=self.warm_first();worker,manifest=self.next_worker();self.adopt(worker,manifest)
        self.budget();self.t.worker_progress(worker,self.progress());self.paid(1);document=self.budget();self.t.worker_progress(worker,self.progress())
        resumed=Telemetry(self.t.directory,self.backend,self.t.state['config'],clock=self.clock,wall_clock=self.clock)
        with self.assertRaises(ValueError):resumed.worker_progress(first[0],first[2])
        p=resumed.worker_progress(worker,self.progress());self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        # A fresh, internally consistent rewrite must also fail after restart.
        raw=self.ledger.path.read_bytes();self.ledger.path.write_bytes(raw.replace(b'v03-seed42',b'v03-seed43',1))
        with self.assertRaises(ValueError):resumed._budget_ledger(worker,manifest,document)

class SharedLedgerRawGateTests(unittest.TestCase):
    """Run only in a parent-coordinated static product-source window."""
    setUpClass=SharedLedgerHandoffTests.__dict__['setUpClass']
    setUp=SharedLedgerHandoffTests.setUp
    paid=SharedLedgerHandoffTests.paid
    progress=SharedLedgerHandoffTests.progress
    budget=SharedLedgerHandoffTests.budget
    gate=SharedLedgerHandoffTests.gate
    next_worker=SharedLedgerHandoffTests.next_worker
    adopt=SharedLedgerHandoffTests.adopt
    def test_new_worker_gate_uses_raw_oracle_and_restart_rechecks_history(self):
        self.paid(200);self.budget();p=self.t.worker_progress(self.w,self.progress())
        self.assertEqual(p['curriculum/consecutive_passes'],0)
        worker,manifest=self.next_worker();self.adopt(worker,manifest);self.gate();document=self.budget()
        index=Path(self.c.history[-1]['summary']['raw_index_path']);raw_index=index.read_bytes();index.unlink()
        # No verifier replacement: a genuinely missing raw file must stop the
        # newly committed gate before a new worker can take the shared cache.
        with self.assertRaises((ValueError,OSError)):self.t._budget_ledger(worker,manifest,document)
        index.write_bytes(raw_index)
        p=self.t.worker_progress(worker,self.progress())
        self.assertEqual(p['curriculum/snapshot_status'],'verified_snapshot')
        self.assertEqual(p['curriculum/consecutive_passes'],1);self.assertEqual(p['curriculum/snapshot_gate_iteration'],200)
        self.assertEqual(p['curriculum/snapshot_worker'],str(worker.resolve()))
        index.unlink()
        resumed=Telemetry(self.t.directory,self.backend,self.t.state['config'],clock=self.clock,wall_clock=self.clock)
        with self.assertRaises((ValueError,OSError)):resumed._budget_ledger(worker,manifest,document)
        self.assertEqual(resumed._ledger_cache,{})

if __name__=="__main__":unittest.main()
