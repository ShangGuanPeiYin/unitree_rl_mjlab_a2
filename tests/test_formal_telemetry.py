"""Online monitoring fault injection without network, credentials or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from src.tasks.velocity.v02_formal.telemetry import Telemetry,MonitoringBlocked,aggregate_updates,evaluation_metrics,finite_metrics

def train_manifest(directory):
    from src.tasks.velocity.v02_formal.spec import TASK,PROTOCOL_ID,CODE_VERSION,INIT_SHA256
    directory=Path(directory)
    (directory/'manifest.json').write_text(json.dumps(dict(task=TASK,protocol_id=PROTOCOL_ID,code_version=CODE_VERSION,source_c_sha256=INIT_SHA256,seed=42,execution='train',arguments={'execution':'train'})))

def train_episode(directory,**values):
    from src.tasks.velocity.v02_formal.spec import TASK,PROTOCOL_ID,CODE_VERSION,INIT_SHA256
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
    def finish(self,summary):pass

class TelemetryTests(unittest.TestCase):
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
            before=len(b.sent);m.evaluation(dict(stage='R0',kind='validation',iteration=11,stage_iteration=11,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[dict(vx=.2,episodes=20,successes=18)],passed=True))
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
            summary=dict(stage='R0',kind='validation',iteration=200,stage_iteration=200,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[dict(vx=.2,episodes=20,successes=18)])
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
        from src.tasks.velocity.v02_formal.telemetry import WandbBackend
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


class SourceRevisionBackendTests(unittest.TestCase):
    def test_pending_revision_does_not_implicitly_overwrite_config_during_sdk_init(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.tasks.velocity.v02_formal.telemetry import WandbBackend
        with tempfile.TemporaryDirectory() as path:
            run=SimpleNamespace(settings=SimpleNamespace(mode='online',base_url='https://example.invalid'),project='p',entity='e',url='https://example.invalid/run',dir=str(Path(path)/'files'),step=7,summary={},define_metric=Mock())
            backend=WandbBackend.__new__(WandbBackend);backend.name='test';backend.project='p';backend.entity='e';backend.approved_config_revision={'parent_config':{'old':True}}
            backend.wandb=SimpleNamespace(init=Mock(return_value=run),Settings=lambda **kwargs:None)
            backend.start('same-run',{'contract':{'new_source':True}})
            self.assertIsNone(backend.wandb.init.call_args.kwargs['config'])
    def test_revision_receipt_marker_without_full_current_remote_config_is_unconfirmed(self):
        from types import SimpleNamespace
        from src.tasks.velocity.v02_formal.telemetry import WandbBackend
        backend=WandbBackend.__new__(WandbBackend);backend.entity='e';backend.project='p';backend.run=SimpleNamespace(id='same-run')
        config={'contract':{'behavior_sha256':{'monitor.py':'old'}},'source_revision_history':[{'receipt_sha256':'receipt'}]}
        backend.wandb=SimpleNamespace(Api=lambda **kwargs:SimpleNamespace(run=lambda path:SimpleNamespace(config=config)))
        backend.revision_expected_config={'contract':{'behavior_sha256':{'monitor.py':'new'}}}
        self.assertFalse(backend.verify_config_revision({'receipt_sha256':'receipt'}))
        config.update(backend.revision_expected_config);self.assertTrue(backend.verify_config_revision({'receipt_sha256':'receipt'}))


class MonitoringClockTests(unittest.TestCase):
    def monitor(self,path):
        clock=Clock();backend=Backend();pauses=[]
        t=Telemetry(path,backend,{'seed':42},clock=clock,wall_clock=clock,pause=pauses.append);t.start();return t,backend,clock,pauses
    def progress(self,i,stage='R0',stage_i=None):return dict(global_iteration=i,stage_iteration=i if stage_i is None else stage_i,env_steps=i*512*48,num_envs=512,**{'curriculum/stage':stage})
    def test_train40_empty_pending_heartbeat_and_restart_use_trusted40_then41(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,_=self.monitor(Path(path)/'telemetry');worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            (worker/'updates.jsonl').write_text(json.dumps(dict(self.progress(40),stage='R0'))+'\n');t.read_worker(worker)
            t.tick(t.worker_progress(worker,self.progress(0)),force=True);self.assertEqual(t.pending_updates,[])
            c.advance(31);t.tick(t.worker_progress(worker,self.progress(0)))
            heartbeat=next(event for event in b.sent if event['kind']=='heartbeat');self.assertEqual(heartbeat['metrics']['global_iteration'],40);self.assertEqual(heartbeat['metrics']['env_steps'],983040)
            resumed=Telemetry(Path(path)/'telemetry',b,{'seed':42},clock=c,wall_clock=c);resumed.restore_pending()
            self.assertEqual(resumed.worker_progress(worker,self.progress(0))['global_iteration'],40)
            # Rebuild from durable train event even if the optional cache was lost.
            resumed.state.pop('worker_progress',None);resumed._save();rebuilt=resumed.worker_progress(worker,self.progress(0));self.assertEqual(rebuilt['global_iteration'],40)
            with (worker/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(self.progress(41),stage='R0'))+'\n')
            resumed.read_worker(worker);resumed.tick(resumed.worker_progress(worker,self.progress(0)),force=True)
            self.assertEqual(b.sent[-1]['metrics']['global_iteration'],41)
    def test_same_stage_reset_rejected_confirmed_next_stage_reset_allowed(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,_=self.monitor(Path(path)/'telemetry');worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            t.worker_progress(worker,self.progress(400))
            (worker/'updates.jsonl').write_text(json.dumps(dict(self.progress(401,stage_i=0),stage='R0'))+'\n')
            with self.assertRaisesRegex(ValueError,'Stage iteration regressed'):t.read_worker(worker)
            self.assertEqual(t.pending_updates,[])
            (worker/'updates.jsonl').write_text(json.dumps(dict(self.progress(401,stage='S1',stage_i=1),stage='S1'))+'\n')
            with self.assertRaisesRegex(ValueError,'Unconfirmed curriculum'):t.read_worker(worker)
            (worker/'budget.json').write_text(json.dumps({'curriculum':{'stage':'S1','iteration':401,'stage_iteration':1}}))
            t.read_worker(worker);p=t.worker_progress(worker,self.progress(400));self.assertEqual(p['global_iteration'],401);self.assertEqual(p['stage_iteration'],1);self.assertEqual(p['curriculum/stage'],'S1')
            self.assertEqual(p['curriculum/stage_budget'],3000)
            (worker/'budget.json').write_text(json.dumps({'curriculum':{'stage':'S2','iteration':801,'stage_iteration':1}}))
            t._cache_worker_progress(worker,self.progress(801,stage='S2',stage_i=1),'verified next-stage update',allow_transition=True)
            self.assertEqual(t.worker_progress(worker,self.progress(400))['curriculum/stage_budget'],4000)
    def test_unconfirmed_network_does_not_turn_50_updates_into_1_update_windows(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,c,pauses=self.monitor(Path(path)/'telemetry');b.down=True;worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
            for i in range(1,51):
                c.advance(3)
                with (worker/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(self.progress(i),stage='R0'))+'\n')
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
                with (worker/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(self.progress(i),stage='R0'))+'\n')
                t.read_worker(worker);t.tick(t.worker_progress(worker,self.progress(0)))
            self.assertEqual(t.state['window_started_wall'],3.)
            c.advance(20);resumed=Telemetry(Path(path)/'telemetry',b,{'seed':42},clock=c,wall_clock=c);resumed.restore_pending();resumed.tick(resumed.worker_progress(worker,self.progress(0)))
            self.assertFalse(any(event['kind']=='train' for event in resumed.events))
            c.advance(1);resumed.tick(resumed.worker_progress(worker,self.progress(0)));self.assertEqual(b.sent[-1]['kind'],'train');self.assertEqual(b.sent[-1]['metrics']['train/window_updates'],3)
            c.advance(30);resumed.tick(self.progress(3));self.assertEqual(b.sent[-1]['kind'],'heartbeat')
            resumed.evaluation(dict(stage='R0',kind='validation',iteration=3,stage_iteration=3,checkpoint_sha256='a'*64,case_table_sha256='b'*64,speeds=[]))
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




class CurriculumSnapshotTests(unittest.TestCase):
    def fixture(self,path):
        from src.tasks.velocity.v02_formal.spec import contract
        from src.tasks.velocity.v02_formal.runner import Curriculum
        t,b,clock,_=MonitoringClockTests().monitor(Path(path)/'telemetry')
        worker=Path(path)/'worker';worker.mkdir();train_manifest(worker)
        manifest=json.loads((worker/'manifest.json').read_text())
        manifest.update(num_envs=512,contract=contract())
        manifest['arguments'].update(output_dir=str(worker.resolve()),num_envs=512)
        (worker/'manifest.json').write_text(json.dumps(manifest))
        return t,b,clock,worker,Curriculum()
    def paid(self,c,count):
        for _ in range(count):c.updated(512)
    def joint(self,c,passed=True):
        from test_formal_model_runner import summary,SHA,CASE,FLAT_CASE
        c.validate(summary(c,passed=passed),summary(c,'flat'),SHA,CASE,FLAT_CASE)
    def progress(self,c):
        return dict(global_iteration=c.iteration,stage_iteration=c.stage_iteration,env_steps=c.env_steps,num_envs=512,
            **{'curriculum/stage':c.stage,'curriculum/consecutive_passes':c.consecutive_passes,
               'curriculum/r0_extra_passed':int(c.r0_extra_passed)})
    def budget(self,worker,c):
        from src.tasks.velocity.v02_formal.telemetry import atomic_json
        atomic_json(worker/'budget.json',dict(curriculum=c.state_dict(),last_complete_update=c.iteration))
    def row(self,t,worker,c):
        with (worker/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(self.progress(c),stage=c.stage))+'\n')
        t.read_worker(worker)
    def test_real_gate_zero_to_one_and_one_to_zero_never_max_or_block_start(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);start=self.progress(c)
            t.worker_progress(w,start);self.joint(c);self.budget(w,c)
            p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.assertEqual(p['curriculum/snapshot_gate_iteration'],200)
            self.assertEqual(p['curriculum/snapshot_as_of_iteration'],200)
            self.assertTrue(p['curriculum/snapshot_source'].startswith('verified_worker_budget:'))
            self.paid(c,200);self.budget(w,c);self.row(t,w,c);p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.joint(c,False);self.budget(w,c);p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],0)
            self.assertEqual(p['curriculum/snapshot_gate_iteration'],400)
            self.assertEqual(p['curriculum/snapshot_as_of_iteration'],400)
    def test_budget_ahead_and_behind_one_update_keep_exact_clock_and_streak(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);self.joint(c)
            start=self.progress(c);self.budget(w,c);t.worker_progress(w,start)
            self.paid(c,1);self.budget(w,c)
            p=t.worker_progress(w,start)
            self.assertEqual(p['global_iteration'],200)
            self.assertEqual(p['curriculum/snapshot_document_iteration'],201)
            self.assertEqual(p['curriculum/snapshot_as_of_iteration'],200)
            self.row(t,w,c);self.paid(c,1);self.row(t,w,c)
            p=t.worker_progress(w,start)
            self.assertEqual(p['global_iteration'],202)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.assertEqual(p['curriculum/snapshot_as_of_iteration'],201)
    def test_stage_start_preserves_r0_extra_and_resets_only_current_stage_streak(self):
        from test_formal_model_runner import summary,SHA,CASE
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);self.joint(c);self.paid(c,200);self.joint(c)
            start=self.progress(c);self.budget(w,c);t.worker_progress(w,start)
            c.validate_perturbed(summary(c,'perturbed'),SHA,CASE);c.advance();self.budget(w,c)
            p=t.worker_progress(w,start)
            self.assertEqual((p['global_iteration'],p['stage_iteration'],p['curriculum/stage']),(400,0,'S1'))
            self.assertEqual(p['curriculum/consecutive_passes'],0)
            self.assertEqual(p['curriculum/r0_extra_passed'],1)
            self.assertEqual(p['curriculum/stage_budget'],3000)
            self.assertEqual(p['curriculum/snapshot_stage'],'S1')
            self.assertEqual(p['curriculum/snapshot_gate_iteration'],400)
            self.paid(c,1);self.budget(w,c);self.row(t,w,c)
            p=t.worker_progress(w,start)
            self.assertEqual((p['stage_iteration'],p['curriculum/consecutive_passes'],p['curriculum/r0_extra_passed']),(1,0,1))
    def test_r0_extra_failure_clears_two_passes_without_stage_change(self):
        from test_formal_model_runner import summary,SHA,CASE
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);self.joint(c);self.paid(c,200);self.joint(c)
            start=self.progress(c);self.budget(w,c);self.assertEqual(t.worker_progress(w,start)['curriculum/consecutive_passes'],2)
            c.validate_perturbed(summary(c,'perturbed',False),SHA,CASE);self.budget(w,c)
            p=t.worker_progress(w,start)
            self.assertEqual((p['curriculum/consecutive_passes'],p['curriculum/r0_extra_passed'],p['curriculum/stage']),(0,0,'R0'))
    def test_adopt_restart_empty_pending_cache_loss_force_and_close(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);start=self.progress(c);self.joint(c)
            self.paid(c,1);self.budget(w,c);self.row(t,w,c)
            t.tick(t.worker_progress(w,start),force=True)
            self.assertEqual(t.pending_updates,[])
            resumed=Telemetry(t.directory,b,{'seed':42},clock=clock,wall_clock=clock);resumed.restore_pending()
            clock.advance(31);resumed.tick(resumed.worker_progress(w,start))
            self.assertEqual(b.sent[-1]['kind'],'heartbeat')
            self.assertEqual(b.sent[-1]['metrics']['curriculum/consecutive_passes'],1)
            resumed.state.pop('worker_progress',None);resumed.state.pop('worker_curriculum',None)
            p=resumed.worker_progress(w,start)
            self.assertEqual((p['global_iteration'],p['curriculum/consecutive_passes']),(201,1))
            resumed.emit_once('resource_close',str(w),p)
            self.assertEqual(b.sent[-1]['metrics']['curriculum/snapshot_as_of_iteration'],201)
    def test_missing_partial_wrong_or_unconfirmed_budget_does_not_invent_zero(self):
        import copy
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);start=self.progress(c)
            for raw in (None,'{',json.dumps({'curriculum':{'consecutive_passes':1}})):
                if raw is not None:(w/'budget.json').write_text(raw)
                p=t.worker_progress(w,start)
                self.assertNotIn('curriculum/consecutive_passes',p);self.assertNotIn('curriculum/r0_extra_passed',p)
                self.assertEqual(p['curriculum/snapshot_status'],'unavailable')
            self.joint(c);self.budget(w,c);good=json.loads((w/'budget.json').read_text())
            invalid=[]
            for field,value in [('consecutive_passes',0),('r0_extra_passed',True),('env_steps',c.env_steps+48),('stage_iteration',0)]:
                bad=copy.deepcopy(good);bad['curriculum'][field]=value;invalid.append(bad)
            bad=copy.deepcopy(good);bad['last_complete_update']=199;invalid.append(bad)
            bad=copy.deepcopy(good);bad['accounting']={'conservative_charge':1};invalid.append(bad)
            for bad in invalid:
                (w/'budget.json').write_text(json.dumps(bad));p=t.worker_progress(w,start)
                self.assertNotIn('curriculum/consecutive_passes',p)
            self.budget(w,c);manifest=json.loads((w/'manifest.json').read_text())
            for kind in ('seed','output_dir','contract','num_envs'):
                bad=copy.deepcopy(manifest)
                if kind=='seed':bad['seed']=43
                elif kind=='output_dir':bad['arguments']['output_dir']=str(w.parent/'other_worker')
                elif kind=='contract':bad['contract']['seed']=43
                else:bad['num_envs']=258
                (w/'manifest.json').write_text(json.dumps(bad));p=t.worker_progress(w,start)
                self.assertNotIn('curriculum/consecutive_passes',p)
    def test_missing_budget_retains_labelled_same_stage_snapshot_not_fallback(self):
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);start=self.progress(c);self.joint(c);self.budget(w,c)
            p=t.worker_progress(w,start);(w/'budget.json').unlink();self.paid(c,1);self.row(t,w,c)
            p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.assertEqual(p['curriculum/snapshot_as_of_iteration'],200)
            self.assertEqual(p['curriculum/snapshot_status'],'last_verified_snapshot')
            t._cache_worker_progress(w,dict(self.progress(c),stage_iteration=0,**{'curriculum/stage':'S1'}),'test transition',allow_transition=True)
            p=t.worker_progress(w,start)
            self.assertNotIn('curriculum/consecutive_passes',p);self.assertNotIn('curriculum/r0_extra_passed',p)
    def test_strict_exited_result_and_invalid_checkpoint_never_invent_counters(self):
        from src.tasks.velocity.v02_formal.model import FormalActorCritic
        from src.tasks.velocity.v02_formal.runner import PPO
        from test_formal_model_runner import FakeEnv
        from src.tasks.velocity.v02_formal.spec import INIT_PATH,INIT_SHA256
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);start=self.progress(c);self.joint(c)
            runner=PPO(FormalActorCritic.from_c(INIT_PATH,INIT_SHA256),FakeEnv())
            checkpoint=w/'checkpoints'/'paused.pt';checkpoint.parent.mkdir()
            digest=runner.save(checkpoint,c,{'seed':42})
            result=dict(checkpoint=str(checkpoint),sha256=digest,curriculum=c.state_dict(),actual_num_envs=512)
            p=t.worker_progress(w,start,result=result)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.assertEqual(p['curriculum/snapshot_source'],'verified_child_result:'+digest)
            t.state.pop('worker_curriculum',None)
            result['sha256']='0'*64;p=t.worker_progress(w,start,result=result)
            self.assertNotIn('curriculum/consecutive_passes',p)
    def test_stale_or_rewritten_gate_history_cannot_replace_verified_snapshot(self):
        import copy
        from src.tasks.velocity.v02_formal.runner import Curriculum
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);before=copy.deepcopy(c);start=self.progress(c)
            self.joint(c);self.budget(w,c);p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.budget(w,before);p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.assertEqual(p['curriculum/snapshot_status'],'last_verified_snapshot')
            # A valid but substituted failing gate is not an appended new failure.
            self.joint(before,False);self.budget(w,before);p=t.worker_progress(w,start)
            self.assertEqual(p['curriculum/consecutive_passes'],1)
            self.assertEqual(p['curriculum/snapshot_status'],'last_verified_snapshot')

    def test_budget_snapshot_does_not_consume_training_rng(self):
        import random,numpy as np,torch
        with tempfile.TemporaryDirectory() as path:
            t,b,clock,w,c=self.fixture(path);self.paid(c,200);self.joint(c);self.budget(w,c)
            py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state().clone()
            t.worker_progress(w,self.progress(c))
            self.assertEqual(py,random.getstate());self.assertEqual(npstate[0],np.random.get_state()[0])
            self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]));self.assertEqual(npstate[2:],np.random.get_state()[2:])
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
