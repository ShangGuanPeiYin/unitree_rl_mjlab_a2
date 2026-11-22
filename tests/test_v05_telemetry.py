"""CPU-only v0.5 telemetry identity, sample accounting and delivery-protection tests.

No SDK run, network request or GPU is created.  The W&B constructor boundary is
replaced by a local fake whose ``send`` is an enqueue and whose ``remote_ids`` is
the only ACK source, so "local enqueue is not a remote ACK" is enforced by the
test fixture rather than by a comment.
"""
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch  # noqa: F401  (keeps the package import order identical to the product)

from src.tasks.velocity.v05 import telemetry as telemetry_module
from src.tasks.velocity.v05.connection import (DeliveryPump, ReliableTelemetry,
                                               ReliableWandbBackend, configure_network)
from src.tasks.velocity.v05.telemetry import (HEARTBEAT_KIND, GROUP, METRIC_PREFIXES,
                                              PANELS, PANEL_COUNT, MonitoringBlocked,
                                              Telemetry, WandbBackend, WorkerAttestation,
                                              aggregate_updates, atomic_json,
                                              diagnostic_record_reason, evaluation_metrics,
                                              record_namespace, sample_accounting,
                                              sample_weight)


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, value):
        self.value += value


class Backend:
    """Local fake: ``send`` is an enqueue, ``remote_ids`` is the only ACK."""

    def __init__(self):
        self.remote = set()
        self.sent = []
        self.down = False
        self.hidden = False
        self.starts = []
        self.panel_verification = None

    def start(self, run_id, config):
        self.starts.append((run_id, dict(config)))
        return dict(project='p', entity='e', run_id=run_id,
                    url='https://example.invalid/run/' + run_id,
                    base_url='https://example.invalid')

    def remote_ids(self):
        if self.down:
            raise ConnectionError('offline')
        return set(self.remote)

    def send(self, event):
        if self.down:
            raise ConnectionError('offline')
        self.sent.append(event)
        if not self.hidden:
            self.remote.add(event['id'])

    def verify_start(self, run_id, config):
        return True

    def ensure_panels(self, existing_url=None):
        self.panel_verification = dict(remote_verified=True, remote_group_count=PANEL_COUNT)
        return 'https://example.invalid/report'

    def finish(self, summary):
        pass


class DelayedBackend(Backend):
    def __init__(self, clock, delay=10):
        super().__init__()
        self.clock = clock
        self.delay = delay
        self.visible = []

    def send(self, event):
        self.sent.append(event)
        self.visible.append((self.clock() + self.delay, event['id']))

    def remote_ids(self):
        if self.down:
            raise ConnectionError('offline')
        return {identifier for when, identifier in self.visible if when <= self.clock()}


def assert_v05_identity(testcase, monitor):
    identity = monitor.state['identity']
    testcase.assertEqual(identity['run_id'], monitor.state['run_id'])
    readiness = monitor.state['readiness']
    testcase.assertEqual(readiness['group'], 'v05')
    testcase.assertEqual(readiness['heartbeat_kind'], 'v05_heartbeat')
    testcase.assertTrue(readiness['remote_verified'])
    testcase.assertTrue(readiness['config_verified'])
    testcase.assertEqual(readiness['run_id'], monitor.state['run_id'])
    initialize = [event for event in monitor.events if event['kind'] == 'initialize']
    testcase.assertEqual(len(initialize), 1)
    testcase.assertEqual(readiness['initialize_event_id'], initialize[0]['id'])


class SleepFree(unittest.TestCase):
    """Startup retries use the real 2 s sleep; the CPU suite stubs it."""

    def setUp(self):
        sleeper = patch('time.sleep', return_value=None)
        sleeper.start()
        self.addCleanup(sleeper.stop)


class IdentityTests(SleepFree):

    def test_v05_group_is_created_and_verified_at_the_real_constructor_boundary(self):
        config = {'seed': 42, 'contract': {'protocol_id': 'v05-stair-climb-first-v1'}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = types.SimpleNamespace(
                settings=types.SimpleNamespace(mode='online', base_url='https://example.invalid'),
                project='p', entity='e', id='run', url='https://example.invalid/run',
                dir=str(root / 'sdk' / 'files'), step=0, summary={}, define_metric=Mock())
            remote = types.SimpleNamespace(config=config | {'telemetry_run_id': 'run'},
                                           group=GROUP)
            backend = WandbBackend.__new__(WandbBackend)
            backend.name = 'v05_seed42_fixture'
            backend.project = 'p'
            backend.entity = 'e'
            backend.wandb = types.SimpleNamespace(
                init=Mock(return_value=run), Settings=lambda **kwargs: None,
                Api=lambda **kwargs: types.SimpleNamespace(run=lambda path: remote))
            identity = backend.start('run', config)
            self.assertEqual(backend.wandb.init.call_args.kwargs.get('group'), 'v05')
            self.assertEqual(identity['run_id'], 'run')
            self.assertTrue(backend.verify_start('run', config))
            remote.group = 'v04'
            self.assertFalse(backend.verify_start('run', config))
            remote.group = GROUP
            remote.config['seed'] = 43
            self.assertFalse(backend.verify_start('run', config))

    def test_readiness_and_single_initialize_are_v05(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {'seed': 42})
            identity = monitor.start()
            self.assertEqual(identity['run_id'], monitor.state['run_id'])
            assert_v05_identity(self, monitor)
            self.assertEqual(monitor.state['panels'], PANELS)
            self.assertEqual(monitor.state['panel_verification']['remote_group_count'],
                             PANEL_COUNT)

    def test_reliable_start_readiness_is_v05(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            monitor = ReliableTelemetry(directory, Backend(), {'seed': 42},
                                        clock=clock, wall_clock=clock)
            monitor.start()
            assert_v05_identity(self, monitor)
            self.assertEqual(monitor.state['readiness']['group'], 'v05')

    def test_monitoring_blocked_is_one_exception_class(self):
        self.assertIs(MonitoringBlocked, telemetry_module.MonitoringBlocked)
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            backend.hidden = True
            with self.assertRaises(MonitoringBlocked):
                Telemetry(directory, backend, {}).start()

    def test_config_or_panel_readback_failure_blocks_start(self):
        for failure in ('config', 'panels'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                backend = Backend()
                backend.verify_start = lambda *args: failure != 'config'
                backend.ensure_panels = lambda existing_url=None: \
                    'https://example.invalid/report'
                backend.panel_verification = dict(
                    remote_verified=True,
                    remote_group_count=PANEL_COUNT - 1 if failure == 'panels' else PANEL_COUNT)
                with self.assertRaises(MonitoringBlocked):
                    Telemetry(directory, backend, {}).start()


class PanelTests(unittest.TestCase):
    def test_exactly_eight_panels_and_v05_metric_routing(self):
        self.assertEqual(len(PANELS), 8)
        self.assertEqual(PANEL_COUNT, 8)
        self.assertIn('ppo/*', PANELS['PPO诊断'])
        self.assertIn('stall/*', PANELS['停车/stall/资源'])
        self.assertIn('probe/*', PANELS['停车/stall/资源'])
        self.assertIn('eval/*', PANELS['评估/门禁分离'])
        self.assertIn('flat_segment/*', PANELS['速度分段'])
        self.assertNotIn('swing', PANELS)
        # Every aggregate namespace belongs to exactly one panel group.
        for prefix in METRIC_PREFIXES:
            owners = [title for title, patterns in PANELS.items()
                      for pattern in patterns
                      if pattern == prefix + '/*' or pattern.startswith(prefix + '/')]
            self.assertTrue(owners, prefix)
        duplicated = []
        for title, patterns in PANELS.items():
            for pattern in patterns:
                if pattern.strip('*') in duplicated:
                    duplicated.append(pattern)
        self.assertEqual(duplicated, [])

    def test_no_v04_identity_remains_in_the_v05_monitoring_modules(self):
        for module in (telemetry_module,):
            source = Path(module.__file__).read_text()
            for forbidden in ('v04_heartbeat', "group='v04'", 'group == "v04"'):
                self.assertNotIn(forbidden, source)
        self.assertNotIn('v04', GROUP)
        self.assertNotIn('v04', HEARTBEAT_KIND)

    def test_report_title_and_eight_group_readback(self):
        with tempfile.TemporaryDirectory():
            reports = types.ModuleType('wandb_workspaces.reports.v2')

            class H1:
                def __init__(self, title):
                    self.title = title

            class PanelGrid:
                def __init__(self, runsets, panels):
                    self.runsets, self.panels = runsets, panels

            class LinePlot:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class Runset:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class Report:
                last_blocks = None
                last_title = None

                def __init__(self, entity=None, project=None, title=None, blocks=None,
                             width=None):
                    self.entity, self.project, self.title = entity, project, title
                    self.blocks = list(blocks or [])
                    self.url = 'https://example.invalid/report'

                def save(self):
                    Report.last_blocks, Report.last_title = self.blocks, self.title

                @classmethod
                def from_url(cls, url):
                    return cls(blocks=Report.last_blocks, title=Report.last_title)

            reports.H1, reports.PanelGrid, reports.LinePlot = H1, PanelGrid, LinePlot
            reports.Runset, reports.Report = Runset, Report
            package = types.ModuleType('wandb_workspaces')
            package.__path__ = []
            sub = types.ModuleType('wandb_workspaces.reports')
            sub.__path__ = []
            package.reports = sub
            sub.v2 = reports
            with patch.dict('sys.modules', {'wandb_workspaces': package,
                                            'wandb_workspaces.reports': sub,
                                            'wandb_workspaces.reports.v2': reports}):
                backend = WandbBackend.__new__(WandbBackend)
                backend.entity, backend.project = 'e', 'p'
                backend.run = types.SimpleNamespace(name='v05_seed42_fixture', id='run')
                url = backend.ensure_panels()
            self.assertEqual(url, 'https://example.invalid/report')
            self.assertIn('v05 SDS 1.1', Report.last_title)
            self.assertEqual(sum(isinstance(block, PanelGrid) for block in Report.last_blocks),
                             8)
            self.assertEqual(backend.panel_verification['remote_group_count'], 8)


class SampleAccountingTests(unittest.TestCase):
    def test_control_samples_and_physics_substeps_are_two_fields(self):
        accounting = sample_accounting(dict(control_samples=24576,
                                            physics_substeps=98304))
        self.assertEqual(accounting, dict(control_samples=24576, physics_substeps=98304))
        self.assertEqual(accounting['physics_substeps'], 4 * accounting['control_samples'])

    def test_legacy_physics_samples_increment_is_a_control_count(self):
        # The frozen callback field really counts control samples: map once.
        accounting = sample_accounting(dict(physics_samples_increment=512))
        self.assertEqual(accounting, dict(control_samples=512, physics_substeps=2048))
        self.assertEqual(accounting['physics_substeps'], 4 * 512)
        # A producer that already multiplied by the decimation is caught by the
        # per-update identity instead of being silently accepted.
        with self.assertRaises(ValueError):
            sample_accounting(dict(global_iteration=1, num_envs=512,
                                   control_samples=2048))
        with self.assertRaises(ValueError):
            sample_accounting(dict(control_samples=24576, physics_substeps=24576))
        with self.assertRaises(ValueError):
            sample_accounting(dict(physics_substeps=6))
        with self.assertRaises(ValueError):
            sample_accounting(dict(control_samples=True))

    def test_missing_counts_are_absent_instead_of_fabricated(self):
        accounting = sample_accounting(dict(global_iteration=1))
        self.assertEqual(accounting, dict(control_samples=None, physics_substeps=None))
        self.assertEqual(sample_weight(dict(num_envs=512, rollout=48)), 512 * 48)


class SeparationTests(unittest.TestCase):
    def test_namespaces_separate_training_evaluation_probe_and_diagnostics(self):
        self.assertEqual(record_namespace(dict(kind='train')), 'train')
        self.assertEqual(record_namespace(dict(kind='diagnostic')), 'diagnostic')
        self.assertEqual(record_namespace(dict(kind='diagnostic',
                                               execution='evaluation')), 'evaluation')
        self.assertEqual(record_namespace(dict(kind='validation')), 'evaluation')
        self.assertEqual(record_namespace(dict(probe=True)), 'probe')
        self.assertEqual(record_namespace(dict(execution='evaluation')), 'evaluation')
        self.assertEqual(record_namespace(dict(execution='probe')), 'probe')

    def test_training_gate_accepts_every_v05_stage_and_slot(self):
        from src.tasks.velocity.v05.spec import STAGE_ORDER
        for stage in STAGE_ORDER:
            row = dict(global_iteration=1, stage=stage, stage_iteration=1,
                       env_steps=48 * 512, num_envs=512, control_samples=24576)
            with self.subTest(stage=stage):
                self.assertIsNone(diagnostic_record_reason(row))
        for slot in ('flat', 'previous', 'current'):
            self.assertIsNone(diagnostic_record_reason(
                dict(global_iteration=1, stage='S1', slot_class=slot, num_envs=512,
                     control_samples=24576)))

    def test_training_gate_rejects_foreign_identity_and_case_records(self):
        base = dict(global_iteration=1, stage='F0', num_envs=512, control_samples=24576)
        for mutation, reason in (
                (dict(base, stage='S9'), 'non_v05_training_stage'),
                (dict(base, slot_class='previous_stairs'), 'non_v05_training_slot'),
                (dict(base, kind='validation'), 'non_training_namespace'),
                (dict(base, kind='diagnostic'), 'non_training_kind'),
                (dict(base, case_id='S1:validation:0.2:001'), 'non_training_namespace'),
                (dict(base, parking_diagnostic=True), 'non_training_namespace'),
                (dict(base, probe=True), 'non_training_namespace'),
                (dict(base, global_iteration=0), 'not_a_paid_training_update')):
            with self.subTest(reason=reason):
                self.assertTrue(diagnostic_record_reason(mutation).startswith(reason))

    def test_evaluation_metrics_use_case_pass_and_not_joint_success(self):
        summary = dict(stage='S1', kind='validation', global_update=200,
                       stage_iteration=200, env_steps=48 * 512 * 200, num_envs=512,
                       checkpoint_sha256='a' * 64, case_table_sha256='b' * 64,
                       control_samples=20 * 3 * 512, physics_substeps=4 * 20 * 3 * 512,
                       speeds=[dict(vx=.2, episodes=20, climb_successes=20,
                                    case_pass_count=19, speed_valid_count=19,
                                    speed_pass_count=18, joint_successes=17)])
        metrics = evaluation_metrics(summary)
        root = 'eval/validation/S1/vx_0.2'
        self.assertEqual(metrics[root + '/case_pass_count'], 19)
        self.assertEqual(metrics[root + '/case_pass_rate'], 19 / 20)
        self.assertEqual(metrics[root + '/joint_successes'], 17)
        self.assertEqual(metrics[root + '/episodes'], 20)
        self.assertEqual(metrics['env_steps'], 48 * 512 * 200)
        self.assertEqual(metrics['physics_substeps'], 4 * 20 * 3 * 512)
        self.assertEqual(metrics['eval/validation/S1/kind'], 'validation')
        bad = dict(summary, stage='S9')
        with self.assertRaises(ValueError):
            evaluation_metrics(bad)
        with self.assertRaises(ValueError):
            evaluation_metrics(dict(summary, kind='unknown'))
        with self.assertRaises(ValueError):
            evaluation_metrics(dict(summary, physics_substeps=1))
        flat = evaluation_metrics(dict(summary, stage='F0', kind='flat_retention',
                                       speeds=[]))
        self.assertEqual(flat['eval/flat_retention/F0/passed'], 0)

    def test_evaluation_and_probe_records_never_enter_training_aggregates(self):
        rows = [dict(global_iteration=1, stage='F0', num_envs=512, control_samples=24576,
                     ppo={'loss': 1.}, reward_sample_means={'reward_total': 2.}),
                dict(global_iteration=1, stage='F0', num_envs=512,
                     control_samples=24576, kind='validation', case_id='x',
                     ppo={'loss': 99.}),
                dict(global_iteration=1, stage='F0', num_envs=512,
                     control_samples=24576, probe=True, ppo={'loss': 77.})]
        result = aggregate_updates(rows, [])
        self.assertEqual(result['train/window_updates'], 1)
        self.assertEqual(result['ppo/loss'], 1.)
        self.assertEqual(result['accounting/control_samples'], 24576)
        self.assertEqual(result['accounting/physics_substeps'], 98304)


class AggregationTests(unittest.TestCase):
    def test_ppo_min_mean_max_and_reward_window_accounting(self):
        weight = 512 * 48
        rows = [dict(num_envs=512, global_iteration=1, stage='F0', stage_iteration=1,
                     env_steps=weight, control_samples=weight,
                     ppo={'kl_min': .001, 'kl_mean': .002, 'kl_max': .004,
                          'gradient_clip_ratio': .5, 'clip_fraction_mean': .3,
                          'std': {'00': .15}},
                     reward_sample_means={'reward_total': 2.,
                                          'reward_event/footfall': .5},
                     swing_metrics={'clearance_count': 2, 'mean_phi': .5},
                     foothold_metrics={'safe_support_count': 3},
                     diagnostic={'stall_penalty': -.005, 'stall_on': 1},
                     motion_terms={'groups': {'current': {'sample_count': 10,
                                                          'actual_vx_body': .3}}}),
                dict(num_envs=512, global_iteration=2, stage='F0', stage_iteration=2,
                     env_steps=weight, control_samples=weight,
                     ppo={'kl_min': .002, 'kl_mean': .004, 'kl_max': .008,
                          'gradient_clip_ratio': 1.0, 'clip_fraction_mean': .4},
                     reward_sample_means={'reward_total': 4.})]
        result = aggregate_updates(rows, [])
        self.assertEqual(result['ppo/kl_min'], .001)
        self.assertEqual(result['ppo/kl_max'], .008)
        self.assertAlmostEqual(result['ppo/kl_mean'], .003)
        # Ratio fields are averaged, never summed.
        self.assertAlmostEqual(result['ppo/gradient_clip_ratio'], .75)
        self.assertEqual(result['ppo/std/00'], .15)
        self.assertEqual(result['swing/clearance_count'], 2)
        self.assertEqual(result['foothold/safe_support_count'], 3)
        self.assertEqual(result['stall/stall_on'], 1)
        self.assertEqual(result['reward/window_total/reward_total'], 6. * weight)
        self.assertEqual(result['accounting/control_samples'], 2 * weight)
        self.assertEqual(result['accounting/physics_substeps'], 8 * weight)
        self.assertEqual(result['train/window_control_samples'], 2 * weight)
        self.assertEqual(result['train/window_physics_substeps'], 8 * weight)
        self.assertEqual(result['motion/groups/current/actual_vx_body'], .3)
        self.assertEqual(result['global_iteration'], 2)
        self.assertEqual(result['curriculum/stage'], 'F0')

    def test_episode_slots_and_case_pass(self):
        episodes = [
            dict(slot_class='current', climb_success=True, case_pass=True,
                 failure=False, return_=1., reward_sums={'reward_total': 1.},
                 duration=1.0, reason='climb'),
            dict(slot_class='current', climb_success=False, case_pass=False,
                 failure=True, reward_sums={'reward_total': -1.}, duration=.5,
                 reason='failure'),
            dict(slot_class='flat', climb_success=False, case_pass=True,
                 failure=False, reward_sums={'reward_total': .5}, duration=1.0,
                 reason='timeout'),
        ]
        result = aggregate_updates([], episodes)
        self.assertEqual(result['episode/current/ended_episodes'], 2)
        self.assertEqual(result['episode/current/climb_success_count'], 1)
        self.assertEqual(result['episode/current/case_pass_count'], 1)
        self.assertEqual(result['episode/flat/case_pass_count'], 1)
        self.assertNotIn('episode/flat/climb_success_count', result)
        self.assertEqual(result['episode/current/reason/climb'], .5)
        self.assertEqual(result['reward/episode_mean/current/reward_total'], 0.)
        self.assertAlmostEqual(result['reward/per_control_sample/current/reward_total'], 0.)


class ProtectionTests(SleepFree):
    def test_120_second_outage_requests_boundary_pause_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = Backend()
            pauses = []
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock,
                                        wall_clock=clock, pause=pauses.append)
            monitor.start()
            backend.hidden = True
            monitor.emit('train', {'global_iteration': 10})
            identifier = monitor.events[-1]['id']
            self.assertNotIn(identifier, monitor.state['acked'])
            backend.down = True
            clock.advance(119)
            monitor.flush()
            self.assertEqual(pauses, [])
            clock.advance(1)
            monitor.flush()
            self.assertEqual(pauses, ['monitoring_blocked'])
            self.assertTrue(monitor.state['monitoring_blocked'])
            backend.down = False
            backend.hidden = False
            backend.remote.add(identifier)
            self.assertTrue(monitor.flush())
            self.assertNotIn('monitoring_blocked', monitor.state)
            # The failed attempt is the same durable event; no fabricated ACK.
            self.assertEqual(len([event for event in backend.sent
                                  if event['kind'] == 'train']), 1)

    def test_local_enqueue_is_not_an_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = Backend()
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock,
                                        wall_clock=clock)
            monitor.start()
            backend.hidden = True
            monitor.emit('train', {'global_iteration': 10}, flush=False)
            identifier = monitor.events[-1]['id']
            self.assertFalse(monitor.flush())
            self.assertEqual(len(backend.sent), 2)  # initialize + train enqueued
            self.assertNotIn(identifier, monitor.state['acked'])
            self.assertFalse(monitor.flush())
            backend.hidden = False
            backend.remote.add(identifier)
            self.assertTrue(monitor.flush())
            self.assertIn(identifier, monitor.state['acked'])

    def test_resumed_session_never_resends_an_old_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = DelayedBackend(clock, delay=10)
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock,
                                        wall_clock=clock)
            monitor.emit('train', {'global_iteration': 10}, flush=False)
            self.assertFalse(monitor.flush())
            identifier = monitor.events[-1]['id']
            sent_before = len(backend.sent)
            resumed = ReliableTelemetry(directory, backend, {}, clock=clock,
                                        wall_clock=clock)
            self.assertFalse(resumed.flush())
            self.assertEqual(len(backend.sent), sent_before)
            self.assertTrue(resumed.state.get('queue_recovery_required'))
            clock.advance(10)
            self.assertTrue(resumed.flush())
            self.assertIn(identifier, resumed.state['acked'])
            self.assertEqual(len(backend.sent), sent_before)

    def test_heartbeat_kind_is_v05_and_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = Backend()
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock,
                                        wall_clock=clock)
            monitor.start()
            pump = DeliveryPump(monitor, lambda: {'global_iteration': 0}, clock=clock)
            for step in (30, 2, 28, 2, 28):
                clock.advance(step)
                pump.poll()
            heartbeats = [event for event in monitor.events
                          if event['kind'] == HEARTBEAT_KIND]
            self.assertEqual(len(heartbeats), 3)
            self.assertFalse([event for event in monitor.events
                              if event['kind'] in ('heartbeat', 'v04_heartbeat',
                                                   'authorized_fork_heartbeat')])

    def test_tick_uses_the_v05_heartbeat_and_never_a_stale_one(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            monitor = ReliableTelemetry(directory, Backend(), {}, clock=clock,
                                        wall_clock=clock)
            monitor.start()
            clock.advance(31)
            monitor.tick({'global_iteration': 0})
            self.assertTrue([event for event in monitor.events
                             if event['kind'] == HEARTBEAT_KIND])
            self.assertFalse([event for event in monitor.events
                              if event['kind'] in ('heartbeat', 'v04_heartbeat')])


class WorkerIntakeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.worker = self.tmp / 'worker'
        self.worker.mkdir()
        self.attestation = WorkerAttestation(
            run_id='seed42', mode='formal', seed=42, contract_sha256='a' * 64,
            num_envs=512, output_dir=str(self.worker),
            protocol_id='v05-stair-climb-first-v1')

    def write(self, name, rows):
        with (self.worker / name).open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    def append(self, name, rows):
        with (self.worker / name).open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    def test_attested_updates_are_ingested_and_other_records_quarantined(self):
        update = dict(global_iteration=1, stage='F0', stage_iteration=1,
                      env_steps=48 * 512, num_envs=512, control_samples=24576,
                      kind='train', run_id='seed42', seed=42,
                      contract_sha256='a' * 64, ppo={'loss': 1.})
        evaluation = dict(global_iteration=1, stage='F0', num_envs=512,
                          control_samples=24576, kind='validation', case_id='x')
        probe = dict(global_iteration=1, stage='F0', num_envs=512,
                     control_samples=24576, probe=True)
        self.write('updates.jsonl', [update, evaluation, probe])
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {})
            monitor.read_worker(self.worker, self.attestation)
            self.assertEqual(len(monitor.pending_updates), 1)
            self.assertEqual(monitor.pending_updates[0]['global_iteration'], 1)
            self.assertEqual(len(monitor.state['quarantine_ids']), 2)
            quarantined = [json.loads(line) for line in
                           (Path(directory) / 'quarantine.jsonl').read_text().splitlines()]
            self.assertEqual([row['reason'].split(':')[0] for row in quarantined],
                             ['non_training_namespace', 'non_training_namespace'])
            self.assertEqual(monitor.worker_progress(self.worker, {})['global_iteration'], 1)

    def test_worker_identity_and_progress_regressions_are_refused(self):
        self.write('updates.jsonl', [dict(global_iteration=2, stage='F0',
                                          stage_iteration=2, env_steps=48 * 512 * 2,
                                          num_envs=512, control_samples=24576)])
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {})
            monitor.read_worker(self.worker, self.attestation)
            self.append('updates.jsonl', [dict(global_iteration=1, stage='F0',
                                               stage_iteration=1, env_steps=48 * 512,
                                               num_envs=512, control_samples=24576)])
            with self.assertRaises(ValueError):
                monitor.read_worker(self.worker, self.attestation)
        wrong = WorkerAttestation(run_id='seed42', mode='formal', seed=43,
                                  contract_sha256='a' * 64, num_envs=512,
                                  output_dir=str(self.worker))
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {})
            self.write('updates.jsonl', [dict(global_iteration=1, stage='F0',
                                              stage_iteration=1, env_steps=48 * 512,
                                              num_envs=512, control_samples=24576,
                                              seed=42)])
            monitor.read_worker(self.worker, wrong)
            self.assertEqual(monitor.pending_updates, [])
            self.assertEqual(len(monitor.state['quarantine_ids']), 1)
        with self.assertRaises(ValueError):
            WorkerAttestation(run_id='seed42', mode='formal', seed=42,
                              contract_sha256='a' * 64, num_envs=8,
                              output_dir=str(self.worker)).validate()

    def test_cursor_prevents_double_ingestion(self):
        row = dict(global_iteration=1, stage='F0', stage_iteration=1,
                   env_steps=48 * 512, num_envs=512, control_samples=24576)
        self.write('updates.jsonl', [row])
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {})
            monitor.read_worker(self.worker, self.attestation)
            monitor.pending_updates = []
            monitor.read_worker(self.worker, self.attestation)
            self.assertEqual(len(monitor.pending_updates), 0)


class UtilityTests(unittest.TestCase):
    def test_atomic_json_is_fsync_backed_and_finite_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'value.json'
            atomic_json(path, {'a': 1, 'b': [1., 2.]})
            self.assertEqual(json.loads(path.read_text()), {'a': 1, 'b': [1., 2.]})
            with self.assertRaises(ValueError):
                atomic_json(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'a': 1, 'b': [1., 2.]})
            self.assertFalse(list(Path(directory).rglob('*.tmp')))

    def test_reliable_backend_readback_uses_the_v05_group(self):
        backend = ReliableWandbBackend.__new__(ReliableWandbBackend)
        backend.reader = types.SimpleNamespace(
            run_query=lambda fields: {'name': 'run', 'group': 'v05',
                                      'config': {'seed': 42, 'telemetry_run_id': 'run'}})
        self.assertTrue(backend.verify_start('run', {'seed': 42}))
        backend.reader.run_query = lambda fields: {'name': 'run', 'group': 'v04',
                                                   'config': {'seed': 42,
                                                              'telemetry_run_id': 'run'}}
        self.assertFalse(backend.verify_start('run', {'seed': 42}))

    def test_proxy_target_cannot_drift(self):
        import os
        with patch.dict(os.environ, {}, clear=True):
            configure_network()
            self.assertEqual(os.environ['WANDB_BASE_URL'],
                             'https://api.forge.coreweave.com')
            self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:7897')
            os.environ['WANDB_BASE_URL'] = 'https://example.invalid'
            with self.assertRaises(ValueError):
                configure_network()


if __name__ == '__main__':
    unittest.main()
