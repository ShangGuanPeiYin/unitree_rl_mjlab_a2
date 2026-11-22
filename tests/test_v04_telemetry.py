"""CPU-only v04 monitoring identity, readiness and delivery regressions.

No SDK run, network request or GPU is created.  The real W&B constructor is
replaced at its documented boundary; local queue state is never called remote
ACK.
"""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v04_bootstrap
a2_v04_bootstrap.activate(str(ROOT))

from src.tasks.velocity.v04 import telemetry as telemetry_module
from src.tasks.velocity.v04.telemetry import (
    PANELS, MonitoringBlocked, Telemetry, WandbBackend, aggregate_updates, evaluation_metrics,
)
from src.tasks.velocity.v04.connection import (
    DeliveryPump, ReliableTelemetry, ReliableWandbBackend, configure_network,
)


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, value):
        self.value += value


class Backend:
    """Local fake: send means enqueue/visibility, never a remote service."""

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
        self.panel_verification = dict(remote_verified=True, remote_group_count=8)
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
        return {event_id for when, event_id in self.visible if when <= self.clock()}


def assert_v04_identity(testcase, monitor):
    identity = monitor.state['identity']
    testcase.assertEqual(identity['run_id'], monitor.state['run_id'])
    readiness = monitor.state['readiness']
    testcase.assertEqual(readiness['group'], 'v04')
    testcase.assertTrue(readiness['remote_verified'])
    testcase.assertTrue(readiness['config_verified'])
    testcase.assertEqual(readiness['run_id'], monitor.state['run_id'])
    initialize = [e for e in monitor.events if e['kind'] == 'initialize']
    testcase.assertEqual(len(initialize), 1)
    testcase.assertEqual(readiness['initialize_event_id'], initialize[0]['id'])


class IdentityTests(unittest.TestCase):
    def setUp(self):
        sleeper = patch('time.sleep', return_value=None)
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def test_v04_group_is_created_and_verified_with_real_constructor_boundary(self):
        config = {'seed': 42, 'contract': {'protocol_id': 'v04-stair-body-speed-v1'}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = types.SimpleNamespace(
                settings=types.SimpleNamespace(mode='online', base_url='https://example.invalid'),
                project='p', entity='e', id='run', url='https://example.invalid/run',
                dir=str(root / 'sdk' / 'files'), step=0, summary={}, define_metric=Mock())
            remote = types.SimpleNamespace(config=config | {'telemetry_run_id': 'run'}, group='v04')
            backend = WandbBackend.__new__(WandbBackend)
            backend.name = 'v04_seed42_fixture'
            backend.project = 'p'
            backend.entity = 'e'
            backend.wandb = types.SimpleNamespace(
                init=Mock(return_value=run), Settings=lambda **kwargs: None,
                Api=lambda **kwargs: types.SimpleNamespace(run=lambda path: remote))
            identity = backend.start('run', config)
            self.assertEqual(backend.wandb.init.call_args.kwargs.get('group'), 'v04')
            self.assertEqual(identity['run_id'], 'run')
            self.assertTrue(backend.verify_start('run', config))
            remote.group = 'v03'
            self.assertFalse(backend.verify_start('run', config))
            remote.group = 'v04'
            remote.config['seed'] = 43
            self.assertFalse(backend.verify_start('run', config))

    def test_readiness_readback_and_single_initialize_are_v04(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            monitor = Telemetry(directory, backend, {'seed': 42})
            identity = monitor.start()
            self.assertEqual(identity['run_id'], monitor.state['run_id'])
            assert_v04_identity(self, monitor)
            self.assertEqual(len(backend.starts), 1)
            self.assertEqual(monitor.state['panels'], PANELS)
            self.assertEqual(monitor.state['panel_verification']['remote_group_count'], 8)

    def test_reliable_start_readiness_is_v04(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = Backend()
            monitor = ReliableTelemetry(directory, backend, {'seed': 42},
                                        clock=clock, wall_clock=clock)
            monitor.start()
            assert_v04_identity(self, monitor)
            self.assertEqual(monitor.state['readiness']['group'], 'v04')

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
                backend.ensure_panels = lambda existing_url=None: 'https://example.invalid/report'
                backend.panel_verification = dict(
                    remote_verified=True,
                    remote_group_count=7 if failure == 'panels' else 8)
                with self.assertRaises(MonitoringBlocked):
                    Telemetry(directory, backend, {}).start()

    def test_trusted_worker_progress_accepts_only_fixed_512(self):
        row = dict(global_iteration=1, stage_iteration=1, env_steps=512 * 48,
                   num_envs=512, stage='S1')
        self.assertEqual(Telemetry._progress_fields(row)['num_envs'], 512)
        for count in (1024, 258, 8):
            with self.subTest(num_envs=count):
                with self.assertRaises(ValueError):
                    Telemetry._progress_fields(dict(row, num_envs=count))

    def test_exactly_eight_panels_and_v04_metrics_are_in_existing_groups(self):
        self.assertEqual(len(PANELS), 8)
        self.assertIn('reward/*', PANELS['分项奖励'])
        self.assertIn('reward_event/*', PANELS['分项奖励'])
        self.assertIn('motion/*', PANELS['运动/接触'])
        self.assertIn('swing/*', PANELS['运动/接触'])
        self.assertIn('foothold/*', PANELS['运动/接触'])
        self.assertNotIn('swing', PANELS)
        self.assertNotIn('foothold', PANELS)

    def test_report_title_is_v04_sds_1_1_and_eight_group_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = types.ModuleType('wandb_workspaces.reports.v2')

            class H1:
                def __init__(self, title): self.title = title

            class PanelGrid:
                def __init__(self, runsets, panels):
                    self.runsets, self.panels = runsets, panels

            class LinePlot:
                def __init__(self, **kwargs): self.kwargs = kwargs

            class Runset:
                def __init__(self, **kwargs): self.kwargs = kwargs

            class Report:
                last_blocks = None
                last_title = None
                def __init__(self, entity=None, project=None, title=None, blocks=None, width=None):
                    self.entity, self.project, self.title = entity, project, title
                    self.blocks = list(blocks or []); self.url = 'https://example.invalid/report'
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
            with patch.dict(sys.modules, {
                    'wandb_workspaces': package,
                    'wandb_workspaces.reports': sub,
                    'wandb_workspaces.reports.v2': reports}):
                backend = WandbBackend.__new__(WandbBackend)
                backend.entity, backend.project = 'e', 'p'
                backend.run = types.SimpleNamespace(name='v04_seed42_fixture', id='run')
                url = backend.ensure_panels()
            self.assertEqual(url, 'https://example.invalid/report')
            self.assertIn('v04 SDS 1.1', Report.last_title)
            self.assertEqual(sum(isinstance(b, PanelGrid) for b in Report.last_blocks), 8)
            self.assertEqual(backend.panel_verification['remote_group_count'], 8)


class ProtectionTests(unittest.TestCase):
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
            event_id = monitor.events[-1]['id']
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
            backend.remote.add(event_id)
            self.assertTrue(monitor.flush())
            self.assertNotIn('monitoring_blocked', monitor.state)
            # The failed attempt is the same durable event; no fabricated ACK.
            self.assertEqual(len([e for e in backend.sent if e['kind'] == 'train']), 1)

    def test_resumed_session_never_resends_old_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = DelayedBackend(clock, delay=10)
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock, wall_clock=clock)
            monitor.emit('train', {'global_iteration': 10}, flush=False)
            self.assertFalse(monitor.flush())
            event_id = monitor.events[-1]['id']
            sent_before = len(backend.sent)
            resumed = ReliableTelemetry(directory, backend, {}, clock=clock, wall_clock=clock)
            self.assertFalse(resumed.flush())
            self.assertEqual(len(backend.sent), sent_before)
            self.assertTrue(resumed.state.get('queue_recovery_required'))
            clock.advance(10)
            self.assertTrue(resumed.flush())
            self.assertIn(event_id, resumed.state['acked'])
            self.assertEqual(len(backend.sent), sent_before)
            self.assertEqual(len({e['id'] for e in backend.sent}), sent_before)

    def test_heartbeat_kind_is_v04_and_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            backend = Backend()
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock, wall_clock=clock)
            monitor.start()
            pump = DeliveryPump(monitor, lambda: {'global_iteration': 0}, clock=clock)
            for step in (30, 2, 28, 2, 28):
                clock.advance(step)
                pump.poll()
            heartbeats = [e for e in monitor.events if e['kind'] == 'v04_heartbeat']
            self.assertEqual(len(heartbeats), 3)
            self.assertFalse([e for e in monitor.events if e['kind'] == 'authorized_fork_heartbeat'])


class EvaluationMetricsTests(unittest.TestCase):
    def test_v04_evaluation_identity_keeps_joint_namespace_and_denominator(self):
        result = evaluation_metrics(dict(
            stage='S1', kind='validation', iteration=200, stage_iteration=200,
            env_steps=2476800, num_envs=512, checkpoint_sha256='a' * 64,
            case_table_sha256='b' * 64, protocol_id='v04-stair-body-speed-v1',
            speeds=[dict(vx=.2, episodes=20, climb_successes=20, speed_valid_count=19,
                         speed_pass_count=18, joint_successes=17)], passed=False))
        root = 'eval/joint/validation/S1/vx_0.2'
        self.assertEqual(result[root + '/joint_successes'], 17)
        self.assertEqual(result[root + '/episodes'], 20)
        self.assertEqual(result['env_steps'], 2476800)
        self.assertEqual(result['eval/protocol_id'], 'v04-stair-body-speed-v1')


class AggregationTests(unittest.TestCase):
    def test_swing_and_foothold_metrics_reach_the_aggregate_without_reward_drift(self):
        weight = 512 * 48
        rows = [dict(num_envs=512, global_iteration=1, env_steps=weight,
                     sample_count=weight,
                     swing_metrics={'clearance_count': 2, 'mean_phi': .5},
                     foothold_metrics={'safe_support_count': 3, 'dwell_seconds': .2},
                     diagnostic_metrics={'reason_count': 1},
                     reward_sample_means={'reward_total': 2.,
                                          'reward_event/swing_clearance': .1},
                     motion_terms={'groups': {'current_stairs/moving': {
                         'sample_count': 10, 'actual_vx_body': .3}}})]
        result = aggregate_updates(rows, [])
        self.assertEqual(result['swing/clearance_count'], 2)
        self.assertEqual(result['swing/mean_phi'], .5)
        self.assertEqual(result['foothold/safe_support_count'], 3)
        self.assertAlmostEqual(result['foothold/dwell_seconds'], .2)
        self.assertEqual(result['diagnostic/reason_count'], 1)
        self.assertEqual(result['reward/window_total/reward_total'], 2. * weight)
        self.assertEqual(result['reward/window_total/reward_event/swing_clearance'], .1 * weight)
        self.assertEqual(result['reward/reward_total'], 2.)
        self.assertEqual(result['train/window_samples'], weight)


class ConnectionBoundaryTests(unittest.TestCase):
    def test_reliable_backend_readback_uses_v04_group(self):
        backend = ReliableWandbBackend.__new__(ReliableWandbBackend)
        backend.reader = types.SimpleNamespace(
            run_query=lambda fields: {'name': 'run', 'group': 'v04',
                                      'config': {'seed': 42, 'telemetry_run_id': 'run'}})
        self.assertTrue(backend.verify_start('run', {'seed': 42}))
        backend.reader.run_query = lambda fields: {'name': 'run', 'group': 'v03',
                                                   'config': {'seed': 42, 'telemetry_run_id': 'run'}}
        self.assertFalse(backend.verify_start('run', {'seed': 42}))

    def test_proxy_target_cannot_drift(self):
        with patch.dict('os.environ', {}, clear=True):
            import os
            configure_network()
            self.assertEqual(os.environ['WANDB_BASE_URL'], 'https://api.forge.coreweave.com')
            self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:7897')
            os.environ['WANDB_BASE_URL'] = 'https://example.invalid'
            with self.assertRaises(ValueError):
                configure_network()


if __name__ == '__main__':
    unittest.main()


class TickHeartbeatIdentityTests(unittest.TestCase):
    def test_tick_uses_v04_heartbeat_and_never_inherited_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock(); backend = Backend()
            monitor = ReliableTelemetry(directory, backend, {}, clock=clock, wall_clock=clock)
            monitor.start()
            clock.advance(31)
            monitor.tick({'global_iteration': 0})
            self.assertTrue([e for e in monitor.events if e['kind'] == 'v04_heartbeat'])
            self.assertFalse([e for e in monitor.events if e['kind'] in ('heartbeat', 'authorized_fork_heartbeat')])
