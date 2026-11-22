"""CPU-only v0.5-r2 telemetry tests: r2 identity, eight panels, route group.

No SDK run, network request or GPU is created.  The W&B constructor boundary is
replaced by a local fake whose ``send`` is an enqueue and whose ``remote_ids`` is
the only ACK source, so "local enqueue is not a remote ACK" is enforced by the
test fixture rather than by a comment.

r2 deltas under test (SDS 4.8/7.4/8.1):

* group ``v05_r2`` / heartbeat kind ``v05_r2_heartbeat``;
* exactly eight panel groups and ``route/*`` belongs to the existing
  ``运动/接触`` group, never a ninth group;
* the 120 s protection and the true remote ACK are unchanged;
* ``physics_substeps == 4 * control_samples`` and no decimation double count.
"""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch  # noqa: F401  (keeps the package import order identical to the product)

from src.tasks.velocity.v05_r2 import spec
from src.tasks.velocity.v05_r2 import telemetry as telemetry_module
from src.tasks.velocity.v05_r2.connection import (DeliveryPump, ReliableTelemetry,
                                                  ReliableWandbBackend, configure_network,
                                                  DELIVERY_TIMEOUT, HEARTBEAT_SECONDS as
                                                  CONNECTION_HEARTBEAT_SECONDS,
                                                  MAX_IN_FLIGHT, POLL_SECONDS)
from src.tasks.velocity.v05_r2.telemetry import (GROUP, HEARTBEAT_KIND, METRIC_PREFIXES,
                                                 PANELS, PANEL_COUNT, UNCONFIRMED_SECONDS,
                                                 MonitoringBlocked, Telemetry, WandbBackend,
                                                 aggregate_updates, evaluation_metrics,
                                                 sample_accounting, sample_weight)


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

    def next_transport_step(self):
        return len(self.sent)

    def verify_start(self, run_id, config):
        return True

    def ensure_panels(self, existing_url=None):
        self.panel_verification = dict(remote_verified=True, remote_group_count=PANEL_COUNT)
        return 'https://example.invalid/report'

    def finish(self, summary):
        pass


class SleepFree(unittest.TestCase):
    """Startup retries use the real 2 s sleep; the CPU suite stubs it."""

    def setUp(self):
        sleeper = patch('time.sleep', return_value=None)
        sleeper.start()
        self.addCleanup(sleeper.stop)


def assert_r2_identity(testcase, monitor):
    readiness = monitor.state['readiness']
    testcase.assertEqual(readiness['group'], 'v05_r2')
    testcase.assertEqual(readiness['heartbeat_kind'], 'v05_r2_heartbeat')
    testcase.assertTrue(readiness['remote_verified'])
    testcase.assertTrue(readiness['config_verified'])
    testcase.assertEqual(readiness['group'], GROUP)
    initialize = [event for event in monitor.events if event['kind'] == 'initialize']
    testcase.assertEqual(len(initialize), 1)
    testcase.assertEqual(readiness['initialize_event_id'], initialize[0]['id'])


class IdentityTests(SleepFree):
    def test_identity_comes_from_the_spec_monitoring_contract(self):
        self.assertEqual(GROUP, spec.contract()['monitoring']['group'])
        self.assertEqual(HEARTBEAT_KIND, spec.contract()['monitoring']['heartbeat_kind'])
        self.assertEqual(PANEL_COUNT, spec.contract()['monitoring']['panel_count'])
        self.assertEqual((GROUP, HEARTBEAT_KIND), ('v05_r2', 'v05_r2_heartbeat'))
        self.assertNotEqual(GROUP, 'v05')
        self.assertNotEqual(HEARTBEAT_KIND, 'v05_heartbeat')

    def test_readiness_and_single_initialize_are_r2(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {'seed': 42})
            identity = monitor.start()
            self.assertEqual(identity['run_id'], monitor.state['run_id'])
            assert_r2_identity(self, monitor)
            self.assertEqual(monitor.state['panels'], PANELS)
            self.assertEqual(monitor.state['panel_verification']['remote_group_count'],
                             PANEL_COUNT)

    def test_group_is_passed_to_the_sdk_and_verified_remotely(self):
        config = {'seed': 42}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = types.SimpleNamespace(
                settings=types.SimpleNamespace(mode='online', base_url='https://example.invalid'),
                project='p', entity='e', id='run', url='https://example.invalid/run',
                dir=str(root / 'sdk' / 'files'), step=0, summary={}, define_metric=Mock())
            remote = types.SimpleNamespace(config=config | {'telemetry_run_id': 'run'},
                                           group=GROUP)
            backend = WandbBackend.__new__(WandbBackend)
            backend.name = 'v05_r2_seed42_fixture'
            backend.project = 'p'
            backend.entity = 'e'
            backend.wandb = types.SimpleNamespace(
                init=Mock(return_value=run), Settings=lambda **kwargs: None,
                Api=lambda **kwargs: types.SimpleNamespace(run=lambda path: remote))
            backend.start('run', config)
            self.assertEqual(backend.wandb.init.call_args.kwargs.get('group'), 'v05_r2')
            self.assertTrue(backend.verify_start('run', config))
            remote.group = 'v05'
            self.assertFalse(backend.verify_start('run', config))

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
    def test_exactly_eight_panels_and_route_routing(self):
        self.assertEqual(len(PANELS), 8)
        self.assertEqual(PANEL_COUNT, 8)
        self.assertIn('route/*', PANELS['运动/接触'])
        self.assertIn('motion/*', PANELS['运动/接触'])
        # No ninth group and no route-named group was created.
        self.assertNotIn('route', PANELS)
        self.assertNotIn('路由', PANELS)
        self.assertEqual([title for title, patterns in PANELS.items()
                          if 'route/*' in patterns], ['运动/接触'])
        # Every aggregate namespace belongs to exactly one existing group.
        for prefix in METRIC_PREFIXES:
            owners = [title for title, patterns in PANELS.items()
                      for pattern in patterns
                      if pattern == prefix + '/*' or pattern.startswith(prefix + '/')]
            self.assertTrue(owners, prefix)
            self.assertEqual(len(owners), 1, prefix)
        self.assertIn('route', METRIC_PREFIXES)

    def test_pilot_stage_outside_its_budget_table_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Telemetry(directory, Backend(), {})
            # A pilot record claiming an S stage cannot borrow the formal table.
            with self.assertRaises(ValueError) as caught:
                monitor._progress_fields(dict(
                    global_iteration=10, stage='S1', stage_iteration=10,
                    env_steps=100, num_envs=512, mode='pilot'))
            self.assertIn('S1', str(caught.exception))
            self.assertIn('pilot', str(caught.exception))
            self.assertNotIn('KeyError', type(caught.exception).__name__)
            # A pilot F0/P1 record is accepted with the pilot caps.
            fields = monitor._progress_fields(dict(
                global_iteration=2400, stage='P1', stage_iteration=400,
                env_steps=100, num_envs=512, mode='pilot'))
            self.assertEqual(fields['curriculum/stage_budget'], 500)
            formal = monitor._progress_fields(dict(
                global_iteration=2400, stage='P1', stage_iteration=400,
                env_steps=100, num_envs=512, mode='formal'))
            self.assertEqual(formal['curriculum/stage_budget'], 1000)

    def test_no_r1_identity_remains_in_the_r2_monitoring_modules(self):
        for module in (telemetry_module,):
            source = Path(module.__file__).read_text()
            for forbidden in ("GROUP = 'v05'", "HEARTBEAT_KIND = 'v05_heartbeat'"):
                self.assertNotIn(forbidden, source)
        self.assertEqual(GROUP, 'v05_r2')
        self.assertEqual(HEARTBEAT_KIND, 'v05_r2_heartbeat')

    def test_route_aggregates_are_read_only_and_max_combined(self):
        rows = [
            dict(global_iteration=1, stage='P1', stage_iteration=1, num_envs=8,
                 control_samples=8 * 48,
                 route=dict(abs_y_mean=.10, abs_y_max=.15, normalized_abs_y_max=.75,
                            saturation_rate=0., heading_deg_mean=1., heading_deg_max=2.,
                            nonfinite_count=0)),
            dict(global_iteration=2, stage='P1', stage_iteration=2, num_envs=8,
                 control_samples=8 * 48,
                 route=dict(abs_y_mean=.20, abs_y_max=.35, normalized_abs_y_max=1.75,
                            saturation_rate=.05, heading_deg_mean=3., heading_deg_max=4.,
                            nonfinite_count=0)),
        ]
        result = aggregate_updates(rows, [])
        self.assertAlmostEqual(result['route/abs_y_mean'], .15)
        self.assertEqual(result['route/abs_y_max'], .35)
        self.assertEqual(result['route/normalized_abs_y_max'], 1.75)
        self.assertAlmostEqual(result['route/saturation_rate'], .025)
        self.assertEqual(result['route/heading_deg_max'], 4.)
        self.assertEqual(result['route/nonfinite_count'], 0)
        # route/* must not be classified into a pass namespace.
        self.assertFalse([key for key in result
                          if key.startswith(('ppo/', 'reward/', 'eval/'))
                          and key.split('/')[-1] in
                          ('abs_y_mean', 'abs_y_max', 'saturation_rate',
                           'heading_deg_mean', 'heading_deg_max', 'nonfinite_count')])

    def test_worker_ppo_nested_route_is_hoisted_out_of_ppo(self):
        row = dict(global_iteration=1, stage='P1', stage_iteration=1, num_envs=8,
                   control_samples=8 * 48,
                   ppo=dict(kl_mean=.01, lr=5e-5,
                            route=dict(abs_y_mean=.12, abs_y_max=.25,
                                       normalized_abs_y_max=1.25,
                                       saturation_rate=0., heading_deg_mean=2.,
                                       heading_deg_max=3., nonfinite_count=0)))
        result = aggregate_updates([row], [])
        self.assertAlmostEqual(result['route/abs_y_mean'], .12)
        self.assertEqual(result['route/normalized_abs_y_max'], 1.25)
        self.assertFalse([key for key in result if key.startswith('ppo/route')])

    def test_episode_route_evidence_is_aggregated(self):
        episodes = [
            dict(slot_class='current', duration=.02, route_max_abs_y=.10,
                 route_max_abs_heading_deg=2., route_final=[0., 0., 1.],
                 route_initial=[0.1, 0., 1.], reward_sums={'reward_total': 0.}),
            dict(slot_class='current', duration=.02, route_max_abs_y=.30,
                 route_max_abs_heading_deg=6., route_final=[1., 0., 1.],
                 route_initial=[0.2, 0., 1.], reward_sums={'reward_total': 0.}),
        ]
        result = aggregate_updates([], episodes)
        self.assertEqual(result['route/episode/current/max_abs_y_max'], .30)
        self.assertAlmostEqual(result['route/episode/current/max_abs_y_mean'], .20)
        self.assertEqual(result['route/episode/current/max_abs_heading_deg_max'], 6.)
        self.assertAlmostEqual(result['route/episode/current/route_final_0'], .5)
        self.assertAlmostEqual(result['route/episode/current/route_initial_1'], 0.)
        self.assertEqual(result['route/episode/current/route_final_episodes'], 2)

    def test_evaluation_metrics_carry_route_read_only(self):
        summary = dict(kind='validation', stage='S5', global_update=500, passed=True,
                       checkpoint_sha256='a' * 64, case_table_sha256='b' * 64,
                       route=dict(abs_y_max=.4, heading_deg_max=12., nonfinite_count=0),
                       speeds=[dict(vx=.2, episodes=20, case_pass_count=20,
                                    climb_successes=20, speed_valid_count=20,
                                    speed_pass_count=20)])
        metrics = evaluation_metrics(summary)
        self.assertAlmostEqual(metrics['eval/validation/S5/route/abs_y_max'], .4)
        self.assertAlmostEqual(metrics['eval/validation/S5/route/heading_deg_max'], 12.)
        self.assertEqual(metrics['eval/validation/S5/passed'], 1)


class AccountingTests(unittest.TestCase):
    def test_physics_substeps_is_exactly_four_times_control_samples(self):
        row = dict(num_envs=512, rollout=48, control_samples=512 * 48)
        accounting = sample_accounting(row)
        self.assertEqual(accounting['control_samples'], 24576)
        self.assertEqual(accounting['physics_substeps'], 98304)
        self.assertEqual(sample_weight(row), 24576)
        with self.assertRaises(ValueError):
            sample_accounting(dict(row, physics_substeps=24576))
        with self.assertRaises(ValueError):
            # A decimation double count inflates control_samples to 4x.
            sample_accounting(dict(num_envs=512, rollout=48,
                                   control_samples=4 * 512 * 48))
        legacy = sample_accounting(dict(num_envs=8, rollout=48,
                                        physics_samples_increment=8 * 48))
        self.assertEqual(legacy['physics_substeps'], 4 * 8 * 48)

    def test_window_accounting_uses_control_samples_once(self):
        rows = [dict(global_iteration=index + 1, stage='P1', stage_iteration=index + 1,
                     num_envs=8, control_samples=8 * 48, elapsed_s=1.)
                for index in range(2)]
        result = aggregate_updates(rows, [])
        self.assertEqual(result['accounting/control_samples'], 2 * 8 * 48)
        self.assertEqual(result['accounting/physics_substeps'], 4 * 2 * 8 * 48)
        self.assertEqual(result['accounting/physics_ratio'], 4)
        self.assertEqual(result['train/window_control_samples'], 2 * 8 * 48)
        self.assertEqual(result['train/window_physics_substeps'], 4 * 2 * 8 * 48)


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
            backend.hidden = False
            backend.remote.add(identifier)
            self.assertTrue(monitor.flush())
            self.assertIn(identifier, monitor.state['acked'])

    def test_transport_constants_are_unchanged(self):
        self.assertEqual(MAX_IN_FLIGHT, 32)
        self.assertEqual(POLL_SECONDS, 2)
        self.assertEqual(DELIVERY_TIMEOUT, 120)
        self.assertEqual(CONNECTION_HEARTBEAT_SECONDS, 30)
        self.assertEqual(UNCONFIRMED_SECONDS, 120)
        self.assertEqual(telemetry_module.HEARTBEAT_SECONDS, 30)
        self.assertEqual(telemetry_module.WINDOW_UPDATES, 10)

    def test_heartbeat_kind_is_r2_and_not_duplicated(self):
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
                              if event['kind'] in ('heartbeat', 'v05_heartbeat',
                                                   'v04_heartbeat')])


if __name__ == '__main__':
    unittest.main()
