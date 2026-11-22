"""CPU-only terminal recovery failures against the original durable sender."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('terminal_reconcile', ROOT / 'logs/a2_ideal_stairs_v02_formal/implementation/reconcile_terminal_online.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
from src.tasks.velocity.v02_formal.telemetry import Telemetry


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Backend:
    """Local commit and remote visibility are independent, as in the SDK."""
    def __init__(self, fixture, *, visible=True, verify_config=True, video_files=True):
        self.fixture = fixture
        self.project = 'project'; self.entity = 'entity'; self.remote_floor = 0
        self.sent = []; self.committed = []; self.finished = []
        self.ids = {e['id'] for e in fixture.events[:840]}
        self.visible = visible; self.verify_config = verify_config; self.video_files = video_files
        self.panel_verification = {}
        self.step = 840
        self.remote = SimpleNamespace(id=fixture.run_id, config=deepcopy(fixture.config) | {'telemetry_run_id': fixture.run_id}, summary={})
        self.remote.scan_history = self.scan
        self.remote.files = self.files
        self.wandb = SimpleNamespace(Api=lambda **kw: SimpleNamespace(run=lambda path: self.remote))
    def start(self, run_id, configuration):
        assert run_id == self.fixture.run_id
        self.run = SimpleNamespace(id=run_id, settings=SimpleNamespace(
            finish_timeout=float(os.environ['WANDB_FINISH_TIMEOUT']), finish_timeout_raises=True))
        return dict(project=self.project, entity=self.entity, run_id=run_id, report_url='report')
    def next_transport_step(self): return self.step
    def remote_ids(self): return set(self.ids)
    def send(self, event):
        # Reproduces explicit-step SDK behavior: an old step cannot be resent.
        if event['transport_step'] < self.step:
            raise AssertionError('SDK dropped an old transport step')
        self.step = event['transport_step'] + 1
        self.committed.append(event['id']); self.sent.append(deepcopy(event))
        if self.visible: self.ids.add(event['id'])
    def ensure_panels(self, existing_url=None):
        self.panel_verification = dict(remote_verified=True, remote_group_count=8, report_url='report')
        return 'report'
    def finish(self, summary):
        self.finished.append(deepcopy(summary)); self.remote.summary.update(summary)
        if not self.verify_config: self.remote.config['seed'] = 17
    def scan(self, **kwargs):
        rows = []
        for event in self.fixture.events:
            if event['id'] in self.ids:
                row = {'event_id': event['id']} | event['metrics']
                for key, path in event.get('videos', {}).items():
                    row[key] = {'path': 'media/videos/' + Path(path).parent.name + '_' + Path(path).name}
                rows.append(row)
        return rows
    def files(self, names):
        return [SimpleNamespace(name=name, size=20) for name in names] if self.video_files else []


class Fixture:
    def __init__(self, directory):
        self.run = Path(directory) / 'run'; self.run.mkdir()
        (self.run / 'supervisor.lock').touch()
        self.run_id = '37b559428d704af09bf16d7ed646595d'
        self.checkpoint = self.run / 'formal_seed42/chunk_0006_n512/checkpoints/budget_exhausted.pt'
        self.checkpoint.parent.mkdir(parents=True); self.checkpoint.write_bytes(b'CPU-only synthetic checkpoint; strict loader injected')
        self.sha = helper.digest(self.checkpoint)
        self.contract = {'behavior_sha256': {'protected.py': 'unchanged'}, 'source_c_sha256': 'original_C'}
        approval = self.run / 'approval.json'; approval.write_text('{}')
        self.config = dict(wandb_mode='online', wandb_project='project', wandb_entity='entity', wandb_run_name='name',
                           review_report=str(approval), seed=42, contract=self.contract)
        self.curriculum = dict(iteration=3000, env_steps=73728000, stage='R0', stage_iteration=3000, consumed={'R0': 3000})
        self.payload = dict(contract=self.contract, curriculum=self.curriculum, num_envs=512, metadata={'seed': 42},
                            learning_state=dict(optimizer_initialized=True, completed_ppo_updates=3000, optimizer_steps=60000),
                            optimizer={'state': dict.fromkeys(range(43), {})})
        result = dict(status='budget_exhausted', checkpoint=str(self.checkpoint), sha256=self.sha,
                      curriculum=self.curriculum, global_iteration=3000, env_steps=73728000, probe=False)
        helper.write(self.checkpoint.parent.parent / 'result.json', result)
        helper.write(self.checkpoint.parent.parent / 'budget.json', {'curriculum': self.curriculum})
        metrics = {'global_iteration': 3000, 'env_steps': 73728000, 'result/status': 'budget_exhausted', 'result/last_sha256': self.sha}
        self.events = [dict(sequence=i, id=f'{self.run_id}:{i:012d}', kind='initialize' if i == 1 else 'heartbeat',
                            metrics={'global_iteration': 0 if i == 1 else 3000}, videos={}, created=float(i), source_ids=[])
                       for i in range(1, 846)]
        videos = {}
        for speed in ('0.2', '0.4', '0.6'):
            path = self.run / 'stop_replay' / ('vx_' + speed) / 'replay.mp4'
            path.parent.mkdir(parents=True); path.write_bytes(b'CPU mock video bytes')
            videos['video/R0/vx_' + speed] = str(path)
        self.events[-2].update(kind='videos', videos=videos, metrics={
            'global_iteration': 3000, 'env_steps': 73728000, 'video/checkpoint_sha256': self.sha})
        self.events[-1].update(kind='stop', metrics=metrics)
        tdir = self.run / 'wandb_telemetry'; tdir.mkdir()
        (tdir / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in self.events))
        self.telemetry = dict(delivery_schema=2, run_id=self.run_id, sequence=845,
                              acked=[event['id'] for event in self.events[:840]], config=self.config,
                              identity=dict(run_id=self.run_id, project='project', entity='entity', report_url='report'),
                              sdk_spools={}, cursors={}, transport={}, next_transport_step=840,
                              pending_updates=[], pending_episodes=[])
        helper.write(tdir / 'state.json', self.telemetry)
        consumption = [dict(iteration=3000, env_steps=73728000, sha256=self.sha)]
        self.final = dict(status='budget_exhausted', iteration=3000, env_steps=73728000,
                          last_sha256=self.sha, last_checkpoint=str(self.checkpoint), curriculum=self.curriculum,
                          consumption=consumption, result=result,
                          diagnostics={'replay': {'status': 'complete'}, 'wandb_finish': {'status': 'unconfirmed', 'sdk_finish': 'requested', 'event_id': self.events[-1]['id']}})
        self.state = dict(phase='terminal', status='budget_exhausted', iteration=3000, env_steps=73728000,
                          checkpoint=str(self.checkpoint), sha256=self.sha, terminal=self.final,
                          curriculum=self.curriculum, config=self.config, num_envs=512,
                          consumption_ledger=consumption, source_hashes=self.contract['behavior_sha256'])
        helper.write(self.run / 'supervisor_state.json', self.state); helper.write(self.run / 'final.json', self.final)
        helper.write(self.run / 'active_supervisor.json', {'pid': 987654321})
        helper.write(self.run / 'child_process.json', {'pid': 987654322})
        self.args = SimpleNamespace(run_dir=str(self.run), expected_run_id=self.run_id, expected_status='budget_exhausted',
                                    expected_iteration=3000, expected_env_steps=73728000,
                                    expected_checkpoint_sha256=self.sha, expected_event_count=845,
                                    expected_ledger_sha256=helper.digest(tdir / 'events.jsonl'),
                                    max_seconds=6., poll_seconds=2., evidence=str(Path(directory) / 'receipt.json'))
    def preflight(self, args, **kwargs):
        return helper.preflight(args, pid_alive=kwargs.pop('pid_alive', lambda pid: False),
                                gpu_query=kwargs.pop('gpu_query', lambda: set()), checkpoint_loader=lambda *a: self.payload,
                                current_contract=lambda: self.contract,
                                approved_sources=lambda path: self.contract['behavior_sha256'], **kwargs)


class TerminalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)
    def run_helper(self, backend, **kwargs):
        clock = Clock()
        def telemetry(directory, backend, config, **kw):
            return Telemetry(directory, backend, config, clock=clock, wall_clock=lambda: 1000 + clock(), **kw)
        with patch('src.tasks.velocity.v02_formal.telemetry.time.sleep', clock.sleep):
            return helper.reconcile(self.f.args, backend_factory=lambda *a: backend,
                                    telemetry_factory=telemetry, check=self.f.preflight,
                                    clock=clock, sleep=clock.sleep, limit=lambda seconds: nullcontext(), **kwargs)
    def before(self):
        return {name: (self.f.run / name).read_bytes() for name in ['supervisor_state.json', 'final.json', 'wandb_telemetry/events.jsonl']}
    def test_success_uses_original_sender_and_only_promotes_finish_diagnostic(self):
        before = self.before(); model = self.f.checkpoint.read_bytes()
        backend = Backend(self.f)
        audit = self.run_helper(backend)
        self.assertTrue(audit['passed']); self.assertEqual(len(backend.finished), 1)
        self.assertEqual(backend.finished[0], self.f.events[-1]['metrics'])
        self.assertEqual([e['id'] for e in backend.sent], [e['id'] for e in self.f.events[840:]])
        self.assertEqual((self.f.run / 'wandb_telemetry/events.jsonl').read_bytes(), before['wandb_telemetry/events.jsonl'])
        self.assertEqual(self.f.checkpoint.read_bytes(), model)
        final = helper.read(self.f.run / 'final.json'); state = helper.read(self.f.run / 'supervisor_state.json')
        self.assertEqual(final['diagnostics']['wandb_finish']['status'], 'remote_verified')
        self.assertEqual(final['diagnostics']['wandb_finish']['original_unconfirmed'], self.f.final['diagnostics']['wandb_finish'])
        final['diagnostics']['wandb_finish'] = self.f.final['diagnostics']['wandb_finish']
        self.assertEqual(final, self.f.final)
        state['terminal'] = self.f.final; self.assertEqual(state, self.f.state)
        self.assertEqual((Path(audit['backups']) / 'final.json').read_bytes(), before['final.json'])
        self.assertEqual(len(audit['remote_verification']['video_files']), 3)
    def test_timeout_local_commit_is_not_remote_ack_or_terminal_confirmation(self):
        before = self.before(); backend = Backend(self.f, visible=False)
        audit = self.run_helper(backend)
        self.assertFalse(audit['passed']); self.assertEqual(len(backend.sent), 1)
        self.assertEqual(len(helper.read(self.f.run / 'wandb_telemetry/state.json')['acked']), 840)
        self.assertEqual(backend.finished, [])
        self.assertEqual(self.before(), before)
    def test_old_uncertain_intent_is_not_resent_at_new_sdk_step(self):
        state = self.f.telemetry
        state['transport'][self.f.events[840]['id']] = dict(step=840, session='dead_previous_session', sdk_spool_directory=None)
        helper.write(self.f.run / 'wandb_telemetry/state.json', state)
        backend = Backend(self.f); audit = self.run_helper(backend)
        self.assertFalse(audit['passed']); self.assertEqual(backend.sent, [])
        self.assertEqual(helper.read(self.f.run / 'final.json'), self.f.final)
    def spool_fixture(self):
        spool = self.f.run / 'old-sdk-spool'; spool.mkdir()
        (spool / 'run-old.wandb').write_bytes(b'synthetic durable queue')
        data = self.f.telemetry
        data['transport'][self.f.events[840]['id']] = dict(step=840, session='dead_previous_session', sdk_spool_directory=str(spool))
        data['next_transport_step'] = 841
        data['sdk_spools'][str(spool)] = dict(run_id=self.f.run_id, entity='entity', project='project',
            sdk_spool_directory=str(spool), sdk_spool_files=[str(spool / 'run-old.wandb')],
            sdk_owner_pid=987654321, sdk_owner_start_time=0.)
        helper.write(self.f.run / 'wandb_telemetry/state.json', data)
        self.f.args.max_seconds = 60.
        return spool
    def test_owned_dead_spool_uses_original_sync_and_remote_readback_without_resend(self):
        spool = self.spool_fixture(); backend = Backend(self.f); backend.step = 841
        commands = []
        def sync(command, **kwargs):
            commands.append(command); backend.ids.add(self.f.events[840]['id'])
            return SimpleNamespace(returncode=0)
        with patch('subprocess.run', sync): audit = self.run_helper(backend)
        self.assertTrue(audit['passed']); self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][-1], str(spool)); self.assertIn('--include-online', commands[0])
        self.assertEqual([event['id'] for event in backend.sent], [event['id'] for event in self.f.events[841:]])
        self.assertEqual(len(helper.read(self.f.run / 'wandb_telemetry/state.json')['acked']), 845)
    def test_sync_exit_zero_without_remote_proof_does_not_fake_ack(self):
        self.spool_fixture(); backend = Backend(self.f); before = self.before()
        with patch('subprocess.run', lambda *args, **kwargs: SimpleNamespace(returncode=0)):
            audit = self.run_helper(backend)
        self.assertFalse(audit['passed']); self.assertEqual(backend.sent, [])
        self.assertEqual(len(helper.read(self.f.run / 'wandb_telemetry/state.json')['acked']), 840)
        self.assertEqual(self.before(), before)
    def test_wrong_remote_config_prevents_confirmation_after_sdk_finish(self):
        before = self.before(); backend = Backend(self.f, verify_config=False)
        audit = self.run_helper(backend)
        self.assertFalse(audit['passed']); self.assertEqual(len(backend.finished), 1)
        self.assertEqual(self.before(), before)
    def test_missing_remote_video_bytes_prevents_confirmation(self):
        backend = Backend(self.f, video_files=False)
        self.assertFalse(self.run_helper(backend)['passed'])
        self.assertEqual(helper.read(self.f.run / 'final.json'), self.f.final)
    def test_unexpected_remote_logical_event_refuses_promotion(self):
        backend = Backend(self.f); backend.ids.add(self.f.run_id + ':000000000846')
        self.assertFalse(self.run_helper(backend)['passed'])
        self.assertEqual(helper.read(self.f.run / 'final.json'), self.f.final)
    def test_partial_terminal_persist_failure_restores_original_bytes(self):
        before = self.before()
        def partial(run, state, final):
            changed = deepcopy(state); changed['terminal'] = final
            helper.write(run / 'supervisor_state.json', changed)
            raise OSError('simulated second atomic write failure')
        audit = self.run_helper(Backend(self.f), persist=partial)
        self.assertFalse(audit['passed']); self.assertTrue(audit['diagnostic_partial_write_rolled_back'])
        self.assertTrue(audit['primary_documents_unchanged']); self.assertEqual(self.before(), before)
    def test_unexpected_primary_mutation_is_never_blindly_rolled_back(self):
        def wrong(run, state, final):
            changed = deepcopy(state); changed['iteration'] = 2999
            helper.write(run / 'supervisor_state.json', changed)
            raise OSError('not a diagnostic-only write')
        audit = self.run_helper(Backend(self.f), persist=wrong)
        self.assertFalse(audit['passed']); self.assertTrue(audit['manual_recovery_required'])
        self.assertFalse(audit['primary_documents_unchanged'])
        self.assertEqual(helper.read(self.f.run / 'supervisor_state.json')['iteration'], 2999)
    def test_sdk_finish_timeout_cannot_be_promoted_by_later_noop_finish(self):
        before = self.before(); backend = Backend(self.f)
        def finish(summary):
            backend.finished.append(summary); raise TimeoutError()
        backend.finish = finish
        audit = self.run_helper(backend)
        self.assertFalse(audit['passed']); self.assertEqual(audit['failure_type'], 'SDKFinishUnconfirmed')
        self.assertEqual(len(backend.finished), 1); self.assertEqual(self.before(), before)
        self.assertTrue(audit['sdk_finish_settings_captured']['finish_timeout_raises'])
    def test_alive_owner_rejected_before_backend_construction(self):
        with self.assertRaisesRegex(ValueError, 'alive'):
            self.f.preflight(self.f.args, pid_alive=lambda pid: True)
    def test_live_cuda_owner_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.f.preflight(self.f.args, gpu_query=lambda: {987654322})
    def test_wrong_expected_sha_rejected(self):
        self.f.args.expected_checkpoint_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.f.preflight(self.f.args)
    def test_corrupt_model_bytes_rejected(self):
        self.f.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes'):
            self.f.preflight(self.f.args)
    def test_nonterminal_rejected(self):
        self.f.state['phase'] = 'train'; helper.write(self.f.run / 'supervisor_state.json', self.f.state)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            self.f.preflight(self.f.args)
    def test_pending_training_queues_rejected(self):
        for key in ('pending_updates', 'pending_episodes'):
            with self.subTest(key=key):
                data = deepcopy(self.f.telemetry); data[key] = [{'kind': 'train'}]
                helper.write(self.f.run / 'wandb_telemetry/state.json', data)
                with self.assertRaisesRegex(ValueError, 'queues'):
                    self.f.preflight(self.f.args)
    def test_wrong_run_id_rejected(self):
        self.f.args.expected_run_id = 'other'
        with self.assertRaisesRegex(ValueError, 'run ID'):
            self.f.preflight(self.f.args)
    def test_missing_initialization_or_duplicate_stop_rejected(self):
        self.f.events[0]['kind'] = 'heartbeat'
        path = self.f.run / 'wandb_telemetry/events.jsonl'
        path.write_text(''.join(json.dumps(e) + '\n' for e in self.f.events))
        self.f.args.expected_ledger_sha256 = helper.digest(path)
        with self.assertRaisesRegex(ValueError, 'initialization'):
            self.f.preflight(self.f.args)
    def test_missing_or_wrong_terminal_video_rejected(self):
        self.f.events[-2]['videos'].pop('video/R0/vx_0.6')
        path = self.f.run / 'wandb_telemetry/events.jsonl'
        path.write_text(''.join(json.dumps(e) + '\n' for e in self.f.events))
        self.f.args.expected_ledger_sha256 = helper.digest(path)
        with self.assertRaisesRegex(ValueError, 'three exact'):
            self.f.preflight(self.f.args)
    def test_duplicate_terminal_stop_rejected(self):
        self.f.events[840]['kind'] = 'stop'
        path = self.f.run / 'wandb_telemetry/events.jsonl'
        path.write_text(''.join(json.dumps(e) + '\n' for e in self.f.events))
        self.f.args.expected_ledger_sha256 = helper.digest(path)
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.f.preflight(self.f.args)
    def test_wrong_adam_and_budget_refused(self):
        self.f.payload['learning_state']['optimizer_steps'] = 59999
        with self.assertRaisesRegex(ValueError, 'Adam'):
            self.f.preflight(self.f.args)
    def test_wrong_ledger_sha_refused_before_any_sender(self):
        self.f.args.expected_ledger_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'ledger SHA'):
            self.f.preflight(self.f.args)
    def test_contract_source_or_pending_revision_refused(self):
        self.f.state['source_hashes'] = {'protected.py': 'changed'}
        helper.write(self.f.run / 'supervisor_state.json', self.f.state)
        with self.assertRaisesRegex(ValueError, 'approval'):
            self.f.preflight(self.f.args)
        self.f.state['source_hashes'] = self.f.contract['behavior_sha256']
        helper.write(self.f.run / 'supervisor_state.json', self.f.state)
        self.f.telemetry['pending_source_revision'] = {'anything': True}
        helper.write(self.f.run / 'wandb_telemetry/state.json', self.f.telemetry)
        with self.assertRaisesRegex(ValueError, 'pending config'):
            self.f.preflight(self.f.args)
    def test_real_alarm_interrupts_bounded_cpu_wait(self):
        import time
        with self.assertRaises(helper.DeadlineExpired):
            with helper.deadline(.01): time.sleep(.1)
    def test_installed_sdk_captures_official_finish_timeout_without_writer(self):
        import wandb
        settings = wandb.Settings()
        # wandb_setup.py invokes this step; bare Settings() does not load env.
        settings.update_from_env_vars({'WANDB_FINISH_TIMEOUT': '17', 'WANDB_FINISH_TIMEOUT_RAISES': 'true'})
        self.assertEqual(settings.finish_timeout, 17.)
        self.assertIs(settings.finish_timeout_raises, True)
    def test_concurrent_lock_refused(self):
        import fcntl
        with (self.f.run / 'supervisor.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_helper(Backend(self.f))
    def test_evidence_cannot_overwrite_existing_receipt(self):
        Path(self.f.args.evidence).write_text('old evidence')
        with self.assertRaisesRegex(ValueError, 'new reconciliation'):
            self.run_helper(Backend(self.f))
    def test_expired_deadline_keeps_primary_unconfirmed(self):
        before = self.before()
        def interrupted(*args): raise helper.DeadlineExpired()
        backend = Backend(self.f); backend.remote_ids = interrupted
        audit = self.run_helper(backend)
        self.assertFalse(audit['passed']); self.assertEqual(audit['failure_type'], 'DeadlineExpired')
        self.assertEqual(self.before(), before)


if __name__ == '__main__':
    unittest.main()
