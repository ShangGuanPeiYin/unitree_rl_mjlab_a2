"""CPU-only v0.5-r2 supervisor / pilot-sequence / W7 probe tests.

No simulator, GPU, W&B session or child process is started.  The supervisor's
``child_runner`` is a fake and the preflight directory is redirected to a
temporary directory, so these tests prove the r2 identity and budget contract
without touching ``logs/a2_ideal_stairs_v05_r2/``.

Covered (SDS 7.1/7.3/8.2 and Q003-r2 3.8):

* pilot reports bind the r2 identity and report ``total_budget == update_cap ==
  2500`` from ``spec.mode_budget('pilot')``; formal admission wants two passing
  pilots and ETH-only initialization;
* the pilot sequence is strictly seed 42 then 43, and a seed-42 failure still
  runs 43 without admitting formal;
* the W7 probe plan is exactly 5 + 50 + 1 = 56 learning updates /
  1,376,256 control samples, with evaluation and S5 no-learning cost listed
  separately and never counted against the formal 30000;
* preflight (no ``--execute``) writes one record under
  ``<log-root>/implementation/preflight``, creates no run directory and starts
  no child/GPU/network work;
* the worker command carries the r2 script name, ``--max-updates`` effective cap
  and the r2 ledger run id.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch  # noqa: F401  (keeps the import order identical to the product)

from src.tasks.velocity.v05_r2 import orchestration as module
from src.tasks.velocity.v05_r2 import spec
from src.tasks.velocity.v05_r2 import evaluation as evaluation_module
from src.tasks.velocity.v05_r2 import recovery
from src.tasks.velocity.v05_r2.recovery import AttemptLedger
from src.tasks.velocity.v05_r2.runner import route_diagnostics as route_block
from src.tasks.velocity.v05_r2.orchestration import (FORMAL_SEED, PILOT_SEEDS,
                                                     PROBE_CONTROL_SAMPLES,
                                                     PROBE_LEARNING_UPDATES,
                                                     FormalAdmissionError, PilotError,
                                                     PilotSequence, Supervisor,
                                                     admit_formal, make_pilot_report,
                                                     parser, pilot_binding, probe_plan,
                                                     probe_statistics, require_pilot_pair,
                                                     resource_guard,
                                                     validate_final_test_summary,
                                                     validate_pilot_report,
                                                     validate_run_contract,
                                                     final_test_request)

DIGEST = 'a' * 64


def verdict(*, status='passed', gates=None, seed=42):
    gates = ({100: None, 200: None, 300: None, 400: True, 500: True}
             if gates is None else gates)
    return dict(mode='pilot', seed=seed, stage='P1', stage_iteration=500,
                gates=gates, f0_consumed=2000, p1_consumed=500,
                total_updates=2500, status=status,
                reasons=[] if status == 'passed' else ['p1_gate_500_not_passed'])


def pilot_report(seed=42, *, status='passed', gates=None):
    return make_pilot_report(
        seed=seed, status=status, verdict=verdict(status=status, gates=gates, seed=seed),
        checkpoint_sha256=DIGEST,
        evidence_index=[dict(path='/tmp/trial.pt', sha256=DIGEST, kind='trial_checkpoint')])


def pipeline_args(mode='pilot', **overrides):
    values = dict(mode=mode, seed=42, device='cuda:0', num_envs=512, eval_num_envs=8,
                  init_checkpoint=spec.INIT_PATH,
                  init_checkpoint_sha256=spec.INIT_SHA256,
                  vx_min=.2, vx_max=.6, vy=0., wz=0., save_interval=100,
                  diagnostic_interval=50,
                  stage_budgets=(spec.PILOT_STAGE_BUDGETS if mode == 'pilot'
                                 else spec.FORMAL_STAGE_BUDGETS),
                  output_dir='/tmp/r2-orchestration-output')
    if mode == 'pilot':
        values['pilot_p1_updates'] = spec.PILOT_P1_UPDATES
    if mode == 'formal':
        values['pilot_report_42'] = '/tmp/report42.json'
        values['pilot_report_43'] = '/tmp/report43.json'
    values.update(overrides)
    argv = []
    for key, value in values.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return parser().parse_args(argv)


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_is_r2_and_bound(self):
        binding = pilot_binding()
        self.assertEqual(binding['task'], 'Unitree-A2-IdealHeightMap-Stairs-V05R2')
        self.assertEqual(binding['recipe_version'], '0.5-r2')
        self.assertEqual(binding['code_version'], '0.5.1')
        self.assertEqual(binding['sds_revision'], '1.0')
        self.assertEqual(binding['checkpoint_schema'], 7)
        self.assertEqual(binding['architecture'], 'v05-route-state-v1')
        self.assertEqual(binding['protocol_id'], 'v05-r2-stair-climb-first-v1')
        self.assertEqual(binding['source_sha'], spec.contract_sha256())
        self.assertEqual(binding['total_budget'], spec.mode_budget('formal')['total'])
        self.assertIn('binding_sha256', binding)

    def test_pilot_report_budget_fields_are_the_effective_cap(self):
        report = pilot_report()
        self.assertEqual(report['schema'], 'v05-r2-pilot-report-v1')
        self.assertEqual(report['total_budget'], 2500)
        self.assertEqual(report['update_cap'], 2500)
        self.assertEqual(report['max_updates'], 2500)
        self.assertEqual(report['total_budget'], report['update_cap'])
        self.assertEqual(report['stage_budgets'], {'F0': 2000, 'P1': 500})
        self.assertEqual(report['effective_cap_source'], 'mode_budget')
        self.assertEqual(report['architecture'], 'v05-route-state-v1')
        self.assertEqual(report['f0']['budget'], 2000)
        self.assertEqual(report['p1']['budget'], 500)
        self.assertEqual(validate_pilot_report(report), report)

    def test_report_metadata_cannot_be_tampered(self):
        for field, value in (('total_budget', 29500), ('update_cap', 30000),
                             ('max_updates', 30000), ('effective_cap_source', 'stage_sum'),
                             ('stage_budgets', {'F0': 2000, 'P1': 1000}),
                             ('architecture', 'a2-ideal-heightmap-v1'),
                             ('checkpoint_schema', 6)):
            with self.subTest(field=field):
                report = pilot_report()
                report = json.loads(json.dumps(report))
                report[field] = value
                with self.assertRaises(PilotError):
                    validate_pilot_report(report)

    def test_stale_binding_is_rejected(self):
        report = pilot_report()
        stale = dict(pilot_binding(), recipe_version='0.5-r1')
        with self.assertRaises(PilotError):
            validate_pilot_report(report, binding_value=stale)

    def test_a_passing_report_needs_both_final_gates(self):
        failed_gate = pilot_report(status='failed',
                                   gates={100: True, 200: True, 300: True,
                                          400: True, 500: False})
        self.assertEqual(failed_gate['status'], 'failed')
        with self.assertRaises(PilotError):
            # A "passed" status with a missing 500 gate cannot be constructed.
            make_pilot_report(seed=42, status='passed',
                              verdict=verdict(gates={100: True, 200: True, 300: True,
                                                     400: True, 500: None}),
                              checkpoint_sha256=DIGEST,
                              evidence_index=[dict(path='/tmp/t', sha256=DIGEST)])

    def test_pair_requires_both_seeds_and_two_passes(self):
        reports = {42: pilot_report(42), 43: pilot_report(43)}
        self.assertEqual(set(require_pilot_pair(reports)), {42, 43})
        with self.assertRaises(PilotError):
            require_pilot_pair({42: pilot_report(42)})
        failed = {42: pilot_report(42), 43: pilot_report(43, status='failed')}
        with self.assertRaises(PilotError):
            require_pilot_pair(failed)

    def test_formal_admission_only_from_eth_after_both_pilots(self):
        reports = {42: pilot_report(42), 43: pilot_report(43)}
        admission = admit_formal(reports=reports,
                                 init_checkpoint_sha256=spec.INIT_SHA256)
        self.assertEqual(admission['status'], 'admitted')
        self.assertEqual(admission['seed'], 42)
        self.assertEqual(admission['total_budget'], 30000)
        self.assertEqual(admission['update_cap'], 30000)
        self.assertEqual(admission['effective_cap_source'], 'mode_budget')
        with self.assertRaises(FormalAdmissionError):
            admit_formal(reports=reports, init_checkpoint_sha256=DIGEST)
        with self.assertRaises(FormalAdmissionError):
            admit_formal(reports=reports, init_checkpoint_sha256=spec.INIT_SHA256,
                         resume_run=True)
        with self.assertRaises(FormalAdmissionError):
            admit_formal(reports=reports, init_checkpoint_sha256=spec.INIT_SHA256,
                         pilot_checkpoint_shas=(spec.INIT_SHA256,))


class PilotSequenceTests(unittest.TestCase):
    def test_sequence_is_seed42_then_seed43_and_keeps_a_failure(self):
        calls = []

        def trial(seed, fresh=True):
            calls.append((seed, fresh))
            return dict(status='passed' if seed == 42 else 'failed',
                        verdict=verdict(status='passed' if seed == 42 else 'failed',
                                        seed=seed),
                        checkpoint_sha256=DIGEST,
                        evidence_index=[dict(path='/tmp/t', sha256=DIGEST)],
                        run_id=f'pilot_seed{seed}')

        result = PilotSequence(trial=trial).run()
        self.assertEqual([seed for seed, _ in calls], [42, 43])
        self.assertTrue(all(fresh for _, fresh in calls))
        self.assertFalse(result['formal_ready'])
        self.assertEqual(result['reports'][42]['status'], 'passed')
        self.assertEqual(result['reports'][43]['status'], 'failed')

    def test_both_passing_pilots_are_formal_ready(self):
        def trial(seed, fresh=True):
            return dict(status='passed', verdict=verdict(seed=seed),
                        checkpoint_sha256=DIGEST,
                        evidence_index=[dict(path='/tmp/t', sha256=DIGEST)])

        result = PilotSequence(trial=trial).run()
        self.assertTrue(result['formal_ready'])
        self.assertEqual(result['status'], 'pilot_complete')
        with self.assertRaises(PilotError):
            PilotSequence(trial=lambda seed, fresh=True: {'status': 'passed'}).run()


class ProbeTests(unittest.TestCase):
    def test_probe_plan_is_fifty_six_updates_never_formal_30000(self):
        plan = probe_plan()
        self.assertEqual(module.W7_WARMUP_UPDATES, 5)
        self.assertEqual(module.W7_MEASURED_UPDATES, 50)
        self.assertEqual(module.W7_RECOVERY_UPDATES, 1)
        self.assertEqual(plan['learning_updates'], PROBE_LEARNING_UPDATES)
        self.assertEqual(plan['learning_updates'], 56)
        self.assertEqual(plan['control_samples'], 1376256)
        self.assertEqual(plan['control_samples'], 56 * 48 * 512)
        self.assertEqual(plan['physics_substeps'], 4 * plan['control_samples'])
        self.assertEqual(plan['adam_steps'], 1120)
        self.assertEqual(plan['total_budget'], 56)
        self.assertEqual(plan['update_cap'], 56)
        self.assertEqual(plan['effective_cap_source'], 'mode_budget')
        self.assertFalse(plan['counts_against_formal'])
        self.assertFalse(plan['formal_updates_started'])
        self.assertEqual(plan['formal_budget_not_started'], 30000)
        self.assertNotEqual(plan['control_samples'], 30000 * 48 * 512)
        # Evaluation and S5 no-learning cost are separate, non-learning entries.
        evaluation = plan['evaluation_lifecycle']
        s5 = plan['s5_three_speed_timeout_eval']
        self.assertFalse(evaluation['learning'])
        self.assertFalse(evaluation['counted_against_formal'])
        self.assertFalse(s5['learning'])
        self.assertFalse(s5['counted_against_formal'])
        self.assertGreater(s5['control_samples'], 0)
        self.assertEqual(spec.mode_budget('probe')['total'], 56)

    def test_probe_statistics_counts_only_the_measured_window(self):
        timings = [1. + index for index in range(56)]
        statistics = probe_statistics(timings, completed=56)
        self.assertEqual(statistics['warmup_completed'], 5)
        self.assertEqual(statistics['measured_completed'], 50)
        self.assertEqual(statistics['recovery_completed'], 1)
        self.assertEqual(statistics['measured_iterations'], [6, 55])
        self.assertFalse(statistics['counts_against_formal'])
        with self.assertRaises(ValueError):
            probe_statistics(timings, completed=57)

    def test_resource_guard_thresholds_and_no_downgrade(self):
        ok = dict(gpu_used_mib=7168, process_gpu_mib=5120,
                  host_available_mib=4096, disk_free_gib=10)
        self.assertTrue(resource_guard(ok)['passed'])
        for field, bad in (('gpu_used_mib', 7169), ('process_gpu_mib', 5121),
                           ('host_available_mib', 4095), ('disk_free_gib', 9.9)):
            with self.subTest(field=field):
                result = resource_guard(dict(ok, **{field: bad}))
                self.assertFalse(result['passed'])
                self.assertEqual(result['limits'],
                                 dict(whole_device_mib=7168, process_mib=5120,
                                      host_available_mib=4096, disk_free_gib=10))
        with self.assertRaises(RuntimeError):
            module.ensure_disk_space('/tmp', minimum_gib=10 ** 9)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.preflight = self.root / 'preflight'

    def supervisor(self, args, *, child=None):
        return Supervisor(args, preflight_dir=self.preflight,
                          child_runner=child)

    def test_pilot_preflight_writes_one_record_and_starts_nothing(self):
        out = self.root / 'pilot_run'
        args = pipeline_args('pilot', output_dir=str(out))
        supervisor = self.supervisor(args)
        record = supervisor.preflight()
        self.assertEqual(record['status'], 'preflight_validated')
        self.assertEqual(record['task'], 'Unitree-A2-IdealHeightMap-Stairs-V05R2')
        self.assertEqual(record['protocol_id'], 'v05-r2-stair-climb-first-v1')
        self.assertEqual(record['checkpoint_schema'], 7)
        self.assertEqual(record['total_budget'], 2500)
        self.assertEqual(record['update_cap'], 2500)
        self.assertEqual(record['effective_cap_source'], 'mode_budget')
        self.assertEqual(record['stage_budget_table'],
                         {'F0': 2000, 'P1': 500})
        self.assertFalse(record['execute'])
        self.assertFalse(record['training_started'])
        self.assertFalse(record['gpu_started'])
        self.assertFalse(record['network_started'])
        self.assertFalse(out.exists())
        written = json.loads(
            (self.preflight / 'preflight_pilot_seed42.json').read_text())
        self.assertEqual(written['binding_sha256'], record['binding_sha256'])

    def test_run_without_execute_returns_preflight_and_never_calls_a_child(self):
        out = self.root / 'pilot_run'
        calls = []
        args = pipeline_args('pilot', output_dir=str(out))
        supervisor = self.supervisor(
            args, child=lambda command, directory: calls.append((command, directory)))
        result = supervisor.run()
        self.assertEqual(result['status'], 'preflight_validated')
        self.assertEqual(calls, [])
        self.assertFalse(out.exists())

    def test_formal_preflight_reports_the_30000_cap(self):
        out = self.root / 'formal_run'
        args = pipeline_args('formal', output_dir=str(out))
        record = self.supervisor(args).preflight()
        self.assertEqual(record['total_budget'], 30000)
        self.assertEqual(record['update_cap'], 30000)
        self.assertEqual(record['stage_budget_table'],
                         spec.mode_budget('formal')['stages'])
        self.assertFalse(out.exists())

    def test_formal_preflight_without_pilot_reports_is_refused(self):
        out = self.root / 'formal_run'
        args = pipeline_args('formal', output_dir=str(out),
                             pilot_report_42=None, pilot_report_43=None)
        with self.assertRaises(ValueError):
            self.supervisor(args).preflight()
        self.assertFalse(out.exists())
        self.assertFalse(self.preflight.exists())

    def test_probe_preflight_reports_56_not_30000(self):
        out = self.root / 'probe_run'
        args = pipeline_args('probe', output_dir=str(out))
        record = self.supervisor(args).preflight()
        self.assertEqual(record['total_budget'], 56)
        self.assertEqual(record['update_cap'], 56)
        self.assertEqual(record['probe_learning_updates'], 56)
        self.assertFalse(out.exists())

    def test_pilot_preflight_refuses_a_wrong_stage_budget(self):
        out = self.root / 'pilot_run'
        args = pipeline_args('pilot', output_dir=str(out),
                             stage_budgets=spec.FORMAL_STAGE_BUDGETS)
        with self.assertRaises(ValueError):
            self.supervisor(args).preflight()

    def test_max_updates_assertion_accepts_each_mode_effective_cap(self):
        cases = (('pilot', 2500), ('formal', 30000), ('probe', 56))
        for mode, cap in cases:
            with self.subTest(mode=mode):
                out = self.root / f'{mode}_run'
                args = pipeline_args(mode, output_dir=str(out), max_updates=cap)
                self.assertEqual(args.max_updates, cap)
                record = self.supervisor(args).preflight()
                self.assertEqual(record['total_budget'], cap)
                self.assertEqual(record['update_cap'], cap)
                self.assertEqual(record['max_updates'], cap)
                self.assertFalse(out.exists())

    def test_max_updates_assertion_rejects_every_other_value(self):
        cases = (('pilot', 30000), ('pilot', 29500), ('pilot', 2000),
                 ('formal', 2500), ('formal', 29500), ('probe', 30000),
                 ('probe', 2500))
        for index, (mode, cap) in enumerate(cases):
            with self.subTest(mode=mode, cap=cap):
                out = self.root / f'bad_{index}'
                args = pipeline_args(mode, output_dir=str(out), max_updates=cap)
                with self.assertRaises(ValueError) as caught:
                    self.supervisor(args).preflight()
                self.assertIn('max-updates', str(caught.exception))
                self.assertFalse(out.exists())
        # The parser itself refuses a non-integer instead of coercing it.
        with self.assertRaises(SystemExit):
            pipeline_args('pilot', max_updates='2500.0')

    def test_worker_command_is_r2_and_carries_the_effective_cap(self):
        args = pipeline_args('pilot', output_dir=str(self.root / 'pilot_run'))
        supervisor = self.supervisor(args)
        command = supervisor.worker_command(self.root / 'trial', seed=42, mode='pilot')
        joined = ' '.join(command)
        self.assertIn('train_a2_ideal_stairs_v05_r2.py', joined)
        self.assertIn('--max-updates 2500', joined)
        self.assertIn('--ledger-run-id v05-r2-pilot-seed42', joined)
        self.assertIn('--stage-budgets F0=2000,P1=500', joined)
        self.assertIn('--pilot-p1-updates 500', joined)
        self.assertIn('--task Unitree-A2-IdealHeightMap-Stairs-V05R2', joined)
        self.assertNotIn('29500', joined)
        formal = supervisor.worker_command(self.root / 'formal', seed=42, mode='formal')
        self.assertIn('--max-updates 30000', ' '.join(formal))
        self.assertEqual(supervisor.ledger_run_id('formal', 42),
                         'v05-r2-formal-seed42')

    def test_run_contract_is_r2_and_rejects_changed_cli(self):
        out = self.root / 'run'
        out.mkdir()
        args = pipeline_args('pilot', output_dir=str(out))
        supervisor = self.supervisor(args)
        record = supervisor.write_run_contract()
        self.assertEqual(record['schema'], 'v05-r2-run-contract-v1')
        self.assertEqual(record['mode'], 'pilot')
        self.assertEqual(validate_run_contract(args, record), record)
        changed = pipeline_args('pilot', output_dir=str(out), seed=43)
        with self.assertRaises(ValueError):
            validate_run_contract(changed, record)


class FakeRngEnv:
    def __init__(self):
        self.rng = torch.Generator().manual_seed(5)


def synthetic_pilot_checkpoint(path, seed, *, passed=True):
    """A genuine schema-7 pilot checkpoint (synthetic tensors, real envelope)."""
    if passed:
        curriculum = dict(stage='P1', stage_iteration=500,
                          spent={name: 0 for name in spec.STAGE_ORDER},
                          streak=dict(gate=2, collapse=0), pending_transition=None)
        curriculum['spent']['F0'] = 2000
        curriculum['spent']['P1'] = 500
        results = [dict(task_key=f'k{update}', kind='validation', stage='P1',
                        global_update=update, outcome='passed', passed=True,
                        gate_streak=2, collapse_streak=0)
                   for update in (2400, 2500)]
        ledger_state = dict(tasks={}, results=results, collapse_streak=0,
                            gate_streak=2, anchor=dict(seq=0, digest='a' * 64),
                            qualified_update=None, pending_transition=None)
        learning = recovery.learning_state(2500, num_envs=512)
    else:
        curriculum = recovery.genesis_curriculum()
        ledger_state = dict(tasks={}, results=[], collapse_streak=0, gate_streak=0,
                            anchor=dict(seq=0, digest='a' * 64),
                            qualified_update=None, pending_transition=None)
        learning = recovery.learning_state(0, num_envs=512)
    model_state = {
        'actor.route_fusion.weight': torch.zeros(512, 3),
        'actor.route_fusion.bias': torch.zeros(512),
        'critic.0.weight': torch.zeros(512, 211),
    }
    state = recovery.build_checkpoint(
        model_state=model_state, optimizer_state={}, normalizer={}, learning=learning,
        rng=recovery.capture_rng(FakeRngEnv()), curriculum=curriculum,
        ledger=ledger_state, evaluation=dict(consumed=[], latest_diagnostic_update=None),
        source=recovery.source_block(mode='pilot', seed=seed, device='cpu'),
        mode='pilot', seed=seed, device='cpu')
    path = Path(path)
    recovery.atomic_torch_save(path, state)
    return dict(path=str(path), sha256=recovery.sha256(path))


class FakeBackend:
    """Immediate remote visibility backend for the supervisor telemetry session."""
    def __init__(self, *args, **kwargs):
        self.remote = set()
        self.sent = []
        self.panel_verification = None

    def start(self, run_id, config):
        return dict(project='p', entity='e', run_id=run_id,
                    url='https://example.invalid/' + run_id,
                    base_url='https://example.invalid')

    def remote_ids(self):
        return set(self.remote)

    def send(self, event):
        self.sent.append(event)
        self.remote.add(event['id'])

    def next_transport_step(self):
        return len(self.sent)

    def verify_start(self, run_id, config):
        return True

    def ensure_panels(self, existing_url=None):
        self.panel_verification = dict(remote_verified=True,
                                       remote_group_count=module.PANEL_COUNT
                                       if hasattr(module, 'PANEL_COUNT') else 8)
        return 'https://example.invalid/report'

    def finish(self, summary):
        pass


class SupervisorRunTests(SupervisorTests):
    def _run(self, args, child, **kwargs):
        supervisor = Supervisor(args, child_runner=child,
                                backend_factory=FakeBackend,
                                preflight_dir=self.preflight,
                                **kwargs)
        with patch.object(module, 'gpu_queue_guard',
                          lambda device: nullcontext(None)):
            return supervisor, supervisor.run()

    def test_pilot_sequence_runs_seed42_then_43_and_writes_reports(self):
        out = self.root / 'pilot_run'
        checkpoints = {}
        calls = []

        def child(command, directory):
            mode = 'pilot'
            seed = 42 if 'pilot_seed42' in str(directory) else 43
            calls.append(seed)
            if seed not in checkpoints:
                checkpoints[seed] = synthetic_pilot_checkpoint(
                    self.root / f'ckpt{seed}.pt', seed, passed=True)
            return dict(status='passed', checkpoint=checkpoints[seed]['path'],
                        sha256=checkpoints[seed]['sha256'])

        args = pipeline_args('pilot', output_dir=str(out), execute=True)
        supervisor, result = self._run(args, child)
        self.assertEqual(calls, [42, 43])
        self.assertEqual(result['status'], 'pilot_complete')
        self.assertTrue(result['formal_ready'])
        self.assertEqual(result['total_budget'], 2500)
        self.assertEqual(result['update_cap'], 2500)
        self.assertTrue((out / 'run_contract.json').is_file())
        self.assertTrue((out / 'pilots' / 'pilot_seed42_report.json').is_file())
        self.assertTrue((out / 'pilots' / 'pilot_seed43_report.json').is_file())
        self.assertTrue((out / 'final.json').is_file())
        report = json.loads((out / 'pilots' / 'pilot_seed42_report.json').read_text())
        self.assertEqual(report['schema'], 'v05-r2-pilot-report-v1')
        self.assertEqual(report['total_budget'], 2500)
        self.assertEqual(report['update_cap'], 2500)
        contract = json.loads((out / 'run_contract.json').read_text())
        self.assertEqual(contract['schema'], 'v05-r2-run-contract-v1')
        self.assertEqual(contract['binding']['architecture'], 'v05-route-state-v1')

    def test_seed42_failure_still_runs_seed43_without_formal_readiness(self):
        out = self.root / 'pilot_run'
        calls = []

        def child(command, directory):
            seed = 42 if 'pilot_seed42' in str(directory) else 43
            calls.append(seed)
            check = synthetic_pilot_checkpoint(self.root / f'ckpt{seed}.pt', seed,
                                               passed=(seed == 43))
            return dict(status='passed' if seed == 43 else 'failed',
                        checkpoint=check['path'], sha256=check['sha256'])

        args = pipeline_args('pilot', output_dir=str(out), execute=True)
        supervisor, result = self._run(args, child)
        self.assertEqual(calls, [42, 43])
        self.assertFalse(result['formal_ready'])
        self.assertEqual(result['reports'][42]['status'], 'failed')
        self.assertEqual(result['reports'][43]['status'], 'passed')

    def test_formal_run_requires_and_binds_both_pilot_reports(self):
        out = self.root / 'formal_run'
        reports = {}
        for seed in (42, 43):
            path = self.root / f'report{seed}.json'
            report = pilot_report(seed)
            path.write_text(json.dumps(report))
            reports[seed] = str(path)
        calls = []

        def child(command, directory):
            calls.append((command, directory))
            return dict(status='passed', final_test=dict(passed=True))

        args = pipeline_args('formal', output_dir=str(out), execute=True,
                             pilot_report_42=reports[42],
                             pilot_report_43=reports[43])
        supervisor, result = self._run(args, child)
        self.assertEqual(len(calls), 1)
        joined = ' '.join(calls[0][0])
        self.assertIn('train_a2_ideal_stairs_v05_r2.py', joined)
        self.assertIn('--max-updates 30000', joined)
        self.assertIn('--ledger-run-id v05-r2-formal-seed42', joined)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['total_budget'], 30000)
        self.assertEqual(result['update_cap'], 30000)
        admission = result['admission']
        self.assertEqual(admission['status'], 'admitted')
        self.assertEqual(admission['total_budget'], 30000)
        contract = json.loads((out / 'run_contract.json').read_text())
        self.assertEqual(set(contract['pilot_reports']), {'42', '43'})


class ProbeClock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        self.value += .25
        return self.value


class ProbeModel(torch.nn.Module):
    """Minimal model with the exact r2 schema-7 tensor identities (CPU)."""
    def __init__(self):
        super().__init__()
        self.actor_weight = torch.nn.Parameter(torch.tensor(1.))
        self.actor = torch.nn.Module()
        self.actor.route_fusion = torch.nn.Module()
        self.actor.route_fusion.weight = torch.nn.Parameter(torch.zeros(512, 3))
        self.actor.route_fusion.bias = torch.nn.Parameter(torch.zeros(512))
        self.actor.normalizer = torch.nn.Module()
        self.actor.normalizer.register_buffer('_mean', torch.zeros(1, 45))
        self.actor.normalizer.register_buffer('_var', torch.ones(1, 45))
        self.actor.normalizer.register_buffer('_std', torch.ones(1, 45))
        self.actor.normalizer.register_buffer('count', torch.tensor(0.))
        self.critic = torch.nn.ModuleList([torch.nn.Linear(211, 512)])
        self.critic_normalizer = torch.nn.Module()
        self.critic_normalizer.register_buffer('mean', torch.zeros(80))
        self.critic_normalizer.register_buffer('var', torch.ones(80))
        self.critic_normalizer.register_buffer('count', torch.tensor(0., dtype=torch.float64))
        mask = torch.ones(80, dtype=torch.bool)
        mask[48:52] = False
        self.critic_normalizer.register_buffer('continuous', mask)
        self.std = torch.nn.Parameter(torch.full((12,), spec.INITIAL_STD))

    def update_critic_statistics(self, obs):
        return None

    def constrain_std(self):
        self.std.data.clamp_(spec.STD_MIN, spec.STD_MAX)


class ProbeEnv:
    device = 'cpu'

    def __init__(self, num_envs=512, **_):
        self.num_envs = int(num_envs)
        self.rng = torch.Generator().manual_seed(3)
        self.stage = 'P1'
        self.stage_iteration = 0
        self.steps = 0

    def _obs(self):
        route = torch.zeros(self.num_envs, 3)
        route[:, 2] = 1.
        route[:, 0] = .01 * (self.steps % 40)
        return {'proprio': torch.zeros(self.num_envs, 45),
                'history': torch.zeros(self.num_envs, 5, 45),
                'heightmap': torch.zeros(self.num_envs, 3, 1, 1),
                'privileged': torch.zeros(self.num_envs, 35),
                'route_error': route}

    def set_stage(self, stage, stage_iteration=0, previous_stage=None):
        self.stage = stage
        self.stage_iteration = int(stage_iteration)

    def reset(self):
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        obs = self._obs()
        episode = dict(slot_class='current', duration=.02,
                       route_final=[.1, 0., 1.], route_initial=[0., 0., 1.],
                       route_max_abs_y=.1, route_max_abs_heading_deg=2.,
                       reward_sums={'reward_total': 0.})
        return obs, torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), \
            torch.zeros(self.num_envs, dtype=torch.bool), {
                'terminal_observation': obs, 'episodes': [episode],
                'reward_terms': {'speed': 0.}, 'diagnostics': {}}

    def close(self):
        pass


class ProbePPO:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.cfg = type('Cfg', (), dict(rollout=48, epochs=5, minibatches=4,
                                        learning_rate=spec.PPO_INITIAL_LR))()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=spec.PPO_INITIAL_LR)
        self.learning_state = dict(optimizer_initialized=False, completed_ppo_updates=0,
                                   optimizer_steps=0)
        self.obs = None
        self.storage = None
        self.control_step_callback = None
        self.update_callback = None
        self.user_diagnostics_callback = None
        self.last_route_diagnostics = None
        self._route = None

    @property
    def learning(self):
        update = self.learning_state['completed_ppo_updates']
        samples = 48 * self.env.num_envs * update
        return dict(update=update, optimizer_step=self.learning_state['optimizer_steps'],
                    control_samples=samples, physics_substeps=4 * samples)

    def collect(self):
        if self.obs is None:
            self.obs, _ = self.env.reset()
        rows = []
        records = []
        for step in range(48):
            rows.append(self.obs['route_error'].clone())
            self.obs, _, _, _, info = self.env.step(torch.zeros(self.env.num_envs, 12))
            records.extend(info.get('episodes', []))
            if self.control_step_callback is not None:
                self.control_step_callback(dict(
                    step_index=step + 1, num_envs=self.env.num_envs,
                    physics_samples_increment=self.env.num_envs, info=info))
            if self.user_diagnostics_callback is not None:
                self.user_diagnostics_callback(info)
        self._route = torch.stack(rows).flatten(0, 1)
        self.storage = {'obs': {'route_error': self._route}}
        self.last_route_diagnostics = route_block(self._route)
        return records

    def update(self):
        base = self.learning_state['completed_ppo_updates']
        if self.update_callback is not None:
            self.update_callback('optimizer_started',
                                 dict(learning_state=self.learning,
                                      expected_optimizer_steps=20))
        for _ in range(20):
            self.optimizer.zero_grad(set_to_none=True)
            sum(parameter.square().sum() for parameter in self.model.parameters()).backward()
            self.optimizer.step()
            self.model.constrain_std()
            self.learning_state['optimizer_steps'] += 1
            if self.update_callback is not None:
                self.update_callback('optimizer_step_completed',
                                     dict(learning_state=self.learning))
        self.learning_state['optimizer_initialized'] = True
        self.learning_state['completed_ppo_updates'] = base + 1
        if self.update_callback is not None:
            self.update_callback('update_completed',
                                 dict(learning_state=self.learning,
                                      optimizer_steps_in_update=20))
        return {}


class ProbeCases:
    def __init__(self):
        self.cases = [dict(speed=speed, index=index) for speed in spec.SPEEDS
                      for index in range(2)]
        self.sha256 = 'c' * 64


class ProbeEvalResult:
    def __init__(self):
        self.summary = dict(gate_pass=True, statistics_unchanged=True,
                            evidence_verified=True, episodes=6,
                            per_speed=[dict(vx=speed, episodes=2,
                                            case_pass_count=2) for speed in spec.SPEEDS],
                            raw_paths={})


class ProbeS5Env:
    device = 'cpu'

    def __init__(self, num_envs=1, **_):
        self.num_envs = int(num_envs)
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {}, {}

    def step(self, action):
        self.steps += 1
        return {}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), \
            torch.zeros(1, dtype=torch.bool), {}

    def close(self):
        pass


_PROBE_LEDGERS = {}


class MemoryAttemptLedger(AttemptLedger):
    """In-memory replacement for the fsync-per-control-step probe ledger.

    The production probe uses the real durable ``AttemptLedger`` (SDS 7.3).  A
    unit test cannot afford ~3800 fsyncs, so the probe test injects this mirror
    through ``patch.object(orchestration, 'AttemptLedger', ...)``; it keeps the
    same mode/learning/curriculum contract and a per-directory singleton so the
    resume re-opens the same boundary.
    """
    def __new__(cls, directory, **kwargs):
        key = str(Path(directory))
        if key in _PROBE_LEDGERS:
            return _PROBE_LEDGERS[key]
        instance = super().__new__(cls)
        _PROBE_LEDGERS[key] = instance
        return instance

    def __init__(self, directory, *, run_id, mode='formal', **kwargs):
        if getattr(self, '_initialized', False):
            return
        key = str(Path(directory))
        self.header = dict(schema=1, run_id=run_id, mode=mode)
        self._mode = mode
        self._completed = 0
        self._num_envs = 0
        self._active = None
        self._checks = []
        self._tasks = {}
        self._results = []
        self._curriculum = recovery.genesis_curriculum()
        self._data = dict(completed=0, control_samples=0, sampled=0)
        self._initialized = True

    @property
    def mode(self):
        return self._mode

    @property
    def curriculum(self):
        return self._curriculum

    @property
    def completed_ppo_updates(self):
        return self._completed

    @property
    def control_samples(self):
        return 48 * self._num_envs * self._completed

    @property
    def physics_substeps(self):
        return 4 * self.control_samples

    @property
    def tasks(self):
        return self._tasks

    @property
    def results(self):
        return self._results

    @property
    def checkpoints(self):
        return self._checks

    @property
    def gate_streak(self):
        return self._curriculum['streak']['gate']

    @property
    def collapse_streak(self):
        return self._curriculum['streak']['collapse']

    @property
    def pending_transition(self):
        return self._curriculum['pending_transition']

    @property
    def qualified_update(self):
        return None

    def anchor(self):
        return dict(seq=len(self._checks), digest='a' * 64)

    def expected_learning(self):
        return recovery.learning_state(self._completed, num_envs=self._num_envs)

    def begin_attempt(self, curriculum, num_envs):
        self._num_envs = num_envs
        self._active = dict(controls=0, num_envs=num_envs, optimizer_steps=0,
                            phase='collect')
        self._data['completed'] = self._completed

    def control_step(self, event):
        self._active['controls'] += 1

    def update_event(self, kind, payload):
        if kind == 'optimizer_started':
            self._active['phase'] = 'optimizer_started'
        elif kind == 'optimizer_step_completed':
            self._active['optimizer_steps'] += 1

    def complete_attempt(self, curriculum, learning):
        self._curriculum = dict(curriculum)
        self._completed += 1
        self._active = None

    def record_checkpoint(self, path, checkpoint_sha, anchor, contract_sha256_value):
        self._checks.append(dict(path=str(path), sha256=checkpoint_sha,
                                 anchor=deepcopy(anchor),
                                 contract=contract_sha256_value))

    def checkpoint_task_state(self):
        return dict(tasks=deepcopy(self._tasks), results=deepcopy(self._results),
                    collapse_streak=self.collapse_streak, gate_streak=self.gate_streak,
                    anchor=self.anchor(), qualified_update=None,
                    pending_transition=None)

    def checkpoint_evaluation(self):
        return dict(consumed=[], latest_diagnostic_update=None)

    def pending_evaluations(self):
        return {}


class DefaultFactoryTests(unittest.TestCase):
    """The real (non-injected) probe factories must import the r2 class names.

    The W7 probe failed on GPU with ``ImportError: cannot import name 'V05Env'``
    because these defaults still pointed at the r1 names while every other test
    injected fake factories.  This CPU regression test exercises the real
    default path: import + construct only (no ``step``, no GPU, no network).
    """

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_factories_import_and_construct_r2_objects_on_cpu(self):
        from src.tasks.velocity.v05_r2.env import V05R2Env
        from src.tasks.velocity.v05_r2.model import V05R2ActorCritic
        from src.tasks.velocity.v05_r2.ppo import PPO

        env = module._default_probe_env_factory(num_envs=1, device='cpu', seed=42,
                                                stage='F0')
        model = None
        try:
            self.assertIsInstance(env, V05R2Env)
            self.assertEqual(type(env).__name__, 'V05R2Env')
            self.assertEqual(env.num_envs, 1)
            self.assertEqual(env.stage, 'F0')
            self.assertEqual(str(env.device), 'cpu')

            model = module._default_probe_model_factory(
                spec.INIT_PATH, spec.INIT_SHA256, 42, 'cpu')
            self.assertIsInstance(model, V05R2ActorCritic)
            self.assertEqual(type(model).__name__, 'V05R2ActorCritic')
            for name, shape in (('actor.route_fusion.weight', (512, 3)),
                                ('actor.route_fusion.bias', (512,)),
                                ('critic.0.weight', (512, 211))):
                self.assertEqual(tuple(model.state_dict()[name].shape), shape)
            self.assertTrue(all(parameter.device.type == 'cpu'
                                for parameter in model.parameters()))

            ppo = module._default_probe_ppo_factory(model, env)
            self.assertIsInstance(ppo, PPO)
            self.assertIs(ppo.model, model)
            self.assertIs(ppo.env, env)
            self.assertEqual(ppo.cfg.rollout, 48)
            self.assertEqual(ppo.learning['update'], 0)
            self.assertFalse(ppo.optimizer.state)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            if model is not None:
                del model
            env.close()

    def test_no_r1_class_or_module_names_remain_in_the_probe_path(self):
        source = Path(module.__file__).read_text()
        self.assertNotIn('from .env import V05Env', source)
        self.assertNotIn('from .model import V05ActorCritic', source)
        self.assertNotIn("import V05Env", source)
        self.assertNotIn('v05.env', source)
        self.assertNotIn('v05.model', source)
        self.assertNotIn('v05.ppo', source)
        # The r2 names are the explicit imports.
        self.assertIn('from .env import V05R2Env', source)
        self.assertIn('from .model import V05R2ActorCritic', source)


class ProbeRunTests(unittest.TestCase):
    def test_probe_run_is_fifty_six_updates_and_never_formal(self):
        clock = ProbeClock()

        def sample_fn():
            return dict(gpu_used_mib=1000., process_gpu_mib=1000.,
                        host_available_mib=9000., disk_free_gib=50., rss_mib=100.)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'probe_run'
            _PROBE_LEDGERS.clear()
            with patch.object(module, 'AttemptLedger', MemoryAttemptLedger):
                record = module.run_resource_probe(
                    output_dir=out, seed=42, device='cuda:0', num_envs=512,
                    eval_num_envs=2, init_checkpoint=spec.INIT_PATH,
                    init_checkpoint_sha256=spec.INIT_SHA256,
                    env_factory=ProbeEnv,
                    model_factory=lambda *args: ProbeModel(),
                    ppo_factory=ProbePPO,
                    evaluation=lambda model, cases, **kwargs: ProbeEvalResult(),
                    make_cases=evaluation_module.make_cases,
                    sample_fn=sample_fn, s5_env_factory=ProbeS5Env,
                    s5_steps_fn=lambda n, h, speed: 3, clock=clock,
                    wall_clock=clock, command=['probe'])
            self.assertEqual(record['status'], 'passed', record['failures'])
            identity = record['identity']
            self.assertEqual(identity['task'], spec.TASK)
            self.assertEqual(identity['mode'], 'probe')
            self.assertEqual(identity['ledger_mode'], 'probe')
            self.assertEqual(identity['evaluation_mode'], 'probe')
            self.assertEqual(identity['learning_updates'], 56)
            self.assertEqual(identity['control_samples'], 1376256)
            self.assertEqual(identity['total_budget'], 56)
            self.assertEqual(identity['update_cap'], 56)
            self.assertEqual(identity['effective_cap_source'], 'mode_budget')
            self.assertEqual(identity['formal_budget_not_started'], 30000)
            # SDS 6.3: the probe uses the same single-offset make_cases contract.
            self.assertEqual(identity['evaluation_eval_seed'], 200342)
            eval_phase = next(phase for phase in record['phases']
                              if phase['phase'] == 'coexistence_evaluation')
            self.assertEqual(eval_phase['eval_seed'], 200342)
            self.assertNotEqual(eval_phase['eval_seed'], 400642)
            costs = record['costs']
            self.assertEqual(costs['learning']['updates'], 56)
            self.assertEqual(costs['learning']['control_samples'], 1376256)
            self.assertEqual(costs['learning']['physics_substeps'], 4 * 1376256)
            self.assertEqual(costs['recovery']['updates'], 1)
            self.assertFalse(costs['counts_against_formal'])
            self.assertFalse(costs['counts_against_pilot'])
            self.assertTrue(costs['evaluation_cost_counted_separately'])
            self.assertTrue(costs['s5_probe_counted_separately'])
            self.assertEqual(costs['s5_no_learning']['control_samples'], 9)
            ledger = _PROBE_LEDGERS[str(out / 'probe' / 'ledger')]
            self.assertEqual(ledger.mode, 'probe')
            self.assertEqual(ledger.completed_ppo_updates, 56)
            self.assertTrue((out / 'probe' / 'probe.json').is_file())
            self.assertTrue((out / 'probe' / 'summary.md').is_file())


class ProbeCaseSeedTests(unittest.TestCase):
    def test_probe_and_gates_share_one_single_offset_rule(self):
        from src.tasks.velocity.v05_r2.evaluation import make_cases
        probe = make_cases('P1', 'validation', 42)
        gate = make_cases('P1', 'validation', 42)
        self.assertEqual(probe.seed, 200342)
        self.assertEqual(gate.seed, 200342)
        self.assertEqual(probe.sha256, gate.sha256)
        self.assertNotEqual(probe.seed, 400642)  # the old double offset
        self.assertNotEqual(probe.seed, 1000642)
        # Seed 43 follows the same rule and produces a different table.
        self.assertEqual(make_cases('P1', 'validation', 43).seed, 200343)
        self.assertNotEqual(make_cases('P1', 'validation', 43).sha256, probe.sha256)


class FinalTestTests(unittest.TestCase):
    def test_final_test_contract_is_three_speeds_one_hundred_cases(self):
        request = final_test_request(checkpoint='/tmp/frozen.pt',
                                     checkpoint_sha256=DIGEST,
                                     run_id='formal_seed42',
                                     output_dir='/tmp/final')
        self.assertEqual(request['checkpoint_schema'], 7)
        self.assertEqual(request['case_pass_required'], 95)
        self.assertEqual(request['speeds'], [.2, .4, .6])
        summary = dict(kind='test', stage='S5', eval_seed=400342,
                       checkpoint_sha256=DIGEST,
                       per_speed=[dict(c=speed, total=100, case_pass_count=95,
                                       speed_pass_count=90) for speed in spec.SPEEDS])
        result = validate_final_test_summary(summary, request=request)
        self.assertTrue(result['passed'])
        self.assertFalse(result['speed_gate'])
        summary['per_speed'][0]['case_pass_count'] = 94
        self.assertFalse(validate_final_test_summary(summary, request=request)['passed'])
        summary['per_speed'][0]['case_pass_count'] = 100
        summary['per_speed'][0]['total'] = 99
        with self.assertRaises(ValueError):
            validate_final_test_summary(summary, request=request)


if __name__ == '__main__':
    unittest.main()
