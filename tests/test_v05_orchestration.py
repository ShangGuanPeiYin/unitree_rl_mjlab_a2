"""CPU tests for the v0.5 pilot/formal state machines, W7 probe and W10 gate.

No simulator, no CUDA, no network and no subprocess is started: the supervisor's
worker entry point is injected, and every checkpoint/verdict used here is either
synthetic or built from the real schema-6 envelope on CPU.
"""
import sys
sys.path.insert(0, 'scripts')
import a2_v05_bootstrap
a2_v05_bootstrap.activate('.')

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.tasks.velocity.v05 import orchestration as o
from src.tasks.velocity.v05 import recovery, spec
from src.tasks.velocity.v05.runner import Curriculum


def digest(text):
    import hashlib
    return hashlib.sha256(str(text).encode()).hexdigest()


def binding_with(**changes):
    """A contract binding with selected fields changed (stale-report tests)."""
    value = deepcopy(spec.contract())
    value.update(changes)
    return o.pilot_binding(value)


def pilot_verdict(**overrides):
    verdict = dict(status='passed', stage='P1', stage_iteration=500,
                   f0_consumed=200, p1_consumed=500, total_updates=700,
                   gates={100: True, 200: True, 300: True, 400: True, 500: True},
                   reasons=[])
    verdict.update(overrides)
    return verdict


def report_for(seed, *, status='passed', verdict=None, checkpoint='a',
               evidence=True, binding_value=None):
    checkpoint_sha = digest(f'checkpoint-{seed}-{checkpoint}')
    index = [dict(path=f'/tmp/evidence-{seed}', sha256=digest(f'ev-{seed}'),
                  kind='trial_checkpoint', update=700)] if evidence else []
    return o.make_pilot_report(seed=seed, status=status,
                               verdict=verdict or pilot_verdict(),
                               checkpoint_sha256=checkpoint_sha,
                               evidence_index=index, binding_value=binding_value)


def pair(*, statuses=('passed', 'passed'), binding_value=None, verdicts=None):
    reports = {}
    for index, seed in enumerate(o.PILOT_SEEDS):
        verdict = (verdicts or {}).get(seed, pilot_verdict())
        reports[seed] = report_for(seed, status=statuses[index], verdict=verdict,
                                   binding_value=binding_value)
    return reports


# ------------------------------------------------------------ report binding
class PilotReportTests(unittest.TestCase):
    def test_report_binds_the_whole_contract_and_evidence_index(self):
        report = report_for(42)
        self.assertEqual(report['schema'], o.REPORT_SCHEMA)
        self.assertEqual(report['seed'], 42)
        self.assertEqual(report['eval_seeds'],
                         dict(validation=200342, perturbed=300342,
                              diagnostic=500342, test=400342))
        self.assertEqual(report['p1']['budget'], 500)
        self.assertEqual(report['p1']['final_gates'], [400, 500])
        self.assertFalse(report['p1']['early_promotion'])
        self.assertEqual(report['total_updates'], 700)
        o.validate_pilot_report(report)
        with self.assertRaises(o.PilotError):
            report_for(41)
        with self.assertRaises(o.PilotError):
            report_for(42, evidence=False)
        with self.assertRaises(o.PilotError):
            o.make_pilot_report(seed=43, status='passed',
                                verdict=pilot_verdict(p1_consumed=400),
                                checkpoint_sha256=digest('x'),
                                evidence_index=[dict(path='p')])

    def test_both_seeds_must_share_the_same_contract_content(self):
        reports = pair()
        checked = o.require_pilot_pair(reports)
        self.assertEqual(set(checked), {42, 43})
        # A different reward/PPO/stage/source identity invalidates the old pilot.
        for field, value in (('reward_config_sha256', digest('old-reward')),
                             ('source_sha', digest('old-source')),
                             ('stages', [dict(value) for value in spec.contract()['stages']][:-1]),
                             ('ppo', dict(spec.contract()['ppo'], gamma=.99))):
            with self.subTest(field):
                stale = {seed: report_for(seed, binding_value=binding_with(**{field: value}))
                         for seed in o.PILOT_SEEDS}
                with self.assertRaises(o.PilotError):
                    o.require_pilot_pair(stale)

    def test_one_failed_pilot_blocks_formal_but_reports_are_kept(self):
        reports = pair(statuses=('passed', 'failed'),
                       verdicts={43: pilot_verdict(status='failed',
                                                   reasons=['p1_gate_500_not_passed'],
                                                   gates={100: True, 200: True, 300: True,
                                                          400: True, 500: False})})
        self.assertEqual(reports[43]['status'], 'failed')
        with self.assertRaises(o.PilotError):
            o.require_pilot_pair(reports)
        with self.assertRaises(o.FormalAdmissionError):
            o.admit_formal(reports=reports,
                           init_checkpoint_sha256=spec.INIT_SHA256)


# --------------------------------------------------------------- admission
class FormalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.reports = pair()

    def test_two_passing_pilots_admit_a_new_eth_formal_run(self):
        admission = o.admit_formal(reports=self.reports,
                                   init_checkpoint_sha256=spec.INIT_SHA256)
        self.assertEqual(admission['status'], 'admitted')
        self.assertEqual(admission['seed'], 42)
        self.assertEqual(admission['total_updates'], 30000)
        self.assertEqual(admission['init_sha256'], spec.INIT_SHA256)
        self.assertEqual(admission['pilot_checkpoints'],
                         {seed: self.reports[seed]['checkpoint_sha256']
                          for seed in o.PILOT_SEEDS})

    def test_formal_never_resumes_and_never_uses_pilot_weights(self):
        with self.assertRaises(o.FormalAdmissionError):
            o.admit_formal(reports=self.reports, init_checkpoint_sha256=spec.INIT_SHA256,
                           resume_run=True)
        with self.assertRaises(o.FormalAdmissionError):
            o.admit_formal(reports=self.reports,
                           init_checkpoint_sha256=self.reports[42]['checkpoint_sha256'])
        with self.assertRaises(o.FormalAdmissionError):
            o.admit_formal(reports=self.reports,
                           init_checkpoint_sha256=spec.INIT_SHA256,
                           pilot_checkpoint_shas=[spec.INIT_SHA256])

    def test_stale_reports_are_refused(self):
        stale = pair(binding_value=binding_with(source_sha=digest('old-code')))
        with self.assertRaises(o.PilotError):
            o.require_pilot_pair(stale)


# ------------------------------------------------------------ pilot sequence
class PilotSequenceTests(unittest.TestCase):
    def _trial(self, *, statuses=('passed', 'passed'), calls=None):

        def trial(seed, fresh=True):
            if calls is not None:
                calls.append((seed, fresh))
            verdict = pilot_verdict(status=statuses[o.PILOT_SEEDS.index(seed)])
            return dict(status=verdict['status'], verdict=verdict,
                        checkpoint_sha256=digest(f'trial-{seed}'),
                        evidence_index=[dict(path=f'/tmp/pilot-{seed}',
                                             sha256=digest(f'raw-{seed}'),
                                             kind='trial_checkpoint', update=700)],
                        failure_reason=(None if verdict['status'] == 'passed'
                                        else 'p1_gate_500_not_passed'))
        return trial

    def test_seed_42_then_43_serially_each_fresh(self):
        calls = []
        sequence = o.PilotSequence(trial=self._trial(calls=calls))
        result = sequence.run()
        self.assertEqual(calls, [(42, True), (43, True)])
        self.assertTrue(result['formal_ready'])
        self.assertEqual(set(result['reports']), {42, 43})
        self.assertTrue(all(report['status'] == 'passed'
                            for report in result['reports'].values()))
        self.assertEqual(result['status'], 'pilot_complete')

    def test_seed_42_failure_still_runs_the_authorized_seed_43(self):
        calls = []
        sequence = o.PilotSequence(trial=self._trial(statuses=('failed', 'passed'),
                                                     calls=calls))
        result = sequence.run()
        self.assertEqual(calls, [(42, True), (43, True)])
        self.assertFalse(result['formal_ready'])
        self.assertEqual(result['reports'][42]['status'], 'failed')
        self.assertEqual(result['reports'][43]['status'], 'passed')
        with self.assertRaises(o.FormalAdmissionError):
            o.admit_formal(reports=result['reports'],
                           init_checkpoint_sha256=spec.INIT_SHA256)

    def test_incomplete_trial_is_a_failed_report(self):
        def trial(seed, fresh=True):
            verdict = pilot_verdict(status='running', stage='P1',
                                    stage_iteration=400, p1_consumed=400,
                                    total_updates=600)
            return dict(status='failed', verdict=verdict,
                        checkpoint_sha256=digest(f'running-{seed}'),
                        evidence_index=[dict(path=f'/tmp/{seed}',
                                             sha256=digest(f'raw-{seed}'),
                                             kind='trial_checkpoint', update=600)],
                        failure_reason='trial_incomplete')
        result = o.PilotSequence(trial=trial).run()
        self.assertFalse(result['formal_ready'])
        for report in result['reports'].values():
            self.assertEqual(report['status'], 'failed')


# ------------------------------------------------------- pilot verdict from disk
def curriculum_state(*, f0, p1, stage='P1', collapse=0):
    spent = {name: 0 for name in spec.STAGE_ORDER}
    spent['F0'] = f0
    spent['P1'] = p1
    return dict(stage=stage, stage_iteration=p1 if stage == 'P1' else f0, spent=spent,
                streak=dict(gate=2 if p1 else 0, collapse=collapse),
                pending_transition=None)


def ledger_state(*, f0, passes, collapse=0):
    results = []
    for iteration in (100, 200, 300, 400, 500):
        passed = passes.get(iteration, False)
        results.append(dict(task_key=digest(f'gate-{iteration}'), kind='validation',
                            stage='P1', global_update=f0 + iteration,
                            outcome='passed' if passed else 'failed',
                            passed=passed, gate_streak=1, collapse_streak=collapse))
    return dict(tasks={}, results=results, anchor=dict(seq=0, digest=digest('anchor')),
                qualified_update=None, pending_transition=None,
                gate_streak=1, collapse_streak=collapse)


class PilotVerdictTests(unittest.TestCase):
    def test_verdict_requires_both_400_and_500_after_the_fixed_500(self):
        verdict = o.pilot_verdict_from_state(
            curriculum_state=curriculum_state(f0=200, p1=500),
            ledger_state=ledger_state(f0=200, passes={400: True, 500: True}))
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(verdict['gates'], {100: False, 200: False, 300: False,
                                            400: True, 500: True})
        failed = o.pilot_verdict_from_state(
            curriculum_state=curriculum_state(f0=200, p1=500),
            ledger_state=ledger_state(f0=200, passes={400: True, 500: False}))
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('p1_gate_500_not_passed', failed['reasons'])

    def test_early_two_passes_must_not_end_the_trial(self):
        verdict = o.pilot_verdict_from_state(
            curriculum_state=curriculum_state(f0=200, p1=200),
            ledger_state=ledger_state(f0=200, passes={100: True, 200: True}))
        self.assertEqual(verdict['status'], 'running')
        self.assertEqual(verdict['total_updates'], 400)

    def test_collapse_or_f0_exhaustion_fails_the_trial(self):
        collapse = o.pilot_verdict_from_state(
            curriculum_state=curriculum_state(f0=200, p1=500, collapse=2),
            ledger_state=ledger_state(f0=200, passes={400: True, 500: True}, collapse=2))
        self.assertEqual(collapse['status'], 'failed')
        self.assertIn('behavior_collapse', collapse['reasons'])
        f0 = o.pilot_verdict_from_state(
            curriculum_state=curriculum_state(f0=2000, p1=0, stage='F0'),
            ledger_state=ledger_state(f0=2000, passes={}))
        self.assertEqual(f0['status'], 'failed')
        self.assertIn('f0_budget_exhausted_without_gate', f0['reasons'])


# ---------------------------------------------------------------- W7 probe
class ProbeTests(unittest.TestCase):
    def test_probe_plan_is_five_plus_fifty_plus_one_and_separate(self):
        plan = o.probe_plan()
        self.assertEqual((plan['warmup_updates'], plan['measured_updates'],
                          plan['recovery_updates']), (5, 50, 1))
        self.assertEqual(plan['control_samples'], 56 * 48 * 512)
        self.assertEqual(plan['physics_substeps'], 4 * 56 * 48 * 512)
        self.assertEqual(plan['adam_steps'], 56 * 20)
        self.assertFalse(plan['counts_against_formal'])
        self.assertTrue(plan['formal_source_unaffected'])
        self.assertTrue(plan['per_control_step_sampling'])
        self.assertTrue(plan['per_update_peak_report'])
        self.assertEqual(plan['evaluation_lifecycle']['num_envs'], 8)
        self.assertEqual(plan['s5_three_speed_timeout_eval']['stage'], 'S5')
        self.assertEqual(plan['s5_three_speed_timeout_eval']['speeds'], [0.2, 0.4, 0.6])
        self.assertFalse(plan['s5_three_speed_timeout_eval']['learning'])

    def test_probe_statistics_use_only_the_50_measured_updates(self):
        timings = [10.0] * 5 + [2.0 + index * .01 for index in range(50)] + [99.0]
        result = o.probe_statistics(timings, completed=56)
        self.assertEqual(result['measured_completed'], 50)
        self.assertEqual(result['measured_iterations'], [6, 55])
        self.assertAlmostEqual(result['median_update_s'], 2.245, places=6)
        self.assertLess(result['p95_update_s'], 99.0)
        self.assertAlmostEqual(result['throughput_control_samples_per_s'],
                               48 * 512 * 50 / sum(timings[5:55]), places=6)
        self.assertFalse(result['counts_against_formal'])
        self.assertEqual(o.probe_statistics(timings[:5], completed=5)['measured_completed'], 0)
        with self.assertRaises(ValueError):
            o.probe_statistics(timings[:5], completed=6)
        with self.assertRaises(ValueError):
            o.probe_statistics([1.0, float('nan')] + timings[2:], completed=56)

    def test_resource_guards_match_the_sds_limits(self):
        ok = dict(gpu_used_mib=7168, process_gpu_mib=5120, host_available_mib=4096,
                  disk_free_gib=10)
        self.assertTrue(o.resource_guard(ok)['passed'])
        expected = {'gpu_used_mib': 'whole_device_mib>7168',
                    'process_gpu_mib': 'process_gpu_mib>5120',
                    'host_available_mib': 'host_available_mib<4096',
                    'disk_free_gib': 'disk_free_gib<10'}
        for key, value in (('gpu_used_mib', 7169), ('process_gpu_mib', 5121),
                           ('host_available_mib', 4095), ('disk_free_gib', 9.99)):
            with self.subTest(key):
                result = o.resource_guard(dict(ok, **{key: value}))
                self.assertFalse(result['passed'])
                self.assertIn(expected[key], result['violations'])
                self.assertEqual(result['limits']['whole_device_mib'], 7168)
                self.assertEqual(result['limits']['process_mib'], 5120)
                self.assertEqual(result['limits']['host_available_mib'], 4096)
                self.assertEqual(result['limits']['disk_free_gib'], 10)

    def test_resource_sampler_reports_per_update_peaks(self):
        values = iter([dict(rss_mib=100), dict(rss_mib=300), dict(rss_mib=200)])
        sampler = o.ResourceSampler(sample_fn=lambda: next(values), clock=lambda: 1.0)
        sampler.sample(update=1, control_step=1)
        sampler.sample(update=1, control_step=2)
        sampler.sample(update=1, control_step=3)
        peak = sampler.end_update(1)
        self.assertEqual(peak['peaks']['rss_mib'], 300)
        self.assertEqual(peak['samples'], 3)
        summary = sampler.summary()
        self.assertTrue(summary['per_control_step'])
        self.assertTrue(summary['counts_against_formal'] is False)


# ---------------------------------------------------------------- W10 final test
class FinalTestTests(unittest.TestCase):
    def request(self, checkpoint_sha=None):
        return o.final_test_request(
            checkpoint='/tmp/frozen.pt',
            checkpoint_sha256=checkpoint_sha or digest('frozen'),
            run_id='formal-seed42', output_dir='/tmp/final_test')

    def summary(self, *, checkpoint_sha, counts=(100, 100, 100), speed_counts=(0, 0, 0)):
        return dict(kind='test', stage='S5', eval_seed=spec.FINAL_TEST_SEED,
                    checkpoint_sha256=checkpoint_sha,
                    per_speed=[dict(c=speed, total=100, case_pass_count=counts[index],
                                    speed_pass_count=speed_counts[index])
                               for index, speed in enumerate(o.SPEEDS)])

    def test_request_is_the_frozen_candidate_only(self):
        request = self.request()
        self.assertEqual(request['seed'], 400342)
        self.assertEqual(request['speeds'], [0.2, 0.4, 0.6])
        self.assertEqual(request['episodes_per_speed'], 100)
        self.assertEqual(request['case_pass_required'], 95)
        self.assertFalse(request['speed_pass_is_gate'])
        self.assertEqual(request['spawn'], 'full_s5_perturbation')
        self.assertEqual(request['candidate'], 'second_s5_pass_frozen')
        self.assertFalse(request['retrain_allowed'])
        self.assertEqual(request['selection'],
                         'frozen_candidate_only_no_history_selection')

    def test_95_of_100_is_required_per_speed_and_speed_not_a_gate(self):
        request = self.request()
        passed = o.validate_final_test_summary(
            self.summary(checkpoint_sha=request['checkpoint_sha256']), request=request)
        self.assertTrue(passed['passed'])
        self.assertEqual(passed['status'], 'final_test_passed')
        self.assertFalse(passed['speed_gate'])
        failed = o.validate_final_test_summary(
            self.summary(checkpoint_sha=request['checkpoint_sha256'],
                         counts=(100, 95, 94)), request=request)
        self.assertFalse(failed['passed'])
        self.assertEqual(failed['status'], 'final_test_failed')
        # A low speed_pass_count is reported but never vetoes the final gate.
        fast = o.validate_final_test_summary(
            self.summary(checkpoint_sha=request['checkpoint_sha256'],
                         speed_counts=(0, 0, 0)), request=request)
        self.assertTrue(fast['passed'])
        with self.assertRaises(ValueError):
            o.validate_final_test_summary(
                self.summary(checkpoint_sha=digest('other')), request=request)
        one_short = o.validate_final_test_summary(
            self.summary(checkpoint_sha=request['checkpoint_sha256'],
                         counts=(99, 100, 100)), request=request)
        self.assertTrue(one_short['passed'])  # 99/100 still clears 95/100
        wrong_total = self.summary(checkpoint_sha=request['checkpoint_sha256'])
        wrong_total['per_speed'][0]['total'] = 99
        with self.assertRaises(ValueError):
            o.validate_final_test_summary(wrong_total, request=request)


# ---------------------------------------------------------------- supervisor
def base_args(tmp, **overrides):
    values = dict(
        mode='pilot', execute=False, task=spec.TASK,
        recipe_version=spec.RECIPE_VERSION, sds_revision=spec.SDS_REVISION,
        seed=42, device='cuda:0', num_envs=512, eval_num_envs=8,
        init_checkpoint=spec.INIT_PATH, init_checkpoint_sha256=spec.INIT_SHA256,
        vx_min=.2, vx_max=.6, vy=0., wz=0., save_interval=100,
        diagnostic_interval=50, stage_budgets=spec.PILOT_STAGE_BUDGETS,
        pilot_p1_updates=500, pilot_report_42=None, pilot_report_43=None,
        output_dir=str(tmp / 'run'), resume_run=False,
        wandb_project=None, wandb_entity=None, wandb_run_name=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.preflight = self.root / 'preflight'

    def supervisor(self, **overrides):
        return o.Supervisor(base_args(self.root, **overrides),
                            preflight_dir=self.preflight)

    def test_pilot_preflight_writes_evidence_but_never_the_run_directory(self):
        supervisor = self.supervisor()
        record = supervisor.preflight()
        self.assertEqual(record['status'], 'preflight_validated')
        self.assertFalse(record['training_started'])
        self.assertFalse(record['gpu_started'])
        self.assertFalse(record['network_started'])
        self.assertEqual(record['mode'], 'pilot')
        self.assertEqual(record['stage_budgets'], spec.PILOT_STAGE_BUDGETS)
        self.assertEqual(record['pilot_p1_updates'], 500)
        self.assertTrue((self.preflight /
                         'preflight_pilot_seed42.json').is_file())
        self.assertFalse((self.root / 'run').exists())
        self.assertFalse(supervisor.out.exists())
        # The non-executing run returns the preflight record unchanged.
        self.assertEqual(supervisor.run(), record)
        self.assertFalse(supervisor.out.exists())

    def test_pilot_arguments_fix_seed_budgets_and_p1(self):
        for overrides in (dict(seed=41), dict(stage_budgets=spec.FORMAL_STAGE_BUDGETS),
                          dict(pilot_p1_updates=400),
                          dict(init_checkpoint_sha256=digest('other')),
                          dict(num_envs=256), dict(save_interval=50),
                          dict(resume_run=True)):
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValueError, o.FormalAdmissionError)):
                    self.supervisor(**overrides).preflight()

    def test_formal_requires_both_reports_and_no_pilot_resume(self):
        report42 = self.root / 'report42.json'
        report43 = self.root / 'report43.json'
        reports = pair()
        report42.write_text(json.dumps(reports[42]))
        report43.write_text(json.dumps(reports[43]))
        args = dict(mode='formal', seed=42, stage_budgets=spec.FORMAL_STAGE_BUDGETS,
                    pilot_p1_updates=None, pilot_report_42=str(report42),
                    pilot_report_43=str(report43))
        record = self.supervisor(**args).preflight()
        self.assertEqual(record['mode'], 'formal')
        self.assertEqual(record['stage_budgets'], spec.FORMAL_STAGE_BUDGETS)
        with self.assertRaises(ValueError):
            self.supervisor(**dict(args, pilot_report_43=None)).preflight()
        with self.assertRaises((ValueError, o.FormalAdmissionError)):
            # Resume is supported, but only against an existing run contract.
            self.supervisor(**dict(args, resume_run=True,
                                   init_checkpoint=None,
                                   init_checkpoint_sha256=None)).preflight()
        with self.assertRaises(ValueError):
            self.supervisor(**dict(args, seed=43)).preflight()

    def test_pause_requests_a_complete_boundary_pause(self):
        supervisor = self.supervisor()
        result = supervisor.pause('monitoring_blocked')
        self.assertEqual(result['status'], 'monitoring_blocked')
        payload = json.loads((self.root / 'run' / 'control.json').read_text())
        self.assertEqual(payload['action'], 'pause')
        self.assertEqual(payload['reason'], 'monitoring_blocked')
        with self.assertRaises(ValueError):
            supervisor.pause('early_stop')

    def test_worker_directories_are_monotonic_and_never_overwritten(self):
        root = self.root / 'workers'
        first = recovery.claim_worker_directory(root)
        second = recovery.claim_worker_directory(root)
        self.assertEqual(first.name, 'attempt-0001')
        self.assertEqual(second.name, 'attempt-0002')
        marker = first / 'keep.txt'
        marker.write_text('original worker evidence')
        third = recovery.claim_worker_directory(root)
        self.assertEqual(third.name, 'attempt-0003')
        self.assertEqual(marker.read_text(), 'original worker evidence')

    def test_worker_command_carries_the_frozen_contract(self):
        command = self.supervisor().worker_command(self.root / 'worker', seed=43,
                                                   mode='pilot')
        joined = ' '.join(command)
        self.assertIn('train_a2_ideal_stairs_v05.py', joined)
        for fragment in ('--mode pilot', '--seed 43', '--num-envs 512',
                         '--eval-num-envs 8', '--save-interval 100',
                         '--diagnostic-interval 50',
                         '--stage-budgets ' + spec.PILOT_STAGE_BUDGETS,
                         '--pilot-p1-updates 500', '--control-file'):
            self.assertIn(fragment, joined)
        self.assertNotIn('--resume', joined)
        self.assertNotIn('--gpu-queue-fd-env', joined)


def frozen_identity():
    """Freeze the source-closure identity used by build_checkpoint in tests."""
    return (patch.object(recovery, 'contract_sha_for', return_value='a' * 64),
            patch.object(recovery, 'recipe_config', return_value={'recipe': '0.5-r1'}))


def synthetic_pilot_checkpoint(root, *, seed, f0=200, p1=500,
                               passes=(400, 500), collapse=0):
    """A valid schema-6 pilot checkpoint with a synthetic durable ledger block."""
    spent = {name: 0 for name in spec.STAGE_ORDER}
    spent['F0'] = f0
    spent['P1'] = p1
    curriculum = dict(stage='P1', stage_iteration=p1, spent=spent,
                      streak=dict(gate=2, collapse=collapse),
                      pending_transition=None)
    results = [dict(task_key=digest(f'gate-{iteration}'), kind='validation', stage='P1',
                    global_update=f0 + iteration,
                    outcome='passed' if iteration in passes else 'failed',
                    passed=iteration in passes, gate_streak=1,
                    collapse_streak=collapse)
               for iteration in (100, 200, 300, 400, 500)]
    ledger = dict(tasks={}, results=results, collapse_streak=collapse, gate_streak=2,
                  anchor=dict(seq=0, digest=digest('anchor')),
                  qualified_update=None, pending_transition=None)
    env = SimpleNamespace(rng=torch.Generator().manual_seed(seed))
    state = recovery.build_checkpoint(
        model_state={'weight': torch.zeros(1)}, optimizer_state={}, normalizer={},
        learning=recovery.learning_state(f0 + p1, num_envs=512),
        rng=recovery.capture_rng(env), curriculum=curriculum, ledger=ledger,
        evaluation=dict(consumed=[], latest_diagnostic_update=None),
        source=recovery.source_block(mode='pilot', seed=seed), mode='pilot', seed=seed)
    path = Path(root) / f'pilot_{seed}.pt'
    recovery.atomic_torch_save(path, state)
    return path, recovery.sha256(path)


class SupervisorTrialTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch as _patch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in frozen_identity():
            patcher.start()
            self.addCleanup(patcher.stop)

    def _child(self, checkpoint, digest_value):
        def child(command, directory):
            return dict(status='chunk_complete', checkpoint=str(checkpoint),
                        sha256=digest_value, returncode=0,
                        evidence_index=[dict(path='/tmp/worker.log',
                                             sha256=digest('log'), kind='worker_log',
                                             update=700)])
        return child

    def test_trial_verdict_is_rederived_from_the_checkpoint(self):
        checkpoint, digest_value = synthetic_pilot_checkpoint(self.root, seed=42)
        supervisor = o.Supervisor(base_args(self.root, seed=42),
                                  child_runner=self._child(checkpoint, digest_value),
                                  preflight_dir=self.root / 'preflight')
        result = supervisor.pilot_trial(42)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['verdict']['status'], 'passed')
        self.assertEqual(result['verdict']['gates'], {100: False, 200: False, 300: False,
                                                      400: True, 500: True})
        self.assertEqual(result['checkpoint_sha256'], digest_value)
        self.assertEqual(len(result['evidence_index']), 2)

    def test_unconfirmed_monitoring_pauses_instead_of_training(self):
        class BrokenBackend:
            def __init__(self, *args, **kwargs):
                pass

            def start(self, *args, **kwargs):
                raise OSError('remote unavailable')

        supervisor = o.Supervisor(base_args(self.root),
                                  backend_factory=BrokenBackend,
                                  preflight_dir=self.root / 'preflight')
        supervisor.out.mkdir(parents=True, exist_ok=True)
        from src.tasks.velocity.v05.telemetry import MonitoringBlocked
        with self.assertRaises(MonitoringBlocked):
            supervisor.connect(config=dict(mode='pilot', seed=42))
        payload = json.loads((self.root / 'run' / 'control.json').read_text())
        self.assertEqual(payload['action'], 'pause')
        self.assertEqual(payload['reason'], 'monitoring_blocked')

    def test_trial_rejects_a_checkpoint_from_another_seed(self):
        checkpoint, digest_value = synthetic_pilot_checkpoint(self.root, seed=42)
        supervisor = o.Supervisor(base_args(self.root, seed=43),
                                  child_runner=self._child(checkpoint, digest_value),
                                  preflight_dir=self.root / 'preflight')
        with self.assertRaises(o.PilotError):
            supervisor.pilot_trial(43)

    def test_pilot_sequence_writes_both_reports(self):
        supervisor = o.Supervisor(base_args(self.root), preflight_dir=self.root / 'preflight')

        def trial(seed, fresh=True):
            verdict = pilot_verdict()
            return dict(status='passed', verdict=verdict,
                        checkpoint_sha256=digest(f'trial-{seed}'),
                        evidence_index=[dict(path=f'/tmp/p-{seed}',
                                             sha256=digest(f'raw-{seed}'),
                                             kind='trial_checkpoint', update=700)])
        supervisor.pilot_trial = trial
        summary = supervisor.run_pilot()
        self.assertTrue(summary['formal_ready'])
        self.assertEqual(summary['status'], 'pilot_complete')
        for seed in o.PILOT_SEEDS:
            self.assertTrue((self.root / 'run' / 'pilots' /
                             f'pilot_seed{seed}_report.json').is_file())
        self.assertTrue((self.root / 'run' / 'final.json').is_file())


# ---------------------------------------------------------------- resume-run
def resume_args(tmp, *, mode='pilot', seed=42, run_dir, **overrides):
    values = dict(mode=mode, seed=seed, resume_run=True,
                  init_checkpoint=None, init_checkpoint_sha256=None,
                  output_dir=str(run_dir))
    if mode == 'formal':
        values.update(stage_budgets=spec.FORMAL_STAGE_BUDGETS, pilot_p1_updates=None)
    values.update(overrides)
    return base_args(tmp, **values)


def write_run(tmp, *, mode='pilot', seed=42, reports=None):
    values = dict(mode=mode, seed=seed, output_dir=str(tmp / 'run'))
    if mode == 'formal':
        values.update(pilot_report_42=str(reports[42]), pilot_report_43=str(reports[43]),
                      stage_budgets=spec.FORMAL_STAGE_BUDGETS, pilot_p1_updates=None)
    supervisor = o.Supervisor(base_args(tmp, **values),
                              preflight_dir=tmp / 'preflight')
    supervisor.out.mkdir(parents=True, exist_ok=True)
    supervisor.write_run_contract()
    return supervisor


def write_pilot_report(path, seed, *, status='passed', binding_value=None, **verdict_overrides):
    verdict = pilot_verdict(**verdict_overrides)
    report = report_for(seed, status=status, verdict=verdict, binding_value=binding_value)
    Path(path).write_text(json.dumps(report))
    return report


def seed_trial_ledger(target, *, mode='pilot', seed=42):
    """A real ledger with one committed synthetic schema-6 checkpoint."""
    ledger = recovery.AttemptLedger(target / 'ledger',
                                    run_id=o.Supervisor.ledger_run_id(mode, seed),
                                    mode=mode)
    env = SimpleNamespace(rng=torch.Generator().manual_seed(seed))
    state = recovery.build_checkpoint(
        model_state={'w': torch.zeros(1)}, optimizer_state={}, normalizer={},
        learning=recovery.learning_state(0, num_envs=512),
        rng=recovery.capture_rng(env), curriculum=ledger.curriculum, ledger=ledger,
        evaluation=ledger.checkpoint_evaluation(),
        source=recovery.source_block(mode=mode, seed=seed), mode=mode, seed=seed)
    path = target / 'checkpoints' / 'update_000000.pt'
    recovery.atomic_torch_save(path, state)
    digest_value = recovery.sha256(path)
    ledger.record_checkpoint(path, digest_value, ledger.anchor(),
                             state['contract_sha256'])
    return ledger, path, digest_value


class ResumeRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.preflight = self.root / 'preflight'
        for patcher in frozen_identity():
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fsync = patch('os.fsync')
        self.fsync.start()
        self.addCleanup(self.fsync.stop)

    def test_new_run_contract_is_written_and_resume_matches_it(self):
        supervisor = write_run(self.root)
        record = json.loads((supervisor.out / 'run_contract.json').read_text())
        self.assertEqual(record['schema'], o.RUN_CONTRACT_SCHEMA)
        self.assertEqual(record['mode'], 'pilot')
        self.assertEqual(record['seed'], 42)
        self.assertEqual(record['configuration']['stage_budgets'],
                         spec.PILOT_STAGE_BUDGETS)
        self.assertEqual(record['configuration']['pilot_p1_updates'], 500)
        self.assertEqual(record['init']['sha256'], spec.INIT_SHA256)
        # The same contract resumes; the preflight does not create any directory.
        args = resume_args(self.root, run_dir=supervisor.out)
        resumed = o.Supervisor(args, preflight_dir=self.preflight)
        record2 = resumed.preflight()
        self.assertTrue(record2['resume'])
        self.assertEqual(record2['init_checkpoint_sha256'], spec.INIT_SHA256)
        self.assertEqual(record2['run_contract_sha256'],
                         recovery.sha256(supervisor.out / 'run_contract.json'))
        o.validate_arguments(args, check_files=True)

    def test_resume_rejects_any_cli_change_from_the_original_contract(self):
        supervisor = write_run(self.root)
        cases = (dict(seed=43), dict(mode='formal'),
                 dict(recipe_version='0.4-r1'), dict(num_envs=256),
                 dict(stage_budgets=spec.FORMAL_STAGE_BUDGETS),
                 dict(pilot_p1_updates=400), dict(eval_num_envs=4))
        for overrides in cases:
            with self.subTest(overrides=overrides):
                args = resume_args(self.root, run_dir=supervisor.out, **overrides)
                with self.assertRaises((ValueError, o.FormalAdmissionError)):
                    o.Supervisor(args, preflight_dir=self.preflight).preflight()

    def test_resume_and_init_are_mutually_exclusive(self):
        supervisor = write_run(self.root)
        args = resume_args(self.root, run_dir=supervisor.out,
                           init_checkpoint=spec.INIT_PATH,
                           init_checkpoint_sha256=spec.INIT_SHA256)
        with self.assertRaises(ValueError):
            o.validate_arguments(args, check_files=False)
        with self.assertRaises(ValueError):
            o.Supervisor(args, preflight_dir=self.preflight).preflight()

    def test_resume_selects_the_ledger_complete_boundary_and_builds_the_command(self):
        supervisor = write_run(self.root)
        target = supervisor.out / 'pilot_seed42'
        ledger = recovery.AttemptLedger(target / 'ledger',
                                        run_id=o.Supervisor.ledger_run_id('pilot', 42),
                                        mode='pilot')
        # Two same-update records: selection must follow ledger order, not mtime.
        first_path = target / 'checkpoints' / 'first.pt'
        env = SimpleNamespace(rng=torch.Generator().manual_seed(1))
        first = recovery.build_checkpoint(
            model_state={'w': torch.zeros(1)}, optimizer_state={}, normalizer={},
            learning=recovery.learning_state(0, num_envs=512),
            rng=recovery.capture_rng(env), curriculum=ledger.curriculum, ledger=ledger,
            evaluation=ledger.checkpoint_evaluation(),
            source=recovery.source_block(mode='pilot', seed=42), mode='pilot', seed=42)
        recovery.atomic_torch_save(first_path, first)
        ledger.record_checkpoint(first_path, recovery.sha256(first_path),
                                 ledger.anchor(), first['contract_sha256'])
        second_path = target / 'checkpoints' / 'second.pt'
        second = recovery.build_checkpoint(
            model_state={'w': torch.zeros(1)}, optimizer_state={}, normalizer={},
            learning=recovery.learning_state(0, num_envs=512),
            rng=recovery.capture_rng(env), curriculum=ledger.curriculum, ledger=ledger,
            evaluation=ledger.checkpoint_evaluation(),
            source=recovery.source_block(mode='pilot', seed=42), mode='pilot', seed=42)
        recovery.atomic_torch_save(second_path, second)
        ledger.record_checkpoint(second_path, recovery.sha256(second_path),
                                 ledger.anchor(), second['contract_sha256'])
        # Touch the older file so an mtime-based pick would choose it.
        os.utime(first_path, (time.time() + 100, time.time() + 100))
        boundary = supervisor.select_resume_boundary('pilot', 42)
        self.assertEqual(Path(boundary['path']), second_path)
        self.assertEqual(boundary['sha256'], recovery.sha256(second_path))
        self.assertEqual(boundary['audit']['status'], 'eligible')
        command = supervisor.worker_command(target, seed=42, mode='pilot',
                                            resume=boundary)
        joined = ' '.join(command)
        self.assertIn(f'--resume {second_path}', joined)
        self.assertIn(f"--resume-sha256 {recovery.sha256(second_path)}", joined)
        self.assertNotIn('--init-checkpoint', joined)
        self.assertIn('--ledger-dir ' + str(target / 'ledger'), joined)
        self.assertIn('--ledger-run-id ' + o.Supervisor.ledger_run_id('pilot', 42), joined)
        self.assertIn('--pilot-p1-updates 500', joined)

    def test_pilot_resume_skips_finished_seed_and_keeps_the_500_rule(self):
        supervisor = write_run(self.root)
        reports_dir = supervisor.out / 'pilots'
        reports_dir.mkdir(parents=True)
        write_pilot_report(reports_dir / 'pilot_seed42_report.json', 42)
        target = supervisor.out / 'pilot_seed43'
        target.mkdir()
        seed_trial_ledger(target, mode='pilot', seed=43)
        calls = []

        def fake_trial(seed, fresh=True, resume=None):
            calls.append(dict(seed=seed, fresh=fresh, resume=resume))
            return dict(status='passed', verdict=pilot_verdict(),
                        checkpoint_sha256=digest(f'trial-{seed}'),
                        evidence_index=[dict(path='/tmp/pilot43',
                                             sha256=digest('p43'),
                                             kind='trial_checkpoint', update=700)],
                        run_id=f'pilot_seed{seed}', failure_reason=None)

        supervisor.pilot_trial = fake_trial
        summary = supervisor.resume_pilot()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['seed'], 43)
        self.assertFalse(calls[0]['fresh'])
        self.assertIsNotNone(calls[0]['resume'])
        self.assertEqual(summary['reports'][42]['checkpoint_sha256'],
                         json.loads((reports_dir / 'pilot_seed42_report.json').read_text())
                         ['checkpoint_sha256'])
        self.assertTrue(summary['formal_ready'])
        report43 = json.loads((reports_dir / 'pilot_seed43_report.json').read_text())
        self.assertEqual(report43['p1']['budget'], 500)
        self.assertLessEqual(report43['total_updates'], o.PILOT_MAX_UPDATES)
        # The resumed trial's command still carries the fixed pilot contract.
        command = supervisor.worker_command(target, seed=43, mode='pilot',
                                            resume=calls[0]['resume'])
        self.assertIn('--pilot-p1-updates 500', ' '.join(command))

    def test_pause_is_converted_to_resume_and_needs_a_durable_boundary(self):
        supervisor = write_run(self.root)
        supervisor.out.mkdir(parents=True, exist_ok=True)
        (supervisor.out / 'control.json').write_text(json.dumps(
            dict(action='pause', reason='monitoring_blocked', requested=time.time())))
        supervisor.clear_pause()
        payload = json.loads((supervisor.out / 'control.json').read_text())
        self.assertEqual(payload['action'], 'resume')
        self.assertEqual(payload['reason'], 'monitoring_blocked')
        with self.assertRaises(o.RecoveryBlocked):
            supervisor.select_resume_boundary('pilot', 42)

    def test_resume_blocks_when_the_recorded_checkpoint_is_missing(self):
        supervisor = write_run(self.root)
        target = supervisor.out / 'pilot_seed42'
        target.mkdir()
        ledger, path, digest_value = seed_trial_ledger(target, mode='pilot', seed=42)
        boundary = supervisor.select_resume_boundary('pilot', 42)
        path.unlink()
        with self.assertRaises(o.RecoveryBlocked):
            supervisor.select_resume_boundary('pilot', 42)

    def test_resume_preserves_the_previous_attempt_history(self):
        supervisor = write_run(self.root)
        prior = dict(mode='pilot', seed=42, phase='paused', attempts=[dict(seed=42)],
                     resume_sessions=[])
        (supervisor.out / 'supervisor_state.json').write_text(json.dumps(prior))
        state = supervisor._load_previous_state()
        self.assertEqual(state['phase'], 'resuming')
        self.assertEqual(state['attempts'], [dict(seed=42)])
        self.assertEqual(len(state['resume_sessions']), 1)

    def test_formal_resume_requires_both_pilot_reports_still_binding(self):
        reports = {}
        for seed in o.PILOT_SEEDS:
            path = self.root / f'report{seed}.json'
            write_pilot_report(path, seed)
            reports[seed] = path
        supervisor = write_run(self.root, mode='formal', seed=42, reports=reports)
        contract = json.loads((supervisor.out / 'run_contract.json').read_text())
        self.assertEqual(set(contract['pilot_reports']), {'42', '43'})
        args = resume_args(self.root, mode='formal', seed=42, run_dir=supervisor.out)
        self.assertTrue(o.Supervisor(args, preflight_dir=self.preflight).preflight()['resume'])
        # A missing report is refused.
        reports[43].unlink()
        with self.assertRaises(o.FormalAdmissionError):
            o.Supervisor(args, preflight_dir=self.preflight).preflight()
        # A changed report file is refused (sha mismatch).
        write_pilot_report(reports[43], 43)
        reports[43].write_text(reports[43].read_text() + ' ')
        with self.assertRaises((o.FormalAdmissionError, ValueError)):
            o.Supervisor(args, preflight_dir=self.preflight).preflight()
        # A stale binding is refused even with a fresh file hash.
        write_pilot_report(reports[43], 43,
                           binding_value=binding_with(source_sha=digest('old-code')))
        with self.assertRaises((o.FormalAdmissionError, ValueError)):
            o.Supervisor(args, preflight_dir=self.preflight).preflight()

    def test_formal_resume_launches_from_the_selected_boundary(self):
        reports = {}
        for seed in o.PILOT_SEEDS:
            path = self.root / f'report{seed}.json'
            write_pilot_report(path, seed)
            reports[seed] = path
        supervisor = write_run(self.root, mode='formal', seed=42, reports=reports)
        target = supervisor.out / 'formal_seed42'
        target.mkdir()
        ledger, path, digest_value = seed_trial_ledger(target, mode='formal', seed=42)
        commands = []

        def fake_child(command, directory):
            commands.append(command)
            return dict(status='chunk_complete', checkpoint=str(path),
                        sha256=digest_value, returncode=0)

        supervisor.child_runner = fake_child
        final = supervisor.resume_formal()
        self.assertTrue(final['resumed'])
        self.assertEqual(final['resume_boundary']['sha256'], digest_value)
        joined = ' '.join(commands[0])
        self.assertIn(f'--resume {path}', joined)
        self.assertIn(f'--resume-sha256 {digest_value}', joined)
        self.assertNotIn('--init-checkpoint', joined)
        self.assertEqual(final['admission']['status'], 'admitted_by_resume')
        self.assertEqual(final['admission']['pilot_checkpoints'],
                         {seed: json.loads(reports[seed].read_text())['checkpoint_sha256']
                          for seed in o.PILOT_SEEDS})



# ------------------------------------------------------------------ W7 probe
class ProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def update_critic_statistics(self, obs):
        return None


class ProbePPO:
    """Minimal PPO with the ledger callback sequence and the probe sampling hook."""
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.cfg = type('Cfg', (), dict(rollout=48, epochs=5, minibatches=4))()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=spec.PPO_INITIAL_LR)
        self.learning_state = dict(optimizer_initialized=False, completed_ppo_updates=0,
                                   optimizer_steps=0)
        self.obs = None
        self.storage = None
        self.control_step_callback = None
        self.update_callback = None
        self.diagnostics_callback = None
        self.user_diagnostics_callback = None

    @property
    def learning(self):
        update = self.learning_state['completed_ppo_updates']
        samples = 48 * self.env.num_envs * update
        return dict(update=update, optimizer_step=self.learning_state['optimizer_steps'],
                    control_samples=samples, physics_substeps=4 * samples)

    def collect(self):
        if self.obs is None:
            self.obs, _ = self.env.reset()
        for step in range(48):
            self.obs, _, _, _, info = self.env.step(
                torch.zeros(self.env.num_envs, 12))
            if self.control_step_callback is not None:
                self.control_step_callback(dict(step_index=step + 1,
                                                num_envs=self.env.num_envs,
                                                physics_samples_increment=self.env.num_envs,
                                                info=info))
            if self.user_diagnostics_callback is not None:
                self.user_diagnostics_callback(info)
        return {}

    def update(self):
        base = self.learning_state['completed_ppo_updates']
        if self.update_callback is not None:
            self.update_callback('optimizer_started',
                                 dict(learning_state=self.learning,
                                      expected_optimizer_steps=20))
        for _ in range(20):
            self.optimizer.zero_grad(set_to_none=True)
            self.model.weight.square().backward()
            self.optimizer.step()
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


class ProbeEnv:
    """CPU stand-in mirroring the V05Env reset-on-stage-change behaviour."""
    device = 'cpu'

    def __init__(self, num_envs=8, device='cpu', seed=11, stage='P1',
                 stage_iteration=0, evaluation=False, fixed_vx=None, **_kwargs):
        self.num_envs = int(num_envs)
        self.rng = torch.Generator().manual_seed(seed)
        self.stage = stage
        self.stage_iteration = int(stage_iteration)
        self.evaluation = bool(evaluation)
        self.fixed_vx = fixed_vx
        self.resets = 0
        self.reset_count = 0
        self.stage_resets = 0
        self.steps = 0
        self.closed = False
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0

    def set_stage(self, stage, stage_iteration=0, previous_stage=None):
        changed = stage != self.stage
        self.stage = stage
        self.stage_iteration = int(stage_iteration)
        if changed:
            self.stage_resets += 1
            self._reset()

    def _obs(self):
        return {'proprio': torch.zeros(self.num_envs, 45)}

    def _reset(self):
        self.resets += 1
        self.reset_count += 1
        torch.rand(1, generator=self.rng)
        self.episode_length_buf.zero_()
        return self._obs()

    def reset(self):
        return self._reset(), {}

    def step(self, action):
        self.steps += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs), zeros, zeros, {}

    def close(self):
        self.closed = True


class ProbeCaseTable:
    def __init__(self, stage, kind, seed):
        self.stage = stage
        self.kind = kind
        self.seed = int(seed)
        self.episodes = 20
        self.cases = [dict(case_id=f'c{index}') for index in range(60)]
        self.sha256 = digest(f'{stage}|{kind}|{self.seed}')


class ProbeEvaluation:
    def __init__(self, polls=6, expected_mode='probe'):
        self.polls = int(polls)
        self.calls = []
        self.expected_mode = expected_mode

    def __call__(self, model, cases, *, device, num_envs, stage, kind, run_id,
                 global_update, checkpoint_sha, output_dir, record_video=False,
                 mode='formal', seed=42, monitor=None):
        if mode != self.expected_mode:
            raise ValueError(f'evaluation mode must be {self.expected_mode!r} for W7; '
                             f'got {mode!r}')
        self.calls.append(dict(stage=stage, kind=kind, num_envs=num_envs,
                               global_update=global_update,
                               checkpoint_sha=checkpoint_sha, mode=mode))
        for _ in range(self.polls):
            if monitor is not None:
                monitor.poll()
        return SimpleNamespace(summary=dict(
            per_speed=[dict(c=speed, total=20, case_pass_count=20,
                            speed_pass_count=20) for speed in spec.SPEEDS],
            gate_pass=True, statistics_unchanged=True, evidence_verified=True,
            episodes=60, raw_paths={}))


def probe_factories(*, sample_fn, polls=6, s5_steps=4):
    environments = []

    def env_factory(**kwargs):
        env = ProbeEnv(**kwargs)
        environments.append(env)
        return env

    return dict(
        env_factory=env_factory,
        model_factory=lambda path, sha, seed, device: ProbeModel(),
        ppo_factory=lambda model, env: ProbePPO(model, env),
        evaluation=ProbeEvaluation(polls=polls),
        make_cases=lambda stage, kind, seed: ProbeCaseTable(stage, kind, seed),
        sample_fn=sample_fn,
        s5_env_factory=env_factory,
        s5_steps_fn=lambda n, h, c: int(s5_steps)), environments


def safe_samples():
    calls = {'count': 0}

    def sample():
        calls['count'] += 1
        return dict(gpu_used_mib=1000.0, process_gpu_mib=500.0,
                    host_available_mib=100000.0, disk_free_gib=100.0,
                    rss_mib=400.0, torch_allocated_mib=200.0,
                    torch_reserved_mib=300.0)
    return sample, calls


class ProbeExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in frozen_identity():
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fsync = patch('os.fsync')
        self.fsync.start()
        self.addCleanup(self.fsync.stop)

    def _probe_args(self, **overrides):
        values = dict(mode='probe', execute=True,
                      stage_budgets=spec.FORMAL_STAGE_BUDGETS,
                      pilot_p1_updates=None,
                      output_dir=str(self.root / 'probe_run'))
        values.update(overrides)
        return base_args(self.root, **values)

    def test_probe_runs_five_plus_fifty_plus_one_and_reports_everything(self):
        sample, calls = safe_samples()
        factories, environments = probe_factories(sample_fn=sample, polls=6,
                                                  s5_steps=4)
        supervisor = o.Supervisor(self._probe_args(), preflight_dir=self.root / 'preflight',
                                  probe_factories=factories)
        record = supervisor.run()
        self.assertEqual(record['status'], 'passed')
        self.assertEqual(record['schema'], o.PROBE_SCHEMA)
        self.assertEqual(record['identity']['mode'], 'smoke')
        self.assertEqual(record['identity']['ledger_mode'], 'smoke')
        self.assertEqual(record['identity']['evaluation_mode'], 'probe')
        self.assertEqual(record['identity']['probe_stage'], 'P1')
        self.assertIn('--mode probe', ' '.join(record['command']) + ' --mode probe')
        statistics = record['statistics']
        self.assertEqual(statistics['warmup_completed'], 5)
        self.assertEqual(statistics['measured_completed'], 50)
        self.assertEqual(statistics['recovery_completed'], 1)
        phases = [entry['phase'] for entry in record['updates']]
        self.assertEqual(phases[:5], ['train_warmup'] * 5)
        self.assertEqual(phases[5:55], ['train_measured'] * 50)
        self.assertEqual(phases[55], 'train_recovery')
        self.assertEqual(len(record['updates']), 56)
        # Separate costs, exact learning units, no formal/pilot pollution.
        costs = record['costs']
        self.assertEqual(costs['learning']['updates'], 56)
        self.assertEqual(costs['learning']['control_samples'], 56 * 48 * 512)
        self.assertEqual(costs['learning']['physics_substeps'], 4 * 56 * 48 * 512)
        self.assertEqual(costs['coexistence_evaluation']['episodes'], 60)
        self.assertEqual(costs['coexistence_evaluation']['control_samples_upper_bound'],
                         60 * 4)
        self.assertEqual(costs['s5_no_learning']['control_samples'], 3 * 4)
        self.assertEqual(costs['s5_no_learning']['physics_substeps'], 4 * 3 * 4)
        self.assertEqual(costs['recovery']['updates'], 1)
        self.assertEqual(costs['recovery']['control_samples'], 48 * 512)
        self.assertEqual(costs['recovery']['physics_substeps'], 4 * 48 * 512)
        self.assertFalse(costs['counts_against_formal'])
        self.assertFalse(costs['counts_against_pilot'])
        self.assertEqual(costs['ledger_control_samples'], 56 * 48 * 512)
        # Thresholds: all four SDS limits observed and passed.
        self.assertTrue(record['thresholds']['passed'])
        self.assertTrue(record['thresholds']['complete'])
        self.assertEqual({item['metric'] for item in record['thresholds']['items']},
                         {'whole_device_mib', 'process_gpu_mib', 'host_available_mib',
                          'disk_free_gib'})
        for item in record['thresholds']['items']:
            self.assertEqual(item['status'], 'passed')
        self.assertEqual(record['worst']['gpu_used_mib'], 1000.0)
        self.assertEqual(record['worst']['host_available_mib'], 100000.0)
        # Phases: train + coexistence eval + S5 no-learning + recovery.
        phase_names = [phase['phase'] for phase in record['phases']]
        self.assertEqual(len(record['phases']), 4)
        self.assertEqual(phase_names,
                         ['train', 'coexistence_evaluation', 's5_no_learning',
                          'recovery'])
        self.assertEqual(record['failures'], [])
        evaluation_call = factories['evaluation'].calls[0]
        self.assertEqual(evaluation_call['mode'], 'probe')
        self.assertEqual(evaluation_call['stage'], 'P1')
        self.assertEqual(evaluation_call['num_envs'], 8)
        recovery_phase = record['phases'][-1]
        self.assertEqual(recovery_phase['status'], 'resumed')
        self.assertEqual(recovery_phase['update'], 56)
        self.assertEqual(recovery_phase['resume_boundary']['sha256'],
                         record['boundaries']['probe_checkpoint']['sha256'])
        self.assertTrue((self.root / 'probe_run' / 'probe' / 'checkpoints' /
                         'probe_boundary_000055.pt').is_file())
        # Artifacts and per-control-step resources.
        probe_dir = self.root / 'probe_run' / 'probe'
        for name in ('probe.json', 'resources.jsonl', 'summary.md'):
            self.assertTrue((probe_dir / name).is_file(), name)
        resources = [json.loads(line) for line in
                     (probe_dir / 'resources.jsonl').read_text().splitlines()]
        self.assertEqual(len(resources), 56 * 48 + 6 + 3 * 4)
        self.assertEqual({item['phase'] for item in resources},
                         {'train_warmup', 'train_measured', 'train_recovery',
                          'coexistence_eval', 's5_no_learning'})
        self.assertEqual(record['artifacts']['resources_sha256'],
                         recovery.sha256(probe_dir / 'resources.jsonl'))
        self.assertEqual(record['artifacts']['summary_sha256'],
                         recovery.sha256(probe_dir / 'summary.md'))
        self.assertTrue((self.root / 'probe_run' / 'final.json').is_file())
        # The first environment was closed before the resumed one; the resumed env
        # performed exactly the single resume reset.
        self.assertTrue(environments[0].closed)
        self.assertEqual(environments[1].resets, 1)

    def test_probe_fails_closed_when_the_evaluation_mode_is_rejected(self):
        sample, _ = safe_samples()
        factories, environments = probe_factories(sample_fn=sample)
        # A B-package that still rejects mode='probe' must abort the probe after
        # the training phase, with recovery/S5 costs at zero (never hidden).
        factories['evaluation'] = ProbeEvaluation(expected_mode='pilot')
        record = o.run_resource_probe(
            output_dir=self.root / 'reject_run', seed=42, device='cuda:0',
            num_envs=512, eval_num_envs=8, command=['probe'], **factories)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual([phase['phase'] for phase in record['phases']], ['train'])
        self.assertEqual(record['costs']['recovery']['control_samples'], 0)
        self.assertEqual(record['costs']['recovery']['updates'], 0)
        self.assertEqual(record['costs']['s5_no_learning']['control_samples'], 0)
        self.assertTrue(record['failures'])
        self.assertIn('evaluation mode', record['failures'][0]['reason'])
        probe_dir = self.root / 'reject_run' / 'probe'
        for name in ('probe.json', 'resources.jsonl', 'summary.md'):
            self.assertTrue((probe_dir / name).is_file(), name)
        resources = [json.loads(line) for line in
                     (probe_dir / 'resources.jsonl').read_text().splitlines()]
        self.assertEqual({item['phase'] for item in resources},
                         {'train_warmup', 'train_measured'})

    def test_probe_stops_immediately_when_a_guard_is_crossed(self):
        state = {'count': 0}

        def sample():
            state['count'] += 1
            return dict(gpu_used_mib=8000.0 if state['count'] >= 10 else 1000.0,
                        process_gpu_mib=500.0, host_available_mib=100000.0,
                        disk_free_gib=100.0, rss_mib=400.0,
                        torch_allocated_mib=200.0, torch_reserved_mib=300.0)

        factories, environments = probe_factories(sample_fn=sample)
        record = o.run_resource_probe(
            output_dir=self.root / 'limit_run', seed=42, device='cuda:0',
            num_envs=512, eval_num_envs=8, command=['probe'], **factories)
        self.assertEqual(record['status'], 'resource_failed')
        self.assertEqual(record['failures'][0]['kind'], 'resource_limit')
        self.assertIn('whole_device_mib>7168',
                      record['failures'][0]['violations'][0])
        self.assertEqual(record['updates'], [])
        self.assertEqual(record['phases'], [])
        self.assertEqual(record['worst']['gpu_used_mib'], 8000.0)
        probe_dir = self.root / 'limit_run' / 'probe'
        for name in ('probe.json', 'resources.jsonl', 'summary.md'):
            self.assertTrue((probe_dir / name).is_file(), name)
        self.assertIn('resource_failed', (probe_dir / 'probe.json').read_text())
        # No auto-downgrade: the configured environment count never changed.
        self.assertEqual(record['identity']['num_envs'], 512)
        self.assertTrue(environments and environments[0].num_envs == 512)

    def test_probe_cli_contract_and_preflight(self):
        sample, _ = safe_samples()
        factories, _ = probe_factories(sample_fn=sample)
        supervisor = o.Supervisor(self._probe_args(execute=False),
                                  preflight_dir=self.root / 'preflight',
                                  probe_factories=factories)
        record = supervisor.preflight()
        self.assertEqual(record['mode'], 'probe')
        self.assertFalse(record['execute'])
        self.assertFalse((self.root / 'probe_run').exists())
        self.assertTrue((self.root / 'preflight' /
                         'preflight_probe_seed42.json').is_file())
        self.assertEqual(supervisor.run(), record)
        self.assertFalse((self.root / 'probe_run').exists())
        for overrides in (dict(seed=43), dict(stage_budgets=spec.PILOT_STAGE_BUDGETS),
                          dict(pilot_p1_updates=500),
                          dict(pilot_report_42='/tmp/report.json'),
                          dict(resume_run=True)):
            with self.subTest(overrides=overrides):
                values = dict(mode='probe', stage_budgets=spec.FORMAL_STAGE_BUDGETS,
                              pilot_p1_updates=None,
                              output_dir=str(self.root / 'probe_run'))
                values.update(overrides)
                args = base_args(self.root, **values)
                with self.assertRaises((ValueError, o.FormalAdmissionError)):
                    o.validate_arguments(args, check_files=False)



if __name__ == '__main__':
    unittest.main()
