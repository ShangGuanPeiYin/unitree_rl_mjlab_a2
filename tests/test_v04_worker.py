"""CPU contracts for the v0.4 CLI entry points and T7 smoke harness.

The module bootstraps the isolated ``src.tasks`` namespace before importing any
v0.4 module, so these tests are valid both under
``scripts/test_a2_ideal_stairs_v04.py`` and when run directly.  No test creates
a simulator, CUDA context, W&B session or network connection.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v04_bootstrap  # noqa: E402

a2_v04_bootstrap.activate(str(ROOT))

from src.tasks.velocity.v04.spec import (  # noqa: E402
    INIT_SHA256, RECIPE_VERSION, TASK)

ENTRY_FILES = (
    'train_a2_ideal_stairs_v04.py',
    'run_a2_ideal_stairs_v04.py',
    'evaluate_a2_ideal_stairs_v04.py',
    'play_a2_ideal_stairs_v04.py',
    'smoke_a2_ideal_stairs_v04.py',
    'test_a2_ideal_stairs_v04.py',
)


def load_script(name):
    path = ROOT / 'scripts' / name
    module_name = 'v04_test_' + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_parser(module, argv):
    with redirect_stderr(io.StringIO()):
        return module.parse_args(argv)


class BootstrapOrdering(unittest.TestCase):
    def test_namespace_is_active_without_legacy_root_scanner(self):
        import src.tasks
        self.assertTrue(a2_v04_bootstrap.is_active(ROOT))
        self.assertEqual(a2_v04_bootstrap.active_root(), str(ROOT.resolve()))
        self.assertIsNone(getattr(src.tasks, '__file__', None))
        self.assertTrue(hasattr(src.tasks, '__a2_v04_bootstrap__'))

    def test_same_root_activation_is_idempotent(self):
        first = sys.modules['src.tasks']
        second = a2_v04_bootstrap.activate(str(ROOT))
        self.assertIs(first, second)

    def test_foreign_or_missing_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                a2_v04_bootstrap.activate(directory)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'src' / 'tasks').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                a2_v04_bootstrap.activate(directory)

    def test_every_entry_activates_before_first_task_import(self):
        import re
        for name in ENTRY_FILES:
            with self.subTest(entry=name):
                text = (ROOT / 'scripts' / name).read_text(encoding='utf-8')
                activation = text.find('a2_v04_bootstrap.activate')
                imports = [match.start() for match in
                           re.finditer(r'^\s*(?:from|import)\s+src\.tasks', text, re.MULTILINE)]
                self.assertNotEqual(activation, -1, f'{name} lacks bootstrap activation')
                if imports:
                    self.assertLess(activation, min(imports),
                                    f'{name} imports src.tasks before bootstrap')

    def test_test_runner_only_bootstraps_then_discovers(self):
        text = (ROOT / 'scripts' / 'test_a2_ideal_stairs_v04.py').read_text(encoding='utf-8')
        self.assertIn('unittest.TestLoader().discover', text)
        self.assertIn("--pattern", text)
        self.assertNotIn('wandb', text.lower())
        self.assertNotIn('subprocess', text)
        self.assertNotIn('import torch', text)
        self.assertNotIn('torch.cuda', text)


class TestLauncherContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = load_script('test_a2_ideal_stairs_v04.py')

    def test_pattern_is_forwarded_and_failure_is_nonzero(self):
        from types import SimpleNamespace
        seen = {}

        def discover(start, pattern=None, top_level_dir=None):
            seen.update(start=start, pattern=pattern, top_level_dir=top_level_dir)
            return object()

        with patch.object(self.t.unittest.TestLoader, 'discover', side_effect=discover), \
                patch.object(self.t.unittest.TextTestRunner, 'run',
                             return_value=SimpleNamespace(wasSuccessful=lambda: False)):
            self.assertEqual(self.t.main(['--pattern', 'test_v04_special.py']), 1)
        self.assertEqual(seen['pattern'], 'test_v04_special.py')
        self.assertTrue(seen['start'].endswith('tests'))

    def test_success_is_zero(self):
        from types import SimpleNamespace
        with patch.object(self.t.unittest.TestLoader, 'discover', return_value=object()), \
                patch.object(self.t.unittest.TextTestRunner, 'run',
                             return_value=SimpleNamespace(wasSuccessful=lambda: True)):
            self.assertEqual(self.t.main([]), 0)


class RunSupervisorCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load_script('run_a2_ideal_stairs_v04.py')

    def base(self, output, **overrides):
        values = dict(
            task=TASK, recipe_version=RECIPE_VERSION,
            init_checkpoint='init.pt', init_sha='2' * 64,
            legacy='manifest.json', num_envs=512, eval_num_envs=8, seed=42,
            device='cuda:0', vx_min='.2', vx_max='.6', vy='0', wz='0',
            max_iterations=23000, chunk=200, save=100, evaluate=200,
            wandb='online')
        values.update(overrides)
        argv = ['--task', values['task'], '--recipe-version', values['recipe_version']]
        if values.get('init_checkpoint'):
            argv += ['--init-checkpoint', values['init_checkpoint']]
            if values.get('init_sha') is not None:
                argv += ['--init-checkpoint-sha256', values['init_sha']]
        if values.get('resume'):
            argv += ['--resume', values['resume']]
            if values.get('resume_sha') is not None:
                argv += ['--resume-sha256', values['resume_sha']]
            if values.get('resume_run'):
                argv += ['--resume-run']
        if values.get('legacy') is not None:
            argv += ['--legacy-source-manifest', values['legacy']]
        if values.get('num_envs') is not None:
            argv += ['--num-envs', str(values['num_envs'])]
        if values.get('eval_num_envs') is not None:
            argv += ['--eval-num-envs', str(values['eval_num_envs'])]
        if values.get('seed') is not None:
            argv += ['--seed', str(values['seed'])]
        if values.get('device') is not None:
            argv += ['--device', values['device']]
        if values.get('vx_min') is not None:
            argv += ['--vx-min', str(values['vx_min'])]
        if values.get('vx_max') is not None:
            argv += ['--vx-max', str(values['vx_max'])]
        if values.get('vy') is not None:
            argv += ['--vy', str(values['vy'])]
        if values.get('wz') is not None:
            argv += ['--wz', str(values['wz'])]
        if values.get('max_iterations') is not None:
            argv += ['--max-iterations', str(values['max_iterations'])]
        if values.get('chunk') is not None:
            argv += ['--worker-chunk-iterations', str(values['chunk'])]
        if values.get('save') is not None:
            argv += ['--save-interval', str(values['save'])]
        if values.get('evaluate') is not None:
            argv += ['--eval-interval', str(values['evaluate'])]
        if values.get('wandb') is not None:
            argv += ['--wandb-mode', values['wandb']]
        if values.get('extra'):
            argv += list(values['extra'])
        argv += ['--output-dir', str(output)]
        return argv

    def test_accepts_the_exact_sds_cli_without_old_approval_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            a = self.r.validate_cli(self.base(output))
            self.assertEqual(a.num_envs, 512)
            self.assertEqual(a.eval_num_envs, 8)
            self.assertEqual(a.seed, 42)
            self.assertEqual(a.device, 'cuda:0')
            self.assertEqual((a.vx_min, a.vx_max, a.vy, a.wz), (.2, .6, 0., 0.))
            self.assertEqual((a.max_iterations, a.worker_chunk_iterations), (23000, 200))
            self.assertEqual((a.save_interval, a.eval_interval), (100, 200))
            self.assertFalse(output.exists())
            self.assertFalse(hasattr(a, 'review_report_required'))

    def test_help_is_left_to_the_delegated_parser(self):
        self.assertIsNone(self.r.validate_cli(['--help']))

    def test_nonfixed_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            cases = [
                dict(num_envs=1024), dict(num_envs=258), dict(eval_num_envs=4),
                dict(eval_num_envs=1), dict(seed=7), dict(device='cpu'),
                dict(vx_min=.3), dict(vx_max=.5), dict(vy=.1), dict(wz=.1),
                dict(max_iterations=200), dict(chunk=100), dict(save=50),
                dict(evaluate=100), dict(wandb='offline'),
                dict(task='Unitree-A2-IdealHeightMap-Stairs-V03'),
                dict(recipe_version='0.3'),
                dict(extra=['--num-envs-candidates', '1024', '512', '258']),
                dict(extra=['--allow-num-env-change']),
            ]
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaises((SystemExit, ValueError)):
                        self.r.validate_cli(self.base(output, **overrides))

    def test_init_and_resume_are_mutually_exclusive_with_explicit_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with self.assertRaises(ValueError):
                self.r.validate_cli(self.base(output, resume='parent.pt', resume_sha='3' * 64))
            with self.assertRaises(ValueError):
                self.r.validate_cli([a for a in self.base(output)
                                     if a not in ('--init-checkpoint', 'init.pt',
                                                  '--init-checkpoint-sha256', '2' * 64)])
            with self.assertRaises(ValueError):
                self.r.validate_cli(self.base(output, init_sha=None))
            with self.assertRaises(ValueError):
                self.r.validate_cli(self.base(output, legacy=None))
            resumed = output
            resumed.mkdir()
            (resumed / 'supervisor_state.json').write_text('{}')
            a = self.r.validate_cli(self.base(output, init_checkpoint=None,
                                              resume='parent.pt', resume_sha='3' * 64,
                                              resume_run=True))
            self.assertEqual(a.resume, 'parent.pt')
            with self.assertRaises(ValueError):
                self.r.validate_cli(self.base(output, init_checkpoint=None,
                                              resume='parent.pt', resume_sha='3' * 64))

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            output.mkdir()
            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                self.r.validate_cli(self.base(output))

    def test_no_old_run_approval_file_is_required_or_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            a = self.r.validate_cli(self.base(output))
            self.assertFalse(output.exists())
            self.assertEqual(a.num_envs, 512)
            with self.assertRaises(SystemExit):
                self.r.validate_cli(self.base(output, extra=['--review-report', 'audit.json']))


class WorkerCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = load_script('train_a2_ideal_stairs_v04.py')

    def argv(self, **overrides):
        values = dict(execution='initialize', num_envs=512, seed=42, device='cuda:0',
                      init='init.pt', init_sha=INIT_SHA256, resume=None, resume_sha=None,
                      manifest='manifest.json', output='out', ledger='ledger', run_id='run')
        values.update(overrides)
        argv = ['--task', TASK, '--recipe-version', RECIPE_VERSION,
                '--execution', values['execution'], '--num-envs', str(values['num_envs']),
                '--seed', str(values['seed']), '--device', values['device'],
                '--output-dir', values['output'], '--ledger-dir', values['ledger'],
                '--ledger-run-id', values['run_id']]
        if values['init']:
            argv += ['--init-checkpoint', values['init'],
                     '--init-checkpoint-sha256', values['init_sha']]
        if values['resume']:
            argv += ['--resume', values['resume']]
        if values['resume_sha']:
            argv += ['--resume-sha256', values['resume_sha']]
        if values['manifest'] is not None:
            argv += ['--legacy-source-manifest', values['manifest']]
        return argv

    def test_fixed_512_defaults_and_chunk(self):
        a = run_parser(self.w, self.argv())
        self.assertEqual(a.num_envs, 512)
        self.assertEqual(a.max_iterations, 23000)
        self.assertEqual(a.worker_chunk_iterations, 200)
        self.assertEqual((a.save_interval, a.eval_interval), (100, 200))
        self.assertEqual(a.recipe_version, RECIPE_VERSION)
        self.assertFalse(hasattr(a, 'review_report'))

    def test_non_512_and_old_candidate_controls_are_rejected(self):
        for count in (1024, 258, 8):
            with self.subTest(count=count):
                with self.assertRaises(SystemExit):
                    run_parser(self.w, self.argv(num_envs=count))
        for flag in (['--allow-num-env-change'], ['--num-envs-candidates', '512'],
                     ['--runtime-approval', 'approval.json'],
                     ['--review-report', 'review.json']):
            with self.subTest(flag=flag[0]):
                with self.assertRaises(SystemExit):
                    run_parser(self.w, self.argv() + flag)

    def test_fixed_budget_seed_device_recipe_and_intervals(self):
        for flags in (
                ['--max-iterations', '200'], ['--seed', '7'], ['--device', 'cpu'],
                ['--recipe-version', '0.3'], ['--save-interval', '50'],
                ['--eval-interval', '100'], ['--worker-chunk-iterations', '201'],
                ['--task', 'Unitree-A2-IdealHeightMap-Stairs-V03']):
            with self.subTest(flags=flags):
                argv = self.argv()
                if flags[0] in argv:
                    index = argv.index(flags[0])
                    argv = argv[:index] + flags + argv[index + 1:]
                else:
                    argv += flags
                with self.assertRaises(SystemExit):
                    run_parser(self.w, argv)

    def test_init_resume_exclusivity_and_explicit_sha(self):
        with self.assertRaises(SystemExit):
            run_parser(self.w, self.argv(init=None))
        with self.assertRaises(SystemExit):
            run_parser(self.w, self.argv(init=None, resume='parent.pt',
                                         resume_sha='3' * 64))
        with self.assertRaises(SystemExit):
            run_parser(self.w, self.argv(init='init.pt', init_sha=None))
        with self.assertRaises(SystemExit):
            run_parser(self.w, self.argv(init='init.pt', resume_sha='3' * 64))
        a = run_parser(self.w, self.argv(init=None, execution='train',
                                         resume='parent.pt', resume_sha='3' * 64))
        self.assertEqual(a.resume, 'parent.pt')

    def test_gate_flags_require_resume_and_case_sha(self):
        with self.assertRaises(SystemExit):
            run_parser(self.w, self.argv() + ['--validation', 'summary.json',
                                              '--validation-cases-sha256', '4' * 64])
        with self.assertRaises(SystemExit):
            run_parser(self.w, self.argv(init=None, execution='train', resume='parent.pt',
                                         resume_sha='3' * 64) + ['--validation', 'summary.json'])

    def test_swing_foothold_monitoring_accumulator_is_cpu_only(self):
        import torch
        accumulator = self.w.DiagnosticMeanAccumulator()
        accumulator.add('swing', {
            'phi': torch.tensor([[0., .5], [1., .5]]),
            'active': torch.tensor([[True, False], [True, True]]),
            'gate_rate': torch.tensor([True, False]),
        })
        accumulator.add('swing', {'phi': torch.tensor([[.5, .5], [.5, .5]])})
        accumulator.add('foothold', {'window_size': 25, 'count': torch.tensor([1, 2])})
        result = accumulator.result()
        self.assertAlmostEqual(result['swing/phi'], .5)
        self.assertAlmostEqual(result['swing/active'], .75)
        self.assertAlmostEqual(result['swing/gate_rate'], .5)
        self.assertAlmostEqual(result['foothold/window_size'], 25.)
        self.assertAlmostEqual(result['foothold/count'], 1.5)
        self.assertNotIn('reward_total', result)
        # The worker emits top-level swing/* / foothold/* keys; telemetry must
        # pass them through to the existing panels without a ninth group.
        from src.tasks.velocity.v04.telemetry import aggregate_updates
        row = {'sample_count': 384, 'num_envs': 8, 'global_iteration': 1,
               'stage': 'S1', 'stage_iteration': 1, 'env_steps': 384,
               'elapsed_s': 1., 'swing/phi': .5, 'swing/gate_rate': .25,
               'foothold/count': 2., 'motion_terms': {}, 'reward_sample_means': {}}
        aggregated = aggregate_updates([row], [])
        self.assertEqual(aggregated['swing/phi'], .5)
        self.assertEqual(aggregated['swing/gate_rate'], .25)
        self.assertEqual(aggregated['foothold/count'], 2.)

    def test_manifest_validation_rejects_bad_schema_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'legacy.json'
            bad.write_text(json.dumps({'schema': 'wrong'}))
            with self.assertRaises(ValueError):
                self.w.verify_legacy_manifest(bad)
            with self.assertRaises((ValueError, OSError)):
                self.w.verify_legacy_manifest(Path(directory) / 'missing.json')

    def test_train_authorization_requires_supervised_resume(self):
        a = run_parser(self.w, self.argv(init=None, execution='train',
                                         resume='parent.pt', resume_sha='3' * 64))
        with patch.object(self.w, 'verify_legacy_manifest', return_value={}):
            with self.assertRaisesRegex(ValueError, 'supervised'):
                self.w.verify_worker_authorization(a)
            a.resume = None
            a.init_checkpoint = 'init.pt'
            a.init_checkpoint_sha256 = INIT_SHA256
            with self.assertRaisesRegex(ValueError, 'supervised'):
                self.w.verify_worker_authorization(a)

    def test_initialize_authorization_does_not_require_gpu(self):
        a = run_parser(self.w, self.argv())
        with patch.object(self.w, 'verify_legacy_manifest', return_value={}):
            self.w.verify_worker_authorization(a)


class SmokeCliAndContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = load_script('smoke_a2_ideal_stairs_v04.py')

    def argv(self, **overrides):
        values = dict(task=TASK, stage='S1', num_envs=8, updates=2, seed=42,
                      device='cuda:0', vx_min='.2', vx_max='.6', vy='0', wz='0',
                      init='init.pt', init_sha=INIT_SHA256,
                      manifest='manifest.json', output='smoke')
        values.update(overrides)
        argv = ['--task', values['task'], '--stage', values['stage'],
                '--num-envs', str(values['num_envs']), '--updates', str(values['updates']),
                '--seed', str(values['seed']), '--device', values['device'],
                '--vx-min', str(values['vx_min']), '--vx-max', str(values['vx_max']),
                '--vy', str(values['vy']), '--wz', str(values['wz']),
                '--init-checkpoint', values['init'],
                '--init-checkpoint-sha256', values['init_sha'],
                '--legacy-source-manifest', values['manifest'],
                '--output-dir', values['output']]
        return argv

    def test_accepts_the_exact_t7_parameters_and_cpu_engineering_mode(self):
        a = run_parser(self.s, self.argv())
        self.assertEqual((a.stage, a.num_envs, a.updates, a.seed), ('S1', 8, 2, 42))
        self.assertEqual((a.vx_min, a.vx_max, a.vy, a.wz), (.2, .6, 0., 0.))
        self.assertEqual(a.init_checkpoint_sha256, INIT_SHA256)
        cpu = run_parser(self.s, self.argv(device='cpu'))
        self.assertEqual(cpu.device, 'cpu')

    def test_nonfixed_smoke_values_are_rejected(self):
        for overrides in (dict(stage='S2'), dict(num_envs=4), dict(num_envs=512),
                          dict(updates=1), dict(updates=55), dict(seed=7),
                          dict(device='cuda:1'), dict(vx_min=.3), dict(vx_max=.5),
                          dict(vy=.1), dict(wz=.1),
                          dict(task='Unitree-A2-IdealHeightMap-Stairs-V03')):
            with self.subTest(overrides=overrides):
                with self.assertRaises(SystemExit):
                    run_parser(self.s, self.argv(**overrides))

    def test_manifest_validation_and_source_sha_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'legacy.json'
            bad.write_text(json.dumps({'schema': 'a2-v04-legacy-manifest-v1',
                                       'manifest_schema': 'a2-v04-legacy-manifest-v1',
                                       'source_code_version': '0.2.1',
                                       'source_checkpoint_schema': 3,
                                       'source_checkpoint_sha256': '0' * 64}))
            with self.assertRaises(ValueError):
                self.s.verify_legacy_manifest(bad)

    def test_container_name_and_explicit_non_formal_semantics(self):
        self.assertEqual(self.s.SMOKE_CONTAINER, 'v04-smoke-state-v1')
        self.assertEqual(self.s.FIXED_UPDATES, 2)
        self.assertEqual(self.s.FIXED_NUM_ENVS, 8)
        self.assertEqual(self.s.FIXED_STAGE, 'S1')

    def test_state_compare_is_exact_and_rng_roundtrips(self):
        import torch
        left = {'a': torch.tensor([1., 2.]), 'b': {'c': torch.tensor([3])}}
        right = {'a': torch.tensor([1., 2.]), 'b': {'c': torch.tensor([3])}}
        self.assertTrue(self.s._compare_state(left, right, 'test')['equal'])
        right['a'][0] = 9
        self.assertFalse(self.s._compare_state(left, right, 'test')['equal'])
        class Fake:
            rng = torch.Generator().manual_seed(42)
        captured = self.s._capture_rng(Fake())
        self.assertTrue(self.s._check_rng(captured)['equal'])
        captured['environment'] = torch.zeros_like(captured['environment'])
        self.assertFalse(self.s._check_rng(captured)['equal'])

    def test_cpu_state_keeps_tensors_on_cpu_and_json_rejects_nan(self):
        import torch
        state = self.s._cpu_state({'x': torch.tensor([1.], device='cpu')})
        self.assertEqual(state['x'].device.type, 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            with self.assertRaises(ValueError):
                self.s._atomic_json(path, {'not_finite': float('nan')})
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
