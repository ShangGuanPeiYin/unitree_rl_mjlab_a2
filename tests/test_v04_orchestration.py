"""CPU-only v04 formal supervisor CLI and preflight contract tests.

No child training, GPU, W&B session or network request is executed.  The
Supervisor lock and command builders are exercised with temporary fixtures.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager, redirect_stderr
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v04_bootstrap
a2_v04_bootstrap.activate(str(ROOT))

from src.tasks.velocity.v04 import orchestration as o
from src.tasks.velocity.v04.spec import CODE_VERSION, PROTOCOL_ID, RECIPE_VERSION, SDS_REVISION, TASK


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = root
        self.source = root / 'source.pt'
        self.source.write_bytes(b'v02 source fixture for CLI tests')
        self.source_sha = digest_bytes(self.source.read_bytes())
        self.init = root / 'init.pt'
        self.init.write_bytes(b'initialization fixture for CLI tests')
        self.init_sha = digest_bytes(self.init.read_bytes())
        self.snapshot_dir = root / 'legacy'
        self.snapshot = self.snapshot_dir / 'sources/scripts/fixture.py'
        self.snapshot.parent.mkdir(parents=True)
        self.snapshot.write_text("print('frozen v02 source fixture')\n")
        self.snapshot_sha = digest_bytes(self.snapshot.read_bytes())
        # The patched verifier also checks that the current project source bytes
        # still match the frozen snapshot and that coverage equals exactly the
        # source contract's behavior set.
        self.current = root / 'scripts/fixture.py'
        self.current.parent.mkdir(parents=True)
        self.current.write_bytes(self.snapshot.read_bytes())
        self.contract = {'behavior_sha256': {'scripts/fixture.py': self.snapshot_sha}}
        self.manifest_path = self.snapshot_dir / 'legacy_manifest.json'
        self.manifest = dict(
            schema='a2-v04-legacy-manifest-v1', manifest_schema='a2-v04-legacy-manifest-v1',
            project_root=str(root), source_checkpoint_path=str(self.source),
            source_checkpoint_sha256=self.source_sha, source_code_version='0.2.1',
            source_checkpoint_schema=3, source_contract=self.contract,
            source_contract_sha256=digest_bytes(canonical(self.contract).encode()),
            source_files=[dict(relative_path='scripts/fixture.py',
                origin_absolute_path=str(self.snapshot),
                snapshot_relative_path='sources/scripts/fixture.py',
                sha256=self.snapshot_sha, bytes=self.snapshot.stat().st_size)],
            audit_created_at='2026-11-13T00:00:00+0800')
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.out = root / 'run'


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.f = Fixture(self.root)

    @contextmanager
    def patched(self):
        with patch.object(o, 'ROOT', self.root), \
             patch.object(o, 'INIT_SHA256', self.f.source_sha), \
             patch.object(o, 'approved_init_kind', lambda path, digest: 'v02_formal_3000'), \
             patch.object(o, 'contract', lambda *a, **k: {'fixture_contract': True}), \
             patch.object(o, 'behavior_hashes', lambda *a, **k: {'fixture.py': 'hash'}):
            yield

    def args(self, *extra, stop=False):
        argv = ['--task', TASK, '--recipe-version', RECIPE_VERSION,
                '--legacy-source-manifest', str(self.f.manifest_path),
                '--output-dir', str(self.f.out), '--num-envs', '512',
                '--eval-num-envs', '8', '--seed', '42', '--device', 'cuda:0',
                '--vx-min', '0.2', '--vx-max', '0.6', '--vy', '0.0', '--wz', '0.0',
                '--max-iterations', '23000', '--save-interval', '100',
                '--eval-interval', '200', '--worker-chunk-iterations', '200',
                '--wandb-mode', 'online', '--wandb-project', 'p',
                '--wandb-entity', 'e', '--wandb-run-name', 'v04_test',
                '--init-checkpoint', str(self.f.init),
                '--init-checkpoint-sha256', self.f.init_sha]
        if stop:
            argv.append('--stop-after-preflight')
        return o.parser().parse_args(argv + list(extra))

    def close(self, supervisor):
        supervisor.lock.close()
        supervisor.queue_lock.close()

    def test_cli_accepts_only_fixed_512_and_eight(self):
        a = self.args()
        self.assertEqual((a.num_envs, a.eval_num_envs, a.seed, a.device), (512, 8, 42, 'cuda:0'))
        self.assertEqual((a.max_iterations, a.worker_chunk_iterations), (23000, 200))
        for value in ('1024', '258', '8'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    o.parser().parse_args([
                        '--legacy-source-manifest', str(self.f.manifest_path),
                        '--output-dir', str(self.root / ('bad' + value)),
                        '--num-envs', value, '--init-checkpoint', str(self.f.init),
                        '--init-checkpoint-sha256', self.f.init_sha])
        for value in ('4', '1'):
            with self.subTest(eval_value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    o.parser().parse_args([
                        '--legacy-source-manifest', str(self.f.manifest_path),
                        '--output-dir', str(self.root / ('bade' + value)),
                        '--eval-num-envs', value, '--init-checkpoint', str(self.f.init),
                        '--init-checkpoint-sha256', self.f.init_sha])

    def test_cli_has_no_candidate_array_or_old_approval_surface(self):
        flags = {option for action in o.parser()._actions for option in action.option_strings}
        for forbidden in ('--num-envs-candidates', '--runtime-approval',
                          '--runtime-zero-checkpoint', '--allow-num-env-change'):
            self.assertNotIn(forbidden, flags)
        self.assertEqual(o.TRAIN_COUNTS, (512,))
        self.assertEqual(o.EVAL_COUNTS, (8,))
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                o.parser().parse_args([
                    '--legacy-source-manifest', str(self.f.manifest_path),
                    '--output-dir', str(self.root / 'candidates'),
                    '--num-envs-candidates', '1024', '512', '258',
                    '--init-checkpoint', str(self.f.init),
                    '--init-checkpoint-sha256', self.f.init_sha])

    def test_init_and_resume_are_exclusive(self):
        base = ['--legacy-source-manifest', str(self.f.manifest_path),
                '--output-dir', str(self.root / 'mutual')]
        init = ['--init-checkpoint', str(self.f.init), '--init-checkpoint-sha256', self.f.init_sha]
        resume = ['--resume', str(self.root / 'checkpoint.pt'),
                  '--resume-sha256', 'a' * 64, '--resume-run']
        cases = [
            base + init + resume,
            base + init + ['--resume-run'],
            base + init + ['--resume-sha256', 'a' * 64],
            base + ['--resume', str(self.root / 'checkpoint.pt'), '--resume-sha256', 'a' * 64],
            base + resume + ['--init-checkpoint-sha256', self.f.init_sha],
            base,
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                with self.assertRaises(ValueError):
                    o.validate_arguments(o.parser().parse_args(argv), check_files=False)
        self.assertEqual(o.validate_arguments(o.parser().parse_args(base + init), check_files=False),
                         o.validate_arguments(o.parser().parse_args(base + init), check_files=False))

    def test_first_resource_fit_never_downgrades(self):
        with self.assertRaises(ValueError):
            o.first_resource_fit(lambda n: {'status': 'passed'}, counts=(1024, 512, 258))
        tried = []
        with self.assertRaises(RuntimeError):
            o.first_resource_fit(lambda n: (tried.append(n) or {'status': 'resource_failure'}))
        self.assertEqual(tried, [512])

    def test_validation_enforces_real_init_and_manifest_bytes(self):
        a = self.args()
        with patch.object(o, 'INIT_SHA256', self.f.source_sha), \
             patch.object(o, 'approved_init_kind', lambda path, digest: 'v02_formal_3000'):
            o.validate_arguments(a, check_files=True)
            self.f.init.write_bytes(b'changed init')
            with self.assertRaises(ValueError):
                o.validate_arguments(a, check_files=True)
            self.f.init.write_bytes(b'initialization fixture for CLI tests')
            manifest = json.loads(self.f.manifest_path.read_text())
            manifest['schema'] = 'old'
            self.f.manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                o.validate_arguments(a, check_files=True)

    def test_actual_v04_legacy_manifest_verifies_all_snapshots(self):
        path = (Path(__file__).resolve().parents[1] /
                'logs/a2_ideal_stairs_v04/implementation/legacy_snapshot/legacy_manifest.json')
        document = o.verify_legacy_manifest(path)
        self.assertEqual(len(document['source_files']), 43)
        self.assertEqual(document['source_code_version'], '0.2.1')

    def test_stop_after_preflight_validates_and_runs_no_training(self):
        a = self.args(stop=True)
        calls = []
        def forbidden_backend(*args):
            calls.append(('backend', args))
            raise AssertionError('preflight must not create an SDK session')
        def forbidden_child(*args, **kwargs):
            calls.append(('child', args))
            raise AssertionError('preflight must not launch a GPU child')
        with patch.object(o, 'ROOT', self.root), \
             patch.object(o, 'INIT_SHA256', self.f.source_sha), \
             patch.object(o, 'approved_init_kind', lambda path, digest: 'v02_formal_3000'), \
             patch.object(o, 'contract', lambda *a, **k: {'fixture_contract': True}), \
             patch.object(o, 'behavior_hashes', lambda *a, **k: {'fixture.py': 'hash'}):
            supervisor = o.Supervisor(a, child_runner=forbidden_child, backend_factory=forbidden_backend)
            try:
                result = supervisor.run()
            finally:
                self.close(supervisor)
        self.assertEqual(result['status'], 'preflight_validated')
        self.assertEqual(result['formal_updates'], 0)
        self.assertFalse(result['training_started'])
        self.assertEqual(calls, [])
        self.assertEqual(supervisor.state['iteration'], 0)
        self.assertTrue((self.f.out / 'preflight.json').exists())
        self.assertFalse((self.f.out / 'final.json').exists())
        self.assertFalse((self.f.out / 'formal_seed42').exists())
        record = json.loads((self.f.out / 'preflight.json').read_text())
        self.assertEqual(record['num_envs'], 512)
        self.assertEqual(record['eval_num_envs'], 8)
        self.assertEqual(record['legacy_source_files'], 1)

    def test_worker_eval_and_play_commands_point_to_v04_scripts(self):
        a = self.args()
        with self.patched():
            supervisor = o.Supervisor(a)
            try:
                worker = supervisor.worker_command(self.f.out / 'formal_seed42/chunk_0000_n512', 512)
                self.assertTrue(any(Path(part).name == 'train_a2_ideal_stairs_v04.py' for part in worker))
                self.assertFalse(any(Path(part).name == 'train_a2_ideal_stairs_v03.py' for part in worker))
                self.assertEqual(worker[worker.index('--num-envs') + 1], '512')
                self.assertEqual(worker[worker.index('--recipe-version') + 1], RECIPE_VERSION)
                self.assertEqual(worker[worker.index('--max-iterations') + 1], '23000')
                self.assertEqual(worker[worker.index('--worker-chunk-iterations') + 1], '200')
                self.assertIn('--legacy-source-manifest', worker)
                self.assertIn('--init-checkpoint', worker)
                resumed = supervisor.worker_command(self.f.out / 'chunk', 512,
                                                    checkpoint=str(self.f.init), digest=self.f.init_sha)
                self.assertIn('--resume', resumed)
                self.assertIn('--resume-sha256', resumed)
                evaluation = supervisor.evaluation_command(
                    self.f.init, self.f.init_sha, 'S1', 'validation',
                    self.f.out / 'eval', self.f.out / 'cases.json')
                self.assertTrue(any(Path(part).name == 'evaluate_a2_ideal_stairs_v04.py' for part in evaluation))
                self.assertEqual(evaluation[evaluation.index('--num-envs') + 1], '8')
                self.assertEqual(evaluation[evaluation.index('--episodes') + 1], '20')
                replay = supervisor.replay_command(
                    self.f.init, self.f.init_sha, 'S5', self.f.out / 'replay', .4)
                self.assertTrue(any(Path(part).name == 'play_a2_ideal_stairs_v04.py' for part in replay))
                self.assertEqual(replay[replay.index('--num-envs') + 1], '1')
                self.assertEqual(replay[replay.index('--vx') + 1], '0.4')
            finally:
                self.close(supervisor)

    def test_single_supervisor_queue_and_lock_surface(self):
        with self.patched():
            lease = o._new_gpu_queue_lease()
            try:
                with self.assertRaises(BlockingIOError):
                    o._new_gpu_queue_lease()
            finally:
                lease.close()
            first = o.Supervisor(self.args())
            try:
                with self.assertRaises(BlockingIOError):
                    o.Supervisor(self.args())
            finally:
                self.close(first)

    def test_preflight_is_idempotent_and_does_not_overwrite_its_file(self):
        a = self.args(stop=True)
        with self.patched():
            supervisor = o.Supervisor(a)
            try:
                first = supervisor.preflight()
                raw = (self.f.out / 'preflight.json').read_bytes()
                second = supervisor.preflight()
                self.assertEqual(first, second)
                self.assertEqual(raw, (self.f.out / 'preflight.json').read_bytes())
            finally:
                self.close(supervisor)


if __name__ == '__main__':
    unittest.main()


class InitializationIdentityTests(unittest.TestCase):
    def test_arbitrary_or_zero_shape_init_is_not_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'arbitrary.pt'
            path.write_bytes(b'not the approved source and not a v04 zero container')
            with self.assertRaises(ValueError):
                o.approved_init_kind(path, digest_bytes(path.read_bytes()))
