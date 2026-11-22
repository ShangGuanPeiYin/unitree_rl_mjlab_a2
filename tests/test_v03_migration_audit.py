"""CPU synthetic helper tests only; these never satisfy real-observation V7."""
import ast
import copy
import importlib.util
import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_a2_ideal_stairs_v03_migration.py'
audit = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location('migration_audit', SCRIPT)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)


def observations(n=1):
    return {k: torch.zeros(n, *shape) for k, shape in
            dict(proprio=(45,), history=(5, 45), heightmap=(3, 48, 32), privileged=(35,)).items()}


class ToyModel(nn.Module):
    def __init__(self, offset=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(offset))
        self.register_buffer('statistics', torch.zeros(2))
    def forward(self, obs):
        return obs['proprio'][:, :12] + self.weight


class PublicAPI(unittest.TestCase):
    def test_audit_entry_exists_before_gpu_execution(self):
        self.assertIsNotNone(audit, 'V3 migration audit script is not implemented')


@unittest.skipIf(audit is None, 'Entry absent during RED')
class AuditHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def capture_invalid_frame(self, obs, context, stage='S1', index=0):
        # Execute the actual main-loop prefix, stopping before Actor inference.
        # This proves production ordering without constructing CUDA/env/NVML.
        tree = ast.parse(SCRIPT.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_run_migration')
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'index')
        statements = []
        for node in loop.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'compare_actor_means':
                break
            statements.append(node)
        scope = dict(audit.__dict__, context=context, obs=obs, stage=stage, index=index,
                     env=SimpleNamespace(command=torch.tensor([[.4, 0., 0.]]), episode_length_buf=torch.tensor([0])),
                     episode=torch.tensor([3]), monitor=SimpleNamespace(poll=lambda: None))
        with self.assertRaises(ValueError):
            exec(compile(ast.Module(body=statements, type_ignores=[]), str(SCRIPT), 'exec'), scope)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure_numeric.pt'
            torch.save(audit.cpu_copy(context), path)
            return torch.load(path, map_location='cpu', weights_only=False)

    def test_first_invalid_frame_preserves_raw_nan_shape_and_dtype_in_failure_evidence(self):
        for mutation in ('nan', 'shape', 'dtype'):
            with self.subTest(mutation=mutation):
                obs = observations()
                if mutation == 'nan': obs['privileged'][0, 0] = float('nan')
                if mutation == 'shape': obs['history'] = torch.zeros(1, 4, 45)
                if mutation == 'dtype': obs['proprio'] = obs['proprio'].double()
                saved = self.capture_invalid_frame(obs, dict(physics_samples_observed=0, physics_tail_unknown=False))
                self.assertEqual(saved.get('stage'), 'S1')
                self.assertEqual(saved.get('stage_observation_index'), 0)
                self.assertIn('pre_obs', saved)
                if mutation == 'nan': self.assertTrue(torch.isnan(saved['pre_obs']['privileged'][0, 0]))
                if mutation == 'shape': self.assertEqual(saved['pre_obs']['history'].shape, (1, 4, 45))
                if mutation == 'dtype': self.assertEqual(saved['pre_obs']['proprio'].dtype, torch.float64)

    def test_cross_stage_invalid_frame_replaces_identity_and_clears_previous_derived_evidence(self):
        bad = observations()
        bad['privileged'][0, 0] = float('nan')
        context = dict(stage='S1', stage_observation_index=199, pre_obs=observations(),
                       pre_episode_id=torch.tensor([99]), pre_episode_step=torch.tensor([199]),
                       source_mean=torch.ones(1, 12), migrated_mean=torch.ones(1, 12),
                       post_info={'stale': True}, reward=torch.ones(1),
                       physics_samples_observed=200, physics_tail_unknown=False)
        saved = self.capture_invalid_frame(bad, context, stage='S2', index=0)
        self.assertEqual(saved['stage'], 'S2')
        self.assertEqual(saved['stage_observation_index'], 0)
        self.assertEqual(saved['pre_episode_id'].item(), 3)
        self.assertEqual(saved['pre_episode_step'].item(), 0)
        self.assertEqual(saved['pre_command'][0, 0].item(), torch.tensor(.4).item())
        self.assertTrue(torch.isnan(saved['pre_obs']['privileged'][0, 0]))
        self.assertEqual(saved['physics_samples_observed'], 200)
        self.assertFalse(saved['physics_tail_unknown'])
        for key in ('source_mean', 'migrated_mean', 'post_info', 'reward'):
            self.assertNotIn(key, saved)

    def test_cli_requires_full_fixed_protocol_and_new_output(self):
        args = audit.parse_args(['--output-dir', '/tmp/a2-migration-not-created'])
        self.assertEqual(args.stages, ['S1', 'S2', 'S3', 'S4', 'S5'])
        self.assertEqual(args.observations_per_stage, 200)
        self.assertEqual(args.num_envs, 1)
        for flags in (['--seed', '43'], ['--num-envs', '2'], ['--observations-per-stage', '199'],
                      ['--stages', 'S1', 'S2'], ['--stages', 'S1', 'S2', 'S3', 'S4', 'S4'],
                      ['--vx-min', '.1'], ['--init-checkpoint-sha256', '0' * 64], ['--device', 'cpu']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                audit.parse_args(['--output-dir', '/tmp/a2-migration-not-created'] + flags)

    def test_real_observation_contract_rejects_bad_shapes_dtypes_and_nonfinite(self):
        audit.validate_observations(observations(), 1)
        for mutation in ('shape', 'dtype', 'nan', 'missing'):
            bad = observations()
            if mutation == 'shape': bad['history'] = torch.zeros(1, 4, 45)
            if mutation == 'dtype': bad['proprio'] = bad['proprio'].double()
            if mutation == 'nan': bad['privileged'][0, 0] = float('nan')
            if mutation == 'missing': del bad['heightmap']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_observations(bad, 1)

    def test_deterministic_comparison_and_state_mutation_detection(self):
        source, model = ToyModel(), ToyModel()
        before = audit.snapshot_state(model)
        left, right, error = audit.compare_actor_means(source, model, observations(), 1)
        self.assertEqual(error, 0.)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(audit.require_unchanged(model, before))
        model.statistics[0] = 1
        with self.assertRaises(ValueError): audit.require_unchanged(model, before)
        with self.assertRaises(ValueError):
            audit.compare_actor_means(source, ToyModel(.01), observations(), 1)
        bad = ToyModel(float('nan'))
        with self.assertRaises(ValueError): audit.compare_actor_means(source, bad, observations(), 1)

    def test_streamed_npz_roundtrip_and_identity_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = audit.BatchWriter(Path(directory), dict(kind='actor_migration_check', stage='S1', seed=42), batch_size=2)
            rows = [dict(obs_proprio=np.full((1, 45), i, dtype=np.float32),
                         pre_episode_id=np.array([i // 2], dtype=np.int64),
                         pre_episode_step=np.array([i % 2], dtype=np.int64),
                         post_step=np.array([i % 2 + 1], dtype=np.int64)) for i in range(3)]
            for row in rows: writer.append(row)
            self.assertEqual(len(writer.entries), 1)
            writer.flush()
            self.assertEqual([e['observations'] for e in writer.entries], [2, 1])
            loaded = []
            for entry in writer.entries:
                file = Path(directory) / entry['path']
                self.assertEqual(audit.sha256(file), entry['sha256'])
                with np.load(file, allow_pickle=False) as archive:
                    self.assertEqual(json.loads(str(archive['identity']))['kind'], 'actor_migration_check')
                    self.assertEqual(archive['obs_proprio'].dtype, np.float32)
                    self.assertEqual(archive['post_step'].dtype, np.int64)
                    loaded.extend(archive['obs_proprio'][:, 0].tolist())
                    np.testing.assert_array_equal(archive['post_step'], archive['pre_episode_step'] + 1)
            self.assertEqual(loaded, [0., 1., 2.])
            self.assertEqual(writer.total_written, 3)
            manifest = json.loads((Path(directory) / 'index.json').read_text())
            self.assertEqual(manifest['batches'], writer.entries)

    def test_transition_rejects_fractional_episode_identity(self):
        obs = observations()
        obs['proprio'][0, 6] = .4
        info = dict(segment_samples=dict(step=torch.tensor([8]), executed_c=torch.tensor([.4]),
                    num_steps=torch.tensor([4]), tile_id=torch.tensor([5]), slot_class=torch.tensor([1]),
                    body_vx=torch.tensor([.3]), local_x=torch.tensor([.1])))
        with self.assertRaises(ValueError):
            audit.pack_transition(obs, torch.zeros(1, 12), torch.zeros(1, 12),
                pre_command=torch.tensor([[.4, 0., 0.]]), pre_episode_id=torch.tensor([2.5]),
                pre_episode_step=torch.tensor([7.]), info=info,
                terminated=torch.tensor([False]), truncated=torch.tensor([False]), stage='S1')

    def test_writer_rejects_nonfinite_or_shape_drift_before_adding_bad_record(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = audit.BatchWriter(Path(directory), dict(kind='actor_migration_check'), batch_size=2)
            writer.append(dict(x=np.zeros((1, 2), np.float32)))
            for bad in (dict(x=np.full((1, 2), np.nan, np.float32)),
                        dict(x=np.zeros((1, 3), np.float32)), dict(y=np.zeros((1, 2), np.float32))):
                with self.assertRaises(ValueError): writer.append(bad)
            writer.flush()
            self.assertEqual(writer.total_written, 1)

    def test_cleanup_continues_to_monitor_when_environment_close_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Resource:
                def __init__(self, name, fails=False): self.name, self.fails = name, fails
                def close(self):
                    (root / self.name).write_text('closed')
                    if self.fails: raise RuntimeError('close failed')
            errors = audit.close_resources(Resource('env', True), Resource('monitor'))
            self.assertTrue((root / 'env').exists())
            self.assertTrue((root / 'monitor').exists())
            self.assertEqual([e['resource'] for e in errors], ['env'])

    def test_pack_transition_preserves_pre_action_episode_and_post_control_identity(self):
        obs = observations()
        obs['proprio'][0, 6] = .4
        info = dict(segment_samples=dict(step=torch.tensor([8]), executed_c=torch.tensor([.4]),
                    num_steps=torch.tensor([4]), tile_id=torch.tensor([5]), slot_class=torch.tensor([1]),
                    body_vx=torch.tensor([.3]), local_x=torch.tensor([.1])))
        row = audit.pack_transition(obs, torch.zeros(1, 12), torch.zeros(1, 12),
              pre_command=torch.tensor([[.4, 0., 0.]]), pre_episode_id=torch.tensor([2]),
              pre_episode_step=torch.tensor([7]), info=info,
              terminated=torch.tensor([True]), truncated=torch.tensor([False]), stage='S1')
        self.assertEqual(row['pre_episode_id'].item(), 2)
        self.assertEqual(row['pre_episode_step'].item(), 7)
        self.assertEqual(row['post_step'].item(), 8)
        self.assertTrue(row['terminated'].item())
        info['segment_samples']['step'] = torch.tensor([0])
        with self.assertRaises(ValueError):
            audit.pack_transition(obs, torch.zeros(1, 12), torch.zeros(1, 12),
                pre_command=torch.tensor([[.4, 0., 0.]]), pre_episode_id=torch.tensor([2]),
                pre_episode_step=torch.tensor([7]), info=info,
                terminated=torch.tensor([True]), truncated=torch.tensor([False]), stage='S1')


class MigrationQueueEntryTests(unittest.TestCase):
    """Actual main + actual shared flock, stopped before NVML/CUDA allocation."""
    def test_busy_shared_queue_rejects_actual_migration_main_before_resource_constructor(self):
        from src.tasks.velocity.v03 import orchestration as orch
        from src.tasks.velocity.rl import ideal_stairs_resources as resources
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'resource_constructor_reached'
            def cpu_stop(*args, **kwargs):
                marker.write_text('runtime reached')
                raise RuntimeError('CPU_QUEUE_SENTINEL_STOP_BEFORE_GPU')
            with patch.object(orch, 'ROOT', root), patch.dict(os.environ, {}, clear=False):
                os.environ.pop('V03_GPU_QUEUE_LOCK_FD', None)
                lease = orch._new_gpu_queue_lease()
                try:
                    with patch.object(resources, 'ResourceMonitor', side_effect=cpu_stop), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(BlockingIOError):
                            audit.main(['--output-dir', str(root / 'blocked_migration')])
                    self.assertFalse(marker.exists(), 'Busy entry reached resource/CUDA allocation boundary')
                    self.assertFalse(torch.cuda.is_initialized())
                finally:
                    lease.close()

    def test_legitimate_inherited_queue_fd_reaches_actual_main_once_and_becomes_close_on_exec(self):
        from src.tasks.velocity.v03 import orchestration as orch
        from src.tasks.velocity.rl import ideal_stairs_resources as resources
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'inherited_runtime_boundary.json'
            with patch.object(orch, 'ROOT', root), patch.dict(os.environ, {}, clear=False):
                lease = orch._new_gpu_queue_lease()
                fd = lease.fileno()
                os.set_inheritable(fd, True)
                os.environ['V03_GPU_QUEUE_LOCK_FD'] = str(fd)
                def cpu_stop(*args, **kwargs):
                    marker.write_text(json.dumps(dict(fd_env_consumed='V03_GPU_QUEUE_LOCK_FD' not in os.environ,
                                                      fd_inheritable=os.get_inheritable(fd))))
                    raise RuntimeError('CPU_QUEUE_SENTINEL_STOP_BEFORE_GPU')
                try:
                    with patch.object(resources, 'ResourceMonitor', side_effect=cpu_stop), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(audit.main(['--output-dir', str(root / 'inherited_migration')]), 1)
                    self.assertTrue(marker.exists(), 'A valid same-open-description FD was rejected/self-locked')
                    observed = json.loads(marker.read_text())
                    self.assertTrue(observed['fd_env_consumed'])
                    self.assertFalse(observed['fd_inheritable'])
                    self.assertNotIn('V03_GPU_QUEUE_LOCK_FD', os.environ)
                    # The borrower must not close/unlock the supervisor's description.
                    with self.assertRaises(BlockingIOError):
                        with orch.gpu_queue_guard('cuda:0'):
                            self.fail('Borrower released the owner queue lease')
                    self.assertFalse(torch.cuda.is_initialized())
                finally:
                    lease.close()
