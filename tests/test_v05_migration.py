"""CPU-only migration evidence for the approved ETH -> V0.5 Actor transfer.

The approved source is the frozen ETH locomotion checkpoint.  Only its Actor MLP
and the ``FrozenETHNormalizer`` buffers are copied; the map/history branches and
the Critic are freshly seeded, the fusion layer is zeroed and all twelve action
std values are written to 0.15.  The migration-zero container is an
initialization artifact, never a schema-6 resume boundary.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)

import copy  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import (  # noqa: E402
    IdealHeightmapActorCritic)
from src.tasks.velocity.v05 import model as v05_model  # noqa: E402
from src.tasks.velocity.v05 import spec  # noqa: E402

sys.path.insert(0, str(ROOT / 'scripts'))
import check_a2_ideal_stairs_v05_migration as migration_script  # noqa: E402

V05ActorCritic = v05_model.V05ActorCritic
FULL_RESUME_FIELDS = {'contract', 'ppo', 'model', 'optimizer', 'learning_state',
                      'transfer_metadata', 'curriculum', 'metadata', 'num_envs', 'rng',
                      'resume_mode', 'reward_state', 'execution', 'ledger', 'ledger_anchor'}


class ApprovedMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=True)
        cls.migrated = V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_approved_sha_copies_every_actor_tensor_and_normalizer(self):
        self.assertEqual(spec.sha256(spec.INIT_PATH), spec.INIT_SHA256)
        source_state = self.source['actor_state_dict']
        self.assertEqual(len(source_state), 13)
        state = self.migrated.state_dict()
        copied = [key for key in source_state
                  if key.startswith('mlp.') or key.startswith('obs_normalizer.')]
        self.assertEqual(len(copied), 12)
        for key in copied:
            target = v05_model._source_key_to_model_key(key)
            # Values must be bit-identical after the copy.  ``torch.equal`` ignores
            # dtype, so the one documented dtype coercion is checked separately.
            self.assertTrue(torch.equal(source_state[key], state[target]), target)
            self.assertEqual(source_state[key].shape, state[target].shape, target)
            if key == 'obs_normalizer.count':
                # ETH stores the counter as int64; the frozen base network owns a
                # float32 counter buffer and copy_ casts the value.  The network
                # definition is not allowed to change (SDS 1), so this one cast is
                # value-exact and recorded rather than silently ignored.
                self.assertEqual(source_state[key].dtype, torch.int64)
                self.assertEqual(state[target].dtype, torch.float32)
                self.assertEqual(float(source_state[key].item()), float(state[target].item()))
            else:
                self.assertEqual(source_state[key].dtype, state[target].dtype, target)
        targets = [v05_model._source_key_to_model_key(key) for key in copied]
        for branch in ('actor.mlp.', 'actor.normalizer.'):
            self.assertTrue(any(key.startswith(branch) for key in targets), branch)
        # The normalizer is a frozen buffer-only module.
        self.assertEqual(list(self.migrated.actor.normalizer.parameters()), [])
        self.assertTrue(torch.equal(self.migrated.actor.normalizer._std,
                                    source_state['obs_normalizer._std']))
        self.assertEqual(self.migrated.actor.normalizer.count.item(),
                         source_state['obs_normalizer.count'].item())

    def test_std_is_overwritten_not_inherited(self):
        source_std = self.source['actor_state_dict']['std']
        self.assertFalse(torch.equal(source_std, torch.full_like(source_std, .15)))
        self.assertTrue(torch.equal(self.migrated.std.detach().cpu(),
                                    torch.full((12,), .15)))
        self.assertEqual(self.migrated.transfer_metadata['std_before'],
                         source_std.tolist())
        self.assertEqual(self.migrated.transfer_metadata['std_after'],
                         self.migrated.std.detach().tolist())
        self.assertEqual(self.migrated.transfer_metadata['std_source'],
                         'overwritten_to_0.15_not_inherited')

    def test_fresh_map_history_fusion_and_critic_equal_seed42_construction(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            fresh = IdealHeightmapActorCritic()
        state = self.migrated.state_dict()
        reference = fresh.state_dict()
        self.assertEqual(len(state), 51)
        fresh_keys = [key for key in reference if key != 'std' and not key.startswith('actor.')]
        self.assertEqual(len(fresh_keys), 24)
        for key in fresh_keys:
            self.assertTrue(torch.equal(reference[key], state[key]), key)
        self.assertEqual(self.migrated.critic_normalizer.count.item(), 0)
        self.assertEqual(float(self.migrated.critic_normalizer.mean.abs().sum()), 0.)
        for key in v05_model.FUSION_KEYS:
            self.assertEqual(float(state[key].detach().abs().sum()), 0., key)
        self.assertEqual(self.migrated.transfer_metadata['critic'], 'fresh_seeded')
        self.assertEqual(self.migrated.transfer_metadata['critic_statistics'], 'fresh')
        self.assertEqual(self.migrated.transfer_metadata['optimizer'], 'fresh_adam')
        self.assertFalse(self.migrated.transfer_metadata['resume'])

    def test_actor_mean_parity_on_the_128_cpu_audit_observations(self):
        reference = v05_model.load_eth_actor_reference(spec.INIT_PATH)
        observations = v05_model.synthetic_observations(128, seed=42)
        result = v05_model.mean_equivalence(self.migrated, reference, observations)
        self.assertEqual(result['observations'], 128)
        self.assertTrue(result['finite'])
        self.assertLessEqual(result['max_abs_error'], 1e-6)
        self.assertTrue(result['passed'])
        self.assertEqual(result['reference'], 'eth_actor_mlp_and_frozen_normalizer')

    def test_zero_fusion_output_ignores_map_history_and_privileged(self):
        observations = v05_model.synthetic_observations(128, seed=42)
        result = v05_model.map_history_invariance(self.migrated, observations)
        self.assertTrue(result['passed'])
        self.assertTrue(result['independent_of_map_history_privileged'])
        self.assertTrue(result['independent_of_rolled_map_history'])
        self.assertEqual(result['fusion_weight_abs_sum'], 0.)
        self.assertEqual(result['fusion_bias_abs_sum'], 0.)
        # Even a wildly different history/map/privileged input changes nothing.
        altered = dict(observations)
        altered['heightmap'] = torch.full_like(observations['heightmap'], 3.)
        altered['history'] = torch.full_like(observations['history'], -3.)
        altered['privileged'] = torch.full_like(observations['privileged'], 7.)
        self.migrated.eval()
        try:
            with torch.no_grad():
                self.assertTrue(torch.equal(self.migrated(observations), self.migrated(altered)))
        finally:
            self.migrated.train()

    def test_migration_is_deterministic_and_independent_of_ambient_rng(self):
        torch.manual_seed(123456)
        first = V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        torch.manual_seed(987654)
        second = V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        self.assertEqual(first.transfer_metadata, second.transfer_metadata)
        other_seed = V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 43)
        self.assertFalse(torch.equal(first.critic[0].weight, other_seed.critic[0].weight))
        # The copied Actor part is seed independent.
        for key in list(first.state_dict()):
            if key.startswith('actor.mlp.') or key.startswith('actor.normalizer.'):
                self.assertTrue(torch.equal(first.state_dict()[key],
                                            other_seed.state_dict()[key]), key)

    def test_transfer_metadata_records_source_and_v05_recipe(self):
        meta = self.migrated.transfer_metadata
        self.assertEqual(meta['kind'], 'eth_actor_mlp_and_frozen_normalizer_only')
        self.assertEqual(meta['sha256'], spec.INIT_SHA256)
        self.assertEqual(meta['source_iteration'], 2000)
        self.assertEqual(meta['recipe'], '0.5-r1')
        self.assertEqual(meta['reward_version'], spec.RECIPE_VERSION)
        self.assertEqual(meta['reward_config_sha256'], spec.reward_config_sha256())
        self.assertEqual(meta['seed'], 42)
        self.assertEqual(meta['scope'], 'initialization')
        self.assertEqual(meta['std_range'], [.10, .30])
        self.assertEqual(meta['actor_statistics'], 'copied_complete_and_frozen')
        self.assertEqual(meta['normalization'], 'frozen_ETH_normalizer_buffers')
        self.assertEqual(meta['fusion'], 'zeros_after_migration')
        self.assertTrue(Path(meta['source_path']).is_absolute())
        self.assertEqual(meta['source_path'], str(Path(spec.INIT_PATH).resolve()))
        self.assertEqual(sorted(meta['source_actor_keys']), sorted(v05_model.ETH_ACTOR_KEYS))
        self.assertEqual(meta['source_normalizer_keys'],
                         sorted(v05_model.NORMALIZER_SOURCE_KEYS))

    def test_per_tensor_audit_rows_cover_every_state_key(self):
        source_state = self.source['actor_state_dict']
        rows = v05_model.tensor_audit(source_state, self.migrated.state_dict())
        self.assertEqual(len(rows), 51)
        self.assertEqual([row['key'] for row in rows],
                         sorted(self.migrated.state_dict()))
        for row in rows:
            self.assertEqual(set(row), {'key', 'source_key', 'group', 'shape', 'dtype',
                                        'numel', 'finite', 'tensor_sha256', 'equal_to_source'})
            self.assertTrue(row['finite'])
            self.assertRegex(row['tensor_sha256'], r'^[0-9a-f]{64}$')
            if row['source_key'] is not None and row['key'] != 'std':
                self.assertTrue(row['equal_to_source'], row['key'])
                self.assertEqual(row['group'], 'actor_copied')
        self.assertEqual(sum(1 for row in rows if row['group'] == 'actor_copied'), 12)
        self.assertEqual(sum(1 for row in rows if row['group'] == 'std_overwritten'), 1)
        # Map encoder (8) + history encoder (4) + fusion (2) stay freshly seeded.
        self.assertEqual(sum(1 for row in rows if row['group'] == 'actor_fresh'), 14)
        self.assertEqual(sum(1 for row in rows if row['group'] == 'fresh_seeded'), 24)

    def test_cpu_audit_never_initializes_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


class SourceRejectionTests(unittest.TestCase):
    def test_wrong_sha_is_rejected_before_any_checkpoint_is_read(self):
        with patch.object(v05_model.torch, 'load') as loader:
            for expected in ('0' * 64, 'f' * 64, spec.INIT_SHA256.upper(), None, ''):
                with self.assertRaises(ValueError):
                    V05ActorCritic.from_eth(spec.INIT_PATH, expected, 42)
            loader.assert_not_called()
        with patch.object(v05_model, 'sha256', return_value='0' * 64):
            with self.assertRaises(ValueError):
                V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        for seed in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError, msg=repr(seed)):
                V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, seed)

    def test_corrupt_source_state_is_rejected(self):
        source = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=True)
        mutations = {
            'missing_mlp': lambda state: state.pop('mlp.2.bias'),
            'missing_normalizer': lambda state: state.pop('obs_normalizer._mean'),
            'missing_std': lambda state: state.pop('std'),
            'extra': lambda state: state.update({'actor.bogus': torch.zeros(1)}),
            'shape': lambda state: state.update({'mlp.0.weight': torch.zeros(1, 45)}),
            'dtype': lambda state: state.update({'mlp.4.weight': state['mlp.4.weight'].double()}),
            'nan': lambda state: state['mlp.0.weight'].view(-1).__setitem__(0, float('nan')),
            'inf': lambda state: state['std'].view(-1).__setitem__(1, float('inf')),
            'zero_normalizer_std': lambda state: state['obs_normalizer._std'].fill_(0.),
            'negative_variance': lambda state: state['obs_normalizer._var'].fill_(-1.),
            'negative_std': lambda state: state['std'].fill_(-1.),
            'bad_count': lambda state: state['obs_normalizer.count'].fill_(-1),
        }
        for name, apply in mutations.items():
            with self.subTest(mutation=name):
                bad = copy.deepcopy(source)
                apply(bad['actor_state_dict'])
                with patch.object(v05_model, 'sha256', return_value=spec.INIT_SHA256), \
                     patch.object(v05_model.torch, 'load', return_value=bad):
                    with self.assertRaises(ValueError):
                        V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)

    def test_source_without_actor_state_dict_is_rejected(self):
        bad = {'iter': 1}
        with patch.object(v05_model, 'sha256', return_value=spec.INIT_SHA256), \
             patch.object(v05_model.torch, 'load', return_value=bad):
            with self.assertRaises(ValueError):
                V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)


class ZeroInitializationTests(unittest.TestCase):
    def build(self, model):
        return v05_model.initialization_state(
            model, seed=42, source_path=spec.INIT_PATH, source_sha256=spec.INIT_SHA256,
            audit_schema=migration_script.AUDIT_SCHEMA)

    def test_initialization_container_is_not_a_resume_checkpoint(self):
        model = V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        state = self.build(model)
        self.assertEqual(set(state), set(v05_model.INITIALIZATION_FIELDS))
        self.assertNotEqual(set(state), FULL_RESUME_FIELDS)
        for field in ('rng', 'ledger', 'ledger_anchor', 'reward_state', 'execution',
                      'curriculum', 'metadata', 'num_envs', 'ppo'):
            self.assertNotIn(field, state)
        self.assertEqual(state['scope'], 'initialization')
        self.assertFalse(state['resume'])
        self.assertFalse(state['physical_trajectory_continuous'])
        self.assertFalse(state['learning_state']['optimizer_initialized'])
        self.assertEqual(state['optimizer'], dict(state=dict(), param_groups=[]))
        self.assertIsNone(state['source_model_sha256'])
        self.assertRegex(state['migrated_model_sha256'], r'^[0-9a-f]{64}$')
        self.assertEqual(state['transfer_metadata']['scope'], 'initialization')
        v05_model.validate_initialization(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'migration_zero.pt'
            torch.save(state, path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            v05_model.validate_initialization(loaded)

    def test_bad_initialization_scope_or_adam_is_rejected(self):
        model = V05ActorCritic.from_eth(spec.INIT_PATH, spec.INIT_SHA256, 42)
        base = self.build(model)
        for mutation in ('scope', 'resume', 'adam', 'counters', 'source_sha', 'extra',
                         'metric_sha'):
            state = copy.deepcopy(base)
            if mutation == 'scope':
                state['scope'] = 'formal_resume'
            if mutation == 'resume':
                state['transfer_metadata'] = dict(state['transfer_metadata'], resume=True)
            if mutation == 'adam':
                state['optimizer'] = dict(state={'0': {}}, param_groups=[])
            if mutation == 'counters':
                state['learning_state'] = dict(optimizer_initialized=True,
                                               completed_ppo_updates=1, optimizer_steps=20)
            if mutation == 'source_sha':
                state['source_checkpoint_sha256'] = '0' * 64
            if mutation == 'extra':
                state['rng'] = {}
            if mutation == 'metric_sha':
                state['migrated_model_sha256'] = 'not-a-digest'
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    v05_model.validate_initialization(state)


class ScriptTests(unittest.TestCase):
    def test_audit_script_writes_every_evidence_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'migration_seed42'
            code = migration_script.main([
                '--mode', 'audit', '--device', 'cpu', '--seed', '42',
                '--project-root', str(ROOT),
                '--source-checkpoint', spec.INIT_PATH,
                '--source-checkpoint-sha256', spec.INIT_SHA256,
                '--output-dir', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(set(path.name for path in output.iterdir()), {
                'audit_verification.json', 'initialization.json', 'mean_equivalence.json',
                'migration_audit.json', 'migration_zero.pt', 'source_inventory.json',
                'tensor_audit.json'})
            audit = json.loads((output / 'migration_audit.json').read_text())
            self.assertEqual(audit['status'], 'passed')
            self.assertEqual(audit['schema'], migration_script.AUDIT_SCHEMA)
            self.assertEqual(audit['seed'], 42)
            self.assertEqual(audit['device'], 'cpu')
            self.assertFalse(audit['cuda_initialized'])
            self.assertEqual(audit['scope'], 'initialization')
            self.assertFalse(audit['resume'])
            self.assertTrue(audit['actor_all_equal'])
            self.assertTrue(audit['fresh_critic_all_equal'])
            self.assertTrue(audit['fresh_adam_empty'])
            self.assertLessEqual(audit['mean_equivalence']['max_abs_error'], 1e-6)
            self.assertTrue(audit['map_history_invariance']['passed'])
            tensor_audit = json.loads((output / 'tensor_audit.json').read_text())
            self.assertEqual(tensor_audit['tensor_count'], 51)
            self.assertTrue(tensor_audit['all_copied_actor_equal'])
            self.assertEqual(tensor_audit['std']['expected'], .15)
            self.assertTrue(all(abs(value - .15) <= 1e-7 for value in tensor_audit['std']['after']))
            initialization = json.loads((output / 'initialization.json').read_text())
            self.assertEqual(initialization['scope'], 'initialization')
            self.assertFalse(initialization['resume'])
            self.assertIn('not a complete schema-6 resume checkpoint', initialization['note'])
            zero = torch.load(output / 'migration_zero.pt', map_location='cpu',
                              weights_only=False)
            self.assertEqual(zero['scope'], 'initialization')
            self.assertEqual(zero['optimizer'], dict(state=dict(), param_groups=[]))
            v05_model.validate_initialization(zero)
            verification = json.loads((output / 'audit_verification.json').read_text())
            self.assertEqual(verification['zero_sha256'],
                             spec.sha256(output / 'migration_zero.pt'))

    @staticmethod
    def argv(output, **overrides):
        values = dict(mode='audit', device='cpu', seed='42', project_root=str(ROOT),
                      source_checkpoint=spec.INIT_PATH,
                      source_checkpoint_sha256=spec.INIT_SHA256, output_dir=str(output))
        values.update(overrides)
        return ['--mode', values['mode'], '--device', values['device'],
                '--seed', values['seed'], '--project-root', values['project_root'],
                '--source-checkpoint', values['source_checkpoint'],
                '--source-checkpoint-sha256', values['source_checkpoint_sha256'],
                '--output-dir', values['output_dir']]

    def test_audit_script_refuses_to_overwrite_and_rejects_bad_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'exists'
            existing.mkdir()
            code = migration_script.main(self.argv(existing))
            self.assertEqual(code, 2)
            self.assertEqual(list(existing.iterdir()), [])
            # Runtime argument violations return the migration error code.
            for overrides in (dict(seed='0'), dict(seed='-3'),
                              dict(source_checkpoint_sha256='0' * 64),
                              dict(device='cuda:0'), dict(mode='freeze')):
                output = Path(directory) / ('bad_' + next(iter(overrides)))
                with self.subTest(overrides=overrides):
                    if overrides.get('device', 'cpu') != 'cpu' or overrides.get('mode') == 'freeze':
                        with self.assertRaises(SystemExit):
                            migration_script.main(self.argv(output, **overrides))
                    else:
                        self.assertEqual(migration_script.main(self.argv(output, **overrides)), 2)
                    self.assertFalse(output.exists())
            for overrides in (dict(source_checkpoint='/missing/model.pt'),
                              dict(project_root=directory),
                              dict(source_checkpoint_sha256='abc')):
                output = Path(directory) / ('bad_input_' + next(iter(overrides)))
                with self.subTest(overrides=overrides):
                    self.assertEqual(migration_script.main(self.argv(output, **overrides)), 2)
                    self.assertFalse(output.exists())
            with self.assertRaises(SystemExit):
                migration_script.parse_args(['--mode', 'audit'])


if __name__ == '__main__':
    unittest.main()
