"""CPU-only T4 migration evidence for the approved 0.2.1 -> 0.4 Actor transfer.

The approved source is the v0.2 formal terminal checkpoint (schema 3, 3000
complete updates / 60000 Adam steps).  Only Actor parameters/buffers and ``std``
are copied; the Critic, its statistics and Adam are freshly built under seed 42.
A migration zero-update checkpoint is an initialization artifact, never a full
schema-5 resume boundary.
"""
import sys
sys.path.insert(0, 'scripts')
import a2_v04_bootstrap
a2_v04_bootstrap.activate('.')

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import IdealHeightmapActorCritic
from src.tasks.velocity.v02_formal.spec import (TASK as SOURCE_TASK,
                                                PROTOCOL_ID as SOURCE_PROTOCOL_ID,
                                                contract as source_contract)
from src.tasks.velocity.v04 import spec
from src.tasks.velocity.v04.model import V04ActorCritic
from src.tasks.velocity.v04.recovery import preload_checkpoint
from src.tasks.velocity.v04.runner import REWARD_STATE


def _closure_ready():
    try:
        spec.contract()
        return True, ''
    except (FileNotFoundError, SyntaxError, UnicodeDecodeError) as error:
        # A concurrently incomplete/undeveloped dependency closure cannot be
        # contracted; report the exact cause instead of pretending success.
        return False, f'v0.4 source closure unavailable: {type(error).__name__}: {error}'


CONTRACT_READY, CONTRACT_REASON = _closure_ready()

FULL_RESUME_FIELDS = {'contract', 'ppo', 'model', 'optimizer', 'learning_state',
                      'transfer_metadata', 'curriculum', 'metadata', 'num_envs', 'rng',
                      'resume_mode', 'reward_state', 'execution', 'ledger', 'ledger_anchor'}


class ApprovedMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=False)
        cls.model = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)

    def test_approved_sha_copies_every_actor_tensor_std_and_statistic(self):
        self.assertEqual(spec.sha256(spec.INIT_PATH), spec.INIT_SHA256)
        self.assertEqual(self.source['contract'], source_contract())
        self.assertEqual(self.source['metadata']['checkpoint_schema'], 3)
        self.assertEqual(self.source['metadata']['seed'], 42)
        actor_keys = sorted(k for k in self.source['model'] if k == 'std' or k.startswith('actor.'))
        self.assertEqual(len(actor_keys), 27)
        state = self.model.state_dict()
        for key in actor_keys:
            self.assertTrue(torch.equal(self.source['model'][key], state[key]), key)
            self.assertEqual(self.source['model'][key].dtype, state[key].dtype, key)
            self.assertEqual(self.source['model'][key].shape, state[key].shape, key)
        # Every copied Actor branch, including the frozen map/history normalizer.
        for branch in ('actor.map_encoder.', 'actor.history_encoder.', 'actor.mlp.',
                       'actor.fusion.', 'actor.normalizer.'):
            self.assertTrue(any(k.startswith(branch) for k in actor_keys), branch)
        self.assertEqual(self.model.actor.normalizer.count.item(),
                         self.source['model']['actor.normalizer.count'].item())
        self.assertTrue(torch.equal(self.model.actor.normalizer._std,
                                    self.source['model']['actor.normalizer._std']))

    def test_fresh_critic_and_statistics_equal_seed42_construction(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            fresh = IdealHeightmapActorCritic()
        state = self.model.state_dict(); reference = fresh.state_dict()
        self.assertEqual(len(state), 51)
        fresh_keys = [k for k in reference if k != 'std' and not k.startswith('actor.')]
        self.assertEqual(len(fresh_keys), 24)
        for key in fresh_keys:
            self.assertTrue(torch.equal(reference[key], state[key]), key)
        self.assertEqual(self.model.critic_normalizer.count.item(), 0)
        self.assertEqual(self.model.transfer_metadata['critic'], 'fresh')
        self.assertEqual(self.model.transfer_metadata['critic_statistics'], 'fresh')
        self.assertEqual(self.model.transfer_metadata['optimizer'], 'fresh_adam')
        self.assertFalse(self.model.transfer_metadata['resume'])
        from src.tasks.velocity.v04.runner import PPO
        self.assertFalse(PPO(self.model, object()).optimizer.state)

    def test_actor_mean_parity_on_the_cpu_audit_inputs(self):
        source_model = IdealHeightmapActorCritic()
        source_model.load_state_dict(self.source['model'], strict=True)
        source_model.eval(); self.model.eval()
        generator = torch.Generator(device='cpu').manual_seed(42)
        observations = {
            'proprio': torch.randn((128, 45), generator=generator),
            'history': torch.randn((128, 5, 45), generator=generator),
            'heightmap': torch.randn((128, 3, 48, 32), generator=generator),
            'privileged': torch.randn((128, 35), generator=generator),
        }
        with torch.no_grad():
            left = source_model(observations); right = self.model(observations)
        maximum = float((left - right).abs().max())
        self.assertTrue(torch.isfinite(left).all() and torch.isfinite(right).all())
        self.assertLessEqual(maximum, 1e-6)
        self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-5))
        source_model.train(); self.model.train()

    def test_migration_is_deterministic_and_independent_of_ambient_rng(self):
        torch.manual_seed(123456)
        first = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        torch.manual_seed(987654)
        second = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        self.assertEqual(first.transfer_metadata, second.transfer_metadata)

    def test_transfer_metadata_records_source_and_v04_recipe(self):
        meta = self.model.transfer_metadata
        self.assertEqual(meta['kind'], 'v02_formal_3000_actor_only')
        self.assertEqual(meta['sha256'], spec.INIT_SHA256)
        self.assertEqual(meta['source_iteration'], 3000)
        self.assertEqual(meta['source_optimizer_steps'], 60000)
        self.assertEqual(meta['source_code_version'], '0.2.1')
        self.assertEqual(meta['source_checkpoint_schema'], 3)
        self.assertEqual(meta['recipe'], '0.4-r1')
        self.assertEqual(meta['reward_version'], spec.RECIPE_VERSION)
        self.assertEqual(meta['reward_config_sha256'], spec.reward_config_sha256())
        self.assertEqual(meta['source_contract'], source_contract())
        self.assertEqual(meta['std_before'], self.source['model']['std'].tolist())
        self.assertEqual(meta['std_after'], self.model.std.detach().tolist())
        self.assertTrue(Path(meta['source_path']).is_absolute())


class SourceRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=False)

    def migrate(self, source):
        with patch('src.tasks.velocity.v04.model.sha256', return_value=spec.INIT_SHA256), \
             patch('src.tasks.velocity.v04.model.torch.load', return_value=source):
            return V04ActorCritic.from_v02('trusted-test-double', spec.INIT_SHA256)

    def test_wrong_sha_is_rejected_before_any_checkpoint_is_read(self):
        with patch('src.tasks.velocity.v04.model.torch.load') as loader:
            for expected in ('0'*64, 'f'*64, spec.INIT_SHA256.upper(), None):
                with self.assertRaises(ValueError):
                    V04ActorCritic.from_v02(spec.INIT_PATH, expected)
            loader.assert_not_called()
        with patch('src.tasks.velocity.v04.model.sha256', return_value='0'*64):
            with self.assertRaises(ValueError):
                V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)

    def test_source_contract_or_metadata_mismatch_is_rejected(self):
        mutations = ('contract_schema', 'contract_protocol', 'contract_source',
                     'metadata_task', 'metadata_protocol', 'metadata_code',
                     'metadata_schema', 'metadata_seed', 'num_envs', 'curriculum_stage',
                     'curriculum_progress', 'curriculum_env_steps', 'learning_updates',
                     'learning_adam', 'ppo', 'resume_mode')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.source)
                if mutation == 'contract_schema': bad['contract']['checkpoint_schema'] = 4
                if mutation == 'contract_protocol': bad['contract']['protocol_id'] = 'wrong'
                if mutation == 'contract_source': bad['contract']['behavior_sha256'] = {'bogus':'0'*64}
                if mutation == 'metadata_task': bad['metadata']['task'] = 'Wrong'
                if mutation == 'metadata_protocol': bad['metadata']['protocol_id'] = 'wrong'
                if mutation == 'metadata_code': bad['metadata']['code_version'] = '0.3.0'
                if mutation == 'metadata_schema': bad['metadata']['checkpoint_schema'] = 5
                if mutation == 'metadata_seed': bad['metadata']['seed'] = 43
                if mutation == 'num_envs': bad['num_envs'] = 258
                if mutation == 'curriculum_stage': bad['curriculum']['stage'] = 'S1'
                if mutation == 'curriculum_progress': bad['curriculum']['iteration'] = 2999
                if mutation == 'curriculum_env_steps': bad['curriculum']['env_steps'] = 73728001
                if mutation == 'learning_updates': bad['learning_state']['completed_ppo_updates'] = 2999
                if mutation == 'learning_adam': bad['learning_state']['optimizer_steps'] = 59999
                if mutation == 'ppo': bad['ppo']['gamma'] = .9
                if mutation == 'resume_mode': bad['resume_mode'] = 'latest'
                with self.assertRaises(ValueError):
                    self.migrate(bad)

    def test_missing_or_nonfinite_source_state_is_rejected(self):
        for mutation, apply in (
                ('missing_actor', lambda state: state.pop('actor.fusion.bias')),
                ('missing_critic', lambda state: state.pop('critic_map_encoder.0.weight')),
                ('missing_std', lambda state: state.pop('std')),
                ('extra', lambda state: state.update({'actor.bogus': torch.zeros(1)})),
                ('shape', lambda state: state.update({'std': torch.full((11,), .1)})),
                ('dtype', lambda state: state.update({'critic.0.weight': state['critic.0.weight'].double()})),
                ('nan_actor', lambda state: state['actor.mlp.0.weight'].view(-1).__setitem__(0, float('nan'))),
                ('inf_critic', lambda state: state['critic.0.weight'].view(-1).__setitem__(0, float('inf'))),
                ('std_below_floor', lambda state: state['std'].fill_(.049)),
                ('std_above_ceiling', lambda state: state['std'].fill_(.301)),
                ('negative_normalizer_var', lambda state: state['actor.normalizer._var'].fill_(-1.)),
                ('negative_critic_var', lambda state: state['critic_normalizer.var'].fill_(-1.))):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.source)
                apply(bad['model'])
                with self.assertRaises(ValueError):
                    self.migrate(bad)


class ResumeBoundaryTests(unittest.TestCase):
    """Schema-5 resume gates that do not require the full source closure."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def preload(self, state):
        path = self.root/'envelope.pt'
        torch.save(state, path)
        return preload_checkpoint(path, spec.sha256(path))

    def test_schema4_checkpoint_relabelled_as_schema5_is_rejected(self):
        legacy = Path('logs/a2_ideal_stairs_v03/seed42_20261113_100126/formal_seed42/'
                      'chunk_0004_n512/checkpoints/latest.pt')
        if not legacy.is_file():
            self.skipTest(f'historical v03 checkpoint unavailable: {legacy}')
        state = torch.load(legacy, map_location='cpu', weights_only=False)
        self.assertEqual(state['contract']['checkpoint_schema'], 4)
        self.assertNotIn('reward_state', state)
        relabelled = dict(state, reward_state=REWARD_STATE)
        self.assertEqual(set(relabelled), FULL_RESUME_FIELDS)
        with self.assertRaisesRegex(ValueError, 'schema5'):
            self.preload(relabelled)
        relabelled['contract'] = dict(relabelled['contract'], checkpoint_schema=5)
        with self.assertRaises(ValueError):
            self.preload(relabelled)

    def test_v04_identity_with_schema4_or_foreign_identity_is_rejected(self):
        base = dict(code_version=spec.CODE_VERSION, checkpoint_schema=spec.SCHEMA_VERSION,
                    task=spec.TASK, protocol_id=spec.PROTOCOL_ID,
                    recipe_version=spec.RECIPE_VERSION, resume_mode='new_episode')
        state = {'contract': base, 'ppo': {}, 'model': {}, 'optimizer': {},
                 'learning_state': {}, 'transfer_metadata': {}, 'curriculum': {},
                 'metadata': {'seed':42}, 'num_envs': 512, 'rng': {},
                 'resume_mode': 'new_episode', 'reward_state': REWARD_STATE,
                 'execution': 'formal', 'ledger': {}, 'ledger_anchor': {}}
        self.assertEqual(set(state), FULL_RESUME_FIELDS)
        with self.assertRaisesRegex(ValueError, 'schema5'):
            self.preload(dict(state, contract=dict(base, checkpoint_schema=4)))
        with self.assertRaisesRegex(ValueError, 'schema5'):
            self.preload(dict(state, contract=dict(base, task='Unitree-A2-IdealHeightMap-Stairs-V03')))

    def test_migration_zero_is_initialization_not_a_full_resume(self):
        migrated = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        zero = dict(
            contract=None,
            model={k: v.detach().cpu().clone() for k, v in migrated.state_dict().items()},
            learning_state=dict(optimizer_initialized=False, completed_ppo_updates=0, optimizer_steps=0),
            optimizer=dict(state=dict(), param_groups=[]), scope='initialization',
            transfer_metadata=migrated.transfer_metadata,
            audit_schema='a2-v04-migration-audit-v1',
            source_checkpoint_sha256=spec.INIT_SHA256)
        self.assertNotEqual(set(zero), FULL_RESUME_FIELDS)
        for field in ('rng', 'ledger', 'ledger_anchor', 'reward_state', 'execution',
                      'curriculum', 'metadata', 'num_envs', 'ppo'):
            self.assertNotIn(field, zero)
        self.assertEqual(zero['scope'], 'initialization')
        self.assertFalse(zero['learning_state']['optimizer_initialized'])
        self.assertEqual(zero['optimizer'], dict(state=dict(), param_groups=[]))
        with self.assertRaisesRegex(ValueError, 'field contract'):
            self.preload(zero)
        for field in ('rng', 'ledger', 'ledger_anchor', 'reward_state', 'execution'):
            promoted = dict(zero); promoted[field] = None; promoted['contract'] = None
            with self.subTest(promoted=field):
                with self.assertRaisesRegex(ValueError, 'field contract'):
                    self.preload(promoted)

    @unittest.skipUnless(CONTRACT_READY, CONTRACT_REASON)
    def test_migration_zero_can_be_an_initialization_source_but_not_resume(self):
        contract = spec.contract()
        migrated = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        zero = dict(
            contract=contract,
            model={k: v.detach().cpu().clone() for k, v in migrated.state_dict().items()},
            learning_state=dict(optimizer_initialized=False, completed_ppo_updates=0, optimizer_steps=0),
            optimizer=dict(state=dict(), param_groups=[]), scope='initialization',
            transfer_metadata=migrated.transfer_metadata,
            audit_schema='a2-v04-migration-audit-v1',
            source_checkpoint_sha256=spec.INIT_SHA256)
        self.assertEqual(contract['checkpoint_schema'], 5)
        self.assertEqual(zero['transfer_metadata']['resume'], False)
        with self.assertRaisesRegex(ValueError, 'field contract'):
            self.preload(zero)


if __name__ == '__main__': unittest.main()


class ZeroInitializationTests(unittest.TestCase):
    """The audited zero-update container is initialization, not resume."""

    def test_zero_scope_loads_only_as_initialization(self):
        model = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        zero = dict(
            contract=spec.contract(), model=model.state_dict(),
            learning_state=dict(optimizer_initialized=False, completed_ppo_updates=0,
                                optimizer_steps=0),
            optimizer=dict(state={}, param_groups=[]), scope='initialization',
            transfer_metadata=copy.deepcopy(model.transfer_metadata),
            audit_schema='a2-v04-migration-audit-v1',
            source_checkpoint_sha256=spec.INIT_SHA256,
            source_model_sha256='0' * 64, migrated_model_sha256='0' * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'migration_zero.pt'
            torch.save(zero, path)
            digest = spec.sha256(path)
            restored = V04ActorCritic.from_zero(path, digest, device='cpu')
            self.assertEqual(restored.state_dict().keys(), model.state_dict().keys())
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertEqual(restored.transfer_metadata['initialization'],
                             'migration_zero_initialization_scope')
            # It is intentionally not a schema-5 resume boundary.
            with self.assertRaises(ValueError):
                preload_checkpoint(path, digest)

    def test_bad_zero_scope_or_nonempty_adam_is_rejected(self):
        model = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        base = dict(
            contract=spec.contract(), model=model.state_dict(),
            learning_state=dict(optimizer_initialized=False, completed_ppo_updates=0,
                                optimizer_steps=0),
            optimizer=dict(state={}, param_groups=[]), scope='initialization',
            transfer_metadata=copy.deepcopy(model.transfer_metadata),
            audit_schema='a2-v04-migration-audit-v1',
            source_checkpoint_sha256=spec.INIT_SHA256,
            source_model_sha256='0' * 64, migrated_model_sha256='0' * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'zero.pt'
            for mutation in ('resume_scope', 'nonzero_adam'):
                state = copy.deepcopy(base)
                if mutation == 'resume_scope':
                    state['scope'] = 'formal_resume'
                else:
                    state['optimizer'] = dict(state={'0': {}}, param_groups=[])
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    V04ActorCritic.from_zero(path, spec.sha256(path))
