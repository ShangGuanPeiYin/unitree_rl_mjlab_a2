"""CPU-only v0.4 Actor transfer; synthetic inputs are not real-observation acceptance."""
import sys
sys.path.insert(0, 'scripts')
import a2_v04_bootstrap
a2_v04_bootstrap.activate('.')

import copy
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import IdealHeightmapActorCritic
from src.tasks.velocity.v02_formal.spec import contract as source_contract
from src.tasks.velocity.v04 import model as target
from src.tasks.velocity.v04 import spec

V04 = target.V04ActorCritic


class PublicAPI(unittest.TestCase):
    def test_explicit_v04_migration_api_exists(self):
        self.assertTrue(callable(V04.from_v02))
        self.assertTrue(callable(target.validate_model_state))
        self.assertEqual(V04.architecture_version, 'v04')
        self.assertEqual(spec.INIT_SHA256,
                         '2ab41ba8cdf18cf1079311af53f50d111e537546030cdf2ccaebcb333f73ff81')


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(spec.INIT_PATH, map_location='cpu', weights_only=False)

    def test_all_actor_tensors_std_and_fresh_seed42_critic(self):
        torch.manual_seed(42)
        fresh = IdealHeightmapActorCritic()
        torch.manual_seed(999)
        model = V04.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        self.assertEqual(len(model.state_dict()), 51)
        for key, value in model.state_dict().items():
            if key == 'std' or key.startswith('actor.'):
                self.assertTrue(torch.equal(value, self.source['model'][key]), key)
            else:
                self.assertTrue(torch.equal(value, fresh.state_dict()[key]), key)
        for prefix in ('critic.', 'critic_map_encoder.', 'critic_history_encoder.'):
            self.assertTrue(any(not torch.equal(v, self.source['model'][k])
                                for k, v in model.state_dict().items() if k.startswith(prefix)))
        self.assertEqual(model.critic_normalizer.count.item(), 0)
        self.assertEqual(model.critic_normalizer.mean.abs().sum().item(), 0.)
        self.assertTrue(all(p.requires_grad for p in model.actor.parameters()))
        self.assertEqual(list(model.actor.normalizer.parameters()), [])
        self.assertEqual(model.recipe, '0.4-r1')
        self.assertEqual(model.recipe_settings, spec.recipe_config())

    def test_transfer_metadata_records_v04_recipe_and_full_source_identity(self):
        model = V04.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        meta = model.transfer_metadata
        self.assertEqual(set(meta), {
            'kind', 'source_path', 'sha256', 'source_iteration', 'source_optimizer_steps',
            'source_code_version', 'source_checkpoint_schema', 'recipe', 'reward_version',
            'reward_config_sha256', 'seed', 'critic', 'critic_statistics', 'optimizer',
            'actor_statistics', 'resume', 'std_before', 'std_after', 'source_contract'})
        self.assertEqual(meta['kind'], 'v02_formal_3000_actor_only')
        self.assertEqual(meta['sha256'], spec.INIT_SHA256)
        self.assertEqual(meta['source_iteration'], 3000)
        self.assertEqual(meta['source_optimizer_steps'], 60000)
        self.assertEqual(meta['source_code_version'], '0.2.1')
        self.assertEqual(meta['source_checkpoint_schema'], 3)
        self.assertEqual(meta['recipe'], spec.RECIPE_VERSION)
        self.assertEqual(meta['reward_version'], spec.RECIPE_VERSION)
        self.assertEqual(meta['reward_config_sha256'], spec.reward_config_sha256())
        self.assertEqual(meta['seed'], 42)
        self.assertEqual(meta['critic'], 'fresh')
        self.assertEqual(meta['critic_statistics'], 'fresh')
        self.assertEqual(meta['optimizer'], 'fresh_adam')
        self.assertEqual(meta['actor_statistics'], 'copied_complete_and_frozen')
        self.assertFalse(meta['resume'])
        self.assertTrue(Path(meta['source_path']).is_absolute())
        self.assertEqual(meta['source_path'], str(Path(spec.INIT_PATH).resolve()))
        self.assertEqual(meta['std_before'], self.source['model']['std'].tolist())
        self.assertEqual(meta['std_after'], model.std.detach().tolist())
        self.assertEqual(meta['std_before'], meta['std_after'])
        self.assertEqual(meta['source_contract'], source_contract())

    def test_fresh_adam_and_no_old_optimizer_state(self):
        model = V04.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        from src.tasks.velocity.v04.runner import PPO
        runner = PPO(model, object())
        self.assertFalse(runner.optimizer.state)
        self.assertEqual(runner.learning_state,
                         dict(optimizer_initialized=False, completed_ppo_updates=0, optimizer_steps=0))
        self.assertTrue(model.std.requires_grad)

    def test_synthetic_actor_mean_parity_and_frozen_statistics(self):
        source_model = IdealHeightmapActorCritic()
        source_model.load_state_dict(self.source['model'], strict=True)
        model = V04.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        stats = {k: v.clone() for k, v in model.state_dict().items() if 'normalizer' in k}
        generator = torch.Generator().manual_seed(812)
        obs = dict(proprio=torch.randn(8, 45, generator=generator),
                   history=torch.randn(8, 5, 45, generator=generator),
                   heightmap=torch.randn(8, 3, 48, 32, generator=generator),
                   privileged=torch.randn(8, 35, generator=generator))
        with torch.no_grad():
            self.assertLessEqual((source_model(obs) - model(obs)).abs().max().item(), 1e-5)
            model.evaluate(obs)
        for key, value in stats.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        altered = dict(obs, privileged=obs['privileged'] + 100)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(obs), model(altered)))

    def test_wrong_sha_and_legacy_entry_are_rejected_before_loading(self):
        with patch.object(target.torch, 'load') as loader:
            with self.assertRaises(ValueError):
                V04.from_v02(spec.INIT_PATH, '0' * 64)
            loader.assert_not_called()
        with patch.object(target, 'sha256', return_value='0' * 64):
            with self.assertRaises(ValueError):
                V04.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        with patch.object(target, 'sha256', return_value=spec.INIT_SHA256), \
             patch.object(target.torch, 'load') as loader:
            with self.assertRaises(ValueError):
                V04.from_v02(spec.INIT_PATH, 'f' * 64)
            loader.assert_not_called()
        with self.assertRaises(ValueError):
            V04.from_eth(spec.INIT_PATH)
        with patch.object(IdealHeightmapActorCritic, 'from_eth') as legacy:
            with self.assertRaises(ValueError):
                V04.from_eth(spec.INIT_PATH)
            legacy.assert_not_called()

    def test_source_identity_and_all_tensor_contract_fail_closed(self):
        mutations = ('schema', 'schema5_relabel', 'protocol', 'source_hash', 'seed', 'updates',
                     'adam', 'paid_steps', 'ppo', 'resume', 'missing', 'shape', 'dtype',
                     'nonfinite', 'count', 'norm_std', 'std_bound')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.source)
                if mutation == 'schema': bad['contract']['checkpoint_schema'] = 4
                if mutation == 'schema5_relabel':
                    bad['contract']['checkpoint_schema'] = 5
                    bad['metadata']['checkpoint_schema'] = 5
                    bad['metadata']['task'] = spec.TASK
                if mutation == 'protocol': bad['contract']['protocol_id'] = 'wrong'
                if mutation == 'source_hash': bad['contract']['behavior_sha256']['wrong'] = '0' * 64
                if mutation == 'seed': bad['metadata']['seed'] = 43
                if mutation == 'updates': bad['curriculum']['iteration'] = 2999
                if mutation == 'adam': bad['learning_state']['optimizer_steps'] = 59999
                if mutation == 'paid_steps': bad['curriculum']['env_steps'] = 73727999
                if mutation == 'ppo': bad['ppo']['gamma'] = .9
                if mutation == 'resume': bad['resume_mode'] = 'latest'
                if mutation == 'missing': del bad['model']['critic_map_encoder.0.weight']
                if mutation == 'shape': bad['model']['std'] = torch.full((11,), .1)
                if mutation == 'dtype': bad['model']['critic.0.weight'] = bad['model']['critic.0.weight'].double()
                if mutation == 'nonfinite': bad['model']['critic.0.weight'][0, 0] = float('nan')
                if mutation == 'count': bad['model']['critic_normalizer.count'].fill_(-1)
                if mutation == 'norm_std': bad['model']['actor.normalizer._std'][0, 0] = 0.
                if mutation == 'std_bound': bad['model']['std'][0] = .31
                with patch.object(target, 'sha256', return_value=spec.INIT_SHA256), \
                     patch.object(target.torch, 'load', return_value=bad):
                    with self.assertRaises(ValueError):
                        V04.from_v02('trusted-test-double', spec.INIT_SHA256)

    def test_general_state_validator_never_accepts_checkpoint_envelope(self):
        model = V04()
        target.validate_model_state(model.state_dict(), model)
        with self.assertRaises(ValueError):
            target.validate_model_state(self.source, model)
        with torch.device('meta'):
            template = V04()
        target.validate_model_state(model.state_dict(), template)
        for key, value in [('critic_normalizer.continuous[48]', True),
                           ('extra', torch.zeros(1)),
                           ('missing', None)]:
            bad = copy.deepcopy(model.state_dict())
            if key.startswith('critic_normalizer'):
                bad['critic_normalizer.continuous'][48] = value
            elif key == 'extra':
                bad[key] = value
            else:
                bad.pop('actor.fusion.bias')
            with self.assertRaises(ValueError):
                target.validate_model_state(bad, template)
        for key, value in [('std', .049), ('std', .301),
                           ('actor.normalizer._var', -1.),
                           ('critic_normalizer.var', -1.),
                           ('critic_normalizer.count', -1.)]:
            bad = copy.deepcopy(model.state_dict())
            if key == 'std':
                bad[key] = torch.full((12,), value)
            elif key == 'critic_normalizer.count':
                bad[key] = torch.tensor(value, dtype=bad[key].dtype)
            else:
                bad[key] = bad[key].clone()
                bad[key].view(-1)[0] = value
            with self.assertRaises(ValueError):
                target.validate_model_state(bad, template)


if __name__ == '__main__': unittest.main()
