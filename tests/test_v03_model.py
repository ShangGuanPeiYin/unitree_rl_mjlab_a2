"""CPU-only actor transfer; synthetic inputs are not real-observation acceptance."""
import copy
import importlib
import unittest
from unittest.mock import patch

import torch
from src.tasks.velocity.rl.ideal_heightmap_actor_critic import IdealHeightmapActorCritic
from src.tasks.velocity.v03.spec import INIT_PATH, INIT_SHA256

try:
    target = importlib.import_module('src.tasks.velocity.v03.model')
except ModuleNotFoundError:
    target = None


class PublicAPI(unittest.TestCase):
    def test_explicit_v03_migration_api_exists(self):
        self.assertIsNotNone(target, 'V3 model module is not implemented')
        self.assertTrue(callable(target.V03ActorCritic.from_v02))
        self.assertTrue(callable(target.validate_model_state))


@unittest.skipIf(target is None, 'V3 implementation absent during RED')
class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(INIT_PATH, map_location='cpu', weights_only=False)

    def test_all_actor_tensors_std_and_fresh_seed42_critic(self):
        torch.manual_seed(42)
        fresh = IdealHeightmapActorCritic()
        torch.manual_seed(999)
        model = target.V03ActorCritic.from_v02(INIT_PATH, INIT_SHA256)
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
        self.assertTrue(all(p.requires_grad for p in model.actor.parameters()))
        self.assertEqual(list(model.actor.normalizer.parameters()), [])
        self.assertEqual(model.transfer_metadata['kind'], 'v02_formal_3000_actor_only')
        self.assertEqual(model.transfer_metadata['source_iteration'], 3000)
        self.assertEqual(model.transfer_metadata['source_optimizer_steps'], 60000)
        self.assertEqual(model.transfer_metadata['critic'], 'fresh')
        self.assertEqual(model.transfer_metadata['critic_statistics'], 'fresh')
        self.assertEqual(model.transfer_metadata['optimizer'], 'fresh_adam')
        self.assertFalse(model.transfer_metadata['resume'])
        from src.tasks.velocity.v03.ppo import PPO
        runner = PPO(model, object())
        self.assertFalse(runner.optimizer.state)
        self.assertTrue(model.std.requires_grad)

    def test_synthetic_actor_mean_parity_and_frozen_statistics(self):
        source_model = IdealHeightmapActorCritic()
        source_model.load_state_dict(self.source['model'], strict=True)
        model = target.V03ActorCritic.from_v02(INIT_PATH, INIT_SHA256)
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
                target.V03ActorCritic.from_v02(INIT_PATH, '0' * 64)
            loader.assert_not_called()
        with patch.object(target, 'sha256', return_value='0' * 64):
            with self.assertRaises(ValueError):
                target.V03ActorCritic.from_v02(INIT_PATH, INIT_SHA256)
        with self.assertRaises(ValueError):
            target.V03ActorCritic.from_eth(INIT_PATH)

    def test_source_identity_and_all_tensor_contract_fail_closed(self):
        mutations = ('schema', 'protocol', 'source_hash', 'seed', 'updates', 'adam',
                     'paid_steps', 'ppo', 'missing', 'shape', 'dtype', 'nonfinite', 'count')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.source)
                if mutation == 'schema': bad['contract']['checkpoint_schema'] = 4
                if mutation == 'protocol': bad['contract']['protocol_id'] = 'wrong'
                if mutation == 'source_hash': bad['contract']['behavior_sha256']['wrong'] = '0' * 64
                if mutation == 'seed': bad['metadata']['seed'] = 43
                if mutation == 'updates': bad['curriculum']['iteration'] = 2999
                if mutation == 'adam': bad['learning_state']['optimizer_steps'] = 59999
                if mutation == 'paid_steps': bad['curriculum']['env_steps'] = 73727999
                if mutation == 'ppo': bad['ppo']['gamma'] = .9
                if mutation == 'missing': del bad['model']['critic_map_encoder.0.weight']
                if mutation == 'shape': bad['model']['std'] = torch.full((11,), .1)
                if mutation == 'dtype': bad['model']['critic.0.weight'] = bad['model']['critic.0.weight'].double()
                if mutation == 'nonfinite': bad['model']['critic.0.weight'][0, 0] = float('nan')
                if mutation == 'count': bad['model']['critic_normalizer.count'].fill_(-1)
                with patch.object(target, 'sha256', return_value=INIT_SHA256), \
                     patch.object(target.torch, 'load', return_value=bad):
                    with self.assertRaises(ValueError):
                        target.V03ActorCritic.from_v02('trusted-test-double', INIT_SHA256)

    def test_general_state_validator_never_accepts_checkpoint_envelope(self):
        model = target.V03ActorCritic()
        target.validate_model_state(model.state_dict(), model)
        with self.assertRaises(ValueError):
            target.validate_model_state(self.source, model)
        with torch.device('meta'):
            template = target.V03ActorCritic()
        target.validate_model_state(model.state_dict(), template)
        bad = copy.deepcopy(model.state_dict())
        bad['critic_normalizer.continuous'][48] = True
        with self.assertRaises(ValueError):
            target.validate_model_state(bad, template)
