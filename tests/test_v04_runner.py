"""CPU runner boundary behavior for the v0.4 schema-5 protocol.

PPO mathematics are imported frozen from ``src.tasks.velocity.v03.ppo``; these
tests only exercise the curriculum/ledger/schema-5 container semantics.  Gate
raw-evidence replay is owned by the evaluation tests.
"""
import sys
sys.path.insert(0, 'scripts')
import a2_v04_bootstrap
a2_v04_bootstrap.activate('.')

import copy
import inspect
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.tasks.velocity.v04 import recovery as v04_recovery
from src.tasks.velocity.v04 import runner as v04_runner
from src.tasks.velocity.v04 import spec
from src.tasks.velocity.v04.model import V04ActorCritic
from src.tasks.velocity.v04.recovery import AttemptLedger, RecoveryBlocked, preload_checkpoint
from src.tasks.velocity.v04.runner import (Curriculum, ProbeCurriculum, PPO, REWARD_STATE,
                                           capture_rng, restore_rng)


def _closure_ready():
    try:
        spec.contract()
        return True, ''
    except (FileNotFoundError, SyntaxError, UnicodeDecodeError) as error:
        # A concurrently incomplete/undeveloped dependency closure cannot be
        # contracted; report the exact cause instead of pretending success.
        return False, f'v0.4 source closure unavailable: {type(error).__name__}: {error}'


CONTRACT_READY, CONTRACT_REASON = _closure_ready()


def learning(n):
    return dict(optimizer_initialized=n > 0, completed_ppo_updates=n, optimizer_steps=20*n)


def controls(ledger, n=512, count=48):
    for step in range(1, count+1):
        ledger.control_step(dict(step_index=step, num_envs=n, physics_samples_increment=n))


def complete(ledger, c, n=512):
    ledger.begin_attempt(c, n)
    controls(ledger, n)
    ledger.update_event('optimizer_started', {'expected_optimizer_steps':20, 'learning_state':learning(c.iteration)})
    for step in range(1, 21):
        state = learning(c.iteration)
        state.update(optimizer_initialized=True, optimizer_steps=20*c.iteration+step)
        ledger.update_event('optimizer_step_completed', {'learning_state':state})
    ledger.update_event('update_completed', {'learning_state':learning(c.iteration+1), 'optimizer_steps_in_update':20})
    c.updated(n)
    ledger.complete_attempt(c, learning(c.iteration))


FIXTURE_CONTRACT = dict(code_version=spec.CODE_VERSION, checkpoint_schema=spec.SCHEMA_VERSION,
                        task=spec.TASK, protocol_id=spec.PROTOCOL_ID,
                        recipe_version=spec.RECIPE_VERSION, resume_mode='new_episode')


def prepared(root, num_envs=512, name='run'):
    """Save one complete schema-5 checkpoint (1 PPO update / 20 Adam steps)."""
    ledger = AttemptLedger(root/name, run_id='seed42')
    env = Env(); env.num_envs = num_envs
    model = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
    runner = PPO(model, env, ledger=ledger)
    for _ in range(20):
        sum(p.sum() for p in model.parameters()).backward()
        runner.optimizer.step(); model.constrain_std(); runner.optimizer.zero_grad()
    c = Curriculum(); complete(ledger, c, num_envs)
    runner.learning_state = learning(1)
    path = root/f'{name}.pt'
    digest = runner.save(path, c, {'seed':42}, ledger=ledger)
    return runner, ledger, c, model, path, digest


class Env:
    device = 'cpu'; num_envs = 512
    def __init__(self): self.rng = torch.Generator().manual_seed(42); self.resets = 0
    def set_stage(self, stage, stage_iteration=0, previous_stage=None): self.stage = (stage, stage_iteration, previous_stage)
    def reset(self):
        self.resets += 1; random.random(); np.random.rand(); torch.rand(1); torch.rand(1, generator=self.rng)
        return {}, {}


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def test_updates_require_pending_gate_and_exact_actual_samples(self):
        c = Curriculum()
        for _ in range(200): c.updated(512)
        self.assertEqual(c.env_steps, 4915200)
        self.assertTrue(c.validation_due)
        with self.assertRaises(RuntimeError): c.updated(512)
        restored = Curriculum(); restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.state_dict(), c.state_dict())
        bad = c.state_dict(); bad['env_steps'] -= 48
        with self.assertRaises(ValueError): Curriculum().load_state_dict(bad)
        bad = c.state_dict(); bad['stage'] = 'S2'; bad['stage_iteration'] = 0
        with self.assertRaises(ValueError): Curriculum().load_state_dict(bad)
        bad = c.state_dict(); bad['consumed']['S1'] = 101
        with self.assertRaises(ValueError): Curriculum().load_state_dict(bad)
        self.assertEqual(c.stages, ['S1', 'S2', 'S3', 'S4', 'S5'])

    def test_fake_summary_without_raw_evidence_never_consumes_gate(self):
        c = Curriculum()
        for _ in range(200): c.updated(512)
        with self.assertRaises((ValueError, ImportError, ModuleNotFoundError)):
            c.validate({'passed':True}, '1'*64, '2'*64)
        self.assertEqual(c.history, [])

    def test_probe_55_plus_resume_one_has_real_zero_based_counters(self):
        c = ProbeCurriculum()
        for _ in range(55): c.updated(512)
        restored = ProbeCurriculum(); restored.load_state_dict(c.state_dict()); restored.updated(512)
        self.assertEqual((restored.iteration, restored.env_steps, restored.environment_stage_iteration),
                         (56, 1376256, 400))
        with self.assertRaises(RuntimeError): restored.updated(512)
        with self.assertRaises(ValueError): Curriculum().load_state_dict(restored.state_dict())
        with self.assertRaises(ValueError): ProbeCurriculum().load_state_dict(Curriculum().state_dict())

    def test_rng_restore_after_reset_preserves_all_cpu_streams(self):
        env = Env(); saved = capture_rng(env)
        expected = (random.random(), np.random.rand(), torch.rand(5), torch.rand(5, generator=env.rng))
        env.reset(); restore_rng(saved, env)
        actual = (random.random(), np.random.rand(), torch.rand(5), torch.rand(5, generator=env.rng))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))


class FrozenPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def test_v04_wrapper_reuses_the_frozen_v03_ppo_math(self):
        import src.tasks.velocity.v03.ppo as v03_ppo
        self.assertIs(v04_runner.CorePPO, v03_ppo.PPO)
        self.assertIs(v04_runner.PPOConfig, v03_ppo.PPOConfig)
        self.assertIs(v04_runner.PPO.__mro__[1], v03_ppo.PPO)
        for method in ('collect', 'update'):
            source = inspect.getsource(getattr(v04_runner.PPO, method))
            self.assertIn(f'super().{method}()', source)
            self.assertNotIn('randperm', source)
            self.assertNotIn('learning_state[', source)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.std = torch.nn.Parameter(torch.full((12,), .2)); self.value = torch.nn.Parameter(torch.tensor(2.))
    def action_distribution(self, obs):
        return torch.distributions.Normal(torch.zeros(len(obs['proprio']), 12)+self.value*0, self.std)
    def evaluate(self, obs): return self.value.expand(len(obs['proprio']), 1)
    def update_critic_statistics(self, obs): pass
    @torch.no_grad()
    def constrain_std(self): self.std.clamp_(.05, .30)


class TinyEnv(Env):
    def __init__(self, fail=False):
        super().__init__(); self.calls = 0; self.fail = fail
    def reset(self):
        return {'proprio':torch.zeros(self.num_envs, 45), 'heightmap':torch.zeros(self.num_envs, 3, 1, 1)}, {}
    def step(self, actions):
        self.calls += 1
        if self.fail: raise RuntimeError('physics may have advanced')
        obs = self.reset()[0]; flag = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, torch.zeros(self.num_envs), flag, flag, {}


class CoreLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def test_real48_and20adam_callbacks_commit_one_exact_learning_update(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AttemptLedger(directory, run_id='core'); c = Curriculum(); env = TinyEnv()
            runner = PPO(TinyModel(), env, ledger=ledger)
            ledger.begin_attempt(c, 512); runner.collect(); runner.update()
            c.updated(512); ledger.complete_attempt(c, runner.learning_state)
            self.assertEqual((env.calls, c.iteration, ledger.completed_ppo_updates, ledger.physics_samples_observed),
                             (48, 1, 1, 24576))
            self.assertEqual(runner.learning_state['optimizer_steps'], 20)
            self.assertEqual(runner.learning_state, learning(1))
            with self.assertRaises(ValueError): runner.update()
    def test_failed_first_control_cannot_be_retried_under_the_same_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AttemptLedger(directory, run_id='crash'); c = Curriculum(); env = TinyEnv(fail=True)
            runner = PPO(TinyModel(), env, ledger=ledger)
            ledger.begin_attempt(c, 512)
            with self.assertRaisesRegex(RuntimeError, 'physics'): runner.collect()
            with self.assertRaises(ValueError): runner.collect()
            self.assertEqual(env.calls, 1); self.assertTrue(ledger.physics_samples_unknown)
    def test_callbacks_cannot_be_detached_before_physics_or_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AttemptLedger(directory, run_id='detach'); c = Curriculum(); env = TinyEnv()
            runner = PPO(TinyModel(), env, ledger=ledger)
            ledger.begin_attempt(c, 512); runner.control_step_callback = None
            with self.assertRaises(ValueError): runner.collect()
            self.assertEqual(env.calls, 0)


class PinnedContractCheckpointTests(unittest.TestCase):
    """Schema-5 container semantics with a pinned contract value.

    The real contract text is owned by ``test_v04_spec``/the closure test below;
    here save/load must still validate every non-contract field, and both sides
    must compare the same full contract object.
    """
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module in (v04_runner, v04_recovery):
            patcher = patch.object(module, 'contract', return_value=dict(FIXTURE_CONTRACT))
            patcher.start(); self.addCleanup(patcher.stop)
    def test_schema5_container_records_new_episode_semantics_and_full_resume(self):
        runner, ledger, c, model, path, digest = prepared(self.root)
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(set(state), {'contract','ppo','model','optimizer','learning_state',
            'transfer_metadata','curriculum','metadata','num_envs','rng','resume_mode',
            'reward_state','execution','ledger','ledger_anchor'})
        self.assertEqual(state['reward_state'], 'new_episode_no_half_trajectory')
        self.assertEqual(REWARD_STATE, 'new_episode_no_half_trajectory')
        self.assertEqual(state['resume_mode'], 'new_episode')
        self.assertEqual(state['execution'], 'formal')
        self.assertEqual(state['contract'], FIXTURE_CONTRACT)
        self.assertEqual(state['num_envs'], 512)
        # The checkpoint anchor is the durable boundary before its own commit event.
        self.assertEqual(ledger.events[-1]['event'], 'checkpoint_committed')
        self.assertEqual(ledger.events[-1]['payload']['anchor'], state['ledger_anchor'])
        self.assertEqual(state['ledger_anchor']['seq'], ledger.anchor()['seq'] - 1)
        self.assertEqual(state['ledger_anchor']['digest'], ledger.events[-1]['previous'])
        saved = preload_checkpoint(path, digest)
        self.assertEqual(saved['learning_state'], learning(1))
        expected = (random.random(), np.random.rand(), torch.rand(4), torch.rand(4, generator=runner.env.rng))
        target = PPO(V04ActorCritic(), Env(), ledger=ledger); restored = Curriculum()
        target.load(path, restored, digest, ledger=ledger)
        actual = (random.random(), np.random.rand(), torch.rand(4), torch.rand(4, generator=target.env.rng))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))
        self.assertEqual(target.env.resets, 1); self.assertIsNone(target.storage)
        self.assertEqual(restored.state_dict(), c.state_dict())
        self.assertEqual(target.learning_state, runner.learning_state)
        self.assertEqual(target.model.transfer_metadata, model.transfer_metadata)
        for k, v in model.state_dict().items(): self.assertTrue(torch.equal(v, target.model.state_dict()[k]), k)
        for i, m in runner.optimizer.state_dict()['state'].items():
            for k, v in m.items(): self.assertTrue(torch.equal(v, target.optimizer.state_dict()['state'][i][k]))
    def test_mutated_schema5_envelopes_are_rejected(self):
        runner, ledger, c, model, path, digest = prepared(self.root)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        mutations = ('schema', 'task', 'protocol', 'behavior', 'recipe', 'ppo', 'reward_state',
                     'resume_mode', 'missing_adam', 'partial_adam', 'learning', 'rng',
                     'num_envs', 'curriculum', 'transfer', 'model', 'ledger_anchor', 'extra_field')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(saved)
                if mutation == 'schema': bad['contract']['checkpoint_schema'] = 4
                if mutation == 'task': bad['contract']['task'] = 'Unitree-A2-IdealHeightMap-Stairs-V03'
                if mutation == 'protocol': bad['contract']['protocol_id'] = 'v03-stair-body-speed-v1'
                if mutation == 'behavior': bad['contract']['behavior_sha256'] = {'bogus':'0'*64}
                if mutation == 'recipe': bad['contract']['recipe_version'] = '0.3'
                if mutation == 'ppo': bad['ppo']['gamma'] = .9
                if mutation == 'reward_state': bad['reward_state'] = 'half_trajectory'
                if mutation == 'resume_mode': bad['resume_mode'] = 'latest'
                if mutation == 'missing_adam': bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
                if mutation == 'partial_adam': next(iter(bad['optimizer']['state'].values()))['step'].fill_(19)
                if mutation == 'learning': bad['learning_state']['completed_ppo_updates'] = 2
                if mutation == 'rng': bad['rng']['torch'] = torch.ones(3, dtype=torch.uint8)
                if mutation == 'num_envs': bad['num_envs'] = 258
                if mutation == 'curriculum': bad['curriculum']['env_steps'] -= 48
                if mutation == 'transfer': bad['transfer_metadata']['recipe'] = 'C'
                if mutation == 'model': bad['model']['critic.0.weight'][0, 0] = float('nan')
                if mutation == 'ledger_anchor': bad['ledger_anchor']['seq'] += 1
                if mutation == 'extra_field': bad['scope'] = 'initialization'
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    preload_checkpoint(path, spec.sha256(path))
    def test_unknown_tail_and_unsaved_complete_updates_block_resume_without_refund(self):
        runner, ledger, c, model, path, digest = prepared(self.root)
        target = PPO(V04ActorCritic(), Env(), ledger=ledger)
        ledger.mark_unknown('power lost before the next durable sample')
        with self.assertRaises(RecoveryBlocked): target.load(path, Curriculum(), digest, ledger=ledger)
        self.assertEqual(c.iteration, 1)
        fresh = prepared(self.root, name='second')
        runner, ledger2, c2, model2, path2, digest2 = fresh
        complete(ledger2, c2, 512)
        with self.assertRaisesRegex(RecoveryBlocked, 'unsaved complete'):
            PPO(V04ActorCritic(), Env(), ledger=ledger2).load(path2, Curriculum(), digest2, ledger=ledger2)
        self.assertEqual(ledger2.completed_ppo_updates, 2)
        self.assertEqual(c2.iteration, 2)
    def test_save_requires_contract_transfer_seed_boundary_and_valid_num_envs(self):
        ledger = AttemptLedger(self.root/'save', run_id='seed42')
        model = V04ActorCritic.from_v02(spec.INIT_PATH, spec.INIT_SHA256)
        env = Env(); runner = PPO(model, env, ledger=ledger)
        for _ in range(20):
            sum(p.sum() for p in model.parameters()).backward()
            runner.optimizer.step(); model.constrain_std(); runner.optimizer.zero_grad()
        c = Curriculum(); complete(ledger, c, 512); runner.learning_state = learning(1)
        path = self.root/'save.pt'
        with self.assertRaises(ValueError):
            runner.save(path, c, {'seed':43}, ledger=ledger)
        keep = model.transfer_metadata
        model.transfer_metadata = {}
        with self.assertRaises(ValueError):
            runner.save(path, c, {'seed':42}, ledger=ledger)
        model.transfer_metadata = keep
        env.num_envs = 999
        with self.assertRaises(ValueError):
            runner.save(path, c, {'seed':42}, ledger=ledger)
        env.num_envs = 512
        with patch.object(v04_runner, 'contract',
                          return_value=dict(FIXTURE_CONTRACT, checkpoint_schema=4)):
            with self.assertRaises(ValueError):
                runner.save(path, c, {'seed':42}, ledger=ledger)
        self.assertFalse(path.exists())
        c.updated(512)
        with self.assertRaises(ValueError):
            runner.save(path, c, {'seed':42}, ledger=ledger)
    def test_environment_count_downgrade_is_rejected(self):
        runner, ledger, c, model, path, digest = prepared(self.root, num_envs=512, name='down')
        env = Env(); env.num_envs = 258
        target = PPO(V04ActorCritic(), env, ledger=ledger); restored = Curriculum()
        with self.assertRaises(ValueError):
            target.load(path, restored, digest, ledger=ledger, allow_num_env_change=True)
        self.assertEqual(restored.iteration, 0)
    def test_probe_and_formal_execution_never_mix(self):
        runner, ledger, c, model, path, digest = prepared(self.root, name='exec')
        with self.assertRaises(ValueError):
            preload_checkpoint(path, digest, execution='probe')
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['execution'] = 'probe'; state['curriculum'] = dict(state['curriculum'], execution='probe')
        torch.save(state, path)
        digest = spec.sha256(path)
        with self.assertRaises(ValueError):
            preload_checkpoint(path, digest, execution='formal')
        target = PPO(V04ActorCritic(), Env(), ledger=AttemptLedger(self.root/'probe', run_id='seed42', execution='probe'))
        with self.assertRaises(ValueError):
            target.load(path, ProbeCurriculum(), digest, ledger=target.ledger)


@unittest.skipUnless(CONTRACT_READY, CONTRACT_REASON)
class RealContractCheckpointTests(unittest.TestCase):
    """Unpatched end-to-end schema-5 save/load under the real v0.4 contract."""
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
    def test_real_contract_identity_reward_config_and_resume(self):
        runner, ledger, c, model, path, digest = prepared(self.root)
        state = torch.load(path, map_location='cpu', weights_only=False)
        contract = state['contract']
        self.assertEqual(contract, spec.contract())
        self.assertEqual((contract['code_version'], contract['checkpoint_schema'],
                          contract['task'], contract['protocol_id']),
                         ('0.4.0', 5, spec.TASK, spec.PROTOCOL_ID))
        self.assertEqual(contract['recipe_version'], '0.4-r1')
        self.assertEqual(contract['reward_version'], '0.4-r1')
        self.assertEqual(contract['reward_config_sha256'], spec.reward_config_sha256())
        self.assertEqual(state['reward_state'], 'new_episode_no_half_trajectory')
        self.assertEqual(preload_checkpoint(path, digest)['reward_state'], REWARD_STATE)
        target = PPO(V04ActorCritic(), Env(), ledger=ledger); restored = Curriculum()
        target.load(path, restored, digest, ledger=ledger)
        self.assertEqual(restored.state_dict(), c.state_dict())
        self.assertIsNone(target.storage)
    def test_contract_configuration_change_and_schema4_are_rejected(self):
        runner, ledger, c, model, path, digest = prepared(self.root, name='config')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        mutations = ('schema', 'recipe_version', 'reward_config', 'reward_sha', 'behavior',
                     'code_version', 'behavior_extra')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(saved)
                if mutation == 'schema': bad['contract']['checkpoint_schema'] = 4
                if mutation == 'recipe_version': bad['contract']['recipe_version'] = '0.3'
                if mutation == 'reward_config': bad['contract']['reward_config']['speed']['exp_scale'] = 5.
                if mutation == 'reward_sha': bad['contract']['reward_config_sha256'] = '0'*64
                if mutation == 'behavior': bad['contract']['behavior_sha256']['bogus'] = '0'*64
                if mutation == 'code_version': bad['contract']['code_version'] = '0.3.0'
                if mutation == 'behavior_extra': bad['contract']['behavior_sha256'].pop(
                    'src/tasks/velocity/v04/model.py')
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    preload_checkpoint(path, spec.sha256(path))
        self.assertIn('src/tasks/velocity/v04/model.py', saved['contract']['behavior_sha256'])
        self.assertIn('src/tasks/velocity/v04/runner.py', saved['contract']['behavior_sha256'])
        self.assertIn('src/tasks/velocity/v04/recovery.py', saved['contract']['behavior_sha256'])
        self.assertIn('src/tasks/velocity/v04/speed.py', saved['contract']['behavior_sha256'])


if __name__ == '__main__': unittest.main()
