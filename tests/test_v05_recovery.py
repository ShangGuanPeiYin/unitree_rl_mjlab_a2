"""CPU-only v0.5 schema-6 recovery, ledger and evaluation-transaction tests.

Everything is built in ``tempfile`` trees; no project log directory, network,
GPU or simulator is touched.  The coverage follows SDS 9.2 item 17 and 7.2/7.3:
same-update newest selection, directory collision, partial Adam, unknown tail,
legacy-schema refusal, new-episode window reset, and a crash before/after every
evaluation-transaction node.
"""
from pathlib import Path
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import Normal

from src.tasks.velocity.v05 import ppo as P
from src.tasks.velocity.v05 import recovery as R

ROOT = Path(__file__).resolve().parents[1]
HEX_A = 'a' * 64
HEX_B = 'b' * 64
HEX_C = 'c' * 64


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_weight = nn.Parameter(torch.tensor(.01))
        self.value_weight = nn.Parameter(torch.tensor(1.))
        self.std = nn.Parameter(torch.full((12,), .15))

    def forward(self, obs):
        return (self.actor_weight * obs['proprio'][:, :1]).expand(-1, 12)

    def action_distribution(self, obs):
        return Normal(self(obs), self.std.clamp(.10, .30))

    def evaluate(self, obs):
        return self.value_weight * obs['proprio'][:, :1]

    def update_critic_statistics(self, obs):
        pass

    @torch.no_grad()
    def constrain_std(self):
        self.std.clamp_(.10, .30)


class TinyEnv:
    """Environment double: generator RNG, stage binding that never resets."""
    device = 'cpu'

    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.rng = torch.Generator()
        self.steps = 0
        self.reset_count = 0
        self.stage = None
        self.stage_iteration = None
        self.window = []

    def observation(self, value=1.):
        return dict(proprio=torch.full((self.num_envs, 45), float(value)),
                    heightmap=torch.zeros(self.num_envs, 3, 2, 2))

    def reset(self):
        self.reset_count += 1
        self.steps = 0
        self.window = []
        self.observation_value = float(torch.randn(1, generator=self.rng).item())
        return self.observation(self.observation_value), {}

    def set_stage(self, stage, stage_iteration=0):
        self.stage = stage
        self.stage_iteration = stage_iteration

    def sample_command(self):
        return float(torch.rand(1, generator=self.rng).item() * .4 + .2)

    def step(self, actions):
        self.steps += 1
        self.window.append(self.steps)
        return (self.observation(1.), torch.full((self.num_envs,), 2.),
                torch.zeros(self.num_envs, dtype=torch.bool),
                torch.zeros(self.num_envs, dtype=torch.bool),
                dict(terminal_observation=self.observation(2), reward_terms={'speed': 1.}))


class Rig:
    """One complete learning boundary plus a durable ledger in a temp tree."""
    def __init__(self, root, *, mode='smoke', num_envs=2, seed=7):
        self.root = Path(root)
        self.mode = mode
        self.num_envs = num_envs
        torch.manual_seed(seed)
        self.model = TinyModel()
        self.env = TinyEnv(num_envs)
        self.env.reset()
        self.ppo = P.PPO(self.model, self.env)
        self.ledger = R.AttemptLedger(self.root / 'run', run_id='seed42', mode=mode)
        self.ppo.control_step_callback = self.ledger.control_step
        self.ppo.update_callback = self.ledger.update_event
        self.curriculum = R.genesis_curriculum()
        self.eval_count = 0

    def next_case(self):
        self.eval_count += 1
        return f'{self.eval_count:064x}'

    def sync(self):
        self.curriculum = self.ledger.curriculum
        return self.curriculum

    def update(self):
        self.ledger.begin_attempt(self.ledger.curriculum, self.env.num_envs)
        self.ppo.collect()
        self.ppo.update()
        self.curriculum = R.updated_curriculum(self.ledger.curriculum,
                                              num_envs=self.env.num_envs)
        self.ledger.complete_attempt(self.curriculum, self.ppo.learning)
        self.sync()

    def charge(self, count=1):
        """Charge complete updates through the exact callback event sequence.

        This exercises the durable ledger arithmetic (48 control samples + 20
        Adam steps + one curriculum increment) without paying for PPO at every
        boundary; the PPO->ledger integration is covered separately.
        """
        for _ in range(count):
            num_envs = self.env.num_envs
            update = self.ledger.completed_ppo_updates
            before = self.ledger.curriculum
            self.ledger.begin_attempt(before, num_envs)
            for step in range(1, 49):
                self.ledger.control_step(dict(step_index=step, num_envs=num_envs,
                                              physics_samples_increment=num_envs))
            self.ledger.update_event(
                'optimizer_started',
                dict(learning_state=R.learning_state(update, num_envs=num_envs),
                     expected_optimizer_steps=20))
            for step in range(1, 21):
                state = R.learning_state(update, num_envs=num_envs)
                state['optimizer_step'] = 20 * update + step
                self.ledger.update_event('optimizer_step_completed',
                                         dict(learning_state=state))
            self.ledger.update_event(
                'update_completed',
                dict(learning_state=R.learning_state(update + 1, num_envs=num_envs),
                     optimizer_steps_in_update=20))
            after = R.updated_curriculum(before, num_envs=num_envs)
            self.ledger.complete_attempt(after, R.learning_state(update + 1,
                                                                num_envs=num_envs))
        self.sync()

    def checkpoint(self, name='update_000001.pt', *, record=True, deep=True):
        state = R.build_checkpoint(
            model_state={key: value.detach().clone()
                         for key, value in self.model.state_dict().items()},
            optimizer_state=self.ppo.optimizer.state_dict(), normalizer={},
            learning=self.ledger.expected_learning(), rng=R.capture_rng(self.env),
            curriculum=self.ledger.curriculum, ledger=self.ledger, mode=self.mode,
            seed=42, device='cpu',
            parameters=list(self.model.parameters()) if deep else None)
        path = self.root / 'run' / 'checkpoints' / name
        R.atomic_torch_save(path, state)
        sha = R.sha256(path)
        if record:
            self.ledger.record_checkpoint(path, sha, self.ledger.anchor(),
                                          state['contract_sha256'])
        return path, sha, state

    def evaluation_files(self, name, text='evidence'):
        directory = self.root / 'run' / 'evaluations' / name
        directory.mkdir(parents=True, exist_ok=True)
        result = directory / 'result.json'
        report = directory / 'report.md'
        result.write_text(json.dumps({'text': text}) + '\n')
        report.write_text('# report\n' + text + '\n')
        return dict(result_path=str(result), result_sha256=R.sha256(result),
                    report_path=str(report), report_sha256=R.sha256(report),
                    evidence_sha256=[R.sha256(result)], summary_sha256=HEX_C)

    def plan(self, sha, *, kind='validation', stage=None, update=None, case=None,
             run_id='seed42'):
        return self.ledger.plan_evaluation(
            run_id=run_id, mode=self.mode, stage=stage or self.ledger.curriculum['stage'],
            global_update=self.ledger.completed_ppo_updates if update is None else update,
            kind=kind, checkpoint_sha256=sha, case_sha256=case or self.next_case())

    def evaluate(self, sha, *, kind='validation', passed=True, collapse_hit=False,
                 text='evidence', **kwargs):
        task = self.plan(sha, kind=kind, **kwargs)
        key = task['task_key']
        if self.ledger.task_state(key) == 'planned':
            self.ledger.commit_evaluation_result(key, **self.evaluation_files(
                key.replace('|', '_')[:80], text))
        consumed = self.ledger.consume_evaluation(
            key, outcome='passed' if passed else 'failed', passed=passed,
            collapse_hit=collapse_hit)['task_key']
        self.sync()
        return consumed

    def last_sha(self):
        return self.ledger.checkpoints[-1]['sha256']

    def gate(self, *, passed=True, kind='validation'):
        return self.evaluate(self.last_sha(), kind=kind, passed=passed)

    def advance_stage(self):
        """Two passing validations plus the S1 extra, then the one transition."""
        self.gate()
        self.gate()
        pending = self.ledger.pending_transition
        if pending['requires'] == 'perturbed':
            self.gate(kind='perturbed')
            self.gate()
            self.gate()
            self.gate(kind='perturbed')
            pending = self.ledger.pending_transition
        self.ledger.commit_transition(from_stage=pending['from_stage'],
                                      to_stage=pending['to_stage'],
                                      at_update=pending['at_update'], gate_streak=2)
        return self.sync()['stage']


class TempCase(unittest.TestCase):
    """Temporary tree; fsync is stubbed for speed except in the durability test.

    The production ledger fsyncs every control sample (33 ms per call on this
    filesystem, which would dominate a CPU suite).  ``FsyncDurabilityTests``
    runs with the real fsync and asserts it is actually called.
    """
    def setUp(self):
        torch.set_num_threads(2)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        patcher = patch('src.tasks.velocity.v05.recovery.os.fsync', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)


class FsyncDurabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_every_ledger_append_fsyncs_and_survives_a_clean_reload(self):
        rig = Rig(self.tmp, num_envs=2)
        with patch('src.tasks.velocity.v05.recovery.os.fsync',
                   wraps=os.fsync) as spy:
            rig.ledger.begin_attempt(rig.curriculum, rig.env.num_envs)
            for step in range(1, 49):
                rig.ledger.control_step(dict(step_index=step, num_envs=2,
                                             physics_samples_increment=2))
        self.assertGreaterEqual(spy.call_count, 48)
        reloaded = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        self.assertEqual(reloaded.sampled_control_samples, 96)
        self.assertEqual(reloaded.active_attempt['controls'], 48)


class LearningIdentityTests(TempCase):
    def test_learning_state_is_one_complete_update(self):
        state = R.learning_state(3)
        self.assertEqual(state, dict(update=3, optimizer_step=60, control_samples=73728,
                                     physics_substeps=294912))
        self.assertEqual(R.learning_state(0), dict(update=0, optimizer_step=0,
                                                   control_samples=0, physics_substeps=0))
        self.assertEqual(R.validate_learning(state), 3)

    def test_learning_state_rejects_partial_or_double_counted_sides(self):
        state = R.learning_state(2)
        for mutation in (dict(state, optimizer_step=39), dict(state, optimizer_step=41),
                         dict(state, physics_substeps=state['control_samples']),
                         dict(state, control_samples=state['control_samples'] + 1),
                         dict(state, update=0)):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    R.validate_learning(mutation)
        self.assertRaises(ValueError, R.validate_learning, dict(state, extra=1))
        self.assertRaises(ValueError, R.validate_learning,
                          R.learning_state(1, num_envs=2), num_envs=512)

    def test_adam_state_must_be_the_complete_20_step_boundary(self):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        for _ in range(20):
            sum(parameter.sum() for parameter in model.parameters()).backward()
            optimizer.step()
            model.constrain_std()
            optimizer.zero_grad()
        state = optimizer.state_dict()
        learning = R.learning_state(1, num_envs=2)
        R.validate_optimizer_state(state, list(model.parameters()), learning)
        self.assertEqual(len(state['state']), len(list(model.parameters())))
        for mutation in ('step19', 'step21', 'missing', 'extra', 'nonfinite', 'dtype',
                         'negative'):
            bad = copy.deepcopy(state)
            first = next(iter(bad['state']))
            if mutation == 'step19':
                bad['state'][first]['step'].fill_(19)
            if mutation == 'step21':
                bad['state'][first]['step'].fill_(21)
            if mutation == 'missing':
                bad['state'].pop(first)
            if mutation == 'extra':
                bad['state']['bogus'] = copy.deepcopy(bad['state'][first])
            if mutation == 'nonfinite':
                bad['state'][first]['exp_avg'].view(-1)[0] = float('nan')
            if mutation == 'dtype':
                bad['state'][first]['exp_avg'] = bad['state'][first]['exp_avg'].double()
            if mutation == 'negative':
                bad['state'][first]['exp_avg_sq'].view(-1)[0] = -1.
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    R.validate_optimizer_state(bad, list(model.parameters()), learning)
        with self.assertRaises(ValueError):
            R.validate_optimizer_state(state, list(model.parameters()),
                                       R.learning_state(2, num_envs=2))
        with self.assertRaises(ValueError):
            R.validate_optimizer_state(state, list(model.parameters()),
                                       dict(learning, optimizer_step=21))
        # A zero-update checkpoint may not carry Adam moments.
        with self.assertRaises(ValueError):
            R.validate_optimizer_state(state, list(model.parameters()),
                                       R.learning_state(0, num_envs=2))
        bad_lr = copy.deepcopy(state)
        bad_lr['param_groups'][0]['lr'] = 3e-4
        with self.assertRaises(ValueError):
            R.validate_optimizer_state(bad_lr, list(model.parameters()), learning)


class SchemaTests(TempCase):
    def rig(self):
        rig = Rig(self.tmp, num_envs=2)
        rig.update()
        return rig

    def test_schema6_envelope_is_strict_and_complete(self):
        rig = self.rig()
        path, sha, state = rig.checkpoint()
        self.assertEqual(set(state), set(R.CHECKPOINT_KEYS))
        self.assertEqual(state['schema'], 6)
        self.assertEqual(state['resume_mode'], 'new_episode')
        R.validate_checkpoint(state, parameters=list(rig.model.parameters()))
        loaded = R.preload_checkpoint(path, sha)
        self.assertEqual(loaded['schema'], 6)
        self.assertEqual(R.sha256(path), sha)

    def test_legacy_schema_is_refused_without_touching_the_body(self):
        rig = self.rig()
        _, _, state = rig.checkpoint()
        legacy = dict(state, schema=5)
        with self.assertRaises(R.LegacySchemaError):
            R.validate_checkpoint(legacy)
        unlabelled = dict(state)
        unlabelled.pop('schema')
        unlabelled['contract'] = {'checkpoint_schema': 5}
        with self.assertRaises(R.LegacySchemaError):
            R.validate_checkpoint(unlabelled)
        future = dict(state, schema=7)
        with self.assertRaises(ValueError):
            R.validate_checkpoint(future)

    def test_field_identity_recipe_and_resume_mode_are_fail_closed(self):
        rig = self.rig()
        _, _, state = rig.checkpoint()
        mutations = [
            ('missing', dict(item for item in state.items() if item[0] != 'normalizer')),
            ('extra', dict(state, reward_state='half_trajectory')),
            ('identity', dict(state, identity=dict(state['identity'], task='V04'))),
            ('recipe', dict(state, recipe=dict(state['recipe'], std_min=.05))),
            ('resume', dict(state, resume_mode='half_trajectory')),
            ('contract', dict(state, contract_sha256=HEX_A)),
            ('rng', dict(state, rng={key: value for key, value in state['rng'].items()
                                     if key != 'env'})),
            ('source_mode', dict(state, source=dict(state['source'], mode='unknown'))),
            ('learning', dict(state, learning=dict(state['learning'], optimizer_step=1))),
            ('curriculum', dict(state, curriculum=dict(state['curriculum'], stage='S9'))),
        ]
        for label, mutation in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    R.validate_checkpoint(mutation)

    def test_evaluation_and_ledger_sections_must_agree(self):
        rig = self.rig()
        _, sha, state = rig.checkpoint()
        consumed_key = rig.evaluate(sha)
        _, _, newer = rig.checkpoint('update_000001b.pt')
        self.assertIn(consumed_key, newer['evaluation']['consumed'])
        bad = dict(newer, evaluation=dict(consumed=[HEX_B], latest_diagnostic_update=None))
        with self.assertRaises(ValueError):
            R.validate_checkpoint(bad)
        bad = dict(newer, ledger=dict(newer['ledger'], gate_streak=9))
        with self.assertRaises(ValueError):
            R.validate_checkpoint(bad)
        bad = dict(newer, curriculum=dict(newer['curriculum'],
                                          streak={'gate': 0, 'collapse': 0}))
        with self.assertRaises(ValueError):
            R.validate_checkpoint(bad)
        bad = dict(newer, ledger=dict(newer['ledger'], anchor={'seq': 0, 'digest': HEX_A}))
        with self.assertRaises(R.RecoveryBlocked):
            R.audit_recovery(bad, rig.ledger)

    def test_preload_requires_the_exact_sha(self):
        rig = self.rig()
        path, sha, _ = rig.checkpoint()
        with self.assertRaises(ValueError):
            R.preload_checkpoint(path, HEX_A)
        with self.assertRaises(ValueError):
            R.preload_checkpoint(path, None)

    def test_preload_can_validate_the_real_model_body_on_request(self):
        try:
            from src.tasks.velocity.v05.model import V05ActorCritic
        except ImportError:  # implemented by another work package
            self.skipTest('v05.model is not implemented in this work package')
        rig = self.rig()
        path, sha, state = rig.checkpoint()
        with self.assertRaises(ValueError):
            R.preload_checkpoint(path, sha, validate_bodies=True)


class NativeModelCheckpointTests(TempCase):
    """Schema-6 round trip with the real v05 network (skipped if absent)."""

    def build(self):
        try:
            from src.tasks.velocity.v05.model import V05ActorCritic
        except ImportError:
            self.skipTest('v05.model is not implemented in this work package')
        torch.manual_seed(42)
        model = V05ActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        for _ in range(20):
            sum(parameter.sum() for parameter in model.parameters()).backward()
            optimizer.step()
            model.constrain_std()
            optimizer.zero_grad()
        env = TinyEnv(8)
        env.reset()
        ledger = R.AttemptLedger(self.tmp / 'run', run_id='seed42', mode='smoke')
        # Charge the real event sequence so the durable curriculum matches.
        ledger.begin_attempt(R.genesis_curriculum(), 8)
        for step in range(1, 49):
            ledger.control_step(dict(step_index=step, num_envs=8,
                                     physics_samples_increment=8))
        ledger.update_event('optimizer_started',
                            dict(learning_state=R.learning_state(0, num_envs=8),
                                 expected_optimizer_steps=20))
        for step in range(1, 21):
            partial = R.learning_state(0, num_envs=8)
            partial['optimizer_step'] = step
            ledger.update_event('optimizer_step_completed', dict(learning_state=partial))
        ledger.update_event('update_completed',
                            dict(learning_state=R.learning_state(1, num_envs=8),
                                 optimizer_steps_in_update=20))
        curriculum = R.updated_curriculum(R.genesis_curriculum(), num_envs=8)
        learning = R.learning_state(1, num_envs=8)
        ledger.complete_attempt(curriculum, learning)
        state = R.build_checkpoint(
            model_state={key: value.detach().clone()
                         for key, value in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),
            normalizer=model.actor.normalizer.state_dict(), learning=learning,
            rng=R.capture_rng(env), curriculum=curriculum, ledger=ledger, mode='smoke',
            seed=42, device='cpu', parameters=list(model.parameters()))
        path = self.tmp / 'run' / 'checkpoints' / 'native.pt'
        R.atomic_torch_save(path, state)
        sha = R.sha256(path)
        ledger.record_checkpoint(path, sha, ledger.anchor(), state['contract_sha256'])
        return V05ActorCritic, model, env, path, sha, state, ledger

    def test_real_model_body_round_trips_and_matches_after_restore(self):
        model_class, model, env, path, sha, state, ledger = self.build()
        loaded = R.preload_checkpoint(path, sha, model_factory=model_class)
        target = model_class()
        sequence = R.RecoverySequence(
            env=env, model=target,
            optimizer=torch.optim.Adam(target.parameters(), lr=5e-5))
        sequence.load_learning(loaded).bind_curriculum(loaded).restore_rng(loaded)
        sequence.new_episode_reset()
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
        self.assertTrue(torch.equal(model.actor.normalizer._mean,
                                    target.actor.normalizer._mean))
        self.assertGreaterEqual(float(target.std.detach().min()), .10 - 1e-6)
        self.assertLessEqual(float(target.std.detach().max()), .30 + 1e-6)
        self.assertEqual(R.audit_recovery(loaded, ledger)['status'], 'eligible')


class LedgerTests(TempCase):
    def rig(self):
        rig = Rig(self.tmp, num_envs=2)
        rig.update()
        return rig

    def test_per_control_persistence_and_partial_collect_is_unknown_after_restart(self):
        rig = self.rig()
        _, sha, state = rig.checkpoint()
        rig.evaluate(sha)
        rig.ledger.begin_attempt(rig.curriculum, rig.env.num_envs)
        for step in range(1, 8):
            rig.ledger.control_step(dict(step_index=step, num_envs=2,
                                         physics_samples_increment=2))
        durable = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        self.assertEqual(durable.sampled_control_samples, 2 * 48 + 7 * 2)
        self.assertEqual(durable.control_samples, 2 * 48)
        with self.assertRaises(R.RecoveryBlocked) as caught:
            R.audit_recovery(state, durable)
        self.assertIn('unknown', str(caught.exception))

    def test_exact_complete_boundary_resumes_with_new_episode_semantics(self):
        rig = self.rig()
        _, sha, state = rig.checkpoint()
        audit = R.audit_recovery(state, R.AttemptLedger(rig.root / 'run', run_id='seed42',
                                                        mode='smoke'))
        self.assertEqual(audit['status'], 'eligible')
        self.assertTrue(audit['new_episode'])
        self.assertFalse(audit['physical_trajectory_continuous'])
        self.assertFalse(audit['half_trajectory_restored'])
        self.assertEqual(audit['reward_state'], 'new_episode_no_half_trajectory')
        self.assertEqual(rig.ledger.control_samples, 2 * 48)
        self.assertEqual(rig.ledger.physics_substeps_observed, 4 * 2 * 48)
        snapshot = rig.ledger.snapshot_at(state['ledger']['anchor']['seq'])
        self.assertEqual(snapshot.checkpoint_task_state(), state['ledger'])
        self.assertEqual(snapshot.checkpoint_evaluation(), state['evaluation'])

    def test_unsaved_complete_updates_block_without_refund(self):
        rig = self.rig()
        _, sha, state = rig.checkpoint()
        for _ in range(2):
            rig.update()
        with self.assertRaises(R.RecoveryBlocked) as caught:
            R.audit_recovery(state, rig.ledger)
        self.assertIn('unsaved complete', str(caught.exception))
        self.assertEqual(rig.ledger.completed_ppo_updates, 3)

    def test_optimizer_started_blocks_even_without_an_adam_callback(self):
        rig = self.rig()
        _, sha, state = rig.checkpoint()
        rig.ledger.begin_attempt(rig.curriculum, rig.env.num_envs)
        for step in range(1, 49):
            rig.ledger.control_step(dict(step_index=step, num_envs=2,
                                         physics_samples_increment=2))
        rig.ledger.update_event('optimizer_started',
                                dict(learning_state=R.learning_state(1, num_envs=2),
                                     expected_optimizer_steps=20))
        with self.assertRaises(R.RecoveryBlocked) as caught:
            R.audit_recovery(state, rig.ledger)
        self.assertIn('optimizer_started', str(caught.exception))
        self.assertEqual(rig.ledger.completed_ppo_updates, 1)

    def test_full_collect_only_discard_preserves_the_budget(self):
        rig = self.rig()
        _, _, state = rig.checkpoint()
        rig.ledger.begin_attempt(rig.curriculum, rig.env.num_envs)
        for step in range(1, 49):
            rig.ledger.control_step(dict(step_index=step, num_envs=2,
                                         physics_samples_increment=2))
        audit = R.audit_recovery(state, R.AttemptLedger(rig.root / 'run', run_id='seed42',
                                                        mode='smoke'),
                                 discard_collect_only=True)
        self.assertEqual(audit['discarded_samples'], 96)
        durable = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        self.assertEqual((durable.sampled_control_samples, durable.discarded_samples,
                          durable.control_samples), (192, 96, 96))
        self.assertEqual(durable.completed_ppo_updates, 1)

    def test_partial_collect_only_a_live_owner_can_close(self):
        rig = self.rig()
        rig.checkpoint()
        rig.ledger.begin_attempt(rig.curriculum, rig.env.num_envs)
        for step in range(1, 4):
            rig.ledger.control_step(dict(step_index=step, num_envs=2,
                                         physics_samples_increment=2))
        with self.assertRaises(ValueError):
            R.AttemptLedger(rig.root / 'run', run_id='seed42',
                            mode='smoke').close_collect_only(known_no_inflight=True,
                                                             proof='not evidence')
        rig.ledger.close_collect_only(known_no_inflight=True,
                                      proof='synchronous env returned; no next step issued')
        self.assertEqual(R.AttemptLedger(rig.root / 'run', run_id='seed42',
                                         mode='smoke').discarded_samples, 6)

    def test_unknown_tail_blocks(self):
        rig = self.rig()
        _, _, state = rig.checkpoint()
        rig.ledger.mark_unknown('power lost before the last control sample was persisted')
        with self.assertRaises(R.RecoveryBlocked) as caught:
            R.audit_recovery(state, rig.ledger)
        self.assertIn('unknown', str(caught.exception))

    def test_tampered_or_truncated_durable_events_are_rejected(self):
        rig = self.rig()
        rig.checkpoint()
        path = rig.root / 'run' / 'ledger.jsonl'
        lines = path.read_text().splitlines()
        event = json.loads(lines[-1])
        event['payload']['update'] = 99
        lines[-1] = json.dumps(event)
        path.write_text('\n'.join(lines) + '\n')
        with self.assertRaises(ValueError):
            R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        path.write_text(path.read_text().rstrip('\n'))
        with self.assertRaises(ValueError):
            R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')

    def test_newest_same_update_checkpoint_wins_by_ledger_order_not_mtime(self):
        rig = self.rig()
        first, first_sha, first_state = rig.checkpoint('first.pt')
        second, second_sha, second_state = rig.checkpoint('second.pt')
        self.assertNotEqual(first_sha, second_sha)
        # A later mtime on the older ledger record must not win.
        os.utime(first, (4e9, 4e9))
        os.utime(second, (1e9, 1e9))
        selected = R.select_checkpoint(rig.ledger, directory=first.parent)
        self.assertEqual(selected['sha256'], second_sha)
        self.assertEqual(selected['path'].name, 'second.pt')
        # The older same-update state is refused instead of silently used.
        with self.assertRaises(R.RecoveryBlocked) as caught:
            R.audit_recovery(first_state, rig.ledger)
        self.assertIn('newer complete checkpoint', str(caught.exception))
        self.assertEqual(R.audit_recovery(second_state, rig.ledger)['status'], 'eligible')
        # A missing newest file blocks; it never falls back to the older file.
        second.unlink()
        with self.assertRaises(R.RecoveryBlocked) as caught:
            R.select_checkpoint(rig.ledger, directory=first.parent)
        self.assertIn('missing', str(caught.exception))

    def test_bounded_memory_keeps_the_durable_chain_and_old_prefixes(self):
        rig = self.rig()
        first, sha, state = rig.checkpoint()
        anchor = state['ledger']['anchor']
        # 1 begin + 48 control + 1 optimizer_started + 20 Adam + 1 update_completed
        # + 1 complete_attempt + 1 checkpoint_committed = 73 durable events.
        self.assertEqual(anchor['seq'], 72)
        bounded = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke',
                                  retain_events=8)
        self.assertEqual(len(bounded.events), 8)
        self.assertEqual(bounded.anchor()['seq'], 73)
        with self.assertRaises(ValueError):
            bounded.state_dict()
        snapshot = bounded.snapshot_at(anchor['seq'])
        self.assertEqual(snapshot.checkpoint_task_state(), state['ledger'])
        self.assertEqual(snapshot.curriculum, state['curriculum'])
        self.assertEqual(snapshot.checkpoint_evaluation(), state['evaluation'])
        self.assertEqual(bounded.snapshot_at(bounded.anchor()['seq']).completed_ppo_updates, 1)
        # The bounded owner can still append, and the chain stays loadable.
        bounded.begin_attempt(snapshot.curriculum, 2)
        self.assertEqual(bounded.active_attempt['controls'], 0)
        self.assertEqual(R.AttemptLedger(rig.root / 'run', run_id='seed42',
                                        mode='smoke').anchor()['seq'], 74)

    def test_reconcile_state_synchronises_a_behind_state_atomically(self):
        rig = self.rig()
        state_path = rig.root / 'run' / 'state.json'
        rig.checkpoint()
        plan = R.reconcile_state(state_path=state_path, ledger=rig.ledger,
                                 directory=rig.root / 'run' / 'checkpoints')
        self.assertEqual(plan['status'], 'created')
        self.assertEqual(R.read_state(state_path)['learning']['update'], 1)
        self.assertEqual(R.reconcile_state(state_path=state_path, ledger=rig.ledger,
                                           directory=rig.root / 'run' / 'checkpoints')['status'],
                         'up_to_date')
        rig.update()
        rig.checkpoint('update_000002.pt')
        plan = R.reconcile_state(state_path=state_path, ledger=rig.ledger,
                                 directory=rig.root / 'run' / 'checkpoints')
        self.assertEqual(plan['status'], 'synchronized')
        state = R.read_state(state_path)
        self.assertEqual(state['learning']['update'], 2)
        self.assertEqual(state['checkpoint']['sha256'],
                         rig.ledger.checkpoints[-1]['sha256'])
        self.assertEqual(state['ledger_anchor'], rig.ledger.anchor())
        # An incompatible/ahead state is blocked, not silently clamped.
        bad = dict(state, learning=R.learning_state(9, num_envs=2))
        R.atomic_json(state_path, bad)
        with self.assertRaises(R.RecoveryBlocked):
            R.reconcile_state(state_path=state_path, ledger=rig.ledger,
                              directory=rig.root / 'run' / 'checkpoints')
        bad = dict(state, curriculum=dict(state['curriculum'], stage='S3'))
        R.atomic_json(state_path, bad)
        with self.assertRaises(R.RecoveryBlocked):
            R.reconcile_state(state_path=state_path, ledger=rig.ledger,
                              directory=rig.root / 'run' / 'checkpoints')


class EvaluationTransactionTests(TempCase):
    def rig(self):
        rig = Rig(self.tmp, num_envs=2)
        rig.update()
        _, sha, state = rig.checkpoint()
        return rig, sha, state

    def count(self, ledger, kind):
        return len([event for event in ledger.events if event['event'] == kind])

    def test_plan_is_persisted_before_any_execution_and_is_idempotent(self):
        rig, sha, _ = self.rig()
        task = rig.plan(sha, case=HEX_A)
        self.assertEqual(task['state'], 'planned')
        self.assertFalse(task['replayed'])
        self.assertEqual(rig.ledger.task_state(task['task_key']), 'planned')
        self.assertEqual(self.count(rig.ledger, 'evaluation_planned'), 1)
        again = rig.plan(sha, case=HEX_A)
        self.assertEqual(again['task_key'], task['task_key'])
        self.assertTrue(again['replayed'])
        self.assertEqual(self.count(rig.ledger, 'evaluation_planned'), 1)
        plan = R.resume_plan(rig.ledger)
        self.assertEqual([item['task_key'] for item in plan['rerun']], [task['task_key']])
        self.assertTrue(plan['blocked_until_complete'])

    def test_consume_requires_a_committed_result_and_evidence(self):
        rig, sha, _ = self.rig()
        key = rig.plan(sha)['task_key']
        with self.assertRaises(ValueError):
            rig.ledger.consume_evaluation(key, outcome='passed', passed=True,
                                          collapse_hit=False)
        files = rig.evaluation_files('case1')
        with self.assertRaises(ValueError):
            rig.ledger.commit_evaluation_result(
                key, **dict(files, result_path=str(Path(files['result_path']).with_name(
                    'absent.json'))))
        tampered = dict(files, result_sha256=HEX_A)
        with self.assertRaises(ValueError):
            rig.ledger.commit_evaluation_result(key, **tampered)
        self.assertEqual(rig.ledger.commit_evaluation_result(key, **files)['state'],
                         'result_committed')
        self.assertEqual(self.count(rig.ledger, 'evaluation_result_committed'), 1)
        self.assertEqual(rig.ledger.task_state(key), 'result_committed')
        Path(files['report_path']).unlink()
        with self.assertRaises(ValueError):
            rig.ledger.consume_evaluation(key, outcome='passed', passed=True,
                                          collapse_hit=False)
        Path(files['report_path']).write_text('# report\nevidence\n')
        os.utime(files['report_path'], None)

    def test_consumption_and_streak_change_are_one_event(self):
        rig, sha, state = self.rig()
        key = rig.evaluate(sha)
        events = [event for event in rig.ledger.events
                  if event['event'] == 'evaluation_consumed']
        self.assertEqual(len(events), 1)
        payload = events[0]['payload']
        self.assertEqual(payload['task_key'], key)
        self.assertEqual(payload['streak_after'], {'gate': 1, 'collapse': 0})
        self.assertEqual(payload['result_sha256'],
                         rig.ledger.tasks[key]['result']['result_sha256'])
        self.assertEqual(rig.ledger.gate_streak, 1)
        self.assertEqual(self.count(rig.ledger, 'transition_committed'), 0)

    def test_replay_of_plan_commit_and_consume_is_idempotent(self):
        rig, sha, _ = self.rig()
        task = rig.plan(sha, case=HEX_A)
        key = task['task_key']
        files = rig.evaluation_files('case1')
        rig.ledger.commit_evaluation_result(key, **files)
        rigor = dict(outcome='passed', passed=True, collapse_hit=False)
        first = rig.ledger.consume_evaluation(key, **rigor)
        before = copy.deepcopy(rig.ledger.events)
        self.assertTrue(rig.plan(sha, case=HEX_A)['replayed'])
        self.assertTrue(rig.ledger.commit_evaluation_result(key, **files)['replayed'])
        self.assertTrue(rig.ledger.consume_evaluation(key, **rigor)['replayed'])
        self.assertEqual(rig.ledger.events, before)
        self.assertEqual(first['streak'], {'gate': 1, 'collapse': 0})
        self.assertEqual(rig.ledger.gate_streak, 1)
        with self.assertRaises(ValueError):
            rig.ledger.consume_evaluation(key, outcome='failed', passed=False,
                                          collapse_hit=False)

    def test_crash_before_result_commit_replays_the_same_intent(self):
        rig, sha, _ = self.rig()
        key = rig.plan(sha)['task_key']
        restarted = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        plan = R.resume_plan(restarted)
        self.assertEqual([item['task_key'] for item in plan['rerun']], [key])
        self.assertEqual(plan['consume'], [])
        # The same case/model must be re-run; the identity cannot be substituted.
        other = restarted.plan_evaluation(run_id='seed42', mode='smoke',
                                          stage=restarted.curriculum['stage'],
                                          global_update=restarted.completed_ppo_updates,
                                          kind='validation',
                                          checkpoint_sha256=rig.ledger.checkpoints[0]['sha256'],
                                          case_sha256=HEX_B)
        self.assertNotEqual(other['task_key'], key)

    def test_crash_after_result_commit_replays_consumption_once(self):
        rig, sha, _ = self.rig()
        key = rig.plan(sha)['task_key']
        files = rig.evaluation_files('case1')
        rig.ledger.commit_evaluation_result(key, **files)
        restarted = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        plan = R.resume_plan(restarted)
        self.assertEqual([item['task_key'] for item in plan['consume']], [key])
        self.assertEqual(restarted.task_state(key), 'result_committed')
        restarted.consume_evaluation(key, outcome='passed', passed=True, collapse_hit=False)
        self.assertEqual(restarted.gate_streak, 1)
        restarted.consume_evaluation(key, outcome='passed', passed=True, collapse_hit=False)
        self.assertEqual(restarted.gate_streak, 1)
        self.assertEqual(R.resume_plan(restarted)['blocked_until_complete'], False)

    def test_crash_after_consumption_replays_the_transition_once(self):
        rig, sha, _ = self.rig()
        rig.evaluate(sha)
        restarted = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        # Gate needs two passes: the second one sets the pending transition.
        rig.evaluate(sha)
        restarted = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        pending = restarted.pending_transition
        self.assertEqual(pending['from_stage'], 'F0')
        self.assertEqual(pending['to_stage'], 'P1')
        plan = R.resume_plan(restarted)
        self.assertEqual(plan['pending_transition'], pending)
        result = restarted.commit_transition(from_stage='F0', to_stage='P1', at_update=1,
                                             gate_streak=2)
        self.assertTrue(result['committed'])
        self.assertEqual(restarted.curriculum['stage'], 'P1')
        self.assertEqual(restarted.curriculum['streak'], {'gate': 0, 'collapse': 0})
        before = len(restarted.events)
        self.assertTrue(restarted.commit_transition(from_stage='F0', to_stage='P1',
                                                    at_update=1, gate_streak=2)['replayed'])
        self.assertEqual(len(restarted.events), before)
        fresh = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        self.assertEqual(fresh.curriculum, restarted.curriculum)

    def test_failed_gate_resets_streak_and_missing_result_requires_rerun(self):
        rig, sha, _ = self.rig()
        rig.evaluate(sha)
        rig.evaluate(sha, passed=False)
        self.assertEqual(rig.ledger.gate_streak, 0)
        self.assertIsNone(rig.ledger.pending_transition)
        key = rig.plan(sha, kind='diagnostic')['task_key']
        files = rig.evaluation_files('diag')
        rig.ledger.commit_evaluation_result(key, **files)
        Path(files['result_path']).unlink()
        self.assertEqual([item['task_key'] for item in rig.ledger.missing_results()], [key])
        with self.assertRaises(ValueError):
            rig.ledger.consume_evaluation(key, outcome='passed', passed=False,
                                          collapse_hit=True)
        plan = R.resume_plan(rig.ledger)
        self.assertEqual([item['task_key'] for item in plan['missing']], [key])
        self.assertTrue(plan['blocked_until_complete'])

    def test_diagnostic_updates_collapse_streak_only(self):
        rig, sha, _ = self.rig()
        rig.evaluate(sha, kind='diagnostic', passed=False, collapse_hit=True)
        self.assertEqual(rig.ledger.collapse_streak, 1)
        self.assertEqual(rig.ledger.gate_streak, 0)
        rig.evaluate(sha, kind='diagnostic', passed=True, collapse_hit=False)
        self.assertEqual(rig.ledger.collapse_streak, 0)
        rig.evaluate(sha, kind='diagnostic', passed=True, collapse_hit=True)
        self.assertEqual(rig.ledger.collapse_streak, 1)
        rig.evaluate(sha, kind='flat_retention', passed=False)
        self.assertEqual(rig.ledger.gate_streak, 0)
        self.assertEqual(rig.ledger.collapse_streak, 1)
        self.assertEqual(rig.ledger.checkpoint_evaluation()['latest_diagnostic_update'], 1)

    def test_full_gate_route_including_the_s1_extra_and_s5_terminal_state(self):
        rig, sha, _ = self.rig()
        for expected in ('P1', 'P2', 'P3', 'P4', 'S1'):
            self.assertEqual(rig.advance_stage(), expected)
        # S1: a second pass is not enough; the extra perturbed gate is required.
        rig.gate()
        rig.gate()
        pending = rig.ledger.pending_transition
        self.assertEqual(pending['from_stage'], 'S1')
        self.assertEqual(pending['to_stage'], 'S2')
        self.assertEqual(pending['requires'], 'perturbed')
        with self.assertRaises(ValueError):
            rig.ledger.commit_transition(from_stage='S1', to_stage='S2', at_update=1,
                                         gate_streak=2)
        rig.gate(kind='perturbed', passed=False)
        self.assertEqual(rig.ledger.gate_streak, 0)
        self.assertIsNone(rig.ledger.pending_transition)
        rig.gate()
        rig.gate()
        rig.gate(kind='perturbed')
        pending = rig.ledger.pending_transition
        self.assertEqual(pending['to_stage'], 'S2')
        self.assertIsNone(pending['requires'])
        rig.ledger.commit_transition(from_stage='S1', to_stage='S2', at_update=1,
                                     gate_streak=2)
        self.assertEqual(rig.sync()['stage'], 'S2')
        for expected in ('S3', 'S4', 'S5'):
            self.assertEqual(rig.advance_stage(), expected)
        # S5 end: terminal qualification, not a stage transition.
        rig.gate()
        rig.gate()
        pending = rig.ledger.pending_transition
        self.assertIsNone(pending['to_stage'])
        self.assertEqual(rig.ledger.qualified_update, 1)
        with self.assertRaises(ValueError):
            rig.ledger.commit_transition(from_stage='S5', to_stage=None, at_update=1,
                                         gate_streak=2)
        # A later transition cannot be committed twice.
        fresh = R.AttemptLedger(rig.root / 'run', run_id='seed42', mode='smoke')
        self.assertEqual(fresh.curriculum, rig.ledger.curriculum)
        self.assertEqual(fresh.qualified_update, 1)

    def test_task_key_binds_every_component(self):
        rig, sha, _ = self.rig()
        base = rig.plan(sha)['task_key']
        for changed in (rig.plan(sha, kind='diagnostic')['task_key'],
                        rig.plan(sha, kind='flat_retention')['task_key'],
                        rig.plan(sha, case=HEX_B)['task_key']):
            self.assertNotEqual(changed, base)
        with self.assertRaises(ValueError):
            R.task_key(run_id='seed42', mode='smoke', stage='P1', global_update=1,
                       kind='unknown', checkpoint_sha256=sha, case_sha256=HEX_A)
        with self.assertRaises(ValueError):
            R.task_key(run_id='seed42', mode='smoke', stage='S9', global_update=1,
                       kind='validation', checkpoint_sha256=sha, case_sha256=HEX_A)
        # A test case cannot be planned before the terminal qualification.
        with self.assertRaises(ValueError):
            rig.plan(sha, kind='test')

    def test_evaluation_must_bind_the_current_complete_boundary(self):
        rig, sha, _ = self.rig()
        with self.assertRaises(ValueError):
            rig.plan(sha, update=0)
        with self.assertRaises(ValueError):
            rig.plan(sha, update=2)


class PilotModeTests(TempCase):
    """SDS 7.1: the pilot's P1 may not advance early, but F0 -> P1 still does."""

    def rig(self):
        rig = Rig(self.tmp, mode='pilot', num_envs=2)
        rig.charge(1)
        sha = rig.checkpoint(deep=False)[1]
        return rig, sha

    def test_pilot_f0_two_passes_still_advance_and_p1_passes_do_not(self):
        rig, sha = self.rig()
        rig.gate()
        rig.gate()
        pending = rig.ledger.pending_transition
        self.assertEqual(pending['from_stage'], 'F0')
        self.assertEqual(pending['to_stage'], 'P1')
        with self.assertRaises(ValueError):
            rig.charge(1)  # a pending transition blocks the next attempt
        rig.ledger.commit_transition(from_stage='F0', to_stage='P1', at_update=1,
                                     gate_streak=2)
        rig.sync()
        self.assertEqual(rig.ledger.curriculum['stage'], 'P1')
        rig.charge(99)  # complete update 100
        rig.checkpoint('update_000100.pt', deep=False)
        rig.gate()
        rig.gate()
        self.assertEqual(rig.ledger.gate_streak, 2)
        self.assertIsNone(rig.ledger.pending_transition)
        self.assertEqual(rig.ledger.completed_ppo_updates, 100)
        rig.charge(100)  # complete update 200
        rig.checkpoint('update_000200.pt', deep=False)
        rig.gate()
        rig.gate()
        self.assertEqual(rig.ledger.gate_streak, 2)
        self.assertIsNone(rig.ledger.pending_transition)
        self.assertEqual(rig.ledger.qualified_update, None)
        # Update 201 must be allowed: the pilot P1 runs to its fixed budget.
        rig.charge(1)
        self.assertEqual(rig.ledger.completed_ppo_updates, 201)
        self.assertEqual(rig.ledger.curriculum['stage'], 'P1')
        # One F0 update + 200 P1 updates: the pilot's P1 kept running.
        self.assertEqual(rig.ledger.curriculum['stage_iteration'], 200)
        self.assertEqual(rig.ledger.curriculum['spent']['P1'], 200)
        rig.ledger.begin_attempt(rig.ledger.curriculum, 2)
        self.assertEqual(rig.ledger.active_attempt['attempt'], 202)

    def test_formal_p1_two_passes_still_require_the_transition(self):
        rig = Rig(self.tmp, mode='formal', num_envs=2)
        rig.charge(1)
        sha = rig.checkpoint(deep=False)[1]
        rig.gate()
        rig.gate()
        self.assertEqual(rig.ledger.pending_transition['from_stage'], 'F0')
        rig.ledger.commit_transition(from_stage='F0', to_stage='P1', at_update=1,
                                     gate_streak=2)
        rig.sync()
        rig.gate()
        rig.gate()
        pending = rig.ledger.pending_transition
        self.assertEqual(pending['from_stage'], 'P1')
        self.assertEqual(pending['to_stage'], 'P2')
        with self.assertRaises(ValueError):
            rig.charge(1)


class RecoveryOrderTests(TempCase):
    def rig(self):
        rig = Rig(self.tmp, num_envs=2)
        rig.update()
        path, sha, state = rig.checkpoint()
        return rig, state

    def test_unique_order_is_enforced(self):
        rig, state = self.rig()
        model = TinyModel()
        env = TinyEnv(2)
        sequence = R.RecoverySequence(env=env, model=model,
                                      optimizer=torch.optim.Adam(model.parameters(),
                                                                 lr=5e-5))
        with self.assertRaises(ValueError):
            sequence.bind_curriculum(state)
        with self.assertRaises(ValueError):
            sequence.new_episode_reset()
        with self.assertRaises(ValueError):
            sequence.restore_rng(state)
        sequence.load_learning(state)
        with self.assertRaises(ValueError):
            sequence.load_learning(state)
        with self.assertRaises(ValueError):
            sequence.restore_rng(state)
        sequence.bind_curriculum(state)
        self.assertEqual(env.stage, state['curriculum']['stage'])
        sequence.restore_rng(state)
        with self.assertRaises(ValueError):
            sequence.restore_rng(state)
        observation = sequence.new_episode_reset()
        self.assertEqual(sequence.state, 'ready')
        with self.assertRaises(ValueError):
            sequence.new_episode_reset()
        self.assertIs(observation, sequence.obs)
        self.assertEqual(sequence.reset_count, 1)
        self.assertEqual(env.reset_count, 1)

    def test_stage_binding_must_not_reset_the_environment(self):
        rig, state = self.rig()
        model = TinyModel()
        env = TinyEnv(2)

        def resetting_bind(stage, iteration):
            env.reset()
            env.set_stage(stage, iteration)

        sequence = R.RecoverySequence(env=env, model=model, bind_stage=resetting_bind)
        sequence.load_learning(state)
        with self.assertRaises(ValueError):
            sequence.bind_curriculum(state)
        self.assertEqual(sequence.state, 'learning')

    def test_new_episode_reset_clears_the_reward_window_and_uses_fresh_obs(self):
        rig, state = self.rig()
        model = TinyModel()
        env = TinyEnv(2)
        env.reset()  # construction reset
        construction_observation = env.observation_value
        env.window.extend([1, 2, 3])
        sequence = R.RecoverySequence(env=env, model=model)
        sequence.load_learning(state).bind_curriculum(state).restore_rng(state)
        observation = sequence.new_episode_reset()
        self.assertEqual(env.window, [])
        self.assertNotEqual(observation['proprio'][0, 0].item(),
                            construction_observation)
        self.assertEqual(observation['proprio'][0, 0].item(), env.observation_value)
        # No reward/window state may be carried by the checkpoint.
        self.assertNotIn('reward_state', state)
        self.assertEqual(state['resume_mode'], 'new_episode')

    def test_two_independent_restores_from_one_checkpoint_are_identical(self):
        rig, state = self.rig()

        def restore():
            model = TinyModel()
            env = TinyEnv(2)
            ppo = P.PPO(model, env)
            sequence = R.RecoverySequence(env=env, model=model, optimizer=ppo.optimizer)
            sequence.load_learning(state).bind_curriculum(state).restore_rng(state)
            observation = sequence.new_episode_reset()
            command = env.sample_command()
            torch.manual_seed(3)
            action = model.action_distribution(observation).sample()
            return dict(observation=observation, command=command, action=action,
                        rng=R.capture_rng(env), learning=ppo.learning)

        first, second = restore(), restore()
        self.assertEqual(first['command'], second['command'])
        self.assertEqual(first['observation']['proprio'].tolist(),
                         second['observation']['proprio'].tolist())
        self.assertTrue(torch.equal(first['action'], second['action']))
        self.assertTrue(torch.equal(first['rng']['env'], second['rng']['env']))
        self.assertEqual(first['learning'], second['learning'])
        # Not a claim of physical continuity: the checkpoint has no trajectory.
        self.assertFalse(state['source']['resume'])

    def test_worker_directory_numbers_are_monotonic_and_never_overwritten(self):
        root = self.tmp / 'workers'
        first = R.claim_worker_directory(root)
        self.assertEqual(first.name, 'attempt-0001')
        (first / 'keep.txt').write_text('existing worker evidence\n')
        second = R.claim_worker_directory(root)
        self.assertEqual(second.name, 'attempt-0002')
        (root / 'attempt-0007').mkdir()
        third = R.claim_worker_directory(root)
        self.assertEqual(third.name, 'attempt-0008')
        self.assertEqual((first / 'keep.txt').read_text(), 'existing worker evidence\n')
        self.assertTrue(second.is_dir())
        with self.assertRaises(ValueError):
            R.claim_worker_directory(root, prefix='bad/../name')

    def test_curriculum_guards(self):
        curriculum = R.genesis_curriculum()
        self.assertEqual(curriculum['stage'], 'F0')
        self.assertEqual(set(curriculum), set(R.CURRICULUM_KEYS))
        R.validate_curriculum(curriculum)
        with self.assertRaises(ValueError):
            R.advance_curriculum(curriculum)  # nothing pending
        updated = R.updated_curriculum(curriculum, num_envs=2)
        self.assertEqual(updated['spent']['F0'], 1)
        self.assertEqual(updated['stage_iteration'], 1)
        R.validate_curriculum(updated, learning=R.learning_state(1, num_envs=2))
        with self.assertRaises(ValueError):
            R.validate_curriculum(dict(updated, stage='S9'))
        with self.assertRaises(ValueError):
            R.validate_curriculum(dict(updated, spent=dict(updated['spent'], F0=3)))
        budget = dict(updated, spent=dict(updated['spent'], F0=2000), stage_iteration=2000)
        with self.assertRaises(ValueError):
            R.updated_curriculum(budget, num_envs=2)


if __name__ == '__main__':
    unittest.main()
