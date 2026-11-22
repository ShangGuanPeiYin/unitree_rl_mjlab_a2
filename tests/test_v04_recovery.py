"""Strict durable accounting: unfinished learning cannot be priced as complete.

v0.4 keeps the v0.3 schema-5 learning ledger arithmetic (48 controls + exactly
20 Adam steps per complete PPO update) and adds the explicit new-episode reward
state contract.  No rollback or budget refund is allowed.
"""
import sys
sys.path.insert(0, 'scripts')
import a2_v04_bootstrap
a2_v04_bootstrap.activate('.')

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


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


def checkpoint(c, ledger):
    from src.tasks.velocity.v04.runner import REWARD_STATE
    return dict(curriculum=c.state_dict(), learning_state=learning(c.iteration),
                ledger=ledger.state_dict(), ledger_anchor=ledger.anchor(),
                execution=c.execution, reward_state=REWARD_STATE)


class LearningIdentityTests(unittest.TestCase):
    def test_learning_state_is_exactly_20_adam_steps_per_complete_update(self):
        from src.tasks.velocity.v04.recovery import learning_state
        for n in (0, 1, 7, 3000):
            self.assertEqual(learning_state(n), dict(
                optimizer_initialized=n > 0, completed_ppo_updates=n, optimizer_steps=20*n))

    def test_adam_step_moments_and_finiteness_are_bound_to_the_update_ledger(self):
        import torch
        from src.tasks.velocity.v04.model import V04ActorCritic
        from src.tasks.velocity.v04.recovery import validate_optimizer_state
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = V04ActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(20):
            sum(p.sum() for p in model.parameters()).backward()
            optimizer.step()
            model.constrain_std()
            optimizer.zero_grad()
        state = optimizer.state_dict()
        validate_optimizer_state(state, model, learning(1))
        self.assertEqual(len(state['state']), 43)
        for mutation in ('step19', 'step21', 'missing', 'extra', 'nonfinite', 'dtype'):
            bad = copy.deepcopy(state)
            first = next(iter(bad['state']))
            if mutation == 'step19': bad['state'][first]['step'].fill_(19)
            if mutation == 'step21': bad['state'][first]['step'].fill_(21)
            if mutation == 'missing': bad['state'].pop(first)
            if mutation == 'extra': bad['state']['bogus'] = copy.deepcopy(bad['state'][first])
            if mutation == 'nonfinite': bad['state'][first]['exp_avg'].view(-1)[0] = float('nan')
            if mutation == 'dtype': bad['state'][first]['exp_avg'] = bad['state'][first]['exp_avg'].double()
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    validate_optimizer_state(bad, model, learning(1))
        with self.assertRaises(ValueError):
            validate_optimizer_state(state, model, learning(2))
        with self.assertRaises(ValueError):
            validate_optimizer_state(state, model, dict(learning(1), optimizer_steps=21))


class LedgerTests(unittest.TestCase):
    def setUp(self):
        from src.tasks.velocity.v04.runner import Curriculum, ProbeCurriculum
        from src.tasks.velocity.v04.recovery import AttemptLedger, RecoveryBlocked, audit_recovery
        self.Curriculum = Curriculum
        self.ProbeCurriculum = ProbeCurriculum
        self.AttemptLedger = AttemptLedger
        self.RecoveryBlocked = RecoveryBlocked
        self.audit_recovery = audit_recovery
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def ledger(self, name='run', execution='formal'):
        return self.AttemptLedger(self.root/name, run_id='seed42', execution=execution)

    def test_per_control_persistence_and_partial_collect_is_unknown_after_restart(self):
        c = self.Curriculum(); ledger = self.ledger()
        ledger.begin_attempt(c, 512); controls(ledger, count=7)
        durable = self.ledger()
        self.assertEqual(durable.physics_samples_observed, 3584)
        self.assertEqual(durable.completed_ppo_updates, 0)
        with self.assertRaisesRegex(self.RecoveryBlocked, 'unknown'):
            self.audit_recovery(checkpoint(self.Curriculum(), self.ledger('base')), durable)

    def test_exact_checkpoint_boundary_and_complete_adam_can_resume(self):
        c = self.Curriculum(); ledger = self.ledger(); complete(ledger, c)
        state = checkpoint(c, ledger)
        audit = self.audit_recovery(state, self.ledger())
        self.assertEqual(audit['status'], 'eligible')
        self.assertEqual(audit['reward_state'], 'new_episode_no_half_trajectory')
        self.assertFalse(audit['physical_trajectory_continuous'])
        self.assertTrue(audit['new_episode'])
        self.assertEqual(ledger.physics_samples_observed, 24576)
        self.assertEqual(ledger.ppo_samples, 24576)
        self.assertEqual(ledger.completed_ppo_updates, 1)

    def test_unsaved_complete_updates_block_without_refund_or_counter_advance(self):
        c = self.Curriculum(); ledger = self.ledger()
        # Logic-only large history; real per-control fsync is covered separately.
        with patch('src.tasks.velocity.v04.recovery.os.fsync', return_value=None):
            for _ in range(100): complete(ledger, c)
            saved = checkpoint(c, ledger)
            for _ in range(37): complete(ledger, c)
        with self.assertRaisesRegex(self.RecoveryBlocked, 'unsaved complete'):
            self.audit_recovery(saved, ledger)
        self.assertEqual((c.iteration, ledger.completed_ppo_updates), (137, 137))

    def test_optimizer_started_is_blocked_even_with_zero_adam_callbacks(self):
        c = self.Curriculum(); ledger = self.ledger(); saved = checkpoint(c, ledger)
        ledger.begin_attempt(c, 512); controls(ledger)
        ledger.update_event('optimizer_started', {'expected_optimizer_steps':20, 'learning_state':learning(0)})
        with self.assertRaisesRegex(self.RecoveryBlocked, 'optimizer'):
            self.audit_recovery(saved, ledger)

    def test_full_collect_only_discard_preserves_physics_and_learning_budget(self):
        c = self.Curriculum(); ledger = self.ledger(); saved = checkpoint(c, ledger)
        ledger.begin_attempt(c, 512); controls(ledger)
        audit = self.audit_recovery(saved, self.ledger(), discard_collect_only=True)
        durable = self.ledger()
        self.assertEqual(audit['discarded_samples'], 24576)
        self.assertEqual((durable.physics_samples_observed, durable.discarded_samples, durable.ppo_samples),
                         (24576, 24576, 0))
        self.assertEqual((c.iteration, durable.completed_ppo_updates), (0, 0))

    def test_partial_collect_can_only_be_closed_by_live_synchronous_owner(self):
        c = self.Curriculum(); ledger = self.ledger()
        ledger.begin_attempt(c, 512); controls(ledger, count=3)
        with self.assertRaises(ValueError):
            self.ledger().close_collect_only(known_no_inflight=True, proof='not evidence after restart')
        ledger.close_collect_only(known_no_inflight=True, proof='synchronous env returned; no next step issued')
        self.assertEqual(self.ledger().discarded_samples, 1536)

    def test_tampered_or_truncated_durable_events_and_wrong_anchor_rejected(self):
        ledger = self.ledger(); c = self.Curriculum(); saved = checkpoint(c, ledger); complete(ledger, c)
        path = self.root/'run'/'ledger.jsonl'
        lines = path.read_text().splitlines(); event = json.loads(lines[-1])
        event['payload']['learning_state']['completed_ppo_updates'] = 100
        lines[-1] = json.dumps(event); path.write_text('\n'.join(lines)+'\n')
        with self.assertRaises(ValueError): self.ledger()
        other = self.ledger('other'); other.begin_attempt(self.Curriculum(), 512)
        with self.assertRaises(self.RecoveryBlocked):
            self.audit_recovery(saved, other)

    def test_probe_counts_start_at_zero_and_formal_audit_rejects_probe(self):
        c = self.ProbeCurriculum(); ledger = self.ledger(execution='probe')
        self.assertEqual((c.stage, c.iteration, c.stage_iteration, c.environment_stage_iteration),
                         ('S5', 0, 0, 400))
        complete(ledger, c); self.assertEqual(c.env_steps, 24576)
        with self.assertRaises(self.RecoveryBlocked):
            self.audit_recovery(checkpoint(c, ledger), ledger, execution='formal')
        with self.assertRaises(ValueError):
            self.Curriculum().load_state_dict(c.state_dict())

    def test_same_update_newer_checkpoint_blocks_earlier_rng_or_gate_state(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = self.AttemptLedger(directory, run_id='newest'); c = self.Curriculum()
            saved = checkpoint(c, ledger)
            ledger.record_checkpoint(Path(directory)/'old.pt', '1'*64, ledger.anchor())
            ledger.record_checkpoint(Path(directory)/'new.pt', '2'*64, ledger.anchor())
            with self.assertRaisesRegex(self.RecoveryBlocked, 'newer complete checkpoint'):
                self.audit_recovery(saved, ledger)

    def test_unknown_tail_explicitly_blocks_even_without_active_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = self.AttemptLedger(directory, run_id='unknown'); saved = checkpoint(self.Curriculum(), ledger)
            ledger.mark_unknown('power lost before final physical sample persistence')
            with self.assertRaisesRegex(self.RecoveryBlocked, 'unknown'):
                self.audit_recovery(saved, ledger)

    def test_checkpoint_reward_state_must_be_new_episode(self):
        c = self.Curriculum(); ledger = self.ledger(); complete(ledger, c)
        saved = checkpoint(c, ledger)
        for value in ('half_trajectory', None, True):
            bad = dict(saved, reward_state=value)
            with self.assertRaisesRegex(self.RecoveryBlocked, 'reward state'):
                self.audit_recovery(bad, self.ledger())

    def test_record_api_is_required_before_any_same_global_gate_can_be_closed(self):
        record = getattr(self.AttemptLedger(self.root/'gate', run_id='gate'), 'record_curriculum', None)
        self.assertIsNotNone(record, 'Consumed gate/history has no durable record_curriculum boundary')
        c = self.Curriculum(); record(c)
        self.assertEqual(self.AttemptLedger(self.root/'gate', run_id='gate').curriculum_state, c.state_dict())


class SchemaRejectionTests(unittest.TestCase):
    """Resume envelopes are rejected without needing the full v0.4 source closure."""

    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def envelope(self, **overrides):
        """A structurally complete schema-5 envelope with non-semantic dummy bodies.

        Identity/schema gates are what these tests exercise; they are evaluated
        before any model, Adam or ledger body is inspected.
        """
        from src.tasks.velocity.v04 import spec
        state = dict(contract=dict(code_version=spec.CODE_VERSION,
                                   checkpoint_schema=spec.SCHEMA_VERSION,
                                   task=spec.TASK, protocol_id=spec.PROTOCOL_ID,
                                   resume_mode='new_episode'),
                     ppo={}, model={}, optimizer={}, learning_state={}, transfer_metadata={},
                     curriculum={}, metadata={'seed':42}, num_envs=512,
                     rng={}, resume_mode='new_episode',
                     reward_state='new_episode_no_half_trajectory', execution='formal',
                     ledger={}, ledger_anchor={})
        state.update(overrides)
        return state

    def preload(self, state):
        from src.tasks.velocity.v04 import recovery
        path = self.root/'envelope.pt'
        self.torch.save(state, path)
        return recovery.preload_checkpoint(path, recovery.sha256(path))

    def test_schema4_v03_checkpoint_relabelled_as_schema5_is_rejected(self):
        legacy = Path('logs/a2_ideal_stairs_v03/seed42_20261113_100126/formal_seed42/'
                      'chunk_0004_n512/checkpoints/latest.pt')
        if not legacy.is_file():
            self.skipTest(f'historical v03 checkpoint unavailable: {legacy}')
        state = self.torch.load(legacy, map_location='cpu', weights_only=False)
        self.assertEqual(state['contract']['checkpoint_schema'], 4)
        from src.tasks.velocity.v04.runner import REWARD_STATE
        state = dict(state, reward_state=REWARD_STATE)
        self.assertRaises(ValueError, self.preload, state)

    def test_v04_identity_with_schema4_or_foreign_protocol_is_rejected(self):
        from src.tasks.velocity.v04 import spec
        for contract in (
                dict(self.envelope()['contract'], checkpoint_schema=4),
                dict(self.envelope()['contract'], code_version='0.3.0'),
                dict(self.envelope()['contract'], task='Unitree-A2-IdealHeightMap-Stairs-V03'),
                dict(self.envelope()['contract'], protocol_id='v03-stair-body-speed-v1')):
            with self.subTest(contract=contract):
                state = self.envelope(contract=contract)
                with self.assertRaisesRegex(ValueError, 'schema5'):
                    self.preload(state)
        self.assertEqual(spec.SCHEMA_VERSION, 5)

    def test_missing_or_extra_schema5_fields_are_rejected(self):
        for key in ('reward_state', 'execution', 'ledger_anchor', 'rng'):
            state = self.envelope(); state.pop(key)
            with self.subTest(missing=key):
                with self.assertRaisesRegex(ValueError, 'field contract'):
                    self.preload(state)
        for key in ('scope', 'reward_config', 'schema4_contract'):
            state = self.envelope(); state[key] = None
            with self.subTest(extra=key):
                with self.assertRaisesRegex(ValueError, 'field contract'):
                    self.preload(state)


if __name__ == '__main__': unittest.main()
