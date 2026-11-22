"""Strict durable accounting: unfinished learning cannot be priced as complete."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch



def learning(n):
    return dict(optimizer_initialized=n > 0, completed_ppo_updates=n, optimizer_steps=20*n)


def controls(ledger, n=258, count=48):
    for step in range(1, count+1):
        ledger.control_step(dict(step_index=step, num_envs=n, physics_samples_increment=n))


def complete(ledger, c, n=258):
    ledger.begin_attempt(c, n)
    controls(ledger, n)
    ledger.update_event('optimizer_started', {'expected_optimizer_steps':20, 'learning_state':learning(c.iteration)})
    for step in range(1,21):
        state=learning(c.iteration)
        state.update(optimizer_initialized=True, optimizer_steps=20*c.iteration+step)
        ledger.update_event('optimizer_step_completed', {'learning_state':state})
    ledger.update_event('update_completed', {'learning_state':learning(c.iteration+1),'optimizer_steps_in_update':20})
    c.updated(n)
    ledger.complete_attempt(c, learning(c.iteration))


def checkpoint(c, ledger):
    return dict(curriculum=c.state_dict(), learning_state=learning(c.iteration),
                ledger=ledger.state_dict(), ledger_anchor=ledger.anchor(), execution=c.execution)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        global Curriculum, ProbeCurriculum, AttemptLedger, RecoveryBlocked, audit_recovery
        try:
            from src.tasks.velocity.v03.runner import Curriculum, ProbeCurriculum
            from src.tasks.velocity.v03.recovery import AttemptLedger, RecoveryBlocked, audit_recovery
        except ImportError as error:
            self.fail(f"Missing V4r behavior: {error}")
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def ledger(self, name='run', execution='formal'):
        return AttemptLedger(self.root/name, run_id='seed42', execution=execution)
    def test_per_control_persistence_and_partial_collect_is_unknown_after_restart(self):
        c=Curriculum();ledger=self.ledger();ledger.begin_attempt(c,258);controls(ledger,count=7)
        durable=self.ledger()
        self.assertEqual(durable.physics_samples_observed,1806)
        self.assertEqual(durable.completed_ppo_updates,0)
        with self.assertRaisesRegex(RecoveryBlocked,'unknown'):
            audit_recovery(checkpoint(Curriculum(),self.ledger('base')),durable)
    def test_exact_checkpoint_boundary_and_complete_adam_can_resume(self):
        c=Curriculum();ledger=self.ledger();complete(ledger,c)
        state=checkpoint(c,ledger);audit=audit_recovery(state,self.ledger())
        self.assertEqual(audit['status'],'eligible')
        self.assertEqual(ledger.physics_samples_observed,12384)
        self.assertEqual(ledger.ppo_samples,12384)
        self.assertEqual(ledger.completed_ppo_updates,1)
    def test_unsaved_complete_updates_block_without_refund_or_counter_advance(self):
        c=Curriculum();ledger=self.ledger()
        # Logic-only large history; real per-control fsync is covered separately.
        with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):
            for _ in range(100):complete(ledger,c)
            saved=checkpoint(c,ledger)
            for _ in range(37):complete(ledger,c)
        with self.assertRaisesRegex(RecoveryBlocked,'unsaved complete'):
            audit_recovery(saved,ledger)
        self.assertEqual((c.iteration,ledger.completed_ppo_updates),(137,137))
    def test_optimizer_started_is_blocked_even_with_zero_adam_callbacks(self):
        c=Curriculum();ledger=self.ledger();saved=checkpoint(c,ledger)
        ledger.begin_attempt(c,258);controls(ledger)
        ledger.update_event('optimizer_started',{'expected_optimizer_steps':20,'learning_state':learning(0)})
        with self.assertRaisesRegex(RecoveryBlocked,'optimizer'):
            audit_recovery(saved,ledger)
    def test_full_collect_only_discard_preserves_physics_and_learning_budget(self):
        c=Curriculum();ledger=self.ledger();saved=checkpoint(c,ledger)
        ledger.begin_attempt(c,258);controls(ledger)
        audit=audit_recovery(saved,self.ledger(),discard_collect_only=True)
        durable=self.ledger()
        self.assertEqual(audit['discarded_samples'],12384)
        self.assertEqual((durable.physics_samples_observed,durable.discarded_samples,durable.ppo_samples),(12384,12384,0))
        self.assertEqual((c.iteration,durable.completed_ppo_updates),(0,0))
    def test_partial_collect_can_only_be_closed_by_live_synchronous_owner(self):
        c=Curriculum();ledger=self.ledger();ledger.begin_attempt(c,258);controls(ledger,count=3)
        with self.assertRaises(ValueError):self.ledger().close_collect_only(known_no_inflight=True,proof='not evidence after restart')
        ledger.close_collect_only(known_no_inflight=True,proof='synchronous env returned; no next step issued')
        self.assertEqual(self.ledger().discarded_samples,774)
    def test_tampered_or_truncated_durable_events_and_wrong_anchor_rejected(self):
        ledger=self.ledger();c=Curriculum();saved=checkpoint(c,ledger);complete(ledger,c)
        path=self.root/'run'/'ledger.jsonl'
        lines=path.read_text().splitlines();event=json.loads(lines[-1])
        event['payload']['learning_state']['completed_ppo_updates']=100
        lines[-1]=json.dumps(event);path.write_text('\n'.join(lines)+'\n')
        with self.assertRaises(ValueError):self.ledger()
        other=self.ledger('other');other.begin_attempt(Curriculum(),512)
        with self.assertRaises(RecoveryBlocked):audit_recovery(saved,other)
    def test_probe_counts_start_at_zero_and_formal_audit_rejects_probe(self):
        c=ProbeCurriculum();ledger=self.ledger(execution='probe')
        self.assertEqual((c.stage,c.iteration,c.stage_iteration,c.environment_stage_iteration),('S5',0,0,400))
        complete(ledger,c);self.assertEqual(c.env_steps,12384)
        with self.assertRaises(RecoveryBlocked):audit_recovery(checkpoint(c,ledger),ledger,execution='formal')


class ExtraAuditTests(unittest.TestCase):
    def test_same_update_newer_checkpoint_blocks_earlier_rng_or_gate_state(self):
        from src.tasks.velocity.v03.recovery import AttemptLedger,RecoveryBlocked,audit_recovery
        from src.tasks.velocity.v03.runner import Curriculum
        with tempfile.TemporaryDirectory() as directory:
            ledger=AttemptLedger(directory,run_id='newest');c=Curriculum();saved=checkpoint(c,ledger)
            ledger.record_checkpoint(Path(directory)/'old.pt','1'*64,ledger.anchor())
            ledger.record_checkpoint(Path(directory)/'new.pt','2'*64,ledger.anchor())
            with self.assertRaisesRegex(RecoveryBlocked,'newer complete checkpoint'):audit_recovery(saved,ledger)
    def test_unknown_tail_explicitly_blocks_even_without_active_attempt(self):
        from src.tasks.velocity.v03.recovery import AttemptLedger,RecoveryBlocked,audit_recovery
        from src.tasks.velocity.v03.runner import Curriculum
        with tempfile.TemporaryDirectory() as directory:
            ledger=AttemptLedger(directory,run_id='unknown');saved=checkpoint(Curriculum(),ledger)
            ledger.mark_unknown('power lost before final physical sample persistence')
            with self.assertRaisesRegex(RecoveryBlocked,'unknown'):audit_recovery(saved,ledger)

class CurriculumDurabilityTests(unittest.TestCase):
    def test_record_api_is_required_before_any_same_global_gate_can_be_closed(self):
        from src.tasks.velocity.v03.recovery import AttemptLedger
        from src.tasks.velocity.v03.runner import Curriculum
        with tempfile.TemporaryDirectory() as directory:
            ledger=AttemptLedger(directory,run_id='gate')
            record=getattr(ledger,'record_curriculum',None)
            self.assertIsNotNone(record,'Consumed gate/history has no durable record_curriculum boundary')
            c=Curriculum();record(c)
            self.assertEqual(AttemptLedger(directory,run_id='gate').curriculum_state,c.state_dict())

class GateCrashTests(unittest.TestCase):
    def test_same_global_recorded_gate_blocks_presave_candidate_and_new_collect(self):
        from unittest.mock import patch
        from src.tasks.velocity.v03.recovery import AttemptLedger,RecoveryBlocked,audit_recovery
        from src.tasks.velocity.v03.runner import Curriculum
        from test_v03_evaluation import make_gate_fixture
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=AttemptLedger(root/'ledger',run_id='gatecrash');c=Curriculum()
            # These 200 synthetic ledger updates test accounting, not hardware power-loss.
            # Per-control real fsync is exercised by LedgerTests above.
            with patch('src.tasks.velocity.v03.recovery.os.fsync',return_value=None):
                for _ in range(200):complete(ledger,c)
            saved=checkpoint(c,ledger)
            summary,case=make_gate_fixture(root/'evidence',iteration=200,stage_iteration=200)
            c.validate(summary,'1'*64,case)
            with self.assertRaises(ValueError):ledger.begin_attempt(c,258)
            ledger.record_curriculum(c)
            with self.assertRaisesRegex(RecoveryBlocked,'newer durable curriculum'):audit_recovery(saved,ledger)
            current=checkpoint(c,ledger)
            self.assertEqual(audit_recovery(current,ledger)['status'],'eligible')
            reloaded=AttemptLedger(root/'ledger',run_id='gatecrash')
            self.assertEqual(reloaded.curriculum_state,c.state_dict())


if __name__=='__main__':unittest.main()
