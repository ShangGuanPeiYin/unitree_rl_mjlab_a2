"""CPU-only v0.5-r2 schema-7 recovery tests.

Covered (SDS 7.3/7.5/7.7):

* the exact 14-key schema-7 envelope, the r2 identity, architecture and the
  five-key route observation contract;
* rejection of an r1 full checkpoint (schema 6 / 208 Critic / r1 identity), of
  ``schema <= 6`` or unlabelled files, of the old and r2 smoke containers, of
  the r1 initialization container, of unknown tails and of partial Adam state;
* the r2 model body must carry ``actor.route_fusion.weight/bias`` and the
  211-input ``critic.0``; a missing/short route tensor is refused;
* ``total_budget``/``update_cap`` come from the single ``spec.mode_budget``
  source (pilot 2500, formal 30000) and :func:`audit_budget` is the fourth
  (resume) consumer of the same table;
* the evaluation transaction ``planned -> result_committed -> consumed`` is
  exactly-once and replaying it never consumes a streak twice.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from src.tasks.velocity.v05_r2 import recovery, spec
from src.tasks.velocity.v05_r2.model import V05R2ActorCritic
from src.tasks.velocity.v05_r2.recovery import (AttemptLedger, CHECKPOINT_KEYS,
                                                ForeignContainerError, LegacySchemaError,
                                                audit_budget, build_checkpoint,
                                                genesis_curriculum, learning_state,
                                                preload_checkpoint, reject_foreign_container,
                                                route_evidence_contract, source_block,
                                                stage_budgets, task_key, total_budget,
                                                updated_curriculum, validate_checkpoint,
                                                validate_learning, validate_route_model_state)


class RngEnv:
    def __init__(self):
        self.rng = torch.Generator().manual_seed(7)
        self.num_envs = 8


def digest(text):
    import hashlib
    return hashlib.sha256(str(text).encode()).hexdigest()


def model_state(model=None):
    """A genuine r2 state_dict (including the two route tensors and the 211 critic)."""
    torch.manual_seed(0)
    model = V05R2ActorCritic() if model is None else model
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def fresh_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=spec.PPO_INITIAL_LR).state_dict()


def make_checkpoint(*, mode='formal', seed=42, model=None, ledgers=None):
    model = V05R2ActorCritic() if model is None else model
    learning = learning_state(0, num_envs=8)
    curriculum = genesis_curriculum()
    ledger = ledgers or dict(tasks={}, results=[], collapse_streak=0, gate_streak=0,
                             anchor=dict(seq=0, digest='a' * 64),
                             qualified_update=None, pending_transition=None)
    state = build_checkpoint(
        model_state=model_state(model), optimizer_state=fresh_optimizer(model),
        normalizer={}, learning=learning, rng=recovery.capture_rng(RngEnv()),
        curriculum=curriculum, ledger=ledger,
        evaluation=dict(consumed=[], latest_diagnostic_update=None),
        source=source_block(mode=mode, seed=seed, device='cpu'),
        mode=mode, seed=seed, device='cpu')
    return state, model


class EnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_schema7_has_the_exact_fourteen_top_level_keys(self):
        self.assertEqual(len(CHECKPOINT_KEYS), 14)
        self.assertEqual(CHECKPOINT_KEYS, frozenset({
            'schema', 'identity', 'contract_sha256', 'model', 'optimizer', 'normalizer',
            'learning', 'rng', 'curriculum', 'ledger', 'evaluation', 'source', 'recipe',
            'resume_mode'}))
        state, _ = make_checkpoint()
        self.assertEqual(set(state), set(CHECKPOINT_KEYS))
        self.assertEqual(state['schema'], 7)
        self.assertEqual(state['identity']['task'], 'Unitree-A2-IdealHeightMap-Stairs-V05R2')
        self.assertEqual(state['identity']['recipe'], '0.5-r2')
        self.assertEqual(state['identity']['code_version'], '0.5.1')
        self.assertEqual(state['identity']['protocol'], 'v05-r2-stair-climb-first-v1')
        self.assertEqual(state['identity']['sds_revision'], '1.0')
        self.assertEqual(state['resume_mode'], 'new_episode')
        self.assertEqual(validate_checkpoint(state), state)

    def test_identity_block_and_architecture_are_r2(self):
        self.assertEqual(spec.ARCHITECTURE, 'v05-route-state-v1')
        self.assertEqual(recovery.architecture(), 'v05-route-state-v1')
        self.assertTrue(recovery.validate_observation_contract())
        self.assertEqual(set(spec.OBSERVATION), {'proprio', 'history', 'heightmap',
                                                 'privileged', 'route_error'})
        self.assertEqual(spec.OBSERVATION['route_error'], [3])

    def test_route_evidence_contract_names_the_checkpoint_evidence_split(self):
        contract = route_evidence_contract()
        self.assertEqual(contract['route_key'], 'route_error')
        self.assertEqual(contract['route_shape'], [3])
        self.assertFalse(contract['checkpoint_persists_rollout'])
        self.assertEqual(contract['terminal_route_evidence'],
                         'evaluation_npz_and_episode_route_final')
        self.assertIn('actor.route_fusion.weight', contract['model_keys'])
        self.assertIn('critic.0.weight', contract['model_keys'])

    def test_model_body_must_carry_the_route_bypass_and_211_critic(self):
        state, model = make_checkpoint()
        validate_route_model_state(state['model'])
        self.assertEqual(tuple(state['model']['actor.route_fusion.weight'].shape), (512, 3))
        self.assertEqual(tuple(state['model']['actor.route_fusion.bias'].shape), (512,))
        self.assertEqual(tuple(state['model']['critic.0.weight'].shape), (512, 211))
        self.assertIn('critic_normalizer.mean', state['model'])
        self.assertEqual(state['model']['critic_normalizer.mean'].numel(), 80)

        broken = deepcopy(state)
        del broken['model']['actor.route_fusion.weight']
        with self.assertRaises(ValueError) as caught:
            validate_checkpoint(broken)
        self.assertIn('route_fusion', str(caught.exception))

        short = deepcopy(state)
        short['model']['critic.0.weight'] = torch.zeros(512, 208)
        with self.assertRaises(ValueError) as caught:
            validate_checkpoint(short)
        self.assertIn('critic.0.weight', str(caught.exception))

    def test_full_model_body_is_validated_when_a_factory_is_available(self):
        state, _ = make_checkpoint()
        validate_checkpoint(state, model_factory=V05R2ActorCritic)
        # Structural tampering (a wrong shape / missing key) is refused; values
        # are not compared against a reference because the contract binds the
        # source closure, not a weight snapshot.
        tampered = deepcopy(state)
        tampered['model']['actor.mlp.0.bias'] = torch.zeros(513)
        with self.assertRaises(ValueError):
            validate_checkpoint(tampered, model_factory=V05R2ActorCritic)


class RejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.state, _ = make_checkpoint()

    def test_r1_full_checkpoint_is_rejected(self):
        r1 = deepcopy(self.state)
        r1['schema'] = 6
        r1['identity'] = dict(task='Unitree-A2-IdealHeightMap-Stairs-V05',
                              recipe='0.5-r1', code_version='0.5.0',
                              protocol='v05-stair-climb-first-v1', sds_revision='1.1')
        r1['recipe'] = {'recipe': '0.5-r1'}
        r1['model']['critic.0.weight'] = torch.zeros(512, 208)
        del r1['model']['actor.route_fusion.weight']
        del r1['model']['actor.route_fusion.bias']
        with self.assertRaises(LegacySchemaError):
            validate_checkpoint(r1)
        # Even with the schema field rewritten, the missing route tensors and the
        # r1 identity are still refused (no strict=False path).
        forged = deepcopy(r1)
        forged['schema'] = 7
        with self.assertRaises((ValueError, LegacySchemaError)):
            validate_checkpoint(forged)

    def test_unlabelled_and_schema_le_6_are_legacy_errors(self):
        for schema in (None, 5, 6, True, False):
            with self.subTest(schema=schema):
                legacy = deepcopy(self.state)
                if schema is None:
                    del legacy['schema']
                else:
                    legacy['schema'] = schema
                with self.assertRaises(LegacySchemaError):
                    validate_checkpoint(legacy)
        with self.assertRaises(LegacySchemaError):
            validate_checkpoint({'contract': {}, 'model': {}})

    def test_old_smoke_containers_are_rejected(self):
        for marker in ('v05-smoke-state-v1', 'v05-r2-smoke-state-v1'):
            with self.subTest(marker=marker):
                container = {'container': marker, 'schema': 7, 'formal_resumable': False}
                with self.assertRaises(ForeignContainerError):
                    reject_foreign_container(container)
                with self.assertRaises(ForeignContainerError):
                    validate_checkpoint(container)
                wrapped = deepcopy(self.state)
                wrapped['trainer'] = marker
                with self.assertRaises(ForeignContainerError):
                    validate_checkpoint(wrapped)
        with self.assertRaises(ForeignContainerError):
            reject_foreign_container({'schema': 7, 'formal_resumable': False})

    def test_r1_initialization_container_is_not_a_resume(self):
        for container in ({'scope': 'initialization', 'schema': 7},
                          {'scope': 'initialization', 'model': {}, 'contract': {}}):
            with self.subTest(container=sorted(container)):
                with self.assertRaises(ForeignContainerError):
                    validate_checkpoint(container)

    def test_unknown_tail_and_partial_adam_are_blocked(self):
        extra = deepcopy(self.state)
        extra['rollout'] = {'route_error': torch.zeros(4, 3)}
        with self.assertRaises(ValueError) as caught:
            validate_checkpoint(extra)
        self.assertIn('missing=', str(caught.exception))
        self.assertIn('extra=', str(caught.exception))

        missing = deepcopy(self.state)
        del missing['ledger']
        with self.assertRaises(ValueError):
            validate_checkpoint(missing)

        model = V05R2ActorCritic()
        state, _ = make_checkpoint(model=model)
        parameters = list(model.parameters())
        partial = deepcopy(state)
        index = 0
        parameter = parameters[index]
        partial['optimizer']['state'][index] = dict(
            step=torch.tensor(20.), exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter))
        with self.assertRaises(ValueError) as caught:
            validate_checkpoint(partial, parameters=parameters)
        self.assertIn('Adam', str(caught.exception))

    def test_unknown_pending_requirement_and_resume_mode_are_blocked(self):
        state, _ = make_checkpoint()
        state['curriculum']['pending_transition'] = dict(
            from_stage='F0', to_stage='P1', at_update=0, requires='manual',
            at_checkpoint='b' * 64)
        with self.assertRaises(ValueError):
            validate_checkpoint(state)
        other = deepcopy(self.state)
        other['resume_mode'] = 'continue'
        with self.assertRaises(ValueError):
            validate_checkpoint(other)

    def test_preload_requires_the_exact_sha(self):
        state, _ = make_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'schema7.pt'
            recovery.atomic_torch_save(path, state)
            actual = recovery.sha256(path)
            self.assertEqual(preload_checkpoint(path, actual)['schema'], 7)
            with self.assertRaises(ValueError):
                preload_checkpoint(path, 'f' * 64)
            with self.assertRaises(ValueError):
                preload_checkpoint(path, None)


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mode_budget_is_the_single_source(self):
        self.assertEqual(spec.mode_budget('pilot')['total'], 2500)
        self.assertEqual(spec.mode_budget('pilot')['stages'], {'F0': 2000, 'P1': 500})
        self.assertEqual(spec.mode_budget('formal')['total'], 30000)
        self.assertEqual(sum(spec.mode_budget('formal')['stages'].values()), 30000)
        self.assertEqual(total_budget('pilot'), 2500)
        self.assertEqual(total_budget('formal'), 30000)
        self.assertEqual(stage_budgets('pilot')['P1'], 500)
        self.assertEqual(stage_budgets('formal')['P1'], 1000)
        with self.assertRaises(ValueError):
            spec.mode_budget('unknown')

    def test_pilot_and_formal_curricula_validate_against_their_own_caps(self):
        pilot = dict(stage='P1', stage_iteration=500,
                     spent={name: 0 for name in spec.STAGE_ORDER},
                     streak=dict(gate=0, collapse=0), pending_transition=None)
        pilot['spent']['F0'] = 2000
        pilot['spent']['P1'] = 500
        audit = audit_budget(learning_state(2500, num_envs=8), pilot, mode='pilot')
        self.assertEqual(audit['total_budget'], 2500)
        self.assertEqual(audit['update_cap'], 2500)
        self.assertEqual(audit['effective_cap_source'], 'mode_budget')
        self.assertEqual(audit['update'], 2500)
        self.assertEqual(audit['total_spent'], 2500)

        over = deepcopy(pilot)
        over['stage_iteration'] = 501
        over['spent']['P1'] = 501
        with self.assertRaises(ValueError):
            audit_budget(learning_state(2501, num_envs=8), over, mode='pilot')
        # The formal table allows P1=1000, so the same 501 is structurally legal
        # only when the learning boundary and the spent sum agree.
        formal_ok = deepcopy(over)
        formal_ok['spent']['F0'] = 2000
        audit_budget(learning_state(2501, num_envs=8), formal_ok, mode='formal')

    def test_resume_audit_rejects_update_above_cap_and_spent_mismatch(self):
        formal = genesis_curriculum()
        formal['spent']['F0'] = 2000
        formal['stage'] = 'F0'
        formal['stage_iteration'] = 2000
        with self.assertRaises(ValueError):
            audit_budget(learning_state(2001, num_envs=8), formal, mode='pilot')
        # Three places agree (2500 cap) but the spent table does not sum to the
        # learning boundary: the resume audit must block instead of accepting.
        pilot = genesis_curriculum()
        pilot['stage'] = 'P1'
        pilot['stage_iteration'] = 500
        pilot['spent']['F0'] = 2000
        pilot['spent']['P1'] = 499
        with self.assertRaises(ValueError):
            audit_budget(learning_state(2500, num_envs=8), pilot, mode='pilot')

    def test_updated_curriculum_uses_the_mode_cap(self):
        pilot = genesis_curriculum()
        pilot['stage'] = 'P1'
        pilot['stage_iteration'] = 500
        pilot['spent']['F0'] = 2000
        pilot['spent']['P1'] = 500
        with self.assertRaises(ValueError):
            updated_curriculum(pilot, num_envs=8, mode='pilot')
        formal = genesis_curriculum()
        formal['stage'] = 'P1'
        formal['stage_iteration'] = 500
        formal['spent']['F0'] = 2000
        formal['spent']['P1'] = 500
        advanced = updated_curriculum(formal, num_envs=8, mode='formal')
        self.assertEqual(advanced['stage_iteration'], 501)

    def test_schema7_learning_block_is_strict(self):
        state, _ = make_checkpoint()
        self.assertEqual(state['learning'], dict(update=0, optimizer_step=0,
                                                 control_samples=0, physics_substeps=0))
        bad = deepcopy(state)
        bad['learning']['physics_substeps'] = 4
        with self.assertRaises(ValueError):
            validate_learning(bad['learning'])


class EvaluationTransactionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = AttemptLedger(self.tmp.name, run_id='r2-unit', mode='formal')
        self.checkpoint_sha = 'a' * 64
        self.case_sha = 'b' * 64
        self.report = Path(self.tmp.name) / 'report.json'
        self.report.write_text('{"kind": "validation"}')
        self.report_sha = recovery.sha256(self.report)
        self.key = task_key(run_id='r2-unit', mode='formal', stage='F0', global_update=0,
                            kind='validation', checkpoint_sha256=self.checkpoint_sha,
                            case_sha256=self.case_sha)

    def _plan(self):
        return self.ledger.plan_evaluation(
            run_id='r2-unit', mode='formal', stage='F0', global_update=0,
            kind='validation', checkpoint_sha256=self.checkpoint_sha,
            case_sha256=self.case_sha, planned_wall=1.)

    def _commit(self):
        return self.ledger.commit_evaluation_result(
            self.key, result_path=str(self.report), result_sha256=self.report_sha,
            report_path=str(self.report), report_sha256=self.report_sha,
            evidence_sha256=['d' * 64], summary_sha256='e' * 64, committed_wall=2.)

    def test_transaction_is_exactly_once_and_idempotent(self):
        first = self._plan()
        self.assertEqual(first['state'], 'planned')
        again = self._plan()
        self.assertEqual(again['state'], 'planned')
        self.assertEqual(len(self.ledger.tasks), 1)
        self.assertEqual(self.ledger.tasks[self.key]['state'], 'planned')
        with self.assertRaises(ValueError):
            self.ledger.consume_evaluation(self.key, outcome='passed', passed=True,
                                           collapse_hit=False)
        self._commit()
        self.assertEqual(self.ledger.tasks[self.key]['state'], 'result_committed')
        consumed = self.ledger.consume_evaluation(self.key, outcome='passed', passed=True,
                                                  collapse_hit=False, consumed_wall=3.)
        self.assertEqual(consumed['streak']['gate'], 1)
        self.assertEqual(len(self.ledger.results), 1)
        replay = self.ledger.consume_evaluation(self.key, outcome='passed', passed=True,
                                                collapse_hit=False, consumed_wall=4.)
        self.assertTrue(replay['replayed'])
        self.assertEqual(replay['streak']['gate'], 1)
        self.assertEqual(len(self.ledger.results), 1)
        self.assertEqual(self.ledger.gate_streak, 1)
        self.assertEqual(self.ledger.checkpoint_evaluation()['consumed'], [self.key])
        # The durable file has one header plus exactly three events (plan, commit,
        # consume); the replay wrote nothing.
        lines = Path(self.ledger.path).read_text().splitlines()
        self.assertEqual(len(lines), 4)

    def test_streak_change_and_consumption_are_one_ledger_event(self):
        self._plan()
        self._commit()
        self.ledger.consume_evaluation(self.key, outcome='failed', passed=False,
                                       collapse_hit=False)
        events = [json.loads(line) for line in
                  Path(self.ledger.path).read_text().splitlines()[1:]]
        self.assertEqual([event['event'] for event in events],
                         ['evaluation_planned', 'evaluation_result_committed',
                          'evaluation_consumed'])
        consumed = events[-1]['payload']
        self.assertEqual(consumed['streak_after'], {'gate': 0, 'collapse': 0})
        self.assertFalse(consumed['passed'])

    def test_streak_derivation_is_the_single_source(self):
        curriculum = genesis_curriculum()
        record = dict(kind='validation', stage='F0', global_update=100,
                      checkpoint_sha256='a' * 64)
        streak, pending = recovery.next_streak_state(curriculum, record, passed=True,
                                                     collapse_hit=False)
        self.assertEqual(streak['gate'], 1)
        self.assertIsNone(pending)
        streak, pending = recovery.next_streak_state(
            dict(curriculum, streak=dict(gate=1, collapse=0)), record, passed=True,
            collapse_hit=False)
        self.assertEqual(streak['gate'], 2)
        self.assertEqual(pending['to_stage'], 'P1')
        self.assertEqual(pending['at_checkpoint'], 'a' * 64)
        pilot_record = dict(record, stage='P1')
        streak, pending = recovery.next_streak_state(
            dict(curriculum, streak=dict(gate=1, collapse=0)), pilot_record, passed=True,
            collapse_hit=False, mode='pilot')
        self.assertEqual(streak['gate'], 2)
        self.assertIsNone(pending)  # pilot P1 never advances early


if __name__ == '__main__':
    unittest.main()
