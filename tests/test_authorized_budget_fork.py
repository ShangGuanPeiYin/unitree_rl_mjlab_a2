"""Explicit authorized fork: paid budget is not the retained model lineage."""
import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

AUTH=Path('logs/a2_ideal_stairs_v03/implementation/authorized_recovery_2700/explicit_authorization.json')

class ForkTests(unittest.TestCase):
    def setUp(self):
        try:self.api=importlib.import_module('scripts.tools.authorized_budget_fork')
        except ImportError as error:self.fail(f'Authorized dual-counter fork unavailable: {error}')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def cursor(self):return self.api.load_authorized_budget(AUTH)
    def test_four_new_updates_gate_on_budget2800_not_lineage2800(self):
        c=self.cursor();self.assertEqual((c.iteration,c.model_iteration,c.remaining_stage_updates),(2796,2700,204))
        for _ in range(4):c.updated(512)
        self.assertTrue(c.validation_due);self.assertEqual((c.iteration,c.model_iteration,c.model_stage_iteration),(2800,2704,2704))
        self.assertTrue(c.checkpoint_due)
        with self.assertRaises(RuntimeError):c.updated(512)
    def test_real_raw_first_gate_is_model2704_with_separate_scheduled_budget2800(self):
        from test_v03_evaluation import make_gate_fixture
        c=self.cursor()
        for _ in range(4):c.updated(512)
        with self.api.fork_evidence_context(c):
            summary,case=make_gate_fixture(self.root/'raw',iteration=2704,stage_iteration=2704,num_envs=512)
        self.assertEqual(summary['iteration'],2704);self.assertEqual(summary['scheduled_budget_iteration'],2800)
        self.assertFalse(c.validate(summary,'1'*64,case));self.assertEqual(c.consecutive_passes,1)
        self.assertEqual(c.history[-1]['iteration'],2800);self.assertEqual(c.history[-1]['summary']['iteration'],2704)
        restored=self.api.BudgetCursor();restored.load_state_dict(c.state_dict());self.assertEqual(restored.state_dict(),c.state_dict())
        with self.assertRaises(ValueError):c.validate(summary,'1'*64,case)
    def test_missing_fork_binding_and_fake_budget_model_identity_are_rejected(self):
        from test_v03_evaluation import make_gate_fixture
        c=self.cursor()
        for _ in range(4):c.updated(512)
        plain,case=make_gate_fixture(self.root/'plain',iteration=2704,stage_iteration=2704,num_envs=512)
        with self.assertRaises(ValueError):c.validate(plain,'1'*64,case)
        with self.api.fork_evidence_context(c):
            wrong,case=make_gate_fixture(self.root/'wrong',iteration=2800,stage_iteration=2800,num_envs=512)
        with self.assertRaises(ValueError):c.validate(wrong,'1'*64,case)
    def test_new_ledger_translates_callbacks_without_changing_actual_model_learning(self):
        c=self.cursor();local=self.api.LocalProgress();ledger=self.api.ForkLedger(self.root/'ledger',run_id='fork-test')
        ledger.record_origin({'source_checkpoint_sha256':'a'*64,'model_base':2700,'budget_base':2796,'lost_updates':96})
        ledger.begin_attempt(local,512)
        for i in range(1,49):ledger.control_step(dict(step_index=i,num_envs=512,physics_samples_increment=512))
        actual=dict(optimizer_initialized=True,completed_ppo_updates=2700,optimizer_steps=54000)
        ledger.update_event('optimizer_started',dict(learning_state=actual,expected_optimizer_steps=20))
        for step in range(1,21):
            ledger.update_event('optimizer_step_completed',dict(learning_state=dict(actual,optimizer_steps=54000+step)))
        actual=dict(optimizer_initialized=True,completed_ppo_updates=2701,optimizer_steps=54020)
        ledger.update_event('update_completed',dict(learning_state=actual,optimizer_steps_in_update=20))
        c.updated(512);local.updated(512);ledger.complete_actual(local,actual)
        self.assertEqual(actual['completed_ppo_updates'],2701);self.assertEqual(c.iteration,2797)
        self.assertEqual((ledger.completed_ppo_updates,ledger.physics_samples_observed),(1,24576))
        loaded=self.api.ForkLedger(self.root/'ledger',run_id='fork-test');self.assertEqual(loaded.state_dict(),ledger.state_dict())
    def test_s1_last_budget_gate_then_same_sha_extra_before_advance(self):
        from test_v03_evaluation import make_gate_fixture
        c=self.cursor()
        for target in (2800,3000):
            while c.iteration<target:c.updated(512)
            with self.api.fork_evidence_context(c):
                summary,case=make_gate_fixture(self.root/str(target),iteration=c.model_iteration,stage_iteration=c.model_stage_iteration,num_envs=512)
            c.validate(summary,'1'*64,case)
        self.assertEqual((c.iteration,c.model_iteration,c.remaining_stage_updates),(3000,2904,0))
        self.assertTrue(c.needs_perturbed)
        with self.assertRaises(RuntimeError):c.updated(512)
        with self.api.fork_evidence_context(c):
            summary,case=make_gate_fixture(self.root/'extra',kind='perturbed',iteration=2904,stage_iteration=2904,num_envs=512)
        c.validate_perturbed(summary,'1'*64,case);c.advance()
        self.assertEqual((c.stage,c.stage_iteration,c.model_stage_iteration,c.iteration,c.model_iteration),('S2',0,0,3000,2904))
    def test_complete_source2700_state_restored_but_budget_stays2796(self):
        a=self.api;source=a.load_source(AUTH);c=self.cursor();ledger=a.ForkLedger(self.root/'ledger',run_id='restore')
        ledger.record_origin(a.origin_record(AUTH,source))
        class Env:
            num_envs=512;device='cpu'
            def __init__(self):self.rng=torch.Generator().manual_seed(123)
            def set_stage(self,*args,**kwargs):pass
            def reset(self):random.random();np.random.rand();torch.rand(1);torch.rand(1,generator=self.rng);return {},{}
        # CPU-only projection is confined to this test: product restore remains exact.
        original_restore=a.restore_rng
        cpu_environment=torch.Generator().manual_seed(777).get_state()
        seen=[]
        def cpu_restore(state,env):
            seen.append(copy.deepcopy(state))
            projected=copy.deepcopy(state);projected['cuda']=[];projected['environment']=cpu_environment
            return original_restore(projected,env)
        domain=patch.object(a,'restore_rng',side_effect=cpu_restore);domain.start();self.addCleanup(domain.stop)
        env=Env();runner=a.create_runner(source,env,ledger)
        self.assertTrue(torch.equal(seen[0]['environment'],source['rng']['environment']))
        self.assertEqual(len(seen[0]['cuda']),len(source['rng']['cuda']))
        self.assertEqual(runner.learning_state,source['learning_state']);self.assertEqual(len(runner.optimizer.state),43)
        actual_optimizer=runner.optimizer.state_dict()
        self.assertEqual(actual_optimizer['param_groups'],source['optimizer']['param_groups'])
        for key,moments in source['optimizer']['state'].items():
            for name,value in moments.items():self.assertTrue(torch.equal(actual_optimizer['state'][key][name],value),(key,name))
        expected=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=env.rng))
        a.restore_rng(source['rng'],env)
        actual=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=env.rng))
        self.assertEqual(actual[:2],expected[:2]);self.assertTrue(torch.equal(actual[2],expected[2]));self.assertTrue(torch.equal(actual[3],expected[3]))
        path=self.root/'fork.pt';sha=a.save_fork(path,runner,c,ledger,AUTH)
        restored=a.preload_fork(path,sha,AUTH)
        self.assertEqual(restored['learning_state']['completed_ppo_updates'],2700)
        self.assertEqual(restored['budget_curriculum']['iteration'],2796)
        self.assertEqual(restored['container_schema'],'authorized-budget-fork-v1')
        for k,v in source['model'].items():self.assertTrue(torch.equal(v,restored['model'][k]),k)
        bad=copy.deepcopy(restored);bad['learning_state']['completed_ppo_updates']=2796;torch.save(bad,path)
        with self.assertRaises(ValueError):a.preload_fork(path,a.sha256(path),AUTH)

    def test_partial_nan_and_bad_adam_callback_stop_without_budget_charge(self):
        for kind in ('nan','adam'):
            c=self.cursor();local=self.api.LocalProgress();ledger=self.api.ForkLedger(self.root/kind,run_id=kind)
            ledger.record_origin({'source':'test-domain'})
            class Broken:
                learning_state=dict(optimizer_initialized=True,completed_ppo_updates=2700,optimizer_steps=54000)
                def collect(inner):
                    ledger.control_step(dict(step_index=1,num_envs=512,physics_samples_increment=512))
                    if kind=='nan':raise FloatingPointError('retained NaN')
                    for i in range(2,49):ledger.control_step(dict(step_index=i,num_envs=512,physics_samples_increment=512))
                def update(inner):
                    ledger.update_event('optimizer_started',dict(learning_state=inner.learning_state,expected_optimizer_steps=20))
                    ledger.update_event('optimizer_step_completed',dict(learning_state=dict(inner.learning_state,optimizer_steps=54002)))
            with self.assertRaises((FloatingPointError,ValueError)):self.api.perform_update(Broken(),c,local,ledger)
            self.assertEqual((c.iteration,c.model_iteration,local.iteration),(2796,2700,0))
            self.assertTrue(ledger.physics_samples_unknown)
            with self.assertRaises(ValueError):ledger.begin_attempt(local,512)
            loaded=self.api.ForkLedger(self.root/kind,run_id=kind);self.assertEqual(loaded.state_dict(),ledger.state_dict())

    def test_telemetry_copy_keeps_all553_old_event_bytes_without_writing_old_run(self):
        original=self.api.RUN/'wandb_telemetry';before=(original/'events.jsonl').read_bytes();old_state=(original/'state.json').read_bytes()
        target,state=self.api.telemetry_copy(self.root)
        self.assertEqual((target/'events.jsonl').read_bytes(),before);self.assertEqual(state['sequence'],553)
        self.assertEqual((original/'state.json').read_bytes(),old_state)
        self.assertEqual((original/'events.jsonl').read_bytes(),before)
        with (target/'events.jsonl').open('ab') as stream:stream.write(b'new-isolated-event\n')
        self.assertTrue((target/'events.jsonl').read_bytes().startswith(before))
        self.assertEqual((original/'events.jsonl').read_bytes(),before)

    def test_actual_evaluate_runtime_components_use_model_identity_and_raw_budget_binding(self):
        from test_v03_evaluation import RuntimeEvaluationTests,RuntimeFakeModel,RuntimeFakeMonitor
        harness=RuntimeEvaluationTests();harness.setUp()
        api,args,state,components,calls=harness.runtime(self.root)
        c=self.cursor()
        for _ in range(4):c.updated(512)
        fork=dict(state,container_schema=self.api.SCHEMA,budget_curriculum=c.state_dict(),num_envs=512)
        # Test-domain preload substitutes simulator/model only; original evaluate/raw/oracle run.
        with patch.object(api,'_runtime_components',return_value=components),patch.object(self.api,'preload_fork',return_value=fork) as loader:
            with self.api.runtime_projection(),self.api.fork_evidence_context(c):summary=api.evaluate(args,RuntimeFakeMonitor())
        self.assertEqual(loader.call_args.args,(args.checkpoint,args.checkpoint_sha256,self.api.DEFAULT_AUTH))
        self.assertEqual((summary['iteration'],summary['global_iteration'],summary['scheduled_budget_iteration']),(2704,2704,2800))
        raw=json.loads(Path(summary['raw_index_path']).read_text());self.assertEqual(raw['identity']['scheduled_budget_iteration'],2800)
        self.assertTrue(summary['evidence_verified']);self.assertEqual([row['joint_successes'] for row in summary['speeds']],[20,20,20])
        self.assertFalse(c.validate(summary,args.checkpoint_sha256,summary['case_table_sha256']))
        summary['scheduled_budget_iteration']=3000
        with self.assertRaises(ValueError):c._check(summary,'validation',args.checkpoint_sha256,summary['case_table_sha256'])

    def test_monitoring_eval_reuses_original_panel_names_with_budget_axis(self):
        from src.tasks.velocity.v03.telemetry import evaluation_metrics
        from test_v03_evaluation import make_gate_fixture
        c=self.cursor()
        for _ in range(4):c.updated(512)
        with self.api.fork_evidence_context(c):summary,_=make_gate_fixture(self.root/'panel',iteration=2704,stage_iteration=2704,num_envs=512)
        expected=evaluation_metrics(summary);actual=self.api.fork_evaluation_metrics(summary,c)
        for key,value in expected.items():
            if key not in ('global_iteration','stage_iteration'):self.assertEqual(actual[key],value,key)
        self.assertEqual(actual['global_iteration'],2800);self.assertEqual(actual['fork/eval_model_iteration'],2704)
        self.assertIn('eval/joint/validation/S1/vx_0.6/joint_success_rate',actual)

    def test_monitoring_rollout_uses_dynamic_policy_hash_and_static_origin(self):
        c=self.cursor();model=torch.nn.Linear(2,1)
        # Real policy digest excludes non-actor layers; use real actor-named module.
        class Policy(torch.nn.Module):
            def __init__(inner):super().__init__();inner.actor=model
        policy=Policy();origin=dict(path='/evidence/origin.pt',sha256='1'*64)
        first=self.api.rollout_identity(policy,c,self.root,origin)
        with torch.no_grad():model.weight.add_(1.)
        second=self.api.rollout_identity(policy,c,self.root,origin)
        self.assertNotEqual(first['policy_state_sha256'],second['policy_state_sha256'])
        self.assertEqual(first['origin_checkpoint'],origin);self.assertEqual(second['origin_checkpoint'],origin)
        self.assertEqual(first['policy_update_before_rollout'],2700);self.assertEqual(first['scheduled_budget_before_rollout'],2796)

    def test_monitoring_heartbeat_and_close_reuse_original_resource_metrics(self):
        from src.tasks.velocity.v03 import orchestration as original
        snapshot={'system/gpu_used_mib':123.,'system/gpu_utilization_percent':40.}
        closed={'system/observed_peak_rss_mib':456.,'system/resource_guard_pass':1}
        with patch.object(original,'resource_snapshot',return_value=snapshot) as sample:
            self.assertEqual(self.api.fork_resource_metrics(self.root),snapshot);self.assertEqual(sample.call_args.args[1:],(0,self.root))
        with patch.object(original,'resource_close_metrics',return_value=closed) as close:
            self.assertEqual(self.api.fork_resource_metrics(self.root,closed=True),closed);close.assert_called_once_with(self.root)

if __name__=='__main__':unittest.main()
