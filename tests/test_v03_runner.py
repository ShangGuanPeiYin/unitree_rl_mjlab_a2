"""CPU runner boundary behavior; raw gate fixtures are owned by V4e."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from src.tasks.velocity.v03.runner import Curriculum, ProbeCurriculum, PPO, capture_rng, restore_rng
from src.tasks.velocity.v03.model import V03ActorCritic
from src.tasks.velocity.v03.spec import INIT_PATH,INIT_SHA256,sha256
from src.tasks.velocity.v03.recovery import AttemptLedger,preload_checkpoint
from test_v03_recovery import complete


class Env:
    device='cpu';num_envs=258
    def __init__(self):self.rng=torch.Generator().manual_seed(42);self.resets=0
    def set_stage(self,stage,stage_iteration=0,previous_stage=None):self.stage=(stage,stage_iteration,previous_stage)
    def reset(self):
        self.resets+=1;random.random();np.random.rand();torch.rand(1);torch.rand(1,generator=self.rng)
        return {},{}


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_updates_require_pending_gate_and_exact_actual_samples(self):
        c=Curriculum()
        for _ in range(100):c.updated(1024)
        for _ in range(100):c.updated(512)
        self.assertEqual(c.env_steps,7372800)
        self.assertTrue(c.validation_due)
        with self.assertRaises(RuntimeError):c.updated(258)
        restored=Curriculum();restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.state_dict(),c.state_dict())
        bad=c.state_dict();bad['env_steps']-=48
        with self.assertRaises(ValueError):Curriculum().load_state_dict(bad)
        bad=c.state_dict();bad['stage']='S2';bad['stage_iteration']=0
        with self.assertRaises(ValueError):Curriculum().load_state_dict(bad)
    def test_fake_summary_without_raw_evidence_never_consumes_gate(self):
        c=Curriculum()
        for _ in range(200):c.updated(258)
        with self.assertRaises((ValueError,ImportError,ModuleNotFoundError)):
            c.validate({'passed':True},'1'*64,'2'*64)
        self.assertEqual(c.history,[])
    def test_rng_restore_after_reset_preserves_all_cpu_streams(self):
        env=Env();saved=capture_rng(env)
        expected=(random.random(),np.random.rand(),torch.rand(5),torch.rand(5,generator=env.rng))
        env.reset();restore_rng(saved,env)
        actual=(random.random(),np.random.rand(),torch.rand(5),torch.rand(5,generator=env.rng))
        self.assertEqual(actual[:2],expected[:2]);self.assertTrue(torch.equal(actual[2],expected[2]));self.assertTrue(torch.equal(actual[3],expected[3]))
    def test_full_model_adam_rng_curriculum_and_ledger_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=AttemptLedger(root/'ledger',run_id='seed42')
            c=Curriculum();model=V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256)
            runner=PPO(model,Env(),ledger=ledger)
            # Real complete Adam state for every trainable parameter, separate from physics fixture.
            for _ in range(20):
                sum(p.sum() for p in model.parameters()).backward();runner.optimizer.step();model.constrain_std();runner.optimizer.zero_grad()
            complete(ledger,c);runner.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20)
            path=root/'checkpoint.pt';digest=runner.save(path,c,{'seed':42},ledger=ledger)
            saved=preload_checkpoint(path,digest)
            self.assertEqual(len(saved['optimizer']['state']),43)
            expected=(random.random(),np.random.rand(),torch.rand(4),torch.rand(4,generator=runner.env.rng))
            target=PPO(V03ActorCritic(),Env(),ledger=ledger);restored=Curriculum()
            target.load(path,restored,digest,ledger=ledger)
            actual=(random.random(),np.random.rand(),torch.rand(4),torch.rand(4,generator=target.env.rng))
            self.assertEqual(actual[:2],expected[:2]);self.assertTrue(torch.equal(actual[2],expected[2]));self.assertTrue(torch.equal(actual[3],expected[3]))
            self.assertEqual(target.env.resets,1);self.assertIsNone(target.storage)
            self.assertEqual(restored.state_dict(),c.state_dict());self.assertEqual(target.learning_state,runner.learning_state)
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,target.model.state_dict()[k]),k)
            for i,m in runner.optimizer.state_dict()['state'].items():
                for k,v in m.items():self.assertTrue(torch.equal(v,target.optimizer.state_dict()['state'][i][k]))
            for mutation in ('schema','source','missing_adam','partial_adam','learning','rng'):
                bad=copy.deepcopy(saved)
                if mutation=='schema':bad['contract']['checkpoint_schema']=3
                if mutation=='source':bad['contract']['behavior_sha256']['bogus']='0'*64
                if mutation=='missing_adam':bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
                if mutation=='partial_adam':next(iter(bad['optimizer']['state'].values()))['step'].fill_(19)
                if mutation=='learning':bad['learning_state']['completed_ppo_updates']=2
                if mutation=='rng':bad['rng']['torch']=torch.ones(3,dtype=torch.uint8)
                torch.save(bad,path)
                with self.assertRaises(ValueError,msg=mutation):preload_checkpoint(path,sha256(path))
    def test_save_requires_exact_learning_curriculum_ledger_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=AttemptLedger(root/'ledger',run_id='seed42');c=Curriculum();c.updated(258)
            runner=PPO(V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256),Env(),ledger=ledger)
            with self.assertRaises(ValueError):runner.save(root/'bad.pt',c,{'seed':42},ledger=ledger)
    def test_explicit_downgrade_keeps_past_samples_and_can_save_before_next_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=AttemptLedger(root/'ledger',run_id='down');c=Curriculum()
            env=Env();env.num_envs=512;model=V03ActorCritic.from_v02(INIT_PATH,INIT_SHA256);runner=PPO(model,env,ledger=ledger)
            for _ in range(20):
                sum(p.sum() for p in model.parameters()).backward();runner.optimizer.step();model.constrain_std();runner.optimizer.zero_grad()
            complete(ledger,c,512);runner.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20)
            path=root/'n512.pt';sha=runner.save(path,c,{'seed':42},ledger=ledger)
            target=PPO(V03ActorCritic(),Env(),ledger=ledger);restored=Curriculum()
            with self.assertRaises(ValueError):target.load(path,restored,sha,ledger=ledger)
            target.load(path,restored,sha,ledger=ledger,allow_num_env_change=True)
            self.assertEqual(restored.env_steps,24576)
            lower=root/'n258.pt';lower_sha=target.save(lower,restored,{'seed':42},ledger=ledger)
            self.assertEqual(preload_checkpoint(lower,lower_sha)['curriculum']['env_steps'],24576)
            restored.updated(258);self.assertEqual(restored.env_steps,36960)
            with self.assertRaises(RuntimeError):
                # A same-global older candidate cannot overwrite the newer saved RNG/state.
                target.load(path,Curriculum(),sha,ledger=ledger,allow_num_env_change=True)

    def test_probe_55_plus_resume_one_has_real_zero_based_counters(self):
        c=ProbeCurriculum()
        for _ in range(55):c.updated(258)
        restored=ProbeCurriculum();restored.load_state_dict(c.state_dict());restored.updated(258)
        self.assertEqual((restored.iteration,restored.env_steps,restored.environment_stage_iteration),(56,693504,400))
        with self.assertRaises(RuntimeError):restored.updated(258)
        with self.assertRaises(ValueError):Curriculum().load_state_dict(restored.state_dict())


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.std=torch.nn.Parameter(torch.full((12,),.2));self.value=torch.nn.Parameter(torch.tensor(2.))
    def action_distribution(self,obs):
        return torch.distributions.Normal(torch.zeros(len(obs['proprio']),12)+self.value*0,self.std)
    def evaluate(self,obs):return self.value.expand(len(obs['proprio']),1)
    def update_critic_statistics(self,obs):pass
    @torch.no_grad()
    def constrain_std(self):self.std.clamp_(.05,.30)

class TinyEnv(Env):
    def __init__(self,fail=False):super().__init__();self.calls=0;self.fail=fail
    def reset(self):
        return {'proprio':torch.zeros(self.num_envs,45),'heightmap':torch.zeros(self.num_envs,3,1,1)},{}
    def step(self,actions):
        self.calls+=1
        if self.fail:raise RuntimeError('physics may have advanced')
        obs=self.reset()[0];flag=torch.zeros(self.num_envs,dtype=torch.bool)
        return obs,torch.zeros(self.num_envs),flag,flag,{}

class CoreLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_real48_and20adam_callbacks_commit_one_exact_learning_update(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=AttemptLedger(directory,run_id='core');c=Curriculum();env=TinyEnv();runner=PPO(TinyModel(),env,ledger=ledger)
            ledger.begin_attempt(c,258);runner.collect();runner.update();c.updated(258);ledger.complete_attempt(c,runner.learning_state)
            self.assertEqual((env.calls,c.iteration,ledger.completed_ppo_updates,ledger.physics_samples_observed),(48,1,1,12384))
            self.assertEqual(runner.learning_state['optimizer_steps'],20)
            with self.assertRaises(ValueError):runner.update()
    def test_failed_first_control_cannot_be_retried_under_the_same_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=AttemptLedger(directory,run_id='crash');c=Curriculum();env=TinyEnv(fail=True);runner=PPO(TinyModel(),env,ledger=ledger)
            ledger.begin_attempt(c,258)
            with self.assertRaisesRegex(RuntimeError,'physics'):runner.collect()
            with self.assertRaises(ValueError):runner.collect()
            self.assertEqual(env.calls,1);self.assertTrue(ledger.physics_samples_unknown)
    def test_callbacks_cannot_be_detached_before_physics_or_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=AttemptLedger(directory,run_id='detach');c=Curriculum();env=TinyEnv();runner=PPO(TinyModel(),env,ledger=ledger)
            ledger.begin_attempt(c,258);runner.control_step_callback=None
            with self.assertRaises(ValueError):runner.collect()
            self.assertEqual(env.calls,0)

class JointGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.count=0
    def fixture(self,c,kind='validation',passed=True,sha='1'*64):
        from test_v03_evaluation import make_gate_fixture
        self.count+=1
        return make_gate_fixture(self.root/str(self.count),stage=c.stage,kind=kind,iteration=c.iteration,
            stage_iteration=c.stage_iteration,checkpoint_sha=sha,num_envs=258,passed=passed)
    def updates(self,c,n=200):
        for _ in range(n):c.updated(258)
    def gate(self,c,passed=True):
        summary,case=self.fixture(c,passed=passed)
        return c.validate(summary,'1'*64,case)
    def test_joint_two_passes_s1_extra_failure_then_fresh_pair_and_replay(self):
        c=Curriculum();self.updates(c);self.assertFalse(self.gate(c));self.assertEqual(c.consecutive_passes,1)
        self.updates(c);self.assertTrue(self.gate(c));self.assertTrue(c.needs_perturbed)
        with self.assertRaises(RuntimeError):c.updated(258)
        with self.assertRaises(RuntimeError):c.advance()
        bad,case=self.fixture(c,'perturbed',False)
        with self.assertRaises(ValueError):c.validate_perturbed(bad,'2'*64,case)
        self.assertFalse(c.validate_perturbed(bad,'1'*64,case));self.assertEqual(c.consecutive_passes,0);self.assertFalse(c.s1_extra_passed)
        with self.assertRaises(ValueError):c.validate_perturbed(bad,'1'*64,case)
        self.updates(c);self.gate(c);self.updates(c);self.gate(c)
        extra,case=self.fixture(c,'perturbed');self.assertTrue(c.validate_perturbed(extra,'1'*64,case));c.advance()
        self.assertEqual((c.stage,c.stage_iteration,c.iteration,c.env_steps),('S2',0,800,9907200))
        restored=Curriculum();restored.load_state_dict(c.state_dict());self.assertEqual(restored.state_dict(),c.state_dict())
        forged=c.state_dict();forged['history'][0]['passed']=False
        with self.assertRaises(ValueError):Curriculum().load_state_dict(forged)
    def test_final_budget_gate_and_extra_take_precedence_and_exhaustion_stops(self):
        edge=Curriculum();exhausted=Curriculum()
        for k in range(1,16):
            self.updates(edge);self.gate(edge,passed=k>=14)
            self.updates(exhausted);self.gate(exhausted,passed=False)
        self.assertTrue(edge.exhausted);self.assertTrue(edge.needs_perturbed)
        extra,case=self.fixture(edge,'perturbed');edge.validate_perturbed(extra,'1'*64,case);edge.advance()
        self.assertEqual((edge.stage,edge.iteration),('S2',3000));self.assertEqual(edge.consumed['S1'],3000)
        self.assertTrue(exhausted.exhausted)
        with self.assertRaises(RuntimeError):exhausted.updated(258)
        with self.assertRaises(RuntimeError):exhausted.advance()
    def test_duplicate_wrong_identity_missing_raw_and_parking_cannot_authorize_gate(self):
        c=Curriculum();self.updates(c);summary,case=self.fixture(c)
        for field,value in [('stage','S2'),('iteration',201),('eval_seed',42),('global_iteration',201),('parking_diagnostic',True)]:
            bad=copy.deepcopy(summary);bad[field]=value
            with self.assertRaises(ValueError):c.validate(bad,'1'*64,case)
        missing=copy.deepcopy(summary);missing['raw_index_path']=str(self.root/'missing.json')
        with self.assertRaises((ValueError,OSError)):c.validate(missing,'1'*64,case)
        self.assertFalse(c.validate(summary,'1'*64,case))
        with self.assertRaises(ValueError):c.validate(summary,'1'*64,case)
    def test_s5_freezes_second_candidate_and_final94_is_terminal_without_reselection(self):
        c=Curriculum()
        for stage in c.stages:
            self.updates(c);self.gate(c);self.updates(c);self.gate(c)
            if stage=='S1':
                extra,case=self.fixture(c,'perturbed');c.validate_perturbed(extra,'1'*64,case)
            if stage!='S5':c.advance()
        self.assertTrue(c.complete);self.assertEqual(c.frozen_candidate_sha,'1'*64)
        with self.assertRaises(RuntimeError):c.updated(258)
        final,case=self.fixture(c,'test',False)
        self.assertFalse(c.validate_final(final,'1'*64,case));self.assertIs(c.final_result,False)
        with self.assertRaises(ValueError):c.validate_final(final,'1'*64,case)
        with self.assertRaises(ValueError):c.validate_final(final,'2'*64,case)
        restored=Curriculum();restored.load_state_dict(c.state_dict());self.assertIs(restored.final_result,False)


if __name__=='__main__':unittest.main()
