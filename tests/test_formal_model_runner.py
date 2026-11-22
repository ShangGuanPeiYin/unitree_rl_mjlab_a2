"""Formal migration, strict joint curriculum and real PPO/RNG counterexamples."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.tasks.velocity.v02_formal.model import FormalActorCritic
from src.tasks.velocity.v02_formal.runner import Curriculum,PPO,capture_rng,restore_rng
from src.tasks.velocity.v02_formal.spec import TASK,PROTOCOL_ID,CODE_VERSION,SCHEMA_VERSION,INIT_PATH,INIT_SHA256,sha256
from src.tasks.velocity.v02.model import V02ActorCritic

SHA='1'*64; CASE='2'*64; FLAT_CASE='3'*64

def summary(c,kind='validation',passed=True,sha=SHA):
    flat=kind in ('flat','final_flat');count=100 if kind=='test' else 20
    result=dict(task=TASK,protocol_id=PROTOCOL_ID,code_version=CODE_VERSION,checkpoint_schema=SCHEMA_VERSION,
        source_c_sha256=INIT_SHA256,seed=42,eval_seed=300142 if kind=='perturbed' else 400142 if kind in ('test','final_flat') else 200142,
        stage='flat' if flat else c.stage,kind=kind,iteration=c.iteration,stage_iteration=c.stage_iteration,
        checkpoint_sha256=sha,case_table_sha256=FLAT_CASE if flat else CASE,training_num_envs=512,statistics_unchanged=True)
    result['speeds']=[dict(vx=v,episodes=count,failures=0 if passed else 1,mean_speed=0. if v==0 else v,mean_vx=v,episode_seconds=30) if flat else dict(vx=v,episodes=count,failures=0,successes=count if passed else 17) for v in ([0.,.2,.4,.6] if flat else [.2,.4,.6])]
    return result


def gate(c,passed=True):return c.validate(summary(c,passed=passed),summary(c,'flat'),SHA,CASE,FLAT_CASE)
def updates(c,count):
    for _ in range(count):c.updated(258)


class FakeEnv:
    device='cpu';num_envs=512
    def __init__(self):self.rng=torch.Generator().manual_seed(142);self.resets=0
    def set_stage(self,stage,stage_iteration=0,previous_stage=None):self.stage=stage
    def reset(self):
        self.resets+=1
        random.random();np.random.rand();torch.rand(1);torch.rand(1,generator=self.rng)
        return {},{}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_complete_c_actor_mean_parity_1000_observations_and_fresh_value_state(self):
        random.seed(99);torch.manual_seed(99)
        source=torch.load(INIT_PATH,map_location='cpu',weights_only=False)
        legacy=V02ActorCritic('C');legacy.load_state_dict(source['model'])
        migrated=FormalActorCritic.from_c(INIT_PATH,INIT_SHA256)
        self.assertTrue(torch.equal(migrated.std,legacy.std))
        self.assertEqual(migrated.critic_normalizer.count.item(),0)
        self.assertTrue(torch.equal(migrated.critic_normalizer.mean,torch.zeros(80)))
        self.assertFalse(torch.equal(migrated.critic[0].weight,legacy.critic[0].weight))
        maximum=0.
        with torch.no_grad():
            for _ in range(10):
                obs=dict(proprio=torch.randn(100,45),history=torch.randn(100,5,45),heightmap=torch.randn(100,3,48,32))
                maximum=max(maximum,float((legacy(obs)-migrated(obs)).abs().max()))
        self.assertLessEqual(maximum,1e-5)
        self.assertFalse(PPO(migrated,FakeEnv()).optimizer.state)
    def test_source_sha_schema_key_shape_and_joint_contract_rejected(self):
        with self.assertRaises(ValueError):FormalActorCritic.from_c(INIT_PATH,'0'*64)
        source=torch.load(INIT_PATH,map_location='cpu',weights_only=False)
        for mutation in ('schema','key','shape','order'):
            bad=copy.deepcopy(source)
            if mutation=='schema':bad['contract']['checkpoint_schema']=3
            if mutation=='key':del bad['model']['actor.fusion.weight']
            if mutation=='shape':bad['model']['std']=torch.ones(11)*.2
            if mutation=='order':bad['contract']['joint_order']=['wrong']
            with patch('src.tasks.velocity.v02_formal.model.sha256',return_value=INIT_SHA256),patch('src.tasks.velocity.v02_formal.model.torch.load',return_value=bad):
                with self.assertRaises(ValueError):FormalActorCritic.from_c('mock',INIT_SHA256)
    def test_schema3_resume_full_optimizer_and_rng_restored_after_reset(self):
        env=FakeEnv();model=FormalActorCritic.from_c(INIT_PATH,INIT_SHA256);runner=PPO(model,env);c=Curriculum()
        # Exercise actual Adam moments without running a physical rollout.
        for _ in range(20):
            loss=sum(p.sum() for p in model.parameters());loss.backward();runner.optimizer.step();model.constrain_std();runner.optimizer.zero_grad()
        runner.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt';digest=runner.save(path,c,{'seed':42})
            expected=(random.random(),np.random.rand(),torch.rand(5),torch.rand(5,generator=env.rng))
            random.seed(12);np.random.seed(12);torch.manual_seed(12)
            target=PPO(FormalActorCritic(),FakeEnv());loaded=target.load(path,Curriculum(),digest)
            actual=(random.random(),np.random.rand(),torch.rand(5),torch.rand(5,generator=target.env.rng))
            self.assertEqual(expected[:2],actual[:2]);self.assertTrue(torch.equal(expected[2],actual[2]));self.assertTrue(torch.equal(expected[3],actual[3]))
            self.assertEqual(target.env.resets,1);self.assertIsNone(target.storage)
            self.assertTrue(target.optimizer.state);self.assertEqual(target.learning_state,runner.learning_state)
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,target.model.state_dict()[key]),key)
            with self.assertRaises(ValueError):target.load(path,Curriculum(),'0'*64)
            bad=copy.deepcopy(loaded);bad['contract']['checkpoint_schema']=2;torch.save(bad,path)
            with self.assertRaises(ValueError):target.load(path,Curriculum(),sha256(path))
    def test_schema3_complete_adam_coverage_and_step_identity_required_even_with_new_sha(self):
        from src.tasks.velocity.v02_formal.recovery import preload_checkpoint
        model=FormalActorCritic.from_c(INIT_PATH,INIT_SHA256);runner=PPO(model,FakeEnv())
        # Twenty actual Adam minibatch steps form one complete approved PPO boundary.
        for _ in range(20):
            sum(p.sum() for p in model.parameters()).backward();runner.optimizer.step();model.constrain_std();runner.optimizer.zero_grad()
        runner.learning_state=dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt';digest=runner.save(path,Curriculum(),{'seed':42})
            original=preload_checkpoint(path,digest);self.assertEqual(len(original['optimizer']['state']),43)
            self.assertEqual(original['curriculum']['iteration'],0) # probe updates do not consume curriculum
            for mutation in ('missing_one','empty','fractional_step','different_step','missing_learning','partial_boundary'):
                bad=copy.deepcopy(original)
                first=next(iter(bad['optimizer']['state']))
                if mutation=='missing_one':del bad['optimizer']['state'][first]
                if mutation=='empty':bad['optimizer']['state'].clear()
                if mutation=='fractional_step':bad['optimizer']['state'][first]['step'].fill_(20.5)
                if mutation=='different_step':bad['optimizer']['state'][first]['step'].fill_(19)
                if mutation=='missing_learning':del bad['learning_state']
                if mutation=='partial_boundary':bad['learning_state']['optimizer_steps']=19
                torch.save(bad,path)
                with self.assertRaises(ValueError,msg=mutation):preload_checkpoint(path,sha256(path))
                with self.assertRaises(ValueError,msg=mutation):PPO(FormalActorCritic(),FakeEnv()).load(path,Curriculum(),sha256(path))
            # Save also refuses a partially lost in-memory Adam state.
            runner.optimizer.state.pop(next(iter(runner.optimizer.state)))
            with self.assertRaises(ValueError):runner.save(path,Curriculum(),{'seed':42})
    def test_fresh_optimizer_explicit_identity_remains_valid_with_conservatively_paid_budget(self):
        from src.tasks.velocity.v02_formal.recovery import preload_checkpoint
        c=Curriculum();c.updated(258)
        runner=PPO(FormalActorCritic.from_c(INIT_PATH,INIT_SHA256),FakeEnv())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fresh.pt';digest=runner.save(path,c,{'seed':42})
            state=preload_checkpoint(path,digest)
            self.assertEqual(state['curriculum']['iteration'],1);self.assertEqual(state['optimizer']['state'],{})
            self.assertEqual(state['learning_state'],dict(optimizer_initialized=False,completed_ppo_updates=0,optimizer_steps=0))
            target=PPO(FormalActorCritic(),FakeEnv());target.env.num_envs=258
            target.load(path,Curriculum(),digest,allow_num_env_change=True)
            self.assertEqual(target.learning_state,runner.learning_state);self.assertFalse(target.optimizer.state)

    def test_rng_inserting_diagnostics_at_same_restart_boundary_has_identical_next_sequence(self):
        env=FakeEnv();saved=capture_rng(env)
        def segment(with_diagnostics):
            if with_diagnostics:
                random.random();np.random.rand(23);torch.rand(33);torch.rand(19,generator=env.rng)
            env.reset();restore_rng(saved,env)
            return random.random(),np.random.rand(),torch.rand(5),torch.rand(5,generator=env.rng)
        direct=segment(False);diagnostic=segment(True)
        self.assertEqual(direct[:2],diagnostic[:2]);self.assertTrue(torch.equal(direct[2],diagnostic[2]));self.assertTrue(torch.equal(direct[3],diagnostic[3]))


class CurriculumTests(unittest.TestCase):
    def test_joint_two_gates_and_r0_extra_failure_clear_and_budget_edge(self):
        c=Curriculum();self.assertEqual(sum(c.budgets.values()),26000)
        updates(c,200);self.assertFalse(gate(c));updates(c,200);self.assertTrue(gate(c));self.assertTrue(c.needs_perturbed)
        with self.assertRaises(RuntimeError):c.updated()
        self.assertFalse(c.validate_perturbed(summary(c,'perturbed',False),SHA,CASE));self.assertEqual(c.consecutive_passes,0)
        updates(c,200);gate(c);updates(c,200);gate(c);self.assertTrue(c.validate_perturbed(summary(c,'perturbed'),SHA,CASE));c.advance()
        self.assertEqual((c.stage,c.stage_iteration,c.iteration),('S1',0,800))
        restored=Curriculum();restored.load_state_dict(c.state_dict());self.assertEqual(restored.state_dict(),c.state_dict())
        # Every failed gate consumes the scheduled checkpoint; budget-edge pair
        # at 2800/3000 still receives the R0 additional check before exhaustion.
        edge=Curriculum()
        for k in range(1,16):updates(edge,200);gate(edge,passed=k>=14)
        self.assertTrue(edge.exhausted);self.assertTrue(edge.needs_perturbed)
        edge.validate_perturbed(summary(edge,'perturbed'),SHA,CASE);edge.advance();self.assertEqual(edge.stage,'S1')
    def test_missing_cases_identity_wrong_flat_same_checkpoint_and_skipped_gate(self):
        c=Curriculum();updates(c,200)
        for mutate in ('count','case','checkpoint','stage','statistics','duration'):
            a=summary(c);b=summary(c,'flat')
            if mutate=='count':a['speeds'][0]['episodes']=19
            if mutate=='case':a['case_table_sha256']='4'*64
            if mutate=='checkpoint':b['checkpoint_sha256']='4'*64
            if mutate=='stage':a['stage']='S1'
            if mutate=='statistics':a['statistics_unchanged']=False
            if mutate=='duration':b['speeds'][0]['episode_seconds']=1
            with self.assertRaises(ValueError):c.validate(a,b,SHA,CASE,FLAT_CASE)
        self.assertEqual(c.history,[])
        with self.assertRaises(RuntimeError):c.updated()
        broken=c.state_dict();broken['stage']='S1';broken['consumed']['S1']=broken['stage_iteration'];broken['consumed']['R0']=0
        with self.assertRaises(ValueError):Curriculum().load_state_dict(broken)
    def test_final_candidate_frozen_and_failure_terminal_no_reselection(self):
        c=Curriculum()
        for stage in c.stages:
            updates(c,200);gate(c);updates(c,200);gate(c)
            if stage=='R0':c.validate_perturbed(summary(c,'perturbed'),SHA,CASE)
            if stage!='S5':c.advance()
        self.assertTrue(c.complete);self.assertEqual(c.frozen_candidate_sha,SHA)
        with self.assertRaises(RuntimeError):c.updated()
        self.assertFalse(c.validate_final(summary(c,'test',False),summary(c,'final_flat'),SHA,CASE,FLAT_CASE))
        with self.assertRaises(ValueError):c.validate_final(summary(c,'test'),summary(c,'final_flat'),SHA,CASE,FLAT_CASE)
        restored=Curriculum();restored.load_state_dict(c.state_dict());self.assertIs(restored.final_result,False)


# Cheap model uses the real inherited 48-step rollout, timeout GAE, 5x4 Adam.
class TinyModel(torch.nn.Module):
    def __init__(self):super().__init__();self.std=torch.nn.Parameter(torch.full((12,),.2));self.bias=torch.nn.Parameter(torch.tensor(2.))
    def action_distribution(self,o):return torch.distributions.Normal(self.bias.expand(len(o['proprio']),12)*0,self.std.clamp(.05,.30))
    def evaluate(self,o):return self.bias.expand(len(o['proprio']),1)
    def update_critic_statistics(self,o):pass
    @torch.no_grad()
    def constrain_std(self):self.std.clamp_(.05,.30)
class TerminalEnv:
    device='cpu';num_envs=258
    def reset(self):return {'proprio':torch.zeros(258,45),'heightmap':torch.zeros(258,3,1,1)},{}
    def step(self,action):
        o=self.reset()[0];term=torch.arange(258)%2==0
        return o,torch.zeros(258),term,~term,dict(terminal_observation=o)
class PPOTests(unittest.TestCase):
    def test_258_timeout_bootstrap_and_true_termination_and_real_metrics(self):
        runner=PPO(TinyModel(),TerminalEnv());seen=[];runner.diagnostics_callback=seen.append
        runner.collect();returns=runner.storage['returns'].reshape(48,258)
        self.assertTrue(torch.equal(returns[:,::2],torch.zeros(48,129)))
        self.assertTrue(torch.allclose(returns[:,1::2],torch.full((48,129),1.98)))
        metrics=runner.update();self.assertEqual(len(seen),48)
        self.assertEqual(runner.learning_state,dict(optimizer_initialized=True,completed_ppo_updates=1,optimizer_steps=20))
        self.assertTrue(all(v['step'].item()==20 for v in runner.optimizer.state.values()))
        for key in ('policy_loss','value_loss','entropy','kl','lr','clip_fraction','gradient_norm','std_mean',*[f'std_{i}' for i in range(12)]):self.assertTrue(np.isfinite(metrics[key]),key)
        self.assertTrue(0<=metrics['clip_fraction']<=1)

if __name__=='__main__':unittest.main()
