"""SDS 0.2 migration, budget, validation identity and resume boundaries (CPU)."""
import copy
from pathlib import Path
import numpy as np
import unittest
import tempfile
import torch
from src.tasks.velocity.v02.model import V02ActorCritic
from src.tasks.velocity.v02.runner import Curriculum, PPO, select_recipe
from src.tasks.velocity.v02.spec import S0_PATH, S0_SHA256, TASK
from src.tasks.velocity.rl.ideal_heightmap_actor_critic import IdealHeightmapActorCritic

torch.set_num_threads(2)
SHA='1'*64

def obs(n):
    g=torch.Generator().manual_seed(101)
    heightmap=torch.zeros(n,3,48,32)
    heightmap[:,0]=torch.rand(n,48,32,generator=g)*.4-.2
    heightmap[:,1]=1
    heightmap[:,2]=torch.rand(n,48,32,generator=g)*.2
    return dict(proprio=torch.randn(n,45,generator=g)*.1,
                history=torch.randn(n,5,45,generator=g)*.1,
                heightmap=heightmap,privileged=torch.zeros(n,35))

def summary(recipe='B',mode='comparison',stage='P4',iteration=2000,stage_iteration=800,kind='fixed',sha=SHA):
    return dict(task=TASK,code_version='0.2.0',recipe_version='0.2',checkpoint_schema=2,
        recipe=recipe,mode=mode,stage=stage,seed=42,source_s0_sha256=S0_SHA256,training_num_envs=512,eval_seed=300042 if kind=='perturbed' else 200042,
        checkpoint_sha256=sha,case_table_sha256='3'*64,kind=kind,iteration=iteration,stage_iteration=stage_iteration,
        speeds=[dict(vx=v,episodes=20,successes=18,failures=0,mean_speed=v,mean_vx=v) for v in (.2,.4,.6)])


def candidate(recipe,passed=True):
    fixed1800=summary(recipe=recipe,iteration=1800,stage_iteration=1800 if recipe=='A' else 600)
    fixed2000=summary(recipe=recipe,stage_iteration=2000 if recipe=='A' else 800)
    if not passed: fixed1800['speeds'][0]['successes']=17
    perturb=summary(recipe=recipe,kind='perturbed',stage_iteration=2000 if recipe=='A' else 800)
    flat=summary(recipe=recipe,stage='flat',kind='flat',stage_iteration=2000 if recipe=='A' else 800)
    flat['speeds'].insert(0,dict(vx=0.,episodes=20,failures=0,mean_speed=.01,mean_vx=0.))
    return dict(fixed1800=fixed1800,fixed2000=fixed2000,perturbed=perturb if passed else None,flat=flat if passed else None)


class CPUEnv:
    """No physical simulation: exercises checkpoint-owned state and reset boundary."""
    def __init__(self,n=4,recipe='B',mode='formal'):
        self.num_envs=n;self.device='cpu';self.recipe=recipe;self.mode=mode
        self.rng=torch.Generator().manual_seed(42);self.resets=0;self.stage=None
    def reset(self):self.resets+=1;return obs(self.num_envs),{}
    def set_stage(self,stage,stage_iteration=0,previous_stage=None):
        self.stage=stage;self.previous_stage=previous_stage


class V02Tests(unittest.TestCase):
    def test_actor_only_exact_s0_transfer_thousand_legal_observations_and_fresh_critic(self):
        state=torch.load(S0_PATH,map_location='cpu',weights_only=False)
        reference=IdealHeightmapActorCritic(); reference.load_state_dict(state['model'])
        torch.manual_seed(42); fresh=V02ActorCritic('A')
        torch.manual_seed(42); model=V02ActorCritic.from_s0(S0_PATH,S0_SHA256,recipe='A')
        maximum=0.
        with torch.no_grad():
            observations=obs(1000)
            for start in range(0,1000,25):
                chunk={k:v[start:start+25] for k,v in observations.items()}
                maximum=max(maximum,float((reference(chunk)-model(chunk)).abs().max()))
            assert maximum <= 1e-5
            for key,value in fresh.state_dict().items():
                if key.startswith('critic'): assert torch.equal(value,model.state_dict()[key])
            assert model.critic_normalizer.count == 0
            assert torch.equal(model.std,state['model']['std'])
            assert model.actor.fusion.weight.abs().sum() > 0
            frozen={k:v.clone() for k,v in model.actor.normalizer.state_dict().items()}
            before=model(obs(2)); changed=obs(2); changed['privileged']+=100
            assert torch.equal(before,model(changed))
            model.update_critic_statistics(changed)
            assert all(torch.equal(v,model.actor.normalizer.state_dict()[k]) for k,v in frozen.items())
        with self.assertRaises(ValueError): V02ActorCritic.from_s0(S0_PATH,'0'*64)
        assert all(p.requires_grad for p in model.actor.parameters())

    def test_recipe_std_floor_and_same_actor(self):
        baseline=V02ActorCritic.from_s0(S0_PATH,S0_SHA256,recipe='A')
        with torch.no_grad():expected_mean=baseline(obs(2))
        for recipe,lower in [('A',.05),('B',.05),('C',.05),('D',.10)]:
            model=V02ActorCritic.from_s0(S0_PATH,S0_SHA256,recipe=recipe)
            s=torch.load(S0_PATH,map_location='cpu',weights_only=False)['model']['std']
            with torch.no_grad():assert torch.equal(model(obs(2)),expected_mean)
            assert torch.equal(model.std.detach(),s.clamp(min=.20 if recipe=='D' else .05,max=.30))
            with torch.no_grad(): model.std.fill_(-1)
            assert model.action_distribution(obs(1)).stddev.min() >= lower
            model.constrain_std(); assert model.std.min() >= lower
            with torch.no_grad(): model.std.fill_(1)
            model.constrain_std(); assert model.std.max() <= .30

    def test_comparison_predetermined_course_and_exact_budget(self):
        c=Curriculum('B','comparison'); assert c.stage=='P1'
        with self.assertRaises(RuntimeError):c.updated()
        baseline=summary(iteration=0,stage_iteration=0)
        c.validate(baseline,expected_checkpoint_sha=SHA)
        for iteration in range(1,2001):
            c.updated()
            if iteration==1800:c.validate(summary(iteration=1800,stage_iteration=600),expected_checkpoint_sha=SHA)
            if iteration in (400,800,1200,2000):
                assert c.stage=={400:'P2',800:'P3',1200:'P4',2000:'P4'}[iteration]
        assert c.consumed==dict(P1=400,P2=400,P3=400,P4=800)
        assert c.exhausted
        with self.assertRaises(RuntimeError): c.updated()
        a=Curriculum('A','formal'); assert a.stage=='P4'
        assert sum(a.budgets.values())==25000
        assert sum(Curriculum('D','formal').budgets.values())==28000

    def test_formal_gates_are_stage_local_distinct_and_identity_checked(self):
        c=Curriculum('B','formal')
        for _ in range(200):c.updated()
        good=summary(mode='formal',stage='P1',iteration=200,stage_iteration=200,kind='validation')
        with self.assertRaises(RuntimeError):c.updated()
        assert not c.validate(good,expected_checkpoint_sha=SHA)
        with self.assertRaises(ValueError):c.validate(good,expected_checkpoint_sha=SHA)
        for _ in range(200):c.updated()
        good.update(iteration=400,stage_iteration=400)
        for field,value in [('seed',43),('checkpoint_sha256','2'*64),('stage','P2')]:
            bad=copy.deepcopy(good);bad[field]=value
            with self.assertRaises(ValueError):c.validate(bad,expected_checkpoint_sha=SHA)
        assert c.validate(good,expected_checkpoint_sha=SHA)
        c.advance();assert (c.stage,c.stage_iteration,c.iteration)==('P2',0,400)
        d=Curriculum('B','formal');d.load_state_dict(c.state_dict());assert d.state_dict()==c.state_dict()
        bad=copy.deepcopy(c.state_dict());bad['iteration']+=1
        with self.assertRaises(ValueError):d.load_state_dict(bad)
        wrong=Curriculum('C','formal')
        with self.assertRaises(ValueError):wrong.load_state_dict(c.state_dict())

    def test_recipe_selector_all_groups_and_minimal_change_priority(self):
        results={r:candidate(r,r!='A') for r in 'ABCD'}
        assert select_recipe(results)=='B'
        assert select_recipe({r:candidate(r,False) for r in 'ABCD'}) is None
        with self.assertRaises(ValueError):select_recipe({'B':candidate('B')})
        for mutate in ('hash','seed','count','source','scale'):
            bad=copy.deepcopy(results)
            if mutate=='hash':bad['B']['perturbed']['checkpoint_sha256']='2'*64
            if mutate=='seed':bad['B']['perturbed']['eval_seed']=42
            if mutate=='count':bad['B']['flat']['speeds'][0]['episodes']=19
            if mutate=='source':del bad['B']['perturbed']['source_s0_sha256']
            if mutate=='scale':bad['B']['fixed2000']['training_num_envs']=258
            with self.assertRaises(ValueError):select_recipe(bad)

    def test_atomic_strict_resume_fresh_optimizer_and_new_episode(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); tmp_path=Path(temporary.name)
        model=V02ActorCritic.from_s0(S0_PATH,S0_SHA256,recipe='B'); env=CPUEnv();ppo=PPO(model,env);c=Curriculum('B','formal')
        assert not ppo.optimizer.state
        ppo.reset();ppo.storage={'physical_cache':'must not persist'}
        path=tmp_path/'state.pt';digest=ppo.save(path,c,{'seed':42})
        target=PPO(V02ActorCritic('B'),CPUEnv());target.load(path,Curriculum('B','formal'),digest)
        assert all(torch.equal(v,target.model.state_dict()[k]) for k,v in model.state_dict().items())
        with self.assertRaises(ValueError):target.load(path,Curriculum('B','comparison'),digest)
        uninitialized=PPO(V02ActorCritic('B'),CPUEnv())
        with self.assertRaises(ValueError):uninitialized.save(tmp_path/'uninitialized.pt',c,{'seed':42})
        assert target.storage is None and target.env.resets==1 and target.env.previous_stage is None
        assert target.env.stage=='P1'
        with self.assertRaises(ValueError):target.load(path,Curriculum('C','formal'),digest)
        with self.assertRaises(ValueError):target.load(path,Curriculum('B','formal'),'0'*64)
        with self.assertRaises(ValueError):PPO(V02ActorCritic('B'),CPUEnv(8)).load(path,Curriculum('B','formal'),digest)


class TinyModel(torch.nn.Module):
    """Cheap differentiable policy for PPO bootstrap/update boundary tests."""
    def __init__(self):
        super().__init__();self.recipe='B';self.recipe_settings={'std_min':.05,'std_max':.30}
        self.std=torch.nn.Parameter(torch.full((12,),.2));self.bias=torch.nn.Parameter(torch.tensor(2.))
    def action_distribution(self,observations):
        return torch.distributions.Normal(self.bias.expand(len(observations['proprio']),12)*0,self.std.clamp(.05,.30))
    def evaluate(self,observations):return self.bias.expand(len(observations['proprio']),1)
    def update_critic_statistics(self,observations):pass
    @torch.no_grad()
    def constrain_std(self):self.std.clamp_(.05,.30)

class TerminalEnv(CPUEnv):
    def step(self,action):
        n=self.num_envs
        return (obs(n),torch.zeros(n),torch.tensor([True,False,True,False]),
            torch.tensor([False,True,False,True]),dict(terminal_observation=obs(n)))

class PPOBootstrapTests(unittest.TestCase):
    def test_timeouts_bootstrap_terminations_zero_and_std_per_dimension(self):
        ppo=PPO(TinyModel(),TerminalEnv());ppo.collect()
        returns=ppo.storage['returns'].reshape(48,4)
        self.assertTrue(torch.equal(returns[:,[0,2]],torch.zeros(48,2)))
        self.assertTrue(torch.allclose(returns[:,[1,3]],torch.full((48,2),1.98)))
        metrics=ppo.update()
        self.assertTrue(all(f'std_{i}' in metrics for i in range(12)))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

if __name__=="__main__": unittest.main()
