"""Single-device PPO orchestration. Importing this module never starts training."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
import subprocess
import time
import numpy as np
import torch
from .algorithm import JointPPO, compute_gae
from .checkpoint import save_checkpoint, restore_checkpoint
from .curriculum import Curriculum
from .model import PointCloudActorCritic
from src.tasks.velocity.rl.eth_actor import EthActor


def tree_map(fn, tree):
    return {k:tree_map(fn,v) for k,v in tree.items()} if isinstance(tree,dict) else fn(tree)


def stack(trees):
    if isinstance(trees[0],dict):
        return {k:stack([x[k] for x in trees]) for k in trees[0]}
    return torch.stack(trees)


class PointCloudRunner:
    def __init__(self, env, settings, *, resource_test=False):
        self.env,self.settings=env,settings
        self.config=settings.validate(resource_test=resource_test)
        self.config['privileged_names']=env.privileged_names
        self.model=PointCloudActorCritic(env.privileged_dim,
            eth=EthActor.load(settings.eth_checkpoint,env.device),
            use_points=settings.variant=='full').to(env.device)
        self.algorithm=JointPPO(self.model)
        self.curriculum=Curriculum()
        self.iteration=0
        self.obs=env.reset()
        self.best_score={}
        if settings.resume:
            # Restore curriculum before the mandatory environment reset.
            metadata=torch.load(settings.resume,map_location='cpu',weights_only=False)
            self.curriculum=Curriculum(**metadata['curriculum'])
            env.set_stage(self.curriculum.stage[0])
            state=restore_checkpoint(settings.resume,self.model,self.algorithm.optimizer,self._reset)
            self.iteration=state['iteration']
            self.best_score=state['config'].get('best_score',{})
            self.resume_event=state['event']

    def _reset(self):
        self.obs=self.env.reset()

    def collect(self):
        observations=[]; actions=[]; logps=[]; values=[]; next_values=[]
        rewards=[]; failures=[]; successes=[]; timeouts=[]
        capacities=dict(contact_capacity_hits=0,constraint_capacity_hits=0)
        diagnostics={}
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.settings.rollout):
                obs=self.obs
                dist=self.model.distribution(obs['actor'])
                action=dist.sample()
                value=self.model.critic(obs['privileged']).squeeze(-1)
                following,reward,failure,success,timeout,info=self.env.step(action)
                terminal=info['terminal_observation']
                for key in capacities: capacities[key]+=int(info[key])
                for key,component in info.get('diagnostics',{}).items(): diagnostics[key]=diagnostics.get(key,0.)+component/self.settings.rollout
                for key,reward_component in info['reward_terms'].items(): diagnostics['reward/'+key]=diagnostics.get('reward/'+key,0.)+float(reward_component.mean())/self.settings.rollout
                diagnostics['point_valid_fraction']=diagnostics.get('point_valid_fraction',0.)+float(obs['actor']['mask'].float().mean())/self.settings.rollout
                correction=dist.mean-self.model.actor.eth(obs['actor']['obs_history'][:,-1])
                diagnostics['correction_rms']=diagnostics.get('correction_rms',0.)+float(correction.square().mean().sqrt())/self.settings.rollout
                # terminal_observation is taken before ANY automatic reset.
                nv=self.model.critic(terminal['privileged']).squeeze(-1)
                observations.append(tree_map(lambda x:x.detach().clone(),obs))
                actions.append(action); logps.append(dist.log_prob(action).sum(-1))
                values.append(value); next_values.append(nv); rewards.append(reward.clone())
                failures.append(failure.clone()); successes.append(success.clone()); timeouts.append(timeout.clone())
                self.obs=following
        data=stack(observations)
        advantage,returns=compute_gae(torch.stack(rewards),torch.stack(values),torch.stack(next_values),
            torch.stack(failures),torch.stack(successes),torch.stack(timeouts))
        data.update(actions=torch.stack(actions),old_log_prob=torch.stack(logps),
            old_value=torch.stack(values),advantages=advantage,returns=returns)
        data=tree_map(lambda x:x.flatten(0,1),data)
        data['advantages']=(data['advantages']-data['advantages'].mean())/data['advantages'].std(unbiased=False).clamp_min(1e-8)
        return data,{**capacities,**diagnostics,'reward':float(torch.stack(rewards).mean()),
                     'failure_count':int(torch.stack(failures).sum()),
                     'success_count':int(torch.stack(successes).sum())}

    def update(self):
        self.model.freeze_eth(self.iteration < 25 and self.curriculum.index == 0)
        start=time.perf_counter()
        batch,metrics=self.collect()
        sample_end=time.perf_counter()
        self.model.train()
        count=batch['actions'].shape[0]
        losses=[]
        for _ in range(self.settings.epochs):
            for ids in torch.randperm(count,device=self.env.device).chunk(self.settings.mini_batches):
                losses.append(self.algorithm.update_batch(tree_map(lambda x:x[ids],batch)))
        if str(self.env.device).startswith('cuda'): torch.cuda.synchronize()
        end=time.perf_counter()
        self.iteration+=1
        metrics.update({k:sum(x[k] for x in losses)/len(losses) for k in losses[0]})
        metrics.update(iteration=self.iteration,stage=self.curriculum.stage[0],
            sample_seconds=sample_end-start,update_seconds=end-sample_end,
            transitions_per_second=count/(end-start),
            peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        return metrics

    def save(self, directory):
        path=Path(directory)/f'checkpoint_{self.iteration:06d}.pt'
        save_checkpoint(path,self.model,self.algorithm.optimizer,asdict(self.curriculum),
                        self.iteration,{**self.config,'privileged_dim':self.env.privileged_dim,'best_score':self.best_score},update_complete=True)
        pointer=Path(directory)/'last.json'
        pointer.write_text(json.dumps({'checkpoint':path.name}))
        return path

    def learn(self, validate):
        out=Path(self.settings.output_dir)
        out.mkdir(parents=True,exist_ok=False)
        (out/'config.json').write_text(json.dumps(self.config,indent=2))
        (out/'git_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
        (out/'worktree.patch').write_text(subprocess.check_output(['git','diff','HEAD'],text=True))
        (out/'dependencies.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
        if hasattr(self,'resume_event'):
            (out/'resume_reset.json').write_text(json.dumps({'event':self.resume_event,'iteration':self.iteration}))
        while self.iteration < self.settings.max_iterations and not (self.curriculum.stopped or self.curriculum.complete):
            metrics=self.update()
            due=self.curriculum.advance_iteration()
            old_stage=self.curriculum.stage[0]
            if due:
                rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
                try:
                    passed,report=validate(self.model,self.curriculum.stage,self.iteration)
                finally:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                    if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])
                (out/f'validation_{self.iteration:06d}.json').write_text(json.dumps(report,indent=2))
                scores=[v['fraction'] for v in report.values()]
                errors=[v['speed'].get('mae',0. if v['speed']['passed'] else float('inf')) for v in report.values()]
                score=[min(scores),sum(scores)/len(scores),-sum(errors)/len(errors)]
                improved=old_stage not in self.best_score or score>self.best_score[old_stage]
                if improved: self.best_score[old_stage]=score
                self.curriculum.validate(passed)
            if self.iteration % self.settings.save_interval == 0:
                saved=self.save(out)
                if due and improved:
                    record={'checkpoint':saved.name,'score':self.best_score[old_stage],'stage':old_stage}
                    (out/f'best_{old_stage}.json').write_text(json.dumps(record))
                    if old_stage in ('P4','P5'): (out/'best.json').write_text(json.dumps(record))
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
            if self.curriculum.stage[0] != old_stage:
                try:
                    self.env.set_stage(self.curriculum.stage[0]); self._reset()
                except Exception as error:
                    (out/'blocked.json').write_text(json.dumps({'iteration':self.iteration,'stage':self.curriculum.stage[0],'reason':str(error),'checkpoint':saved.name}))
                    raise
        (out/'completion.json').write_text(json.dumps({'iteration':self.iteration,'curriculum':asdict(self.curriculum)}))
