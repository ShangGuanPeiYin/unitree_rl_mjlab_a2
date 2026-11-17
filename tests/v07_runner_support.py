"""Native real-model/real-nonempty-Adam checkpoint fixtures; rollouts synthetic."""
from copy import deepcopy
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import model,ppo,recovery,curriculum,runner,continuation_v07 as migration

class RecordingEnv:
    device='cpu';num_envs=512
    def __init__(self):self.events=[];self._rng=dict(env=[1]);self._sampler=dict(sampler=[2])
    def rng_state(self):return deepcopy(self._rng)
    def sampler_rng_state(self):return deepcopy(self._sampler)
    def load_rng_state(self,value):self._rng=deepcopy(value)
    def load_sampler_rng_state(self,value):self._sampler=deepcopy(value)
    def set_schedule(self,schedule,iteration=None):
        self.events.append(('schedule',schedule.generation,iteration));self.generation=schedule.generation;self.stage_iteration=iteration
    def reset(self):self.events.append(('reset',self.generation,self.stage_iteration));return dict(proprio=torch.zeros(512,45)),{}

def bare_runner(path,case_runner=None,require=False):
    m=model.V07ActorCritic();env=RecordingEnv();p=ppo.PPO(m,env)
    c=curriculum.CurriculumManager(num_envs=512,seed=42)
    r=runner.V07Runner(model=m,ppo=p,curriculum=c,env=env,directory=path,device='cpu',
        case_runner=case_runner,require_resume_evidence=require)
    return r

def migrated_checkpoint(path):
    source=migration.load_designated_source(ROOT/migration.SOURCE_PATH)
    state,_=migration.migrate_stop_4817(source,source_sha256=migration.SOURCE_SHA256)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(state,path)
    return dict(path=str(path),sha256=recovery.sha256(path))

def loaded_runner(path,checkpoint):
    r=bare_runner(path);r.load(checkpoint['path'],checkpoint['sha256']);return r

def set_synthetic_boundary(r,update=6100,substage='D20',generation=73,gu=1283):
    """Historical count setup only; no claim that these updates were executed."""
    state=r.curriculum.state_dict();state.update(phase='D',substage=substage,frontier_tile_id={'D25':17,'D22':22,'D20':18}[substage],
        generation=generation,generation_updates=gu,last_update=update)
    state['spent_by_phase'].update(F0=200,H=2500,D=update-2700)
    state['spent_by_substage'].update(D25=2083,D22=0,D20=update-4783)
    r.curriculum.load_state_dict(state);r.curriculum.load_regression_policy_state(__import__('src.tasks.velocity.v07.regression_policy',fromlist=['new_state']).new_state(generation))
    r.ppo.learning_state.update(completed_ppo_updates=update,optimizer_steps=update*20,optimizer_initialized=True)
    for value in r.ppo.optimizer.state.values():value['step'].fill_(update*20)
    r.update=update;r._sync_phase()
