"""The real stop4224 state is the only authorized C1 migration fixture."""
from test_v06_c1_contract import run_c1

PRELUDE = '''
from copy import deepcopy
import torch
from src.tasks.velocity.v06 import continuation_c1 as c1, recovery, spec
source=torch.load(c1.SOURCE_PATH,map_location='cpu',weights_only=False)
'''


def test_c1_migration_preserves_learning_state_and_resets_only_current_eligibility():
    run_c1(PRELUDE + '''
before=c1.tree_digest(source)
new,diff=c1.migrate_stop_4224(source,source_sha256=c1.SOURCE_SHA256)
assert c1.tree_digest(source)==before
for key in ['model','optimizer','normalizer','learning','rng']:
    assert c1.tree_digest(source[key])==c1.tree_digest(new[key]),key
assert new['initialization']['transfer_metadata'] is not source['initialization']['transfer_metadata']
assert new['curriculum']['generation']==68
assert new['curriculum']['generation_updates']==0
assert new['curriculum']['spent_by_phase']==source['curriculum']['spent_by_phase']
assert new['curriculum']['spent_by_substage']==source['curriculum']['spent_by_substage']
assert new['curriculum']['ema']==dict(s=None,f=None,q=None,batches=0)
assert new['curriculum']['episode_fifo']==[]
assert new['gate']['streak']==0 and new['pending_transition'] is None
assert new['gate']['records']==source['gate']['records']
assert new['curriculum']['gate_log']==source['curriculum']['gate_log']
assert new['curriculum']['retention_next_due']==dict(flat=4400,h00=4400)
assert new['curriculum']['retention_qualified_at']==dict(flat=200,h00=400)
assert new['curriculum']['retention_last_completed']==dict(flat=4200,h00=4200)
assert new['learning']['learning_rate']==1e-5
recovery.validate_checkpoint(new,num_envs=512)
recovery.resume_contract(new,num_envs=512)
try: c1.migrate_stop_4224(new,source_sha256=c1.SOURCE_SHA256)
except recovery.RecoveryBlocked: pass
else: raise AssertionError('C1 migration happened twice')
''')


def test_c1_rejects_every_non_designated_or_incomplete_source():
    run_c1(PRELUDE + '''
mutations=[
('wrong_sha',lambda x:None),
('user_stop',lambda x:x['ledger'].update(stop_state='user_requested_stop')),
('regression',lambda x:x['ledger'].update(stop_state='behavior_regression')),
('reason',lambda x:x['ledger'].update(stop_reason='phase_budget_exhausted')),
('pending',lambda x:x['ledger'].update(pending_update=4224)),
('partial',lambda x:x['ledger'].update(incomplete_update={'update':4225,'resumable':False})),
('update',lambda x:x['learning'].update(update=4223)),
('adam',lambda x:x['learning'].update(adam_steps=84479)),
('samples',lambda x:x['learning'].update(control_samples=103809023)),
('spent',lambda x:x['curriculum']['spent_by_substage'].update(D20=23)),
('metadata',lambda x:x['initialization'].pop('transfer_metadata')),
('eth',lambda x:x['initialization']['transfer_metadata'].update(sha256='0'*64)),
('lr',lambda x:x['optimizer']['param_groups'][0].update(lr=.0003)),
('association',lambda x:x['optimizer']['param_groups'][0]['params'].reverse()),
]
for label,change in mutations:
    state=deepcopy(source);change(state)
    try: c1.migrate_stop_4224(state,source_sha256=('0'*64 if label=='wrong_sha' else c1.SOURCE_SHA256))
    except (recovery.RecoveryBlocked,ValueError): pass
    else: raise AssertionError('accepted '+label)
''')


def test_c1_generation_gate_minimum_and_existing_retention_clock():
    run_c1(PRELUDE + '''
from src.tasks.velocity.v06.curriculum import CurriculumManager
new,_=c1.migrate_stop_4224(source,source_sha256=c1.SOURCE_SHA256)
m=CurriculumManager(num_envs=512,seed=42,state=new['curriculum'])
# Only legitimate complete learning boundaries are billed; no EMA windows here.
for u in range(4225,4301):m.on_update(u)
assert m.state.generation_updates==76
m.record_gate(passed=True,update=4300)
assert m.state.gate_streak==0 and m.state.substage=='D25'
for u in range(4301,4401):m.on_update(u)
assert m.retention_due(4400)==('flat','h00')
''')


def test_c1_worker_bare_model_loaded_ready_and_first_new_update():
    run_c1(PRELUDE + '''
import sys,tempfile
from pathlib import Path
sys.path.insert(0,'tests')
from test_v06_q006 import Q006Env,Q006PPO
from src.tasks.velocity.v06 import c1_runtime
import train_a2_ideal_stairs_v06 as worker
from src.tasks.velocity.v06 import ppo as ppo_module
class TwentyAdam(Q006PPO):
    def update(self):
        for _ in range(20):self._adam_step()
        self.learning_state['completed_ppo_updates']+=1
        self.learning_state['optimizer_steps']+=20
        return dict(loss=1.,kl=.01,update=self.learning_state['completed_ppo_updates'])
class RecordingEnv(Q006Env):
    stage_iteration=0
    generation=0
    def set_schedule(self,schedule,iteration=None):
        super().set_schedule(schedule,iteration)
        self.stage_iteration=int(iteration or 0);self.generation=schedule.generation
        self.phase=schedule.phase;self.substage=schedule.substage
        self.roles=schedule.roles;self.tile_ids=schedule.tile_ids
worker.build_environment=lambda contract,**kw:RecordingEnv(512)
worker.build_telemetry=lambda *args,**kw:None
worker.build_case_runner=lambda *args,**kw:None
ppo_module.PPOTrainer=lambda model,env:TwentyAdam(model,num_envs=512)
with tempfile.TemporaryDirectory() as tmp:
    runner,receipt=c1_runtime.prepare_runner(dict(spec.recipe_config(),device='cpu'),Path(tmp))
    assert runner.resume_loaded['verification']['aligned']
    assert runner.resume_ready['verification']['aligned']
    assert runner.update==4224 and runner.curriculum.state.generation==68
    assert runner.env.stage_iteration==1524 and runner.env.generation==68
    assert runner.env.phase=='D' and runner.env.substage=='D25'
    assert runner.ppo.optimizer.param_groups[0]['lr']==1e-5
    loaded=recovery.preload_checkpoint(runner.resume_loaded['path'],mode='resume')
    ready=recovery.preload_checkpoint(runner.resume_ready['path'],mode='resume')
    for state in [loaded,ready]:
        for key in ['model','optimizer','normalizer','learning']:
            assert c1.tree_digest(state[key])==c1.tree_digest(source[key]),key
        assert state['source']['continuation']['parent']['checkpoint_sha256']==c1.SOURCE_SHA256
    assert c1.tree_digest(runner.model.transfer_metadata)==c1.tree_digest(source['initialization']['transfer_metadata'])
    assert runner.model.transfer_metadata is not source['initialization']['transfer_metadata']
    out=runner.train_one_update()
    assert out['update']==4225 and runner.learning['adam_steps']==84500
    assert runner.learning['control_samples']==103833600
    assert {int(x['step'].item()) for x in runner.ppo.optimizer.state.values()}=={84500}
    new=runner.save_verified('engineering_4225')
    reloaded=recovery.preload_checkpoint(new['path'],mode='resume')
    assert reloaded['curriculum']['generation']==68
    assert reloaded['source']['continuation']['parent']['checkpoint_sha256']==c1.SOURCE_SHA256
    from src.tasks.velocity.v06.runner import V06Runner
    from src.tasks.velocity.v06.curriculum import CurriculumManager
    from src.tasks.velocity.v06.model import V06ActorCritic
    bare2=V06ActorCritic()
    ppo2=TwentyAdam(bare2,num_envs=512)
    runner2=V06Runner(model=bare2,ppo=ppo2,env=RecordingEnv(512),
        curriculum=CurriculumManager(num_envs=512,seed=42),directory=Path(tmp)/'second',
        seed=42,device='cpu',mode='resume',require_resume_evidence=True)
    runner2.load(new['path'],new['sha256'])
    assert runner2.update==4225 and runner2.curriculum.state.generation==68
    assert runner2.env.stage_iteration==1525 and runner2.env.generation==68
    assert runner2.source['continuation']['parent']['checkpoint_sha256']==c1.SOURCE_SHA256
    second=runner2.save_verified('resume_loaded')
    try:runner2._check_resume_evidence(stage='learning')
    except Exception as error:assert 'resume_ready' in str(error)
    else:raise AssertionError('C1 second load bypassed ready')
    runner2.save_verified('resume_ready')
    runner2._check_resume_evidence(stage='learning')
    stopped=deepcopy(reloaded);stopped.pop('checkpoint_sha256',None)
    stopped['ledger']['stop_state']='user_requested_stop'
    stopped_path=Path(tmp)/'user_stop.pt';recovery.atomic_save(stopped_path,stopped)
    try:runner2.load(stopped_path)
    except recovery.RecoveryBlocked:pass
    else:raise AssertionError('second resume reopened a new user stop')
''')


def test_c1_real_environment_has_no_training_override_and_can_switch_frontier():
    run_c1(PRELUDE + '''
import tempfile
from pathlib import Path
from collections import Counter
import train_a2_ideal_stairs_v06 as worker
from src.tasks.velocity.v06 import c1_runtime
from src.tasks.velocity.v06.curriculum import CurriculumManager,allocate_roles
new,_=c1.migrate_stop_4224(source,source_sha256=c1.SOURCE_SHA256)
original_build=worker.build_environment
class EnvironmentChecked(Exception):pass
def check_environment(contract,**kwargs):
    actual=original_build(dict(contract,num_envs=8,device='cpu'),**kwargs)
    assert actual._tile_override is None, 'training env accidentally pinned to one tile'
    manager=CurriculumManager(num_envs=8,seed=42,state=new['curriculum'])
    schedule=manager.schedule();actual.set_schedule(schedule,1524);actual.reset()
    assert actual.generation==68 and actual.stage_iteration==1524
    assert Counter(actual.roles)==Counter(allocate_roles(8,'D'))
    assert Counter(allocate_roles(512,'D'))==dict(flat=102,frontier=164,near=123,easy=123)
    def check_tiles(env,frontier):
        for role,tile in zip(env.roles,env.tile_ids.tolist()):
            assert (tile==0 if role=='flat' else tile==frontier if role=='frontier'
                    else tile in [frontier-2,frontier-1] if role=='near'
                    else 1<=tile<=frontier-3), (role,tile,frontier)
    check_tiles(actual,17)
    manager.state.substage='D20';manager.state.frontier_tile_id=18;manager.state.generation=69
    actual.set_schedule(manager.schedule(),1525);actual.reset()
    assert actual.substage=='D20' and actual.frontier_tile_id==18 and actual.generation==69
    check_tiles(actual,18)
    assert .3<=float(actual.command[:,0].min())<=float(actual.command[:,0].max())<=.6
    evaluation=original_build(dict(contract,num_envs=4,device='cpu'),phase='D',substage='D25',
        evaluation=True,frontier_tile_id=17)
    evaluation.reset();assert evaluation._tile_override==17
    assert set(evaluation.tile_ids.tolist())=={17}
    raise EnvironmentChecked()
worker.build_environment=check_environment
with tempfile.TemporaryDirectory() as tmp:
    try:c1_runtime.prepare_runner(dict(spec.recipe_config(),device='cpu'),Path(tmp))
    except EnvironmentChecked:pass
    else:raise AssertionError('real environment check did not run')
''')
