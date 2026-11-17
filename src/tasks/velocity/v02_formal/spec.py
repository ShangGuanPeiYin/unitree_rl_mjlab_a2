"""Frozen SDS Formal 1.2 identities and recipe; legacy sources remain immutable."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

CODE_VERSION = '0.2.1'
RECIPE_VERSION = '0.2'
SCHEMA_VERSION = 3
SDS_REVISION = '1.2'
TASK = 'Unitree-A2-IdealHeightMap-Stairs-V02-Formal'
PROTOCOL_ID = 'v02-formal-climb-v1'
ROOT = Path(__file__).resolve().parents[4]
INIT_PATH = str(ROOT/'logs/a2_ideal_stairs_v02/local_sds_20261111_1621/comparisons/C/chunk_0010_n512/checkpoints/budget_exhausted.pt')
INIT_SHA256 = '7d94810bb86e6997fe2a0f03883a6a0e4ee594140bb3ff65deb86642c3938d9a'

@dataclass(frozen=True)
class Stage:
    id: str
    tile_id: int
    N: int
    h: float
    budget: int

STAGES = tuple(Stage(*v) for v in (('flat',0,0,0.,0), ('R0',4,1,.08,3000), ('S1',5,4,.08,3000), ('S2',6,8,.12,4000), ('S3',7,12,.16,4000), ('S4',8,20,.20,5000), ('S5',9,40,.20,7000)))
STAGE_BY_ID = {s.id:s for s in STAGES}
STAGE_BY_TILE = {s.tile_id:s for s in STAGES}
SLOT_CLASSES = ('flat_walk','flat_stand','previous_stairs','current_stairs')

def sha256(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest()

def behavior_hashes():
    from ..v02.spec import behavior_hashes as legacy_hashes
    out=legacy_hashes()
    paths=list((ROOT/'src/tasks/velocity/v02_formal').glob('*.py'))
    paths+=list((ROOT/'scripts').glob('*v02_formal*.py'))
    out.update({str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)})
    return out

def recipe_config(recipe='C'):
    if recipe!='C': raise ValueError('Formal protocol requires unchanged C recipe')
    return dict(recipe='C',new_rewards=True,std_min=.05,std_max=.30,initial_std_floor=None,entropy=.005,progressive=True)

def contract(recipe='C',mode='formal'):
    if mode!='formal': raise ValueError('Formal protocol only')
    return dict(code_version=CODE_VERSION,recipe_version=RECIPE_VERSION,checkpoint_schema=SCHEMA_VERSION,sds_revision=SDS_REVISION,task=TASK,protocol_id=PROTOCOL_ID,
        recipe=recipe_config(recipe),mode=mode,source_c_sha256=INIT_SHA256,stages=[asdict(s) for s in STAGES],behavior_sha256=behavior_hashes(),
        observation={'proprio':[45],'history':[5,45],'heightmap':[3,48,32],'privileged':[35]},physics_dt=.005,control_dt=.02,map_hz=10,action_scale=.25,
        rollout=48,epochs=5,minibatches=4,save_interval=100,formal_eval_interval=200,initialization='explicit_actor_only_C',resume_mode='new_episode',
        seed=42,vx_range=[.2,.6],vy=0.,wz=0.,initial_lr=1e-4,lr_min=1e-5,lr_max=3e-4,desired_kl=.01,clip=.2,gamma=.99,gae_lambda=.95,value_coefficient=1.,max_grad_norm=1.,clipped_value_loss=True,
        total_budget=26000,slot_classes=list(SLOT_CLASSES),mix_R0=[.4,.1,0.,.5],mix_later=[.25,.05,.2,.5],success='safe_dynamic_platform_climb',parking_in_training=False)
