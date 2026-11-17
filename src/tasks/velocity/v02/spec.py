"""Versioned SDS revision 1.1 configuration; isolated from the legacy task."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

CODE_VERSION = '0.2.0'
RECIPE_VERSION = '0.2'
SCHEMA_VERSION = 2
SDS_REVISION = '1.1'
TASK = 'Unitree-A2-IdealHeightMap-Stairs-V02'
ROOT = Path(__file__).resolve().parents[4]
S0_PATH = str(ROOT / 'logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/chunk_0002_n512/checkpoints/stage_1_start.pt')
S0_SHA256 = '7db7ea4f65d94aa762c6101bb1a8a06f61aa26e78af0622b7420712854f5efdb'

@dataclass(frozen=True)
class Stage:
    id: str
    tile_id: int
    N: int
    h: float
    budget: int

STAGES = tuple(Stage(*v) for v in (
    ('flat',0,0,0.,0), ('P1',1,1,.02,1000), ('P2',2,1,.04,1000),
    ('P3',3,1,.06,1000), ('P4',4,1,.08,2000), ('S1',5,4,.08,3000),
    ('S2',6,8,.12,4000), ('S3',7,12,.16,4000), ('S4',8,20,.20,5000),
    ('S5',9,40,.20,7000)))
STAGE_BY_ID = {s.id:s for s in STAGES}

def recipe_config(recipe):
    if recipe not in ('A','B','C','D'):
        raise ValueError('Recipe must be A/B/C/D')
    return dict(recipe=recipe, new_rewards=recipe in ('C','D'),
                std_min=.10 if recipe=='D' else .05, std_max=.30,
                initial_std_floor=.20 if recipe=='D' else None,
                entropy=.005, progressive=recipe!='A')

def formal_stages(recipe):
    recipe_config(recipe)
    return [s.id for s in STAGES[4 if recipe=='A' else 1:]]

def comparison_stage(recipe, completed_updates):
    recipe_config(recipe)
    if not 0 <= completed_updates <= 2000:
        raise ValueError('Comparison budget outside 0..2000')
    if recipe=='A': return 'P4'
    return ('P1','P2','P3','P4')[min(completed_updates//400,3)]

def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def behavior_hashes():
    # Imported legacy algorithms remain immutable and are covered transitively.
    sources=list((ROOT/'src/tasks/velocity/v02').glob('*.py'))
    sources += [ROOT/p for p in (
        'src/tasks/velocity/rl/ideal_heightmap_actor_critic.py',
        'src/tasks/velocity/rl/ideal_stairs_runner.py',
        'src/tasks/velocity/rl/ideal_stairs_resources.py',
        'src/tasks/velocity/mdp/ideal_stairs_env.py',
        'src/tasks/velocity/mdp/ideal_stairs_contacts.py',
        'src/tasks/velocity/mdp/ideal_heightmap.py',
        'src/tasks/velocity/config/a2/ideal_stairs_env_cfg.py',
        'src/tasks/velocity/config/a2/eth_env_cfg.py',
        'src/tasks/velocity/config/a2/__init__.py',
        'src/terrains/ideal_stairs_atlas.py', 'src/terrains/open_stairs.py',
        'scripts/train.py', 'scripts/play.py',
        'scripts/evaluate_a2_ideal_stairs.py',
        'scripts/train_a2_ideal_stairs_v02.py',
        'scripts/evaluate_a2_ideal_stairs_v02.py',
        'scripts/run_a2_ideal_stairs_v02.py',
        'scripts/play_a2_ideal_stairs_v02.py')]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(sources)}

def contract(recipe, mode):
    if mode not in ('comparison','formal'): raise ValueError('Invalid training mode')
    return dict(code_version=CODE_VERSION,recipe_version=RECIPE_VERSION,
        checkpoint_schema=SCHEMA_VERSION,sds_revision=SDS_REVISION,task=TASK,
        recipe=recipe_config(recipe),mode=mode,source_s0_sha256=S0_SHA256,
        stages=[asdict(s) for s in STAGES],behavior_sha256=behavior_hashes(),
        observation={'proprio':[45],'history':[5,45],'heightmap':[3,48,32],'privileged':[35]},
        physics_dt=.005,control_dt=.02,map_hz=10,action_scale=.25,
        rollout=48,epochs=5,minibatches=4,save_interval=100,
        formal_eval_interval=200,comparison_eval_updates=[0,1800,2000],
        comparison_budget=2000,initialization='explicit_actor_only_s0',resume_mode='new_episode')
