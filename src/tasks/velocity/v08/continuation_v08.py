"""Attested6400 migration: preserve learning state; create a new v08 course."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
import torch

SOURCE_PATH='/home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt'
SOURCE_SHA256='30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2'
SOURCE_STATE_DIGEST='390170c2a41c329994db3094ad561b9c5e28d20a519d6de6fa2d201a6f70ea23'
SDS_SHA256='8aca6444581f0fa4c3f62e82061c2a47b00d98c72df48add93b4168138189afe'
CONTINUATION_ID='v08-stairs-walk-stop-reverse-6400'
PRESERVED=('model','optimizer','normalizer','learning','rng')

def tree_digest(value):
    digest = hashlib.sha256()

    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous()
            digest.update(b'tensor'); digest.update(str(a.dtype).encode())
            digest.update(str(tuple(a.shape)).encode()); digest.update(a.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(b'ndarray'); digest.update(str(x.dtype).encode())
            digest.update(str(x.shape).encode()); digest.update(x.tobytes())
        elif isinstance(x, dict):
            digest.update(b'dict')
            for k in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(k); visit(x[k])
            digest.update(b'end')
        elif isinstance(x, (tuple, list)):
            digest.update(type(x).__name__.encode())
            for item in x: visit(item)
            digest.update(b'end')
        elif isinstance(x, np.generic): visit(x.item())
        elif x is None or isinstance(x, (str, int, float, bool)):
            digest.update(type(x).__name__.encode())
            digest.update(json.dumps(x, ensure_ascii=False, allow_nan=False).encode())
            digest.update(b';')
        else: raise ValueError(f'unsupported state type {type(x)}')
    visit(value)
    return digest.hexdigest()


def load_designated_source(path=SOURCE_PATH, *, source_sha256=SOURCE_SHA256):
    path=Path(path)
    if path.resolve()!=Path(SOURCE_PATH).resolve() or source_sha256!=SOURCE_SHA256:
        raise ValueError('only the canonical source6400 is authorized')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError('source6400 SHA differs')
    return torch.load(path,map_location='cpu',weights_only=False)

def migrate_stop_6400(state, *, source_sha256):
    from . import spec
    from .curriculum import CurriculumManager
    original={k:v for k,v in state.items() if k!='checkpoint_sha256'}
    if source_sha256!=SOURCE_SHA256 or tree_digest(original)!=SOURCE_STATE_DIGEST:
        raise ValueError('6400 source state differs, including Adam and RNG')
    new={k:deepcopy(original[k]) for k in PRESERVED}
    new.update(schema=10,identity=dict(task=spec.TASK_ID,recipe=spec.RECIPE_VERSION),
        curriculum=CurriculumManager(num_envs=512,seed=42).state_dict(),
        ledger=dict(pending_boundary=None,pending_update=None,formal_updates=0,engineering_updates=0,
            control_samples=0,physics_substeps=0,evaluation_control_samples=0,events=[],
            stop_state=None,error_type=None),
        source=dict(checkpoint_path=SOURCE_PATH,checkpoint_sha256=SOURCE_SHA256,
            parent=dict(terminal=deepcopy(original['ledger']['terminal']),
                ledger=deepcopy(original['ledger']),pending_transition=deepcopy(original.get('pending_transition')))),
        initialization=deepcopy(original['initialization']),evaluation={},resume_mode='fresh_physical_episode',
        recipe=spec.contract(),environment=None,freeze=None,mode=None)
    diff=dict(first_new_update=6401,source_sha256=SOURCE_SHA256,source_state_digest=SOURCE_STATE_DIGEST,
        no_pending_replay=True,preserved={k:tree_digest(new[k]) for k in PRESERVED},
        recovery='saved compatible RNG followed by one schedule and fresh physical episode')
    return new,diff
