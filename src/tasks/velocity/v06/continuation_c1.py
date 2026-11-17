"""One-time, pure stop4224 -> C1 state migration; never alters the source file."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch

from . import spec, recovery, model as model_module
from .c1_contract import (CONTINUATION_ID, PLAN_SHA256, SOURCE_PATH, SOURCE_SHA256,
                          SOURCE_CLOSURE)

# Full-tree identity of the byte-verified designated source; also binds every
# Adam parameter-ID association and all six original RNG/metadata blocks.
SOURCE_STATE_DIGEST = 'a8cdf8f03a516838f5131adff0ac789fd449adb6db3ffd14538254a512a039cf'


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


def migrate_stop_4224(state, *, source_sha256):
    """Return a deep-copied state + explicit diff, without I/O or RNG changes.

    The new C1 checkpoint uses normal recovery and cannot re-enter migration.
    """
    def require(condition, reason):
        if not condition:
            raise recovery.RecoveryBlocked('C1 migration refused: ' + reason)
    require(getattr(spec, 'CONTINUATION_ID', None) == CONTINUATION_ID,
            'C1 must be selected before imports')
    require(source_sha256 == SOURCE_SHA256, 'not the designated checkpoint SHA')
    require(isinstance(state, dict), 'state is not a checkpoint mapping')
    original = {k: v for k, v in state.items() if k != 'checkpoint_sha256'}
    identity = original.get('identity') or {}; ledger = original.get('ledger') or {}
    learning = original.get('learning') or {}; curriculum = original.get('curriculum') or {}
    require(identity.get('recipe') == '0.6-go2refs-r1' and identity.get('schema') == 8
            and identity.get('architecture') == 'v06-route-state-v1',
            'source identity or already migrated state')
    require(ledger.get('stop_state') == 'gate_failed'
            and ledger.get('stop_reason') == 'substage_budget_exhausted',
            'only the exact substage budget stop is authorized')
    require(all(ledger.get(k) is None for k in
                ['pending_boundary', 'pending_update', 'incomplete_update']),
            'pending or partial PPO state')
    require(learning == dict(update=4224, optimizer_step=84480, adam_steps=84480,
        control_samples=103809024, physics_substeps=415236096, num_envs=512,
        rollout=48, epochs=5, minibatches=4, learning_rate=1e-5),
        'learning counts or actual learning rate differ')
    require((curriculum.get('phase'), curriculum.get('substage'), curriculum.get('level_tick'),
        curriculum.get('frontier_tile_id'), curriculum.get('generation')) == ('D','D25',100,17,67),
        'course identity differs')
    require(curriculum.get('finished') is True and curriculum.get('stop_status') == 'budget_exhausted',
            'not the authorized finished budget state')
    require((original.get('source') or {}).get('source_closure_sha256') == SOURCE_CLOSURE,
            'source closure differs')
    model_module.validate_initialization(original.get('initialization'))
    require(tree_digest(original) == SOURCE_STATE_DIGEST,
            'the state content differs from the exact source')
    new = deepcopy(original); cur = new['curriculum']; changed = {}
    for key, value in dict(finished=False, stop_status=None, generation=68, generation_updates=0,
        ema=dict(s=None,f=None,q=None,batches=0), episode_fifo=[], ready_streak=0,
        hard_streak=0, gate_streak=0, minimum_level_failure_streak=0, cooldown=5,
        window_metrics={}).items():
        changed['curriculum.' + key] = dict(before=deepcopy(cur.get(key)), after=deepcopy(value))
        cur[key] = deepcopy(value)
    new['gate']['streak'] = 0
    new['gate']['identity']['recipe'] = spec.RECIPE_VERSION
    new['pending_transition'] = None
    for key, value in dict(stop_state=None, stop_reason=None, error_type=None,
                          lifecycle_state='running', terminal=None).items():
        new['ledger'][key] = value
    # Eligibility belongs to the current generation; its history stays intact.
    diagnostic = new['ledger'].get('diagnostic_collapse') or {}
    diagnostic.update(streak=0, generation=68, last=None)
    new['ledger']['diagnostic_collapse'] = diagnostic
    parent = dict(checkpoint_path=SOURCE_PATH, checkpoint_sha256=SOURCE_SHA256,
        identity=deepcopy(original['identity']), source=deepcopy(original['source']),
        contract_sha256=original['contract_sha256'], recipe=deepcopy(original['recipe']),
        stopped_terminal=deepcopy(original['ledger'].get('terminal')),
        pending_transition=deepcopy(original.get('pending_transition')))
    new['identity'] = recovery.identity_block()
    new['contract_sha256'] = recovery.contract_sha(spec.contract())
    new['recipe'] = spec.recipe_config()
    new['source'] = recovery.source_block(seed=42, device='cuda:0', mode='resume')
    new['source']['continuation'] = dict(continuation_id=CONTINUATION_ID, plan_sha256=PLAN_SHA256,
        parent=parent, rollback_from_update=4224, resume_from_update=4224, lost_updates=0,
        replay_billed_updates=False, generation_before=67, generation_after=68,
        full_evaluation_rng_isolation='not_established_Q008')
    # Old case tables and records retain old identities as historical evidence.
    diff = dict(schema='a2-v06-c1-migration-diff-v1', continuation_id=CONTINUATION_ID,
        source_checkpoint_sha256=SOURCE_SHA256, source_state_digest=SOURCE_STATE_DIGEST,
        plan_sha256=PLAN_SHA256, new_source_closure_sha256=new['source']['source_closure_sha256'],
        changes=changed, runner_changes=dict(gate_streak=[original['gate']['streak'],0],
            pending_transition=[deepcopy(original.get('pending_transition')),None],
            stop_state=['gate_failed',None], stop_reason=['substage_budget_exhausted',None]),
        budget_before=dict(D25=1500,D20=1500,D=3000,total=30000),
        budget_after=dict(D25=4500,D20=4500,D=9000,total=36000,formal=36100),
        preserved={k:tree_digest(original[k]) for k in ['model','optimizer','normalizer','learning','rng']},
        no_pending_replay=True, first_new_update=4225, source_provenance=parent)
    diff['envelope_changes'] = {key:dict(before=deepcopy(original[key]),after=deepcopy(new[key]))
        for key in ['identity','contract_sha256','recipe','source']}
    diff['gate_identity_change'] = dict(before=deepcopy(original['gate']['identity']),
                                        after=deepcopy(new['gate']['identity']))
    diff['ledger_changes'] = {key:dict(before=deepcopy(original['ledger'].get(key)),
                                      after=deepcopy(new['ledger'].get(key)))
        for key in ['stop_state','stop_reason','error_type','lifecycle_state','terminal',
                    'diagnostic_collapse']}
    for key, want in diff['preserved'].items():
        require(tree_digest(new[key]) == want, 'preserved '+key+' changed')
    recovery.validate_checkpoint(new, num_envs=512)
    return new, diff


def load_designated_source(path=SOURCE_PATH, *, source_sha256=SOURCE_SHA256):
    """Check actual bytes before migration; no generalized stopped-run resume."""
    if source_sha256 != SOURCE_SHA256 or recovery.sha256(path) != SOURCE_SHA256:
        raise recovery.RecoveryBlocked('C1 requires the byte-verified stop004224 source')
    return torch.load(path, map_location='cpu', weights_only=False)
