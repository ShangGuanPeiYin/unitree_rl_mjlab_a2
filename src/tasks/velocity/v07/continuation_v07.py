"""One-time native schema9 migration of the uniquely attested stop4817."""
from copy import deepcopy
import hashlib,json
import numpy as np
import torch
from . import spec,recovery,regression_policy

CONTINUATION_ID='v07-d22-cautious-curriculum'
PLAN_SHA256='303ae40a5c5e3fe82f9f0df3aae8d153136e016439087c8db9701aeb17197aca'
SOURCE_PATH='logs/a2_ideal_stairs_v06_c1/continue_seed42_from004224_trial01/checkpoints/stop_004817.pt'
SOURCE_SHA256='a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64'
SOURCE_CLOSURE='eae063c549d310f66a1931ec49ea3a22d16f3d6bcc0007a045e1779e94d7b54b'
SOURCE_STATE_DIGEST='07224e3c8bb0ff3bccb1cd063b7260fc78636f91c808927fa9a228ddc9afcd81'

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

def load_designated_source(path=SOURCE_PATH,*,source_sha256=SOURCE_SHA256):
    if source_sha256!=SOURCE_SHA256 or recovery.sha256(path)!=SOURCE_SHA256:
        raise recovery.RecoveryBlocked('v07 requires the exact byte-verified stop004817 source')
    return torch.load(path,map_location='cpu',weights_only=False)

def migrate_stop_4817(state,*,source_sha256):
    def require(test,reason):
        if not test:raise recovery.RecoveryBlocked('v07 migration refused: '+reason)
    require(spec.SCHEMA_VERSION==9 and spec.RECIPE_VERSION=='0.7-d22-eval-regression-r1','native v07 contract is not active')
    require(source_sha256==SOURCE_SHA256,'not the designated SHA')
    require(isinstance(state,dict),'checkpoint mapping missing')
    original={k:v for k,v in state.items() if k!='checkpoint_sha256'}
    identity=original.get('identity') or {};ledger=original.get('ledger') or {};cur=original.get('curriculum') or {}
    require(original.get('schema')==8 and identity.get('recipe')=='0.6-go2refs-c1' and identity.get('architecture')=='v06-route-state-v1','not the original schema8 C1 identity')
    require(ledger.get('stop_state')=='curriculum_stalled' and ledger.get('stop_reason')=='curriculum_stalled','only the exact minimum-level stop is authorized')
    require(all(ledger.get(k) is None for k in ['pending_update','pending_boundary','incomplete_update','evaluation_transaction']),'pending or partial learning/evaluation')
    require(original.get('learning')==dict(update=4817,optimizer_step=96340,adam_steps=96340,control_samples=118382592,physics_substeps=473530368,num_envs=512,rollout=48,epochs=5,minibatches=4,learning_rate=7.593750000000002e-05),'learning counters or actual lr differ')
    require((cur.get('phase'),cur.get('substage'),cur.get('frontier_tile_id'),cur.get('level_tick'),cur.get('generation'))==('D','D25',17,100,72),'source course differs')
    require(cur.get('finished') is True and cur.get('stop_status')=='minimum_level_failure','source terminal differs')
    require((original.get('source') or {}).get('source_closure_sha256')==SOURCE_CLOSURE,'source closure differs')
    require(tree_digest(original)==SOURCE_STATE_DIGEST,'state tree differs, including Adam ID/metadata/RNG associations')
    new=deepcopy(original);new['schema']=9;current=new['curriculum'];changes={}
    for key,value in dict(finished=False,stop_status=None,generation=73,generation_updates=0,
        ema=dict(s=None,f=None,q=None,batches=0),episode_fifo=[],ready_streak=0,hard_streak=0,gate_streak=0,
        minimum_level_failure_streak=0,cooldown=spec.COOLDOWN_CHECKS,window_metrics={}).items():
        changes['curriculum.'+key]=dict(before=deepcopy(current.get(key)),after=deepcopy(value));current[key]=deepcopy(value)
    current['spent_by_substage']['D22']=0
    changes['curriculum.spent_by_substage.D22']=dict(before=None,after=0)
    new['regression_policy_state']=regression_policy.new_state(73)
    new['gate']['streak']=0;new['gate']['identity']['recipe']=spec.RECIPE_VERSION;new['gate']['identity']['task']=spec.TASK_ID
    new['pending_transition']=None
    for key,value in dict(stop_state=None,stop_reason=None,error_type=None,lifecycle_state='running',terminal=None).items():new['ledger'][key]=value
    new['ledger']['curriculum_transaction']=None
    diagnostic=new['ledger'].get('diagnostic_collapse') or {};diagnostic.update(streak=0,generation=73,last=None);new['ledger']['diagnostic_collapse']=diagnostic
    parent=dict(checkpoint_path=SOURCE_PATH,checkpoint_sha256=SOURCE_SHA256,identity=deepcopy(original['identity']),
        source=deepcopy(original['source']),contract_sha256=original['contract_sha256'],recipe=deepcopy(original['recipe']),
        historical_evaluation=deepcopy(original['evaluation']),historical_gate=deepcopy(original['gate']),
        stopped_terminal=deepcopy(original['ledger'].get('terminal')),legacy_pending_transition=deepcopy(original.get('pending_transition')))
    # Old C1 tables are retained only as parent history, never native qualification.
    new['evaluation']=dict(cases={},latest=None,seed=42,offset_source='historical_tables_in_parent')
    new['gate']['records']=[]
    new['identity']=recovery.identity_block();new['contract_sha256']=recovery.contract_sha(spec.contract());new['recipe']=spec.recipe_config()
    new['source']=recovery.source_block(seed=42,device='cuda:0',mode='resume')
    new['source']['continuation']=dict(continuation_id=CONTINUATION_ID,plan_sha256=PLAN_SHA256,parent=parent,
        rollback_from_update=4817,resume_from_update=4817,lost_updates=0,replay_billed_updates=False,
        generation_before=72,generation_after=73,full_evaluation_rng_isolation='not_established_Q008')
    diff=dict(schema='a2-v07-stop4817-migration-v1',continuation_id=CONTINUATION_ID,plan_sha256=PLAN_SHA256,
        source_checkpoint_sha256=SOURCE_SHA256,source_state_digest=SOURCE_STATE_DIGEST,first_new_update=4818,
        changes=changes,source_provenance=parent,no_pending_replay=True,
        budget_before=dict(D25=4500,D20=4500,D=9000,total=36000,formal=36100),
        budget_after=dict(D25=6000,D22=6000,D20=9000,D=21000,total=48000,formal=48100),
        preserved={k:tree_digest(original[k]) for k in ['model','optimizer','normalizer','learning','rng']})
    for key in ['schema','identity','contract_sha256','recipe','source','gate','evaluation','pending_transition','regression_policy_state']:
        diff.setdefault('envelope_changes',{})[key]=dict(before=deepcopy(original.get(key)),after=deepcopy(new.get(key)))
    diff['ledger_changes']={key:dict(before=deepcopy(original['ledger'].get(key)),after=deepcopy(new['ledger'].get(key))) for key in ['stop_state','stop_reason','error_type','lifecycle_state','terminal','diagnostic_collapse','curriculum_transaction']}
    for key,digest in diff['preserved'].items():require(tree_digest(new[key])==digest,'preserved '+key+' changed')
    recovery.validate_checkpoint(new,num_envs=512)
    return new,diff
