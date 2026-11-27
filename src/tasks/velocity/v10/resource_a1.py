"""Explicitly approved old-full + new-targeted A1 resource association.

A targeted measurement is never relabelled as a fresh twenty-measurement probe.
This CPU parent retains original evidence and serializes isolated GPU children.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from . import spec
from .identity import code_fingerprint
from .probe_budget import GlobalProbeBudget,ProbeBudgetInsufficient,streaming_sha256

A1_PROTOCOL='A1-targeted-resource-v1'
A1_STACK_SAMPLES=5*1024*48
A1_TOTAL_SAMPLES=2*A1_STACK_SAMPLES
A1_REQUIRED_STAGES={'sampling_initial_begin','replay_full_initial_released','save_begin','save_serialized','save_disk_validated','old_replay_released_before_load','load_begin','load_complete','restored_state_transferred_and_released'}|{f'{name}_{i}_{suffix}' for i in range(4) for name,suffix in (('sampling','begin'),('sampling','complete'),('update','complete'),('rollout','released'))}
OLD_SOURCE='b48e3fd1601ff2926272ed014d846fd480fe24a8c56d77100d646f933b92991f'
OLD_GIT='07fe14fe753ade340f3d64e06dbae51a9ad0bdb5'
OLD_BASELINE_DIR=spec.ROOT/'logs/a2_v10/formal_A_20261127_approved/resource_probe'
OLD_FILES={
    'selection.json':'4f1e2c2738e1ad6cb8a2431f434c23634db4f232ac3023e56ad178d0dd7ecc04',
    'n1024_dagger/result.json':'37d422907bd3305ab10ac3da78223b49404b1b96e9446f6415055d8d27d0383e',
    'n1024_ppo/result.json':'6ccbbf3d5a0cb784cba91416721a6a9c6078536b3abd4674945d4447b2b072db',
}
RULINGS=spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'
PROPOSAL=spec.ROOT/'doc/sds/reviews/v10_A1_validation_budget_proposal_20261127.md'
INVARIANT_FILES=(
    'src/tasks/velocity/v10/model.py','src/tasks/velocity/v10/spec.py',
    'src/tasks/velocity/v10/env.py','src/tasks/velocity/v10/geometry.py','src/tasks/velocity/v10/contacts.py','src/tasks/velocity/v10/tasks.py',
    'src/tasks/velocity/v10/replay.py','src/tasks/velocity/v10/dagger.py','src/tasks/velocity/v10/ppo.py',
    'src/tasks/velocity/v10/rewards.py','src/tasks/velocity/v10/runtime.py',
    'src/tasks/velocity/v09/model.py','src/tasks/velocity/v09/map.py','src/tasks/velocity/v07/model.py',
    'src/tasks/velocity/v07/env_cfg.py','src/tasks/velocity/v07/contacts.py','src/tasks/velocity/v07/events.py',
    'src/tasks/velocity/rl/ideal_heightmap_actor_critic.py','src/tasks/velocity/config/a2/eth_env_cfg.py',
    'src/tasks/velocity/config/a2/env_cfgs.py','src/tasks/velocity/velocity_env_cfg.py',
)


def _read(path):return json.loads(Path(path).read_text())


def _write(path,row):
    from .checkpoint import atomic_json
    atomic_json(path,row)


def _old_blob_sha(path):
    process=subprocess.Popen(['git','cat-file','blob',f'{OLD_GIT}:{path}'],cwd=spec.ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    digest=hashlib.sha256()
    with process.stdout as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest() if process.wait()==0 else None


def _invariant_paths():
    assets={p.relative_to(spec.ROOT).as_posix() for p in (spec.ROOT/'src/assets/robots/unitree_a2').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    return sorted(set(INVARIANT_FILES)|assets)


def verify_old_evidence(baseline_dir=OLD_BASELINE_DIR):
    root=Path(baseline_dir).resolve()
    for relative,digest in OLD_FILES.items():
        if streaming_sha256(root/relative)!=digest:raise ValueError('old evidence SHA mismatch: '+relative)
    selection=_read(root/'selection.json')
    runs={stack:_read(root/f'n1024_{stack}/result.json') for stack in ('dagger','ppo')}
    for row in [selection,*runs.values()]:
        if row.get('passed') is not True or row.get('num_envs')!=1024 or row.get('code_sha256')!=OLD_SOURCE or row.get('sds_sha256')!=spec.SDS_SHA256:
            raise ValueError('old full resource identity invalid')
    for stack,row in runs.items():
        if (row.get('stack')!=stack or row.get('warmup_rounds')!=5 or row.get('measurement_rounds')!=20
            or row.get('restore_checked') is not True or any(row.get('replay_counts',{}).get(r,0)<32768 for r in spec.ROLES)
            or row.get('gpu_total_peak_mib',1e9)>7680 or row.get('gpu_task_peak_mib',1e9)>7168 or row.get('host_available_min_bytes',0)<4*1024**3):
            raise ValueError('old full resource evidence incomplete')
    physics={r['physical_sha256'] for r in runs.values()}
    if len(physics)!=1:raise ValueError('old full physics mismatch')
    return dict(num_envs=1024,physical_sha256=next(iter(physics)),runs=runs,selection=selection,files=dict(OLD_FILES),source_sha256=OLD_SOURCE)


# Exact approved recording-only expression. No arbitrary frame.update is removed.
_RECORDING_UPDATE = """frame.update(yaw=yaw.clone(),foot_position_local=feet_local.clone(),top_upward_force=upward.clone(),
    failure_masks=dict(body_hit=body_hit.clone(),long_hit=long_hit.clone(),
        attitude=(roll.abs()>math.radians(70))|(pitch.abs()>math.radians(70)),
        route_fall=d.root_link_pos_w[:,2]<route_h-.5,bounds=~bounds,timeout=frame['timeout'].clone()))"""


def _env_recording_equivalence(relative):
    before=subprocess.check_output(['git','cat-file','blob',f'{OLD_GIT}:{relative}'],cwd=spec.ROOT).decode()
    old=ast.parse(before);new=ast.parse((spec.ROOT/relative).read_text())
    expected=ast.dump(ast.parse(_RECORDING_UPDATE).body[0],include_attributes=False)
    classes=[n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='V10Env']
    if len(classes)!=1:return False
    functions=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='step']
    if len(functions)!=1:return False
    body=functions[0].body;matches=[i for i,n in enumerate(body) if ast.dump(n,include_attributes=False)==expected]
    if len(matches)!=1:return False
    index=matches[0]
    if index==0 or index+1>=len(body):return False
    previous=body[index-1];following=body[index+1]
    if not (isinstance(previous,ast.Assign) and len(previous.targets)==1 and isinstance(previous.targets[0],ast.Name) and previous.targets[0].id=='frame'
            and isinstance(following,ast.Assign) and isinstance(following.value,ast.Call) and isinstance(following.value.func,ast.Name) and following.value.func.id=='compute_reward'):
        return False
    del body[index]
    return ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)


def _invariance_record(relative):
    old=_old_blob_sha(relative);new=streaming_sha256(spec.ROOT/relative)
    record=dict(baseline_sha256=old,current_sha256=new,unchanged=old is not None and old==new,evidence_type='exact_bytes')
    if relative=='src/tasks/velocity/v10/env.py' and old!=new:
        record.update(evidence_type='exact_approved_recording_ast_only',recording_only_ast_equivalent=_env_recording_equivalence(relative),
                      approved_removed_statement_sha256=hashlib.sha256(ast.dump(ast.parse(_RECORDING_UPDATE).body[0],include_attributes=False).encode()).hexdigest())
    return record


def build_invariance_evidence(path):
    rows={}
    for relative in _invariant_paths():
        rows[relative]=_invariance_record(relative)
    row=dict(schema='a2-v10-A1-invariants-v1',baseline_git=OLD_GIT,baseline_code_sha256=OLD_SOURCE,
             new_code_sha256=code_fingerprint(),files=rows,unchanged=all(r['unchanged'] for r in rows.values()),
             invariants_satisfied=all(r['unchanged'] or r.get('recording_only_ast_equivalent') is True for r in rows.values()),
             facts=dict(num_envs=1024,rollout=48,observation=[45,[5,45],[3,40,30],3,35],
                        replay_tasks=6,replay_per_task=32768,replay_dtype='float32',dagger_logical_batch=3072,
                        ppo_epochs=5,ppo_minibatches=4,microbatch_max=256))
    _write(path,row);return row


def _source_difference_document():
    current=set((spec.ROOT/'src').rglob('*.py'))|set((spec.ROOT/'scripts').glob('*a2_v10*.py'))|set((spec.ROOT/'tests').glob('test_v10*.py'))
    names={p.relative_to(spec.ROOT).as_posix() for p in current}
    old=subprocess.check_output(['git','ls-tree','-r','--name-only',OLD_GIT,'src','scripts','tests'],cwd=spec.ROOT,text=True).splitlines()
    names.update(p for p in old if p.endswith('.py') and (p.startswith('src/') or '/test_v10' in p or 'a2_v10' in p))
    changed=[]
    for relative in sorted(names):
        before=_old_blob_sha(relative);source=spec.ROOT/relative;after=streaming_sha256(source) if source.is_file() else None
        if before!=after:changed.append(dict(path=relative,baseline_sha256=before,current_sha256=after))
    row=dict(schema='a2-v10-A1-source-diff-v1',baseline_git=OLD_GIT,baseline_code_sha256=OLD_SOURCE,
             new_code_sha256=code_fingerprint(),changed_files=changed,
             note='Actual current bytes include protected pre-existing dirty/untracked files; independent review must classify the differences.')
    return row


def build_source_difference(path):
    row=_source_difference_document();_write(path,row);return row


def build_a1_contract_template(path,evidence_dir):
    evidence_dir=Path(evidence_dir);evidence_dir.mkdir(parents=True,exist_ok=True)
    invariants=evidence_dir/'invariants.json';diff=evidence_dir/'source_diff.json'
    build_invariance_evidence(invariants);build_source_difference(diff)
    row=dict(schema='a2-v10-A1-resource-linkage-v1',protocol=A1_PROTOCOL,approved=False,
             sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(RULINGS),new_code_sha256=code_fingerprint(),
             proposal=dict(path=str(PROPOSAL),sha256=streaming_sha256(PROPOSAL)),baseline_dir=str(OLD_BASELINE_DIR),baseline_git=OLD_GIT,
             old_evidence=dict(OLD_FILES),invariance_evidence=dict(path=str(invariants.resolve()),sha256=streaming_sha256(invariants)),
             source_diff=dict(path=str(diff.resolve()),sha256=streaming_sha256(diff)),
             independent_review=dict(passed=False,report_path=None,report_sha256=None),
             scope=dict(num_envs=1024,fill_rollouts=1,warmup_rounds=2,measurement_rounds=2,per_stack_samples=A1_STACK_SAMPLES,total_samples=A1_TOTAL_SAMPLES,
                        full_twenty_measurement_revalidation=False,probe_state_discarded=True))
    _write(path,row);return row


def _bound_document(ref):
    path=Path(ref['path'])
    if streaming_sha256(path)!=ref['sha256']:raise ValueError('linked evidence SHA changed')
    return _read(path)


def validate_a1_contract(path):
    row=_read(path)
    if row.get('approved') is not True:raise ValueError('explicit approved A1 resource reuse approval required')
    if (row.get('schema')!='a2-v10-A1-resource-linkage-v1' or row.get('protocol')!=A1_PROTOCOL
        or row.get('sds_sha256')!=spec.SDS_SHA256 or row.get('rulings_sha256')!=streaming_sha256(RULINGS)
        or row.get('new_code_sha256')!=code_fingerprint() or row.get('baseline_git')!=OLD_GIT or row.get('old_evidence')!=OLD_FILES):
        raise ValueError('A1 resource contract identity mismatch')
    proposal=row['proposal']
    if Path(proposal['path']).resolve()!=PROPOSAL.resolve() or proposal['sha256']!=streaming_sha256(PROPOSAL):raise ValueError('A1 proposal binding mismatch')
    review=row.get('independent_review',{})
    if review.get('passed') is not True or not review.get('report_path') or streaming_sha256(review['report_path'])!=review.get('report_sha256'):
        raise ValueError('independent A1 invariance review required')
    evidence=_bound_document(row['invariance_evidence']);diff=_bound_document(row['source_diff'])
    if (evidence.get('baseline_git')!=OLD_GIT or evidence.get('new_code_sha256')!=row['new_code_sha256']
        or evidence.get('invariants_satisfied') is not True or set(evidence['files'])!=set(_invariant_paths())
        or diff.get('baseline_git')!=OLD_GIT or diff.get('new_code_sha256')!=row['new_code_sha256']):raise ValueError('A1 invariance/source difference evidence invalid')
    for relative,record in evidence['files'].items():
        expected=_invariance_record(relative)
        if record!=expected or not (expected['unchanged'] or expected.get('recording_only_ast_equivalent') is True):
            raise ValueError('A1 invariant actually changed: '+relative)
    if diff!=_source_difference_document():raise ValueError('A1 source difference list incomplete or changed')
    for record in diff['changed_files']:
        source=spec.ROOT/record['path'];actual=streaming_sha256(source) if source.is_file() else None
        if record.get('baseline_sha256')!=_old_blob_sha(record['path']) or record.get('current_sha256')!=actual:raise ValueError('A1 source difference changed')
    scope=row.get('scope',{})
    if (scope.get('num_envs')!=1024 or scope.get('fill_rollouts')!=1 or scope.get('warmup_rounds')!=2 or scope.get('measurement_rounds')!=2
        or scope.get('per_stack_samples')!=A1_STACK_SAMPLES or scope.get('total_samples')!=A1_TOTAL_SAMPLES
        or scope.get('full_twenty_measurement_revalidation') is not False):raise ValueError('A1 targeted protocol altered')
    old=verify_old_evidence(row['baseline_dir'])
    return dict(contract=row,contract_path=str(Path(path).resolve()),contract_sha256=streaming_sha256(path),old=old)


def a1_probe_stack(*,stack,output_dir,linkage_contract,device='cuda:0',global_probe_ledger=None,probe_budget_authorization=None,probe_sequence_id=None,num_envs=1024):
    if num_envs!=1024:raise ValueError('A1 targeted protocol has no N fallback')
    link=validate_a1_contract(linkage_contract)
    audit=GlobalProbeBudget(global_probe_ledger,probe_budget_authorization)
    audit.assert_authorized(probe_sequence_id)
    attempt=audit.require_protocol_attempt(A1_PROTOCOL,link['contract_sha256'],probe_sequence_id)
    if Path(output_dir).resolve()!=Path(attempt['output_dir'])/f'n1024_{stack}':raise ValueError('A1 child output differs from original attempt')
    from .resources import _probe_stack_impl
    result=_probe_stack_impl(num_envs=1024,stack=stack,output_dir=output_dir,device=device,global_probe_ledger=global_probe_ledger,
        probe_budget_authorization=probe_budget_authorization,probe_sequence_id=probe_sequence_id,warmup_rounds=2,measurement_rounds=2,
        protocol=A1_PROTOCOL,expected_physical_sha256=link['old']['physical_sha256'])
    result.update(linkage_contract_path=link['contract_path'],linkage_contract_sha256=link['contract_sha256'])
    _write(Path(output_dir)/'result.json',result);return result


def run_a1_probes(*,output_dir,linkage_contract,device='cuda:0',global_probe_ledger=None,probe_budget_authorization=None,probe_sequence_id=None):
    link=validate_a1_contract(linkage_contract);out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('A1 output directory must be new; no targeted remeasure retry')
    budget=GlobalProbeBudget(global_probe_ledger,probe_budget_authorization)
    if not probe_sequence_id:raise ValueError('unique A1 sequence identity required')
    budget.begin_sequence(probe_sequence_id,required_samples=A1_TOTAL_SAMPLES)
    runs=[];passed=False;claimed=False
    try:
        budget.claim_protocol_attempt(A1_PROTOCOL,link['contract_sha256'],probe_sequence_id,out);claimed=True
        grant=budget.assert_authorized(probe_sequence_id);out.mkdir(parents=True,exist_ok=True)
        for stack in ('dagger','ppo'):
            scope=f'{probe_sequence_id}:1024:{stack}';budget.acquire_stack_capacity(probe_sequence_id,A1_STACK_SAMPLES,scope_id=scope)
            child=out/f'n1024_{stack}'
            cmd=[sys.executable,str(spec.ROOT/'scripts/run_a2_v10.py'),'_probe-a1-stack','--stack',stack,'--output-dir',str(child),'--device',device,
                 '--linkage-contract',str(linkage_contract),'--global-probe-ledger',str(global_probe_ledger),
                 '--probe-budget-authorization',str(probe_budget_authorization),'--probe-sequence-id',probe_sequence_id]
            try:
                with (out/f'{stack}.log').open('w') as log:completed=subprocess.run(cmd,cwd=spec.ROOT,stdout=log,stderr=subprocess.STDOUT)
            finally:budget.release_stack_capacity(scope,reason='A1 child exited')
            result=_read(child/'result.json') if (child/'result.json').exists() else dict(passed=False,error=f'child exit {completed.returncode}')
            runs.append(result)
            if not result.get('passed'):break
            if completed.returncode!=0:raise ValueError('A1 child exit disagrees with result')
        passed=len(runs)==2 and all(r.get('passed') is True for r in runs)
        selection=dict(resource_protocol=A1_PROTOCOL,passed=passed,num_envs=1024 if passed else None,runs=runs,
            sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(RULINGS),code_sha256=code_fingerprint(),
            global_probe_ledger=str(budget.path),probe_sequence_id=probe_sequence_id,probe_budget_authorization_sha256=grant['sha256'],
            global_probe_totals=budget.global_report()['global_totals'],probe_state_discarded=True,full_twenty_measurement_revalidation=False,physical_sha256=link['old']['physical_sha256'],
            linkage_contract_path=link['contract_path'],linkage_contract_sha256=link['contract_sha256'])
        if passed:
            certificate=make_linkage_certificate(link,selection,out)
            selection.update(resource_linkage_certificate_path=str(out/'resource_linkage_certificate.json'),resource_linkage_certificate_sha256=streaming_sha256(out/'resource_linkage_certificate.json'))
        _write(out/'selection.json',selection);return selection
    except BaseException:
        passed=False;raise
    finally:
        if claimed:budget.finish_protocol_attempt(A1_PROTOCOL,link['contract_sha256'],passed=passed)
        budget.finish_sequence(probe_sequence_id,passed=passed,reason='A1 targeted sequence finished; no retries or fallback')


def _association_document(link,selection,output):
    output=Path(output).resolve()
    if (selection.get('passed') is not True or selection.get('resource_protocol')!=A1_PROTOCOL or selection.get('num_envs')!=1024
        or selection.get('sds_sha256')!=spec.SDS_SHA256 or selection.get('code_sha256')!=code_fingerprint()
        or selection.get('rulings_sha256')!=streaming_sha256(RULINGS) or selection.get('physical_sha256')!=link['old']['physical_sha256']
        or selection.get('probe_state_discarded') is not True or selection.get('full_twenty_measurement_revalidation') is not False
        or selection.get('linkage_contract_path')!=link['contract_path'] or selection.get('linkage_contract_sha256')!=link['contract_sha256']):
        raise ValueError('A1 selection identity invalid')
    runs=selection.get('runs',[])
    if len(runs)!=2 or {r.get('stack') for r in runs}!={'dagger','ppo'}:raise ValueError('A1 requires exactly two distinct targeted stacks')
    references={str((Path(link['contract']['baseline_dir'])/rel).resolve()):sha for rel,sha in OLD_FILES.items()}
    new_results={}
    for row in runs:
        stack=row['stack'];path=output/f'n1024_{stack}/result.json'
        if not path.is_file() or _read(path)!=row:raise ValueError('A1 selection run differs from raw targeted result')
        stages={r.get('stage') for r in row.get('memory_stages',[])}
        replay=row.get('restored_replay_manifest',{})
        cost=row.get('cost',{})
        totals={k:sum(cost.get(k,{}).values()) for k in ('charged','executed','uncertain','updated')}
        if (row.get('passed') is not True or row.get('resource_protocol')!=A1_PROTOCOL or row.get('num_envs')!=1024
            or row.get('sds_sha256')!=spec.SDS_SHA256 or row.get('warmup_rounds')!=2 or row.get('measurement_rounds')!=2
            or row.get('fill_rollouts')!=1 or len(row.get('measured_seconds',[]))!=2
            or row.get('physical_sha256')!=link['old']['physical_sha256'] or row.get('code_sha256')!=code_fingerprint()
            or row.get('restore_checked') is not True or any(row.get('replay_counts',{}).get(role,0)<32768 for role in spec.ROLES)
            or row.get('gpu_total_peak_mib',1e9)>7680 or row.get('gpu_task_peak_mib',1e9)>7168 or row.get('host_available_min_bytes',0)<4*1024**3
            or not A1_REQUIRED_STAGES<=stages or len(row.get('checkpoint_sha256',''))!=64 or len(row.get('next_sample_manifest',''))!=64
            or replay.get('capacity')!=32768 or len(replay.get('sha256',''))!=64 or any(replay.get('counts',{}).get(role,0)!=32768 for role in spec.ROLES)
            or totals!=dict(charged=A1_STACK_SAMPLES,executed=A1_STACK_SAMPLES,uncertain=0,updated=4*1024*48)
            or row.get('global_probe_ledger')!=selection['global_probe_ledger'] or row.get('probe_sequence_id')!=selection['probe_sequence_id']
            or row.get('probe_budget_authorization_sha256')!=selection['probe_budget_authorization_sha256']
            or row.get('linkage_contract_sha256')!=link['contract_sha256']):
            raise ValueError('A1 targeted resource evidence incomplete')
        references[str(path)]=streaming_sha256(path);new_results[stack]=dict(path=str(path),sha256=references[str(path)])
    references[link['contract_path']]=link['contract_sha256']
    for name in ('invariance_evidence','source_diff','proposal'):
        ref=link['contract'][name];references[str(Path(ref['path']).resolve())]=ref['sha256']
    review=link['contract']['independent_review'];references[str(Path(review['report_path']).resolve())]=review['report_sha256']
    references[str(RULINGS.resolve())]=streaming_sha256(RULINGS)
    unsigned={k:v for k,v in selection.items() if k not in ('resource_linkage_certificate_path','resource_linkage_certificate_sha256')}
    from .cases import digest
    return dict(schema='a2-v10-A1-resource-association-v1',passed=True,resource_protocol=A1_PROTOCOL,
        sds_sha256=spec.SDS_SHA256,rulings_sha256=streaming_sha256(RULINGS),old_code_sha256=OLD_SOURCE,new_code_sha256=code_fingerprint(),
        physical_sha256=link['old']['physical_sha256'],num_envs=1024,old_warmup=5,old_measurement=20,new_warmup=2,new_measurement=2,
        full_twenty_measurement_revalidation=False,accepted_evidence='old full long-run + explicitly approved new targeted supplement',
        references=references,targeted_results=new_results,selection_content_sha256=digest(unsigned),
        source_diff_sha256=link['contract']['source_diff']['sha256'],invariance_evidence_sha256=link['contract']['invariance_evidence']['sha256'])


def make_linkage_certificate(link,selection,output):
    certificate=_association_document(link,selection,output)
    _write(Path(output)/'resource_linkage_certificate.json',certificate);return certificate


def verify_linkage_selection(selection):
    certificate_path=Path(selection['resource_linkage_certificate_path']).resolve()
    if streaming_sha256(certificate_path)!=selection['resource_linkage_certificate_sha256']:raise ValueError('A1 association certificate SHA mismatch')
    certificate=_read(certificate_path)
    link=validate_a1_contract(selection['linkage_contract_path'])
    expected=_association_document(link,selection,certificate_path.parent)
    if certificate!=expected:raise ValueError('A1 association certificate schema/identity/references/selection mismatch')
    for path,sha in certificate['references'].items():
        if streaming_sha256(path)!=sha:raise ValueError('A1 association reference changed')
    return certificate
