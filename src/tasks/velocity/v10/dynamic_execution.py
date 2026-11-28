"""Explicit user-authorized N migration, with carried historical costs and fixed rounds.

The old 1024 resource receipt remains historical. This profile never certifies
resources for the requested N and never executes a probe.
"""
import copy
import json
from pathlib import Path
from . import cases,spec
from .identity import code_fingerprint
from .orchestration import Orchestrator,StopTraining,atomic_json,file_sha,read_json

SCHEMA='a2-v10-dynamic-execution-v1'
AUTHORIZATION=spec.ROOT/'doc/sds/reviews/v10_configurable_envs_user_override_20261127.md'
RULINGS=spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'
ORIGINAL_RUN=spec.ROOT/'logs/a2_v10/formal_A1_20261127_approved'
ORIGINAL_CP=ORIGINAL_RUN/'learning/D1_direct_user/D1_000050_boundary.pt'
ROUND_BUDGETS=dict(D1=500,D2=1000,D3=1500,PPO=1000)
GATES=dict(D1=[200,400,500],D2=[400,800,1000],D3=[600,1200,1500],PPO=[250,500,750,1000])


def _n(n):
    if not isinstance(n,int) or isinstance(n,bool) or n<10:raise ValueError('num_envs must be integer >=10')
    from .tasks import allocate_roles
    if any(v<=0 for v in allocate_roles(n).values()):raise ValueError('each of the six tasks requires at least one environment')
    return n


def _bound(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or not path.is_file() or file_sha(path)!=ref['sha256']:raise ValueError('execution profile reference changed')
    return path


def _ledger(path):
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('existing learning ledger required; history is never reset')
    from .checkpoint import BudgetLedger
    return BudgetLedger(path).report()



def _ledger_prefix(path,size=None):
    import hashlib
    h=hashlib.sha256();count=0
    with Path(path).open('rb') as f:
        remaining=Path(path).stat().st_size if size is None else size
        while remaining:
            block=f.read(min(8*1024*1024,remaining))
            if not block:raise ValueError('original ledger prefix truncated')
            h.update(block);count+=len(block);remaining-=len(block)
    return dict(bytes=count,sha256=h.hexdigest())

def _checkpoint_metadata(path):
    import torch
    from .teachers import teacher_identity
    row=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if row.get('sds_sha256')!=spec.SDS_SHA256 or row.get('recipe')!=spec.RECIPE or row.get('teachers')!=teacher_identity():raise ValueError('migration checkpoint recipe/teacher identity mismatch')
    return dict(from_num_envs=row['num_envs'],from_code_sha256=row['code_sha256'],physical_sha256=row['physical_sha256'],
                rounds={k:int(row.get('rounds',{}).get(k,0)) for k in ROUND_BUDGETS},accounting=row.get('accounting',{}))


def fresh_sources():
    """Initial teacher files and expected physics, without a student checkpoint."""
    import hashlib
    physics=spec.ROOT/'doc/sds/reviews/v10_server_physical_contract.json'
    row=read_json(physics)
    digest=hashlib.sha256(json.dumps(row['contract'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=row['sha256']:raise ValueError('server physical contract damaged')
    refs={}
    for key,path,expected in (('blind',spec.BLIND_PATH,spec.BLIND_SHA256),('stairs',spec.STAIRS_PATH,spec.STAIRS_SHA256)):
        if file_sha(path)!=expected:raise ValueError('fresh teacher identity differs: '+key)
        refs[key]=dict(path=str(path.resolve()),sha256=expected)
    refs['physics']=dict(path=str(physics.resolve()),sha256=file_sha(physics))
    return refs,digest


def build_execution_profile(path,*,num_envs,origin_checkpoint=ORIGINAL_CP,learning_ledger_path=ORIGINAL_RUN/'learning_costs.jsonl',
                            evaluation_ledger_path=spec.ROOT/'logs/a2_v10/formal_A_20261127_approved/evaluation_costs.jsonl',
                            authorization_path=AUTHORIZATION,physical_sha256=None,write=True,fresh=False):
    n=_n(num_envs)
    if fresh:
        sources,physics=fresh_sources();origin=None
        metadata=dict(physical_sha256=physics,rounds={k:0 for k in ROUND_BUDGETS},accounting={})
    else:
        origin=Path(origin_checkpoint).resolve();metadata=_checkpoint_metadata(origin)
    if physical_sha256 is not None and metadata['physical_sha256']!=physical_sha256:raise ValueError('migration physical contract differs')
    if fresh and not Path(learning_ledger_path).exists():
        accounting=dict(charged={},executed={},uncertain={},updated={},pending=[])
    else:accounting=_ledger(learning_ledger_path)
    if fresh and any(accounting[k] for k in ('charged','executed','updated','pending')):
        raise ValueError('fresh profile requires an unused learning ledger')
    if accounting['pending']:raise ValueError('learning pending step requires explicit recovery before creating migration profile')
    checkpoint_accounting=metadata.pop('accounting',{})
    if any(accounting['updated'].get(k,0)>checkpoint_accounting.get('updated',{}).get(k,0) for k in accounting['updated']):
        raise ValueError('origin checkpoint would discard newer completed updates; resume existing dynamic profile')
    rounds=metadata.pop('rounds');costs={k:dict(accounting[k]) for k in ('charged','executed','uncertain','updated')}
    if any(not 0<=rounds[k]<=ROUND_BUDGETS[k] for k in rounds):raise ValueError('migration round outside fixed course budget')
    sample_caps={k:costs['charged'].get('learning:'+k,0)+(r-rounds[k])*n*48 for k,r in ROUND_BUDGETS.items()}
    document=dict(schema=SCHEMA,num_envs=n,rollout=48,seed=42,code_sha256=code_fingerprint(),sds_sha256=spec.SDS_SHA256,
        rulings_sha256=file_sha(RULINGS),authorization=dict(path=str(Path(authorization_path).resolve()),sha256=file_sha(authorization_path)),
        origin_checkpoint=None if fresh else dict(path=str(origin),sha256=file_sha(origin),**metadata),physical_sha256=metadata['physical_sha256'],
        origin_rounds=rounds,accounting_origin=costs,learning_ledger_prefix=({'bytes':0,'sha256':__import__('hashlib').sha256(b'').hexdigest()} if fresh else _ledger_prefix(learning_ledger_path)),learning_ledger_path=str(Path(learning_ledger_path).resolve()),
        evaluation_ledger_path=str(Path(evaluation_ledger_path).resolve()),round_budgets=dict(ROUND_BUDGETS),save_rounds=50,gates=copy.deepcopy(GATES),
        p0_rounds=100,dagger_updates_per_round=10 if n==512 else 20,sample_caps=sample_caps,
        nominal_scaled_budgets={k:r*n*48 for k,r in ROUND_BUDGETS.items()},
        p0_sample_cap=costs['charged'].get('learning:PPO',0)+max(0,100-rounds['PPO'])*n*48,
        resource_status='unverified_for_requested_N',resource_probe_requested=False,automatic_N_fallback=False,
        recovery_registry_path=str(Path(learning_ledger_path).resolve())+'.dynamic_latest.json',
        supported_migration='original50_to_requested_N_then_same_profile_recovery',historical_costs_refunded=0)
    if fresh:
        document.update(start_mode='fresh',initial_sources=sources,supported_migration='fresh_from_S2_then_same_profile_recovery')
    if write:
        path=Path(path)
        if path.exists() and read_json(path)!=document:raise ValueError('existing execution profile cannot be overwritten')
        if not path.exists():atomic_json(path,document)
    return document


def validate_execution_profile(path,*,ledger_path=None,num_envs=None,physical_sha256=None):
    path=Path(path).resolve();d=read_json(path);n=_n(d.get('num_envs'))
    if (d.get('schema')!=SCHEMA or d.get('code_sha256')!=code_fingerprint() or d.get('sds_sha256')!=spec.SDS_SHA256
        or d.get('rulings_sha256')!=file_sha(RULINGS) or d.get('rollout')!=48 or d.get('seed')!=42
        or d.get('round_budgets')!=ROUND_BUDGETS or d.get('gates')!=GATES or d.get('save_rounds')!=50 or d.get('p0_rounds')!=100
        or d.get('dagger_updates_per_round')!=(10 if n==512 else 20)
        or d.get('resource_status')!='unverified_for_requested_N' or d.get('resource_probe_requested') is not False
        or d.get('automatic_N_fallback') is not False or d.get('historical_costs_refunded')!=0):raise ValueError('execution profile identity/recipe mismatch')
    if _bound(d['authorization']).resolve()!=AUTHORIZATION.resolve():raise ValueError('explicit dynamic-N user authorization required')
    if d.get('start_mode')=='fresh':
        sources,physics=fresh_sources()
        if (d.get('initial_sources')!=sources or d['origin_checkpoint'] is not None or d['physical_sha256']!=physics
            or d['origin_rounds']!={k:0 for k in ROUND_BUDGETS}
            or any(d['accounting_origin'].get(k) for k in ('charged','executed','uncertain','updated'))):
            raise ValueError('fresh initialization identity mismatch')
    else:
        _bound(d['origin_checkpoint'])
        if d['origin_checkpoint'].get('physical_sha256')!=d['physical_sha256']:raise ValueError('origin physical identity mismatch')
    if num_envs is not None and n!=num_envs:raise ValueError('execution profile requested N differs')
    if physical_sha256 is not None and d['physical_sha256']!=physical_sha256:raise ValueError('execution profile physical identity differs')
    learning=Path(d['learning_ledger_path']).resolve()
    if ledger_path is not None and learning!=Path(ledger_path).resolve():raise ValueError('execution profile original ledger differs')
    prefix=d['learning_ledger_prefix']
    if _ledger_prefix(learning,prefix['bytes'])!=prefix:raise ValueError('original learning ledger history changed')
    if file_sha(spec.SDS_PATH)!=spec.SDS_SHA256:raise ValueError('SDS file changed')
    current=_ledger(learning);origin=d['accounting_origin'];rounds=d['origin_rounds']
    if set(rounds)!=set(ROUND_BUDGETS) or any(not isinstance(rounds[k],int) or not 0<=rounds[k]<=r for k,r in ROUND_BUDGETS.items()):raise ValueError('invalid retained origin rounds')
    for field in ('charged','executed','updated'):
        if any(current[field].get(k,0)<v for k,v in origin[field].items()):raise ValueError('execution ledger history went backwards')
    caps={k:origin['charged'].get('learning:'+k,0)+(r-rounds[k])*n*48 for k,r in ROUND_BUDGETS.items()}
    if caps!=d['sample_caps'] or d.get('nominal_scaled_budgets')!={k:r*n*48 for k,r in ROUND_BUDGETS.items()}:raise ValueError('dynamic sample caps changed')
    return dict(path=str(path),sha256=file_sha(path),document=d,origin_checkpoint=d['origin_checkpoint'],
        learning_ledger_path=str(learning),evaluation_ledger_path=d['evaluation_ledger_path'],num_envs=n,round_budgets=d['round_budgets'],
        save_rounds=50,gates=d['gates'],sample_caps=caps)


class DynamicOrchestrator(Orchestrator):
    """Same behavior gates; explicit new profile replaces only static N admission/accounting."""
    def __init__(self,run_dir,profile,device='cuda:0',resume_reason=None,runner=None):
        self.profile=profile
        super().__init__(run_dir,AUTHORIZATION,device=device,resume_reason=resume_reason,runner=runner,
            learning_ledger=profile['learning_ledger_path'],evaluation_ledger=profile['evaluation_ledger_path'])
        existing=self.state.get('execution_profile_sha256')
        if existing and existing!=profile['sha256']:raise StopTraining('dynamic_execution_profile_changed')
        self.state['execution_profile_sha256']=profile['sha256']
        self.state['resource_status']='unverified_for_requested_N'

    def _assert_identity(self):
        checked=validate_execution_profile(self.profile['path'],ledger_path=self.learning_ledger,num_envs=self.profile['num_envs'])
        if checked['sha256']!=self.profile['sha256'] or self.state['code_sha256']!=code_fingerprint():raise StopTraining('dynamic_profile_or_source_changed')
        if self.state.get('rulings_sha256')!=file_sha(RULINGS):raise StopTraining('dynamic_rulings_changed')

    def prepare(self):
        self._assert_identity();d=self.profile['document'];freeze=self.root/'frozen';freeze.mkdir(exist_ok=True)
        frozen=dict(code_sha256=d['code_sha256'],sds_sha256=spec.SDS_SHA256,rulings_sha256=d['rulings_sha256'],
                    execution_profile_sha256=self.profile['sha256'],evaluation_plan=cases.evaluation_plan(),suites={})
        for course in ('D1','D2','D3','PPO'):
            suites=[('A',20),('B',20)]+([('DAGGER_FINAL',100)] if course=='D3' else [('PPO_FINAL',100)] if course=='PPO' else [])
            for suite,replicas in suites:
                manifest=cases.suite(course,suite,replicas);digest=cases.digest(manifest);path=freeze/f'{course}_{suite}.json'
                row=dict(sha256=digest,cases=manifest,sds_sha256=spec.SDS_SHA256)
                if path.exists() and read_json(path)!=row:raise StopTraining('dynamic_frozen_cases_changed')
                if not path.exists():atomic_json(path,row)
                frozen['suites'][f'{course}_{suite}']=dict(sha256=digest,path=str(path),replicas=replicas,max_samples=sum(r['max_steps'] for r in manifest))
        path=freeze/'execution_plan.json'
        if path.exists() and read_json(path)!=frozen:raise StopTraining('dynamic_frozen_plan_changed')
        if not path.exists():atomic_json(path,frozen)
        self.frozen=frozen
        self.state.update(num_envs=d['num_envs'],physical_sha256=d['physical_sha256'],frozen_sha256=file_sha(path),
            evaluation_max_samples=cases.evaluation_plan()['max_samples'],status='running',
            schedule=dict(num_envs=d['num_envs'],rollout=48,p0_rounds=100,gates=d['gates'],save_every_rounds=50,learning_budgets=d['sample_caps']))
        if not self.state.get('learning_checkpoint') and d['origin_checkpoint'] is not None:self.state['learning_checkpoint']=d['origin_checkpoint']['path']
        self.persist()
        print(json.dumps(dict(phase='prepare',num_envs=d['num_envs'],remaining_rounds={k:r-d['origin_rounds'][k] for k,r in ROUND_BUDGETS.items()},
            sample_caps=d['sample_caps'],historical_charged=d['accounting_origin']['charged'],resource_status=self.state['resource_status']),sort_keys=True),flush=True)

    def _migrate_controller_source(self):self._assert_identity()

    def evaluate(self,key,course,suite,model_path,reference):
        print(json.dumps(dict(phase='evaluation',operation=key,course=course,suite=suite,reference=reference)),flush=True)
        return super().evaluate(key,course,suite,model_path,reference)

    def _recover_evaluation_accounting(self):
        from .gpu_lock import gpu_execution_lock
        from .checkpoint import BudgetLedger
        with gpu_execution_lock(blocking=False):
            ledger=BudgetLedger(self.evaluation_ledger);before=ledger.report();ledger.reconcile_uncertain();after=ledger.report()
        if before['pending']:
            self.commit_event('dynamic-evaluation-recovery:'+cases.digest(before),dict(evaluation_recovery=dict(
                charged_before=before['charged'],charged_after=after['charged'],uncertain=after['uncertain'],refund=0)))

    def _finalize_formal_handoff(self):
        # Dynamic CP admission has its own explicit profile. Never forge a legacy
        # N1024 handoff credential or transplant its source/resource acceptance.
        if self.state.get('teacher_admitted') is not True:raise StopTraining('dynamic_teachers_not_admitted')

    def _teacher_receipts(self):
        return {f'teacher_admission_{course}_{suite}':dict(path=str(self._operation_dir(f'teacher_admission_{course}_{suite}')/'result.json'),
                sha256=file_sha(self._operation_dir(f'teacher_admission_{course}_{suite}')/'result.json'))
                for course in ('D1','D2','D3') for suite in ('A','B')}

    def _dynamic_resume(self,key,course,target,out,resume):
        path=out/'worker_checkpoint_manifest.json'
        if key not in self.state['operations'] or not path.exists():return resume
        row=read_json(path)
        if row.get('operation_id')!=key:return resume
        expected=dict(course=course,num_envs=self.profile['num_envs'],code_sha256=self.state['code_sha256'],
            physical_sha256=self.state['physical_sha256'],learning_ledger_path=str(self.learning_ledger),target_round=target,
            execution_profile_sha256=self.profile['sha256'])
        cp=row.get('checkpoint',{});candidate=Path(cp.get('path','')).resolve()
        if any(row.get(k)!=v for k,v in expected.items()) or candidate.parent!=out.resolve() or file_sha(candidate)!=cp.get('sha256'):
            raise StopTraining('dynamic_recovery_manifest_changed')
        if not isinstance(row.get('round'),int) or not 0<=row['round']<=target:raise StopTraining('dynamic_recovery_round_invalid')
        return str(candidate)

    def worker(self,key,course,target,resume=None,acceptance=None):
        out=self.root/'learning'/course;resume=self._dynamic_resume(key,course,target,out,resume)
        self.state['execution_teacher_receipts']=self._teacher_receipts();self.persist()
        command=[__import__('sys').executable,spec.ROOT/'scripts/run_a2_v10.py','worker','--output-dir',out,'--device',self.device,
            '--num-envs',str(self.state['num_envs']),'--course',course,'--stop-round',str(target),'--ledger-path',self.learning_ledger,
            '--controller-state','__CONTROLLER_STATE__','--execution-profile',self.profile['path']]
        if resume:command+=['--resume',resume]
        if acceptance:command+=['--acceptance',acceptance]
        print(json.dumps(dict(phase=course,operation=key,target_round=target,num_envs=self.state['num_envs'],resource_status=self.state['resource_status'])),flush=True)
        result=self._launch(key,command,out/'worker_result.json')
        if (result.get('gate_reached') is not True or result.get('course')!=course or result.get('round')!=target
            or result.get('physical_sha256')!=self.state['physical_sha256'] or file_sha(result['checkpoint'])!=result.get('checkpoint_sha256')):
            raise StopTraining('dynamic_worker_target_not_reached')
        budget_key='PPO' if course.startswith('P') else course
        cap=self.profile['document']['p0_sample_cap'] if course=='P0' else self.profile['sample_caps'][budget_key]
        if result.get('budget',{}).get('charged',{}).get('learning:'+budget_key,0)>cap:raise StopTraining('dynamic_worker_charged_cap_exceeded')
        self.state['learning_checkpoint']=result['checkpoint'];self.persist();return result['checkpoint']

    def _train_dagger(self):
        if self.state.get('accepted_dagger'):return self.state['accepted_dagger']
        checkpoint=self.state['learning_checkpoint']
        for course in ('D1','D2','D3'):
            self.state['phase']=course;self.persist();completed=self.state.get('courses',{}).get(course)
            if completed:checkpoint=completed['checkpoint'];continue
            if course!='D1' or checkpoint is None:checkpoint=self.worker(f'entry_{course}',course,0,checkpoint)
            for target in self.state['schedule']['gates'][course]:
                checkpoint=self.worker(f'learning_{course}_{target}',course,target,checkpoint)
                grade=self.gate(course,target,checkpoint,{'teachers':None})
                if grade['passed']:self._promote_course(course,target,checkpoint,grade);break
            else:raise StopTraining('course_budget_exhausted:'+course)
        final=self.final('DAGGER_FINAL','D3',checkpoint,{'teachers':None})
        self.certificate('V10_DAGGER_ACCEPTED',checkpoint,final)
        return checkpoint



def _guard_other_dynamic_progress(root,learning,n):
    current=_ledger(learning)
    for path in root.glob('execution_profile_N*.json'):
        old=read_json(path)
        if old.get('schema')!=SCHEMA or old.get('num_envs')==n:continue
        baseline=old.get('accounting_origin',{}).get('charged',{})
        if any(v>baseline.get(k,0) for k,v in current['charged'].items()):
            raise StopTraining('second_dynamic_N_switch_not_supported; existing dynamic costs/progress have no refund')

def run_dynamic_execution(*,run_dir,profile_path,num_envs,device='cuda:0',teachers_only=False,check_only=False,resume_reason=None,fresh=False):
    """One original-run parent lease prevents old/new N controllers interleaving."""
    import fcntl
    root=Path(run_dir).resolve();path=Path(profile_path).resolve();n=_n(num_envs)
    learning=root/'learning_costs.jsonl';registry=Path(str(learning)+'.dynamic_latest.json')
    if registry.exists():
        latest=read_json(registry)
        if latest.get('num_envs')!=n:
            raise StopTraining('second_dynamic_N_switch_not_supported; retain latest checkpoint and resume its existing N profile')
    if learning.exists():_guard_other_dynamic_progress(root,learning,n)
    if check_only:
        if path.exists():profile=validate_execution_profile(path,ledger_path=learning,num_envs=n);document=profile['document']
        else:document=build_execution_profile(path,num_envs=n,learning_ledger_path=learning,write=False,**(dict(fresh=True,evaluation_ledger_path=root/'evaluation_costs.jsonl') if fresh else {}))
        return dict(passed=True,executed=False,profile_exists=path.exists(),num_envs=n,resource_status=document['resource_status'],
            remaining_rounds={k:r-document['origin_rounds'][k] for k,r in ROUND_BUDGETS.items()},sample_caps=document['sample_caps'],
            dagger_updates_per_round=document['dagger_updates_per_round'],supported_migration=document['supported_migration'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise StopTraining('another_original_or_dynamic_controller_is_running') from None
        from .gpu_lock import gpu_execution_lock
        with gpu_execution_lock(blocking=False):
            if fresh and not learning.exists():learning.touch(exist_ok=False)
            _guard_other_dynamic_progress(root,learning,n)
            if not path.exists():build_execution_profile(path,num_envs=n,learning_ledger_path=learning,**(dict(fresh=True,evaluation_ledger_path=root/'evaluation_costs.jsonl') if fresh else {}))
            profile=validate_execution_profile(path,ledger_path=learning,num_envs=n)
        # Once any profile has consumed new samples, a second fresh profile must
        # not silently reload original50, even if a crash preceded a save/registry.
        current=_ledger(learning);origin=profile['document']['accounting_origin']
        dynamic_root=root/f'dynamic_N{n}'
        if not (dynamic_root/'controller_state.json').exists() and current['charged']!=origin['charged']:
            raise StopTraining('new_dynamic_profile_would_discard_existing_progress')
        return DynamicOrchestrator(dynamic_root,profile,device,resume_reason).run(teachers_only=teachers_only)
