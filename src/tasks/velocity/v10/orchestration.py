"""Durable serial controller for the reviewed double-teacher DAgger -> PPO SDS.

The controller performs only CPU bookkeeping/comparison. Physics, evaluation,
probes, learning and export always run in separate, awaited child processes.
No failed gate is interpreted as permission to change a recipe or case suite.
"""
from dataclasses import dataclass,asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from . import cases,spec
from .identity import code_fingerprint

RULINGS_PATH=spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'


class StopTraining(RuntimeError):pass


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno());temp=f.name
    os.replace(temp,path)


def read_json(path):return json.loads(Path(path).read_text())


def execution_plan(n):
    if n not in (1024,512):raise ValueError('formal N must be 1024 or 512')
    return dict(num_envs=n,rollout=48,learning_budgets=dict(spec.BUDGETS),
                p0_rounds=spec.P0_BUDGET//(n*48),
                gates={c:list(spec.gate_rounds(c,n)) for c in (*('D1','D2','D3'),'PPO')},
                save_every_rounds=spec.SAVE_SAMPLES//(n*48))


def validate_engineering(path,code_sha):
    try:row=read_json(path)
    except (OSError,ValueError) as error:raise StopTraining('engineering_admission_missing') from error
    review=row.get('independent_review',{})
    if (row.get('passed') is not True or row.get('sds_sha256')!=spec.SDS_SHA256
        or row.get('code_sha256')!=code_sha or review.get('passed') is not True
        or len(review.get('report_sha256',''))!=64 or not review.get('report_path')):raise StopTraining('engineering_admission_invalid')
    if review.get('report_path'):
        try:valid=file_sha(review['report_path'])==review['report_sha256']
        except OSError:valid=False
        if not valid:raise StopTraining('independent_review_changed')
    return row



def preflight_formal_handoff(plan_path,engineering_admission,source_compatibility):
    """Read-only admission: no controller/ledger creation, reconciliation or child."""
    from .source_compatibility import validate_compatibility
    from .handoff import validate_plan
    bridge=validate_compatibility(source_compatibility)
    validate_engineering(engineering_admission,bridge['to_code_sha256'])
    plan=validate_plan(plan_path)
    if plan['compatibility']['sha256']!=bridge['sha256']:raise StopTraining('handoff_compatibility_mismatch')
    return dict(passed=True,executed=False,plan=plan,compatibility=bridge,
                initial_round=plan['initial_round'],next_D1_gate=spec.gate_rounds('D1',1024)[0])

def is_retention_failure(comparison):
    return any(reason=='relative_success' or reason.startswith(('relative_metric:','pair_coverage:'))
               for row in comparison['groups'].values() for reason in row['reasons'])


@dataclass
class RetentionState:
    consecutive_failures:int=0
    failures:int=0
    rollbacks:int=0
    def record(self,passed):
        if passed:self.consecutive_failures=0;return False
        self.consecutive_failures+=1;self.failures+=1
        if self.consecutive_failures<2:return False
        if self.rollbacks>=2:raise StopTraining('third_rollback_required')
        self.rollbacks+=1;self.consecutive_failures=0
        return True


def compare_results(candidate,reference,final=False):
    from .evaluation import grade_group
    for key in ('sds_sha256','suite_sha256','physical_sha256'):
        if candidate.get(key)!=reference.get(key):raise StopTraining('evaluation_identity_mismatch:'+key)
    if set(candidate['groups'])!=set(reference['groups']):raise StopTraining('evaluation_group_mismatch')
    groups={key:grade_group(candidate['groups'][key],reference['groups'][key],final=final) for key in candidate['groups']}
    if any('invalid_denominator' in row['reasons'] for row in groups.values()):raise StopTraining('invalid_evaluation_denominator')
    return dict(passed=all(row['passed'] for row in groups.values()),groups=groups,
                candidate_model_sha256=candidate['model_sha256'],reference_model_sha256=reference['model_sha256'])


class EventJournal:
    """Append-only, hash-chained protocol facts independent of rollback snapshots."""
    def __init__(self,path):self.path=Path(path)
    def read(self):
        if not self.path.exists():return []
        rows=[];previous='0'*64;ids=set()
        for line in self.path.read_text().splitlines():
            row=json.loads(line);body={k:v for k,v in row.items() if k!='sha256'}
            if row.get('seq')!=len(rows)+1 or row.get('previous_sha256')!=previous or row['id'] in ids or row.get('sha256')!=cases.digest(body):
                raise StopTraining('controller_event_chain_damaged')
            rows.append(row);ids.add(row['id']);previous=row['sha256']
        return rows
    def append(self,event_id,payload):
        import fcntl
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            rows=self.read()
            for row in rows:
                if row['id']==event_id:
                    if row['payload']!=payload:raise StopTraining('controller_event_identity_mismatch:'+event_id)
                    return row
            row=dict(seq=len(rows)+1,previous_sha256=rows[-1]['sha256'] if rows else '0'*64,id=event_id,payload=payload)
            row['sha256']=cases.digest(row)
            f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            return row


class Orchestrator:
    def __init__(self,run_dir,engineering_admission,resource_selection=None,device='cuda:0',resume_reason=None,runner=None,global_probe_ledger=None,probe_budget_authorization=None,probe_sequence_id=None,evaluation_ledger=None,formal_handoff_plan=None,source_compatibility=None,learning_ledger=None):
        self.root=Path(run_dir).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.evaluation_ledger=Path(evaluation_ledger).resolve() if evaluation_ledger else self.root/'evaluation_costs.jsonl'
        self.learning_ledger=Path(learning_ledger).resolve() if learning_ledger else self.root/'learning_costs.jsonl'
        self.handoff_plan=Path(formal_handoff_plan).resolve() if formal_handoff_plan else None
        self.source_compatibility=Path(source_compatibility).resolve() if source_compatibility else None
        self.handoff=None
        self.state_path=self.root/'controller_state.json';self.approval=Path(engineering_admission).resolve()
        self.selection=Path(resource_selection).resolve() if resource_selection else None
        self.device=device;self.resume_reason=resume_reason;self.runner=runner or subprocess.run
        from .probe_budget import DEFAULT_GLOBAL_LEDGER
        self.global_probe_ledger=Path(global_probe_ledger or DEFAULT_GLOBAL_LEDGER).resolve()
        self.probe_budget_authorization=Path(probe_budget_authorization).resolve() if probe_budget_authorization else None
        self.probe_sequence_id=probe_sequence_id
        self.state=(read_json(self.state_path) if self.state_path.exists() else dict(
            schema='a2-v10-controller-v1',sds_sha256=spec.SDS_SHA256,rulings_sha256=file_sha(RULINGS_PATH),code_sha256=code_fingerprint(),
            status='new',operations={},teacher_admission={},gates={},finals={},certificates={},
            learning_checkpoint=None,accepted_dagger=None,p0_checkpoint=None,last_passed_ppo=None,
            retention=asdict(RetentionState()),rerun_reasons=[],phase='prepare'))
        self._completed=False
        self.events=EventJournal(self.root/'controller_events.jsonl')
        self._replay_events()

    def _apply_event(self,event):
        payload=event['payload']
        if payload.get('sds_sha256')!=spec.SDS_SHA256 or payload.get('rulings_sha256')!=self.state.get('rulings_sha256') or self.state.get('rulings_sha256')!=file_sha(RULINGS_PATH):
            raise StopTraining('controller_event_protocol_changed')
        for path,digest in payload.get('references',{}).items():
            try:valid=file_sha(path)==digest
            except OSError:valid=False
            if not valid:raise StopTraining('controller_event_reference_changed:'+path)
        certificate=payload.get('certificate')
        if certificate is not None:
            path=Path(payload['certificate_path'])
            if path.exists() and read_json(path)!=certificate:raise StopTraining('acceptance_certificate_changed')
            if not path.exists():atomic_json(path,certificate)
        if event['seq']<=self.state.get('event_sequence',0):return
        patch=copy.deepcopy(payload['patch'])
        if event['id'].startswith('formal-source-transition:') and self.state.get('code_sha256')==patch.get('code_sha256'):
            for field in ('operations','teacher_admission','teacher_admitted'):patch.pop(field,None)
        if event['id'].startswith('formal-handoff-adopt:') and self.state.get('formal_handoff')==patch.get('formal_handoff'):
            patch.pop('learning_checkpoint',None)
        if 'retention' in patch:
            for field in ('rollbacks','failures'):
                if patch['retention'][field]<self.state['retention'][field]:raise StopTraining('rollback_counters_went_backwards')
        if 'ppo_checked' in patch and not set(self.state.get('ppo_checked',[]))<=set(patch['ppo_checked']):raise StopTraining('processed_gate_went_backwards')
        for key,value in patch.items():
            if key in ('finals','certificates','gates','courses','promotion_cancellations'):
                for name,row in value.items():
                    prior=self.state.setdefault(key,{}).get(name)
                    if key=='finals' and prior and prior['candidate_sha256']!=row['candidate_sha256']:raise StopTraining('final_cross_candidate_merge')
                    self.state[key][name]=copy.deepcopy(row)
            else:self.state[key]=copy.deepcopy(value)
        self.state['event_sequence']=event['seq'];self.state['event_sha256']=event['sha256']

    def _replay_events(self):
        events=self.events.read()
        if not events:return
        # Snapshots/checkpoint.extra are caches, not authorities for these monotonic facts.
        for key in ('finals','certificates','courses','promotion_cancellations'):self.state[key]={}
        self.state.update(retention=asdict(RetentionState()),rollback_history=[],ppo_checked=[],last_passed_ppo=None,
                          accepted_dagger=None,accepted_ppo=None)
        for key in ('dagger_certificate','ppo_certificate','ppo_resume'):self.state.pop(key,None)
        self.state['event_sequence']=0
        for event in events:self._apply_event(event)

    def commit_event(self,event_id,patch,references=None,certificate=None,certificate_path=None):
        payload=dict(sds_sha256=spec.SDS_SHA256,rulings_sha256=self.state['rulings_sha256'],
                     patch=patch,references=references or {})
        if certificate is not None:payload.update(certificate=certificate,certificate_path=str(certificate_path))
        event=self.events.append(event_id,payload)
        self._apply_event(event);self.persist();return event

    def persist(self):atomic_json(self.state_path,self.state)

    def _operation_dir(self,key):
        base=self.root/'operations'
        if self.state.get('source_transition'):base=base/('handoff_'+self.state['code_sha256'][:16])
        return base/key

    def _evaluation_root(self):
        base=self.root/'evaluation'
        if self.state.get('source_transition'):base=base/('handoff_'+self.state['code_sha256'][:16])
        return base

    def _migrate_controller_source(self):
        if self.state.get('code_sha256')==code_fingerprint():return
        if not self.handoff_plan or not self.source_compatibility:raise StopTraining('controller_code_or_SDS_changed')
        from .source_compatibility import validate_compatibility,validate_compatible_artifact
        bridge=validate_compatibility(self.source_compatibility)
        if self.state.get('code_sha256')!=bridge['from_code_sha256']:raise StopTraining('unapproved_controller_origin_source')
        validate_engineering(self.approval,bridge['to_code_sha256'])
        refs={}
        for purpose,operation in [('controller_state','controller_resume'),('frozen_execution_plan','freeze_plan')]:
            entries=[r for r in bridge['document']['artifacts'] if r['purpose']==purpose and operation in r['operation_ids']]
            if len(entries)!=1:raise StopTraining('exact_controller_source_transition_artifacts_required')
            value=validate_compatible_artifact(entries[0]['path'],bridge,purpose,operation_id=operation)
            if purpose=='controller_state' and value!=self.state:raise StopTraining('controller_origin_snapshot_mismatch')
            if purpose=='frozen_execution_plan' and value.get('code_sha256')!=bridge['from_code_sha256']:raise StopTraining('old_execution_plan_source_mismatch')
            refs[entries[0]['path']]=entries[0]['sha256']
        transition=dict(from_code_sha256=bridge['from_code_sha256'],to_code_sha256=bridge['to_code_sha256'],compatibility_sha256=bridge['sha256'],
                        prior_operations=copy.deepcopy(self.state['operations']),prior_teacher_admission=copy.deepcopy(self.state['teacher_admission']))
        refs[bridge['path']]=bridge['sha256']
        self.commit_event('formal-source-transition:'+bridge['sha256'],dict(code_sha256=bridge['to_code_sha256'],source_transition=transition,
            operations={},teacher_admission={},teacher_admitted=False),refs)

    def _assert_identity(self):
        if self.state.get('sds_sha256')!=spec.SDS_SHA256 or self.state.get('code_sha256')!=code_fingerprint():
            raise StopTraining('controller_code_or_SDS_changed')
        if file_sha(spec.SDS_PATH)!=spec.SDS_SHA256:raise StopTraining('SDS_file_changed')
        if self.state.get('rulings_sha256')!=file_sha(RULINGS_PATH):raise StopTraining('frozen_rulings_changed')
        validate_engineering(self.approval,self.state['code_sha256'])
        if self.state.get('learning_ledger') and self.state['learning_ledger']!=str(self.learning_ledger):raise StopTraining('learning_ledger_changed')
        self.state['learning_ledger']=str(self.learning_ledger)
        if self.state.get('evaluation_ledger') and self.state['evaluation_ledger']!=str(self.evaluation_ledger):raise StopTraining('evaluation_ledger_changed')
        self.state['evaluation_ledger']=str(self.evaluation_ledger)

    def _validate_receipt(self,key,result):
        if key.startswith(('teacher_admission_','gate_','final_')):
            try:
                report=Path(result['resource_report_path'])
                valid=(result.get('resource_admitted') is True and report.is_absolute() and file_sha(report)==result['resource_report_sha256'] and read_json(report).get('passed') is True)
            except (KeyError,OSError,ValueError):valid=False
            if not valid:raise StopTraining('evaluation_resource_receipt_invalid:'+key)
        if key.startswith(('entry_','learning_')):
            try:valid=(result.get('gate_reached') is True and result['checkpoint_sha256']==file_sha(result['checkpoint']))
            except (KeyError,OSError):valid=False
            if not valid:raise StopTraining('worker_incomplete_receipt:'+key)

    def _launch(self,key,command,result_path):
        """Durable idempotent child invocation, with no automatic resource reruns."""
        self._assert_identity()
        operation=self.state['operations'].get(key)
        op_dir=self._operation_dir(key);op_dir.mkdir(parents=True,exist_ok=True)
        durable=op_dir/'result.json';result_path=Path(result_path)
        if operation and operation['status']=='succeeded':
            if not durable.exists() or file_sha(durable)!=operation['result_sha256']:raise StopTraining('operation_receipt_changed:'+key)
            result=read_json(durable);self._validate_receipt(key,result);return result
        if operation:
            # A parent crash after a completed child may adopt only its exact result receipt.
            if operation['status']=='running' and result_path.exists():
                result=read_json(result_path);self._validate_receipt(key,result);atomic_json(durable,result)
                operation.update(status='succeeded',result_sha256=file_sha(durable),adopted_after_parent_restart=True)
                self.persist();return result
            if not self.resume_reason:raise StopTraining('interrupted_operation_requires_reason:'+key)
            if key.startswith('final_'):raise StopTraining('final_suite_not_automatically_retried:'+key)
            self.state['rerun_reasons'].append(dict(operation=key,reason=self.resume_reason))
        # Stale worker_result from another target must not be adopted after a crash.
        if result_path.exists():result_path.unlink()
        attempt=(operation or {}).get('attempts',0)+1
        self.state['operations'][key]=dict(status='running',command=list(map(str,command)),attempts=attempt)
        self.persist()
        snapshot=op_dir/'controller_snapshot.json';atomic_json(snapshot,dict(self.state,worker_operation_id=key))
        command=[str(x).replace('__CONTROLLER_STATE__',str(snapshot)) for x in command]
        env=dict(os.environ,MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'),PYTHONUNBUFFERED='1')
        try:
            with (op_dir/f'attempt_{attempt}.log').open('w') as log:
                completed=self.runner(command,cwd=spec.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if not result_path.exists():raise StopTraining(f'child_no_result:{key}:exit={completed.returncode}')
            result=read_json(result_path)
            self._validate_receipt(key,result)
            if completed.returncode!=0 and not (key=='resource_probe' and result.get('passed') is False):
                raise StopTraining(f'child_failed:{key}:exit={completed.returncode}')
            atomic_json(durable,result)
            self.state['operations'][key].update(status='succeeded',result_sha256=file_sha(durable))
            self.persist();return result
        except BaseException as error:
            self.state['operations'][key].update(status='failed',error=repr(error));self.persist();raise

    def prepare(self):
        self._assert_identity()
        if self.handoff_plan:
            if not self.source_compatibility:raise StopTraining('formal_handoff_requires_compatibility')
            checked=preflight_formal_handoff(self.handoff_plan,self.approval,self.source_compatibility)
            self.pending_handoff=checked['plan']
            if (self.pending_handoff['learning_ledger_path']!=str(self.learning_ledger) or self.pending_handoff['evaluation_ledger_path']!=str(self.evaluation_ledger)):
                raise StopTraining('handoff_original_ledgers_required')
        # Freeze actual suites before opening any A/B or final set; all worker regenerations
        # are checked against these hashes. TRAIN is an RNG-defined flow, not an eval set.
        freeze=self.root/'frozen';freeze.mkdir(exist_ok=True)
        plan=cases.evaluation_plan()
        if len(plan['runs'])!=71 or plan['max_samples']!=1685390840:raise StopTraining('frozen_evaluation_plan_changed')
        frozen=dict(sds_sha256=spec.SDS_SHA256,code_sha256=self.state['code_sha256'],evaluation_plan=plan,
                    rulings_sha256=self.state['rulings_sha256'],training_flow=dict(seed=42,role_order=list(spec.ROLES),initialization='specified uniform x/joint offsets; resumable Python RNG'),suites={})
        for c in ('D1','D2','D3','PPO'):
            for suite,replicas in [('A',20),('B',20)]+([('DAGGER_FINAL',100)] if c=='D3' else [('PPO_FINAL',100)] if c=='PPO' else []):
                manifest=cases.suite(c,suite,replicas);digest=cases.digest(manifest)
                key=f'{c}_{suite}';path=freeze/(key+'.json')
                document=dict(sha256=digest,cases=manifest,sds_sha256=spec.SDS_SHA256)
                if path.exists() and read_json(path)!=document:raise StopTraining('frozen_case_manifest_changed:'+key)
                if not path.exists():atomic_json(path,document)
                frozen['suites'][key]=dict(sha256=digest,path=str(path),replicas=replicas,groups=len(cases.groups(c)),max_samples=sum(r['max_steps'] for r in manifest))
        config=freeze/(('execution_plan_handoff_'+self.state['code_sha256']+'.json') if self.state.get('source_transition') else 'execution_plan.json')
        if config.exists() and read_json(config)!=frozen:raise StopTraining('frozen_execution_plan_changed')
        atomic_json(config,frozen);self.frozen=frozen
        self.state['frozen_sha256']=file_sha(config);self.state['evaluation_max_samples']=plan['max_samples']
        self.state['status']='running';self.persist()
        from .probe_budget import GlobalProbeBudget
        budget=GlobalProbeBudget(self.global_probe_ledger,self.probe_budget_authorization)
        authorization=budget.validate_authorization() # No authorization means no child, even on a new run ID.
        sequence=self.probe_sequence_id or self.state.get('probe_sequence_id') or ('probe-'+cases.digest(str(self.root))[:24])
        if self.state.get('global_probe_ledger') and self.state['global_probe_ledger']!=str(self.global_probe_ledger):raise StopTraining('global_probe_ledger_changed')
        if self.state.get('probe_sequence_id') and self.state['probe_sequence_id']!=sequence:raise StopTraining('probe_sequence_identity_changed')
        self.state.update(global_probe_ledger=str(self.global_probe_ledger),probe_sequence_id=sequence,probe_budget_authorization_sha256=authorization['sha256'])
        self.persist()
        if self.selection is None:
            out=self.root/'resource_probe'
            selection=self._launch('resource_probe',[sys.executable,spec.ROOT/'scripts/run_a2_v10.py','probe','--output-dir',out,'--device',self.device,'--global-probe-ledger',self.global_probe_ledger,
                '--probe-budget-authorization',self.probe_budget_authorization,'--probe-sequence-id',sequence],out/'selection.json')
            self.selection=out/'selection.json'
        else:selection=read_json(self.selection)
        self.admit_resource_selection(selection,authorization,sequence)

    def admit_resource_selection(self,selection,authorization,sequence):
        """Consume the real selector receipt without constructing GPU state."""
        validate_engineering(self.approval,self.state['code_sha256'])
        expected_source=self.state['code_sha256']
        if self.source_compatibility:
            from .source_compatibility import validate_compatible_resource
            inherited=validate_compatible_resource(self.selection,self.source_compatibility)
            if inherited!=selection:raise StopTraining('inherited_resource_selection_changed')
            expected_source=inherited['code_sha256']
        if (selection.get('passed') is not True or selection.get('num_envs') not in (1024,512)
            or selection.get('sds_sha256')!=spec.SDS_SHA256 or selection.get('code_sha256')!=expected_source):raise StopTraining('resource_selection_invalid')
        if (selection.get('global_probe_ledger')!=str(self.global_probe_ledger) or selection.get('probe_sequence_id')!=sequence
            or selection.get('probe_budget_authorization_sha256')!=authorization['sha256']):raise StopTraining('resource_selection_global_budget_identity_invalid')
        targeted=selection.get('resource_protocol') in ('A1-targeted-resource-v1','A1-targeted-resource-v2')
        if targeted:
            from .resource_a1 import verify_linkage_selection
            try:verify_linkage_selection(selection,source_compatibility=self.source_compatibility)
            except (ValueError,OSError,KeyError) as error:raise StopTraining('A1_resource_linkage_invalid') from error
            if selection['num_envs']!=1024:raise StopTraining('A1_resource_N_changed')
        expected_warmup,expected_measurement=((1,2) if selection.get('resource_protocol')=='A1-targeted-resource-v2' else (2,2)) if targeted else (5,20)
        n=selection['num_envs'];runs=[r for r in selection['runs'] if r.get('num_envs')==n]
        if len(runs)!=2 or {r.get('stack') for r in runs}!={'dagger','ppo'}:raise StopTraining('both_complete_resource_stacks_required')
        for row in runs:
            if (row.get('passed') is not True or row.get('restore_checked') is not True or row.get('sds_sha256')!=spec.SDS_SHA256
                or row.get('code_sha256')!=expected_source or row.get('gpu_total_peak_mib',1e9)>7680
                or row.get('gpu_task_peak_mib',1e9)>7168 or row.get('host_available_min_bytes',0)<4*1024**3
                or row.get('warmup_rounds')!=expected_warmup or row.get('measurement_rounds')!=expected_measurement
                or any(row.get('replay_counts',{}).get(role,0)<32768 for role in spec.ROLES)):
                raise StopTraining('resource_stack_incomplete')
        physics={r['physical_sha256'] for r in runs}
        if len(physics)!=1:raise StopTraining('probe_physics_mismatch')
        physical=next(iter(physics));approval=read_json(self.approval)
        if approval.get('physical_sha256') and approval['physical_sha256']!=physical:raise StopTraining('engineering_probe_physics_mismatch')
        if self.state.get('physical_sha256') and (self.state['physical_sha256']!=physical or self.state['num_envs']!=n):raise StopTraining('formal_N_or_physics_changed')
        if self.state.get('resource_selection_sha256') and self.state['resource_selection_sha256']!=file_sha(self.selection):raise StopTraining('resource_selection_receipt_changed')
        self.state.update(num_envs=n,physical_sha256=physical,resource_selection_sha256=file_sha(self.selection),schedule=execution_plan(n))
        self.persist()

    def evaluate(self,key,course,suite,model_path,reference):
        replicas=100 if suite.endswith('FINAL') else 20
        out=self._evaluation_root()/key
        command=[sys.executable,'-m','src.tasks.velocity.v10.eval_worker','--course',course,'--suite',suite,'--replicas',str(replicas),
                 '--reference',reference,'--output-dir',out,'--ledger',self.evaluation_ledger,
                 '--eval-limit',str(self.state['evaluation_max_samples']),'--device',self.device]
        if model_path is not None:command+=['--model-path',model_path]
        if key.startswith('teacher_admission_'):command+=['--stop-on-failed-group']
        result=self._launch(key,command,out/'result.json')
        self._validate_receipt(key,result)
        if not self._evaluation_identity_matches(result,course,suite,reference,model_path,
                allow_aborted=key.startswith('teacher_admission_')):
            raise StopTraining('evaluation_worker_receipt_mismatch:'+key)
        return result

    def _recover_worker_checkpoint(self,key,course,target,out,resume):
        if not self.handoff or key not in self.state['operations']:return resume
        path=Path(out)/'worker_checkpoint_manifest.json'
        if not path.exists():return resume
        row=read_json(path)
        if row.get('operation_id')!=key:return resume
        expected=dict(schema='a2-v10-worker-recovery-v1',course=course,num_envs=self.state['num_envs'],
            physical_sha256=self.state['physical_sha256'],code_sha256=self.state['code_sha256'],
            learning_ledger_path=str(self.learning_ledger),operation_id=key,target_round=target,
            formal_handoff_sha256=self.handoff['sha256'])
        cp=row.get('checkpoint',{})
        if any(row.get(k)!=v for k,v in expected.items()):raise StopTraining('worker_recovery_manifest_identity_mismatch')
        candidate=Path(cp.get('path','')).resolve()
        if candidate.parent!=Path(out).resolve() or not candidate.is_file() or file_sha(candidate)!=cp.get('sha256'):
            raise StopTraining('worker_recovery_checkpoint_changed')
        nstep=self.state['num_envs']*48
        if (not isinstance(row.get('round'),int) or not 0<=row['round']<=target
            or not isinstance(row.get('confirmed_samples'),int) or row['confirmed_samples']//nstep!=row['round']):raise StopTraining('worker_recovery_progress_invalid')
        from .checkpoint import BudgetLedger
        budget=BudgetLedger(self.learning_ledger).report();budget_key='learning:'+('PPO' if course.startswith('P') else course)
        if budget['executed'].get(budget_key,0)<row['confirmed_samples']:raise StopTraining('worker_recovery_external_ledger_behind')
        return str(candidate)

    def worker(self,key,course,target,resume=None,acceptance=None):
        out=self.root/'learning'/course
        resume=self._recover_worker_checkpoint(key,course,target,out,resume)
        command=[sys.executable,spec.ROOT/'scripts/run_a2_v10.py','worker','--output-dir',out,'--device',self.device,
                 '--num-envs',str(self.state['num_envs']),'--course',course,'--stop-round',str(target),
                 '--ledger-path',self.learning_ledger,'--controller-state','__CONTROLLER_STATE__']
        if self.handoff:command+=['--formal-handoff',self.handoff['path']]
        if resume:command+=['--resume',resume]
        if acceptance:command+=['--acceptance',acceptance]
        result=self._launch(key,command,out/'worker_result.json')
        if (not result.get('gate_reached') or result.get('course')!=course or result.get('round')<target
            or result.get('physical_sha256')!=self.state['physical_sha256'] or not Path(result.get('checkpoint','')).is_file()
            or result.get('checkpoint_sha256')!=file_sha(result['checkpoint'])):raise StopTraining('learning_worker_target_not_reached:'+key)
        overshoot=result['consumed']-target*self.state['num_envs']*48
        if not 0<=overshoot<self.state['num_envs']*48:raise StopTraining('learning_worker_sample_mismatch')
        budget_key='PPO' if course.startswith('P') else course
        cap=spec.P0_BUDGET if course=='P0' else spec.BUDGETS[budget_key]
        charged=result.get('budget',{}).get('charged',{}).get('learning:'+budget_key,result['consumed'])
        if charged>cap:raise StopTraining('learning_worker_charged_cap_exceeded')
        self.state['learning_checkpoint']=result['checkpoint'];self.persist();return result['checkpoint']

    def teacher_admission(self):
        from .evaluation_capacity import evaluation_capacity
        from .teachers import teacher_identity
        for course in ('D1','D2','D3'):
            pending=[suite for suite in ('A','B') if not self.state['teacher_admission'].get(f'teacher_admission_{course}_{suite}',{}).get('passed')]
            if not pending:continue
            required=sum(sum(cases.max_steps(row)*20 for row in cases.groups(course)) for _ in pending)
            stage='teachers:'+course+':'+cases.digest(teacher_identity())
            with evaluation_capacity(self.evaluation_ledger,stage,required,self.state['evaluation_max_samples']):
                for suite in pending:
                    key=f'teacher_admission_{course}_{suite}'
                    result=self.evaluate(key,course,suite,None,'teachers')
                    if result.get('aborted_for_teacher_failure'):
                        self.state['teacher_admission'][key]=dict(passed=False,reason='teacher_capability_gap',
                            tested_case_count=result.get('tested_case_count'),planned_case_count=result.get('planned_case_count'),groups=result['groups'])
                        self.persist();raise StopTraining('teacher_capability_gap:'+key)
                    comparison=compare_results(result,result)
                    self.state['teacher_admission'][key]=comparison;self.persist()
                    if not comparison['passed']:raise StopTraining('teacher_capability_gap:'+key)
        self.state['teacher_admitted']=True;self.persist()

    def _evaluation_identity_matches(self,row,course,suite,reference,model_path,allow_aborted=False):
        """Shared identity predicate for result consumption and capacity savings.

        This proves correspondence/completeness, never behavioral acceptance.
        A teacher early-stop can be consumed only in its original admission path;
        it remains incomplete and can never justify saved future capacity.
        """
        try:
            replicas=100 if suite.endswith('FINAL') else 20
            frozen=self.frozen['suites'][f'{course}_{suite}']
            document=read_json(frozen['path']);manifest=document['cases']
            if document.get('sds_sha256')!=spec.SDS_SHA256 or document.get('sha256')!=frozen['sha256'] or cases.digest(manifest)!=frozen['sha256']:return False
            from .teachers import teacher_identity
            model_sha=cases.digest(teacher_identity()) if reference=='teachers' else file_sha(model_path)
            identity=dict(course=course,suite=suite,reference=reference,replicas=replicas,suite_sha256=frozen['sha256'],
                          model_sha256=model_sha,code_sha256=self.state['code_sha256'],sds_sha256=spec.SDS_SHA256,
                          rulings_sha256=self.state['rulings_sha256'],physical_sha256=self.state['physical_sha256'])
            if any(row.get(k)!=v for k,v in identity.items()):return False
            aborted=row.get('aborted_for_teacher_failure') is True
            if aborted:
                if not allow_aborted or reference!='teachers' or row.get('complete') is not False:return False
            elif row.get('complete') is not True:return False
            expected={}
            for case in manifest:
                if case.get('course')!=course or case.get('suite')!=suite:return False
                expected.setdefault(case['group_id'],[]).append(case['case_sha256'])
            groups=row['groups']
            if not groups or (not aborted and set(groups)!=set(expected)) or (aborted and not set(groups)<=set(expected)):return False
            for group,actual in groups.items():
                case_ids=expected[group]
                if len(case_ids)!=replicas or len(set(case_ids))!=replicas or len(actual)!=replicas:return False
                if any(r.get('group_id')!=group for r in actual):return False
                ids=[r['case_sha256'] for r in actual]
                if len(set(ids))!=replicas or set(ids)!=set(case_ids):return False
            return True
        except (AttributeError,KeyError,TypeError,ValueError,OSError):return False

    def _capacity_receipt_matches(self,key,course,suite,reference,model_path):
        """A durable, resource-qualified exact complete result saves capacity."""
        try:
            operation=self.state['operations'].get(key,{})
            path=self._operation_dir(key)/'result.json'
            if operation.get('status')!='succeeded' or not path.is_file() or operation.get('result_sha256')!=file_sha(path):return False
            row=read_json(path);self._validate_receipt(key,row)
            return self._evaluation_identity_matches(row,course,suite,reference,model_path)
        except (StopTraining,KeyError,TypeError,ValueError,OSError):return False

    def _stage_capacity(self,stage_id,course,suites,model_paths,operation_prefix):
        from contextlib import nullcontext
        from .evaluation_capacity import evaluation_capacity
        required=0
        for suite in suites:
            replicas=100 if suite.endswith('FINAL') else 20
            cost=sum(cases.max_steps(row)*replicas for row in cases.groups(course))
            for model,model_path in model_paths.items():
                key=(f'{operation_prefix}_{model}' if suite.endswith('FINAL') else f'{operation_prefix}_{suite}_{model}')
                if not self._capacity_receipt_matches(key,course,suite,model,model_path):required+=cost
        return evaluation_capacity(self.evaluation_ledger,stage_id,required,self.state['evaluation_max_samples']) if required else nullcontext()

    def gate(self,course,target,checkpoint,references):
        key=f'{course}_{target}'
        if key in self.state['gates']:return self._gate_unleased(course,target,checkpoint,references)
        digest=file_sha(checkpoint)
        with self._stage_capacity('gate:'+key+':'+digest,course,('A','B'),{'candidate':checkpoint,**references},f'gate_{key}'):
            return self._gate_unleased(course,target,checkpoint,references)

    def _gate_unleased(self,course,target,checkpoint,references):
        key=f'{course}_{target}';checkpoint_sha=file_sha(checkpoint)
        prior=self.state['gates'].get(key)
        if prior:
            if prior['candidate_sha256']!=checkpoint_sha:raise StopTraining('gate_candidate_changed')
            return prior
        result=dict(candidate_sha256=checkpoint_sha,passed=False,retention_failed=False,suites={})
        for suite in ('A','B'):
            candidate=self.evaluate(f'gate_{key}_{suite}_candidate',course,suite,checkpoint,'candidate')
            checks={}
            for reference,path in references.items():
                ref=self.evaluate(f'gate_{key}_{suite}_{reference}',course,suite,path,reference)
                checks[reference]=compare_results(candidate,ref)
            result['suites'][suite]=checks
            result['retention_failed']|=any(is_retention_failure(x) for x in checks.values())
            if file_sha(checkpoint)!=checkpoint_sha:raise StopTraining('candidate_mutated_between_A_B')
            if not all(c['passed'] for c in checks.values()):break
        result['passed']=len(result['suites'])==2 and all(c['passed'] for checks in result['suites'].values() for c in checks.values())
        self.state['gates'][key]=result;self.persist();return result

    def open_final(self,name,checkpoint_sha,checkpoint=None):
        if name in self.state['finals']:raise StopTraining('final_already_opened:'+name)
        patch=dict(finals={name:dict(status='opened',candidate_sha256=checkpoint_sha)})
        refs={str(checkpoint):checkpoint_sha} if checkpoint else {}
        self.commit_event('final-open:'+name,patch,refs)

    def final(self,name,course,checkpoint,references):
        existing=self.state['finals'].get(name)
        if existing and existing['status'] in ('passed','failed'):
            return self._final_unleased(name,course,checkpoint,references)
        # Reject insufficient capacity before the irreversible final-open event.
        with self._stage_capacity('final:'+name+':'+file_sha(checkpoint),course,(name,),{'candidate':checkpoint,**references},f'final_{name}'):
            return self._final_unleased(name,course,checkpoint,references)

    def _final_unleased(self,name,course,checkpoint,references):
        digest=file_sha(checkpoint)
        existing=self.state['finals'].get(name)
        if existing and existing['status']=='passed':
            if existing['candidate_sha256']!=digest:raise StopTraining('final_candidate_changed')
            return existing
        if existing:
            candidate_op=self.state['operations'].get(f'final_{name}_candidate',{})
            artifact=self._evaluation_root()/f'final_{name}_candidate'/'result.json'
            if (existing['status']!='opened' or existing['candidate_sha256']!=digest
                or not (candidate_op.get('status')=='succeeded' or candidate_op.get('status')=='running' and artifact.exists())):
                raise StopTraining('final_already_opened:'+name)
            # Continue only from a completed immutable candidate result, never rerun it.
        else:self.open_final(name,digest,checkpoint)
        try:
            candidate=self.evaluate(f'final_{name}_candidate',course,name,checkpoint,'candidate')
            checks={}
            for reference,path in references.items():
                ref=self.evaluate(f'final_{name}_{reference}',course,name,path,reference)
                checks[reference]=compare_results(candidate,ref,final=True)
            passed=all(row['passed'] for row in checks.values())
            result=dict(status='passed' if passed else 'failed',candidate_sha256=digest,comparisons=checks,
                        suite_sha256=candidate['suite_sha256'],candidate_result_sha256=file_sha(self._evaluation_root()/f'final_{name}_candidate'/'result.json'))
            verdict_path=self.root/'final_results'/(name+'.json');atomic_json(verdict_path,result)
            result['result_sha256']=file_sha(verdict_path)
            if not passed:
                self.commit_event('final-failed:'+name,dict(finals={name:result}),{str(checkpoint):digest,str(verdict_path):file_sha(verdict_path)})
                raise StopTraining('independent_final_failed:'+name)
            self.certificate('V10_DAGGER_ACCEPTED' if name=='DAGGER_FINAL' else 'V10_DAGGER_PPO_ACCEPTED',checkpoint,result)
            return result
        except BaseException as error:
            accepted_id='acceptance:'+('V10_DAGGER_ACCEPTED' if name=='DAGGER_FINAL' else 'V10_DAGGER_PPO_ACCEPTED')
            acceptance_recorded=any(e['id']==accepted_id for e in self.events.read())
            if not acceptance_recorded and self.state['finals'][name]['status'] not in ('passed','failed'):
                failed=dict(self.state['finals'][name],status='failed',reason=repr(error))
                self.commit_event('final-failed:'+name,dict(finals={name:failed}),{str(checkpoint):digest})
            raise

    def certificate(self,name,checkpoint,final):
        final_name='DAGGER_FINAL' if name=='V10_DAGGER_ACCEPTED' else 'PPO_FINAL'
        certificate=dict(token=name,passed=True,sds_sha256=spec.SDS_SHA256,rulings_sha256=self.state['rulings_sha256'],code_sha256=self.state['code_sha256'],
            model_sha256=file_sha(checkpoint),physical_sha256=self.state['physical_sha256'],suite_sha256=final['suite_sha256'],
            result_sha256=final['result_sha256'],frozen_plan_sha256=self.state['frozen_sha256'])
        path=self.root/(name+'.json')
        refs={str(checkpoint):file_sha(checkpoint),str(self.root/'final_results'/(final_name+'.json')):final['result_sha256']}
        for receipt in self._evaluation_root().glob(f'final_{final_name}_*/result.json'):
            refs[str(receipt)]=file_sha(receipt)
        accepted='accepted_dagger' if final_name=='DAGGER_FINAL' else 'accepted_ppo'
        cert_key='dagger_certificate' if final_name=='DAGGER_FINAL' else 'ppo_certificate'
        patch=dict(finals={final_name:final},certificates={name:str(path)},**{accepted:str(checkpoint),cert_key:str(path)})
        self.commit_event('acceptance:'+name,patch,refs,certificate,path)
        return str(path)

    def _promote_course(self,course,target,checkpoint,grade):
        """Journal promotion and cancellation facts together, with no budget effect."""
        digest=file_sha(checkpoint)
        if (grade.get('passed') is not True or grade.get('candidate_sha256')!=digest
            or set(grade.get('suites',{}))!={'A','B'}
            or any(set(checks)!={'teachers'} for checks in grade['suites'].values())
            or any(check.get('passed') is not True or check.get('candidate_model_sha256')!=digest
                   for checks in grade['suites'].values() for check in checks.values())):
            raise StopTraining('course_promotion_requires_same_candidate_A_B_pass')
        event_id=f'course-promotion:{course}:{target}'
        grade_path=self.root/'promotion_evidence'/f'{course}_{target}_passed_gate.json'
        if grade_path.exists():
            if read_json(grade_path)!=grade:raise StopTraining('immutable_promotion_grade_changed')
        else:atomic_json(grade_path,grade)
        grade_sha=file_sha(grade_path)
        references={str(checkpoint):digest,str(grade_path):grade_sha}
        operations=[]
        for future in self.state['schedule']['gates'][course]:
            if future<=target:continue
            for suite in ('A','B'):
                frozen=self.frozen['suites'][f'{course}_{suite}']
                bound=int(frozen['max_samples'])
                if frozen.get('path'):references[frozen['path']]=file_sha(frozen['path'])
                for model in ('candidate','teachers'):
                    operation=f'gate_{course}_{future}_{suite}_{model}'
                    if operation in self.state['operations']:continue
                    operations.append(dict(operation_id=operation,target=future,suite=suite,model=model,
                        case_max_samples=bound,suite_sha256=frozen['sha256'],reference_scope='this_gate_invocation_only'))
        completed=dict(passed=True,passed_target=target,checkpoint=str(checkpoint),sha256=digest,promotion_event_id=event_id,
                       passed_grade_path=str(grade_path),passed_grade_sha256=grade_sha)
        patch=dict(courses={course:completed})
        if operations:
            patch['promotion_cancellations']={f'{course}:{target}':dict(course=course,passed_target=target,
                checkpoint=str(checkpoint),candidate_sha256=digest,trigger_event_id=event_id,
                trigger_grade_path=str(grade_path),trigger_grade_sha256=grade_sha,reason='early_promotion',
                cancelled_operations=operations,original_case_max_samples=sum(r['case_max_samples'] for r in operations),
                charged_samples=0,refunded_samples=0,transferred_samples=0,shared_reference_cache='preserved',
                resolution='cancelled_before_start_not_cache_hit')}
        self.commit_event(event_id,patch,references)

    def _train_dagger(self):
        if self.state.get('accepted_dagger'):return self.state['accepted_dagger']
        checkpoint=self.state['learning_checkpoint']
        for course in ('D1','D2','D3'):
            self.state['phase']=course;self.persist()
            completed=self.state.get('courses',{}).get(course)
            if completed:
                checkpoint=completed['checkpoint'];continue
            if not (course=='D1' and self.state.get('formal_handoff')):
                checkpoint=self.worker(f'entry_{course}',course,0,checkpoint)
            promoted=False
            for target in self.state['schedule']['gates'][course]:
                checkpoint=self.worker(f'learning_{course}_{target}',course,target,checkpoint)
                grade=self.gate(course,target,checkpoint,{'teachers':None})
                if grade['passed']:
                    self._promote_course(course,target,checkpoint,grade)
                    promoted=True;break
            if not promoted:raise StopTraining('course_budget_exhausted:'+course)
        final=self.final('DAGGER_FINAL','D3',checkpoint,{'teachers':None})
        certificate=self.certificate('V10_DAGGER_ACCEPTED',checkpoint,final)
        self.state['accepted_dagger']=checkpoint;self.state['dagger_certificate']=certificate;self.persist()
        return checkpoint

    def _train_ppo(self,dagger):
        references={'teachers':None,'dagger':dagger}
        if self.state.get('accepted_ppo'):return self.state['accepted_ppo']
        if self.state.get('last_passed_ppo'):
            checkpoint=self.state['last_passed_ppo']
            final=self.final('PPO_FINAL','PPO',checkpoint,references)
            self.certificate('V10_DAGGER_PPO_ACCEPTED',checkpoint,final)
            return checkpoint
        p0round=self.state['schedule']['p0_rounds']
        if not self.state.get('p0_checkpoint'):
            self.state['phase']='P0';self.persist()
            checkpoint=self.worker('entry_P0','P0',0,dagger,self.state['dagger_certificate'])
            checkpoint=self.worker('learning_P0','P0',p0round,checkpoint)
            self.state['p0_checkpoint']=checkpoint;self.persist()
        if self.state.get('ppo_resume'):
            checkpoint=self.state['ppo_resume']
        else:
            checkpoint=self.worker('entry_P1','P1',p0round,self.state['p0_checkpoint'])
            self.state['ppo_resume']=checkpoint;self.persist()
        for target in self.state['schedule']['gates']['PPO']:
            if target in self.state.get('ppo_checked',[]):continue
            self.state['phase']='P1';self.persist()
            checkpoint=self.worker(f'learning_PPO_{target}','P1',target,checkpoint)
            grade=self.gate('PPO',target,checkpoint,references)
            if grade['passed']:
                retention=RetentionState(**self.state['retention']);retention.record(True)
                self.commit_event(f'ppo-gate:{target}',dict(last_passed_ppo=checkpoint,ppo_resume=checkpoint,retention=asdict(retention),
                    ppo_checked=list(self.state.get('ppo_checked',[]))+[target],gates={f'PPO_{target}':grade}),{str(checkpoint):file_sha(checkpoint)})
                final=self.final('PPO_FINAL','PPO',checkpoint,references)
                certificate=self.certificate('V10_DAGGER_PPO_ACCEPTED',checkpoint,final)
                self.state.update(accepted_ppo=checkpoint,ppo_certificate=certificate);self.persist();return checkpoint
            retention=RetentionState(**self.state['retention']);stop=None;rollback=False
            try:rollback=retention.record(not grade['retention_failed'])
            except StopTraining as error:stop=error
            history=copy.deepcopy(self.state.get('rollback_history',[]))
            if rollback:
                checkpoint=self.state['last_passed_ppo'] or self.state['p0_checkpoint']
                history.append(dict(gate=target,resume=checkpoint,refund=0,number=retention.rollbacks))
            checked=list(self.state.get('ppo_checked',[]))+[target]
            patch=dict(retention=asdict(retention),ppo_resume=checkpoint,ppo_checked=checked,rollback_history=history,
                       gates={f'PPO_{target}':grade},p0_checkpoint=self.state['p0_checkpoint'])
            self.commit_event(f'ppo-gate:{target}',patch,{str(checkpoint):file_sha(checkpoint)})
            if stop:raise stop
        raise StopTraining('PPO_budget_exhausted')

    def _recover_evaluation_accounting(self):
        """Only the GPU lease proves an orphan child cannot still execute its step."""
        if not self.handoff_plan:return
        from .gpu_lock import gpu_execution_lock
        from .checkpoint import BudgetLedger
        ledger=BudgetLedger(self.evaluation_ledger)
        with gpu_execution_lock(blocking=False):
            before=ledger.report()
            ledger.reconcile_uncertain()
            after=ledger.report()
        if before['pending']:
            self.commit_event('handoff-evaluation-recovery:'+cases.digest(before),dict(evaluation_recovery=dict(
                charged_before=before['charged'],charged_after=after['charged'],uncertain=after['uncertain'],refund=0)),
                {str(self.handoff_plan):file_sha(self.handoff_plan)})

    def _finalize_formal_handoff(self):
        if not self.handoff_plan:return
        from .handoff import finalize_handoff,validate_handoff
        from .source_compatibility import validate_compatibility
        bridge=validate_compatibility(self.source_compatibility)
        if self.state.get('formal_handoff'):
            self.handoff=validate_handoff(self.state['formal_handoff']['path'],ledger_path=self.learning_ledger,num_envs=1024,physical_sha256=self.state['physical_sha256'])
            if self.handoff['sha256']!=self.state['formal_handoff']['sha256']:raise StopTraining('adopted_handoff_changed')
            return
        receipts={}
        for course in ('D1','D2','D3'):
            for suite in ('A','B'):
                key=f'teacher_admission_{course}_{suite}'
                path=self._operation_dir(key)/'result.json'
                receipts[key]=dict(path=str(path),sha256=file_sha(path))
        self.handoff=finalize_handoff(self.handoff_plan,receipts)
        if self.handoff['compatibility']['sha256']!=bridge['sha256']:raise StopTraining('handoff_source_bridge_changed')
        origin=self.handoff['origin_checkpoint']
        self.commit_event('formal-handoff-adopt:'+self.handoff['sha256'],dict(
            formal_handoff=dict(path=self.handoff['path'],sha256=self.handoff['sha256'],initial_round=self.handoff['initial_round']),
            learning_checkpoint=origin['path']),{self.handoff['path']:self.handoff['sha256'],origin['path']:origin['sha256']})

    def run(self,teachers_only=False):
        import fcntl
        # One controller owns GPU scheduling and all persistent transactions at a time.
        with (self.root/'controller.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise StopTraining('another_controller_is_running') from error
            try:
                self._migrate_controller_source();self.prepare();self._recover_evaluation_accounting();self.teacher_admission()
                if teachers_only:
                    return dict(passed=True,teacher_admitted=True,training_executed=False,state=str(self.state_path))
                self._finalize_formal_handoff()
                dagger=self._train_dagger();ppo=self._train_ppo(dagger)
                export=self.root/'deployment'
                result=self._launch('export_student',[sys.executable,spec.ROOT/'scripts/run_a2_v10.py','export','--checkpoint',ppo,
                    '--acceptance',self.state['ppo_certificate'],'--output-dir',export],export/'contract.json')
                if (result.get('sds_sha256')!=spec.SDS_SHA256 or result.get('checkpoint_sha256')!=file_sha(ppo)
                    or result.get('actor_sha256')!=file_sha(export/'student_actor.pt')):raise StopTraining('deployment_export_receipt_invalid')
                self.state.update(status='complete',phase='complete',deployment=str(export));self.persist();self._completed=True
                return dict(passed=True,checkpoint=ppo,dagger_checkpoint=dagger,deployment=str(export),state=str(self.state_path))
            except BaseException as error:
                self.state.update(status='stopped',stop_reason=repr(error));self.persist()
                if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
                return dict(passed=False,reason=str(error),state=str(self.state_path),dagger_checkpoint=self.state.get('accepted_dagger'))
