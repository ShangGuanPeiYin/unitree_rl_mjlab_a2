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


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    def __init__(self,run_dir,engineering_admission,resource_selection=None,device='cuda:0',resume_reason=None,runner=None):
        self.root=Path(run_dir).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.state_path=self.root/'controller_state.json';self.approval=Path(engineering_admission).resolve()
        self.selection=Path(resource_selection).resolve() if resource_selection else None
        self.device=device;self.resume_reason=resume_reason;self.runner=runner or subprocess.run
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
        patch=payload['patch']
        if 'retention' in patch:
            for field in ('rollbacks','failures'):
                if patch['retention'][field]<self.state['retention'][field]:raise StopTraining('rollback_counters_went_backwards')
        if 'ppo_checked' in patch and not set(self.state.get('ppo_checked',[]))<=set(patch['ppo_checked']):raise StopTraining('processed_gate_went_backwards')
        for key,value in patch.items():
            if key in ('finals','certificates','gates'):
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
        for key in ('finals','certificates'):self.state[key]={}
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

    def _assert_identity(self):
        if self.state.get('sds_sha256')!=spec.SDS_SHA256 or self.state.get('code_sha256')!=code_fingerprint():
            raise StopTraining('controller_code_or_SDS_changed')
        if file_sha(spec.SDS_PATH)!=spec.SDS_SHA256:raise StopTraining('SDS_file_changed')
        if self.state.get('rulings_sha256')!=file_sha(RULINGS_PATH):raise StopTraining('frozen_rulings_changed')
        validate_engineering(self.approval,self.state['code_sha256'])

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
        op_dir=self.root/'operations'/key;op_dir.mkdir(parents=True,exist_ok=True)
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
        snapshot=op_dir/'controller_snapshot.json';atomic_json(snapshot,self.state)
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
        config=freeze/'execution_plan.json'
        if config.exists() and read_json(config)!=frozen:raise StopTraining('frozen_execution_plan_changed')
        atomic_json(config,frozen);self.frozen=frozen
        self.state['frozen_sha256']=file_sha(config);self.state['evaluation_max_samples']=plan['max_samples']
        self.state['status']='running';self.persist()
        if self.selection is None:
            out=self.root/'resource_probe'
            selection=self._launch('resource_probe',[sys.executable,spec.ROOT/'scripts/run_a2_v10.py','probe','--output-dir',out,'--device',self.device],out/'selection.json')
            self.selection=out/'selection.json'
        else:selection=read_json(self.selection)
        if (selection.get('passed') is not True or selection.get('num_envs') not in (1024,512)
            or selection.get('sds_sha256')!=spec.SDS_SHA256 or selection.get('code_sha256')!=self.state['code_sha256']):raise StopTraining('resource_selection_invalid')
        n=selection['num_envs'];runs=[r for r in selection['runs'] if r.get('num_envs')==n]
        if len(runs)!=2 or {r.get('stack') for r in runs}!={'dagger','ppo'}:raise StopTraining('both_complete_resource_stacks_required')
        for row in runs:
            if (row.get('passed') is not True or row.get('restore_checked') is not True or row.get('sds_sha256')!=spec.SDS_SHA256
                or row.get('code_sha256')!=self.state['code_sha256'] or row.get('gpu_total_peak_mib',1e9)>7680
                or row.get('gpu_task_peak_mib',1e9)>7168 or row.get('host_available_min_bytes',0)<4*1024**3
                or row.get('warmup_rounds')!=5 or row.get('measurement_rounds')!=20
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
        out=self.root/'evaluation'/key
        command=[sys.executable,'-m','src.tasks.velocity.v10.eval_worker','--course',course,'--suite',suite,'--replicas',str(replicas),
                 '--reference',reference,'--output-dir',out,'--ledger',self.root/'evaluation_costs.jsonl',
                 '--eval-limit',str(self.state['evaluation_max_samples']),'--device',self.device]
        if model_path is not None:command+=['--model-path',model_path]
        if key.startswith('teacher_admission_'):command+=['--stop-on-failed-group']
        result=self._launch(key,command,out/'result.json')
        self._validate_receipt(key,result)
        expected=self.frozen['suites'][f'{course}_{suite}']
        if (result.get('suite_sha256')!=expected['sha256'] or result.get('physical_sha256')!=self.state['physical_sha256']
            or result.get('sds_sha256')!=spec.SDS_SHA256 or result.get('code_sha256')!=self.state['code_sha256']
            or result.get('rulings_sha256')!=self.state['rulings_sha256'] or result.get('replicas')!=replicas):raise StopTraining('evaluation_worker_receipt_mismatch:'+key)
        expected_model=(cases.digest(dict(blind=spec.BLIND_SHA256,stairs=spec.STAIRS_SHA256)) if reference=='teachers' else file_sha(model_path))
        if result.get('model_sha256')!=expected_model:raise StopTraining('evaluation_model_changed:'+key)
        aborted=result.get('aborted_for_teacher_failure') is True
        if aborted:
            if not key.startswith('teacher_admission_'):raise StopTraining('unexpected_partial_evaluation')
        else:
            expected_groups={r['group_id'] for r in cases.groups(course)}
            if result.get('complete') is not True or set(result.get('groups',{}))!=expected_groups:
                raise StopTraining('incomplete_evaluation_suite:'+key)
        return result

    def worker(self,key,course,target,resume=None,acceptance=None):
        out=self.root/'learning'/course
        command=[sys.executable,spec.ROOT/'scripts/run_a2_v10.py','worker','--output-dir',out,'--device',self.device,
                 '--num-envs',str(self.state['num_envs']),'--course',course,'--stop-round',str(target),
                 '--ledger-path',self.root/'learning_costs.jsonl','--controller-state','__CONTROLLER_STATE__']
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
        for course in ('D1','D2','D3'):
            for suite in ('A','B'):
                key=f'teacher_admission_{course}_{suite}'
                if self.state['teacher_admission'].get(key,{}).get('passed'):continue
                result=self.evaluate(key,course,suite,None,'teachers')
                if result.get('aborted_for_teacher_failure'):
                    self.state['teacher_admission'][key]=dict(passed=False,reason='teacher_capability_gap',
                        tested_case_count=result.get('tested_case_count'),planned_case_count=result.get('planned_case_count'),groups=result['groups'])
                    self.persist();raise StopTraining('teacher_capability_gap:'+key)
                comparison=compare_results(result,result)
                self.state['teacher_admission'][key]=comparison;self.persist()
                if not comparison['passed']:raise StopTraining('teacher_capability_gap:'+key)
        self.state['teacher_admitted']=True;self.persist()

    def gate(self,course,target,checkpoint,references):
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
        digest=file_sha(checkpoint)
        existing=self.state['finals'].get(name)
        if existing and existing['status']=='passed':
            if existing['candidate_sha256']!=digest:raise StopTraining('final_candidate_changed')
            return existing
        if existing:
            candidate_op=self.state['operations'].get(f'final_{name}_candidate',{})
            artifact=self.root/'evaluation'/f'final_{name}_candidate'/'result.json'
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
                        suite_sha256=candidate['suite_sha256'],candidate_result_sha256=file_sha(self.root/'evaluation'/f'final_{name}_candidate'/'result.json'))
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
        for receipt in (self.root/'evaluation').glob(f'final_{final_name}_*/result.json'):
            refs[str(receipt)]=file_sha(receipt)
        accepted='accepted_dagger' if final_name=='DAGGER_FINAL' else 'accepted_ppo'
        cert_key='dagger_certificate' if final_name=='DAGGER_FINAL' else 'ppo_certificate'
        patch=dict(finals={final_name:final},certificates={name:str(path)},**{accepted:str(checkpoint),cert_key:str(path)})
        self.commit_event('acceptance:'+name,patch,refs,certificate,path)
        return str(path)

    def _train_dagger(self):
        if self.state.get('accepted_dagger'):return self.state['accepted_dagger']
        checkpoint=self.state['learning_checkpoint']
        for course in ('D1','D2','D3'):
            self.state['phase']=course;self.persist()
            completed=self.state.get('courses',{}).get(course)
            if completed:
                checkpoint=completed['checkpoint'];continue
            checkpoint=self.worker(f'entry_{course}',course,0,checkpoint)
            promoted=False
            for target in self.state['schedule']['gates'][course]:
                checkpoint=self.worker(f'learning_{course}_{target}',course,target,checkpoint)
                grade=self.gate(course,target,checkpoint,{'teachers':None})
                if grade['passed']:
                    promoted=True;break
            if not promoted:raise StopTraining('course_budget_exhausted:'+course)
            self.state.setdefault('courses',{})[course]=dict(passed=True,checkpoint=checkpoint,sha256=file_sha(checkpoint));self.persist()
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

    def run(self):
        import fcntl
        # One controller owns GPU scheduling and all persistent transactions at a time.
        with (self.root/'controller.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise StopTraining('another_controller_is_running') from error
            try:
                self.prepare();self.teacher_admission()
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
