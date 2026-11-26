"""GPU-isolated fixed-suite execution; first-terminal freeze and explicit padding."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from . import cases,spec
from .checkpoint import BudgetLedger,atomic_json,sha256,checkpoint_validate
from .evaluation import EpisodeMetrics,grade_group
from .identity import code_fingerprint


class BatchAccounting:
    def __init__(self,max_steps):
        if not max_steps or len(set(max_steps))!=1:raise ValueError('batch must have identical case bounds')
        self.n=len(max_steps);self.upper_bound=sum(max_steps)
        if self.upper_bound!=self.n*max(max_steps):raise ValueError('batch bound differs from case sum')
        self.active=set(range(self.n));self.first_terminals={}
        self.total=self.effective=self.padding=self.steps=0
    def step(self,terminals):
        self.steps+=1;self.total+=self.n;self.effective+=len(self.active);self.padding+=self.n-len(self.active)
        for i in terminals:
            if i in self.active:self.first_terminals[i]=self.steps;self.active.remove(i)
        if self.total>self.upper_bound:raise RuntimeError('evaluation batch exceeded frozen bound')


def _frame_rows(frame,n):
    fields=('command','position','v_body','v_world','roll_deg','pitch_deg','clearance','safe_support','stable','top_event','failure','timeout')
    values={k:frame[k].detach().cpu().numpy() for k in fields}
    return [dict({k:values[k][i] for k in fields},phase=frame['phase'][i]) for i in range(n)]


def teacher_group_failed(results):
    if len(results)!=20:raise ValueError('admission group must contain exactly 20 cases')
    return sum(bool(r['passed']) for r in results)<18


def run_suite(*,course,suite_name,replicas,model_path,reference,output_dir,ledger_path,eval_limit,device='cuda:0',stop_on_failed_group=False):
    """One bounded suite. Teachers use their task routing; candidates use only the student."""
    from .env import V10Env
    from .model import V10ActorCritic
    from .teachers import DualTeacher
    from .runtime import physical_contract
    if stop_on_failed_group and (reference!='teachers' or suite_name not in ('A','B') or replicas!=20):
        raise ValueError('early stopping only applies to teacher admission A/B')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=cases.suite(course,suite_name,replicas)
    suite_sha=cases.digest(manifest)
    atomic_json(out/'cases.json',dict(sds_sha256=spec.SDS_SHA256,sha256=suite_sha,cases=manifest))
    code_sha=code_fingerprint();rulings_sha=sha256(spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md')
    model_sha=(cases.digest({'blind':spec.BLIND_SHA256,'stairs':spec.STAIRS_SHA256}) if reference=='teachers' else sha256(model_path))
    model=V10ActorCritic.from_stairs(device=device)
    state=None
    if reference!='teachers':
        state=torch.load(model_path,map_location='cpu',weights_only=False)
        if state.get('sds_sha256')!=spec.SDS_SHA256:raise ValueError('evaluation source SDS mismatch')
        model.load_state_dict(state['model'],strict=True)
        # Evaluation never retains optimizer/replay or deserializes GBs again per group.
        state={k:state[k] for k in ('schema','sds_sha256','recipe','code_sha256','teachers','num_envs','physical_sha256')}
    model.eval().requires_grad_(False)
    teachers=DualTeacher(model).to(device) if reference=='teachers' else None
    ledger=BudgetLedger(ledger_path);ledger.reconcile_uncertain()
    uid=uuid.uuid4().hex;results=[];executed=effective=padding=0
    before=ledger.report();physical_sha=None;new_caches=[];aborted=False
    # Group manifests are contiguous and each contains precisely 20/100 actual cases.
    group_results={};start=time.monotonic()
    for offset in range(0,len(manifest),replicas):
        batch=manifest[offset:offset+replicas];account=BatchAccounting([c['max_steps'] for c in batch])
        env=V10Env(num_envs=len(batch),course=course,seed=spec.SEED,device=device,cases=batch,auto_reset=False)
        try:
            contract=physical_contract(env)
            if physical_sha is None:
                physical_sha=contract['sha256'];atomic_json(out/'physical_contract.json',contract)
            if physical_sha!=contract['sha256']:raise RuntimeError('physics changed within suite')
            if state is not None:
                checkpoint_validate(state,num_envs=state['num_envs'],physical_sha256=physical_sha)
            key=cases.digest(dict(model=model_sha,cases=cases.digest(batch),protocol=spec.SDS_SHA256,code=code_sha,rulings=rulings_sha,physics=physical_sha))
            cache=out.parent/'cache'/f'{key}.json'
            if cache.exists():
                cached=json.loads(cache.read_text())
                if cached.get('key')!=key:raise RuntimeError('damaged reference cache')
                report=Path(cached.get('resource_report_path','__missing__'))
                if (report.is_file() and sha256(report)==cached.get('resource_report_sha256')
                    and json.loads(report.read_text()).get('passed') is True):
                    results+=cached['results'];group_results[batch[0]['group_id']]=cached['results']
                    if stop_on_failed_group and teacher_group_failed(cached['results']):
                        aborted=True;break
                    continue
            metrics=[EpisodeMetrics(c) for c in batch];first={}
            obs=env.observe()
            while account.active:
                active=set(account.active)
                with torch.no_grad():
                    if teachers is not None:
                        action=teachers.label(obs,[t=='stairs' for t in env.metadata()['teacher']])
                    else:action=model(obs)
                    for i in range(len(batch)):
                        if i not in active:action[i]=0.
                step_uid=f'eval:{uid}:{offset}:{account.steps}'
                if not ledger.reserve(step_uid,'ALL',len(batch),limit=eval_limit,kind='evaluation',details=dict(suite=suite_sha,group=batch[0]['group_id'])):
                    raise RuntimeError('refusing replay of reserved evaluation step')
                obs,_,done,info=env.step(action)
                ledger.mark_executed(step_uid)
                rows=_frame_rows(info['frame'],len(batch))
                for i in active:metrics[i].update(rows[i])
                terminal=[int(ep['env_id']) for ep in info['episodes'] if int(ep['env_id']) in active]
                for i in terminal:
                    # Verify actual trace endpoints; do not trust environment success boolean.
                    result=metrics[i].finish(completed=True)
                    first[i]=result
                    np.savez_compressed(out/f'trace_{batch[i]["case_sha256"]}.npz',rows=metrics[i].rows[:metrics[i].count])
                account.step(terminal)
            ordered=[first[i] for i in range(len(batch))]
            group_results[batch[0]['group_id']]=ordered;results+=ordered
            executed+=account.total;effective+=account.effective;padding+=account.padding
            if reference in ('teachers','dagger'):
                atomic_json(cache,dict(key=key,results=ordered))
                new_caches.append(str(cache))
            atomic_json(out/'progress.json',dict(completed_groups=len(group_results),physical_samples=executed,effective_samples=effective,padding_samples=padding))
            if stop_on_failed_group and teacher_group_failed(ordered):
                aborted=True;break
        finally:env.close()
    result=dict(sds_sha256=spec.SDS_SHA256,code_sha256=code_sha,rulings_sha256=rulings_sha,suite_sha256=suite_sha,model_sha256=model_sha,
                physical_sha256=physical_sha,course=course,suite=suite_name,replicas=replicas,reference=reference,
                results=results,groups=group_results,aborted_for_teacher_failure=aborted,complete=not aborted,
                tested_case_count=len(results),planned_case_count=len(manifest),physical_samples=executed,effective_samples=effective,padding_samples=padding,
                ledger_before=before,ledger_after=ledger.report(),elapsed_seconds=time.monotonic()-start,_new_caches=new_caches)
    atomic_json(out/'result_unadmitted.json',result)
    return result


def compare_suites(candidate,reference,*,final):
    if any(candidate[k]!=reference[k] for k in ('sds_sha256','suite_sha256','physical_sha256')):
        raise ValueError('paired evaluation identity mismatch')
    if set(candidate['groups'])!=set(reference['groups']):raise ValueError('paired group mismatch')
    groups={k:grade_group(candidate['groups'][k],reference['groups'][k],final=final) for k in candidate['groups']}
    return dict(passed=all(g['passed'] for g in groups.values()),groups=groups,
                candidate_model_sha256=candidate['model_sha256'],reference_model_sha256=reference['model_sha256'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--course',choices=('D1','D2','D3','PPO'),required=True)
    p.add_argument('--suite',choices=('A','B','DAGGER_FINAL','PPO_FINAL'),required=True)
    p.add_argument('--replicas',type=int,choices=(20,100),required=True)
    p.add_argument('--model-path',default=None)
    p.add_argument('--stop-on-failed-group',action='store_true')
    p.add_argument('--reference',choices=('candidate','teachers','dagger'),required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--ledger',required=True)
    p.add_argument('--eval-limit',type=int,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    from .resources import ResourceMonitor
    from .gpu_lock import gpu_execution_lock
    with gpu_execution_lock():
        with ResourceMonitor() as monitor:
            torch.cuda.reset_peak_memory_stats()
            result=run_suite(course=a.course,suite_name=a.suite,replicas=a.replicas,model_path=a.model_path,reference=a.reference,
                      output_dir=a.output_dir,ledger_path=a.ledger,eval_limit=a.eval_limit,device=a.device,stop_on_failed_group=a.stop_on_failed_group)
            torch.cuda.synchronize()
        summary=monitor.summary()
        report=Path(a.output_dir).resolve()/'resource_report.json'
        atomic_json(report,summary)
        if not summary['passed']:raise RuntimeError('evaluation GPU/host admission failed')
        resource=dict(resource_report_path=str(report),resource_report_sha256=sha256(report),resource_admitted=True)
        for cache_path in result.pop('_new_caches',[]):
            cache=Path(cache_path);entry=json.loads(cache.read_text());entry.update(resource);atomic_json(cache,entry)
        result.update(resource)
        # Only this immutable success receipt may be adopted after a controller crash.
        atomic_json(Path(a.output_dir)/'result.json',result)
        torch.cuda.empty_cache()

if __name__=='__main__':main()
