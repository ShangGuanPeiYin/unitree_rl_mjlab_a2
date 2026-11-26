"""Isolated full-stack resource admission with actual rollout/update/replay/save."""
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import torch
from . import spec
from .checkpoint import BudgetLedger,atomic_json,checkpoint_load,trace_memory,release_host_memory,state_fingerprint,rng_state
from .model import V10ActorCritic
from .teachers import DualTeacher
from .replay import TaskReservoir
from .dagger import DAggerLearner
from .ppo import PPOLearner
from .runtime import collect_rollout,physical_contract,save_worker,seed_all
from .identity import code_fingerprint

def sample_resources():
    import psutil
    total=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    own=0
    for line in apps.strip().splitlines():
        parts=line.split(',')
        if len(parts)==2 and parts[0].strip()==str(os.getpid()):own+=int(parts[1].strip())
    return dict(gpu_total_mib=max(int(x) for x in total.strip().splitlines()),gpu_task_mib=own,gpu_task_seen=own>0,
                host_available_bytes=int(psutil.virtual_memory().available),process_rss_bytes=psutil.Process().memory_info().rss)

class ResourceMonitor:
    def __init__(self):self.rows=[];self.stop_event=threading.Event();self.errors=[];self.memory_stages=[]
    def _run(self):
        while not self.stop_event.is_set():
            try:self.rows.append(dict(time=time.time(),**sample_resources()))
            except Exception as e:self.errors.append(repr(e))
            self.stop_event.wait(.1)
    def __enter__(self):
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();return self
    def __exit__(self,*_):self.stop_event.set();self.thread.join()
    def memory_stage(self,row):self.memory_stages.append(row)
    def summary(self):
        if not self.rows or self.errors:raise RuntimeError('resource telemetry unavailable: '+str(self.errors))
        peak_total=max(r['gpu_total_mib'] for r in self.rows);peak_own=max(r['gpu_task_mib'] for r in self.rows);host=min(r['host_available_bytes'] for r in self.rows+self.memory_stages)
        return dict(gpu_total_peak_mib=peak_total,gpu_task_peak_mib=peak_own,host_available_min_bytes=host,
            cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            process_rss_peak_bytes=max(max(r.get('process_rss_bytes',0),r.get('process_peak_rss_bytes',0)) for r in self.rows+self.memory_stages),
            resource_samples=len(self.rows),gpu_task_valid_samples=sum(r['gpu_task_seen'] for r in self.rows),passed=peak_own>0 and peak_total<=7680 and peak_own<=7168 and host>=4*1024**3)

def probe_stack(*,num_envs,stack,output_dir,device='cuda:0',global_probe_ledger=None,probe_budget_authorization=None,probe_sequence_id=None):
    """One fresh process for one N/stack, never exports a formal learner state."""
    from .env import V10Env
    if num_envs not in (1024,512) or stack not in ('dagger','ppo'):raise ValueError('invalid SDS probe')
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('probe output directory must be new or empty')
    from .probe_budget import GlobalProbeBudget,FULL_SEQUENCE_DEMAND
    if not global_probe_ledger or not probe_sequence_id:raise ValueError('global probe budget authorization required')
    ledger=GlobalProbeBudget(global_probe_ledger,authorization_path=probe_budget_authorization,scope_id=f'{probe_sequence_id}:{num_envs}:{stack}')
    authorization=ledger.assert_authorized(probe_sequence_id,required_samples=26*num_envs*48)
    output.mkdir(parents=True,exist_ok=True);seed_all();env=None
    monitor=ResourceMonitor();result=dict(num_envs=num_envs,stack=stack,passed=False,warmup_rounds=5,measurement_rounds=20,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),global_probe_ledger=str(Path(global_probe_ledger).resolve()),probe_sequence_id=probe_sequence_id,probe_budget_authorization_sha256=authorization['sha256'])
    try:
        with monitor:
            torch.cuda.reset_peak_memory_stats()
            env=V10Env(num_envs,'D3',42,device)
            physical=physical_contract(env);atomic_json(output/'physical_contract.json',physical)
            model=V10ActorCritic.from_stairs(device=device);reservoir=TaskReservoir()
            teachers=DualTeacher(model).to(device)
            atomic_json(output/'cost_receipt.json',dict(global_ledger_path=str(global_probe_ledger),scope_id=f'{probe_sequence_id}:{num_envs}:{stack}'))
            limit=26*num_envs*48
            # Real same-frame data fill every allocated reservoir slot; repeated observations
            # are admitted only for resource stress, never carried into formal training.
            trace_memory(monitor.memory_stage,'sampling_initial_begin')
            initial,_=collect_rollout(env,model,teachers,phase='D3',ledger=ledger,kind='probe',budget_limit=limit)
            while min(reservoir.seen.values())<reservoir.capacity:reservoir.add(initial)
            del initial,_
            release_host_memory();trace_memory(monitor.memory_stage,'replay_full_initial_released')
            if stack=='dagger':learner=DAggerLearner(model);keep=None;phase='D3'
            else:
                del teachers;teachers=None;gc.collect();torch.cuda.empty_cache()
                keep=model.initialize_ppo();model.set_phase('P1');learner=PPOLearner(model);phase='P1'
            durations=[]
            for iteration in range(25):
                torch.cuda.synchronize();start=time.monotonic()
                trace_memory(monitor.memory_stage,f'sampling_{iteration}_begin')
                rollout,rollout_metrics=collect_rollout(env,model,teachers if stack=='dagger' else None,phase=phase,ledger=ledger,kind='probe',budget_limit=limit)
                trace_memory(monitor.memory_stage,f'sampling_{iteration}_complete')
                metrics=(learner.update(rollout,reservoir,num_envs) if stack=='dagger' else learner.update(rollout,keep,reservoir,'P1'))
                ledger.record_update(f'{probe_sequence_id}:{num_envs}:{stack}:update:{iteration}', 'D3' if stack=='dagger' else 'PPO',num_envs*48,kind='probe')
                if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):raise FloatingPointError('nonfinite probe update')
                trace_memory(monitor.memory_stage,f'update_{iteration}_complete')
                del rollout,rollout_metrics
                release_host_memory();trace_memory(monitor.memory_stage,f'rollout_{iteration}_released')
                torch.cuda.synchronize();durations.append(time.monotonic()-start)
                if not monitor.summary()['passed']:raise RuntimeError('resource ceiling exceeded during full update')
                atomic_json(output/'progress.json',dict(iteration=iteration+1,measurement=iteration>=5,metrics=metrics))
            checkpoint=output/'probe_restore.pt'
            expected_replay=reservoir.integrity_manifest();expected_next=reservoir.next_sample_manifest()
            learner_state=learner.state_dict() if stack=='ppo' else dict(optimizer=learner.optimizer.state_dict())
            expected_model=state_fingerprint(model.state_dict());expected_learner=state_fingerprint(learner_state)
            expected_rng=state_fingerprint(rng_state());del learner_state
            digest=save_worker(checkpoint,model,learner,reservoir,env,ledger,{'probe':25},physical['sha256'],keep,memory_trace=monitor.memory_stage)
            # save_worker performs complete disk SHA + ZIP CRC validation before release.
            reservoir.release_storage();release_host_memory()
            trace_memory(monitor.memory_stage,'old_replay_released_before_load')
            restored=checkpoint_load(checkpoint,num_envs=num_envs,physical_sha256=physical['sha256'],memory_trace=monitor.memory_stage)
            if state_fingerprint(restored['rng'])!=expected_rng:raise RuntimeError('checkpoint RNG mismatch')
            model.load_state_dict(restored['model']);reservoir.take_state_dict(restored.pop('reservoir'))
            if stack=='dagger':learner.optimizer.load_state_dict(restored['learner']['optimizer'])
            else:learner.load_state_dict(restored['learner'])
            env.load_state_dict(restored['env'])
            if state_fingerprint(model.state_dict())!=expected_model:raise RuntimeError('restored model mismatch')
            actual_learner=learner.state_dict() if stack=='ppo' else dict(optimizer=learner.optimizer.state_dict())
            if state_fingerprint(actual_learner)!=expected_learner:raise RuntimeError('restored Adam/RNG mismatch')
            del actual_learner
            if reservoir.integrity_manifest()!=expected_replay or reservoir.next_sample_manifest()!=expected_next:raise RuntimeError('restored replay/RNG/next-sample mismatch')
            for task in reservoir.seen:
                if reservoir.seen[task]<reservoir.capacity:raise RuntimeError('probe replay not filled')
            restored.clear();del restored;release_host_memory()
            trace_memory(monitor.memory_stage,'restored_state_transferred_and_released')
            result.update(measured_seconds=durations[5:],cost=ledger.report(),physical_sha256=physical['sha256'],
                restore_checked=True,replay_counts=dict(reservoir.seen),checkpoint_sha256=digest,
                restored_replay_manifest=expected_replay,next_sample_manifest=expected_next)
            # Unlink only after a successful real restore and equality checks.
            checkpoint.unlink()
        summary=monitor.summary();result.update(summary)
    except Exception as error:
        ledger.reconcile_uncertain()
        result.update(error=repr(error),passed=False)
        if monitor.rows:
            result['sampled_resource_peak']=dict(gpu_total_mib=max(r['gpu_total_mib'] for r in monitor.rows),gpu_task_mib=max(r['gpu_task_mib'] for r in monitor.rows),host_available_bytes=min(r['host_available_bytes'] for r in monitor.rows))
        result['cost']=ledger.report()
    finally:
        if env is not None:env.close()
        result['global_probe_totals']=ledger.global_report()['global_totals']
        result['memory_stages']=monitor.memory_stages
        atomic_json(output/'result.json',result)
        with (output/'resources.jsonl').open('w') as f:
            for row in monitor.rows:f.write(json.dumps(row)+'\n')
    return result

def run_probes(output_dir,device='cuda:0',global_probe_ledger=None,probe_budget_authorization=None,probe_sequence_id=None):
    root=Path(output_dir)
    if root.exists() and any(root.iterdir()):raise ValueError('probe root must be new or empty')
    from .probe_budget import GlobalProbeBudget,FULL_SEQUENCE_DEMAND
    if not global_probe_ledger or not probe_sequence_id:raise ValueError('global probe budget authorization required')
    budget=GlobalProbeBudget(global_probe_ledger,authorization_path=probe_budget_authorization)
    budget.begin_sequence(probe_sequence_id,required_samples=FULL_SEQUENCE_DEMAND)
    try:return _run_authorized_probes(root,device,global_probe_ledger,probe_budget_authorization,probe_sequence_id)
    finally:budget.finish_sequence(probe_sequence_id,passed=(root/'selection.json').exists() and json.loads((root/'selection.json').read_text()).get('passed',False),reason='sequence executor finished')

def _run_authorized_probes(root,device,global_probe_ledger,probe_budget_authorization,probe_sequence_id):
    from .probe_budget import GlobalProbeBudget,FULL_SEQUENCE_DEMAND
    budget=GlobalProbeBudget(global_probe_ledger,authorization_path=probe_budget_authorization)
    authorization=budget.assert_authorized(probe_sequence_id)
    receipt=dict(global_probe_ledger=str(Path(global_probe_ledger).resolve()),probe_sequence_id=probe_sequence_id,probe_budget_authorization_sha256=authorization['sha256'])
    root.mkdir(parents=True,exist_ok=True);runs=[]
    for n in (1024,512):
        passed=True
        for stack in ('dagger','ppo'):
            output=root/f'n{n}_{stack}'
            command=[sys.executable,str(spec.ROOT/'scripts/run_a2_v10.py'),'_probe-stack','--num-envs',str(n),'--stack',stack,'--device',device,'--output-dir',str(output),'--global-probe-ledger',str(global_probe_ledger),'--probe-budget-authorization',str(probe_budget_authorization),'--probe-sequence-id',probe_sequence_id]
            with (root/f'n{n}_{stack}.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=spec.ROOT)
            result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else dict(passed=False,error=f'child exit {completed.returncode}')
            runs.append(result)
            if (not result['passed'] or result.get('sds_sha256')!=spec.SDS_SHA256 or result.get('code_sha256')!=code_fingerprint()):passed=False;break
        if passed:
            result=dict(passed=True,num_envs=n,runs=runs,probe_state_discarded=True,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),global_probe_totals=budget.global_report()['global_totals'],**receipt);atomic_json(root/'selection.json',result);return result
    result=dict(passed=False,num_envs=None,runs=runs,sds_sha256=spec.SDS_SHA256,code_sha256=code_fingerprint(),global_probe_totals=budget.global_report()['global_totals'],**receipt);atomic_json(root/'selection.json',result);return result
