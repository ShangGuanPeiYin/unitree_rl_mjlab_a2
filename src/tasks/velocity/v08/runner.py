"""Native v08 PPO lifecycle, cost ledger, real case evaluation and recovery."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
import torch
from . import spec,recovery
from .continuation_v08 import tree_digest,SOURCE_SHA256
from .ledger import CostLedger
from .telemetry import V08Telemetry
from .operator_control import ControlProtocol

def boundary_plan(mode,global_update,local_update):
    if mode=='probe':
        return ('diagnostic',) if local_update==50 else ('save_reload',) if local_update==55 else ()
    out=[]
    if global_update%50==0:out.append('diagnostic')
    if global_update%100==0:out.append('gate')
    if global_update%200==0:out.append('h00')
    if global_update%100==0:out.append('save')
    return tuple(out)

def probe_accounting(durations,*,save_update,reload_update,completed_update):
    if len(durations)!=56 or (save_update,reload_update,completed_update)!=(6455,6455,6456):
        raise ValueError('probe requires56 completed updates with real save/reload55')
    if any(not isinstance(x,(float,int)) or not __import__('math').isfinite(x) or x<0 for x in durations):
        raise ValueError('invalid probe measurement')
    measured=sum(durations[5:55])
    return dict(warmup_updates=5,measurement_updates=50,recovery_updates=1,
        warmup_seconds=sum(durations[:5]),measured_seconds=measured,recovery_seconds=durations[55],
        measured_control_samples=50*512*48,measured_physics_substeps=50*512*48*4,
        measured_samples_per_second=50*512*48/measured if measured>0 else None,
        save_global_update=6455,reload_global_update=6455,post_reload_global_update=6456)

class OperatorInterrupted(RuntimeError):
    pass

class V08Runner:
    def __init__(self,model,ppo,curriculum,env,directory,seed=42,device='cpu',mode='formal',
                 case_runner=None,telemetry=None,**kwargs):
        if kwargs:raise ValueError('unsupported runner arguments: '+','.join(kwargs))
        if seed!=42:raise ValueError('unique training seed42 required')
        self.model,self.ppo,self.curriculum,self.env=model,ppo,curriculum,env
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.seed,self.device,self.mode=seed,str(device),mode
        self.ledger=CostLedger(mode=mode,num_envs=env.num_envs)
        self.telemetry=telemetry or V08Telemetry(self.directory/'telemetry')
        self.operator_control=ControlProtocol(self.directory/'control.json',self.directory/'control_evidence')
        self.case_runner=case_runner;self.source=None;self.freeze=None;self.case_tables=None
        self.evaluation={};self.initialization={};self.last_diagnostics=None;self.rng_receipt=None
        self.saved_task_provenance=None;self.update_durations=[];self.recent_episodes=[];self._last_monitor_poll=0.
        self.ppo.control_step_callback=self._control_step
    @property
    def update(self):return 6400+self.ledger.updates
    @property
    def learning(self):
        lr=self.ppo.optimizer.param_groups[0]['lr']
        return dict(update=self.update,optimizer_step=self.ppo.learning_state['optimizer_steps'],
            adam_steps=self.ppo.learning_state['optimizer_steps'],control_samples=self.ledger.learning_samples,
            physics_substeps=self.ledger.learning_samples*4,num_envs=self.env.num_envs,rollout=48,epochs=5,minibatches=4,learning_rate=lr)
    def _load_learning(self,state):
        self.model.load_state_dict(state['model'],strict=True)
        self.ppo.optimizer.load_state_dict(state['optimizer'])
        self.ppo.load_learning_counters(state['learning'])
        self.ppo.storage=None
        if tree_digest(recovery.normalizer_state(self.model))!=tree_digest(state['normalizer']):
            raise recovery.RecoveryBlocked('loaded model normalizers differ')
        if tree_digest(self.ppo.optimizer.state_dict())!=tree_digest(state['optimizer']):
            raise recovery.RecoveryBlocked('loaded optimizer differs')
    def initialize(self,state):
        if state['learning']['update']!=6400:raise recovery.RecoveryBlocked('migration must start6400')
        self._load_learning(state)
        self.source=deepcopy(state['source']);self.initialization=deepcopy(state['initialization'])
        self.freeze=recovery.build_freeze();self.case_tables=recovery.freeze_case_tables()
        self.rng_receipt=recovery.restore_rng(state['rng'],self.env,device=self.device,
            allow_cpu_cuda_provenance=self.mode!='formal' and self.device=='cpu')
        self.rng_receipt['new_streams']=dict(env_task_sampler='SeedSequence([42,8801]), provisional constructor draws discarded',
            curriculum_sampler='independent instance SeedSequence([42,8801]); schedule currently deterministic, no draws',
            evaluation='sha256 namespaced case seeds, separate environments and restored training global RNG')
        # A new v08 deterministic schedule and exactly one post-restore reset.
        if hasattr(self.env,'reset_task_sampler'):self.env.reset_task_sampler(seed=42)
        self.env.set_schedule(self.curriculum.schedule(),stage_iteration=0)
        self.ppo.reset()
        self.start_telemetry()
    def start_telemetry(self):
        rng=recovery.capture_rng(self.env)
        try:self.telemetry.start(self.directory.name,dict(mode=self.mode,learning=self.learning,rng=self.rng_receipt,freeze_sha256=self.freeze['sha256']))
        finally:recovery.restore_rng(rng,self.env,device=self.device,allow_cpu_cuda_provenance=self.mode!='formal' and self.device=='cpu')
    def checkpoint_state(self):
        return dict(schema=10,identity=dict(task=spec.TASK_ID,recipe=spec.RECIPE_VERSION),
            model=deepcopy(self.model.state_dict()),optimizer=deepcopy(self.ppo.optimizer.state_dict()),
            normalizer=recovery.normalizer_state(self.model),learning=self.learning,
            rng=recovery.capture_rng(self.env),curriculum=self.curriculum.state_dict(),ledger=self.ledger.state_dict(),
            source=deepcopy(self.source),initialization=deepcopy(self.initialization),
            environment=deepcopy(self.env.state_dict()),saved_task_provenance=deepcopy(self.saved_task_provenance),
            freeze=deepcopy(self.freeze),case_tables=deepcopy(self.case_tables),mode=self.mode,recipe=spec.contract(),
            resume_mode='fresh_physical_episode',evaluation=deepcopy(self.evaluation),
            last_diagnostics=deepcopy(self.last_diagnostics),rng_receipt=deepcopy(self.rng_receipt),
            update_durations=list(self.update_durations),recent_episodes=deepcopy(self.recent_episodes))
    def save_verified(self,reason):
        recovery.verify_freeze(self.freeze)
        seq=self.ledger.state.get('save_sequence',0)+1;self.ledger.state['save_sequence']=seq
        state=self.checkpoint_state();recovery.validate_checkpoint(state,mode=self.mode)
        path=self.directory/'checkpoints'/f'{reason}_{self.update:06d}_{seq:04d}.pt'
        if path.exists():raise recovery.RecoveryBlocked('checkpoint path already exists')
        sha=recovery.atomic_save(path,state)
        loaded=recovery.load_verified(path,sha);recovery.validate_checkpoint(loaded,mode=self.mode)
        if tree_digest(loaded)!=tree_digest(state):raise recovery.RecoveryBlocked('save/reload state differs')
        recovery.write_json(path.with_suffix('.json'),dict(path=str(path),sha256=sha,reason=reason,
            update=self.update,sequence=seq,mode=self.mode,freeze_sha256=self.freeze['sha256']))
        return path,sha
    def load(self,path,sha):
        state=recovery.load_verified(path,sha);recovery.validate_checkpoint(state,mode=self.mode)
        recovery.verify_learning_journal(self.directory,state['learning']['update'])
        ledger=state['ledger']
        if ledger.get('pending_update') is not None:raise recovery.RecoveryBlocked('incomplete PPO cannot be replayed')
        if ledger.get('stop_state') is not None:raise recovery.RecoveryBlocked('terminal/operator/error checkpoint cannot auto-resume')
        self._load_learning(state);self.curriculum.load_state_dict(state['curriculum'])
        self.ledger=CostLedger(mode=self.mode,num_envs=self.env.num_envs,state=ledger)
        self.source=deepcopy(state['source']);self.initialization=deepcopy(state['initialization'])
        self.freeze=deepcopy(state['freeze']);self.case_tables=deepcopy(state['case_tables'])
        self.evaluation=deepcopy(state['evaluation']);self.last_diagnostics=deepcopy(state['last_diagnostics'])
        self.saved_task_provenance=deepcopy(state['environment'])
        self.update_durations=list(state.get('update_durations',[]));self.recent_episodes=deepcopy(state.get('recent_episodes',[]))
        self.rng_receipt=recovery.restore_rng(state['rng'],self.env,device=self.device,
            allow_cpu_cuda_provenance=self.mode!='formal' and self.device=='cpu')
        # Env owns one fresh physical reset; full logical snapshot is retained as provenance.
        self.env.load_state_dict(state['environment'],restore_physics=False)
        self.ppo.obs=self.env.get_observations()
        return dict(update=self.update,physical_trajectory_continuous=False,
                    recovery_mode='fresh_physical_episode',rng=self.rng_receipt)
    def _control_step(self,payload):
        # Read the atomic request at every real control boundary; the environment
        # latches its requested speed for the *next* actor observation.
        if self._poll_control():raise OperatorInterrupted('operator request observed at control boundary')
        if hasattr(self.telemetry,'heartbeat') and time.monotonic()-self._last_monitor_poll>=2.:
            self._last_monitor_poll=time.monotonic()
            self.telemetry.heartbeat(dict(global_iteration=self.update))
            if self.telemetry.state.get('monitoring_blocked'):
                from .connection import MonitoringBlocked
                raise MonitoringBlocked('remote delivery exceeded120s contract')
    def _poll_control(self):
        control=self.operator_control.poll()
        if control is None:return False
        if control['action']=='request_speed':
            ids=control.get('env_ids')
            if ids is None:ids=list(range(self.env.num_envs))
            if any(i>=self.env.num_envs for i in ids):raise ValueError('control env id outside batch')
            self.env.request_speed(ids,float(control['speed']))
            self.ppo.obs=self.env.get_observations()
            self.ledger.state['events'].append(dict(event='request_speed',update=self.update,request=control))
            return False
        self.ledger.state.update(stop_state='operator_pause' if control['action']=='pause' else 'user_requested_stop')
        return True
    def _evaluate(self,kind,phase=None):
        from .evaluation import table_from_json,evaluate_cases
        from .evaluation_records import pack_result,write_sidecar,restore_result
        phase=self.curriculum.phase if phase is None else phase
        frozen=self.case_tables[f'{kind}:{phase}'];table=table_from_json(frozen['table'],frozen['sha256'])
        key=f'{kind}:{self.update}'
        identity=dict(update=self.update,phase=table.phase,kind=kind,case_table_sha256=table.sha256,
            source_sha256=SOURCE_SHA256,recipe_sha256=recovery.digest(recovery.jsonable(spec.contract())),
            freeze_sha256=self.freeze['sha256'],model_state_sha256=tree_digest(self.model.state_dict()),
            optimizer_state_sha256=tree_digest(self.ppo.optimizer.state_dict()))
        sidecar=self.directory/'evaluations'/f'{kind}_{table.phase}_{self.update:06d}.json'
        if key in self.evaluation:
            reference=self.evaluation[key]
            if reference['identity']!=identity:raise RuntimeError('cached evaluation input identity differs')
            result=recovery.load_evaluation_reference(reference)
        else:
            if sidecar.exists():
                # Crash after durable evidence but before boundary commit: use
                # actual evidence, never rerun this already billed evaluation.
                payload=__import__('json').loads(sidecar.read_text())
                result=restore_result(payload,identity)
                cost=recovery.evaluation_control_samples(result)
                self.ledger.state['evaluation_control_samples']+=cost
            else:
                if self.case_runner is None:raise RuntimeError('real policy/physics case runner missing')
                self.case_runner.phase=table.phase
                before=getattr(self.case_runner,'control_samples',0)
                try:
                    result=evaluate_cases(self.model,table,self.case_runner,optimizer=self.ppo.optimizer,
                        counters=self.ppo.learning_state,device=self.device)
                finally:
                    cost=getattr(self.case_runner,'control_samples',0)-before
                    self.ledger.state['evaluation_control_samples']+=cost
                payload=pack_result(result,identity)
                if not result.summary.get('engineering_error') and recovery.evaluation_control_samples(result)!=cost:
                    raise RuntimeError('collector actual cost differs from raw batch evidence')
            binding=write_sidecar(sidecar,payload)
            # Full raw trajectories live exactly once in immutable sidecars.
            # Native checkpoints retain SHA references, summary and transaction.
            self.evaluation[key]=dict(binding=binding,identity=identity,summary=deepcopy(result.summary),
                control_samples=cost,raw_bytes=sidecar.stat().st_size)
        self.telemetry.gate(result.summary)
        if result.summary.get('engineering_error'):raise RuntimeError('evaluation engineering error; full denominator retained')
        if kind=='gate':self.curriculum.record_gate(result.summary,update=self.update)
        return result
    def _complete_boundary(self):
        pending=self.ledger.state.get('pending_boundary')
        if not pending:return
        while pending['index']<len(pending['events']):
            event=pending['events'][pending['index']]
            if event=='diagnostic':
                self.diagnostic()
                self.ledger.state['events'].append(dict(event=event,update=self.update))
            elif event in ('gate','h00'):self._evaluate(event,phase=pending.get('phase'))
            elif event=='save':self.save_verified('interval')
            elif event=='save_reload':
                # Mark completed before snapshot; restored56 must not save55 again.
                pending['index']+=1
                if pending['index']==len(pending['events']):self.ledger.state['pending_boundary']=None
                path,sha=self.save_verified('probe55')
                self.load(path,sha)
                self.ledger.state['events'].append(dict(event='reload55',update=self.update,path=str(path),sha256=sha))
                return
            pending['index']+=1
            # Persist completed event index and bindings before any next event.
            self.save_verified('boundary_event')
        self.ledger.state['pending_boundary']=None
    def step(self):
        if self.ledger.state['stop_state'] is not None:return False
        self._complete_boundary()
        if self.curriculum.finished:return False
        if self._poll_control():self.save_verified('operator');return False
        recovery.verify_freeze(self.freeze)
        next_update=self.update+1
        intent=self.directory/'transactions'/f'update_{next_update:06d}.intent.json'
        if intent.exists():raise recovery.RecoveryBlocked('PPO intent already exists; no duplicate learning')
        recovery.write_json(intent,dict(update=next_update,mode=self.mode,freeze_sha256=self.freeze['sha256'],
            before=self.learning,planned_control_samples=48*self.env.num_envs))
        self.ledger.state['pending_update']=dict(update=next_update,status='PPO_started')
        try:
            records,metrics=self.ppo.train_step()
            if self.ppo.learning_state['completed_ppo_updates']!=next_update:raise ValueError('PPO update counter differs')
            self.ledger.commit(next_update);self.curriculum.on_update(next_update)
            self.ledger.state['pending_update']=None
            recovery.write_json(intent.with_name(intent.name.replace('.intent.json','.commit.json')),
                dict(update=self.update,mode=self.mode,learning=self.learning,cost=self.ledger.state_dict(),freeze_sha256=self.freeze['sha256']))
            self.last_diagnostics=deepcopy(self.ppo.last_update_diagnostics)
            self.recent_episodes.extend(deepcopy(records))
            self.telemetry.update(dict(learning=self.learning,metrics=metrics))
            events=boundary_plan(self.mode,self.update,self.ledger.updates)
            if events:
                self.ledger.state['pending_boundary']=dict(update=self.update,phase=self.curriculum.phase,events=list(events),index=0)
                if events==('diagnostic',):
                    self.diagnostic();self.ledger.state['events'].append(dict(event='diagnostic',update=self.update))
                    self.ledger.state['pending_boundary']=None
                else:
                    self.save_verified('boundary')
                    self._complete_boundary()
            # Complete gate at the cap before deciding whether the phase ended.
            if not self.curriculum.finished:
                self.env.set_schedule(self.curriculum.schedule(),stage_iteration=self.curriculum.state.generation_updates)
                self.ppo.obs=self.env.get_observations()
            if self.curriculum.finished:
                self.ledger.state['stop_state']='curriculum_gates_passed' if self.curriculum.stop_status=='complete' else self.curriculum.stop_status
                self.save_verified('terminal')
            return True
        except OperatorInterrupted:
            self.ledger.state['pending_update']=None
            self.ledger.state['discarded_rollout_control_samples']=self.ledger.state.get('discarded_rollout_control_samples',0)+self.ppo.pending_control_samples
            recovery.write_json(intent.with_name(intent.name.replace('.intent.json','.aborted.json')),
                dict(update=next_update,adam_unchanged=True,reason=self.ledger.state['stop_state'],
                    discarded_rollout_control_samples=self.ppo.pending_control_samples))
            self.save_verified('operator')
            return False
        except BaseException as error:
            self.ledger.state['partial_physics_cost_unknown']=True
            self.ledger.state.update(stop_state='monitoring_blocked' if type(error).__name__=='MonitoringBlocked' else 'engineering_error',error_type=type(error).__name__)
            self.ledger.state['error']=str(error)
            # Preserve partial state honestly; invalid partial Adam cannot pass
            # native validation and receives a separate unvalidated audit file.
            path=self.directory/'exceptions'/f'exception_{next_update:06d}_{time.time_ns()}.pt'
            state=self.checkpoint_state();sha=recovery.atomic_save(path,state)
            recovery.write_json(path.with_suffix('.json'),dict(sha256=sha,validated=False,error_type=type(error).__name__,error=str(error)))
            raise
    def diagnostic(self):
        if not isinstance(self.last_diagnostics,dict) or self.last_diagnostics.get('update')!=self.update:
            raise RuntimeError('real current PPO diagnostics missing')
        rows=self.recent_episodes
        data=dict(update=self.update,ppo=self.last_diagnostics,episodes=len(rows),
            successes=sum(x.get('success') is True for x in rows),
            reasons=dict(__import__('collections').Counter(str(x.get('reason','missing')) for x in rows)),
            raw_episodes=rows,learning=self.learning)
        recovery.write_json(self.directory/'diagnostics'/f'diagnostic_{self.update:06d}.json',data)
        self.telemetry.emit('diagnostic',data)
        self.recent_episodes=[]
        return data
    def run(self,updates):
        for _ in range(int(updates)):
            started=time.monotonic()
            if not self.step():break
            self.update_durations.append(time.monotonic()-started)
        probe=None
        if self.mode=='probe' and self.ledger.updates==56:
            reloaded=[x for x in self.ledger.state['events'] if x.get('event')=='reload55']
            if len(reloaded)!=1:raise RuntimeError('probe requires exactly one real reload55')
            event=reloaded[0]
            if recovery.sha256(event['path'])!=event['sha256']:raise RuntimeError('saved probe55 SHA differs')
            probe=probe_accounting(self.update_durations,save_update=event['update'],
                reload_update=event['update'],completed_update=self.update)
        if self.mode=='probe' and probe is None:
            probe=dict(complete=False,completed_updates=self.ledger.updates,reason=self.ledger.state['stop_state'])
        path,sha=self.save_verified('exit')
        receipt=dict(mode=self.mode,learning=self.learning,cost=self.ledger.state_dict(),checkpoint=str(path),checkpoint_sha256=sha,
                     phase=self.curriculum.phase,stop_status=self.ledger.state['stop_state'] or self.curriculum.stop_status,
                     final_acceptance_complete=False,probe=probe)
        receipt['total_new_control_samples_spent']=self.ledger.state['control_samples']+self.ledger.state['evaluation_control_samples']+self.ledger.state.get('discarded_rollout_control_samples',0)
        receipt['history_report_path']=str(self.directory/f'run_result_{self.update:06d}_{self.ledger.state["save_sequence"]:04d}.json')
        recovery.write_json(receipt['history_report_path'],receipt)
        recovery.write_json(self.directory/'run_result.json',receipt)
        self.telemetry.finish(receipt)
        return receipt

def environment_case_runner(model,env_factory,device='cpu',max_steps=50000):
    """Real policy/physics first-episode collection; extra async steps are billed."""
    def run(cases):
        if isinstance(cases,dict):batch=[cases];single=True
        else:batch=list(cases);single=False
        if not 1<=len(batch)<=8:raise ValueError('evaluation requires1..8 environments')
        env=env_factory(batch);rows={};start_samples=run.control_samples
        try:
            observations=env.get_observations()
            with torch.no_grad():
                for _ in range(max_steps):
                    actions=model.act(observations,deterministic=True)
                    observations,reward,terminated,truncated,info=env.step(actions)
                    run.control_samples+=len(batch)
                    for row in info.get('episodes',[]):
                        i=row.get('env_id')
                        if type(i) is not int or not 0<=i<len(batch):raise ValueError('episode env_id missing/invalid')
                        if i not in rows:
                            if row.get('case_id')!=batch[i]['case_id']:raise ValueError('episode case identity differs')
                            rows[i]=deepcopy(row)
                    if len(rows)==len(batch):break
            for i in range(len(batch)):
                if i not in rows:rows[i]=dict(case_id=batch[i]['case_id'],complete=False,raw_complete=False,finite=True,
                    success=False,reason='collector_timeout',segment='engineering')
            batch_cost=run.control_samples-start_samples
            batch_id=recovery.digest([c['case_id'] for c in batch])
            for row in rows.values():
                row.update(collector_batch_id=batch_id,collector_batch_control_samples=batch_cost,collector_batch_size=len(batch))
            output=[rows[i] for i in range(len(batch))]
            return output[0] if single else output
        finally:env.close()
    run.supports_batch=True;run.control_samples=0
    return run
