"""Executable native v08 modes; engineering output can never initialize formal."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import torch
from . import spec,recovery,orchestration
from . import continuation_v08 as migration

SOURCE_PATH=migration.SOURCE_PATH
SOURCE_SHA256=migration.SOURCE_SHA256
FORMAL_OUTPUT=Path('/home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs_v08/continue_seed42_from006400_trial01')
verify_learning_journal=recovery.verify_learning_journal
MODES=('formal','smoke','probe','baseline','coverage','final','replay','freeze','migration')

def wandb_enabled():
    import os
    flag=str(os.environ.get('A2_V08_WANDB','')).strip().lower()
    if flag in ('0','false','no','off'):return False
    if flag in ('1','true','yes','on'):return True
    return bool(os.environ.get('WANDB_API_KEY'))

def build_telemetry(output_dir,seed=42,backend=None):
    from .telemetry import V08Telemetry,WandbBackend
    directory=Path(output_dir)/'telemetry'
    if backend is None and not wandb_enabled():return V08Telemetry(directory)
    from .connection import ReliableConnection
    return ReliableConnection(directory,backend=backend or WandbBackend(name=Path(output_dir).name))

def build_parser():
    parser=argparse.ArgumentParser(description='A2 v08 exact6400 migration and native lifecycle')
    parser.add_argument('--source-checkpoint',required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--num-envs',required=True,type=int)
    parser.add_argument('--updates',required=True,type=int)
    parser.add_argument('--mode',required=True,choices=MODES)
    parser.add_argument('--policy-checkpoint')
    parser.add_argument('--policy-sha256')
    parser.add_argument('--record-video',action='store_true')
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--acceptance-file')
    parser.add_argument('--baseline-report')
    parser.add_argument('--freeze-file')
    parser.add_argument('--request-speed',type=float)
    parser.add_argument('--case-id')
    parser.add_argument('--phase',choices=spec.PHASE_ORDER,default='A')
    parser.add_argument('--max-control-steps',type=int,default=50000)
    return parser

def resolve_request(args):
    c=vars(args).copy();mode=c['mode']
    if Path(c['source_checkpoint']).resolve()!=Path(SOURCE_PATH).resolve() or c['source_sha256']!=SOURCE_SHA256:
        raise ValueError('only canonical6400 source allowed')
    if c['seed']!=42:raise ValueError('seed42 is fixed')
    if c['device']!='cpu' and c['device']!='cuda:0':raise ValueError('device must be cpu or cuda:0')
    if mode=='formal' and (c['num_envs']!=512 or not 1<=c['updates']<=18000 or c['device']!='cuda:0'):
        raise ValueError('formal requires512 envs/cuda:0/1..18000 updates')
    if mode=='formal' and Path(c['output_dir']).resolve()!=FORMAL_OUTPUT:raise ValueError('formal output must be the unique approved canonical directory')
    if mode=='smoke' and (c['num_envs'],c['updates'])!=(8,2):raise ValueError('smoke requires8env/2updates')
    if mode=='probe' and (c['num_envs'],c['updates'])!=(512,56):raise ValueError('probe requires512env/56updates')
    if mode not in ('formal','smoke','probe') and (c['updates']!=0 or not 1<=c['num_envs']<=8):
        raise ValueError('zero-learning modes require0updates/1..8env')
    if c['request_speed'] is not None and (not math.isfinite(c['request_speed']) or not -.6<=c['request_speed']<=.6):
        raise ValueError('request speed must be finite in[-.6,+.6]')
    if mode in ('formal','probe','baseline','final','coverage') and c['request_speed'] is not None:
        raise ValueError('frozen task cases control requests in this mode')
    if mode in ('final','replay') and (not c['policy_checkpoint'] or not c['policy_sha256']):
        raise ValueError('final/replay requires native policy checkpoint and SHA')
    if bool(c['policy_checkpoint'])!=bool(c['policy_sha256']):raise ValueError('policy file and SHA must be supplied together')
    if mode=='replay' and c['num_envs']!=1:raise ValueError('replay requires one physical environment')
    if c['record_video'] and mode!='replay':raise ValueError('video recording is a replay operation')
    if c['max_control_steps']<=0:raise ValueError('positive max control steps required')
    if bool(c['resume_checkpoint'])!=bool(c['resume_sha256']):raise ValueError('resume requires both file and SHA')
    if c['resume_checkpoint'] and mode not in ('formal','smoke','probe'):raise ValueError('native resume only for learning mode')
    if mode=='probe' and c['resume_checkpoint']:raise ValueError('probe55reload is owned by runner, no external resume')
    return c

def frozen_inputs(contract):
    if contract.get('freeze_file'):
        frozen=json.loads(Path(contract['freeze_file']).read_text())
        # freeze artifact includes cases; raw source freeze is also accepted.
        if not isinstance(frozen,dict) or set(frozen)!=set(('freeze','case_tables')):
            raise ValueError('freeze artifact requires both source/config freeze and frozen case tables')
        freeze=frozen['freeze']
        recovery.verify_freeze(freeze)
        cases=frozen['case_tables']
        recovery.verify_case_tables(cases)
    else:
        freeze=recovery.build_freeze();cases=recovery.freeze_case_tables()
    return freeze,cases

def build_environment(contract,*,phase='A',initial_cases=None,evaluation=False,num_envs=None,stage_iteration=0):
    from .env import V08Env
    return V08Env(num_envs=contract['num_envs'] if num_envs is None else num_envs,device=contract['device'],
        seed=42,phase=phase,initial_cases=initial_cases,evaluation=evaluation,stage_iteration=stage_iteration)

def build_case_runner(model,contract,phase='A'):
    from .runner import environment_case_runner
    def factory(cases):
        return build_environment(contract,phase=run.phase,initial_cases=cases,evaluation=True,num_envs=len(cases))
    base=environment_case_runner(model,factory,device=contract['device'],max_steps=contract['max_control_steps'])
    limit=min(int(contract['num_envs']),8)
    def run(cases):
        single=isinstance(cases,dict);batch=[cases] if single else list(cases);rows=[]
        if not batch or len(batch)>8:raise ValueError('evaluation batch must contain1..8cases')
        try:
            for i in range(0,len(batch),limit):rows.extend(base(batch[i:i+limit]))
        finally:run.control_samples=base.control_samples
        return rows[0] if single else rows
    run.supports_batch=True;run.control_samples=0;run.phase=phase
    return run

def prepare_runner(contract,output_dir):
    from .model import V08ActorCritic
    from .ppo import PPOTrainer
    from .curriculum import CurriculumManager
    from .runner import V08Runner
    frozen,cases=frozen_inputs(contract)
    admission=None
    if contract['mode']=='formal':
        admission=orchestration.admit(acceptance_file=contract.get('acceptance_file'),
            baseline_report=contract.get('baseline_report'),frozen=frozen,case_tables=cases,formal=True)
    source=migration.load_designated_source(contract['source_checkpoint'],source_sha256=contract['source_sha256'])
    state,diff=migration.migrate_stop_6400(source,source_sha256=SOURCE_SHA256)
    model=V08ActorCritic().to(contract['device'])
    manager=CurriculumManager(num_envs=contract['num_envs'],seed=42)
    env=build_environment(contract,phase=manager.phase,initial_cases=manager.schedule().cases)
    ppo=PPOTrainer(model,env)
    runner=V08Runner(model,ppo,manager,env,output_dir,seed=42,device=contract['device'],mode=contract['mode'],
        telemetry=build_telemetry(output_dir,seed=42))
    runner.case_runner=build_case_runner(model,contract,phase=manager.phase)
    try:
        if contract.get('resume_checkpoint'):
            runner.load(contract['resume_checkpoint'],contract['resume_sha256'])
            runner.start_telemetry()
        else:runner.initialize(state)
    except BaseException:
        env.close();raise
    if runner.freeze!=frozen or runner.case_tables!=cases:raise ValueError('runner frozen inputs differ')
    suffix=f'_resume_{runner.update:06d}_{runner.ledger.state.get("save_sequence",0):04d}' if contract.get('resume_checkpoint') else ''
    recovery.write_json(Path(output_dir)/('migration_diff'+suffix+'.json'),diff)
    recovery.write_json(Path(output_dir)/('admission'+suffix+'.json'),admission)
    recovery.write_json(Path(output_dir)/('frozen_inputs'+suffix+'.json'),dict(freeze=frozen,case_tables=cases))
    path,sha=runner.save_verified('ready')
    recovery.write_json(Path(output_dir)/('ready'+suffix+'.json'),dict(checkpoint=str(path),sha256=sha,learning=runner.learning,
        first_new_update=runner.update+1,physical_trajectory_continuous=False,rng=runner.rng_receipt))
    if contract.get('request_speed') is not None:
        runner.env.request_speed(list(range(contract['num_envs'])),contract['request_speed'])
        runner.ppo.obs=runner.env.get_observations()
    return runner

def zero_learning(contract,output_dir):
    from .model import V08ActorCritic
    from .ppo import build_optimizer
    from .evaluation import table_from_json,evaluate_cases,CaseTable,make_cases
    from .evaluation_records import pack_result,write_sidecar
    frozen,cases=frozen_inputs(contract);mode=contract['mode'];admission=None
    if mode in ('baseline','final'):
        admission=orchestration.admit(acceptance_file=contract.get('acceptance_file'),
            baseline_report=contract.get('baseline_report'),frozen=frozen,case_tables=cases,formal=mode=='final')
    source=migration.load_designated_source();model=V08ActorCritic().to(contract['device'])
    policy=source
    if mode in ('final','replay'):
        policy=recovery.load_verified(contract['policy_checkpoint'],contract['policy_sha256'])
        recovery.validate_checkpoint(policy,mode='formal')
        if policy['ledger'].get('pending_update') is not None:raise ValueError('partial native policy cannot be evaluated')
    model.load_state_dict(policy['model']);optimizer=build_optimizer(model);optimizer.load_state_dict(policy['optimizer'])
    if mode=='replay':return replay_cases(model,contract,output_dir,frozen,cases,policy)
    table=table_from_json(cases['baseline:A' if mode=='baseline' else 'final:E' if mode=='final' else 'gate:E']['table'])
    runner=build_case_runner(model,contract,phase=table.phase)
    identity=dict(update=policy['learning']['update'],phase=table.phase,kind=table.kind,case_table_sha256=table.sha256,
        source_sha256=SOURCE_SHA256,recipe_sha256=recovery.digest(recovery.jsonable(spec.contract())),freeze_sha256=frozen['sha256'])
    transaction=recovery.EvaluationTransaction(output_dir,identity,table,Path(output_dir)/'evaluation.json',runner)
    cached=transaction.recover()
    if cached is None:
        transaction.start()
        result=evaluate_cases(model,table,transaction,optimizer=optimizer,counters=deepcopy(policy['learning']),device=contract['device'])
        result,reference=transaction.finish(result)
    else:result,reference=cached
    payload=pack_result(result,identity);binding=reference['binding']
    complete=result.summary['valid']==result.summary['expected'] and not result.summary['engineering_error']
    report=dict(mode=mode,complete=complete,source_checkpoint_sha256=SOURCE_SHA256,freeze_sha256=frozen['sha256'],
        case_tables_sha256=recovery.digest(cases),acceptance_sha256=admission['acceptance']['sha256'] if admission else None,
        evaluation=dict(payload=payload,binding=binding),evaluation_transaction=reference,
        control_samples=reference['control_samples'],physics_substeps=reference['control_samples']*4,
        learning_updates=0,evidence_kind='real_policy_and_physics',policy_capability_established=mode=='final' and result.summary['gate_pass'],
        policy_checkpoint_sha256=contract.get('policy_sha256') or SOURCE_SHA256)
    recovery.write_json(Path(output_dir)/(mode+'_result.json'),report)
    return report

def prescribed_replay_cases(table):
    wanted=[('flat_stop',.5,None),('up_stop',.5,None),('E1',.5,None),('E2',.5,None),
        ('E3',.5,'up_reverse'),('E4',.5,None),('E4',.3,None),('E4',.6,None)]
    return [deepcopy(next(c for c in table.cases if c['task']==task and c['speed']==speed
        and (script is None or c['script']==script))) for task,speed,script in wanted]

def replay_cases(model,contract,output_dir,frozen,case_tables,policy):
    from .evaluation import table_from_json,normalize_record
    from .env import V08Env
    from .events import euler_heading
    table=table_from_json(case_tables['final:E']['table'])
    selected=prescribed_replay_cases(table)
    if contract.get('case_id'):
        selected=[deepcopy(c) for c in table.cases if c['case_id']==contract['case_id']]
        if len(selected)!=1:raise ValueError('case-id absent from frozen final table')
    model.eval();records=[];total_steps=0
    settings=dict(width=1280,height=720,fps=25,control_dt=.02,frame_stride=2,
        azimuth='90+body_yaw_degrees',elevation=-28.,distance=3.8,speed=1.)
    for case in selected:
        env=V08Env(num_envs=1,device=contract['device'],seed=42,phase='E',initial_cases=[case],evaluation=True)
        renderer=None;writer=None;last_capture=-1;current_step=0
        video_path=Path(output_dir)/(case['case_id']+'.mp4')
        trace_path=Path(output_dir)/(case['case_id']+'.jsonl')
        terminal_row=None
        try:
            if contract['record_video']:
                import imageio.v2 as imageio
                import mujoco
                from mjlab.viewer.offscreen_renderer import OffscreenRenderer
                cfg=deepcopy(env.cfg.viewer);cfg.width=1280;cfg.height=720;cfg.distance=3.8
                cfg.origin_type=cfg.OriginType.WORLD;cfg.max_extra_envs=0
                renderer=OffscreenRenderer(env.sim.mj_model,cfg,env.scene);renderer.initialize()
                writer=imageio.get_writer(str(video_path),fps=25,codec='libx264',macro_block_size=1)
                camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(env.sim.mj_model,camera)
                camera.elevation=-28.;camera.distance=3.8
            def capture(info=None,force=False):
                nonlocal last_capture
                if writer is None or current_step==last_capture or (not force and current_step%2):return
                import numpy as np
                from PIL import Image,ImageDraw,ImageFont
                camera.lookat[:]=env.robot.data.root_link_pos_w[0].detach().cpu().numpy()
                yaw=float(euler_heading(env.robot.data.root_link_quat_w)[2][0])
                camera.azimuth=90.+math.degrees(yaw)
                renderer.update(env.sim.data,camera=camera)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                executed=float(info['command_executed'][0,0]) if info is not None else float(env.command[0,0])
                requested=float(env.requested_c[0]);segment=env.segment(0)
                text=f"case={case['case_id']}\nrequested={requested:+.3f} m/s  actor={executed:+.3f} m/s  phase={segment}  t={current_step*.02:.2f}s"
                font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
                draw.rectangle((0,0,1280,70),fill='black');draw.multiline_text((12,8),text,fill='white',font=font,spacing=5)
                writer.append_data(np.asarray(image));last_capture=current_step
            # Observe physical terminal state before the automatic reset. This
            # hook renders only; it never writes pose, velocity or contact state.
            env.pre_reset_callback=lambda live,info:capture(info,force=True)
            if contract.get('request_speed') is not None:env.request_speed([0],contract['request_speed'])
            observations=env.get_observations();capture()
            with trace_path.open('x') as trace,torch.no_grad():
                for current_step in range(1,contract['max_control_steps']+1):
                    observations,reward,done,truncated,info=env.step(model.act(observations,deterministic=True))
                    total_steps+=1
                    record=dict(step=current_step,control_time=current_step*.02,
                        requested_c=float(info['command_requested'][0]),actor_c=float(info['command_executed'][0,0]),
                        segment=info['episodes'][0]['segment'] if info['episodes'] else env.segment(0))
                    trace.write(json.dumps(record,sort_keys=True)+'\n')
                    if info['episodes']:
                        terminal_row=deepcopy(info['episodes'][0]);break
                    capture(info)
            if terminal_row is None:terminal_row=dict(case_id=case['case_id'],complete=False,raw_complete=False,
                finite=True,success=False,reason='collector_timeout',segment='engineering')
            result=normalize_record(terminal_row,case)
            result.update(control_steps=current_step,trace_path=str(trace_path),
                video_path=str(video_path) if writer else None,video_settings=settings)
            records.append(result)
            recovery.write_json(Path(output_dir)/(case['case_id']+'_result.json'),result)
        finally:
            if writer is not None:writer.close()
            if renderer is not None:renderer.close()
            env.close()
    report=dict(mode='replay',evidence_kind='real_policy_and_physics',learning_updates=0,
        policy_checkpoint_sha256=contract['policy_sha256'],freeze_sha256=frozen['sha256'],
        case_table_sha256=table.sha256,control_samples=total_steps,physics_substeps=total_steps*4,
        records=records,video_settings=settings,prescribed_case_ids=[c['case_id'] for c in selected],
        failures_retained=True,protocol_override=contract.get('request_speed'),final_acceptance_complete=False)
    recovery.write_json(Path(output_dir)/'replay_result.json',report)
    return report

def admit_output(contract):
    output=Path(contract['output_dir'])
    if not output.exists():
        if contract.get('resume_checkpoint'):raise ValueError('native resume must use existing original run directory')
        output.mkdir(parents=True);return False
    path=contract.get('resume_checkpoint');sha=contract.get('resume_sha256')
    if not path or not sha:raise ValueError('output directory must be new unless strict native recovery')
    candidate=Path(path).resolve();checkpoint_dir=(output/'checkpoints').resolve()
    if candidate.parent!=checkpoint_dir:raise ValueError('native resume checkpoint must belong to this run')
    receipts=[]
    for metadata in checkpoint_dir.glob('*.json'):
        value=json.loads(metadata.read_text())
        if value.get('mode')!=contract['mode']:raise ValueError('foreign checkpoint mode in run directory')
        sequence=int(value.get('sequence',metadata.stem.rsplit('_',1)[-1]))
        receipts.append((int(value['update']),sequence,value))
    if not receipts:raise ValueError('native durable checkpoint receipts missing')
    latest=max(receipts)[2]
    verify_learning_journal(output,int(latest['update']))
    recovery.verify_evaluation_journal(output)
    if Path(latest['path']).resolve()!=candidate or latest['sha256']!=sha or recovery.sha256(candidate)!=sha:
        raise ValueError('native recovery requires latest durable checkpoint; rollback is not authorized')
    return True

def main(argv=None,allowed_modes=None):
    contract=resolve_request(build_parser().parse_args(argv))
    if allowed_modes is not None and contract['mode'] not in allowed_modes:raise ValueError('mode not supported by this entrypoint')
    output=Path(contract['output_dir'])
    resumed=admit_output(contract)
    mode=contract['mode']
    if mode=='freeze':
        frozen,cases=frozen_inputs(contract)
        report=dict(freeze=frozen,case_tables=cases)
        recovery.write_json(output/'frozen_inputs.json',report)
    elif mode=='migration':
        source=migration.load_designated_source();state,report=migration.migrate_stop_6400(source,source_sha256=SOURCE_SHA256)
        recovery.write_json(output/'migration_diff.json',report)
    elif mode in ('baseline','coverage','final','replay'):report=zero_learning(contract,output)
    else:
        runner=prepare_runner(contract,output)
        try:report=runner.run(contract['updates'])
        finally:runner.env.close()
    print(json.dumps(dict(mode=mode,output_dir=str(output),command_completed=True,
        protocol_complete=report.get('complete'),final_acceptance_complete=report.get('policy_capability_established',False)),sort_keys=True))
    return 0
