#!/usr/bin/env python3
"""A2 v1.0 explicit phase executor; this CLI never bypasses a behavior gate."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    for name in ('smoke','probe','_probe-stack','probe-a1','_probe-a1-stack','probe-a1-v2','_probe-a1-stack-v2','worker'):
        p=sub.add_parser(name);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0')
        if name in ('smoke','worker','_probe-stack','_probe-a1-stack','_probe-a1-stack-v2'):p.add_argument('--num-envs',type=int,default=6 if name=='smoke' else 1024)
        if name in ('smoke','worker'):p.add_argument('--course',choices=['D1','D2','D3','P0','P1'],default='D3' if name=='smoke' else 'D1')
        if name=='smoke':p.add_argument('--steps',type=int,default=48)
        if name in ('probe','_probe-stack','probe-a1','_probe-a1-stack','probe-a1-v2','_probe-a1-stack-v2'):
            p.add_argument('--global-probe-ledger',default=str(ROOT/'logs/a2_v10/global_probe_costs.jsonl'))
            p.add_argument('--probe-budget-authorization',required=True)
            p.add_argument('--probe-sequence-id',required=True)
        if name in ('_probe-stack','_probe-a1-stack','_probe-a1-stack-v2'):p.add_argument('--stack',choices=['dagger','ppo'],required=True)
        if name in ('probe-a1','_probe-a1-stack','probe-a1-v2','_probe-a1-stack-v2'):p.add_argument('--linkage-contract',required=True)
        if name=='_probe-a1-stack-v2':p.add_argument('--startup-headroom',required=True)
        if name=='worker':
            p.add_argument('--stop-round',type=int,required=True);p.add_argument('--resume');p.add_argument('--acceptance');p.add_argument('--ledger-path');p.add_argument('--controller-state')
            p.add_argument('--trial-charged-cap',type=int);p.add_argument('--trial-id');p.add_argument('--trial-guard');p.add_argument('--formal-handoff')
    p=sub.add_parser('export');p.add_argument('--checkpoint',required=True);p.add_argument('--acceptance',required=True);p.add_argument('--output-dir',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    if command in ('probe-a1-v2','_probe-a1-stack-v2'):
        # Import by file path: importing src.tasks would load Torch via auto-registration.
        import importlib.util,os
        if 'torch' in sys.modules:raise RuntimeError('v2 startup must precede Torch import')
        helper_spec=importlib.util.spec_from_file_location('v10_headroom_preimport',ROOT/'src/tasks/velocity/v10/headroom.py')
        helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
        if command=='probe-a1-v2':
            report=helper.wait_for_headroom()
            path=Path(str(Path(args['output_dir']).resolve())+'.startup_headroom.json');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as f:json.dump(report,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
            if not report['passed']:
                print(json.dumps(dict(passed=False,reason='startup_headroom_below_12GiB_continuous_60s',headroom_report=str(path),physical_samples=0,protocol_claimed=False)))
                return 1
            args['startup_headroom']=str(path)
        else:helper.validate_startup_report(args['startup_headroom'],live=True)
    if command in ('smoke','worker','_probe-stack','_probe-a1-stack','_probe-a1-stack-v2'):
        from src.tasks.velocity.v10.gpu_lock import gpu_execution_lock
    if command=='smoke':
        from src.tasks.velocity.v10.runtime import smoke
        with gpu_execution_lock():result=smoke(**args)
    elif command=='worker':
        from src.tasks.velocity.v10.runtime import worker
        with gpu_execution_lock():result=worker(**args)
    elif command=='probe':
        from src.tasks.velocity.v10.resources import run_probes
        result=run_probes(**args)
    elif command=='probe-a1':
        from src.tasks.velocity.v10.resource_a1 import run_a1_probes
        result=run_a1_probes(**args)
    elif command=='_probe-a1-stack':
        from src.tasks.velocity.v10.resource_a1 import a1_probe_stack
        with gpu_execution_lock():result=a1_probe_stack(**args)
    elif command=='probe-a1-v2':
        from src.tasks.velocity.v10.resource_a1 import run_a1_v2_probes
        result=run_a1_v2_probes(**args)
    elif command=='_probe-a1-stack-v2':
        from src.tasks.velocity.v10.resource_a1 import a1_v2_probe_stack
        with gpu_execution_lock():result=a1_v2_probe_stack(**args)
    elif command=='_probe-stack':
        from src.tasks.velocity.v10.resources import probe_stack
        with gpu_execution_lock():result=probe_stack(**args)
    else:result=export(**args)
    print(json.dumps(result,sort_keys=True));return 0 if result.get('passed',True) else 1

def export(checkpoint,acceptance,output_dir):
    import torch
    from src.tasks.velocity.v10 import spec
    from src.tasks.velocity.v10.model import V10ActorCritic
    from src.tasks.velocity.v10.checkpoint import checkpoint_load,sha256,atomic_json
    certificate=json.loads(Path(acceptance).read_text())
    if (certificate.get('token') not in ('V10_DAGGER_ACCEPTED','V10_DAGGER_PPO_ACCEPTED') or certificate.get('model_sha256')!=sha256(checkpoint)
       or certificate.get('sds_sha256')!=spec.SDS_SHA256 or certificate.get('passed') is not True):raise ValueError('accepted student required for deployment export')
    state=checkpoint_load(checkpoint,num_envs=torch.load(checkpoint,map_location='cpu',weights_only=False)['num_envs'],physical_sha256=certificate['physical_sha256'],restore_random=False)
    model=V10ActorCritic();model.load_state_dict(state['model']);model.eval().requires_grad_(False)
    example=dict(proprio=torch.zeros(1,45),history=torch.zeros(1,5,45),heightmap=torch.zeros(1,3,40,30),route_error=torch.zeros(1,3))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    traced=torch.jit.trace(model.actor,(example,),strict=True);traced.save(str(output/'student_actor.pt'))
    contract=dict(sds_sha256=spec.SDS_SHA256,checkpoint_sha256=sha256(checkpoint),actor_sha256=sha256(output/'student_actor.pt'),
        physical_sha256=certificate['physical_sha256'],teacher_runtime=False,inputs={k:list(v.shape[1:]) for k,v in example.items()},
        action=dict(dimension=12,scale=.25,target='q0 + .25 * action',clip=None),control_dt=.02,map_refresh_steps=5,history_frames=5,
        proprio_order='angular_velocity3 projected_gravity3 command3 relative_joint_pos12 joint_velocity12 previous_applied_action12',
        route='clip(local_y/.20,-4,4), sin(yaw_error), cos(yaw_error)')
    atomic_json(output/'contract.json',contract);return dict(passed=True,output_dir=str(output),actor_sha256=contract['actor_sha256'])

if __name__=='__main__':raise SystemExit(main())
