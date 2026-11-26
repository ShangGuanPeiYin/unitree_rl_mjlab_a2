"""Opt-in real CUDA engineering checks; these advance counted physical samples.

Run only under the controller's exclusive GPU/engineering budget reservation:
V10_RUN_GPU_TESTS=1 MUJOCO_GL=egl python -m pytest -q tests/test_v10_env_gpu.py
"""
import os
import pytest
import torch
from src.tasks.velocity.v10.cases import suite
from src.tasks.velocity.v10.env import V10Env

pytestmark=pytest.mark.skipif(os.environ.get('V10_RUN_GPU_TESTS')!='1',reason='explicit counted CUDA engineering run required')


def test_real_six_role_observation_and_snapshots():
    all_rows=suite('D3','TRAIN',1)
    selected=[next(c for c in all_rows if c['task']==r) for r in ('F_WALK','F_STAND','F_STOP','P_WALK','P_STOP','S_ROUTE')]
    env=V10Env(6,'D3',42,'cuda:0',cases=selected)
    try:
        snapshot=None
        for _ in range(20):
            a,b=env.observe(),env.observe()
            for key in a:assert torch.equal(a[key],b[key])
            obs,reward,done,info=env.step(torch.zeros(6,12,device='cuda:0'))
            assert torch.equal(obs['proprio'],obs['history'][:,-1])
            assert torch.equal(obs['proprio'][:,6],env.command[:,0])
            assert torch.isfinite(reward).all()
            if snapshot is None:
                old=info
                snapshot={k:v.clone() for k,v in info['frame'].items() if isinstance(v,torch.Tensor)}
        for key,value in snapshot.items():assert torch.equal(value,old['frame'][key])
    finally:env.close()


def test_real_platform_route_fall_protection():
    rows=suite('D3','TRAIN',1)
    selected=[next(c for c in rows if c['task']=='P_WALK' and c['height']==h and c['width']==4. and c['encoding']=='G') for h in (2.,4.,8.)]
    env=V10Env(3,'D3',42,'cuda:0',cases=selected)
    try:
        state=env.robot.data.default_root_state.clone();state[:,3:]=0;state[:,3]=1
        state[:,:3]=env.origins[env.tile_ids]+torch.tensor([[1.,.5,1.],[1.,.5,3.],[1.,.5,7.]],device='cuda:0')
        env.robot.write_root_state_to_sim(state);env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
        _,_,done,info=env.step(torch.zeros(3,12,device='cuda:0'))
        assert done.all() and info['failure'].all()
    finally:env.close()


def test_real_first_terminal_freezes_and_padding_is_zero():
    rows=suite('D3','TRAIN',1)
    case=next(c for c in rows if c['task']=='P_WALK' and c['height']==8.)
    env=V10Env(1,'D3',42,'cuda:0',cases=[case],auto_reset=False)
    try:
        state=env.robot.data.default_root_state.clone();state[:,3:]=0;state[:,3]=1
        state[:,:3]=env.origins[env.tile_ids]+torch.tensor([[1.,0.,7.]],device='cuda:0')
        env.robot.write_root_state_to_sim(state);env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
        _,_,done,info=env.step(torch.ones(1,12,device='cuda:0'))
        assert done.item() and len(info['episodes'])==1
        frozen=env.terminal_records.copy();count=env.metrics[0].count;steps=env.states[0].step
        _,_,done,info=env.step(torch.ones(1,12,device='cuda:0'))
        assert done.item() and not info['episodes']
        assert not info['frame']['action'].any()
        assert not info['frame']['top'].any() and not info['frame']['success'].any() and not info['frame']['failure'].any()
        assert env.metrics[0].count==count and env.states[0].step==steps and env.terminal_records==frozen
    finally:env.close()
