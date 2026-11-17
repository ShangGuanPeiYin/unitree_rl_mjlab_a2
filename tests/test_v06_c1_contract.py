"""C1 runs in clean interpreters so original v06 tests retain their contract."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_c1(code, *, c1=True):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if c1:
        env['A2_V06_CONTINUATION'] = 'v06-c1-speed030060-budget36000'
    else:
        env.pop('A2_V06_CONTINUATION', None)
    preamble = "import sys; sys.path.insert(0,'scripts'); import a2_v06_bootstrap; a2_v06_bootstrap.activate(); "
    result = subprocess.run([sys.executable, '-c', preamble + '\n' + code], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_c1_all_phase_commands_and_evaluation_case_protocols():
    run_c1('''
from src.tasks.velocity.v06 import spec, evaluation, speed
from src.tasks.velocity.v06.env import V06Env
from types import SimpleNamespace
import torch
assert spec.RECIPE_VERSION == '0.6-go2refs-c1'
assert spec.TOTAL_BUDGET == 36000
assert spec.D_SUBSTAGE_BUDGETS == (4500,4500)
assert spec.phase_budget('D') == 9000
for phase, sub in [('F0',None),('H',None),('D','D25'),('L','L12'),('V',None),('R',None)]:
    for u in [-100,0,500,1000,1200,4000]:
        obj=SimpleNamespace(phase=phase,evaluation=False,stage_iteration=u)
        assert V06Env._spawn_ranges(obj)['vx'] == (.3,.6), (phase,u,V06Env._spawn_ranges(obj))
    for kind in ['validation','diagnostic','flat_retention','stair_retention'] + (['test'] if phase=='R' else []):
        table=evaluation.make_cases(phase,sub,kind,42,tile_id=(spec.tile_id_for('H00') if phase=='H' and kind not in ('flat_retention','stair_retention') else None))
        assert table.speeds == [.3,.5,.6], (phase,kind,table.speeds)
        per=100 if kind=='test' else (20 if kind=='validation' else 5)
        assert len(table.cases)==3*per
        if kind=='flat_retention': assert table.target_tile_id==0
        if kind=='stair_retention': assert table.target_tile_id==1
        if kind=='test': assert table.spawn_profile=='full_R_expanded'
for c,d in [(.3,1.5),(.5,2.5),(.6,3.)]:
    assert speed.flat_case_pass(speed.flat_case_metrics(speed=c,steps=500,max_abs_y=0,max_abs_yaw_deg=0,forward=d))
    assert not speed.flat_case_pass(speed.flat_case_metrics(speed=c,steps=500,max_abs_y=0,max_abs_yaw_deg=0,forward=d-5e-7))
assert (spec.GATE_CASES,spec.GATE_PASS_MIN,spec.FINAL_CASES,spec.FINAL_PASS_MIN)==(20,18,100,95)
from src.tasks.velocity.v06 import ledger, curriculum
assert ledger.ACTIVE_BRANCH_CAP==36000 and ledger.FORMAL_CUMULATIVE_CAP==36100
assert ledger.FULL_CAP_COST==ledger.training_cost(36100)
assert curriculum.SUBSTAGE_BUDGETS['D25']==4500
''')


def test_legacy_v06_contract_and_v_ramp_remain_original():
    run_c1('''
from src.tasks.velocity.v06 import spec
from src.tasks.velocity.v06.env import V06Env
from types import SimpleNamespace
import torch
assert spec.RECIPE_VERSION=='0.6-go2refs-r1' and spec.TOTAL_BUDGET==30000
for u,want in [(0,(.3,.8)),(500,(.25,.7)),(1000,(.2,.6)),(1200,(.2,.6))]:
    got=V06Env._spawn_ranges(SimpleNamespace(phase='V',evaluation=False,stage_iteration=u))['vx']
    assert all(abs(a-b)<1e-12 for a,b in zip(got,want)), (u,got)
''', c1=False)


def test_c1_seed_order_pose_content_is_preserved():
    run_c1('''
import numpy as np
from src.tasks.velocity.v06 import evaluation
for sub in ['D25','D20']:
    table=evaluation.make_cases('D',sub,'validation',42)
    rng=np.random.default_rng(200342)
    for case in table.cases:
        want=[rng.uniform(-1.1,-.9),rng.uniform(-.05,.05),rng.uniform(-np.pi/36,np.pi/36)]
        assert case['initial_pose']==want
assert evaluation.seed_for('test',42)==400342
''')
