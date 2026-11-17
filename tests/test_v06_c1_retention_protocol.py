"""The additional GPU protocol run is zero-learning and accounts real work."""
from test_v06_c1_contract import run_c1


def test_retention_plan_uses_original_standard_cases_and_preflight_has_no_gpu_or_run():
    run_c1('''
import validate_a2_ideal_stairs_v06_c1 as driver
from src.tasks.velocity.v06 import evaluation
from pathlib import Path
import torch
plan=driver.retention_protocol_plan()
assert plan['cases_total']==30 and plan['learning_updates']==0
assert plan['curriculum_eligibility'] is False
assert plan['speeds']==[.3,.5,.6]
assert [x['kind'] for x in plan['sets']]==['flat_retention','stair_retention']
for selected,target in zip(plan['sets'],[0,1]):
    table=evaluation.table_from_json(selected['case_table'],expected_sha=selected['case_table_sha256'])
    assert table.table_seed==200342 and table.target_tile_id==target
    assert table.spawn_profile=='standard' and len(table.cases)==15
    assert [sum(c['vx']==v for c in table.cases) for v in [.3,.5,.6]]==[5,5,5]
    assert all(-1.1<=c['initial_pose'][0]<=-.9 and -.05<=c['initial_pose'][1]<=.05 for c in table.cases)
import tempfile
with tempfile.TemporaryDirectory() as directory:
    driver.ROOT=Path(directory)
    output=driver.ROOT/driver.IMPLEMENTATION/'gpu/q010_v3/retention-protocol'
    assert not output.exists()
    assert driver.main(['--stage','retention-protocol','--seed','42','--device','cuda:0'])==0
    assert not output.exists() and not torch.cuda.is_initialized()
''')


def test_cost_separates_first_episode_samples_from_extra_batch_simulation():
    run_c1('''
import validate_a2_ideal_stairs_v06_c1 as driver
records=[dict(steps=2,physical_control_samples=10,extra_control_samples_after_case=8,
              physical_physics_substeps=40,extra_physics_substeps_after_case=32),
         dict(steps=10,physical_control_samples=10,extra_control_samples_after_case=0,
              physical_physics_substeps=40,extra_physics_substeps_after_case=0)]
cost=driver.evaluation_cost(records)
assert cost==dict(case_control_steps=12,actual_eval_control_samples=20,
                 extra_eval_control_samples=8,actual_eval_physics_substeps=80,
                 extra_eval_physics_substeps=32)
records[0]['physical_physics_substeps']=41
try:driver.evaluation_cost(records)
except ValueError:pass
else:raise AssertionError('unbalanced physical cost was accepted')
''')
