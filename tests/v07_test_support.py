"""Complete JSON evidence fixtures; no environment/model/PPO is constructed."""
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import evaluation,evaluation_records

def validation_payload(update=5900,generation=73,successes=0,phase='D',substage='D20',kind='validation'):
    table=evaluation.make_cases(phase,substage,kind,42)
    rows=[]
    for i,case in enumerate(table.cases):
        passed=i<successes;env_id=int(case['episode_index'])%8
        ep=dict(env_id=env_id,tile_id=case['tile_id'],control_samples=10,terminal_step=10,
            success=passed,failure=not passed,timeout=False,reason='success' if passed else 'nonfoot',
            active_samples=10,stall_samples=0,first_step_valid=True,mean_vx=case['vx'],mean_speed=case['vx'],max_abs_y=0.,max_abs_yaw_deg=0.)
        rows.append(dict(case_id=case['case_id'],vx=case['vx'],climb_success=passed,failure=not passed,
            flat_pass=False,measurement='stair',reason=ep['reason'],mean_vx=case['vx'],mean_speed=case['vx'],
            steps=10,evidence_kind='terminal',first_episode=ep,first_observed_at_batch_step=10,
            active_samples=10,stall_samples=0,first_step_valid=True,
            physical_control_samples=10,physical_physics_substeps=40,
            extra_control_samples_after_case=0,extra_physics_substeps_after_case=0))
    isolation=dict(statistics_unchanged=True,statistics_changed=[],snapshot_sha='f'*64)
    summary=evaluation.summarize(rows,table);summary.update(isolation)
    hashes=dict(case_table_sha256=table.sha256,records_sha256=evaluation_records._digest(rows),summary_sha256=evaluation_records._digest(summary))
    identity=dict(update=update,generation=generation,phase=phase,substage=substage,kind=kind,
        case_table_sha256=table.sha256,round=table.round,model_sha256='a'*64,
        optimizer_sha256='b'*64,source_closure_sha256='c'*64,contract_sha256='d'*64)
    result=evaluation.EvalResult(table,summary,rows,isolation,hashes)
    return evaluation_records.pack_result(result,identity),identity

def rehash_records(payload):
    payload['evidence_sha256']['records_sha256']=evaluation_records._digest(payload['records'])
    return payload
