"""Native v07 sampling and cautious curriculum behavior."""
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v07 import curriculum

class CycleIndices:
    def __init__(self):self.position=0
    def integers(self,count):
        value=self.position%count;self.position+=1;return value

@pytest.mark.parametrize('frontier,near',[(17,(15,16)),(22,(16,17)),(18,(17,22)),(19,(22,18)),(21,(19,20))])
def test_near_sampling_follows_curriculum_order_instead_of_tile_arithmetic(frontier,near):
    assert curriculum.role_tiles(('flat','frontier','near','near'),frontier,CycleIndices())==(0,frontier,*near)

def test_final_task_easy_sampling_includes_d22_and_stops_at_d20():
    expected=tuple(range(1,18))+(22,18)
    assert curriculum.role_tiles(('easy',)*19,21,CycleIndices())==expected

@pytest.mark.parametrize('tile',[23,-1])
def test_sampling_refuses_unknown_physical_frontier(tile):
    with pytest.raises(ValueError):curriculum.role_tiles(('frontier',),tile,CycleIndices())

from copy import deepcopy
import json,hashlib
from v07_test_support import validation_payload

def manager_at(tmp_path,substage='D25',update=4817,gu=0):
    m=curriculum.CurriculumManager(num_envs=512,seed=42)
    state=m.state_dict();state.update(phase='D',substage=substage,level_tick=100,generation=73,
        generation_updates=gu,last_update=update,frontier_tile_id={'D25':17,'D22':22,'D20':18}[substage])
    state['spent_by_phase'].update(F0=200,H=2500,D=update-2700)
    state['spent_by_substage'].update(D25=update-2734,D20=34,D22=0)
    m.load_state_dict(state)
    def sink(payload):
        assert len(m.state.episode_fifo)>=64  # Persistence precedes consumption.
        path=tmp_path/f"window_{payload['identity']['update']}_{payload['identity']['generation']}.json"
        path.write_text(json.dumps(payload,sort_keys=True))
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records_sha256=payload['records_sha256'])
    m.window_sink=sink
    return m

def outcomes(failures,generation=73,tile=17,step=0):
    return [dict(env_id=102+i,step=step+i,role='frontier',tile_id=tile,success=i>=failures,
                 failure=i<failures,timeout=False,reason='nonfoot' if i<failures else 'success',
                 tracking_score=.8,command=.4,generation=generation) for i in range(64)]

def test_two_all_failure_windows_in_first_five_updates_do_not_regress_d20(tmp_path):
    m=manager_at(tmp_path,'D20')
    for i in range(1,6):
        m.observe(outcomes(64,tile=18,step=i*100));m.on_update(4817+i)
    assert m.state.substage=='D20' and m.state.generation==73 and m.state.generation_updates==5
    assert not m.state.finished and m.state.ema['batches']==5 and m.state.minimum_level_failure_streak==0
    assert len(list(tmp_path.glob('window_*.json')))==5

def test_historical_four_failure_windows_are_diagnostic_only_in_lowest_d25(tmp_path):
    m=manager_at(tmp_path)
    for i,count in enumerate([56,17,22,8],1):
        m.observe(outcomes(count,step=i*100));m.on_update(4817+i)
    assert not m.state.finished and m.state.stop_status is None
    assert m.state.minimum_level_failure_streak==0
    assert m.state.ema['f']==pytest.approx(.819839796875)

def test_protection_does_not_block_original_two_valid_gates_promotion(tmp_path):
    m=manager_at(tmp_path)
    for update in [5000,5100]:
        payload,identity=validation_payload(update,73,60,substage='D25')
        m.on_update(update,validation=payload,validation_identity=identity)
    assert m.state.substage=='D22' and m.state.generation==74 and m.state.generation_updates==0
    assert m.regression_policy_state['generation']==74 and m.regression_policy_state['low_success_evaluations']==[]

def test_current_boolean_gate_without_complete_evidence_is_refused(tmp_path):
    m=manager_at(tmp_path);before=deepcopy(m.state_dict())
    with pytest.raises(ValueError):m.on_update(4917,gates=[dict(kind='validation',passed=True,phase='D',substage='D25',update=4917)])
    assert m.state_dict()==before

def test_window_persistence_failure_does_not_pop_fifo_or_apply_ema(tmp_path):
    m=manager_at(tmp_path);m.observe(outcomes(56));before=deepcopy(m.state_dict())
    def fail(payload):raise OSError('injected immutable 64-record persistence failure')
    m.window_sink=fail
    with pytest.raises(OSError):m.on_update(4818)
    assert m.state_dict()==before and len(m.state.episode_fifo)==64 and m.state.ema['batches']==0

def test_protection_does_not_exempt_exhausted_substage_budget(tmp_path):
    m=manager_at(tmp_path,'D20',update=11699)
    m.state.spent_by_substage.update(D25=0,D20=8999)
    m.on_update(11700)
    assert m.state.finished and m.state.stop_status=='budget_exhausted'

def test_policy_duplicate_without_applied_curriculum_transaction_cannot_silently_continue(tmp_path):
    from src.tasks.velocity.v07 import regression_policy
    m=manager_at(tmp_path,'D20',update=6099,gu=1282)
    for update in [5900,6000,6100]:
        payload,identity=validation_payload(update,73,0,substage='D20')
        m.regression_policy_state,_=regression_policy.apply_validation(m.regression_policy_state,payload,
            expected_identity=identity,generation_updates=update-4817)
    before=deepcopy(m.state_dict());policy=deepcopy(m.regression_policy_state)
    with pytest.raises(regression_policy.PolicyEvidenceError):m.on_update(6100,validation=payload,validation_identity=identity)
    assert m.state_dict()==before and m.regression_policy_state==policy
