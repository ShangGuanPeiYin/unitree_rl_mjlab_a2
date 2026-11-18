import json
from src.tasks.velocity.v08 import spec

def test_native_contract_has_v08_identity_no_v_curriculum_and_all_thresholds():
    c=spec.contract();assert c['identity']['code_version']=='0.8.0'
    assert c['phase_order']==list('ABCDE') and sum(c['phase_budgets'].values())==18000
    assert c['standing']['mid_pitch']==spec.MID_PITCH and c['direction']['deadline_steps']==400
    reward=c['reward'];assert '+ 10*I_success' not in reward['composition']
    assert reward['timeout_reward']==-10
    assert 'h00_retention_collapse' not in json.dumps(reward)
