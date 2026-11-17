"""Bare native reload must preserve policy, generation and accepted RNG order."""
from copy import deepcopy
import pytest
from v07_runner_support import *
from src.tasks.velocity.v07 import regression_policy,evaluation_records
from v07_test_support import validation_payload

def test_first_and_second_native_load_keep_learning_and_policy_and_apply_schedule_before_reset(tmp_path):
    checkpoint=migrated_checkpoint(tmp_path/'migrated.pt')
    r=loaded_runner(tmp_path/'first',checkpoint)
    assert r.env.events==[('schedule',73,2117),('reset',73,2117)]
    assert r.v07_initialization_audit['sequence']==['restore_checkpoint_rng','original_schedule_once','original_ppo_reset_once']
    for key in ['model','optimizer','normalizer','learning']:
        assert migration.tree_digest(r.checkpoint_state()[key])==migration.tree_digest(torch.load(checkpoint['path'],weights_only=False)[key])
    set_synthetic_boundary(r,5900,gu=1083)
    payload,identity=validation_payload(5900,73,4)
    r.curriculum.regression_policy_state,_=regression_policy.apply_validation(r.curriculum.regression_policy_state,payload,expected_identity=identity,generation_updates=1083)
    saved=r.save('native');policy=deepcopy(r.curriculum.regression_policy_state)
    second=loaded_runner(tmp_path/'second',saved)
    assert second.curriculum.regression_policy_state==policy
    assert second.env.events==[('schedule',73,3200),('reset',73,3200)]
    assert second.curriculum.state.generation_updates==1083 and second.curriculum.state.generation==73
    assert second.curriculum.state.retention_next_due==dict(flat=5000,h00=5000)

@pytest.mark.parametrize('action',['pause','stop',None])
def test_r1_resume_contract_latch_guard_keeps_evidence_readable_and_does_not_mutate_it(tmp_path,action):
    checkpoint=migrated_checkpoint(tmp_path/'source.pt');r=loaded_runner(tmp_path/'loaded',checkpoint)
    state=r.checkpoint_state();state['ledger']['error_type']='ControlProtocolError'
    audit=dict(error_type='ControlProtocolError',raw_base64='e30=',parse_error='retained error')
    if action is not None:audit['latched_request']=dict(action=action,resume_authorized=False)
    state['ledger']['terminal']['evidence']['control_error']=audit
    digest=migration.tree_digest(state)
    if action is None:assert recovery.resume_contract(state)['resume_mode']=='new_episode'
    else:
        with pytest.raises(recovery.RecoveryBlocked,match='unresolved.*operator'):recovery.resume_contract(state)
    assert migration.tree_digest(state)==digest and state['ledger']['stop_state'] is None
