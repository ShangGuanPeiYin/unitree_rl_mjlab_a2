from copy import deepcopy
import importlib
import importlib.util
import pytest
from v07_test_support import validation_payload,rehash_records

def policy():
    assert importlib.util.find_spec('src.tasks.velocity.v07.regression_policy') is not None,'native cautious regression policy is missing'
    return importlib.import_module('src.tasks.velocity.v07.regression_policy')

def step(module,state,update,successes,*,gu=None,substage='D20',phase='D',generation=73):
    payload,identity=validation_payload(update,generation,successes,phase,substage)
    return module.apply_validation(state,payload,expected_identity=identity,
        generation_updates=(update-4817 if gu is None else gu))

@pytest.mark.parametrize('substage,phase,target,outcome',[('D20','D','D22','regress'),('D22','D','D25','regress'),('L40','L','L20','regress'),('L20','L','L12','regress'),('D25','D',None,'stop_lowest'),('L12','L',None,'stop_lowest')])
def test_only_third_adjacent_post_protection_bad_validation_can_regress(substage,phase,target,outcome):
    p=policy();state=p.new_state(73)
    for update in [5900,6000]:
        state,d=step(p,state,update,0,substage=substage,phase=phase)
        assert d['outcome']=='continue'
    state,d=step(p,state,6100,0,substage=substage,phase=phase)
    assert d['outcome']==outcome and d['target_substage']==target and d['success_counts']==[0,0,0] and d['improvement']==0

@pytest.mark.parametrize('counts,want',[( [12,12,12],'regress'),([0,3,6],'continue'),([0,6,12],'continue'),([0,13,0],'continue'),([20,25,30],'continue')])
def test_literal_success_examples_follow_the_sds(counts,want):
    p=policy();state=p.new_state(73)
    for update,count in zip([5900,6000,6100],counts):state,d=step(p,state,update,count)
    assert d['outcome']==want

def test_generation_update_floor_not_global_or_phase_spend():
    p=policy();state=p.new_state(73)
    for update in [5600,5700,5800]:
        state,d=step(p,state,update,0,gu=999)
        assert d['outcome']=='protected' and state['low_success_evaluations']==[]
    for update,gu in [(5900,1000),(6000,1100),(6100,1200)]:state,d=step(p,state,update,0,gu=gu)
    assert d['outcome']=='regress'

def test_improvement_keeps_rolling_last_three_instead_of_clearing_history():
    p=policy();state=p.new_state(73)
    for update,count in zip([5900,6000,6100],[0,3,6]):state,d=step(p,state,update,count)
    assert len(state['low_success_evaluations'])==3 and d['improvement']==6
    state,d=step(p,state,6200,6)
    assert d['outcome']=='regress' and d['success_counts']==[3,6,6]

def test_gap_breaks_low_success_adjacency():
    p=policy();state=p.new_state(73)
    for update in [5900,6000,6200]:state,d=step(p,state,update,0)
    assert d['outcome']=='continue' and len(state['low_success_evaluations'])==1

def test_replayed_current_transaction_never_consumes_bad_evaluation_twice():
    p=policy();state=p.new_state(73);payload,identity=validation_payload()
    once,d=p.apply_validation(state,payload,expected_identity=identity,generation_updates=1083)
    again,replay=p.apply_validation(once,payload,expected_identity=identity,generation_updates=1083)
    assert once==again and replay['duplicate'] is True and len(again['low_success_evaluations'])==1
    other,_=validation_payload(successes=1)
    with pytest.raises(p.PolicyEvidenceError):p.apply_validation(once,other,expected_identity=identity,generation_updates=1083)

@pytest.mark.parametrize('mutation',['missing_first','wrong_env','wrong_tile','wrong_model','missing_case','wrong_hash','old_generation','diagnostic','retention'])
def test_incomplete_or_mismatched_actual_evidence_is_never_counted_as_zero(mutation):
    p=policy();state=p.new_state(73);payload,identity=validation_payload()
    if mutation=='missing_first':del payload['records'][0]['first_episode'];rehash_records(payload)
    elif mutation=='wrong_env':payload['records'][0]['first_episode']['env_id']=7;rehash_records(payload)
    elif mutation=='wrong_tile':payload['records'][0]['first_episode']['tile_id']=17;rehash_records(payload)
    elif mutation=='wrong_model':identity=deepcopy(identity);identity['model_sha256']='0'*64
    elif mutation=='missing_case':payload['records'].pop();rehash_records(payload)
    elif mutation=='wrong_hash':payload['evidence_sha256']['records_sha256']='0'*64
    elif mutation=='old_generation':identity=deepcopy(identity);identity['generation']=74
    elif mutation=='diagnostic':payload,identity=validation_payload(kind='diagnostic')
    elif mutation=='retention':payload,identity=validation_payload(kind='stair_retention')
    with pytest.raises(p.PolicyEvidenceError):p.apply_validation(state,payload,expected_identity=identity,generation_updates=1083)
    assert state==p.new_state(73)


def test_a_validation_between_formal_global_boundaries_is_not_a_policy_observation():
    p=policy();state=p.new_state(73);payload,identity=validation_payload(update=5925)
    with pytest.raises(p.PolicyEvidenceError):p.apply_validation(state,payload,expected_identity=identity,generation_updates=1108)

@pytest.mark.parametrize('field',['table_hash','decision_identity'])
def test_persisted_last_receipt_is_cross_checked_even_during_protection(field):
    p=policy();state,_=step(p,p.new_state(73),5800,0,gu=983)
    if field=='table_hash':state['last_applied']['case_table_sha256']='invalid'
    else:state['last_decision']=dict(outcome='regress',identity=dict(state['last_applied'],substage='D25'))
    with pytest.raises(p.PolicyEvidenceError):p.validate_state(state)


def test_b03_r1_one_bad_receipt_cannot_replay_a_forged_regression_action():
    p=policy();payload,identity=validation_payload()
    state,decision=p.apply_validation(p.new_state(73),payload,expected_identity=identity,generation_updates=1083)
    assert decision['outcome']=='continue' and len(state['low_success_evaluations'])==1
    bad=deepcopy(state)
    bad['last_decision']=dict(outcome='regress',reason='persistent_low_validation_success',
        target_substage='D25',success_counts=[0,0,0],improvement=0,total_successes=0,
        per_speed_successes=[0,0,0],generation_updates=1083,identity=deepcopy(state['last_applied']),duplicate=False)
    with pytest.raises(p.PolicyEvidenceError):p.validate_state(bad)
    with pytest.raises(p.PolicyEvidenceError):p.apply_validation(bad,payload,expected_identity=identity,generation_updates=1083)


def test_b03_r1_queue_end_must_bind_all_transaction_fields_not_just_update():
    p=policy();state,_=step(p,p.new_state(73),5900,0)
    bad=deepcopy(state);bad['last_applied'].update(generation=72,phase='L',substage='L40')
    if 'last_decision' in bad:bad['last_decision']['identity']=deepcopy(bad['last_applied'])
    with pytest.raises(p.PolicyEvidenceError):p.validate_state(bad)


def test_consumed_regression_replay_is_no_action_and_legal_generation_reset_keeps_receipt():
    p=policy();state=p.new_state(73)
    for update in [5900,6000,6100]:state,decision=step(p,state,update,0)
    assert decision['outcome']=='regress'
    payload,identity=validation_payload(6100)
    same,replay=p.apply_validation(state,payload,expected_identity=identity,generation_updates=1283)
    assert same==state and replay['duplicate'] is True and replay['outcome']=='continue' and replay['target_substage'] is None
    migrated=p.reset_generation(state,74)
    assert p.validate_state(migrated,generation=74) and migrated['low_success_evaluations']==[]
    assert migrated['last_applied']==state['last_applied']
    with pytest.raises(p.PolicyEvidenceError):p.apply_validation(migrated,payload,expected_identity=identity,generation_updates=1283)
