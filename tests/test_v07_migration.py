"""Only the exact byte-attested stop4817 can enter native schema9 once."""
from copy import deepcopy
import importlib,importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import a2_v07_bootstrap
a2_v07_bootstrap.activate(ROOT)

def migration():
    assert importlib.util.find_spec('src.tasks.velocity.v07.continuation_v07') is not None,'unique native migration is missing'
    return importlib.import_module('src.tasks.velocity.v07.continuation_v07')

def test_exact_source_keeps_all_learning_rng_and_adam_but_enters_generation73_once():
    m=migration();old=m.load_designated_source(ROOT/m.SOURCE_PATH)
    before=m.tree_digest(old);new,diff=m.migrate_stop_4817(old,source_sha256=m.SOURCE_SHA256)
    assert m.tree_digest(old)==before
    assert new['schema']==9 and new['identity']['recipe']=='0.7-d22-eval-regression-r1'
    for name in ['model','optimizer','normalizer','learning','rng']:
        assert m.tree_digest(new[name])==m.tree_digest(old[name])
    assert new['curriculum']['generation']==73 and new['curriculum']['generation_updates']==0
    assert new['curriculum']['spent_by_substage']['D22']==0
    assert new['curriculum']['spent_by_substage']['D25']==2083 and new['curriculum']['spent_by_substage']['D20']==34
    assert new['regression_policy_state']['generation']==73 and new['regression_policy_state']['low_success_evaluations']==[]
    assert new['pending_transition'] is None and new['ledger']['stop_state'] is None
    assert new['curriculum']['retention_next_due']==dict(flat=5000,h00=5000)
    assert diff['first_new_update']==4818
    with pytest.raises(m.recovery.RecoveryBlocked):m.migrate_stop_4817(new,source_sha256=m.SOURCE_SHA256)

@pytest.mark.parametrize('change',['sha','stop','partial','adam_id','rng','metadata','counter'])
def test_other_content_or_termination_is_refused(change):
    m=migration();state=m.load_designated_source(ROOT/m.SOURCE_PATH);digest=m.tree_digest(state);sha=m.SOURCE_SHA256
    if change=='sha':sha='0'*64
    elif change=='stop':state['ledger']['stop_state']='user_requested_stop'
    elif change=='partial':state['ledger']['pending_update']=1
    elif change=='adam_id':
        ids=state['optimizer']['param_groups'][0]['params'];ids[0],ids[1]=ids[1],ids[0]
    elif change=='rng':state['rng']['torch'][0]^=1
    elif change=='metadata':state['initialization']['transfer_metadata']['sha256']='0'*64
    elif change=='counter':state['learning']['adam_steps']+=1
    with pytest.raises(m.recovery.RecoveryBlocked):m.migrate_stop_4817(state,source_sha256=sha)

def test_old_c1_evaluation_tables_remain_parent_history_without_native_qualification():
    m=migration();old=m.load_designated_source(ROOT/m.SOURCE_PATH)
    new,_=m.migrate_stop_4817(old,source_sha256=m.SOURCE_SHA256)
    assert new['evaluation']['cases']=={} and new['evaluation']['latest'] is None
    assert new['gate']['records']==[] and new['gate']['streak']==0
    assert new['source']['continuation']['parent']['historical_evaluation']==old['evaluation']
    assert new['source']['continuation']['parent']['historical_gate']==old['gate']
