"""Native curriculum commits survive save failure without action/PPO replay."""
from copy import deepcopy
import pytest
from v07_runner_support import *
from v07_test_support import validation_payload
from test_v07_curriculum import outcomes
from src.tasks.velocity.v07 import regression_policy,evaluation,evaluation_records


def boundary_runner(tmp_path,substage='D20'):
    cp=migrated_checkpoint(tmp_path/'source.pt');r=loaded_runner(tmp_path/'run',cp)
    set_synthetic_boundary(r,6100,substage=substage)
    for update in (5900,6000):
        payload,identity=validation_payload(update,73,0,substage=substage)
        r.curriculum.regression_policy_state,_=regression_policy.apply_validation(r.curriculum.regression_policy_state,payload,expected_identity=identity,generation_updates=update-4817)
    payload,_=validation_payload(6100,73,0,substage=substage)
    table=evaluation.table_from_json(payload['table'])
    payload['identity']=r._evaluation_identity(table,6100)
    binding=evaluation_records.write_sidecar(tmp_path/'validation.json',payload)
    task=dict(kind='validation',gate_kind='validation',retention_kind=None,tile=table.cases[0]['tile_id'],phase='D',substage=substage,case_table_sha256=table.sha256)
    pending=dict(update=6100,phase='D',substage=substage,generation=73,tasks=[task],results=[],curriculum_decided=False,committed=True,
        validation_result=dict(payload=payload,case_records=binding))
    pending['results']=[r._result_id(6100,task)];r.pending_boundary=pending
    return r,pending

def test_regression_commit_failure_bare_reload_finishes_same_poststate_once(tmp_path,monkeypatch):
    r,pending=boundary_runner(tmp_path);before_learning=deepcopy(r.learning)
    events=r._finalize_curriculum(pending,None)
    assert r.substage=='D22' and r.curriculum.state.generation==74
    assert r.curriculum_transaction['stage']=='applied'
    def fail(*args,**kwargs):raise OSError('injected curriculum checkpoint write failure')
    with monkeypatch.context() as patch:
        patch.setattr(r,'_save_evaluation_commit',fail)
        with pytest.raises(OSError):r.run(max_updates=6100)
    path=r.directory/'checkpoints'/'exception_006100.pt'
    cp=dict(path=str(path),sha256=recovery.sha256(path));r2=loaded_runner(tmp_path/'resume',cp)
    def forbidden(*args,**kwargs):raise AssertionError('committed curriculum/validation replayed')
    monkeypatch.setattr(r2.curriculum,'on_update',forbidden);r2.case_runner=forbidden
    assert r2._complete_pending_boundary()==6100
    assert r2.substage=='D22' and r2.curriculum.state.generation==74 and r2.curriculum.state.generation_updates==0
    assert r2.learning==before_learning and r2.curriculum_transaction is None and r2.pending_boundary is None
    assert len([e for e in r2.transitions if e['kind']=='substage_regress' and e.get('update')==6100])==1
    assert r2.curriculum.regression_policy_state['generation']==74
    again=loaded_runner(tmp_path/'again',r2.save('roundtrip'))
    assert again._complete_pending_boundary() is None and again.curriculum.state.generation==74

def test_window_write_failure_bare_reload_consumes_same64_once_without_rebilling(tmp_path,monkeypatch):
    cp=migrated_checkpoint(tmp_path/'source.pt');r=loaded_runner(tmp_path/'run',cp)
    set_synthetic_boundary(r,4818,substage='D25',gu=1)
    r.curriculum.observe(outcomes(56));before=deepcopy(r.curriculum.state_dict())
    pending=dict(update=4818,phase='D',substage='D25',generation=73,tasks=[],results=[],curriculum_decided=False,committed=True)
    r.pending_boundary=pending
    from src.tasks.velocity.v07 import window_evidence
    original=window_evidence.write_window
    def io_fail(path,payload):original(path,payload);raise OSError('after raw64 installed')
    with monkeypatch.context() as patch:
        patch.setattr(window_evidence,'write_window',io_fail)
        with pytest.raises(OSError):r._finalize_curriculum(pending,None)
    assert r.curriculum.state_dict()==before and r.curriculum_transaction['stage']=='prepared'
    cp=r.save('exception');r2=loaded_runner(tmp_path/'resume',cp)
    assert r2._complete_pending_boundary()==4818
    assert len(r2.curriculum.state.episode_fifo)==0 and r2.curriculum.state.ema['batches']==1
    assert r2.curriculum.state.ema['f']==pytest.approx(56/64)
    assert r2.learning['update']==4818 and r2.learning['adam_steps']==96360
    assert len(list((tmp_path/'run'/'frontier_windows').glob('*.json')))==1
    assert len(list((tmp_path/'resume'/'frontier_windows').glob('*.json')))==1

def test_unproven_duplicate_or_corrupt_applied_transaction_blocks_learning_and_restore(tmp_path):
    r,pending=boundary_runner(tmp_path);r._finalize_curriculum(pending,None)
    with pytest.raises(recovery.RecoveryBlocked):r._check_resume_evidence(stage='learning')
    r.curriculum_transaction['post_state']['substage']='D20'
    cp=r.save('corrupt_evidence')
    with pytest.raises(recovery.RecoveryBlocked):loaded_runner(tmp_path/'refused',cp)


def test_lowest_three_bad_validation_stops_with_new_reason_and_complete_evidence(tmp_path):
    r,pending=boundary_runner(tmp_path,substage='D25')
    events=r._finalize_curriculum(pending,None)
    assert r.curriculum.state.finished and r.curriculum.state.stop_status=='persistent_low_validation_success'
    r._commit_curriculum_transaction(pending,events)
    assert r.stop_state=='curriculum_stalled' and r.stop_reason=='persistent_low_validation_success'
    assert len(r.stop_evidence['event']['detail']['validations'])==3
    assert r.learning['update']==6100 and r.learning['adam_steps']==122000
    stop=r.directory/'checkpoints'/'stop_006100.pt'
    with pytest.raises(recovery.RecoveryBlocked):recovery.preload_checkpoint(stop,mode='resume')

def test_transaction_preload_requires_exact_checkpoint_sampler_before_allowed_new_episode(tmp_path):
    r,pending=boundary_runner(tmp_path);r._finalize_curriculum(pending,None)
    state=r.checkpoint_state()
    sampler=state['curriculum']['sampler_rng_state']
    sampler['state']['state']+=1
    path=tmp_path/'wrong_sampler.pt';torch.save(state,path)
    with pytest.raises(recovery.RecoveryBlocked):loaded_runner(tmp_path/'refused_sampler',dict(path=str(path),sha256=recovery.sha256(path)))

@pytest.mark.parametrize('stop',['behavior_regression','behavior_collapse','user_requested_stop'])
def test_independent_stop_wins_before_cautious_curriculum_decision(tmp_path,stop):
    r,pending=boundary_runner(tmp_path)
    before=deepcopy(r.curriculum.state_dict());policy=deepcopy(r.curriculum.regression_policy_state)
    if stop=='user_requested_stop':r.stop_flag=True
    else:r._stop(stop,dict(reason='independent evaluation trigger'))
    r._finalize_curriculum(pending,None)
    assert r.stop_state==stop
    assert r.curriculum.state_dict()==before and r.curriculum.regression_policy_state==policy
    assert r.substage=='D20' and r.curriculum.state.generation==73

def test_real_evaluation_handoff_write_failure_automatic_exception_load_does_not_re_evaluate(tmp_path,monkeypatch):
    r,pending=boundary_runner(tmp_path)
    template=deepcopy(pending.pop('validation_result')['payload']);pending['results']=[]
    rows={row['case_id']:row for row in template['records']};calls=[]
    def cases(case):calls.append(case['case_id']);return deepcopy(rows[case['case_id']])
    r.case_runner=cases
    original=evaluation_records.write_sidecar
    def fail_after_install(path,payload):original(path,payload);raise OSError('after actual60 sidecar install')
    with monkeypatch.context() as patch:
        patch.setattr(evaluation_records,'write_sidecar',fail_after_install)
        with pytest.raises(OSError):r.run(max_updates=6100)
    assert len(calls)==60 and r.evaluation_transaction['stage']=='evaluated'
    path=r.directory/'checkpoints'/'exception_006100.pt'
    cp=dict(path=str(path),sha256=recovery.sha256(path));r2=loaded_runner(tmp_path/'resumed60',cp)
    def forbidden(*args,**kwargs):raise AssertionError('actual60 evaluated twice')
    r2.case_runner=forbidden
    assert r2._complete_pending_boundary()==6100
    assert len(calls)==60 and r2.substage=='D22' and r2.curriculum.state.generation==74
    assert r2.learning['update']==6100 and r2.learning['adam_steps']==122000
    assert r2.curriculum.regression_policy_state['low_success_evaluations']==[]
    assert r2.pending_boundary is None and r2.evaluation_transaction is None

def test_cached_valid_window_cannot_be_substituted_for_this_fifo_consumption(tmp_path):
    from src.tasks.velocity.v07 import window_evidence
    from src.tasks.velocity.v07.runner import _digest
    cp=migrated_checkpoint(tmp_path/'source.pt');r=loaded_runner(tmp_path/'run',cp)
    set_synthetic_boundary(r,4818,substage='D25',gu=1);r.curriculum.observe(outcomes(56))
    pending=dict(update=4818,phase='D',substage='D25',generation=73,tasks=[],results=[],curriculum_decided=False,committed=True)
    r.pending_boundary=pending;r._finalize_curriculum(pending,None)
    tx=r.curriculum_transaction;window=tx['windows'][0]
    window['payload']['identity']['update']=4819
    window['binding']=window_evidence.write_window(tmp_path/'other_valid_window.json',window['payload'])
    tx['post_sha256']=_digest(dict(state=tx['post_state'],policy=tx['post_policy'],events=tx['events'],windows=tx['windows']))
    saved=r.save('substituted_window')
    with pytest.raises(recovery.RecoveryBlocked):loaded_runner(tmp_path/'refused',saved)

# B CORE R1: consumption cannot disappear merely by dropping its proof list.
def r1_window_boundary(tmp_path,*,advance=False,independent=False,with_rows=True):
    cp=migrated_checkpoint(tmp_path/'source.pt');r=loaded_runner(tmp_path/'run',cp)
    set_synthetic_boundary(r,5000,substage='D25',gu=183)
    if with_rows:r.curriculum.observe(outcomes(56))
    pending=dict(update=5000,phase='D',substage='D25',generation=73,tasks=[],results=[],curriculum_decided=False,committed=True)
    if advance:
        payload,_=validation_payload(5000,73,60,substage='D25')
        table=evaluation.table_from_json(payload['table']);payload['identity']=r._evaluation_identity(table,5000)
        binding=evaluation_records.write_sidecar(tmp_path/'current_validation.json',payload)
        task=dict(kind='validation',gate_kind='validation',retention_kind=None,tile=17,phase='D',substage='D25',case_table_sha256=table.sha256)
        pending.update(tasks=[task],results=[r._result_id(5000,task)],validation_result=dict(payload=payload,case_records=binding))
        r.curriculum.state.gate_streak=1
    r.pending_boundary=pending
    if independent:r._stop('user_requested_stop',dict(reason='independent stop before window consumption'))
    r._finalize_curriculum(pending,None)
    return r

def r1_rehash_post(tx):
    from src.tasks.velocity.v07.runner import _digest
    tx['post_sha256']=_digest(dict(state=tx['post_state'],policy=tx['post_policy'],events=tx['events'],windows=tx['windows']))

@pytest.mark.parametrize('advance',[False,True])
@pytest.mark.parametrize('omit_before',[False,True])
def test_r1_consumed_window_is_required_even_if_post_hash_is_recomputed(tmp_path,advance,omit_before):
    r=r1_window_boundary(tmp_path,advance=advance)
    assert r._validate_curriculum_transaction()['stage']=='applied'
    tx=r.curriculum_transaction;window=tx['windows'][0]
    Path(window['binding']['path']).unlink();tx['windows']=[]
    if omit_before:tx['before_state']['episode_fifo']=[]
    r1_rehash_post(tx)
    with pytest.raises(recovery.RecoveryBlocked):r._validate_curriculum_transaction()


def test_r1_no_raw_fifo_no_window_cannot_prove_ema_consumption_even_with_rehashed_before(tmp_path):
    from src.tasks.velocity.v07.runner import _digest
    r=r1_window_boundary(tmp_path);tx=r.curriculum_transaction
    Path(tx['windows'][0]['binding']['path']).unlink();tx['windows']=[];tx['before_state']['episode_fifo']=[]
    tx['before_sha256']=_digest(dict(state=tx['before_state'],policy=tx['before_policy']))
    r1_rehash_post(tx)
    with pytest.raises(recovery.RecoveryBlocked):r._validate_curriculum_transaction()

@pytest.mark.parametrize('kind',['none','consumed','generation_reset','independent_stop'])
def test_r1_legal_window_counts_generation_reset_and_independent_stop_remain_valid(tmp_path,kind):
    r=r1_window_boundary(tmp_path,advance=kind=='generation_reset',independent=kind=='independent_stop',with_rows=kind!='none')
    tx=r._validate_curriculum_transaction()
    assert len(tx['windows'])==(0 if kind in ('none','independent_stop') else 1)
    if kind=='generation_reset':assert tx['post_state']['generation']==74 and tx['post_state']['ema']['batches']==0
