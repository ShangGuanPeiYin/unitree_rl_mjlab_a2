"""Native v07 cautious D/L decisions from one verified, complete validation.

Pure state transition: no model execution, disk lookup, RNG or curriculum ticks.
The caller supplies its current model/source/task identity and persists both the
actual result and the returned state before another learning update.
"""
from copy import deepcopy
from . import spec,evaluation,evaluation_records

class PolicyEvidenceError(ValueError):
    """A policy input or persisted state is incomplete or bound elsewhere."""

_STATE_KEYS={'version','generation','low_success_evaluations','last_applied','last_fingerprint'}
_ID_KEYS=('update','generation','phase','substage','kind','case_table_sha256')
_HASH_KEYS=('case_table_sha256','model_sha256','optimizer_sha256','source_closure_sha256','contract_sha256')
_OUTCOMES={'protected','continue','regress','stop_lowest'}

def _integer(value,label,minimum=0):
    if type(value) is not int or value<minimum:raise PolicyEvidenceError(label+' must be an integer >= '+str(minimum))
    return value

def _sha(value,label):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise PolicyEvidenceError(label+' must be SHA256')
    return value

def transaction_key(identity):
    if not isinstance(identity,dict) or any(k not in identity for k in _ID_KEYS):
        raise PolicyEvidenceError('incomplete validation transaction identity')
    for key in ('update','generation'):_integer(identity[key],key)
    _sha(identity['case_table_sha256'],'transaction table SHA')
    return {key:deepcopy(identity[key]) for key in _ID_KEYS}

def new_state(generation):
    return dict(version=spec.REGRESSION_POLICY_CONFIG['version'],generation=_integer(generation,'generation'),
        low_success_evaluations=[],last_applied=None,last_fingerprint=None)

def reset_generation(state,generation):
    """Clear this generation's queue; retain the last applied receipt as evidence."""
    validate_state(state)
    result=new_state(generation)
    for key in ('last_applied','last_fingerprint'):result[key]=deepcopy(state[key])
    return result

def validate_state(state,*,generation=None):
    cfg=spec.REGRESSION_POLICY_CONFIG
    if not isinstance(state,dict) or set(state)!=_STATE_KEYS or state.get('version')!=cfg['version']:
        raise PolicyEvidenceError('unknown or incomplete regression_policy_state')
    gen=_integer(state['generation'],'policy generation')
    if generation is not None and gen!=int(generation):raise PolicyEvidenceError('policy generation differs from curriculum')
    queue=state['low_success_evaluations']
    if not isinstance(queue,list) or len(queue)>cfg['history_size']:raise PolicyEvidenceError('invalid low-success queue size')
    previous=None
    for entry in queue:
        if not isinstance(entry,dict) or set(entry)!={'identity','evidence_sha256','per_speed_successes','total_successes','generation_updates'}:
            raise PolicyEvidenceError('incomplete low-success entry')
        identity=entry['identity'];key=transaction_key(identity)
        if key['generation']!=gen or key['kind']!='validation' or key['phase'] not in ('D','L'):
            raise PolicyEvidenceError('queue contains a different generation or evaluation kind')
        order=spec.D_SUBSTAGES if key['phase']=='D' else spec.L_SUBSTAGES
        if key['substage'] not in order:raise PolicyEvidenceError('queue substage is invalid')
        for name in _HASH_KEYS:_sha(identity.get(name),name)
        ev=entry['evidence_sha256']
        if not isinstance(ev,dict) or set(ev)!={'case_table_sha256','records_sha256','summary_sha256'}:raise PolicyEvidenceError('queue evidence hashes missing')
        for name,value in ev.items():_sha(value,name)
        if ev['case_table_sha256']!=key['case_table_sha256']:raise PolicyEvidenceError('queue table SHA mismatch')
        counts=entry['per_speed_successes']
        if not isinstance(counts,list) or len(counts)!=len(cfg['speeds']):raise PolicyEvidenceError('queue speed counts missing')
        if any(type(x) is not int or not 0<=x<=cfg['per_speed_cases'] for x in counts):raise PolicyEvidenceError('queue speed count invalid')
        if type(entry['total_successes']) is not int or sum(counts)!=entry['total_successes'] or sum(counts)>cfg['low_success_max']:
            raise PolicyEvidenceError('queue does not contain low success')
        if _integer(entry['generation_updates'],'entry generation updates')<cfg['protection_updates']:raise PolicyEvidenceError('protected entry in low-success queue')
        if previous is not None and (key['update']-previous['update']!=cfg['validation_interval'] or any(key[n]!=previous[n] for n in ('phase','substage','generation','kind'))):
            raise PolicyEvidenceError('queue evaluations are not adjacent in one task')
        previous=key
    last=state['last_applied']
    if last is None:
        if state['last_fingerprint'] is not None or queue:raise PolicyEvidenceError('policy lacks its applied receipt')
    else:
        key=transaction_key(last)
        if key['generation']>gen:raise PolicyEvidenceError('last receipt is from a future generation')
        _sha(state['last_fingerprint'],'last fingerprint')
        if set(last)!=set(_ID_KEYS) or key['kind']!='validation' or key['phase'] not in ('D','L'):
            raise PolicyEvidenceError('last receipt is not an exact D/L validation identity')
        if key['substage'] not in (spec.D_SUBSTAGES if key['phase']=='D' else spec.L_SUBSTAGES):
            raise PolicyEvidenceError('last receipt substage is invalid')
        if queue and transaction_key(queue[-1]['identity'])!=key:
            raise PolicyEvidenceError('queue end and last receipt differ in task/generation/table identity')
    return True

def _validated_result(payload,identity):
    cfg=spec.REGRESSION_POLICY_CONFIG;key=transaction_key(identity)
    if key['update']%cfg['validation_interval']:
        raise PolicyEvidenceError('validation is not at a formal global evaluation boundary')
    if key['phase'] not in ('D','L') or key['kind']!='validation':raise PolicyEvidenceError('only current D/L validation enters cautious policy')
    if key['substage'] not in (spec.D_SUBSTAGES if key['phase']=='D' else spec.L_SUBSTAGES):raise PolicyEvidenceError('validation substage is invalid')
    for name in _HASH_KEYS:_sha(identity.get(name),name)
    try:
        result=evaluation_records.restore_result(payload,identity)
        if tuple(result.table.speeds)!=tuple(cfg['speeds']) or result.table.episodes!=cfg['per_speed_cases'] or len(result.records)!=len(cfg['speeds'])*cfg['per_speed_cases']:
            raise PolicyEvidenceError('policy requires exactly three speeds x20 actual cases')
        # Reuse the production first-episode validator, lazily after module import
        # to avoid a curriculum/runner import cycle. It does not run the policy.
        from .runner import _validate_first_case
        expected={case['case_id']:(case,index) for batch in evaluation.batch_cases(result.table,size=spec.EVALUATION_NUM_ENVS) for index,case in enumerate(batch)}
        for row in result.records:
            case,index=expected[row['case_id']];episode=row.get('first_episode')
            if not isinstance(episode,dict) or row.get('evidence_kind')!='terminal':raise PolicyEvidenceError('validation lacks its frozen first terminal')
            steps=_integer(row.get('steps'),'case steps',1)
            if row.get('first_observed_at_batch_step')!=steps or type(episode.get('env_id')) is not int or episode['env_id']!=index:
                raise PolicyEvidenceError('first terminal case/env/step association differs')
            _validate_first_case(episode,index=index,batch_step=steps,evidence_kind='terminal',flat=False)
            if episode.get('tile_id')!=case['tile_id'] or row.get('measurement')!='stair' or row['vx']!=case['vx']:
                raise PolicyEvidenceError('first terminal belongs to a different physical case')
            if (type(row.get('climb_success')) is not bool or type(row.get('failure')) is not bool
                or row['failure']!=episode['failure'] or row['climb_success']!=bool(episode['success'] and not episode['failure'])
                or row.get('reason')!=episode['reason']):raise PolicyEvidenceError('summary outcome differs from frozen terminal')
            actual=_integer(row.get('physical_control_samples'),'actual controls',steps)
            extra=_integer(row.get('extra_control_samples_after_case'),'extra controls')
            if extra!=actual-steps or row.get('physical_physics_substeps')!=actual*spec.DECIMATION or row.get('extra_physics_substeps_after_case')!=extra*spec.DECIMATION:
                raise PolicyEvidenceError('first-case and physical costs do not balance')
        counts=[int(x['case_pass_count']) for x in result.summary['per_speed']]
        return result,counts
    except (evaluation.EvaluationError,KeyError,TypeError,ValueError) as error:
        if isinstance(error,PolicyEvidenceError):raise
        raise PolicyEvidenceError('invalid complete validation: '+str(error)) from error

def apply_validation(state,payload,*,expected_identity,generation_updates):
    """Return (new_state, decision), with replay/conflict and full evidence checks."""
    validate_state(state);cfg=spec.REGRESSION_POLICY_CONFIG
    gu=_integer(generation_updates,'generation_updates')
    result,counts=_validated_result(payload,expected_identity)
    key=transaction_key(expected_identity);fingerprint=evaluation_records._digest(payload)
    if key['generation']!=state['generation']:raise PolicyEvidenceError('validation belongs to another current generation')
    if state['last_applied']==key:
        if state['last_fingerprint']!=fingerprint:raise PolicyEvidenceError('conflicting evidence for one consumed validation')
        # An already consumed validation is never an executable action cache.
        # The runner's separately anchored curriculum transaction must finish an
        # applied transition; replaying this pure policy cannot apply it twice.
        success=[x['total_successes'] for x in state['low_success_evaluations']]
        return deepcopy(state),dict(outcome='continue',reason='validation_already_applied',
            target_substage=None,success_counts=success,
            improvement=(success[-1]-success[0] if len(success)==cfg['history_size'] else None),
            total_successes=sum(counts),per_speed_successes=counts,
            generation_updates=gu,identity=deepcopy(key),duplicate=True)
    if state['last_applied'] is not None and key['update']<=state['last_applied']['update']:
        raise PolicyEvidenceError('validation is older than the last applied transaction')
    new=deepcopy(state);total=sum(counts);queue=new['low_success_evaluations']
    decision=dict(outcome='continue',reason='validation_not_severely_low',target_substage=None,
        success_counts=[],improvement=None,total_successes=total,per_speed_successes=counts,
        generation_updates=gu,identity=deepcopy(key),duplicate=False)
    if gu<cfg['protection_updates']:
        queue.clear();decision.update(outcome='protected',reason='generation_updates_below_protection')
    elif total>cfg['low_success_max']:
        queue.clear()
    else:
        if queue and (key['update']-queue[-1]['identity']['update']!=cfg['validation_interval'] or any(key[n]!=queue[-1]['identity'][n] for n in ('phase','substage','generation'))):queue.clear()
        queue.append(dict(identity=deepcopy(expected_identity),evidence_sha256=deepcopy(result.evidence_sha256),
            per_speed_successes=counts,total_successes=total,generation_updates=gu))
        new['low_success_evaluations']=queue=queue[-cfg['history_size']:]
        success=[x['total_successes'] for x in queue];decision.update(reason='waiting_for_three_adjacent_low_validations',success_counts=success)
        if len(queue)==cfg['history_size']:
            delta=success[-1]-success[0];decision['improvement']=delta
            if delta>=cfg['min_improvement']:decision['reason']='low_validation_success_improving'
            else:
                order=spec.D_SUBSTAGES if key['phase']=='D' else spec.L_SUBSTAGES;index=order.index(key['substage'])
                decision.update(outcome='regress' if index else 'stop_lowest',reason='persistent_low_validation_success',target_substage=order[index-1] if index else None)
    new.update(last_applied=deepcopy(key),last_fingerprint=fingerprint)
    validate_state(new)
    return new,decision
