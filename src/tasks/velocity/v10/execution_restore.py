"""Explicit configurable-N migration; default checkpoint identity stays strict."""
from pathlib import Path
from . import spec
from .identity import code_fingerprint
from .checkpoint import SCHEMA,sha256
from .teachers import teacher_identity


def checked_profile(value,**kwargs):
    from .dynamic_execution import validate_execution_profile
    path=value.get('path') if isinstance(value,dict) else value
    if not path:raise ValueError('explicit execution profile path required')
    checked=validate_execution_profile(path,**kwargs)
    if isinstance(value,dict) and checked['sha256']!=value.get('sha256'):raise ValueError('execution profile changed')
    return checked


def provenance(profile):
    return dict(path=profile['path'],sha256=profile['sha256'],num_envs=profile['num_envs'],origin_checkpoint=profile['origin_checkpoint'])


def authorize_checkpoint(state,path,profile,*,num_envs,physical_sha256):
    if (num_envs!=profile['num_envs'] or physical_sha256!=profile['document']['physical_sha256'] or state.get('schema')!=SCHEMA
        or state.get('sds_sha256')!=spec.SDS_SHA256 or state.get('recipe')!=spec.RECIPE or state.get('teachers')!=teacher_identity()
        or state.get('physical_sha256')!=physical_sha256):raise ValueError('execution migration checkpoint identity differs')
    origin=profile['origin_checkpoint']
    if origin is not None and Path(path).resolve()==Path(origin['path']).resolve():
        if (sha256(path)!=origin['sha256'] or state.get('num_envs')!=origin['from_num_envs']
            or state.get('code_sha256')!=origin['from_code_sha256']):raise ValueError('exact execution origin checkpoint required')
        if any(state.get('rounds',{}).get(k,0)!=r for k,r in profile['document']['origin_rounds'].items()):raise ValueError('origin retained rounds differ')
    elif (state.get('num_envs')!=num_envs or state.get('code_sha256')!=code_fingerprint()
          or state.get('extra',{}).get('execution_profile')!=provenance(profile)):
        raise ValueError('same-profile current-source continuation required')
    if any(k not in state for k in ('model','learner','reservoir','rng','env','budgets','rounds')):raise ValueError('complete migration state required')


def bind_profile(state,profile):
    return dict(state.get('extra',{}),execution_profile=provenance(profile))


def validate_admission(profile,snapshot):
    if (snapshot.get('teacher_admitted') is not True or snapshot.get('execution_profile_sha256')!=profile['sha256']
        or not isinstance(snapshot.get('worker_operation_id'),str) or not snapshot['worker_operation_id']):
        raise ValueError('dynamic execution requires matching admitted controller operation')
    from .handoff import _teachers
    _teachers(snapshot.get('execution_teacher_receipts',{}),dict(document={'physical_sha256':profile['document']['physical_sha256']},compatibility={'to_code_sha256':code_fingerprint()}))
