"""Frozen case manifests with actual (not just seed) initial-state differences."""
import hashlib
import json
import math
import random
from . import spec


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def groups(course):
    if course not in ('D1','D2','D3','PPO'):
        raise ValueError(course)
    widths=(1.2,4.) if course=='D1' else (.8,1.2,4.)
    result=[]
    for task in spec.ROLES:
        if task=='F_STAND':
            variants=[dict(task=task,speed=0.,stop_mode='none')]
        elif task.startswith('F_'):
            variants=[dict(task=task,speed=v,stop_mode=m) for v in spec.SPEEDS
                      for m in (('instant','ramp_2s') if task=='F_STOP' else ('none',))]
        else:
            variants=[]
            for h in ((2.,4.,8.) if task.startswith('P_') else (10,20,40)):
                for w in widths:
                    for encoding in ('G','U'):
                        for v in spec.SPEEDS:
                            modes=(('instant','ramp_2s') if task=='P_STOP' else
                                   ('walk','instant','ramp_2s') if task=='S_ROUTE' and course in ('D3','PPO') else
                                   ('top',) if task=='S_ROUTE' else ('none',))
                            for mode in modes:
                                variants.append(dict(task=task,speed=v,stop_mode=mode,width=w,encoding=encoding,
                                                     **({'height':h} if task.startswith('P_') else {'stairs':h})))
        for row in variants:
            row['group_id']=digest(row)
            result.append(row)
    return result


def max_steps(row):
    task=row['task']; v=row['speed']; mode=row['stop_mode']
    decel=2. if mode=='ramp_2s' else 0.
    if task in ('F_WALK','F_STAND'):
        seconds=20.02
    elif task=='F_STOP':
        seconds=33.02+decel
    elif task=='P_WALK':
        seconds=10./v+10.
    elif task=='P_STOP':
        seconds=2*(5./v+10.)+20.+decel
    else:
        seconds=(.2*row['stairs']+2.1)/v+15.
        if mode=='walk': seconds += 10./v+10.
        elif mode in ('instant','ramp_2s'): seconds += 2*(5./v+10.)+20.+decel
    return math.ceil(seconds/spec.DT)


def suite(course,name,replicas):
    if name not in ('A','B','DAGGER_FINAL','PPO_FINAL','TRAIN'):
        raise ValueError(name)
    if replicas <= 0:
        raise ValueError(replicas)
    result=[]
    for group in groups(course):
        for index in range(replicas):
            identity=dict(course=course,suite=name,group=group['group_id'],index=index)
            seed=int(digest(identity)[:16],16)
            rng=random.Random(seed)
            row=dict(group,**identity,seed=seed,
                     spawn_x=(-1. if group['task']=='S_ROUTE' else 1.)+rng.uniform(-.1,.1) if group['task']=='S_ROUTE' else 1.+rng.uniform(-.05,.05),
                     initial_joint_offsets=[rng.uniform(-.01,.01) for _ in range(12)],
                     spawn_y=0.,yaw=0.,roll=0.,pitch=0.,initial_linear_velocity=[0.,0.,0.],
                     initial_angular_velocity=[0.,0.,0.],initial_joint_velocity=[0.]*12,clearance=.4)
            row['max_steps']=max_steps(row)
            row['case_sha256']=digest(row)
            result.append(row)
    return result


def evaluation_plan():
    """All scheduled runs at zero cache hits; each entry is a bounded suite invocation."""
    runs=[]
    # D1/D2 admission covers isolated stair endpoints, D3 admission complete routes.
    for course in ('D1','D2','D3'):
        for ab in ('A','B'):
            runs.append(dict(kind='teacher_admission',course=course,suite=ab,model='teachers',replicas=20))
        for gate in (.4,.8,1.):
            for ab in ('A','B'):
                for model in ('candidate','teachers'):
                    runs.append(dict(kind='dagger_gate',course=course,fraction=gate,suite=ab,model=model,replicas=20))
    for model in ('candidate','teachers'):
        runs.append(dict(kind='dagger_final',course='D3',suite='DAGGER_FINAL',model=model,replicas=100))
    for gate in (.25,.5,.75,1.):
        for ab in ('A','B'):
            for model in ('candidate','teachers','dagger'):
                runs.append(dict(kind='ppo_gate',course='PPO',fraction=gate,suite=ab,model=model,replicas=20))
    for model in ('candidate','teachers','dagger'):
        runs.append(dict(kind='ppo_final',course='PPO',suite='PPO_FINAL',model=model,replicas=100))
    totals={}
    for run in runs:
        key=(run['course'],run['suite'],run['replicas'])
        if key not in totals:
            totals[key]=sum(c['max_steps'] for c in suite(*key))
        run['max_samples']=totals[key]
    maximum=sum(r['max_samples'] for r in runs)
    seen=set(); cached=0
    for r in runs:
        key=(r['course'],r['suite'],r['model'])
        if r['model']!='candidate' and key in seen: cached+=r['max_samples']
        seen.add(key)
    return dict(runs=runs,max_samples=maximum,expected_cache_saving=cached,
                learning_samples=sum(spec.BUDGETS.values()),probe_separate=True)
