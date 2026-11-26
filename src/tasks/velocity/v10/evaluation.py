"""Independent per-episode quality windows and paired per-group acceptance.

Trace rows are post-action physical measurements plus action-time phase/command.
No incomplete episode/window is dropped from the success denominator.
"""
import math
import numpy as np
from . import cases, spec


def _walk_window(rows):
    if len(rows)*spec.DT < 1.:
        return None
    velocity=np.asarray([r['v_body'] for r in rows])[:,:2]
    command=np.asarray([r['command'] for r in rows])
    pitch=np.asarray([r['pitch_deg'] for r in rows])
    return dict(rmse=float(np.sqrt(np.mean((velocity[:,0]-command)**2+velocity[:,1]**2))),
                pitch_rms=float(np.sqrt(np.mean(pitch**2))),pitch_p99=float(np.percentile(np.abs(pitch),99)))


def _stand_window(rows,origin=None):
    if len(rows) != 500:
        return None
    velocity=np.asarray([r['v_world'] for r in rows])[:,:2]
    pos=np.asarray([r['position'] for r in rows])[:,:2]
    speed=np.linalg.norm(velocity,axis=1)
    return dict(mean_speed=float(speed.mean()),peak_speed=float(speed.max()),
                drift=float(np.linalg.norm(pos-(pos[0] if origin is None else np.asarray(origin)[:2]),axis=1).max()),
                max_attitude=max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows),
                min_clearance=min(r['clearance'] for r in rows),
                support_fraction=float(np.mean([all(r['safe_support']) for r in rows])))


def _validate_row(r):
    numbers=[r['command'],*r['position'],*r['v_body'],*r['v_world'],
             r['roll_deg'],r['pitch_deg'],r['clearance']]
    if not np.isfinite(np.asarray(numbers,dtype=float)).all():
        raise FloatingPointError('nonfinite evaluation physical measurement')


def _endpoint_reasons(case,trace):
    if not trace: return ['missing_endpoint']
    segments=[]
    for r in trace:
        if not segments or segments[-1][0]['phase']!=r['phase']:segments.append([])
        segments[-1].append(r)
    task=case['task'];mode=case['stop_mode'];speed=case['speed'];reasons=[]
    stopping=task in ('F_STOP','P_STOP') or task=='S_ROUTE' and mode in ('instant','ramp_2s')
    expected=['STAND'] if task=='F_STAND' else ['CLIMB'] if task=='S_ROUTE' else ['WALK']
    if task=='S_ROUTE' and mode!='top':expected.append('POST_WALK')
    if stopping:expected+= (['DECEL'] if mode=='ramp_2s' else [])+['HOLD','RESTART']
    if [part[0]['phase'] for part in segments]!=expected:reasons.append('phase_order')
    previous_x=case.get('spawn_x',None)
    for part in segments:
        phase=part[0]['phase'];count=len(part)
        if phase in ('STAND','HOLD'):
            if count!=1000:reasons.append('hold_duration')
            commands=np.zeros(count)
        elif phase=='DECEL':
            if count!=100:reasons.append('decel_duration')
            commands=speed*(1-np.arange(count)/100)
        else:
            commands=np.full(count,speed)
        if not np.allclose([r['command'] for r in part],commands,rtol=0,atol=1e-6):reasons.append('command_timing')
        if task=='F_WALK' and phase=='WALK' and count!=1000:reasons.append('walk_duration')
        if task=='F_STOP' and ((phase=='WALK' and count!=400) or (phase=='RESTART' and count!=250)):reasons.append('flat_segment_duration')
        distance=None
        if task=='P_WALK' and phase=='WALK' or task=='S_ROUTE' and mode=='walk' and phase=='POST_WALK':distance=10.
        elif stopping and task in ('P_STOP','S_ROUTE') and (phase in ('POST_WALK','RESTART') or task=='P_STOP' and phase=='WALK'):distance=5.
        if distance is not None and (previous_x is None or part[-1]['position'][0]-previous_x<distance-1e-6):reasons.append('missing_net_distance')
        if phase=='CLIMB' and not part[-1].get('top_event',False):reasons.append('missing_top_endpoint')
        previous_x=part[-1]['position'][0]
    return reasons


def grade_case(case,trace,*,completed):
    for row in trace: _validate_row(row)
    reasons=_endpoint_reasons(case,trace) if completed else []; windows={}
    if not trace or not completed:
        reasons.append('incomplete_task')
    if any(r.get('failure',False) or r.get('timeout',False) for r in trace):
        reasons.append('terminal_failure')
    # Continuous pitch quality protection applies only to planar portions.
    bad=0
    for r in trace:
        planar=r.get('planar',r['phase'] not in ('CLIMB',))
        bad=bad+1 if planar and abs(r['pitch_deg'])>15 else 0
        if bad>=10: reasons.append('continuous_pitch_violation');break
    task=case['task']
    grouped={}
    # Split by contiguous phase, including repeated phases as separate segments.
    for r in trace:
        if not grouped or r['phase'] != list(grouped.values())[-1][0]['phase']:
            grouped[len(grouped)]=[]
        grouped[len(grouped)-1].append(r)
    walk_index=0; stand_index=0
    top_index=next((i for i,r in enumerate(trace) if r.get('top_event')),None)
    for segment in grouped.values():
        phase=segment[0]['phase']
        if phase in ('WALK','POST_WALK','RESTART'):
            rows=segment[100:] # exclude first two seconds except stair first platform segment
            if task=='S_ROUTE' and phase=='POST_WALK' and top_index is not None:
                top_x=trace[top_index]['position'][0]
                # Include transition causing top as the first physical state.
                segment=[trace[top_index]]+segment
                crossing=next((i for i,r in enumerate(segment) if r['position'][0]-top_x>=1.),None)
                if crossing is None:
                    reasons.append('missing_first_meter')
                    parts=[('first_meter',segment),('post_meter',[])]
                else:
                    parts=[('first_meter',segment[:crossing+1]),('post_meter',segment[crossing+1:])]
            else:
                parts=[(f'walk_{walk_index}',rows)]
            for label,part in parts:
                metric=_walk_window(part)
                if metric is None: reasons.append('missing_window:'+label)
                else:
                    windows[label]=metric
                    if metric['rmse']>.10 or metric['pitch_rms']>5 or metric['pitch_p99']>10:
                        reasons.append('walk_quality:'+label)
            walk_index+=1
        if phase in ('STAND','HOLD'):
            label=f'stand_{stand_index}';stand_index+=1
            if len(segment)!=1000: reasons.append('incomplete_hold:'+label)
            # First fully stable 25-frame block must finish by 3 seconds.
            streak=0; established=False
            for r in segment[:150]:
                streak=streak+1 if r['stable'] else 0
                if streak>=25: established=True;break
            if not established: reasons.append('late_stabilization:'+label)
            metric=_stand_window(segment[500:1000],origin=segment[499]['position'] if len(segment)>=500 else None)
            if metric is None: reasons.append('missing_window:'+label)
            else:
                windows[label]=metric
                if (metric['mean_speed']>.05 or metric['peak_speed']>.10 or metric['drift']>.10
                    or metric['max_attitude']>10 or metric['min_clearance']<.30 or metric['support_fraction']<.90):
                    reasons.append('stand_quality:'+label)
    required_walk=0 if task=='F_STAND' or task=='S_ROUTE' and case['stop_mode']=='top' else 2 if task in ('F_STOP','P_STOP') or task=='S_ROUTE' and case['stop_mode'] in ('instant','ramp_2s') else 1
    required_stand=1 if task in ('F_STAND','F_STOP','P_STOP') or task=='S_ROUTE' and case['stop_mode'] in ('instant','ramp_2s') else 0
    if walk_index<required_walk or stand_index<required_stand: reasons.append('missing_required_segment')
    if task=='S_ROUTE' and top_index is None: reasons.append('missing_top')
    expected_windows=(['first_meter','post_meter']+(['walk_1'] if required_walk==2 else [])
                      if task=='S_ROUTE' and required_walk else [f'walk_{i}' for i in range(required_walk)])
    expected_windows += [f'stand_{i}' for i in range(required_stand)]
    return dict(expected_windows=expected_windows,case_sha256=case['case_sha256'],group_id=case['group_id'],passed=not reasons,
                reasons=sorted(set(reasons)),windows=windows,steps=len(trace))


def grade_group(candidate,reference,*,final):
    n=100 if final else 20
    reasons=[]; minimum=95 if final else 18
    refs={r['case_sha256']:r for r in reference}
    for result in candidate+reference:
        for metric in result['windows'].values():
            if not all(math.isfinite(float(v)) for v in metric.values()):
                raise FloatingPointError('nonfinite evaluation window metric')
    if (len(candidate)!=n or len(reference)!=n or len(refs)!=n
        or len({r['case_sha256'] for r in candidate})!=n
        or {r['case_sha256'] for r in candidate}!=set(refs)):
        reasons.append('invalid_denominator')
    successes=sum(bool(r['passed']) for r in candidate)
    ref_success=sum(bool(r['passed']) for r in reference)
    if successes<minimum: reasons.append('absolute_success')
    if successes<ref_success-(2 if final else 1): reasons.append('relative_success')
    keys=set(k for r in reference+candidate for k in r['windows'])
    keys.update(k for r in reference+candidate for k in r.get('expected_windows',[]))
    coverage={}; deltas={}
    for key in keys:
        pairs=[(r['windows'][key],refs[r['case_sha256']]['windows'][key]) for r in candidate
               if r['case_sha256'] in refs and key in r['windows'] and key in refs[r['case_sha256']]['windows']]
        coverage[key]=len(pairs)/n
        if coverage[key] < (.95 if final else .9): reasons.append('pair_coverage:'+key)
        if not pairs: continue
        deltas[key]={}
        for field in ('rmse','pitch_rms','drift'):
            if field not in pairs[0][0]: continue
            c=float(np.mean([p[0][field] for p in pairs])); r=float(np.mean([p[1][field] for p in pairs]))
            allowed=max(r*.05,.02) if field=='rmse' else 1. if field=='pitch_rms' else .02
            deltas[key][field]=dict(candidate=c,reference=r,increase=c-r,allowed=allowed)
            if c-r>allowed+1e-12: reasons.append('relative_metric:'+key+':'+field)
    excluded=[r['case_sha256'] for r in candidate if r['case_sha256'] not in refs or set(r['windows'])!=set(refs[r['case_sha256']]['windows'])]
    return dict(passed=not reasons,reasons=reasons,successes=successes,reference_successes=ref_success,
                denominator=n,pair_coverage=coverage,deltas=deltas,excluded_cases=excluded)


class EpisodeMetrics:
    """Compact numeric trace, shared by terminal reward quality and acceptance.

    Stores no tensors/computation graph. The public grade_case remains separately
    usable against externally recorded traces for independent verification.
    """
    PHASES=('WALK','STAND','DECEL','HOLD','RESTART','CLIMB','POST_WALK')

    def __init__(self,case):
        self.case=dict(case)
        self.rows=np.empty((int(case['max_steps'])+2,20),dtype=np.float64)
        self.count=0

    def update(self,frame):
        if self.count>=len(self.rows):
            raise RuntimeError('quality trace exceeded frozen case bound')
        _validate_row(frame)
        phase=frame['phase']
        self.rows[self.count]=[
            self.PHASES.index(phase),float(frame['command']),
            *list(frame['position'])[:2],*list(frame['v_body'])[:2],*list(frame['v_world'])[:2],
            float(frame['roll_deg']),float(frame['pitch_deg']),float(frame['clearance']),
            *[bool(x) for x in frame['safe_support']],bool(frame['stable']),
            bool(frame.get('top_event',False)),bool(frame.get('failure',False)),
            bool(frame.get('timeout',False)),bool(frame.get('planar',phase!='CLIMB'))]
        self.count+=1

    def finish(self,*,completed=True):
        trace=[]
        for row in self.rows[:self.count]:
            trace.append(dict(phase=self.PHASES[int(row[0])],command=row[1],position=row[2:4],
                              v_body=row[4:6],v_world=row[6:8],roll_deg=row[8],pitch_deg=row[9],
                              clearance=row[10],safe_support=row[11:15].astype(bool),stable=bool(row[15]),
                              top_event=bool(row[16]),failure=bool(row[17]),timeout=bool(row[18]),planar=bool(row[19])))
        return grade_case(self.case,trace,completed=completed)
