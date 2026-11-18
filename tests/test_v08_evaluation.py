from copy import deepcopy
import random
import numpy as np
import pytest
import torch

def test_complete_denominator_missing_raw_nonfinite_and_physical_separate():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(kind='baseline')
    def runner(case):
        i=int(case['case_id'].rsplit('-',1)[1])
        row=failure_raw(case)
        if i==0: row['raw_complete']=False
        elif i==1: row['sample']=float('nan')
        elif i==2: raise RuntimeError('sensor')
        return row
    result=e.evaluate_cases(None,table,runner)
    assert result.summary['expected']==60 and result.summary['success']==0
    assert result.summary['invalid']>0 and result.summary['error']>0
    assert result.summary['physical_failure']>0 and not result.summary['gate_pass']
    assert sum(result.summary[k] for k in ('valid','invalid','error'))==60

def test_success_claim_cannot_hide_stationary_or_wrong_direction():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(phase='E')
    flat=next(x for x in table.cases if x['task']=='flat_stop')
    raw=flat_raw(flat,stationary=True)
    row=e.normalize_record(raw,flat)
    assert row['status']=='failure' and row['reason']=='insufficient_flat_progress'
    case=next(x for x in table.cases if x['task']=='E1')
    row=e.normalize_record(direction_raw(case,mode='reverse'),case)
    assert row['status']=='failure' and row['reason']=='wrong_direction'

def window(kind):
    n=150 if kind=='mid' else 500
    return dict(kind=kind,count=300 if kind=='mid' else 1000,xy=[[0.,0.]]*n,values=[[0.,0.,0.,0.,0.,.4,0.,1.,1.,1.]]*n)

def test_rng_and_model_mode_isolated_even_on_exception():
    from src.tasks.velocity.v08 import evaluation as e
    model=torch.nn.Linear(1,1);model.train()
    before=(random.getstate(),deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    def runner(case):
        random.random();np.random.random();torch.rand(1)
        raise RuntimeError('intentional')
    result=e.evaluate_cases(model,e.make_cases(kind='baseline'),runner)
    assert random.getstate()==before[0]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2]) and model.training
    assert result.isolation['statistics_unchanged']

def test_direction_evidence_binds_request_and_first_cruise_clock():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E1')
    raw=direction_raw(case)
    assert e.normalize_record(raw,case)['status']=='success'
    raw['direction_checks'][0]['cruise_start_step']=0
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw['direction_checks'][0].update(cruise_start_step=151,completion_step=450,elapsed_steps=200)
    assert e.normalize_record(raw,case)['status']=='invalid'

def test_summary_rejects_unrecognized_status():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(kind='baseline')
    with pytest.raises(e.EvaluationError,match='status'):
        e.summarize([dict(case_id=table.cases[0]['case_id'],status='magic',success=True,reason='success')],table)

def test_model_mutation_is_restored_and_blocks_evaluation():
    from src.tasks.velocity.v08 import evaluation as e
    model=torch.nn.Linear(1,1);before=deepcopy(model.state_dict())
    def runner(c):
        with torch.no_grad():model.weight.add_(1.)
        return None
    with pytest.raises(e.EvaluationError,match='mutated'):
        e.evaluate_cases(model,e.make_cases(kind='baseline'),runner)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())

def test_raw_window_binary_support_and_nonnegative_norms_are_required():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases().cases if c['task']=='up_stop')
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[window('top')])
    raw['stop_windows'][0]['values']=[list(x) for x in raw['stop_windows'][0]['values']]
    raw['stop_windows'][0]['values'][0][7]=4.
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw['stop_windows'][0]['values'][0][7]=1.
    raw['stop_windows'][0]['values'][0][0]=-1.
    assert e.normalize_record(raw,case)['status']=='invalid'


def test_direction_contract_only_reverse_e3_but_full_roundtrip_counts():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(phase='E')
    for c in table.cases:
        expected=c['task'] in ('E1','E2','E4') or (c['task']=='E3' and c['script'].endswith('reverse'))
        assert e.needs_direction_check(c)==expected
    case=next(c for c in table.cases if c['task']=='E3' and c['script']=='up_same')
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[window('mid'),window('top')],max_stair_abs_yaw=0.,trace=[motion_frame(0,1.,.3),motion_frame(1,1.,.3)])
    assert e.normalize_record(raw,case)['status']=='invalid'
    assert e.normalize_record(direction_raw(case),case)['status']=='success'
    case=next(c for c in table.cases if c['task']=='E4')
    raw.update(case_id=case['case_id'],stop_windows=[window('top'),window('bottom')])
    assert e.normalize_record(raw,case)['status']=='invalid'

def motion_frame(step,x,command,*,request=None,transition=False,requested=None,y=0.,yaw=0.):
    return dict(control_step=step,local_x=x,local_y=y,yaw=yaw,executed_c=command,requested_c=command if requested is None else requested,transition=transition,segment='walk',request_event=request,next_requested_c=(command if requested is None else requested) if request is None else request)

def direction_raw(case,*,mode='correct',lying=False):
    from src.tasks.velocity.v08 import spec
    if case['task'] in ('up_stop','down_stop'):return endpoint_raw(case)
    direction=-1 if case['task'] in ('E1','E4') else 1 if case['task']=='E2' else case['resume_direction']
    old_direction=case['initial_direction'];old=old_direction*case['speed'];target=direction*case['speed']
    top=case['spawn_mode']=='top';stop=150 if top else 50
    parked=case['task'] in ('E3','E4');hold=spec.MID_HOLD_STEPS if case['task']=='E3' else spec.ENDPOINT_HOLD_STEPS
    r=stop+100+hold if parked else stop;start=r+101;deadline=r+400
    point=case['trigger_p']*spec.terrain_spec(case['tile_id']).N*spec.terrain_spec(case['tile_id']).step_depth if case['task'] in ('E1','E2','E3') else 8.80001
    initial_x=case['x'];trace=[];x=initial_x
    for step in range(deadline+1):
        event=None;transition=False;segment='walk'
        if step<=stop:
            x=point if step==stop else initial_x+(point-initial_x)*step/stop
            command=old*step/100 if top and step<=100 else old
            requested=old;transition=top and 1<=step<=100
            if step==0 and top:event=old
            if step==stop:event=0. if parked else target
        elif parked and step<=r:
            command=old*(1-(step-stop)/100) if step<=stop+100 else 0.
            requested=0.;transition=step<=stop+100;segment='deceleration' if transition else 'hold'
            if step==r:event=target
        else:
            v0=0. if parked else old;transition=step<=r+100
            command=v0+(target-v0)*(step-r)/100 if transition else target
            requested=target
            if mode=='old_command':command=v0
        if step>=start:
            if mode in ('correct','old_command'):x+=direction*.006000001
            elif mode=='reverse':x-=direction*.006000001
            elif mode=='late' and step>=deadline:x+=direction*.006000001
        frame=motion_frame(step,x,command,request=event,transition=transition,requested=requested)
        frame['segment']='initial' if step==0 else segment;frame.update(top_arrived=(step==stop and case['task']=='E4'),bottom_arrived=False);trace.append(frame)
    end=start+99 if mode in ('correct','old_command') else deadline
    progress=direction*(trace[end]['local_x']-trace[start-1]['local_x'])
    reported=.61 if lying else progress
    check=dict(direction=direction,net_progress=reported,elapsed_steps=end-r,request_step=r,cruise_start_step=start,completion_step=end)
    kinds=['mid','top' if direction>0 else 'bottom'] if case['task']=='E3' else ['top','bottom'] if case['task']=='E4' else ['top' if direction>0 else 'bottom']
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[],max_stair_abs_yaw=0.,direction_checks=[] if not __import__('src.tasks.velocity.v08.evaluation',fromlist=['needs_direction_check']).needs_direction_check(case) else [check],trace=trace)
    if parked:raw['stop_windows'].append(bound_window(trace,stop,hold,kinds[0],case))
    return finish_raw(raw,case,target,kinds[-1])

def flat_raw(case,*,stationary=False,lying=False):
    dx=0. if stationary else .8*case['speed']*8/400
    trace=[motion_frame(i,case['x']+i*dx,case['speed'],request=0. if i==400 else None) for i in range(401)]
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[],flat_progress=dict(net_dx=.8*case['speed']*8 if lying else 400*dx,max_abs_y=0.,max_abs_yaw=0.,walk_steps=400),trace=trace)
    return finish_raw(raw,case,case['speed'],'flat',already_stopped=True)

@pytest.mark.parametrize('mode',['stationary','reverse','late','old_command'])
def test_direction_claim_cannot_override_continuous_raw_movement(mode):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E1')
    raw=direction_raw(case,mode=mode,lying=True)
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('mode',['stationary','reverse','late'])
def test_actual_failed_movement_is_policy_failure_when_raw_summary_agrees(mode):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E1')
    assert e.normalize_record(direction_raw(case,mode=mode),case)['status']=='failure'

def test_complete_movement_raw_recomputed_with_command_trace_and_missing_data_rejected():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E1')
    raw=direction_raw(case)
    assert e.normalize_record(raw,case)['status']=='success'
    del raw['trace'][60]
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw=direction_raw(case);raw['trace'][151]['local_x']=float('nan')
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw=direction_raw(case);del raw['trace']
    assert e.normalize_record(raw,case)['status']=='invalid'

def test_flat_claim_and_route_recomputed_from_400_action_boundaries():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases().cases if c['task']=='flat_stop')
    assert e.normalize_record(flat_raw(case),case)['status']=='success'
    assert e.normalize_record(flat_raw(case,stationary=True),case)['status']=='failure'
    assert e.normalize_record(flat_raw(case,stationary=True,lying=True),case)['status']=='invalid'
    raw=flat_raw(case);raw['trace'][100]['executed_c']=-case['speed']
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw=flat_raw(case);raw['trace'][100]['local_y']=.3
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw['flat_progress']['max_abs_y']=.3
    assert e.normalize_record(raw,case)['status']=='failure'


def test_request_events_and_action_latched_targets_must_be_causal():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E1')
    raw=direction_raw(case);raw['trace'][50]['next_requested_c']=case['speed']
    assert e.normalize_record(raw,case)['status']=='invalid'
    raw=direction_raw(case);raw['trace'][49]['next_requested_c']=0.
    assert e.normalize_record(raw,case)['status']=='invalid'

def test_flat_first_action_window_cannot_hide_midwalk_request_interruptions():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases().cases if c['task']=='flat_stop')
    raw=flat_raw(case);raw['trace'][100]['request_event']=0.;raw['trace'][100]['next_requested_c']=0.
    assert e.normalize_record(raw,case)['status']=='invalid'


@pytest.mark.parametrize('task',['E1','E2'])
def test_same_target_repeat_cannot_impersonate_direct_reversal(task):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task)
    raw=direction_raw(case);target=(-1 if task=='E1' else 1)*case['speed']
    for f in raw['trace']:
        f['executed_c']=target;f['requested_c']=target;f['next_requested_c']=target
        if f['request_event'] is not None:f['request_event']=target
        if f['control_step']==0:f['request_event']=None
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('task',['E1','E2'])
def test_wrong_initial_cruise_is_invalid_even_if_request_frame_has_correct_old_speed(task):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task)
    raw=direction_raw(case)
    raw['trace'][1]['executed_c']*=-1
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('task,script',[('E1',None),('E2',None),('E3','up_reverse'),('E3','down_reverse'),('E4',None)])
def test_case_specific_direct_or_parked_reversal_has_valid_old_command(task,script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task and (script is None or c['script']==script))
    assert e.normalize_record(direction_raw(case),case)['status']=='success'

@pytest.mark.parametrize('task,script',[('E3','up_reverse'),('E3','down_reverse'),('E4',None)])
def test_parked_reversal_requires_real_zero_hold_before_restart(task,script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task and (script is None or c['script']==script))
    raw=direction_raw(case);r=raw['direction_checks'][0]['request_step']
    raw['trace'][r]['executed_c']=case['speed']
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('task,script',[('E1',None),('E2',None),('E3','up_reverse')])
def test_trigger_request_must_match_first_directional_crossing(task,script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task and (script is None or c['script']==script))
    raw=direction_raw(case)
    for f in raw['trace']:f['local_x']+=.2
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('task,script',[('up_stop',None),('down_stop',None),('E3','up_same')])
def test_final_review_isolated_windows_and_short_trace_are_invalid(task,script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task and (script is None or c['script']==script))
    kinds=e._required_windows(case)
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,
             reason='success',segment='hold',stop_windows=[window(k) for k in kinds],max_stair_abs_yaw=0.)
    if task!='up_stop':raw['trace']=[motion_frame(0,1.,.3),motion_frame(1,1.,.3)]
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('task',['flat_stop','up_stop','down_stop','E1','E2','E3','E4'])
@pytest.mark.parametrize('missing',[True,False])
def test_final_review_all_failed_terminals_require_raw_through_terminal(task,missing):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task)
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=False,
             reason='fall',segment='walk',physical_failure=True,terminal_step=1,control_samples=1,duration=.02)
    if not missing:raw['trace']=[]
    assert e.normalize_record(raw,case)['status']=='invalid'

def test_final_review_overflow_is_engineering_error_even_with_self_declared_complete():
    from src.tasks.velocity.v08 import evaluation as e
    table=e.make_cases(kind='baseline');case=table.cases[0]
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=False,
             reason='contact_overflow',segment='walk',physical_failure=False,trace=[])
    row=e.normalize_record(raw,case)
    assert row['status']=='error'
    result=e.summarize([row],table)
    assert result['engineering_error'] and result['valid']==0 and result['error']==1


def failure_raw(case,**extra):
    """Synthetic two-boundary failure; no claim of a policy/physics outcome."""
    c=case['initial_direction']*case['speed'];top=case['spawn_mode']=='top'
    a=motion_frame(0,case['x'],0. if top else c,request=c if top else None,requested=c,y=case['y'],yaw=case['yaw'])
    b=motion_frame(1,case['x'],c/100 if top else c,requested=c,transition=top,y=case['y'],yaw=case['yaw'])
    return dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=False,reason='fall',segment='walk',physical_failure=True,failure_kind='physical',trace=[a,b],terminal_step=1,control_samples=1,duration=.02,**extra)

def bound_window(trace,stop,hold,kind,case):
    end=stop+100+hold;n=150 if kind=='mid' else 500
    from src.tasks.velocity.v08 import spec
    origin=spec.terrain_spec(case['tile_id']).frame_origin_world[:2]
    w=window(kind);w.update(stop_request_step=stop,hold_start_step=stop+101,start_step=end-n+1,end_step=end,origin_xy=list(origin),xy=[[f['local_x']+origin[0],f['local_y']+origin[1]] for f in trace[end-n+1:end+1]])
    return w

def finish_raw(raw,case,old,kind,already_stopped=False):
    """Full deterministic kinematic protocol fixture, not trained-policy evidence.

    Positions intentionally illustrate the task protocol only. They are not
    claimed to result from physically feasible actions or a policy rollout.
    """
    from src.tasks.velocity.v08 import spec
    trace=raw['trace'];length=spec.terrain_spec(case['tile_id']).N*spec.terrain_spec(case['tile_id']).step_depth
    if not already_stopped:
        destination=length+.80001 if kind=='top' else -.80001
        start_x=trace[-1]['local_x'];start=len(trace)-1
        for k in range(1,401):
            f=motion_frame(start+k,start_x+(destination-start_x)*k/400,old,request=0. if k==400 else None)
            f.update(top_arrived=kind=='top' and k==400,bottom_arrived=kind=='bottom' and k==400);trace.append(f)
    stop=len(trace)-1
    for k in range(1,1101):
        transition=k<=100;c=old*(1-k/100) if transition else 0.
        f=motion_frame(stop+k,trace[stop]['local_x'],c,requested=0.,transition=transition)
        f.update(segment='deceleration' if transition else 'hold',top_arrived=False,bottom_arrived=False);trace.append(f)
    trace[0].update(local_x=case['x'],local_y=case['y'],yaw=case['yaw'])
    raw['stop_windows'].append(bound_window(trace,stop,1000,kind,case))
    raw.update(terminal_step=len(trace)-1,control_samples=len(trace)-1,duration=(len(trace)-1)*.02,initial_pose=[case['x'],case['y'],case['yaw']])
    return raw

def endpoint_raw(case):
    c=case['initial_direction']*case['speed'];top=case['spawn_mode']=='top'
    trace=[motion_frame(0,case['x'],0. if top else c,requested=c,request=c if top else None)]
    if top:
        for k in range(1,101):trace.append(motion_frame(k,case['x'],c*k/100,requested=c,transition=True))
    for f in trace:f.update(top_arrived=False,bottom_arrived=False)
    raw=dict(case_id=case['case_id'],complete=True,raw_complete=True,finite=True,success=True,reason='success',segment='hold',stop_windows=[],max_stair_abs_yaw=0.,trace=trace)
    return finish_raw(raw,case,c,'top' if c>0 else 'bottom')

@pytest.mark.parametrize('task,script',[('flat_stop',None),('up_stop',None),('down_stop',None),('E1',None),('E2',None),('E3','up_same'),('E3','down_same'),('E3','up_reverse'),('E3','down_reverse'),('E4',None)])
def test_final_review_all_success_scripts_require_complete_trace_and_bound_hold(task,script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task and (script is None or c['script']==script))
    make=flat_raw if task=='flat_stop' else direction_raw
    raw=make(case)
    assert e.normalize_record(raw,case)['status']=='success',e.normalize_record(raw,case).get('detail')
    for mutation in ('missing','gap','terminal','clock','position','hold'):
        bad=deepcopy(raw)
        if mutation=='missing':del bad['trace']
        elif mutation=='gap':del bad['trace'][1]
        elif mutation=='terminal':bad['terminal_step']+=1
        elif mutation=='clock':bad['stop_windows'][-1]['end_step']-=1
        elif mutation=='position':bad['stop_windows'][-1]['xy'][0][0]+=.2
        else:
            bad['trace']=bad['trace'][:-1];bad['terminal_step']-=1;bad['control_samples']-=1;bad['duration']-=.02
        assert e.normalize_record(bad,case)['status']=='invalid',(task,script,mutation)

@pytest.mark.parametrize('script',['up_same','down_same'])
def test_final_review_e3_same_requires_trigger_restart_and_real_endpoint_without_reverse_gate(script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='E3' and c['script']==script)
    raw=direction_raw(case,mode='stationary')
    # No .60m/8s condition applies: this example remains still through the
    # direction deadline, then continues to its actual endpoint and holds.
    assert e.normalize_record(raw,case)['status']=='success'
    for mutation in ('birth','trigger','restart','endpoint'):
        bad=deepcopy(raw)
        if mutation=='birth':bad['trace'][0]['local_x']+=.2
        elif mutation=='trigger':
            stop=bad['stop_windows'][0]['stop_request_step'];bad['trace'][stop]['local_x']-=case['initial_direction']*.1
        elif mutation=='restart':bad['trace'][bad['stop_windows'][0]['end_step']]['request_event']=None
        else:
            stop=bad['stop_windows'][-1]['stop_request_step'];bad['trace'][stop]['top_arrived' if case['resume_direction']>0 else 'bottom_arrived']=False
        assert e.normalize_record(bad,case)['status']=='invalid',mutation

@pytest.mark.parametrize('task',['flat_stop','up_stop','down_stop','E1','E2','E3','E4'])
def test_final_review_complete_real_failure_needs_no_future_success_windows(task):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task)
    raw=failure_raw(case)
    assert e.normalize_record(raw,case)['status']=='failure'
    raw['trace'][-1]['executed_c']+=.001
    assert e.normalize_record(raw,case)['status']=='invalid'

@pytest.mark.parametrize('task,script',[('up_stop',None),('E4',None),('E3','up_same')])
def test_final_review_full_parking_trace_cannot_leave_its_physical_region(task,script):
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']==task and (script is None or c['script']==script))
    raw=direction_raw(case);w=raw['stop_windows'][0]
    # Outside the scoring window, still within the prescribed real dwell.
    raw['trace'][w['hold_start_step']]['local_x']=-.1 if w['kind']=='mid' else 7.99
    assert e.normalize_record(raw,case)['status']=='invalid'

def test_final_review_window_coordinate_origin_must_match_frozen_terrain():
    from src.tasks.velocity.v08 import evaluation as e
    case=next(c for c in e.make_cases(phase='E').cases if c['task']=='up_stop')
    raw=direction_raw(case)
    raw['stop_windows'][0]['origin_xy'][0]+=100.
    for xy in raw['stop_windows'][0]['xy']:xy[0]+=100.
    assert e.normalize_record(raw,case)['status']=='invalid'
