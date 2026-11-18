"""Batched scripted episodes; endpoint events never reset physical state."""
from copy import deepcopy
import math
import torch
from . import spec
from .terrain import build_terrain_specs, timeout_steps

ALIASES={'flat':'flat_stop','up':'up_stop','down':'down_stop'}
def canonical_case(case):
    c=deepcopy(dict(case));c['task']=ALIASES.get(c.get('task','up_stop'),c.get('task','up_stop'))
    if c['task'] not in ('flat_stop','up_stop','down_stop','E1','E2','E3','E4'):raise ValueError('unknown v08 task')
    c['speed']=float(c.get('cruise_speed',c.get('speed',c.get('vx',.5))))
    if not math.isfinite(c['speed']) or not spec.SPEED_MIN<=c['speed']<=spec.SPEED_MAX:raise ValueError('cruise amplitude outside [.3,.6]')
    script=c.get('script','up_same');default=-1 if c['task'] in ('down_stop','E2') or script.startswith('down_') else 1
    c['initial_direction']=int(c.get('start_direction',c.get('initial_direction',default)))
    c['resume_direction']=int(c.get('resume_direction',(-c['initial_direction'] if script.endswith('reverse') else c['initial_direction'])))
    if c['initial_direction'] not in (-1,1) or c['resume_direction'] not in (-1,1):raise ValueError('direction must be +/-1')
    c['tile_id']=int(c.get('tile_id',0 if c['task']=='flat_stop' else 21))
    for name in ('x','y','yaw'):
        if c.get('spawn_'+name) is not None:c[name]=c['spawn_'+name]
    return c

class TaskMachine:
    WALK,DECEL,HOLD,DONE=range(4)
    def __init__(self,cases,device='cpu'):
        self.output_device=torch.device(device);self.device=torch.device("cpu");device=self.device;self.cases=[canonical_case(c) for c in cases];b=len(cases)
        for name in ('steps','mode','hold_steps','leg_steps','direction_steps','direction','goal','stop_kind','request_step','cruise_start_step','leg_transition_budget'):
            setattr(self,name,torch.zeros(b,dtype=torch.long,device=device))
        for name in ('start_x','previous_x','direction_start','max_stair_yaw'):
            setattr(self,name,torch.zeros(b,device=device))
        for name in ('arrival_paid','success_paid','failure_paid','mid_done','top_done','direction_pending','direction_started','direction_pass','flat_valid'):
            setattr(self,name,torch.zeros(b,dtype=torch.bool,device=device))
        self.budget_events=[[] for _ in cases];self.leg_ids=[0]*b;self.events=[[] for _ in cases];self.direction_checks=[[] for _ in cases];self.limits=[0]*b;self.points=[0.]*b
        self.reset(torch.arange(b),torch.zeros(b,device=device))
    def reset(self,ids,x):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        for name,value in vars(self).items():
            if isinstance(value,torch.Tensor) and value.ndim:value[ids]=0
        x=x.detach().cpu() if isinstance(x,torch.Tensor) else torch.as_tensor(x)
        self.start_x[ids]=x;self.previous_x[ids]=x;self.flat_valid[ids]=True
        for i in ids.tolist():
            c=self.cases[i];self.limits[i]=spec.FLAT_WALK_STEPS if c['task']=='flat_stop' else int(timeout_steps(c['tile_id'],torch.tensor(c['speed'])).item());tile=build_terrain_specs()[c['tile_id']];point=c.get('trigger_p');point=sum(spec.TRIGGER_RANGES.get(c['task'],(0.,0.)))/2 if point is None else point;self.points[i]=float(point)*tile.N*tile.step_depth;self.direction[i]=c['initial_direction'];self.goal[i]=c['initial_direction'];self.leg_transition_budget[i]=spec.TRANSITION_STEPS if c['initial_direction']<0 else 0;self.events[i]=[];self.direction_checks[i]=[];self.leg_ids[i]=0;self.budget_events[i]=[dict(event='travel',leg=0,step=0,steps=self.limits[i])]
            if c['initial_direction']<0:self._budget(i,'startup_transition',spec.TRANSITION_STEPS)
    def set_cases(self,cases):self.cases=[canonical_case(c) for c in cases]
    def _limit(self,i):return self.limits[i]
    def advance(self,x,top,bottom,stand_ok,physical_failure,executed,timeout,transition=None,requested=None):
        transition=torch.zeros_like(top) if transition is None else transition
        requested=executed if requested is None else requested
        # One batched device-to-host snapshot; no CUDA scalar access in FSM.
        packed=torch.stack((x,top,bottom,stand_ok,physical_failure,executed,timeout,transition,requested),-1).detach().cpu()
        v=packed.unbind(-1)
        # Labels describe the command already used by this action. The first
        # zero hold action arrives while the logical machine is still DECEL.
        action_mode=self.mode.clone()
        first_hold=(self.mode==self.DECEL)&(v[5].abs()<=spec.COMMAND_TOLERANCE)&(v[8].abs()<=spec.COMMAND_TOLERANCE)&~v[7].bool()
        action_mode[first_hold]=self.HOLD
        self.action_segments=[('mid' if int(self.stop_kind[i])==1 else 'top' if int(self.goal[i])>0 else 'bottom')+'_'+('walk','deceleration','hold','done')[int(action_mode[i])] for i in range(len(self.cases))]
        out=self._advance_cpu(v[0],v[1].bool(),v[2].bool(),v[3].bool(),v[4].bool(),v[5],v[6].bool(),v[7].bool(),v[8])
        result=torch.stack((out["request"],out["arrival"].float(),out["success"].float(),out["failure"].float()),-1).to(self.output_device)
        return dict(request=result[:,0],arrival=result[:,1].bool(),success=result[:,2].bool(),failure=result[:,3].bool(),reason=out["reason"])
    def _advance_cpu(self,x,top,bottom,stand_ok,physical_failure,executed,timeout,transition=None,requested=None):
        b=len(self.cases);device=self.device
        transition=torch.zeros(b,dtype=torch.bool,device=device) if transition is None else transition
        requested=executed if requested is None else requested
        request=torch.full((b,),float('nan'),device=device)
        arrival=torch.zeros(b,dtype=torch.bool,device=device);success=arrival.clone();failure=arrival.clone()
        reason=['' for _ in range(b)];self.steps+=1
        for i,c in enumerate(self.cases):
            if self.mode[i]==self.DONE:continue
            self.leg_steps[i]+=1;task=c['task'];mode=int(self.mode[i]);speed=c['speed']
            fail=bool(physical_failure[i] or timeout[i]);why='physical_failure' if physical_failure[i] else 'timeout'
            if self.direction_pending[i]:
                self.direction_steps[i]+=1
                if not transition[i] and float(executed[i])*int(self.direction[i])>spec.COMMAND_TOLERANCE:
                    if not self.direction_started[i]:self.direction_start[i]=self.previous_x[i];self.direction_started[i]=True;self.cruise_start_step[i]=self.steps[i]
                    if float(self.direction[i]*(x[i]-self.direction_start[i]))>=spec.REVERSE_PROGRESS:
                        self.direction_pending[i]=False;self.direction_pass[i]=True
                        self.direction_checks[i].append(self._direction_record(i,x,True))
                if self.direction_pending[i] and self.direction_steps[i]>=spec.REVERSE_PROGRESS_STEPS:
                    fail=True;why='cannot_reverse';self.direction_checks[i].append(self._direction_record(i,x,False))
            if mode==self.WALK:
                endpoint=bool(top[i] if self.goal[i]>0 else bottom[i])
                if task=='flat_stop':
                    endpoint=int(self.steps[i])>=spec.FLAT_WALK_STEPS
                    if endpoint and (float(x[i]-self.start_x[i])<spec.FLAT_PROGRESS_FACTOR*speed*spec.FLAT_WALK_SECONDS or not self.flat_valid[i]):fail=True;why='flat_progress'
                elif int(self.leg_steps[i])>=self._limit(i)+int(self.leg_transition_budget[i]):
                    fail=True;why='timeout'
                crossed=False
                if task in spec.TRIGGER_RANGES and not self.mid_done[i]:
                    point=self.points[i]
                    crossed=bool(self.previous_x[i]<point<=x[i]) if self.direction[i]>0 else bool(self.previous_x[i]>point>=x[i])
                if crossed and not fail:
                    self.mid_done[i]=True;self.events[i].append(dict(event='trigger',step=int(self.steps[i]),x=float(x[i])))
                    if task=='E3':
                        request[i]=0.;self.mode[i]=self.DECEL;self.stop_kind[i]=1;self.hold_steps[i]=0;self._budget(i,'mid_deceleration',spec.TRANSITION_STEPS);self._budget(i,'mid_hold',spec.MID_HOLD_STEPS)
                    else:
                        self.direction[i]*=-1;self.goal[i]=self.direction[i];request[i]=speed*self.direction[i];self.leg_steps[i]=0;self.leg_transition_budget[i]=spec.TRANSITION_STEPS;self._new_leg(i,'reverse_transition')
                        self.direction_pending[i]=True;self.direction_steps[i]=0;self.direction_started[i]=False;self.request_step[i]=self.steps[i]
                elif endpoint and not fail:
                    # An intermediate trigger cannot be skipped by reaching the end.
                    if task in spec.TRIGGER_RANGES and not self.mid_done[i]:fail=True;why='missed_trigger'
                    elif self.direction_pending[i]:fail=True;why='cannot_reverse'
                    else:
                        self.mode[i]=self.DECEL;self.stop_kind[i]=2 if task=='E4' and not self.top_done[i] else 0;self.hold_steps[i]=0;request[i]=0.
                        self._budget(i,'endpoint_deceleration',spec.TRANSITION_STEPS);self._budget(i,'endpoint_hold',spec.ENDPOINT_HOLD_STEPS)
                        final=self.stop_kind[i]==0
                        if final and not self.arrival_paid[i]:arrival[i]=True;self.arrival_paid[i]=True
                        self.events[i].append(dict(event='arrival' if final else 'intermediate_top',step=int(self.steps[i]),x=float(x[i])))
            elif mode in (self.DECEL,self.HOLD):
                if abs(float(executed[i]))<=spec.COMMAND_TOLERANCE and abs(float(requested[i]))<=spec.COMMAND_TOLERANCE and not transition[i]:
                    self.mode[i]=self.HOLD;self.hold_steps[i]+=1
                    needed=spec.MID_HOLD_STEPS if self.stop_kind[i]==1 else spec.ENDPOINT_HOLD_STEPS
                    if self.hold_steps[i]>=needed:
                        if not stand_ok[i]:fail=True;why='mid_stop_unstable' if self.stop_kind[i]==1 else 'final_unstable'
                        elif self.stop_kind[i] in (1,2):
                            mid=self.stop_kind[i]==1
                            self.direction[i]=c['resume_direction'] if mid else -1;self.goal[i]=self.direction[i];request[i]=speed*self.direction[i]
                            self.mode[i]=self.WALK;self.leg_steps[i]=0;self.leg_transition_budget[i]=spec.TRANSITION_STEPS;self.hold_steps[i]=0;self._new_leg(i,'restart_transition')
                            self.events[i].append(dict(event='mid_stable' if mid else 'top_stable',step=int(self.steps[i]),x=float(x[i])))
                            if not mid:self.top_done[i]=True
                            if not mid or self.direction[i]!=c['initial_direction']:
                                self.direction_pending[i]=True;self.direction_steps[i]=0;self.direction_started[i]=False;self.request_step[i]=self.steps[i]
                        else:success[i]=True
                elif int(self.leg_steps[i])>self._limit(i)+int(self.leg_transition_budget[i])+spec.TRANSITION_STEPS+(spec.MID_HOLD_STEPS if self.stop_kind[i]==1 else spec.ENDPOINT_HOLD_STEPS):
                    fail=True;why='timeout'
            if fail:
                failure[i]=not self.failure_paid[i];self.failure_paid[i]=True;self.mode[i]=self.DONE;success[i]=False;arrival[i]=False;reason[i]=why;request[i]=float('nan')
            elif success[i]:
                success[i]=not self.success_paid[i];self.success_paid[i]=True;self.mode[i]=self.DONE;reason[i]='success'
        self.previous_x.copy_(x)
        return dict(request=request,arrival=arrival,success=success,failure=failure,reason=reason)
    def _budget(self,i,event,steps):
        self.budget_events[i].append(dict(event=event,leg=self.leg_ids[i],step=int(self.steps[i]),steps=int(steps)))
    def _new_leg(self,i,event):
        self.leg_ids[i]+=1;self._budget(i,"travel",self.limits[i]);self._budget(i,event,spec.TRANSITION_STEPS)
    def _direction_record(self,i,x,success):
        return dict(success=success,direction=int(self.direction[i]),net_progress=float(self.direction[i]*(x[i]-self.direction_start[i])) if self.direction_started[i] else 0.,elapsed_steps=int(self.steps[i]-self.request_step[i]),request_step=int(self.request_step[i]),cruise_start_step=int(self.cruise_start_step[i]),completion_step=int(self.steps[i]))
    def state_dict(self):
        return dict(tensors={k:v.clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor)},cases=deepcopy(self.cases),events=deepcopy(self.events),direction_checks=deepcopy(self.direction_checks),limits=list(self.limits),points=list(self.points),budget_events=deepcopy(self.budget_events),leg_ids=list(self.leg_ids))
    def load_state_dict(self,state):
        self.budget_events=deepcopy(state['budget_events']);self.leg_ids=list(state['leg_ids']);self.limits=list(state['limits']);self.points=list(state['points']);self.cases=deepcopy(state['cases']);self.events=deepcopy(state['events']);self.direction_checks=deepcopy(state['direction_checks'])
        for k,v in state['tensors'].items():getattr(self,k).copy_(v.to(self.device))
