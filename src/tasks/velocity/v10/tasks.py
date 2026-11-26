"""Task-only clocks: action-time command is settled before next-time events."""
import math
from . import cases
ROLES=('F_WALK','F_STAND','F_STOP','P_WALK','P_STOP','S_ROUTE')
WEIGHTS=(.20,.10,.15,.15,.10,.30)
PHASES=('WALK','STAND','CLIMB','POST_WALK','DECEL','HOLD','RESTART')
CRUISE=('WALK','CLIMB','POST_WALK','RESTART')

def allocate_roles(n):
    raw=[n*w for w in WEIGHTS];base=[math.floor(x) for x in raw]
    for i in sorted(range(6),key=lambda i:(-(raw[i]-base[i]),i))[:n-sum(base)]:base[i]+=1
    return dict(zip(ROLES,base))

def combinations(role,course):
    return [dict(x) for x in cases.groups('D3' if course in ('P0','P1') else course) if x['task']==role]

class TaskState:
    def __init__(self,row,course,x):
        self.row=dict(row);self.task=row['task'];self.course=course
        self.speed=float(row.get('speed',0));self.mode=row.get('stop_mode',row.get('route','none'))
        self.N=int(row.get('stairs',row.get('N',0)))
        self.phase='STAND' if self.task=='F_STAND' else 'CLIMB' if self.task=='S_ROUTE' else 'WALK'
        self.teacher='stairs' if self.task=='S_ROUTE' else 'blind'
        self.command=0. if self.phase=='STAND' else self.speed
        self.g_level=0. if self.task=='S_ROUTE' else 1.
        self.step=0;self.phase_steps=0;self.start_x=x;self.segment_x=x;self.top_x=None
        self.top_count=0;self.top_latched=False;self.done=False
        self.max_steps=int(row.get('max_steps',self._max_steps()))
    def _max_steps(self):
        row=dict(self.row)
        row['stop_mode']='ramp_2s' if self.mode=='ramp' else self.mode;row['stairs']=self.N
        return cases.max_steps(row)
    def _enter(self,phase,x):
        self.phase=phase;self.phase_steps=0;self.segment_x=x
        self.command=0. if phase in ('HOLD','STAND') else self.speed
    def _stop(self,x):
        self._enter('DECEL' if self.mode in ('ramp','ramp_2s') else 'HOLD',x)
    def advance(self,x,top_condition,failure,platform_projection=True,approach_level=False):
        if self.done:raise RuntimeError('task must reset after terminal')
        self.step+=1;self.phase_steps+=1
        top=False;finished=False;timeout=False
        if self.task=='S_ROUTE':
            if self.top_latched:
                self.g_level=min(1.,self.g_level+1/25) if platform_projection else 0.
            else:
                self.g_level=1. if approach_level else 0.
                self.top_count=self.top_count+1 if top_condition else 0
                if self.top_count>=2 and not failure:
                    self.top_latched=True;self.teacher='blind';self.top_x=x;self.g_level=0.;top=True
                    if self.course in ('D1','D2') or self.mode=='top':finished=True
                    else:self._enter('POST_WALK',x)
                if x>.2*self.N+1 and not self.top_latched:failure=True
        if not top and not failure:
            p=self.phase
            if self.task in ('F_WALK','F_STAND'):
                finished=self.step>=1000
            elif p=='WALK' and self.task=='F_STOP' and self.phase_steps>=400:self._stop(x)
            elif p in ('WALK','POST_WALK') and self.task in ('P_WALK','P_STOP','S_ROUTE'):
                distance=10. if self.task=='P_WALK' or (self.task=='S_ROUTE' and self.mode=='walk') else 5.
                if x-self.segment_x>=distance:
                    if distance==10.:finished=True
                    else:self._stop(x)
                elif self.phase_steps>=math.ceil((distance/self.speed+10)/.02):timeout=True
            elif p=='DECEL':
                if self.phase_steps>=100:self._enter('HOLD',x)
                else:self.command=self.speed*(1-self.phase_steps/100)
            elif p=='HOLD' and self.phase_steps>=1000:self._enter('RESTART',x)
            elif p=='RESTART':
                if self.task=='F_STOP':finished=self.phase_steps>=250
                else:
                    finished=x-self.segment_x>=5.
                    if not finished and self.phase_steps>=math.ceil((5/self.speed+10)/.02):timeout=True
            elif p=='CLIMB' and self.phase_steps>=math.ceil(((.2*self.N+2.1)/self.speed+15)/.02):timeout=True
        if not finished and self.step>=self.max_steps:timeout=True
        failure=bool(failure or timeout)
        if failure:top=False;finished=False
        self.done=failure or finished
        return dict(top=top,finished=finished,failure=failure,timeout=timeout,done=self.done)
    def state_dict(self):return dict(self.__dict__)
