"""SDS v0.8: fixed task slots, independent sampler and capped A→E gates."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import math
import numpy as np
from . import spec

PHASE_BUDGETS = dict(spec.PHASE_BUDGETS)
SCRIPTS = ('up_same', 'up_reverse', 'down_same', 'down_reverse')
TASK_ALIASES = {'flat':'flat_stop','up':'up_stop','down':'down_stop'}

def spawn_bounds(phase, updates=0, *, evaluation=False):
    phase = str(phase).upper()
    if phase not in spec.PHASE_ORDER:
        raise ValueError('unknown v08 phase')
    fraction = 1. if phase in ('C','D','E') or (phase == 'B' and evaluation) else (min(max(int(updates),0)/spec.R_RAMP_UPDATES,1.) if phase == 'B' else 0.)
    out = {}
    for key, start, end in (('x',spec.SPAWN_X,spec.R_SPAWN_X), ('y',spec.SPAWN_Y,spec.R_SPAWN_Y), ('yaw',spec.SPAWN_YAW_RADIANS,spec.R_SPAWN_YAW_RADIANS)):
        out[key] = tuple(float(a+(b-a)*fraction) for a,b in zip(start,end))
    return out

def role_case(role, index=0):
    if role in ('E1','E2','E3','E4'):
        task, terrain = role, 'L40'
    elif role == 'flat':
        task, terrain = 'flat_stop', 'FLAT'
    else:
        task, terrain = role.split('_',1)
        task = TASK_ALIASES[task]
    tile = spec.FLAT_TILE_ID if terrain == 'FLAT' else spec.tile_id_for(terrain)
    script = SCRIPTS[index%4] if task == 'E3' else task
    initial = -1 if task in ('down_stop','E2') or (task == 'E3' and script.startswith('down')) else 1
    resume = -initial if script.endswith('reverse') else initial
    top = initial < 0
    geometry = spec.terrain_spec(tile)
    return dict(task=task, script=script, terrain_name=terrain, tile_id=tile,
                initial_direction=initial, start_direction=initial, resume_direction=resume,
                requested_after_stop=resume, spawn_mode='top' if top else 'bottom',
                spawn_x=geometry.N*geometry.step_depth+spec.TOP_SPAWN_OFFSET if top else None,
                trigger_range=spec.TRIGGER_RANGES.get(task))

@dataclass(frozen=True)
class ScheduleAssignment:
    phase: str
    roles: tuple
    tile_ids: tuple
    cases: tuple
    generation: int = 0
    substage: object = None
    frontier_tile_id: int = field(default_factory=lambda: spec.tile_id_for('L40'))
    def as_dict(self):
        return deepcopy(asdict(self))

@dataclass
class CurriculumState:
    phase: str = 'A'
    generation: int = 0
    generation_updates: int = 0
    last_update: int = spec.SOURCE_UPDATE
    spent_by_phase: dict = field(default_factory=lambda: dict.fromkeys(spec.PHASE_ORDER,0))
    gate_streak: int = 0
    last_gate_update: object = None
    finished: bool = False
    stop_status: str = 'running'
    sampler_rng_state: dict = field(default_factory=dict)
    gate_history: list = field(default_factory=list)

class CurriculumManager:
    def __init__(self, num_envs=512, seed=42, state=None):
        self.num_envs, self.seed = int(num_envs), int(seed)
        if self.num_envs < 1: raise ValueError('num_envs must be positive')
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed,spec.SAMPLER_SEED_OFFSET]))
        self.state = CurriculumState()
        if state is not None: self.load_state_dict(state)
    def schedule(self):
        st = self.state
        alloc = spec.PHASE_ALLOCATIONS[st.phase]
        raw = {k:self.num_envs*n/512 for k,n in alloc.items()}
        counts = {k:math.floor(n) for k,n in raw.items()}
        order = sorted(alloc,key=lambda k:(-(raw[k]-counts[k]),list(alloc).index(k)))
        for k in order[:self.num_envs-sum(counts.values())]: counts[k] += 1
        bounds = spawn_bounds(st.phase,st.generation_updates)
        roles=[]; cases=[]
        for role, n in counts.items():
            for index in range(n):
                row=role_case(role,index)
                row.update(case_id=f'train-{st.phase}-{st.generation}-{len(cases):04d}',
                           spawn_x_bounds=bounds['x'],spawn_y_bounds=bounds['y'],spawn_yaw_bounds=bounds['yaw'],
                           speed_bounds=(spec.SPEED_MIN,spec.SPEED_MAX), training=True)
                cases.append(row); roles.append(role)
        return ScheduleAssignment(st.phase,tuple(roles),tuple(x['tile_id'] for x in cases),tuple(cases),st.generation)
    def on_update(self, update=None):
        st=self.state
        if st.finished: return False
        update=st.last_update+1 if update is None else int(update)
        if update != st.last_update+1: raise ValueError('updates must be consecutive real learning updates')
        st.last_update=update; st.generation_updates+=1; st.spent_by_phase[st.phase]+=1
        if st.spent_by_phase[st.phase]>=PHASE_BUDGETS[st.phase]:
            st.finished=True; st.stop_status='budget_exhausted'
        return st.finished
    def record_gate(self, result, update=None):
        st=self.state
        if hasattr(result,'summary'): result=result.summary
        if result.get('kind') in ('h00','stair_retention'): return False
        update=st.last_update if update is None else int(update)
        if update != st.last_update or update <= spec.SOURCE_UPDATE or update%100:
            raise ValueError('full gate must bind current 100-update boundary')
        if st.last_gate_update == update: return False
        if st.finished and st.stop_status!='budget_exhausted': return False
        passed = result.get('gate_pass') is True
        if result.get('phase',st.phase)!=st.phase: raise ValueError('gate phase differs from curriculum')
        if result.get('engineering_error',False): passed=False
        consecutive=st.last_gate_update is None or update-int(st.last_gate_update)==100
        st.gate_streak=(st.gate_streak+1 if consecutive else 1) if passed else 0
        st.last_gate_update=update
        st.gate_history.append(dict(update=update,phase=st.phase,passed=passed,streak=st.gate_streak))
        if st.gate_streak < spec.GATE_STREAK_REQUIRED: return False
        i=spec.PHASE_ORDER.index(st.phase)
        if i==len(spec.PHASE_ORDER)-1:
            st.finished=True; st.stop_status='complete'
        else:
            st.phase=spec.PHASE_ORDER[i+1]; st.generation+=1;st.generation_updates=0
            st.gate_streak=0; st.last_gate_update=None;st.finished=False;st.stop_status='running'
        return True
    def state_dict(self):
        self.state.sampler_rng_state=deepcopy(self._rng.bit_generator.state)
        return dict(schema='v08-curriculum-v1',num_envs=self.num_envs,seed=self.seed,state=asdict(self.state))
    def load_state_dict(self, payload):
        payload=deepcopy(payload)
        if payload.get('schema')!='v08-curriculum-v1': raise ValueError('invalid v08 curriculum schema')
        if int(payload['num_envs'])!=self.num_envs or int(payload['seed'])!=self.seed: raise ValueError('curriculum sampler identity differs')
        st=CurriculumState(**payload['state'])
        if st.phase not in spec.PHASE_ORDER or set(st.spent_by_phase)!=set(spec.PHASE_ORDER): raise ValueError('invalid phase ledger')
        if any(type(n) is not int or n<0 or n>PHASE_BUDGETS[k] for k,n in st.spent_by_phase.items()): raise ValueError('invalid spent budget')
        if st.last_update != spec.SOURCE_UPDATE+sum(st.spent_by_phase.values()): raise ValueError('global update differs from spent ledger')
        self._rng.bit_generator.state=st.sampler_rng_state
        self.state=st
    @property
    def phase(self): return self.state.phase
    @property
    def generation(self): return self.state.generation
    @property
    def substage(self): return None
    @property
    def finished(self): return self.state.finished
    @property
    def stop_status(self): return self.state.stop_status

CurriculumAssignment=ScheduleAssignment
