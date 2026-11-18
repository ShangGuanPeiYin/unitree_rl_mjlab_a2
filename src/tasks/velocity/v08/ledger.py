"""New v08 cost ledger: 6400 source history is never billed a second time."""
from copy import deepcopy

class CostLedger:
    def __init__(self,*,mode,num_envs,state=None):
        if mode not in ('formal','smoke','probe'): raise ValueError('invalid learning container')
        self.mode,self.num_envs=mode,int(num_envs)
        self.state=deepcopy(state) if state is not None else dict(pending_boundary=None,pending_update=None,
            formal_updates=0,engineering_updates=0,control_samples=0,physics_substeps=0,
            evaluation_control_samples=0,events=[],stop_state=None,error_type=None)
        for key in ('evaluation_control_samples','discarded_rollout_control_samples'):
            if type(self.state.get(key,0)) is not int or self.state.get(key,0)<0:raise ValueError('invalid auxiliary control sample cost')
        n=self.updates
        if type(n) is not int or n<0 or n>18000: raise ValueError('invalid updates')
        if (self.state['formal_updates'] if mode!='formal' else self.state['engineering_updates'])!=0:
            raise ValueError('mixed engineering/formal costs')
        if self.state['control_samples']!=n*self.num_envs*48 or self.state['physics_substeps']!=n*self.num_envs*48*4:
            raise ValueError('cost counters differ')
    @property
    def updates(self): return self.state['formal_updates']+self.state['engineering_updates']
    @property
    def learning_samples(self): return 157286400+self.state['control_samples']
    def commit(self,update):
        if update!=6401+self.updates: raise ValueError('duplicate/nonconsecutive billed update')
        key='formal_updates' if self.mode=='formal' else 'engineering_updates'
        self.state[key]+=1;self.state['control_samples']+=48*self.num_envs
        self.state['physics_substeps']+=48*self.num_envs*4
    def state_dict(self): return deepcopy(self.state)
