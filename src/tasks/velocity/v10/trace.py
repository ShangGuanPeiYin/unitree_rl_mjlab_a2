"""Versioned diagnostic side channel; the legacy 20 grading columns stay intact.

Action fields describe the observation used to produce this action. Measurement
fields describe the resulting physical frame. Only active cases are recorded.
"""
import numpy as np

FAILURES=('body_hit','long_hit','attitude','route_fall','bounds','timeout')


def _array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value)


def action_snapshot(obs,stairs_mask=None):
    """Reuse the actual blind command adapter, never duplicate its controller."""
    prop=obs['proprio'];route=obs['route_error']
    requested=_array(prop[:,6:9]).copy()
    n=len(requested)
    private=requested.copy()
    mask=np.zeros(n,dtype=bool) if stairs_mask is None else _array(stairs_mask).astype(bool)
    if stairs_mask is not None:
        from .teachers import blind_yaw_command
        blind=_array(blind_yaw_command(prop,route))
        private=np.where(mask[:,None],requested,blind)
    r=_array(route)
    return dict(requested_command=requested,teacher_command=private,
                action_yaw=np.arctan2(r[:,1],r[:,2]),action_route_error=r.copy(),teacher_is_stairs=mask,
                teacher_command_valid=np.full(n,stairs_mask is not None,dtype=bool))


class TraceBatchV2:
    def __init__(self,n,max_steps):
        self.n=n;self.capacities=np.asarray(max_steps,dtype=np.int64)
        if self.capacities.shape!=(n,) or (self.capacities<=0).any():
            raise ValueError('invalid trace capacities')
        self.max_steps=int(self.capacities.max());self.counts=np.zeros(n,dtype=np.int64)
        self.buffers={};self.frozen=set()

    def append(self,action,frame,active):
        if self.frozen.intersection(active):raise ValueError('trace case already frozen')
        values={**action,'yaw_next':frame['yaw'],
                'foot_position_local':frame['foot_position_local'],
                'contact_raw':frame['contact_raw'],'top_upward_force':frame['top_upward_force'],
                **{'failure_'+k:frame['failure_masks'][k] for k in FAILURES}}
        shapes={'requested_command':(3,),'teacher_command':(3,), 'action_yaw':(),
                'teacher_is_stairs':(),'teacher_command_valid':(),'action_route_error':(3,),
                'yaw_next':(),'foot_position_local':(4,3),
                'contact_raw':(4,),'top_upward_force':(4,),
                **{'failure_'+k:() for k in FAILURES}}
        boolean={'teacher_is_stairs','teacher_command_valid','contact_raw',*('failure_'+k for k in FAILURES)}
        arrays={}
        ids=sorted(active)
        if any(i<0 or i>=self.n for i in ids):raise ValueError('invalid active trace index')
        if any(self.counts[i]>=self.capacities[i] for i in ids):raise ValueError('trace capacity exceeded')
        for k,shape in shapes.items():
            a=_array(values[k])
            if a.shape!=(self.n,*shape):raise ValueError('invalid trace shape: '+k)
            if not np.isfinite(a[ids]).all():raise ValueError('nonfinite trace: '+k)
            if k in boolean and not np.isin(a[ids],(0,1)).all():raise ValueError('invalid trace mask: '+k)
            arrays[k]=a.astype(bool) if k in boolean else a
        if not self.buffers:
            # A bounded number of contiguous arrays replaces millions of retained
            # per-frame Python dicts/small arrays. Preserve the source precision.
            self.buffers={k:np.empty((self.n,self.max_steps,*a.shape[1:]),dtype=a.dtype)
                          for k,a in arrays.items()}
        for k,a in arrays.items():
            if a.dtype!=self.buffers[k].dtype:raise ValueError('trace dtype changed: '+k)
            self.buffers[k][ids,self.counts[ids]]=a[ids]
        self.counts[ids]+=1

    def save(self,i,path,rows):
        if i in self.frozen:raise ValueError('trace case already frozen')
        count=int(self.counts[i]);rows=_array(rows)
        if not count or rows.shape!=(count,20) or not np.isfinite(rows).all():
            raise ValueError('trace/grading row alignment invalid')
        values={k:a[i,:count] for k,a in self.buffers.items()}
        np.savez_compressed(path,rows=rows,schema_version=np.asarray(2,dtype=np.int64),
                            schema_name=np.asarray('a2-v10-trace-v2'),
                            timing=np.asarray('action=pre-step observation; measurement=post-step; first-terminal-inclusive'),
                            **values)
        self.frozen.add(i)
