"""Six independent standard CPU/FP32 reservoirs with reproducible sampling."""
import copy
import hashlib
import json
import torch

TASKS=('F_WALK','F_STAND','F_STOP','P_WALK','P_STOP','S_ROUTE')

def index_batch(batch, indices):
    indices=torch.as_tensor(indices,dtype=torch.long,device='cpu')
    return {k:({n:t[indices.to(t.device)] for n,t in v.items()} if k=='obs' else
               v[indices.to(v.device)] if isinstance(v,torch.Tensor) else [v[i] for i in indices.tolist()]) for k,v in batch.items()}

def concat_batches(parts):
    return {k:({n:torch.cat([p[k][n] for p in parts]) for n in parts[0][k]} if k=='obs' else
               torch.cat([p[k] for p in parts]) if isinstance(parts[0][k],torch.Tensor) else
               sum([list(p[k]) for p in parts],[])) for k in parts[0]}

def snapshot(batch):
    return {k:({n:t.detach().to('cpu',torch.float32).clone() for n,t in v.items()} if k=='obs' else
               v.detach().to(device='cpu',dtype=torch.float32 if v.is_floating_point() else v.dtype).clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in batch.items()}

class TaskReservoir:
    def __init__(self,capacity=32768,seed=42):
        self.capacity=int(capacity); self.rng=torch.Generator().manual_seed(seed)
        self.data={k:None for k in TASKS}; self.seen={k:0 for k in TASKS}
    def add(self,batch):
        # Slots copy incoming values; there is no need for a full-rollout clone first.
        for task in TASKS:
            ids=[i for i,k in enumerate(batch['task']) if k==task]
            if not ids: continue
            incoming=index_batch(batch,ids)
            if self.data[task] is None:
                # Allocate bounded contiguous storage once; metadata remains detached.
                n=self.capacity
                self.data[task]={k:({name:torch.empty((n,*t.shape[1:]),dtype=torch.float32) for name,t in v.items()} if k=='obs' else
                                    torch.empty((n,*v.shape[1:]),dtype=torch.float32 if v.is_floating_point() else v.dtype) if isinstance(v,torch.Tensor) else [None]*n) for k,v in incoming.items()}
            dst=self.data[task]
            for i in range(len(ids)):
                self.seen[task]+=1; count=self.seen[task]
                j=count-1 if count<=self.capacity else int(torch.randint(count,(1,),generator=self.rng))
                if j>=self.capacity: continue
                for k,v in incoming.items():
                    if k=='obs':
                        for name,t in v.items(): dst[k][name][j].copy_(t[i].detach())
                    elif isinstance(v,torch.Tensor): dst[k][j].copy_(v[i].detach())
                    else: dst[k][j]=copy.deepcopy(v[i])
    def sample_per_task(self,n):
        pieces=[]
        for task in TASKS:
            size=min(self.seen[task],self.capacity)
            if not size: raise ValueError(f'empty reservoir task {task}')
            pieces.append(index_batch(self.data[task],torch.randint(size,(n,),generator=self.rng)))
        return concat_batches(pieces)
    def sample_logical(self,current):
        pieces=[]
        for task in TASKS:
            ids=torch.tensor([i for i,t in enumerate(current['task']) if t==task])
            if not len(ids): raise ValueError(f'current rollout missing {task}')
            take=lambda n:index_batch(current,ids[torch.randint(len(ids),(n,),generator=self.rng)])
            pieces.append(take(256)); size=min(self.seen[task],self.capacity)
            h=min(size,256)
            if h: pieces.append(index_batch(self.data[task],torch.randint(size,(h,),generator=self.rng)))
            if h<256: pieces.append(take(256-h))
        return concat_batches(pieces)
    def state_dict(self,copy_storage=True):
        state=dict(capacity=self.capacity,data=self.data,seen=dict(self.seen),rng=self.rng.get_state())
        return copy.deepcopy(state) if copy_storage else state
    def load_state_dict(self,state,copy_storage=True):
        if int(state['capacity'])!=self.capacity: raise ValueError('reservoir capacity mismatch')
        self.data=copy.deepcopy(state['data']) if copy_storage else state['data']; self.seen=dict(state['seen']); self.rng.set_state(state['rng'])

    def release_storage(self):
        """Release the current owner only after its checkpoint has been validated."""
        self.data={k:None for k in TASKS};self.seen={k:0 for k in TASKS}
    def take_state_dict(self,state):
        """Destructive ownership transfer, leaving no storage aliases in decoded state."""
        if int(state['capacity'])!=self.capacity:raise ValueError('reservoir capacity mismatch')
        data=state['data'];seen=dict(state['seen']);rng=state['rng']
        self.data=data;self.seen=seen;self.rng.set_state(rng)
        del state['data'];state.clear()
    def integrity_manifest(self):
        digest=hashlib.sha256();counts={}
        for task in TASKS:
            count=min(self.seen[task],self.capacity);counts[task]=count
            data=self.data[task]
            if count and data is None:raise ValueError('missing reservoir storage')
            if not count:continue
            for key in sorted(data):
                values=data[key]
                tensors=values if key=='obs' else {key:values} if isinstance(values,torch.Tensor) else None
                if tensors is not None:
                    for name,t in sorted(tensors.items()):
                        if t.device.type!='cpu' or (key in ('obs','actions') and t.dtype!=torch.float32):raise ValueError('reservoir must retain CPU FP32 observations/actions')
                        if t.shape[0]!=self.capacity:raise ValueError('reservoir capacity storage mismatch')
                        valid=t[:count]
                        if valid.is_floating_point():
                            for chunk in valid.split(256):
                                if not torch.isfinite(chunk).all():raise ValueError('nonfinite reservoir storage')
                        digest.update((task+key+name+str(t.dtype)+str(list(t.shape))).encode())
                        view=memoryview(valid.numpy()).cast('B')
                        for offset in range(0,len(view),8*1024*1024):digest.update(view[offset:offset+8*1024*1024])
                else:
                    if len(values)!=self.capacity:raise ValueError('reservoir metadata capacity mismatch')
                    for v in values[:count]:digest.update(json.dumps(v,sort_keys=True,separators=(',',':')).encode()+b'\0')
        digest.update(self.rng.get_state().numpy().tobytes())
        return dict(sha256=digest.hexdigest(),capacity=self.capacity,counts=counts,seen=dict(self.seen))
    def next_sample_manifest(self,n=256):
        state=self.rng.get_state()
        try:
            batch=self.sample_per_task(n);digest=hashlib.sha256()
            for key,value in sorted(batch.items()):
                tensors=value if key=='obs' else {key:value} if isinstance(value,torch.Tensor) else None
                if tensors is not None:
                    for name,t in sorted(tensors.items()):digest.update(name.encode()+memoryview(t.contiguous().numpy()).cast('B'))
                else:digest.update(json.dumps(value,sort_keys=True).encode())
            return digest.hexdigest()
        finally:self.rng.set_state(state)
