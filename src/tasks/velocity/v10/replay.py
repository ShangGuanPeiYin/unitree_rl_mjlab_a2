"""Six independent standard CPU/FP32 reservoirs with reproducible sampling."""
import copy
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
        batch=snapshot(batch)
        for task in TASKS:
            ids=[i for i,k in enumerate(batch['task']) if k==task]
            if not ids: continue
            incoming=index_batch(batch,ids)
            if self.data[task] is None:
                # Allocate bounded contiguous storage once; metadata remains detached.
                n=self.capacity
                self.data[task]={k:({name:torch.empty((n,*t.shape[1:]),dtype=torch.float32) for name,t in v.items()} if k=='obs' else
                                    torch.empty((n,*v.shape[1:]),dtype=v.dtype) if isinstance(v,torch.Tensor) else [None]*n) for k,v in incoming.items()}
            dst=self.data[task]
            for i in range(len(ids)):
                self.seen[task]+=1; count=self.seen[task]
                j=count-1 if count<=self.capacity else int(torch.randint(count,(1,),generator=self.rng))
                if j>=self.capacity: continue
                for k,v in incoming.items():
                    if k=='obs':
                        for name,t in v.items(): dst[k][name][j].copy_(t[i])
                    elif isinstance(v,torch.Tensor): dst[k][j].copy_(v[i])
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
            current_task=index_batch(current,ids)
            take=lambda n:index_batch(current_task,torch.randint(len(ids),(n,),generator=self.rng))
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
