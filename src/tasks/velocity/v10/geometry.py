"""Finite collision primitives are the sole source of map and support geometry."""
from dataclasses import dataclass
from functools import lru_cache
import torch

@dataclass(frozen=True)
class Terrain:
    kind: str
    N: int = 0
    H: float = 0.
    W: float = 4.
    mode: str = 'G'

@dataclass(frozen=True)
class Primitive:
    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    top: float
    thickness: float
    @property
    def center(self):
        return ((self.x0+self.x1)/2,(self.y0+self.y1)/2,self.top-self.thickness/2)
    @property
    def half(self):
        return ((self.x1-self.x0)/2,(self.y1-self.y0)/2,self.thickness/2)

@lru_cache(None)
def primitives(t):
    if t.kind=='flat': return (Primitive('ground',-2,22,-4,4,0,.1),)
    front = .2*t.N if t.kind=='stairs' else 0.
    high = front if t.kind=='stairs' else t.H
    out=[]
    if t.kind=='stairs':
        out.append(Primitive('approach_ground',-2,front,-2,2,0,.1))
        out += [Primitive(f'tread_{k}',(k-1)*.2,k*.2,-.4,.4,k*.2,.01) for k in range(1,t.N+1)]
    if t.mode=='G':out.append(Primitive('under_platform',front,front+14,-2,2,0,.1))
    out.append(Primitive('platform',front,front+14,-t.W/2,t.W/2,high,.01))
    return tuple(out)


@lru_cache(None)
def _primitive_tensor(t,device,dtype):
    return torch.tensor([[p.x0,p.x1,p.y0,p.y1,p.top] for p in primitives(t)],device=device,dtype=dtype)


def query_local(points,t):
    p=_primitive_tensor(t,points.device,points.dtype)
    x,y=points[...,0,None],points[...,1,None]
    hit=(x>=p[:,0])&(x<=p[:,1])&(y>=p[:,2])&(y<=p[:,3])
    out=torch.where(hit,p[:,4],-torch.inf).amax(-1)
    valid=torch.isfinite(out)
    return torch.where(valid,out,torch.zeros_like(out)),valid

@lru_cache(None)
def _exposed_segments(t,height):
    ps=[p for p in primitives(t) if abs(p.top-height)<1e-6]
    if not ps:return ()
    xs=sorted({v for p in ps for v in (p.x0,p.x1)})
    ys=sorted({v for p in ps for v in (p.y0,p.y1)})
    cells={(i,j) for i in range(len(xs)-1) for j in range(len(ys)-1)
           if any(p.x0<=(xs[i]+xs[i+1])/2<=p.x1 and p.y0<=(ys[j]+ys[j+1])/2<=p.y1 for p in ps)}
    edges=[]
    for i,j in cells:
        if (i-1,j) not in cells:edges.append((xs[i],ys[j],xs[i],ys[j+1]))
        if (i+1,j) not in cells:edges.append((xs[i+1],ys[j],xs[i+1],ys[j+1]))
        if (i,j-1) not in cells:edges.append((xs[i],ys[j],xs[i+1],ys[j]))
        if (i,j+1) not in cells:edges.append((xs[i],ys[j+1],xs[i+1],ys[j+1]))
    return tuple(edges)


@lru_cache(None)
def _segments_tensor(t,device,dtype):
    return torch.tensor([(*edge,h) for h in {p.top for p in primitives(t)}
                         for edge in _exposed_segments(t,h)],device=device,dtype=dtype)


def safe_top(points,height,t,inset=.042):
    """Euclidean erosion of the real coplanar union; internal seams disappear."""
    p=_primitive_tensor(t,points.device,points.dtype)
    x,y=points[...,0,None],points[...,1,None]
    inside=((x>=p[:,0])&(x<=p[:,1])&(y>=p[:,2])&(y<=p[:,3])&((height[...,None]-p[:,4]).abs()<1e-5)).any(-1)
    e=_segments_tensor(t,points.device,points.dtype)
    nearx=torch.maximum(e[:,0],torch.minimum(x,e[:,2]))
    neary=torch.maximum(e[:,1],torch.minimum(y,e[:,3]))
    distance=(x-nearx).square()+(y-neary).square()
    distance=torch.where((height[...,None]-e[:,4]).abs()<1e-5,distance,torch.inf)
    return inside & (distance.amin(-1)>=inset**2-1e-9)


def route_height(x,t):
    if t.kind=='flat':return torch.zeros_like(x)
    if t.kind=='platform':return torch.full_like(x,t.H)
    return torch.where(x<0,torch.zeros_like(x),.2*(torch.floor(x/.2)+1).clamp(max=t.N))


def corridor(points,t):
    x,y=points[...,0],points[...,1]
    if t.kind=='flat':return (x>=-2)&(x<=22)&(y.abs()<=4)
    if t.kind=='platform':return (x>=0)&(x<=14)&(y.abs()<=t.W/2)
    front=.2*t.N
    return (x>=-2)&(x<=front+14)&torch.where(x<0,y.abs()<=2,torch.where(x<front,y.abs()<=.4,y.abs()<=t.W/2))


def terrain_from_case(c):
    task=c['task']
    return Terrain('flat' if task.startswith('F_') else 'platform' if task.startswith('P_') else 'stairs',
                   N=int(c.get('stairs',c.get('N',0))), H=float(c.get('height',c.get('H',0))),
                   W=float(c.get('width',c.get('W',4))),mode=c.get('encoding',c.get('mode','G')))
