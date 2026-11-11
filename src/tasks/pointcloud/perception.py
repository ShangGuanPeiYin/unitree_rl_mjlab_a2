"""Synthetic dual LiDAR preprocessing; history stays in acquisition body frames."""
from dataclasses import dataclass
import math
import torch

@dataclass
class DualLidarPattern:
    seed: int = 42
    group: int = 0
    rear: bool = False

    def generate_rays(self, model=None, device='cpu'):
        if not 0 <= self.group < 5:
            raise ValueError('group must be in [0,4]')
        elevation = torch.linspace(-math.pi/4, math.pi/4, 128)
        azimuth = torch.arange(64) * (2*math.pi/64)
        el, az = torch.meshgrid(elevation, azimuth, indexing='ij')
        rays = torch.stack((el.cos()*az.cos(), el.cos()*az.sin(), el.sin()), -1).reshape(-1,3)
        # Pitch +30 degrees makes the forward optical axis point down.
        c, s = math.cos(math.pi/6), math.sin(math.pi/6)
        rotation = torch.tensor([[c,0,s],[0,1,0],[-s,0,c]])
        rays = rays @ rotation.T
        if self.rear:
            rays[:, :2] *= -1
        permutation = torch.randperm(8192, generator=torch.Generator().manual_seed(self.seed))
        rays = rays[torch.tensor_split(permutation,5)[self.group]].to(device)
        origins = torch.tensor([-.45 if self.rear else .45,0,.10], device=device).expand_as(rays).clone()
        return origins, rays


def hits_to_body(hit_pos_w, distances, body_pos_w, rotation_world_from_body, self_hits):
    """Filter first hits, never recast through robot occluders. self_hits is bool [B,N]."""
    points = torch.bmm(hit_pos_w-body_pos_w[:,None,:], rotation_world_from_body)
    mask = (distances >= .15) & (distances <= 10.) & ~self_hits
    return points, mask & torch.isfinite(points).all(-1)


def voxel_means(points, mask):
    lo = points.new_tensor([-1.2,-.6,-1.])
    hi = points.new_tensor([1.2,.6,.5])
    valid = mask & torch.isfinite(points).all(-1) & (points>=lo).all(-1) & (points<=hi).all(-1)
    p = points[valid]
    if not p.numel():
        return p
    keys = torch.floor((p-lo)/.10).long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.bincount(inverse)
    means = p.new_zeros((len(count),3))
    means.index_add_(0,inverse,p)
    return means/count[:,None]


def _voxel_batch(points, mask):
    # Inclusive ROI has 25*13*16 bins. Batch scatter avoids per-env host sync.
    lo=points.new_tensor([-1.2,-.6,-1.]); hi=points.new_tensor([1.2,.6,.5])
    valid=mask & torch.isfinite(points).all(-1) & (points>=lo).all(-1) & (points<=hi).all(-1)
    safe=torch.where(valid[...,None],points,lo)
    key=torch.floor((safe-lo)/.10).long()
    key=torch.minimum(key,torch.tensor([24,12,15],device=points.device))
    key=(key[...,0]*13+key[...,1])*16+key[...,2]
    count=points.new_zeros(len(points),5200)
    count.scatter_add_(1,key,valid.to(points.dtype))
    sums=points.new_zeros(len(points),5200,3)
    sums.scatter_add_(1,key[...,None].expand(-1,-1,3),torch.where(valid[...,None],points,0))
    means=sums/count.clamp_min(1)[...,None]
    # Fixed voxel order, sampled evenly from valid bins. No duplicate samples.
    order=torch.arange(5200,device=points.device)[None].expand(len(points),-1)
    order=torch.where(count>0,order,5200).sort(dim=1).values
    return means,order,(count>0).sum(1)


def preprocess_dual(front, front_mask, rear, rear_mask, chunk_size=128):
    """Batched voxel means; front/rear quotas 35 each, unused quota transfers.

    Bounded temporary allocation uses chunks; operations within a chunk remain
    batched on-device, including empty inputs and unequal per-environment counts.
    """
    if chunk_size<1:
        raise ValueError('chunk_size must be positive')
    out=front.new_zeros(len(front),70,3)
    mask=torch.zeros(len(front),70,dtype=torch.bool,device=front.device)
    slots=torch.arange(70,device=front.device)[None]
    for start in range(0,len(front),chunk_size):
        end=min(start+chunk_size,len(front))
        fm,fo,fc=_voxel_batch(front[start:end],front_mask[start:end])
        rm,ro,rc=_voxel_batch(rear[start:end],rear_mask[start:end])
        nf=fc.clamp_max(35); nr=rc.clamp_max(35)
        nf=nf+torch.minimum(70-nf-nr,fc-nf)
        nr=nr+torch.minimum(70-nf-nr,rc-nr)
        def select(means,order,count,n,indices):
            rank=torch.div(indices.clamp_min(0)*(count-1).clamp_min(0)[:,None],(n-1).clamp_min(1)[:,None],rounding_mode='floor')
            rank=rank.clamp(0,5199)
            voxel=order.gather(1,rank).clamp_max(5199)
            return means.gather(1,voxel[...,None].expand(-1,-1,3))
        first=select(fm,fo,fc,nf,slots)
        second=select(rm,ro,rc,nr,slots-nf[:,None])
        valid=slots<(nf+nr)[:,None]
        chosen=torch.where((slots<nf[:,None])[...,None],first,second)
        out[start:end]=torch.where(valid[...,None],chosen,0)
        mask[start:end]=valid
    return out,mask


class PointCloudHistory:
    """Newest group first. Delay is 0..3 control steps, per environment.

    advance() occurs exactly once per policy step. No truth pose or labels enter
    this buffer. Queue capacity includes the five groups plus maximum delay.
    """
    def __init__(self, num_envs, device='cpu', dt=.02):
        self.dt = dt
        self.points = torch.zeros(num_envs,8,70,3,device=device)
        self.mask = torch.zeros(num_envs,8,70,dtype=torch.bool,device=device)
        self.valid = torch.zeros(num_envs,8,dtype=torch.bool,device=device)
        self.delay = torch.zeros(num_envs,dtype=torch.long,device=device)
        self.scan_group = torch.zeros(num_envs,dtype=torch.long,device=device)

    def reset(self, env_ids, delay_steps=0):
        delay = torch.as_tensor(delay_steps,device=self.delay.device)
        if ((delay<0)|(delay>3)).any():
            raise ValueError('delay_steps must be in [0,3]')
        self.points[env_ids] = 0
        self.mask[env_ids] = False
        self.valid[env_ids] = False
        self.delay[env_ids] = delay
        self.scan_group[env_ids] = 0

    def advance(self, points, mask):
        if points.shape != self.points[:,0].shape or mask.shape != self.mask[:,0].shape:
            raise ValueError('expected [B,70,3] points and [B,70] mask')
        self.points[:,1:] = self.points[:,:-1].clone()
        self.mask[:,1:] = self.mask[:,:-1].clone()
        self.valid[:,1:] = self.valid[:,:-1].clone()
        finite = mask.bool() & torch.isfinite(points).all(-1)
        self.points[:,0] = torch.where(finite[...,None],points,0)
        self.mask[:,0] = finite
        self.valid[:,0] = True
        self.scan_group = (self.scan_group+1)%5

    def read(self):
        index = self.delay[:,None]+torch.arange(5,device=self.delay.device)[None]
        batch = torch.arange(len(self.delay),device=self.delay.device)[:,None]
        points, mask = self.points[batch,index], self.mask[batch,index]
        ages = index.to(points.dtype)*self.dt
        return points,mask,ages


def explicit_lidar_cfg(name, frame, *, rear=False, seed=42):
    """Two sensors only; automatic sim.sense deliberately does not cast them."""
    from mjlab.sensor.raycast_sensor import RayCastSensorCfg, RayCastSensor
    class ExplicitSensor(RayCastSensor):
        def raycast_kernel(self, rc):
            pass  # Explicit adapter owns timing, including terminal observations.
        def explicit_cast(self, rc):
            super().raycast_kernel(rc)
    @dataclass
    class ExplicitCfg(RayCastSensorCfg):
        def build(self):
            return ExplicitSensor(self)
    return ExplicitCfg(name=name, frame=frame,
                       pattern=DualLidarPattern(seed=seed,group=0,rear=rear),
                       exclude_parent_body=False, include_geom_groups=(0,3),
                       max_distance=10., ray_alignment='base')


class DualLidarScanAdapter:
    """Two explicit sensors, one 1639-slot group per environment and control step.

    Group padding never becomes a valid return. All robot collision geoms remain
    occluders. Public RayCastData lacks geometry IDs, hence pinned mjlab internals.
    """
    def __init__(self, sim, sensors, robot_geom_ids):
        self.sim, self.sensors = sim, sensors
        self.robot_geom_ids = torch.as_tensor(robot_geom_ids,device=sim.device)
        if len(sensors)!=2:
            raise ValueError('exactly two explicit sensors required')
        self.patterns=[]
        for sensor in sensors:
            if sensor.cfg.exclude_parent_body or not hasattr(sensor,'explicit_cast'):
                raise ValueError('use explicit_lidar_cfg with self occlusion enabled')
            pattern=sensor.cfg.pattern
            origins=[]; directions=[]; valid=[]
            for group in range(5):
                o,d=DualLidarPattern(pattern.seed,group,pattern.rear).generate_rays(device=sim.device)
                n=len(d)
                origins.append(torch.cat((o,o[:1639-n])))
                directions.append(torch.cat((d,d[:1639-n])))
                valid.append(torch.arange(1639,device=sim.device)<n)
            self.patterns.append((torch.stack(origins),torch.stack(directions),torch.stack(valid)))

    def scan(self, groups, body_pos_w, rotation_world_from_body):
        import warp as wp
        import mujoco_warp as mjwarp
        ctx = self.sim._sensor_context
        if ctx is None:
            raise RuntimeError('registered ray sensors require SensorContext')
        mjwarp.refit_bvh(self.sim.wp_model,self.sim.wp_data,ctx.render_context)
        processed=[]
        for sensor,(origins,directions,valid) in zip(self.sensors,self.patterns):
            world_origins=body_pos_w[:,None]+torch.bmm(origins[groups],rotation_world_from_body.transpose(1,2))
            world_rays=torch.bmm(directions[groups],rotation_world_from_body.transpose(1,2))
            wp.to_torch(sensor._ray_pnt).copy_(world_origins)
            wp.to_torch(sensor._ray_vec).copy_(world_rays)
            sensor._cached_world_origins=world_origins
            sensor._cached_world_rays=world_rays
            sensor._cached_frame_pos=body_pos_w
            sensor._cached_frame_mat=rotation_world_from_body
            sensor.explicit_cast(ctx.render_context)
            sensor.postprocess_rays()
            data=sensor.data
            geom=wp.to_torch(sensor._ray_geomid)
            self_hits=torch.isin(geom,self.robot_geom_ids)
            points,mask=hits_to_body(data.hit_pos_w,data.distances,body_pos_w,rotation_world_from_body,self_hits)
            processed.extend((points,mask & valid[groups]))
        return preprocess_dual(*processed)
