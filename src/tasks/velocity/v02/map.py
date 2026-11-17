"""Batched ideal map cache, with sample-frame hold (no motion compensation)."""
import torch
from .terrain import query_top_surface

class IdealHeightMap:
    def __init__(self, num_envs, device='cpu'):
        self.num_envs = num_envs
        self.device = torch.device(device)
        x = -.8+(torch.arange(48, device=device)+.5)*.05
        y = -.8+(torch.arange(32, device=device)+.5)*.05
        self.grid = torch.stack(torch.meshgrid(x,y,indexing='ij'),-1)
        self.map = torch.zeros(num_envs,3,48,32,device=device)
        self.map[:,2] = 1.
        self.steps = torch.full((num_envs,),5,device=device,dtype=torch.long)
        self.sample_position = torch.zeros(num_envs,3,device=device)
        self.sample_yaw = torch.zeros(num_envs,device=device)
        self.sample_points_world = torch.zeros(num_envs,48,32,2,device=device)
    def update(self, base_pos, yaw, tile_ids, env_ids=None, force=False):
        ids = torch.arange(self.num_envs,device=self.device) if env_ids is None else torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if not force:
            self.steps[ids] += 1
            valid = self.map[ids,1].bool()
            self.map[ids,2] = torch.where(valid,(self.steps[ids,None,None]*.02/.5).clamp(0,1),1.)
        refresh = ids if force else ids[self.steps[ids]>=5]
        if len(refresh):
            pos, angle = base_pos[refresh], yaw[refresh]
            c,s = angle.cos()[:,None,None], angle.sin()[:,None,None]
            gx,gy = self.grid[...,0],self.grid[...,1]
            points = torch.stack((pos[:,None,None,0]+c*gx-s*gy,pos[:,None,None,1]+s*gx+c*gy),-1)
            z,valid = query_top_surface(points,tile_ids[refresh])
            self.map[refresh,0] = torch.where(valid,((z-pos[:,None,None,2])/1.5).clamp(-1,1),0.)
            self.map[refresh,1] = valid.float()
            self.map[refresh,2] = (~valid).float()
            self.steps[refresh] = 0
            self.sample_position[refresh] = pos
            self.sample_yaw[refresh] = angle
            self.sample_points_world[refresh] = points
        return self.map
    def reset(self, base_pos, yaw, tile_ids, env_ids=None):
        return self.update(base_pos,yaw,tile_ids,env_ids,force=True)

class ProprioHistory:
    def __init__(self, num_envs, device='cpu'):
        self.history = torch.zeros(num_envs,5,45,device=device)
        self.initialized = torch.zeros(num_envs,dtype=torch.bool,device=device)
    def update(self, observation):
        fresh = ~self.initialized
        self.history = torch.cat((self.history[:,1:],observation[:,None]),1)
        self.history[fresh] = observation[fresh,None].expand(-1,5,-1)
        self.initialized[:] = True
        return self.history.flatten(1)
    def reset(self, observation, env_ids):
        ids = torch.as_tensor(env_ids,device=self.history.device,dtype=torch.long)
        self.history[ids] = observation[ids,None].expand(-1,5,-1)
        self.initialized[ids] = True
        return self.history.flatten(1)
