"""Actual geom-pair contact extraction for the independent ideal-stairs task."""
import math
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

def support_from_top_forces(tops, geom_top, geom_tile):
    """Aggregate actual upward forces on coplanar faces in the assigned tile."""
    _, groups = torch.unique(torch.stack((geom_tile.float(),geom_top),-1),dim=0,return_inverse=True)
    combined=torch.zeros_like(tops)
    combined.scatter_add_(-1,groups[None,None].expand_as(tops),tops)
    upward, group=combined.max(-1)
    eligible=groups[None,None,:]==group[:,:,None]
    representative=torch.where(eligible,tops,-torch.inf).argmax(-1)
    return upward,representative


LEGS = ('FL', 'FR', 'RL', 'RR')


def reduce_contacts(geom, world, point, normal, force, valid, foot_lookup,
                    nonfoot_lookup, terrain_tile, geom_top, assigned_tiles, num_links):
    """Forces point geom[0] -> geom[1]; retain only assigned-tile contacts.

    Returns all foot forces, upward top forces by foot/geom, and nonfoot link forces.
    The contact point is MuJoCo's actual midpoint, not a foot position proxy.
    """
    b, g = assigned_tiles.numel(), terrain_tile.numel()
    device = force.device
    foot_force = torch.zeros(b * 4, 3, device=device)
    top_force = torch.zeros(b * 4 * g, device=device)
    link_force = torch.zeros(b * num_links, 3, device=device)
    for robot_side in (0, 1):
        rg, tg = geom[:, robot_side], geom[:, 1 - robot_side]
        sign = 1. if robot_side == 1 else -1.
        f, n = force * sign, normal * sign
        selected = valid & (terrain_tile[tg] == assigned_tiles[world]) & (terrain_tile[tg] >= 0)
        feet = foot_lookup[rg]
        use = selected & (feet >= 0)
        fi = world[use] * 4 + feet[use]
        foot_force.index_add_(0, fi, f[use])
        top = use & ((point[:, 2] - geom_top[tg]).abs() <= .005) & (n[:, 2] >= math.cos(math.pi / 6))
        ti = (world[top] * 4 + feet[top]) * g + tg[top]
        top_force.index_add_(0, ti, f[top, 2])
        links = nonfoot_lookup[rg]
        use = selected & (links >= 0)
        link_force.index_add_(0, world[use] * num_links + links[use], f[use])
    return foot_force.reshape(b, 4, 3), top_force.reshape(b, 4, g), link_force.reshape(b, num_links, 3)


class ActualContacts:
    def __init__(self, sim, tile_by_geom, geom_top, device):
        self.sim, self.device = sim, device
        m, d = sim.mj_model, sim.wp_data
        names = [m.geom(i).name or '' for i in range(m.ngeom)]
        self.foot_lookup = torch.full((m.ngeom,), -1, device=device, dtype=torch.long)
        for k, leg in enumerate(LEGS):
            matches = [i for i, n in enumerate(names) if n.endswith(leg + '_foot_collision')]
            if len(matches) != 1:
                raise ValueError(f'Expected one {leg} football geom, found {matches}')
            self.foot_lookup[matches[0]] = k
        nonfoot = [i for i,n in enumerate(names) if 'collision' in n and int(self.foot_lookup[i]) < 0]
        bodies = sorted(set(int(m.geom_bodyid[i]) for i in nonfoot))
        self.link_names = [m.body(i).name for i in bodies]
        self.nonfoot_lookup = torch.full((m.ngeom,), -1, device=device, dtype=torch.long)
        for i in nonfoot:
            self.nonfoot_lookup[i] = bodies.index(int(m.geom_bodyid[i]))
        self.body_link_mask = torch.tensor([name.endswith('base_link') for name in self.link_names], device=device)
        if not self.body_link_mask.any():
            raise ValueError('Missing monitored base_link geometry')
        self.tile_by_geom = torch.as_tensor(tile_by_geom, device=device, dtype=torch.long)
        self.geom_top = torch.as_tensor(geom_top, device=device, dtype=torch.float32)
        with wp.ScopedDevice(device):
            self.ids = wp.array(np.arange(d.contact.geom.shape[0], dtype=np.int32), dtype=wp.int32)
            self.forces = wp.zeros(self.ids.size, dtype=wp.spatial_vector)
        self.top_forces=torch.zeros(sim.num_envs,4,m.ngeom,device=device)
        self.counts = torch.zeros(sim.num_envs, 4, dtype=torch.long, device=device)
        self.nonfoot_duration = torch.zeros(sim.num_envs, len(bodies), device=device)
        self.support_geom = torch.full((sim.num_envs,4),-1,device=device,dtype=torch.long)
        self.support = self.counts.bool()
        self.found = self.counts.bool()
        self.contact_duration = self.counts.float()
        self.foot_force = torch.zeros(sim.num_envs,4,3,device=device)
        self.link_force = torch.zeros(sim.num_envs,len(bodies),3,device=device)

    def reset(self, ids):
        self.top_forces[ids]=0
        self.counts[ids] = 0
        self.nonfoot_duration[ids] = 0
        self.support_geom[ids] = -1
        self.support[ids] = False
        self.found[ids] = False
        self.contact_duration[ids] = 0
        self.foot_force[ids] = 0
        self.link_force[ids] = 0

    def sample(self, assigned_tiles, physics_dt=None, control=False):
        d = self.sim.wp_data
        if int(wp.to_torch(d.nacon)[0]) > self.ids.size:
            raise RuntimeError('Contact buffer overflow')
        self.forces.zero_()
        mjw.contact_force(self.sim.wp_model, d, self.ids, True, self.forces)
        count = wp.to_torch(d.nacon)[0]
        valid = (torch.arange(self.ids.size,device=self.device) < count) & ((wp.to_torch(d.contact.type) & 1) != 0)
        # Ignore inactive capacity slots before indexing world/tile metadata.
        geom = wp.to_torch(d.contact.geom)[valid].long()
        world = wp.to_torch(d.contact.worldid)[valid].long()
        force = wp.to_torch(self.forces)[valid,:3]
        self.foot_force, tops, self.link_force = reduce_contacts(
            geom,world,wp.to_torch(d.contact.pos)[valid],wp.to_torch(d.contact.frame)[valid,0],
            force,torch.ones_like(world,dtype=torch.bool),self.foot_lookup,self.nonfoot_lookup,
            self.tile_by_geom,self.geom_top,assigned_tiles,len(self.link_names))
        self.top_forces=tops
        self.found.zero_()
        for robot_side in (0,1):
            rg,tg=geom[:,robot_side],geom[:,1-robot_side]
            feet=self.foot_lookup[rg]
            selected=(feet>=0)&(self.tile_by_geom[tg]==assigned_tiles[world])&(self.tile_by_geom[tg]>=0)
            self.found[world[selected],feet[selected]]=True
        if physics_dt is not None:
            self.contact_duration=torch.where(self.found,self.contact_duration+physics_dt,0.)
            hit = self.link_force.norm(dim=-1) > 10
            self.nonfoot_duration = torch.where(hit,self.nonfoot_duration+physics_dt,0.)
        if control:
            upward, gid = support_from_top_forces(tops,self.geom_top,self.tile_by_geom)
            raw = upward > 5
            self.counts = torch.where(raw,self.counts+1,0)
            self.support = self.counts >= 2
            self.support_geom = torch.where(self.support,gid,-1)
        return self
