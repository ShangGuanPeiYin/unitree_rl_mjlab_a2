"""v0.8 contact extraction and filtered contact state (self-contained, no v0.2).

SDS v0.8 section 3.4:

    raw_contact      = contacts.found AND foot_force_world_z > 5 N
    filtered_contact = raw_contact OR previous_raw_contact

The contact-force reduction below is the low-level MuJoCo/Warp computation the
frozen v0.2 benchmark used (it is not a curriculum rule); it is re-implemented
inside the v0.8 production line so that ``v08`` never imports ``v02`` or
``v02_formal`` at runtime.  All tile-specific values (tile ids, geom top faces)
come from :mod:`src.tasks.velocity.v08.terrain`.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from . import spec
from .terrain import SAFE_INSET  # noqa: F401  (single-source re-export)

#: SDS 3.4 foot force threshold (world z), read from the single frozen source.
SUPPORT_FORCE_GT = float(spec.SUPPORT_FORCE_GT)
#: Number of consecutive control samples a top-face contact must persist.
SUPPORT_DEBOUNCE = int(spec.SUPPORT_DEBOUNCE)
FOOT_ORDER = tuple(spec.FOOT_ORDER)


def support_from_top_forces(tops, geom_top, geom_tile):
    """Aggregate upward forces on coplanar faces inside the assigned tile."""
    _, groups = torch.unique(torch.stack((geom_tile.float(), geom_top), -1),
                             dim=0, return_inverse=True)
    combined = torch.zeros_like(tops)
    combined.scatter_add_(-1, groups[None, None].expand_as(tops), tops)
    upward, group = combined.max(-1)
    eligible = groups[None, None, :] == group[:, :, None]
    representative = torch.where(eligible, tops, -torch.inf).argmax(-1)
    return upward, representative


def reduce_contacts(geom, world, point, normal, force, valid, foot_lookup,
                    nonfoot_lookup, terrain_tile, geom_top, assigned_tiles,
                    num_links):
    """Forces point ``geom[0] -> geom[1]``; retain only assigned-tile contacts.

    Returns per-foot forces, upward top forces by foot/geom, and non-foot link
    forces.  The contact point is MuJoCo's actual midpoint, not a foot-position
    proxy.
    """
    b, g = assigned_tiles.numel(), terrain_tile.numel()
    device = force.device
    foot_force = torch.zeros(b * 4, 3, device=device)
    top_force = torch.zeros(b * 4 * g, device=device)
    link_force = torch.zeros(b * num_links, 3, device=device)
    for robot_side in (0, 1):
        rg, tg = geom[:, robot_side], geom[:, 1 - robot_side]
        sign = 1.0 if robot_side == 1 else -1.0
        f, n = force * sign, normal * sign
        selected = (valid & (terrain_tile[tg] == assigned_tiles[world])
                    & (terrain_tile[tg] >= 0))
        feet = foot_lookup[rg]
        use = selected & (feet >= 0)
        fi = world[use] * 4 + feet[use]
        foot_force.index_add_(0, fi, f[use])
        top = (use & ((point[:, 2] - geom_top[tg]).abs() <= spec.CONTACT_TOP_TOLERANCE)
               & (n[:, 2] >= math.cos(spec.SUPPORT_NORMAL_MAX_ANGLE)))
        ti = (world[top] * 4 + feet[top]) * g + tg[top]
        top_force.index_add_(0, ti, f[top, 2])
        links = nonfoot_lookup[rg]
        use = selected & (links >= 0)
        link_force.index_add_(0, world[use] * num_links + links[use], f[use])
    return (foot_force.reshape(b, 4, 3), top_force.reshape(b, 4, g),
            link_force.reshape(b, num_links, 3))


class ActualContacts:
    """Raw geom-pair contact extraction for the v0.8 scene."""

    def __init__(self, sim, tile_by_geom, geom_top, device):
        self.sim, self.device = sim, device
        model = sim.mj_model
        data = sim.wp_data
        names = [model.geom(i).name or '' for i in range(model.ngeom)]
        self.foot_lookup = torch.full((model.ngeom,), -1, device=device,
                                      dtype=torch.long)
        for k, leg in enumerate(FOOT_ORDER):
            matches = [i for i, name in enumerate(names)
                       if name.endswith(leg + '_foot_collision')]
            if len(matches) != 1:
                raise ValueError(f'expected one {leg} foot collision geom, got {matches}')
            self.foot_lookup[matches[0]] = k
        nonfoot = [i for i, name in enumerate(names)
                   if 'collision' in name and int(self.foot_lookup[i]) < 0]
        bodies = sorted(set(int(model.geom_bodyid[i]) for i in nonfoot))
        self.link_names = [model.body(i).name for i in bodies]
        self.nonfoot_lookup = torch.full((model.ngeom,), -1, device=device,
                                         dtype=torch.long)
        for i in nonfoot:
            self.nonfoot_lookup[i] = bodies.index(int(model.geom_bodyid[i]))
        self.body_link_mask = torch.tensor(
            [name.endswith('base_link') for name in self.link_names], device=device)
        if not self.body_link_mask.any():
            raise ValueError('missing monitored base_link geometry')
        self.tile_by_geom = torch.as_tensor(tile_by_geom, device=device, dtype=torch.long)
        self.geom_top = torch.as_tensor(geom_top, device=device, dtype=torch.float32)
        with wp.ScopedDevice(device):
            self.ids = wp.array(np.arange(data.contact.geom.shape[0], dtype=np.int32),
                                dtype=wp.int32)
            self.forces = wp.zeros(self.ids.size, dtype=wp.spatial_vector)
        self.top_forces = torch.zeros(sim.num_envs, 4, model.ngeom, device=device)
        self.counts = torch.zeros(sim.num_envs, 4, dtype=torch.long, device=device)
        self.nonfoot_duration = torch.zeros(sim.num_envs, len(bodies), device=device)
        self.support_geom = torch.full((sim.num_envs, 4), -1, device=device,
                                       dtype=torch.long)
        self.support = self.counts.bool()
        self.found = self.counts.bool()
        self.contact_duration = self.counts.float()
        self.foot_force = torch.zeros(sim.num_envs, 4, 3, device=device)
        self.link_force = torch.zeros(sim.num_envs, len(bodies), 3, device=device)

    def reset(self, ids):
        self.top_forces[ids] = 0
        self.counts[ids] = 0
        self.nonfoot_duration[ids] = 0
        self.support_geom[ids] = -1
        self.support[ids] = False
        self.found[ids] = False
        self.contact_duration[ids] = 0
        self.foot_force[ids] = 0
        self.link_force[ids] = 0

    def sample(self, assigned_tiles, physics_dt=None, control=False):
        data = self.sim.wp_data
        if int(wp.to_torch(data.nacon)[0]) > self.ids.size:
            raise RuntimeError('contact buffer overflow')
        self.forces.zero_()
        with wp.ScopedDevice(self.device):
            mjw.contact_force(self.sim.wp_model, data, self.ids, True, self.forces)
        count = wp.to_torch(data.nacon)[0]
        valid = ((torch.arange(self.ids.size, device=self.device) < count)
                 & ((wp.to_torch(data.contact.type) & 1) != 0))
        geom = wp.to_torch(data.contact.geom)[valid].long()
        world = wp.to_torch(data.contact.worldid)[valid].long()
        force = wp.to_torch(self.forces)[valid, :3]
        self.foot_force, tops, self.link_force = reduce_contacts(
            geom, world, wp.to_torch(data.contact.pos)[valid],
            wp.to_torch(data.contact.frame)[valid, 0], force,
            torch.ones_like(world, dtype=torch.bool), self.foot_lookup,
            self.nonfoot_lookup, self.tile_by_geom, self.geom_top, assigned_tiles,
            len(self.link_names))
        self.top_forces = tops
        self.found.zero_()
        for robot_side in (0, 1):
            rg, tg = geom[:, robot_side], geom[:, 1 - robot_side]
            feet = self.foot_lookup[rg]
            selected = ((feet >= 0) & (self.tile_by_geom[tg] == assigned_tiles[world])
                        & (self.tile_by_geom[tg] >= 0))
            self.found[world[selected], feet[selected]] = True
        if physics_dt is not None:
            self.contact_duration = torch.where(
                self.found, self.contact_duration + physics_dt, 0.0)
            hit = self.link_force.norm(dim=-1) > spec.NONFOOT_FORCE_GT
            self.nonfoot_duration = torch.where(
                hit, self.nonfoot_duration + physics_dt, 0.0)
        if control:
            upward, gid = support_from_top_forces(tops, self.geom_top,
                                                  self.tile_by_geom)
            raw = upward > spec.SUPPORT_FORCE_GT
            self.counts = torch.where(raw, self.counts + 1, torch.zeros_like(self.counts))
            self.support = self.counts >= SUPPORT_DEBOUNCE
            self.support_geom = torch.where(self.support, gid,
                                            torch.full_like(gid, -1))
        return self


class ContactState:
    """Per-foot filtered contact state over the v0.8 contact reduction."""

    def __init__(self, sim, tile_by_geom, geom_top, device):
        self.inner = ActualContacts(sim, tile_by_geom, geom_top, device)
        self.device = torch.device(device)
        self.num_envs = int(sim.num_envs)
        self.raw = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=device)
        self.previous_raw = torch.zeros_like(self.raw)
        self.filtered = torch.zeros_like(self.raw)
        self.first_contact = torch.zeros_like(self.raw)
        self.overflow = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    # ------------------------------------------------------------- properties
    @property
    def found(self):
        return self.inner.found

    @property
    def foot_force(self):
        return self.inner.foot_force

    @property
    def foot_force_world_z(self):
        return self.inner.foot_force[..., 2]

    @property
    def link_force(self):
        return self.inner.link_force

    @property
    def body_link_mask(self):
        return self.inner.body_link_mask

    @property
    def nonfoot_duration(self):
        return self.inner.nonfoot_duration

    @property
    def contact_duration(self):
        return self.inner.contact_duration

    @property
    def top_forces(self):
        return self.inner.top_forces

    @property
    def support(self):
        return self.inner.support

    @property
    def support_geom(self):
        return self.inner.support_geom

    @property
    def link_names(self):
        return self.inner.link_names

    # ----------------------------------------------------------------- sample
    def check_overflow(self) -> bool:
        """True when the live contact count exceeds the preallocated buffer."""
        try:
            count = int(wp.to_torch(self.inner.sim.wp_data.nacon)[0])
        except Exception:  # pragma: no cover - only with a non-warp mock
            return False
        return count > int(self.inner.ids.size)

    def begin_control_step(self):
        """Only the next control boundary may release the overflow latch."""
        self.overflow.zero_()

    def sample(self, assigned_tiles, physics_dt=None, control=False):
        if self.check_overflow():
            # Never index past the buffer. Retain the latch through all later
            # samples; Env fails closed before it can consume stale contacts.
            self.overflow[:] = True
            return self
        if self.overflow.any():
            return self
        self.inner.sample(assigned_tiles, physics_dt=physics_dt, control=control)
        raw = self.inner.found & (self.inner.foot_force[..., 2] > SUPPORT_FORCE_GT)
        self.previous_raw = self.raw.clone()
        self.first_contact = raw & ~self.raw
        self.raw = raw
        self.filtered = raw | self.previous_raw
        return self

    def reset(self, ids):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self.inner.reset(ids)
        self.raw[ids] = False
        self.previous_raw[ids] = False
        self.filtered[ids] = False
        self.first_contact[ids] = False
        self.overflow[ids] = False

    # ---------------------------------------------------------------- support
    def platform_support(self, tile_ids, geom_step, geom_tile, geom_bounds):
        """Actual ``> 5 N`` top-face contact with the two-metre platform only."""
        from .terrain import build_terrain_specs
        specs = build_terrain_specs()
        n = tile_ids.new_tensor([int(s.N) for s in specs])
        step_n = n[tile_ids]
        is_platform = ((geom_step[None, :] == step_n[:, None])
                       & (geom_tile[None, :] == tile_ids[:, None])
                       & ((geom_bounds[:, 1] - geom_bounds[:, 0])[None, :] > spec.PLATFORM_CONTACT_MIN_LENGTH))
        force = torch.where(is_platform[:, None, :], self.inner.top_forces,
                            -torch.inf)
        upward, surface_ids = force.max(-1)
        return upward, surface_ids

    def state_dict(self):
        return dict(raw=self.raw.detach().to('cpu').tolist(),
                    previous_raw=self.previous_raw.detach().to('cpu').tolist(),
                    filtered=self.filtered.detach().to('cpu').tolist(),
                    overflow=self.overflow.detach().to('cpu').tolist())


def support_normal_ok(cos_normal, *, minimum=None):
    """Top-face normal test (``cos(30 deg)`` by default), kept in one place."""
    limit = math.cos(spec.SUPPORT_NORMAL_MAX_ANGLE) if minimum is None else float(minimum)
    return cos_normal >= limit
