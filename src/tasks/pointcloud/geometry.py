"""Privileged collision-top queries; never an actor observation."""
import torch


def transform(points, origin, yaw, inverse=False):
    while origin.ndim < points.ndim:
        origin = origin.unsqueeze(-2)
        yaw = yaw.unsqueeze(-1)
    p = points - origin if inverse else points
    c, s = torch.cos(yaw), torch.sin(yaw)
    if inverse:
        s = -s
    out = torch.stack((c*p[..., 0]-s*p[..., 1], s*p[..., 0]+c*p[..., 1], p[..., 2]), -1)
    return out if inverse else out + origin


def stair_surfaces(count=40, height=.20, depth=.20, device='cpu'):
    """Rows xmin,xmax,ymin,ymax,top_z; only actual collision upper faces."""
    rows = [[-2, count*depth+2, -2, 2, 0]]
    rows += [[i*depth, (i+1)*depth, -.4, .4, (i+1)*height] for i in range(count)]
    rows += [[count*depth, count*depth+2, -.4, .4, count*height]]
    return torch.tensor(rows, device=device)


def query_top(points, surfaces, min_distance=-.01, max_distance=.40):
    """points [B,...,3]; surfaces [S,5] or [B,S,5], in terrain frame."""
    if surfaces.ndim == 2:
        surfaces = surfaces.unsqueeze(0)
    while surfaces.ndim < points.ndim + 1:
        surfaces = surfaces.unsqueeze(1)
    p = points.unsqueeze(-2)
    dz = p[..., 2] - surfaces[..., 4]
    ok = ((p[..., 0] >= surfaces[..., 0]) & (p[..., 0] <= surfaces[..., 1]) &
          (p[..., 1] >= surfaces[..., 2]) & (p[..., 1] <= surfaces[..., 3]) &
          (dz >= min_distance-1e-6) & (dz <= max_distance+1e-6))
    z, ids = torch.where(ok, surfaces[..., 4], -torch.inf).max(-1)
    valid = torch.isfinite(z)
    return torch.where(valid, z, 0), valid, torch.where(valid, ids, -1)


def fit_normal(points_world, valid):
    """Least squares world z=ax+by+c; mask insufficient/rank deficient fits."""
    center=(points_world*valid[...,None]).sum(-2,keepdim=True)/valid.sum(-1,keepdim=True).clamp(min=1)[...,None]
    centered=points_world-center
    x = torch.cat((centered[..., :2], torch.ones_like(points_world[..., :1])), -1)
    x = x * valid[..., None]
    z = centered[..., 2:3] * valid[..., None]
    good = (valid.sum(-1) >= 6) & (torch.linalg.matrix_rank(x) == 3)
    coeff = torch.linalg.pinv(x) @ z
    n = torch.cat((-coeff[..., :2, 0], torch.ones_like(coeff[..., :1, 0])), -1)
    return torch.nn.functional.normalize(n, dim=-1), good
