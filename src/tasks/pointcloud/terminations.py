"""Per-episode counters and exclusive termination masks."""
import math
import torch


class EpisodeEnds:
    def __init__(self, batch, device='cpu'):
        self.side = torch.zeros(batch, dtype=torch.long, device=device)
        self.drop = self.side.clone()
        self.top = self.side.clone()

    def reset(self, ids):
        self.side[ids] = self.drop[ids] = self.top[ids] = 0

    def update(self, body, feet, roll_pitch, top_contact_force, illegal_contact, flat,
               elapsed, stair_end=8., stair_height=8., step_depth=.2, step_height=.2):
        stair_end = torch.as_tensor(stair_end, device=body.device)
        stair_height = torch.as_tensor(stair_height, device=body.device)
        step_depth = torch.as_tensor(step_depth, device=body.device)
        step_height = torch.as_tensor(step_height, device=body.device)
        x, y, z = body.unbind(-1)
        stairs = ~flat
        self.side = torch.where(stairs & (x >= 0) & (y.abs() > .50), self.side+1, 0)
        ref = torch.where(x < 0, 0., (torch.floor(x/step_depth)+1)*step_height).clamp(max=stair_height)
        self.drop = torch.where(stairs & (z-ref < .15), self.drop+1, 0)
        region = torch.where(flat, (x < -5)|(x > 25)|(y.abs() > 5), (x < -2)|(x > stair_end+2)|(y.abs() > 2))
        failure = region | illegal_contact | ((torch.cos(roll_pitch[:,0])*torch.cos(roll_pitch[:,1])) < math.cos(math.radians(70))) | (self.side >= 5) | (self.drop >= 5)
        def inside(p):
            end = stair_end
            while end.ndim < p.ndim-1:
                end = end.unsqueeze(-1)
            return (p[..., 0] >= end+.02)&(p[..., 0] <= end+2-.02)&(p[..., 1].abs() <= .38)
        top = stairs & inside(body) & inside(feet).all(-1) & (z-stair_height >= .25) & (z-stair_height <= .65) & (roll_pitch.abs().amax(-1) <= math.radians(20)) & ((top_contact_force > 5).sum(-1) >= 2)
        self.top = torch.where(top, self.top+1, 0)
        success = (self.top >= 50) & ~failure
        timeout = (elapsed >= torch.where(flat, 30., 90.)) & ~failure & ~success
        return failure, success, timeout
