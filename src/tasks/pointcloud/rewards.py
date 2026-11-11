"""SDS reward primitives; inputs are privileged and stay outside actor history."""
import torch


def velocity_tracking(heading_velocity, command):
    return torch.exp(-((heading_velocity[..., :2]-command[..., :2])**2).sum(-1)/.25)


def alignment(body_up_world, normal_world, valid):
    return -(1-(body_up_world*normal_world).sum(-1).clamp(-1, 1))*valid


def edge_penalty(sample_valid, stance):
    return -((~sample_valid).to(torch.float32)*stance[..., None]).sum((-1,-2))/11


def swing_penalty(sole_z, h0, h1, phase, valid):
    target = (1-phase)*h0+phase*h1+4*.10*phase*(1-phase)
    return -10*((sole_z-target).square()*valid).sum(-1)


class SwingTargets:
    """Latch support on real contact and target once per planned swing."""
    def __init__(self, batch, device='cpu'):
        self.h0 = torch.zeros(batch, 4, device=device)
        self.h1 = self.h0.clone()
        self.last_support = self.h0.clone()
        self.known = torch.zeros_like(self.h0, dtype=torch.bool)
        self.valid = self.known.clone()
        self.was_swing = self.known.clone()

    def reset(self, ids):
        for x in (self.h0, self.h1, self.last_support, self.known, self.valid, self.was_swing):
            x[ids] = 0

    def update(self, contact, support_z, support_valid, swing, target_z, target_valid):
        touch = contact & support_valid
        self.last_support = torch.where(touch, support_z, self.last_support)
        self.known |= touch
        start = swing & ~self.was_swing
        self.h0 = torch.where(start, self.last_support, self.h0)
        valid = target_valid & self.known & (target_z-self.h0 >= -.02-1e-6) & (target_z-self.h0 <= .22+1e-6)
        self.h1 = torch.where(start, target_z, self.h1)
        self.valid = torch.where(start, valid, self.valid)
        self.was_swing.copy_(swing)
        return self.h0, self.h1, self.valid & swing
