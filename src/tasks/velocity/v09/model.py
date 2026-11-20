"""SDS v1.1 native-map transfer from the uniquely identified v07/6400 model.

Only two FC weight matrices change. All other parameters and buffers, including
std and both normalizers, are copied exactly; old training state is never read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import CriticNormalizer
from src.tasks.velocity.v07.model import (
    V07ActorCritic, ObservationError, state_sha256, route_error_from_pose,
    synthetic_observations as _source_observations,
)
from . import spec

FC_KEYS = ('actor.map_encoder.7.weight', 'critic_map_encoder.7.weight')
S1_ADAPT_PREFIXES = ('actor.map_encoder.', 'critic_map_encoder.', 'actor.fusion.')
HEIGHTMAP_SHAPE = spec.MAP_SHAPE
ARCHITECTURE = spec.ARCHITECTURE
SDS_SHA256 = spec.SDS_SHA256
SOURCE_SHA256 = spec.SOURCE_SHA256
SOURCE_CHECKPOINT = spec.SOURCE_PATH


class MigrationError(ValueError):
    """The source or migrated state violates the exact 6400 transfer contract."""


class FrozenCriticNormalizer(CriticNormalizer):
    """Keep source statistics even if a legacy caller explicitly requests update."""
    @torch.no_grad()
    def update(self, value):
        return None


def overlap_matrix(*, device='cpu', dtype=torch.float32):
    """Unnormalised interval overlap, rows=new positions, columns=old positions."""
    old = torch.arange(6, dtype=torch.float64, device=device)
    new = torch.arange(5, dtype=torch.float64, device=device) * (6.0 / 5.0)
    return (torch.minimum(new[:, None] + 6.0 / 5.0, old[None] + 1)
            - torch.maximum(new[:, None], old[None])).clamp_min(0).to(dtype=dtype)


def migrate_map_fc(weight):
    if tuple(weight.shape) != (64, 768) or weight.dtype != torch.float32:
        raise MigrationError('map FC source must be FP32 [64,768]')
    source = weight.reshape(64, 32, 6, 4)
    overlap = overlap_matrix(device=weight.device, dtype=weight.dtype)
    return torch.einsum('jk,ockw->ocjw', overlap, source).reshape(64, 640)


def migrate_model_state(source):
    """Pure complete transfer plus per-entry, independently checkable manifest."""
    reference = V07ActorCritic().state_dict()
    if not isinstance(source, dict) or set(source) != set(reference):
        raise MigrationError('source model keys differ from the complete original v07 model')
    transferred, manifest = {}, []
    for key, expected in reference.items():
        value = source[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != expected.dtype or not torch.isfinite(value).all()):
            raise MigrationError(f'invalid source tensor: {key}')
        result = migrate_map_fc(value) if key in FC_KEYS else value.detach().clone()
        transferred[key] = result
        manifest.append(dict(key=key, operation='overlap_6_to_5' if key in FC_KEYS else 'exact_copy',
                             source_shape=list(value.shape), target_shape=list(result.shape),
                             dtype=str(value.dtype), source_sha256=state_sha256({key: value}),
                             target_sha256=state_sha256({key: result}),
                             exact_equal=torch.equal(value, result)))
    return transferred, manifest


def _load_source(path, expected_sha):
    if expected_sha != SOURCE_SHA256:
        raise MigrationError('only the SDS-designated 6400 source SHA is permitted')
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise MigrationError(f'6400 checkpoint SHA256 mismatch: {actual}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise MigrationError('source checkpoint has no complete model state')
    return checkpoint['model']


class V09ActorCritic(V07ActorCritic):
    architecture_version = ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.recipe = spec.RECIPE_VERSION
        self.actor.map_encoder[7] = nn.Linear(640, 64)
        self.critic_map_encoder[7] = nn.Linear(640, 64)
        self.critic_normalizer = FrozenCriticNormalizer()
        self.set_s1_course('S1.1')

    @staticmethod
    def _validate(obs, critic=False, consumer='forward'):
        if not isinstance(obs, dict):
            raise ObservationError(f'{consumer}: observations must be a mapping')
        shapes = dict(proprio=(45,), history=(5, 45), heightmap=HEIGHTMAP_SHAPE,
                      route_error=(3,))
        if critic:
            shapes['privileged'] = (35,)
        batch = device = None
        for key, shape in shapes.items():
            value = obs.get(key)
            if (not isinstance(value, torch.Tensor) or value.ndim != len(shape) + 1
                    or tuple(value.shape[1:]) != shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ObservationError(f'{consumer}: {key} must be finite FP32 [B,{shape}]')
            if batch is not None and (value.shape[0] != batch or value.device != device):
                raise ObservationError(f'{consumer}: observation batch/device differs')
            batch, device = value.shape[0], value.device

    def update_critic_statistics(self, obs):
        self._validate(obs, critic=True, consumer='frozen_critic_statistics')

    def train(self, mode=True):
        super().train(mode)
        self.actor.normalizer.eval()
        self.critic_normalizer.eval()
        return self

    def optimizer_parameters(self):
        """Include frozen parameters now so S1.2 retains map Adam momentum."""
        return list(self.parameters())

    def set_s1_course(self, course):
        if course not in ('S1.1', 'S1.2'):
            raise ValueError('S1 course must be S1.1 or S1.2')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name != 'std' and (
                course == 'S1.2' or name.startswith(S1_ADAPT_PREFIXES)))
        self.s1_course = course
        self.actor.normalizer.eval()
        self.critic_normalizer.eval()
        return self

    @classmethod
    def migrate_6400_checkpoint(cls, path=SOURCE_CHECKPOINT, expected_sha=SOURCE_SHA256, device='cpu'):
        source = _load_source(path, expected_sha)
        transferred, manifest = migrate_model_state(source)
        model = cls()
        model.load_state_dict(transferred, strict=True)
        model.transfer_metadata = dict(
            sds_sha256=SDS_SHA256, recipe=spec.RECIPE_VERSION, source_path=str(path),
            source_sha256=SOURCE_SHA256, architecture=ARCHITECTURE,
            source_model_sha256=state_sha256(source), model_sha256=state_sha256(transferred),
            changed_keys=list(FC_KEYS), entries=manifest,
            copied_entries=len(manifest) - 2, transformed_entries=2,
            old_training_state_loaded=False, resume_mode='new_episode')
        return model.to(device)

    @torch.no_grad()
    def normalizer_saturation(self, obs):
        """Fractions clipped by each frozen input-normalization path."""
        self._validate(obs, critic=True, consumer='normalizer_saturation')
        actor = self.actor.normalizer
        critic = self.critic_normalizer
        raw = torch.cat((obs['proprio'], obs['privileged']), -1)
        values = {
            'actor_proprio': (obs['proprio'] - actor._mean) / actor._std.clamp_min(1e-5),
            'actor_history': (obs['history'] - actor._mean) / actor._std.clamp_min(1e-5),
            'critic_proprio_privileged': ((raw - critic.mean) / critic.var.sqrt().clamp_min(1e-5))[:, critic.continuous],
            'critic_history': (obs['history'] - critic.mean[:45]) / critic.var[:45].sqrt().clamp_min(1e-5),
        }
        result = {key: (value.abs() > 5).float().mean().item() for key, value in values.items()}
        valid = obs['heightmap'][:, 1].bool()
        result['heightmap_valid_fraction'] = valid.float().mean().item()
        result['heightmap_valid_saturation'] = ((obs['heightmap'][:, 0][valid].abs() >= 1).float().mean().item()
                                               if bool(valid.any()) else None)
        result['route_lateral_saturation'] = (obs['route_error'][:, 0].abs() >= 4).float().mean().item()
        return result


class Frozen6400Teacher(V07ActorCritic):
    """Original 48x32 architecture, permanently eval and without gradients."""
    def __init__(self):
        super().__init__()
        self.critic_normalizer = FrozenCriticNormalizer()
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    def update_critic_statistics(self, obs):
        self._validate(obs, critic=True, consumer='frozen_teacher_statistics')

    @classmethod
    def from_6400_checkpoint(cls, path=SOURCE_CHECKPOINT, expected_sha=SOURCE_SHA256, device='cpu'):
        source = _load_source(path, expected_sha)
        # Validate all original keys, shapes, dtypes and finite values as well.
        migrate_model_state(source)
        model = cls()
        model.load_state_dict(source, strict=True)
        return model.to(device)

    @torch.no_grad()
    def labels(self, obs):
        return self(obs).detach(), self.evaluate(obs).detach()


def synthetic_observations(batch=64, seed=42, device='cpu'):
    obs = _source_observations(batch=batch, seed=seed, device=device)
    obs['heightmap'] = obs['heightmap'][:, :, :40, :30].clone()
    return obs
