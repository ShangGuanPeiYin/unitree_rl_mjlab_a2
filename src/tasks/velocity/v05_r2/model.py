"""V0.5-r2 ActorCritic: ETH Actor migration plus the zero-initialized route
bypass (Actor) and the 211-dimensional fresh Critic, with schema-7 validation.

SDS ``SDS_A2_IdealHeightMap_Stairs40_v0.5_r2.md`` revision 1.0 sections 2.1-2.6
and 7.5 fix the network and its initialization.  Relative to the r1 network
(``src/tasks/velocity/v05/model.py`` + ``src/tasks/velocity/rl/
ideal_heightmap_actor_critic.py``, both read-only here) this module changes
exactly two things:

* the Actor gains ``nn.Linear(3, 512)`` (``actor.route_fusion``, two new tensors)
  whose weight and bias are zero, wired inside the first activation::

      features = concat(map_encoder(heightmap) -> 64,
                        history_encoder(normalizer_eth(history)) -> 64)      # 128
      h = ELU( actor.mlp[0](normalizer_eth(proprio))        # Linear(45, 512)
               + actor.fusion(features)                     # zero after migration
               + actor.route_fusion(route_error) )          # new, zero-initialized
      mean = actor.mlp[2:](h)                               # 512-256-128-12

* the Critic input grows from 208 to **211** by appending the fixed-scale
  ``route_error`` as the last three features::

      input211 = concat(raw80, critic_map_encoder(heightmap) -> 64,
                        critic_history_encoder(history_n) -> 64, route_error -> 3)
      value    = critic(input211)          # Linear(211,512)-ELU-...-Linear(128,1)

  ``critic_normalizer`` still owns only the 80 proprio+privileged dimensions and
  the map/history encoders are unchanged.

Construction order is the unmodified shared Actor structure followed by the one
new bypass layer; the bypass is therefore really constructed (it consumes the run
RNG) and only then overwritten with zeros, as SDS 2.3 requires.  The base class
also builds its 208-input Critic, which r2 immediately replaces by the 211-input
Critic: the shared ``rl/`` module is never modified, and the whole construction is
a deterministic function of the run seed, so "the same seed and the same
construction order" defines the fresh Critic reference used by the migration
audit.

``V05R2ActorCritic.from_eth`` performs the seven ordered SDS 2.2 steps:

1. validate SHA256, actor tensor keys/shapes/dtypes/finiteness and normalizer
   ``std > 0`` on the CPU (no CUDA is touched);
2. seed the run RNG and construct the complete r2 network (including
   ``actor.route_fusion`` and the 211-input Critic);
3. copy the complete ETH Actor MLP and ``FrozenETHNormalizer`` parameters and
   buffers (the normalizer owns no trainable parameter, so it stays frozen);
4. leave the map/history encoders, ``actor.fusion``, the Critic and the new
   ``route_fusion`` freshly seeded; zero ``actor.fusion`` and
   ``actor.route_fusion`` (four tensors, two of them new in r2);
5. write all twelve action standard deviations to ``0.15`` (the ETH std is never
   inherited);
6. leave Critic statistics, Adam state and counters untouched/empty;
7. hand back the model plus the transfer metadata the audit persists.

The ETH checkpoint also contains ``critic_state_dict`` (first layer ``(512,72)``).
That block is a **forbidden non-source key** (SDS 2.4/7.5): it is neither required
to be absent nor ever copied.  The r2 Critic is fresh.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Integral
from pathlib import Path, PurePosixPath

import torch
from torch import nn
from torch.distributions import Normal

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import (CriticNormalizer,
                                                               FusedActor,
                                                               FrozenETHNormalizer,
                                                               IdealHeightmapActorCritic,
                                                               history_encoder,
                                                               map_encoder, mlp)

from .spec import (ACTOR_HIDDEN, ACTION_DIM, ARCHITECTURE, CODE_VERSION,
                   CRITIC_FIRST_KEY, CRITIC_FIRST_SHAPE, CRITIC_HIDDEN, CRITIC_INPUT_DIMS,
                   CRITIC_LAYER_KEYS, CRITIC_NORMALIZER_DIMS, DESIGN_DECISION_SHA256,
                   ETH_ACTOR_SOURCE_KEYS, ETH_FORBIDDEN_SOURCE_BLOCKS, ETH_NORMALIZER_DIMS,
                   FLAT_PROTOCOL_ID, IMPLEMENTATION_DECISION_SHA256, INITIAL_STD,
                   INIT_SHA256, OBSERVATION, PPO_INITIAL_LR, PROTOCOL_ID, RECIPE_VERSION,
                   ROUTE_ERROR_DIM, ROUTE_ERROR_KEY, ROUTE_FUSION_BIAS_KEY, ROUTE_FUSION_KEYS,
                   ROUTE_FUSION_SHAPES, ROUTE_FUSION_WEIGHT_KEY, ROUTE_LATERAL_CLIP,
                   ROUTE_LATERAL_SCALE, ROUTE_STATE, ROUTE_YAW_AXIS, SCHEMA_VERSION,
                   SDS_REVISION, SDS_SHA256, SMOKE_CONTAINER, STD_MAX, STD_MIN, TASK,
                   behavior_hashes, contract as spec_contract, identity_block,
                   observation_contract, recipe_config, reward_config,
                   reward_config_sha256, route_state as spec_route_state, sha256)

# Static ETH Actor contract: no module construction (and therefore no RNG draw)
# happens before the run seed is applied in step 2.
_ETH_MLP_SHAPES = {
    'mlp.0.weight': (512, 45), 'mlp.0.bias': (512,),
    'mlp.2.weight': (256, 512), 'mlp.2.bias': (256,),
    'mlp.4.weight': (128, 256), 'mlp.4.bias': (128,),
    'mlp.6.weight': (12, 128), 'mlp.6.bias': (12,),
}
_ETH_NORMALIZER_SHAPES = {
    'obs_normalizer._mean': (1, 45), 'obs_normalizer._var': (1, 45),
    'obs_normalizer._std': (1, 45), 'obs_normalizer.count': (),
}
ETH_ACTOR_KEYS = tuple(sorted({*_ETH_MLP_SHAPES, *_ETH_NORMALIZER_SHAPES, 'std'}))
NORMALIZER_SOURCE_KEYS = tuple(sorted(_ETH_NORMALIZER_SHAPES))
HEX_DIGITS = set('0123456789abcdef')
INITIALIZATION_SCOPE = 'initialization'
INITIALIZATION_FIELDS = ('contract', 'model', 'learning_state', 'optimizer', 'scope',
                         'transfer_metadata', 'audit_schema', 'source_checkpoint_sha256',
                         'source_model_sha256', 'migrated_model_sha256', 'seed',
                         'source_path', 'resume', 'physical_trajectory_continuous', 'notes')
#: SDS 7.5: the exact top-level schema-7 envelope (14 keys, no more, no less).
SCHEMA7_TOP_LEVEL_KEYS = ('schema', 'identity', 'contract_sha256', 'model', 'optimizer',
                          'normalizer', 'learning', 'rng', 'curriculum', 'ledger',
                          'evaluation', 'source', 'recipe', 'resume_mode')
IDENTITY_FIELDS = ('task', 'recipe', 'code_version', 'protocol', 'sds_revision')
RESUME_MODE = 'new_episode'
SMOKE_CONTAINERS = ('v05-smoke-state-v1', SMOKE_CONTAINER)
#: SDS 2.6 A: the audited synthetic route states (local_y in metres, e_world in
#: radians).  ``0.8001`` is the saturation probe past the 0.80 m clip point.
ROUTE_AUDIT_LOCAL_Y = (-.8001, -.8, -.4, -.2, 0., .2, .4, .8, .8001)
ROUTE_AUDIT_E_WORLD = (0., math.radians(5.), -math.radians(5.),
                       math.radians(10.), -math.radians(10.),
                       math.radians(90.), -math.radians(90.),
                       math.pi, -math.pi)
ROUTE_AUDIT_BATCH = 128


class EthMigrationError(ValueError):
    """Raised when the ETH source or the migrated state violates SDS 2.2."""


class LegacySchemaError(ValueError):
    """A schema <= 6 (or unlabelled / smoke) checkpoint was offered to the r2 reader."""


def _is_hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX_DIGITS


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 1:
        raise ValueError('seed must be a positive integer')
    return int(seed)


def _load_validated_eth(path, expected_sha):
    """CPU-only step 1: SHA, exact actor keys, shapes, dtypes, finiteness, std>0."""
    if not _is_hex_digest(expected_sha) or expected_sha != INIT_SHA256:
        raise EthMigrationError('Only the approved ETH Actor SHA256 may be migrated')
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != INIT_SHA256:
        raise EthMigrationError('ETH checkpoint SHA256 mismatch')
    source = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(source, dict) or not isinstance(source.get('actor_state_dict'), dict):
        raise EthMigrationError('ETH checkpoint has no actor_state_dict')
    state = source['actor_state_dict']
    if set(state) != set(ETH_ACTOR_KEYS):
        missing = sorted(set(ETH_ACTOR_KEYS) - set(state))
        extra = sorted(set(state) - set(ETH_ACTOR_KEYS))
        raise EthMigrationError(f'ETH actor key contract mismatch: missing={missing} extra={extra}')
    shapes = {**_ETH_MLP_SHAPES, **_ETH_NORMALIZER_SHAPES, 'std': (12,)}
    for key, expected in shapes.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise EthMigrationError(f'ETH tensor shape mismatch: {key}')
        expected_dtype = torch.int64 if key == 'obs_normalizer.count' else torch.float32
        if value.dtype != expected_dtype:
            raise EthMigrationError(f'ETH tensor dtype mismatch: {key}')
        if not torch.isfinite(value).all():
            raise EthMigrationError(f'Non-finite ETH tensor: {key}')
    if (state['obs_normalizer._std'] <= 0).any() or (state['obs_normalizer._var'] < 0).any():
        raise EthMigrationError('ETH normalizer std/var is invalid')
    if int(state['obs_normalizer.count'].item()) < 0 or (state['std'] <= 0).any():
        raise EthMigrationError('ETH normalizer count or std is invalid')
    for key, value in state.items():
        if value.device.type != 'cpu':
            raise EthMigrationError('ETH migration must validate on the CPU')
    # ``critic_state_dict`` (and any Adam block) is a forbidden non-source key:
    # recorded so the audit can prove it was present and not migrated.
    forbidden = {key: True for key in ETH_FORBIDDEN_SOURCE_BLOCKS if key in source}
    return source, state, forbidden


def _source_key_to_model_key(key):
    if key == 'std':
        return 'std'
    if key.startswith('mlp.'):
        return 'actor.' + key
    if key.startswith('obs_normalizer.'):
        return 'actor.normalizer.' + key[len('obs_normalizer.'):]
    raise KeyError(key)


class RouteFusedActor(FusedActor):
    """The unchanged r1 Actor plus the r2 ``route_fusion`` bypass (SDS 2.3).

    ``super().__init__()`` builds exactly the frozen r1 structure in the frozen
    r1 order (normalizer, map encoder, history encoder, 45-512-256-128-12 MLP and
    the zero-initialized 128->512 fusion).  The one new ``Linear(3, 512)`` is then
    constructed -- it therefore really consumes the run RNG -- and explicitly
    overwritten with zeros.  ``forward`` adds the three-way sum inside the first
    ELU; everything after it is the shared ``mlp[2:]`` tail.
    """

    def __init__(self):
        super().__init__()
        self.route_fusion = nn.Linear(ROUTE_ERROR_DIM, ACTOR_HIDDEN[0])
        nn.init.zeros_(self.route_fusion.weight)
        nn.init.zeros_(self.route_fusion.bias)

    def features(self, obs):
        return torch.cat((self.map_encoder(obs['heightmap']),
                          self.history_encoder(self.normalizer(obs['history']))), -1)

    def forward(self, obs):
        features = self.features(obs)
        pre_activation = (self.mlp[0](self.normalizer(obs['proprio']))
                          + self.fusion(features)
                          + self.route_fusion(obs[ROUTE_ERROR_KEY]))
        return self.mlp[2:](self.mlp[1](pre_activation))


class V05R2ActorCritic(IdealHeightmapActorCritic):
    """Ideal-heightmap network with the r2 identity, route bypass and 211 Critic."""

    architecture_version = ARCHITECTURE
    std_min = STD_MIN
    std_max = STD_MAX

    def __init__(self):
        self.recipe = RECIPE_VERSION
        self.recipe_settings = recipe_config()
        super().__init__()
        # Replace the shared 208-input Critic and the pre-route Actor with the r2
        # shapes.  The shared ``rl/`` module stays byte-identical (SDS 2.3/2.4).
        self.actor = RouteFusedActor()
        self.critic = mlp([CRITIC_INPUT_DIMS, *CRITIC_HIDDEN, 1])

    # ------------------------------------------------------------- validation
    @staticmethod
    def _validate(obs, critic=False, consumer='forward'):
        """Strict five-key observation validation (SDS 2.1 / 4.7).

        Every required key must be present, ``float32``, on one device, in one
        batch and fully finite *before* any layer runs.  A missing
        ``route_error`` is a hard error naming the key and the consumer; it is
        never zero filled.
        """
        if not hasattr(obs, '__contains__'):
            raise ValueError(f'{consumer}: observation must be a mapping')
        keys = ('proprio', 'history', 'heightmap', 'privileged', ROUTE_ERROR_KEY) \
            if critic else ('proprio', 'history', 'heightmap', ROUTE_ERROR_KEY)
        shapes = {'proprio': (45,), 'history': (5, 45), 'heightmap': (3, 48, 32),
                  'privileged': (35,), ROUTE_ERROR_KEY: (ROUTE_ERROR_DIM,)}
        for key in keys:
            if key not in obs:
                raise ValueError(f'{consumer}: observation is missing {key!r}; the r2 '
                                 f'five-key observation requires it and it is never '
                                 f'zero filled')
        batch = None
        device = None
        for key in keys:
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'{consumer}: observation {key!r} must be a tensor')
            shape = shapes[key]
            if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape:
                raise ValueError(f'{consumer}: observation {key!r} must be [B,'
                                 f'{",".join(str(item) for item in shape)}], '
                                 f'got {list(value.shape)}')
            if value.dtype != torch.float32:
                raise ValueError(f'{consumer}: observation {key!r} must be float32, '
                                 f'got {value.dtype}')
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError(f'{consumer}: observation {key!r} device {value.device} '
                                 f'differs from {device}')
            if batch is None:
                batch = value.shape[0]
            elif value.shape[0] != batch:
                raise ValueError(f'{consumer}: observation {key!r} batch {value.shape[0]} '
                                 f'differs from {batch}')
            if not torch.isfinite(value).all():
                raise ValueError(f'{consumer}: observation {key!r} contains NaN/Inf')

    # ---------------------------------------------------------------- forward
    def forward(self, obs):
        self._validate(obs, consumer='forward')
        return self.actor(obs)

    def evaluate(self, obs):
        self._validate(obs, critic=True, consumer='evaluate')
        raw = self.critic_normalizer(torch.cat((obs['proprio'], obs['privileged']), -1))
        # History uses critic-owned current-proprio population moments, never ETH stats.
        history = ((obs['history'] - self.critic_normalizer.mean[:45]) /
                   self.critic_normalizer.var[:45].sqrt().clamp_min(1e-5)).clamp(-5, 5)
        return self.critic(torch.cat((raw, self.critic_map_encoder(obs['heightmap']),
                                      self.critic_history_encoder(history),
                                      obs[ROUTE_ERROR_KEY]), -1))

    def update_critic_statistics(self, obs):
        self._validate(obs, critic=True, consumer='update_critic_statistics')
        self.critic_normalizer.update(torch.cat((obs['proprio'], obs['privileged']), -1).detach())

    # ------------------------------------------------------------- deployment
    @torch.no_grad()
    def constrain_std(self):
        """Project the trainable std after each optimizer step (SDS 2.5)."""
        self.std.clamp_(STD_MIN, STD_MAX)

    def action_distribution(self, obs):
        """Unsquashed Normal with the same [0.10, 0.30] std constraint."""
        self._validate(obs, consumer='action_distribution')
        return Normal(self(obs), self.std.clamp(STD_MIN, STD_MAX))

    def act(self, obs, deterministic=False):
        consumer = 'act' if deterministic else 'act(action_distribution)'
        self._validate(obs, consumer=consumer)
        return self(obs) if deterministic else self.action_distribution(obs).sample()

    # ------------------------------------------------------------- migration
    @classmethod
    def from_eth(cls, path, expected_sha, seed, device='cpu'):
        """The seven ordered SDS 2.2 initialization steps; never a resume."""
        seed = _check_seed(seed)
        source, eth_state, forbidden = _load_validated_eth(path, expected_sha)
        torch.manual_seed(seed)
        model = cls()
        mlp_state = {key[len('mlp.'):]: value for key, value in eth_state.items()
                     if key.startswith('mlp.')}
        normalizer_state = {key[len('obs_normalizer.'):]: value
                            for key, value in eth_state.items()
                            if key.startswith('obs_normalizer.')}
        model.actor.mlp.load_state_dict(mlp_state, strict=True)
        model.actor.normalizer.load_state_dict(normalizer_state, strict=True)
        with torch.no_grad():
            model.actor.fusion.weight.zero_()
            model.actor.fusion.bias.zero_()
            model.actor.route_fusion.weight.zero_()
            model.actor.route_fusion.bias.zero_()
            model.std.fill_(INITIAL_STD)
        if (model.actor.fusion.weight.any() or model.actor.fusion.bias.any()):
            raise EthMigrationError('Actor fusion layer is not zero after migration')
        if (model.actor.route_fusion.weight.any()
                or model.actor.route_fusion.bias.any()):
            raise EthMigrationError('Actor route_fusion layer is not zero after migration')
        if not torch.equal(model.std, torch.full_like(model.std, INITIAL_STD)):
            raise EthMigrationError('Action std is not fully 0.15 after migration')
        if tuple(model.critic[0].weight.shape) != CRITIC_FIRST_SHAPE:
            raise EthMigrationError('Critic first layer is not the 211-input r2 shape')
        # The Actor normalizer is a buffer-only module: assert it owns no
        # trainable parameter so "frozen" is a structural property, not a flag.
        if list(model.actor.normalizer.parameters()):
            raise EthMigrationError('ETH Actor normalizer must stay parameter-free/frozen')
        model.actor.normalizer.eval()
        model.transfer_metadata = dict(
            kind='eth_actor_mlp_and_frozen_normalizer_only',
            source_path=str(Path(path).resolve()), sha256=INIT_SHA256,
            source_iteration=int(source.get('iter', 0)) if isinstance(source, dict) else 0,
            source_actor_keys=list(ETH_ACTOR_KEYS),
            source_normalizer_keys=list(NORMALIZER_SOURCE_KEYS),
            forbidden_source_blocks=sorted(forbidden),
            forbidden_source_blocks_migrated=[],
            architecture=ARCHITECTURE, code_version=CODE_VERSION,
            recipe=RECIPE_VERSION, reward_version=RECIPE_VERSION,
            reward_config_sha256=reward_config_sha256(),
            sds_sha256=SDS_SHA256, design_decision_sha256=DESIGN_DECISION_SHA256,
            implementation_decision_sha256=IMPLEMENTATION_DECISION_SHA256,
            seed=seed, actor_statistics='copied_complete_and_frozen',
            normalization='frozen_ETH_normalizer_buffers',
            fusion='zeros_after_migration',
            route_fusion='zeros_after_migration',
            route_fusion_keys=list(ROUTE_FUSION_KEYS),
            route_fusion_tensor_count=len(ROUTE_FUSION_KEYS),
            critic='fresh_seeded_211', critic_input_dims=CRITIC_INPUT_DIMS,
            critic_statistics='fresh', optimizer='fresh_adam',
            std_before=eth_state['std'].detach().tolist(),
            std_after=model.std.detach().tolist(),
            std_source='overwritten_to_0.15_not_inherited',
            std_range=[STD_MIN, STD_MAX], resume=False, scope=INITIALIZATION_SCOPE)
        return model.to(device)


#: Alias kept so consumers written against the r1 class name (and the sibling
#: r2 modules that import ``V05ActorCritic``) bind the r2 network.
V05ActorCritic = V05R2ActorCritic


# ------------------------------------------------------- route observation
def route_error_from_pose(local_y, yaw_world, yaw_axis=ROUTE_YAW_AXIS):
    """Reference implementation of the SDS 4.2 route observation.

    ``route_error = [clip(local_y/0.20, -4, 4), sin(e_world), cos(e_world)]`` with
    ``e_world = wrap_to_pi(yaw_world - yaw_axis)`` in radians.  The environment
    owns the authoritative computation from its own slot tile; this helper is the
    single documented formula used by the migration audit and the route tests, so
    no consumer can invent a different scale or unit.

    ``yaw_axis`` is **frozen at 0.0**: every current atlas tile is a pure
    translation, so any other axis would silently change the route semantics.
    Like ``spec`` and the r2 environment, this helper fails closed instead of
    computing an unsupported observation (SDS 4.2 / 13.5: a rotated or axially
    offset atlas requires a route-state contract upgrade).
    """
    local_y = torch.as_tensor(local_y, dtype=torch.float32)
    yaw_world = torch.as_tensor(yaw_world, dtype=torch.float32)
    try:
        axis = float(yaw_axis)
    except (TypeError, ValueError):
        raise ValueError('route_error_from_pose requires yaw_axis == '
                         f'{ROUTE_YAW_AXIS} as a real number') from None
    if not math.isfinite(axis) or axis != ROUTE_YAW_AXIS:
        raise ValueError(
            f'route_error_from_pose supports yaw_axis == {ROUTE_YAW_AXIS} only: the '
            'current atlas tiles are translations only, so a non-zero corridor axis '
            'would require a route-state contract upgrade (SDS 4.2/13.5)')
    if not torch.isfinite(local_y).all() or not torch.isfinite(yaw_world).all():
        raise ValueError('route_error_from_pose requires finite local_y and yaw_world')
    if local_y.shape != yaw_world.shape:
        raise ValueError('local_y and yaw_world must have the same shape')
    delta = yaw_world - axis
    e_world = torch.atan2(torch.sin(delta), torch.cos(delta))
    lateral = (local_y / ROUTE_LATERAL_SCALE).clamp(-ROUTE_LATERAL_CLIP, ROUTE_LATERAL_CLIP)
    return torch.stack((lateral, torch.sin(e_world), torch.cos(e_world)), -1)


def synthetic_observations(batch=ROUTE_AUDIT_BATCH, seed=42, device='cpu'):
    """Finite synthetic observations used only by migration/route audits.

    ``route_error`` covers the SDS 2.6 A grid: the all-zero state, both lateral
    clip ends (``local_y = +-0.8`` and ``+-0.8001`` metres) and
    ``e_world in {0, +-5, +-10, +-90, +-180}`` degrees, distributed over mixed
    batch positions instead of contiguous blocks.
    """
    if isinstance(batch, bool) or not isinstance(batch, Integral) or int(batch) < 1:
        raise ValueError('batch must be a positive integer')
    batch = int(batch)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    observations = {
        'proprio': torch.randn(batch, 45, generator=generator, dtype=torch.float32),
        'history': torch.randn(batch, 5, 45, generator=generator, dtype=torch.float32),
        'heightmap': torch.randn(batch, 3, 48, 32, generator=generator, dtype=torch.float32),
        'privileged': torch.randn(batch, 35, generator=generator, dtype=torch.float32),
    }
    combinations = [(local_y, e_world) for e_world in ROUTE_AUDIT_E_WORLD
                    for local_y in ROUTE_AUDIT_LOCAL_Y]
    rows = []
    while len(rows) < batch:
        for index in torch.randperm(len(combinations), generator=generator).tolist():
            rows.append(combinations[index])
            if len(rows) == batch:
                break
    local_y = torch.tensor([row[0] for row in rows], dtype=torch.float32)
    e_world = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    route = torch.stack(((local_y / ROUTE_LATERAL_SCALE).clamp(-ROUTE_LATERAL_CLIP,
                                                              ROUTE_LATERAL_CLIP),
                         torch.sin(e_world), torch.cos(e_world)), -1)
    observations[ROUTE_ERROR_KEY] = route
    return {key: value.to(device) for key, value in observations.items()}


def route_coverage(route_error):
    """Coverage evidence for the audited route grid (SDS 2.6 A / 9.3-21)."""
    route = torch.as_tensor(route_error, dtype=torch.float32).detach().cpu()
    if route.ndim != 2 or route.shape[1] != ROUTE_ERROR_DIM:
        raise ValueError('route_error must be [N,3]')
    lateral = route[:, 0]
    heading = torch.atan2(route[:, 1], route[:, 2])
    def _has(values, target, tol=1e-5):
        return bool(((values - target).abs() <= tol).any())
    return dict(samples=int(route.shape[0]),
                zero_present=bool((lateral.abs() <= 1e-6).any()),
                lateral_plus_clip=_has(lateral, ROUTE_LATERAL_CLIP),
                lateral_minus_clip=_has(lateral, -ROUTE_LATERAL_CLIP),
                lateral_saturated_points=tuple(local_y for local_y in ROUTE_AUDIT_LOCAL_Y
                                               if abs(local_y) >= .8),
                heading_radians=[angle for angle in ROUTE_AUDIT_E_WORLD
                                 if _has(heading, angle, tol=1e-5)],
                dtype=str(route.dtype), finite=bool(torch.isfinite(route).all()))


# ---------------------------------------------------------- audit primitives
def _tensor_sha256(value):
    digest = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_audit(source_state, migrated_state):
    """Per-tensor audit rows: key, shape, dtype, SHA and equality with the source."""
    reverse = {}
    for key in source_state:
        reverse[_source_key_to_model_key(key)] = key
    rows = []
    for key in sorted(migrated_state):
        value = migrated_state[key]
        source_key = reverse.get(key)
        if key == 'std':
            group = 'std_overwritten'
        elif source_key is not None:
            group = 'actor_copied'
        elif key in ROUTE_FUSION_KEYS:
            group = 'route_fusion_new'
        elif key.startswith('actor.'):
            group = 'actor_fresh'
        else:
            group = 'fresh_seeded'
        rows.append(dict(
            key=key, source_key=source_key, group=group,
            shape=list(value.shape), dtype=str(value.dtype), numel=int(value.numel()),
            finite=bool(torch.isfinite(value).all()), tensor_sha256=_tensor_sha256(value),
            equal_to_source=(None if source_key is None
                             else bool(torch.equal(value, source_state[source_key])))))
    return rows


def load_eth_actor_reference(path, expected_sha=INIT_SHA256, *, device='cpu', seed=0):
    """Base network carrying only the ETH Actor, for mean-equivalence evidence.

    Its map/history/critic branches come from a local seeded construction and its
    fusion layer is zero exactly like the base initializer, so its deterministic
    mean must equal the migrated r2 mean.  Ambient RNG is preserved.
    """
    _, eth_state, _ = _load_validated_eth(path, expected_sha)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_check_seed(seed) if seed else 1)
        reference = IdealHeightmapActorCritic()
    mlp_state = {key[len('mlp.'):]: value for key, value in eth_state.items()
                 if key.startswith('mlp.')}
    normalizer_state = {key[len('obs_normalizer.'):]: value
                        for key, value in eth_state.items()
                        if key.startswith('obs_normalizer.')}
    reference.actor.mlp.load_state_dict(mlp_state, strict=True)
    reference.actor.normalizer.load_state_dict(normalizer_state, strict=True)
    with torch.no_grad():
        reference.std.copy_(eth_state['std'])
    return reference.to(device)


def _actor_means(model, observations):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(observations)
    finally:
        model.train(was_training)


def mean_equivalence(model, reference, observations=None, *, atol=1e-6, rtol=1e-5):
    """Compare deterministic Actor means for the same proprio on 128 observations."""
    observations = synthetic_observations() if observations is None else observations
    was_training = (model.training, reference.training)
    model.eval()
    reference.eval()
    try:
        with torch.no_grad():
            reference_mean = reference(observations)
            migrated_mean = model(observations)
    finally:
        model.train(was_training[0])
        reference.train(was_training[1])
    if reference_mean.shape != migrated_mean.shape:
        raise ValueError('Actor mean shape mismatch')
    maximum = float((reference_mean - migrated_mean).abs().max())
    finite = bool(torch.isfinite(reference_mean).all() and torch.isfinite(migrated_mean).all())
    close = bool(torch.allclose(reference_mean, migrated_mean, atol=atol, rtol=rtol))
    passed = bool(finite and maximum <= atol and close)
    return dict(observations=int(reference_mean.shape[0]), atol=atol, rtol=rtol,
                max_abs_error=maximum, finite=finite, allclose=close, passed=passed,
                reference='eth_actor_mlp_and_frozen_normalizer',
                compared='deterministic_actor_mean',
                claim='initialization_mean_equivalence_only_not_all_tensors_equal')


def _zero_fusion_comparison(model, observations, altered):
    base = _actor_means(model, observations)
    changed = _actor_means(model, altered)
    maximum = float((base - changed).abs().max()) if base.numel() else 0.
    return dict(max_abs_difference=maximum, bitwise_equal=bool(torch.equal(base, changed)),
                passed=bool(torch.isfinite(base).all() and torch.isfinite(changed).all()
                            and maximum == 0.))


def map_history_invariance(model, observations=None, *, seed=43):
    """Zero-fusion output must not depend on map/history (nor on privileged data)."""
    observations = synthetic_observations(seed=42) if observations is None else observations
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    altered = dict(
        observations,
        heightmap=torch.randn(observations['heightmap'].shape, generator=generator,
                              dtype=observations['heightmap'].dtype).to(observations['heightmap'].device),
        history=torch.randn(observations['history'].shape, generator=generator,
                            dtype=observations['history'].dtype).to(observations['history'].device),
        privileged=torch.randn(observations['privileged'].shape, generator=generator,
                               dtype=observations['privileged'].dtype).to(observations['privileged'].device))
    rolled = dict(observations,
                  heightmap=observations['heightmap'].roll(1, 0),
                  history=observations['history'].roll(2, 0))
    changed = _zero_fusion_comparison(model, observations, altered)
    mask_changed = _zero_fusion_comparison(model, observations, rolled)
    return dict(observations=int(observations['proprio'].shape[0]), seed=int(seed),
                independent_of_map_history_privileged=changed['passed'],
                independent_of_rolled_map_history=mask_changed['passed'],
                max_abs_difference=max(changed['max_abs_difference'],
                                       mask_changed['max_abs_difference']),
                bitwise_equal=bool(changed['bitwise_equal'] and mask_changed['bitwise_equal']),
                fusion_weight_abs_sum=float(model.actor.fusion.weight.detach().abs().sum()),
                fusion_bias_abs_sum=float(model.actor.fusion.bias.detach().abs().sum()),
                passed=bool(changed['passed'] and mask_changed['passed']))


def route_zero_invariance(model, observations=None, *, seed=44):
    """Changing ``route_error`` alone must not change the zero-bypass initial mean."""
    observations = synthetic_observations(seed=42) if observations is None else observations
    route = observations[ROUTE_ERROR_KEY]
    variants = {
        'negated': dict(observations, **{ROUTE_ERROR_KEY: -route}),
        'zeroed': dict(observations, **{ROUTE_ERROR_KEY: torch.zeros_like(route)}),
        'clipped_ends': dict(observations, **{ROUTE_ERROR_KEY: torch.stack((
            torch.full_like(route[:, 0], ROUTE_LATERAL_CLIP),
            torch.zeros_like(route[:, 1]), torch.zeros_like(route[:, 2])), -1)}),
        'randomized': dict(observations, **{ROUTE_ERROR_KEY: torch.rand(
            route.shape, generator=torch.Generator(device='cpu').manual_seed(int(seed)),
            dtype=route.dtype).to(route.device)}),
    }
    results = {name: _zero_fusion_comparison(model, observations, altered)
               for name, altered in variants.items()}
    return dict(observations=int(route.shape[0]), seed=int(seed), variants=results,
                max_abs_difference=max(item['max_abs_difference']
                                       for item in results.values()),
                route_fusion_weight_abs_sum=float(
                    model.actor.route_fusion.weight.detach().abs().sum()),
                route_fusion_bias_abs_sum=float(
                    model.actor.route_fusion.bias.detach().abs().sum()),
                passed=bool(all(item['passed'] for item in results.values())))


def route_fusion_zero_audit(model):
    """SDS 2.6 A item 3: the two new tensors are elementwise zero."""
    expected = {key: tuple(model.state_dict()[key].shape) for key in ROUTE_FUSION_KEYS}
    zero = {key: bool((model.state_dict()[key] == 0).all()) for key in ROUTE_FUSION_KEYS}
    return dict(keys=list(ROUTE_FUSION_KEYS), shapes={key: list(value)
                                                      for key, value in expected.items()},
                elementwise_zero=zero, tensor_count=len(ROUTE_FUSION_KEYS),
                expected_shapes={key: list(shape)
                                 for key, shape in ROUTE_FUSION_SHAPES.items()},
                shape_matches=bool(all(expected[key] == ROUTE_FUSION_SHAPES[key]
                                       for key in ROUTE_FUSION_KEYS)),
                passed=bool(all(zero.values()) and expected == ROUTE_FUSION_SHAPES))


def route_bypass_trainability(model, observations=None):
    """SDS 2.6 B: the bypass is in Adam's parameter set and receives a gradient.

    A single backward pass on a fixed finite loss with a non-zero route
    observation must give both new tensors a finite ``.grad``.  No optimizer step
    is taken (the frozen ETH part is never modified), and the accumulated
    gradients are cleared again before returning.
    """
    observations = synthetic_observations(seed=45) if observations is None else observations
    parameters = [model.actor.route_fusion.weight, model.actor.route_fusion.bias]
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_INITIAL_LR)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group['params']}
    optimizer_ptrs = {parameter.data_ptr() for group in optimizer.param_groups
                      for parameter in group['params']}
    in_optimizer = {name: bool(id(parameter) in optimizer_ids
                               and parameter.data_ptr() in optimizer_ptrs)
                    for name, parameter in zip(ROUTE_FUSION_KEYS, parameters)}
    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        mean = model(observations)
        loss = (mean.square().mean() + mean.sum() * 0.)
        loss.backward()
        gradients = {}
        for name, parameter in zip(ROUTE_FUSION_KEYS, parameters):
            value = parameter.grad
            gradients[name] = dict(present=value is not None,
                                   finite=bool(value is not None
                                               and torch.isfinite(value).all()),
                                   abs_max=(float(value.detach().abs().max())
                                            if value is not None else None))
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    loss_value = float(loss.detach())
    finite_loss = math.isfinite(loss_value)
    finite_grads = all(item['present'] and item['finite']
                       for item in gradients.values())
    nonzero = all(item['present'] and item['abs_max'] not in (None, 0.)
                  for item in gradients.values())
    return dict(parameters=list(ROUTE_FUSION_KEYS), in_optimizer=in_optimizer,
                gradient_rows=gradients, loss=loss_value, loss_finite=finite_loss,
                adam_parameter_count=sum(len(group['params'])
                                         for group in optimizer.param_groups),
                optimizer_step_called=False,
                passed=bool(all(in_optimizer.values()) and finite_loss and finite_grads),
                gradient_nonzero=bool(nonzero))


def critic_fresh_reference(seed, *, device='cpu'):
    """The fresh Critic reference: same seed, same construction order (SDS 2.4)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_check_seed(seed))
        reference = V05R2ActorCritic()
    return reference.to(device)


def critic_211_audit(model, observations=None, *, seed=42):
    """SDS 2.6 C: 211-input Critic, route tail, sensitivity and fresh equality."""
    observations = synthetic_observations(seed=int(seed)) if observations is None else observations
    route = observations[ROUTE_ERROR_KEY]
    captured = {}

    def pre_hook(_module, inputs):
        captured['input'] = inputs[0]

    handle = model.critic.register_forward_pre_hook(pre_hook)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = model.evaluate(observations)
    finally:
        handle.remove()
        model.train(was_training)
    first_input = captured.get('input')
    tail_equal = bool(first_input is not None
                      and torch.equal(first_input[:, -ROUTE_ERROR_DIM:], route))
    tail_shape = list(first_input.shape) if first_input is not None else None
    generator = torch.Generator(device='cpu').manual_seed(int(seed) + 1)
    altered_route = torch.rand(route.shape, generator=generator, dtype=route.dtype).to(route.device)
    altered_map = torch.randn(observations['heightmap'].shape, generator=generator,
                              dtype=observations['heightmap'].dtype).to(observations['heightmap'].device)
    altered_history = torch.randn(observations['history'].shape, generator=generator,
                                  dtype=observations['history'].dtype).to(observations['history'].device)
    with torch.no_grad():
        changed_route = model.evaluate(dict(observations, **{ROUTE_ERROR_KEY: altered_route}))
        changed_map = model.evaluate(dict(observations, heightmap=altered_map,
                                          history=altered_history))
    route_difference = float((base - changed_route).abs().max())
    map_difference = float((base - changed_map).abs().max())
    reference = critic_fresh_reference(seed)
    state, expected = model.state_dict(), reference.state_dict()
    fresh_keys = [key for key in expected if not key.startswith('actor.') and key != 'std']
    fresh_equal = {key: bool(torch.equal(expected[key], state[key])) for key in fresh_keys}
    del reference
    result = dict(
        critic_in_features=int(model.critic[0].in_features),
        critic_out_features=int(model.critic[0].out_features),
        critic_first_layer_shape=list(model.critic[0].weight.shape),
        critic_normalizer_dims=int(model.critic_normalizer.mean.numel()),
        critic_input_row_dims=int(first_input.shape[-1]) if first_input is not None else None,
        route_tail_equal=tail_equal, route_tail_shape=tail_shape,
        route_tail_index=[int(first_input.shape[-1]) - ROUTE_ERROR_DIM,
                          int(first_input.shape[-1])]
        if first_input is not None else None,
        route_changes_value=bool(route_difference > 0.),
        route_value_max_abs_difference=route_difference,
        map_history_changes_value=bool(map_difference > 0.),
        map_history_value_max_abs_difference=map_difference,
        fresh_seed=int(seed), fresh_keys=fresh_keys, fresh_equal=fresh_equal,
        fresh_all_equal=bool(all(fresh_equal.values())),
        critic_map_encoder_keys=sorted(key for key in state
                                       if key.startswith('critic_map_encoder.')),
        critic_history_encoder_keys=sorted(key for key in state
                                           if key.startswith('critic_history_encoder.')),
    )
    result['passed'] = bool(
        result['critic_in_features'] == CRITIC_INPUT_DIMS
        and result['critic_out_features'] == CRITIC_HIDDEN[0]
        and result['critic_normalizer_dims'] == CRITIC_NORMALIZER_DIMS
        and tail_equal and result['route_changes_value']
        and result['map_history_changes_value'] and result['fresh_all_equal'])
    return result


def audit_initialization(model, *, source_state, reference, observations=None, seed=42):
    """Run the five SDS 2.6 audits and return one passed/failed record.

    Items: (A1) 128-observation ETH Actor mean equivalence within 1e-6,
    (A2) route/map/history invariance of the zero-fusion initial output,
    (A3) the two new tensors elementwise zero, (B) Adam membership and gradient
    of the bypass, (C) the 211-input Critic and its fresh-reference equality.
    ``reference`` is the ETH-Actor reference network built by
    :func:`load_eth_actor_reference`; it is never constructed implicitly.
    """
    if reference is None:
        raise ValueError('audit_initialization requires the ETH actor reference network')
    observations = synthetic_observations(ROUTE_AUDIT_BATCH, seed=seed) \
        if observations is None else observations
    equivalence = mean_equivalence(model, reference, observations)
    invariance = map_history_invariance(model, observations)
    zero = route_zero_invariance(model, observations)
    fusion_zero = route_fusion_zero_audit(model)
    trainability = route_bypass_trainability(model, observations)
    critic = critic_211_audit(model, observations, seed=seed)
    rows = tensor_audit(source_state, {key: value.detach().cpu()
                                       for key, value in model.state_dict().items()})
    copied = [row for row in rows if row['source_key'] is not None and row['key'] != 'std']
    result = dict(
        observations=int(observations['proprio'].shape[0]), seed=int(seed),
        route_coverage=route_coverage(observations[ROUTE_ERROR_KEY]),
        actor_mean_equivalence=equivalence,
        zero_fusion_invariance=invariance,
        route_zero_invariance=zero,
        route_fusion_zero=fusion_zero,
        route_bypass_trainability=trainability,
        critic_211=critic,
        actor_copied_tensors=len(copied),
        actor_all_copied_equal=bool(all(row['equal_to_source'] for row in copied)),
        new_tensor_count=len(ROUTE_FUSION_KEYS),
    )
    result['passed'] = bool(
        result['actor_all_copied_equal'] and equivalence['passed']
        and invariance['passed'] and zero['passed'] and fusion_zero['passed']
        and trainability['passed'] and critic['passed'])
    return result


# --------------------------------------------------------------- validation
def _require_relative_path(value):
    if not isinstance(value, str) or not value or value.startswith('/') or '\\' in value:
        raise ValueError(f'unsafe contract source path: {value!r}')
    path = PurePosixPath(value)
    if any(part in ('', '.', '..') for part in path.parts) or path.is_absolute():
        raise ValueError(f'unsafe contract source path: {value!r}')
    return value


def validate_identity(value):
    """SDS 7.5: the ``identity`` block is exactly five keys with frozen values."""
    if not isinstance(value, dict):
        raise ValueError('checkpoint identity must be a mapping')
    if set(value) != set(IDENTITY_FIELDS):
        missing = sorted(set(IDENTITY_FIELDS) - set(value))
        extra = sorted(set(value) - set(IDENTITY_FIELDS))
        raise ValueError(f'checkpoint identity key contract mismatch: '
                         f'missing={missing} extra={extra}')
    expected = identity_block()
    for key in IDENTITY_FIELDS:
        if value[key] != expected[key]:
            raise ValueError(f'checkpoint identity {key} mismatch: '
                             f'{value[key]!r} != {expected[key]!r}')
    return value


def validate_schema7_keys(state):
    """Exact schema-7 envelope gate; rejects r1, schema<=6 and smoke containers.

    Runs before any tensor body is inspected so a legacy or foreign envelope is
    refused without needing the current model.  ``strict=False`` loading is never
    used anywhere in this module.
    """
    if not isinstance(state, dict):
        raise ValueError('schema-7 checkpoint must be a mapping')
    for key in ('container', 'smoke_container', 'kind', 'format', 'checkpoint_kind'):
        value = state.get(key)
        if isinstance(value, str) and value in SMOKE_CONTAINERS:
            raise ValueError(f'smoke container {value!r} is not a resumable r2 source')
    if 'formal_resumable' in state and not state['formal_resumable']:
        raise ValueError('formal_resumable=False containers are engineering-only '
                         'smoke states, never a pilot/formal resume source')
    if state.get('scope') == INITIALIZATION_SCOPE:
        raise ValueError('an initialization container is migration evidence, not a resume')
    schema = state.get('schema')
    if schema is None:
        if 'contract' in state or 'identity' not in state:
            raise LegacySchemaError('legacy checkpoint without an r2 schema marker '
                                    '(schema <= 6) is not resumable')
        raise LegacySchemaError('checkpoint has no schema marker; only schema 7 is accepted')
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError(f'checkpoint schema must be the integer 7, got {schema!r}')
    if schema < SCHEMA_VERSION:
        raise LegacySchemaError(f'legacy schema {schema} (<= 6) is not resumable')
    if schema > SCHEMA_VERSION:
        raise ValueError(f'unknown checkpoint schema {schema}; only 7 is accepted')
    if set(state) != set(SCHEMA7_TOP_LEVEL_KEYS):
        missing = sorted(set(SCHEMA7_TOP_LEVEL_KEYS) - set(state))
        extra = sorted(set(state) - set(SCHEMA7_TOP_LEVEL_KEYS))
        raise ValueError(f'schema-7 field contract mismatch: missing={missing} extra={extra}')
    validate_identity(state['identity'])
    if state['resume_mode'] != RESUME_MODE:
        raise ValueError(f'resume_mode must be {RESUME_MODE!r}')
    if state['recipe'] != recipe_config():
        raise ValueError('checkpoint recipe differs from the frozen 0.5-r2 recipe')
    if not _is_hex_digest(state['contract_sha256']):
        raise ValueError('contract_sha256 is not a SHA256 digest')
    model = state['model']
    if not isinstance(model, dict):
        raise ValueError('checkpoint model must be a tensor mapping')
    for key in ROUTE_FUSION_KEYS:
        if key not in model:
            raise ValueError(f'checkpoint model is missing the r2 bypass tensor {key!r}')
        if tuple(model[key].shape) != ROUTE_FUSION_SHAPES[key]:
            raise ValueError(f'checkpoint model bypass tensor {key!r} must be '
                             f'{list(ROUTE_FUSION_SHAPES[key])}')
    if CRITIC_FIRST_KEY not in model:
        raise ValueError('checkpoint model is missing the critic first layer')
    if tuple(model[CRITIC_FIRST_KEY].shape) != CRITIC_FIRST_SHAPE:
        raise ValueError(f'checkpoint model {CRITIC_FIRST_KEY} must be '
                         f'{list(CRITIC_FIRST_SHAPE)}, got '
                         f'{list(model[CRITIC_FIRST_KEY].shape)}')
    for key in CRITIC_LAYER_KEYS:
        if key not in model:
            raise ValueError(f'checkpoint model is missing {key!r}')
    return state


def validate_contract(value, *, require_closure=True, seed=None, mode=None):
    """Validate schema-7 identity, frozen documents, recipe and the source closure."""
    if not isinstance(value, dict):
        raise ValueError('schema-7 contract must be a mapping')
    identity = dict(code_version=CODE_VERSION, recipe_version=RECIPE_VERSION,
                    sds_revision=SDS_REVISION, checkpoint_schema=SCHEMA_VERSION,
                    architecture=ARCHITECTURE, task=TASK, protocol_id=PROTOCOL_ID,
                    flat_protocol_id=FLAT_PROTOCOL_ID)
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise ValueError(f'schema-7 contract {key} mismatch: '
                             f'{value.get(key)!r} != {expected!r}')
    if value.get('identity') != identity_block():
        raise ValueError('schema-7 contract identity block mismatch')
    documents = dict(sds_sha256=SDS_SHA256, design_decision_sha256=DESIGN_DECISION_SHA256,
                     implementation_decision_sha256=IMPLEMENTATION_DECISION_SHA256)
    for key, expected in documents.items():
        if value.get(key) != expected:
            raise ValueError(f'schema-7 contract {key} mismatch: '
                             f'{value.get(key)!r} != {expected!r}')
    if value.get('reward_version') != RECIPE_VERSION:
        raise ValueError('schema-7 contract reward_version mismatch')
    if value.get('reward_config_sha256') != reward_config_sha256():
        raise ValueError('schema-7 contract reward_config_sha256 mismatch')
    if value.get('reward_config') != reward_config():
        raise ValueError('schema-7 contract reward_config mismatch')
    if value.get('recipe') != recipe_config():
        raise ValueError('schema-7 contract recipe record mismatch')
    if value.get('observation') != OBSERVATION:
        raise ValueError('schema-7 contract observation must be the five r2 keys')
    if value.get('route_state') != spec_route_state():
        raise ValueError('schema-7 contract route_state mismatch')
    if value.get('observation_contract') != observation_contract():
        raise ValueError('schema-7 contract observation_contract mismatch')
    from .spec import mode_budget

    budget = mode_budget(value.get('mode'))
    if (value.get('total_budget') != budget['total']
            or value.get('update_cap') != budget['total']
            or value.get('stage_budgets') != budget['stages']
            or value.get('effective_cap_source') != budget['source']):
        raise ValueError('schema-7 contract budget is not the single mode_budget source')
    if seed is not None and value.get('seed') != int(seed):
        raise ValueError('schema-7 contract seed mismatch')
    if mode is not None and value.get('mode') != mode:
        raise ValueError('schema-7 contract mode mismatch')
    hashes = value.get('behavior_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('schema-7 contract lacks a nonempty behavior_sha256 manifest')
    for relative, digest in hashes.items():
        _require_relative_path(relative)
        if not _is_hex_digest(digest):
            raise ValueError(f'invalid behavior_sha256 digest for {relative!r}')
    if require_closure:
        current = behavior_hashes()
        if hashes != current:
            changed = sorted(set(hashes) ^ set(current)) or sorted(
                key for key in current if hashes.get(key) != current[key])
            raise ValueError('schema-7 source contract closure SHA mismatch: '
                             f'{changed[:8]}')
    return value


def validate_model_tensors(state, model):
    """Exact tensor key/shape/dtype/finiteness contract plus normalization and std."""
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        if not isinstance(state, dict):
            raise ValueError('model state must be a mapping')
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f'exact r2 model key contract mismatch: missing={missing[:8]} '
                         f'extra={extra[:8]}')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid model tensor: {key}')
    if ((state['actor.normalizer._std'] <= 0).any()
            or (state['actor.normalizer._var'] < 0).any()
            or state['actor.normalizer.count'].item() < 0
            or (state['critic_normalizer.var'] < 0).any()
            or state['critic_normalizer.count'].item() < 0):
        raise ValueError('Invalid normalization state')
    if ((state['std'] < STD_MIN).any() or (state['std'] > STD_MAX).any()):
        raise ValueError('Invalid exploration standard deviation')
    mask = torch.ones(CRITIC_NORMALIZER_DIMS, dtype=torch.bool,
                      device=state['critic_normalizer.continuous'].device)
    mask[48:52] = False
    if not torch.equal(state['critic_normalizer.continuous'], mask):
        raise ValueError('Critic continuous-feature mask mismatch')
    if int(state['actor.normalizer._mean'].shape[-1]) != ETH_NORMALIZER_DIMS:
        raise ValueError('Actor normalizer must stay 45-dimensional')
    if int(state['critic_normalizer.mean'].shape[-1]) != CRITIC_NORMALIZER_DIMS:
        raise ValueError('Critic normalizer must stay 80-dimensional')
    for key, shape in ROUTE_FUSION_SHAPES.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(f'route bypass tensor {key!r} must be {list(shape)}')
    if tuple(state[CRITIC_FIRST_KEY].shape) != CRITIC_FIRST_SHAPE:
        raise ValueError(f'{CRITIC_FIRST_KEY} must be {list(CRITIC_FIRST_SHAPE)}')
    return state


def validate_model_state(state, model, *, contract=None, required_fields=None,
                         require_closure=True):
    """Validate an r2 model payload for the runner/recovery call sites.

    Two explicit modes, so nothing is ever accepted implicitly:

    * tensor mode (``contract is None``): ``state`` must be exactly the model
      ``state_dict``; envelope keys such as ``rng`` or ``ledger`` are rejected.
    * envelope mode (``contract`` is the expected contract mapping): ``state``
      must have exactly the fields in ``required_fields`` (defaulting to the
      minimal ``{'contract','model'}``), ``state['contract']`` must pass
      :func:`validate_contract` and ``state['model']`` must be a valid tensor
      state.  Unknown tails are rejected, never ignored.

    Returns ``'model'`` or ``'schema7'`` so callers can log the accepted shape.
    """
    if contract is None:
        if not isinstance(state, dict):
            raise ValueError('model state must be a mapping')
        if any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError('model state in tensor mode must contain only tensors; '
                             'pass contract= for a schema-7 envelope')
        validate_model_tensors(state, model)
        return 'model'
    if not isinstance(state, dict):
        raise ValueError('schema-7 payload must be a mapping')
    fields = {'contract', 'model'} if required_fields is None else set(required_fields)
    if set(state) != fields:
        unknown = sorted(set(state) - fields)
        missing = sorted(fields - set(state))
        raise ValueError(f'schema-7 field contract mismatch: unknown={unknown} missing={missing}')
    if state['contract'] != contract:
        raise ValueError('schema-7 contract does not match the expected r2 contract')
    validate_contract(state['contract'], require_closure=require_closure)
    validate_model_tensors(state['model'], model)
    return 'schema7'


def load_model_state(model, state):
    """The only sanctioned model loader: exact key set with ``strict=True``."""
    if not isinstance(state, dict):
        raise ValueError('model state must be a mapping')
    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(f'refusing to load a non-r2 model state: missing={missing[:8]} '
                         f'extra={extra[:8]}; strict=False is never used')
    model.load_state_dict(state, strict=True)
    return model


def initialization_state(model, *, seed, source_path, source_sha256, audit_schema,
                         contract=None, source_state=None):
    """Build the ``scope='initialization'`` container (never a full resume checkpoint)."""
    seed = _check_seed(seed)
    if not _is_hex_digest(source_sha256) or source_sha256 != INIT_SHA256:
        raise EthMigrationError('initialization container requires the approved ETH SHA256')
    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state = dict(
        contract=deepcopy(spec_contract() if contract is None else contract),
        model=model_state,
        learning_state=dict(optimizer_initialized=False, completed_ppo_updates=0,
                            optimizer_steps=0),
        optimizer=dict(state=dict(), param_groups=[]), scope=INITIALIZATION_SCOPE,
        transfer_metadata=deepcopy(getattr(model, 'transfer_metadata', {})),
        audit_schema=audit_schema, source_checkpoint_sha256=source_sha256,
        source_model_sha256=(_state_sha256(source_state) if source_state is not None else None),
        migrated_model_sha256=_state_sha256(model_state), seed=seed,
        source_path=str(Path(source_path).resolve()),
        resume=False, physical_trajectory_continuous=False,
        notes=('initialization artifact: model tensors plus empty Adam and zero counters; '
               'it is not a schema-7 resume boundary and holds no curriculum/ledger/RNG'))
    return state


def _state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(_tensor_sha256(state[key]).encode())
    return digest.hexdigest()


def validate_initialization(state, *, require_contract=True, require_closure=False):
    """Validate the audited migration-zero container; rejects any resume promotion.

    The stored ``behavior_sha256`` is *recorded* evidence of the source tree that
    produced the artifact, not a live resume gate: an initialization container may
    legitimately be older than the working tree, and a real run re-derives the
    current contract from ``from_eth``.  Pass ``require_closure=True`` to demand
    that the recorded manifest still equals the current product closure.
    """
    if not isinstance(state, dict) or set(state) != set(INITIALIZATION_FIELDS):
        unknown = sorted(set(state) - set(INITIALIZATION_FIELDS)) if isinstance(state, dict) else None
        raise ValueError(f'initialization field contract mismatch: unknown={unknown}')
    if state['scope'] != INITIALIZATION_SCOPE:
        raise ValueError('initialization scope mismatch')
    if state['learning_state'] != dict(optimizer_initialized=False, completed_ppo_updates=0,
                                       optimizer_steps=0):
        raise ValueError('initialization has nonzero learning counters')
    if state['optimizer'] != dict(state=dict(), param_groups=[]):
        raise ValueError('initialization has nonempty Adam state')
    if state['source_checkpoint_sha256'] != INIT_SHA256:
        raise ValueError('initialization source SHA mismatch')
    if not _is_hex_digest(state['migrated_model_sha256']):
        raise ValueError('initialization migrated_model_sha256 is not a SHA256 digest')
    if state['source_model_sha256'] is not None \
            and not _is_hex_digest(state['source_model_sha256']):
        raise ValueError('initialization source_model_sha256 is not a SHA256 digest')
    metadata = state['transfer_metadata']
    if not isinstance(metadata, dict) or metadata.get('resume') is not False:
        raise ValueError('initialization transfer metadata must not claim resume')
    model_state = state['model']
    if not isinstance(model_state, dict):
        raise ValueError('initialization model must be a tensor mapping')
    for key, shape in ROUTE_FUSION_SHAPES.items():
        if key not in model_state or tuple(model_state[key].shape) != shape:
            raise ValueError(f'initialization model is missing the r2 bypass {key!r}')
        if model_state[key].any():
            raise ValueError('initialization route bypass must be zero')
    if tuple(model_state[CRITIC_FIRST_KEY].shape) != CRITIC_FIRST_SHAPE:
        raise ValueError('initialization Critic must be the 211-input r2 shape')
    if not torch.equal(model_state['std'],
                       torch.full_like(model_state['std'], INITIAL_STD)):
        raise ValueError('initialization std must be fully 0.15')
    if require_contract:
        validate_contract(state['contract'], require_closure=require_closure)
    return state
