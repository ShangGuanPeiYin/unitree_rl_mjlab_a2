"""v0.6 ActorCritic: ETH Actor migration plus zero-initialized heightmap/history/
route bypasses and a fresh 211-wide Critic.

Contract (SDS ``SDS_A2_IdealHeightMap_Stairs40_v0.6.md`` section 2.2/7.2):

* only the ETH Actor MLP (``mlp.0/2/4/6``) and the frozen 45-dimensional input
  normalizer are migrated; the ETH std, Critic and Adam are never inherited;
* the heightmap/history fusion (``actor.fusion``) and the route fusion
  (``actor.route_fusion``) are constructed, enter the single optimizer and are
  then overwritten with zeros so the initial action mean equals ETH;
* the fresh Critic consumes 211 features = 80 raw proprio+privileged,
  64 map features, 64 history features and the 3 route features;
* the initial action std is exactly ``initial_std`` (0.30) and is projected to
  ``[std_min, std_max]`` = ``[0.05, 1.00]`` after every Adam step.

All numbers are read from :mod:`src.tasks.velocity.v06.spec` (the single frozen
number source, owned by agent A).  This module contains no independent tuning
value; the only literal here is the SDS-published ETH checkpoint SHA, used as an
identity contract when the spec module does not expose it.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from src.tasks.velocity.rl.ideal_heightmap_actor_critic import (
    CriticNormalizer, FusedActor, FrozenETHNormalizer, IdealHeightmapActorCritic,
    history_encoder, map_encoder, mlp)

from . import spec

# ---------------------------------------------------------------- spec access
#: SDS 2.2 identity of the approved ETH initialization source.  Read from the
#: frozen spec module when it exposes it; the literal is only the published hash.
INIT_SHA256 = getattr(spec, 'INIT_SHA256',
                      '8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722')
INIT_CHECKPOINT = getattr(
    spec, 'INIT_CHECKPOINT',
    getattr(spec, 'INIT_PATH',
            '/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt'))

#: Canonical PPO field -> tolerated spec ``PPO_CONFIG`` key spellings.  The
#: first spelling is the one requested from agent A (see
#: ``implementation/coordination/QC_training_stack.md``).
_PPO_ALIASES = {
    'num_envs': ('num_envs', 'train_envs', 'n_envs'),
    'eval_num_envs': ('eval_num_envs', 'evaluation_envs', 'n_eval_envs'),
    'rollout': ('rollout', 'rollout_steps'),
    'epochs': ('epochs', 'n_epochs'),
    'minibatches': ('minibatches', 'n_minibatches'),
    'learning_rate': ('learning_rate', 'lr', 'initial_learning_rate'),
    'min_learning_rate': ('min_learning_rate', 'min_lr', 'learning_rate_min'),
    'max_learning_rate': ('max_learning_rate', 'max_lr', 'learning_rate_max'),
    'desired_kl': ('desired_kl', 'kl_desired'),
    'kl_high': ('kl_high', 'kl_upper'),
    'kl_low': ('kl_low', 'kl_lower'),
    'gamma': ('gamma', 'discount'),
    'gae_lambda': ('gae_lambda', 'lambda_', 'lam'),
    'clip': ('clip', 'clip_ratio', 'clip_coef'),
    'entropy': ('entropy', 'entropy_coef', 'entropy_coefficient'),
    'value_coef': ('value_coef', 'value_coefficient', 'vf_coef'),
    'max_grad_norm': ('max_grad_norm', 'grad_clip', 'gradient_clip'),
    'clipped_value_loss': ('clipped_value_loss', 'use_clipped_value_loss'),
    'initial_std': ('initial_std', 'init_std', 'std_init'),
    'std_min': ('std_min', 'min_std'),
    'std_max': ('std_max', 'max_std'),
    'adam_betas': ('adam_betas', 'betas'),
    'adam_eps': ('adam_eps', 'eps'),
    'weight_decay': ('weight_decay',),
}
PPO_REQUIRED = tuple(_PPO_ALIASES)


def _lookup(mapping, name, aliases, *, source):
    if not isinstance(mapping, dict):
        raise TypeError(f'{source} must be a mapping, got {type(mapping).__name__}')
    for key in aliases:
        if key in mapping:
            return mapping[key]
    raise KeyError(f'{source} is missing {name!r} (accepted keys: {aliases})')


def _read_ppo_settings():
    raw = getattr(spec, 'PPO_CONFIG')
    if not hasattr(raw, '__getitem__'):
        raise TypeError('spec.PPO_CONFIG must be a mapping')
    settings = {}
    missing = []
    for name in PPO_REQUIRED:
        try:
            settings[name] = _lookup(dict(raw), name, _PPO_ALIASES[name], source='spec.PPO_CONFIG')
        except KeyError:
            missing.append(name)
    if missing:
        raise KeyError(f'spec.PPO_CONFIG is missing the frozen PPO fields {missing}')
    for name in ('kl_high', 'kl_low'):
        if settings[name] is None:
            settings[name] = (2.0 * float(settings['desired_kl']) if name == 'kl_high'
                              else 0.5 * float(settings['desired_kl']))
    settings['adam_betas'] = tuple(float(value) for value in settings['adam_betas'])
    for name in ('num_envs', 'eval_num_envs', 'rollout', 'epochs', 'minibatches'):
        settings[name] = int(settings[name])
    return settings


#: Normalized, validated SDS section 4 PPO settings (read-only by convention).
PPO_SETTINGS = _read_ppo_settings()

NUM_ENVS = PPO_SETTINGS['num_envs']
EVAL_NUM_ENVS = PPO_SETTINGS['eval_num_envs']
ROLLOUT = PPO_SETTINGS['rollout']
EPOCHS = PPO_SETTINGS['epochs']
MINIBATCHES = PPO_SETTINGS['minibatches']
INITIAL_STD = float(PPO_SETTINGS['initial_std'])
STD_MIN = float(PPO_SETTINGS['std_min'])
STD_MAX = float(PPO_SETTINGS['std_max'])
ADAM_BETAS = PPO_SETTINGS['adam_betas']
ADAM_EPS = float(PPO_SETTINGS['adam_eps'])
WEIGHT_DECAY = float(PPO_SETTINGS['weight_decay'])

# ------------------------------------------------------------------- network
PROPRIO_DIM = int(spec.PROPRIO_DIM)
HISTORY_LEN = int(spec.HISTORY_LEN)
HEIGHTMAP_SHAPE = tuple(spec.HEIGHTMAP_SHAPE)
ROUTE_DIM = int(spec.ROUTE_DIM)
PRIVILEGED_DIM = int(spec.PRIVILEGED_DIM)
CONTROL_DT = float(spec.CONTROL_DT)
N_JOINTS = int(spec.N_JOINTS)
CRITIC_INPUT_DIMS = int(spec.CRITIC_WIDTH)
ARCHITECTURE = spec.ARCHITECTURE
CODE_VERSION = spec.CODE_VERSION
RECIPE_VERSION = spec.RECIPE_VERSION
SCHEMA_VERSION = int(getattr(spec, 'CHECKPOINT_SCHEMA', spec.SCHEMA_VERSION))
TASK_ID = getattr(spec, 'TASK_ID', spec.TASK)
PROTOCOL_ID = spec.PROTOCOL_ID
TRAINING_SEED = int(getattr(spec, 'TRAINING_SEED', 42))
ROUTE_ERROR_KEY = 'route_error'
OBSERVATION_KEYS = ('proprio', 'history', 'heightmap', ROUTE_ERROR_KEY)
OBSERVATION_KEYS_CRITIC = OBSERVATION_KEYS + ('privileged',)
OBSERVATION_SHAPES = {
    'proprio': (PROPRIO_DIM,),
    'history': (HISTORY_LEN, PROPRIO_DIM),
    'heightmap': HEIGHTMAP_SHAPE,
    ROUTE_ERROR_KEY: (ROUTE_DIM,),
    'privileged': (PRIVILEGED_DIM,),
}

ACTOR_HIDDEN = (512,)
CRITIC_FIRST_SHAPE = (512, CRITIC_INPUT_DIMS)
#: Heightmap/history fusion bypass and the route bypass (all zero at migration).
FUSION_KEYS = ('actor.fusion.weight', 'actor.fusion.bias')
ROUTE_FUSION_KEYS = ('actor.route_fusion.weight', 'actor.route_fusion.bias')
BYPASS_KEYS = FUSION_KEYS + ROUTE_FUSION_KEYS
#: Addressable optimizer groups (SDS 4: actor / critic / fusion / std norms).
GRADIENT_GROUPS = ('actor', 'critic', 'fusion', 'std')
RESUME_MODE = 'new_episode'
HEX_DIGITS = set('0123456789abcdef')

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
NORMALIZER_KEYS = tuple(sorted(_ETH_NORMALIZER_SHAPES))
FORBIDDEN_SOURCE_BLOCKS = ('critic_state_dict', 'optimizer_state_dict', 'optimizer')


class EthMigrationError(ValueError):
    """The ETH source or the migrated state violates SDS 2.2."""


class ObservationError(ValueError):
    """An observation bundle violates the frozen five-key contract."""


def _is_hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX_DIGITS


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 1:
        raise ValueError('seed must be a positive integer')
    return int(seed)


# ------------------------------------------------------------------ actor net
class RouteFusedActor(FusedActor):
    """Frozen ETH Actor structure plus the zero-initialized route bypass.

    ``super().__init__()`` builds the shared structure in the shared order
    (normalizer, map encoder, history encoder, 45-512-256-128-12 MLP, the
    128->512 fusion); ``route_fusion`` is then really constructed (it consumes
    the run RNG) and overwritten with zeros.
    """

    def __init__(self):
        super().__init__()
        self.route_fusion = nn.Linear(ROUTE_DIM, ACTOR_HIDDEN[0])
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


class V06ActorCritic(IdealHeightmapActorCritic):
    """v0.6 route-state ActorCritic (ETH Actor + zero bypass + fresh 211 Critic)."""

    architecture_version = ARCHITECTURE
    std_min = STD_MIN
    std_max = STD_MAX

    def __init__(self):
        self.recipe = RECIPE_VERSION
        super().__init__()
        self.actor = RouteFusedActor()
        self.critic = mlp([CRITIC_INPUT_DIMS, 512, 256, 128, 1])
        with torch.no_grad():
            self.std.fill_(INITIAL_STD)

    # ------------------------------------------------------------ validation
    @staticmethod
    def _validate(obs, critic=False, consumer='forward'):
        if not isinstance(obs, dict):
            raise ObservationError(f'{consumer}: observation must be a mapping')
        keys = OBSERVATION_KEYS_CRITIC if critic else OBSERVATION_KEYS
        batch = device = None
        for key in keys:
            if key not in obs:
                raise ObservationError(
                    f'{consumer}: observation is missing {key!r}; v06 requires all of '
                    f'{list(keys)} and never zero fills a missing key')
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise ObservationError(f'{consumer}: observation {key!r} must be a tensor')
            shape = OBSERVATION_SHAPES[key]
            if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape:
                raise ObservationError(
                    f'{consumer}: observation {key!r} must be [B,{",".join(map(str, shape))}], '
                    f'got {list(value.shape)}')
            if value.dtype != torch.float32:
                raise ObservationError(
                    f'{consumer}: observation {key!r} must be float32, got {value.dtype}')
            if device is None:
                device = value.device
            elif value.device != device:
                raise ObservationError(f'{consumer}: observation {key!r} device differs')
            if batch is None:
                batch = value.shape[0]
            elif value.shape[0] != batch:
                raise ObservationError(f'{consumer}: observation {key!r} batch differs')
            if not torch.isfinite(value).all():
                raise ObservationError(f'{consumer}: observation {key!r} contains NaN/Inf')

    # --------------------------------------------------------------- forward
    def forward(self, obs):
        self._validate(obs, consumer='forward')
        return self.actor(obs)

    def evaluate(self, obs):
        self._validate(obs, critic=True, consumer='evaluate')
        raw = self.critic_normalizer(torch.cat((obs['proprio'], obs['privileged']), -1))
        # History uses Critic-owned current-proprio moments, never the frozen ETH stats.
        history = ((obs['history'] - self.critic_normalizer.mean[:PROPRIO_DIM])
                   / self.critic_normalizer.var[:PROPRIO_DIM].sqrt().clamp_min(1e-5)).clamp(-5, 5)
        return self.critic(torch.cat((raw, self.critic_map_encoder(obs['heightmap']),
                                      self.critic_history_encoder(history),
                                      obs[ROUTE_ERROR_KEY]), -1))

    def update_critic_statistics(self, obs):
        self._validate(obs, critic=True, consumer='update_critic_statistics')
        self.critic_normalizer.update(
            torch.cat((obs['proprio'], obs['privileged']), -1).detach())

    # ------------------------------------------------------------ deployable
    @torch.no_grad()
    def constrain_std(self):
        """Project the trainable std after every Adam step (SDS 4)."""
        self.std.clamp_(STD_MIN, STD_MAX)

    def action_distribution(self, obs):
        self._validate(obs, consumer='action_distribution')
        return Normal(self(obs), self.std.clamp(STD_MIN, STD_MAX))

    def act(self, obs, deterministic=False):
        self._validate(obs, consumer='act')
        return self(obs) if deterministic else self.action_distribution(obs).sample()

    # -------------------------------------------------------------- grouping
    def parameter_groups(self):
        """Map the four SDS gradient-norm groups to their parameter names."""
        groups = {name: [] for name in GRADIENT_GROUPS}
        for name, _parameter in self.named_parameters():
            if name == 'std':
                groups['std'].append(name)
            elif name.startswith(('actor.fusion.', 'actor.route_fusion.',
                                  'actor.map_encoder.', 'actor.history_encoder.')):
                groups['fusion'].append(name)
            elif name.startswith('actor.'):
                groups['actor'].append(name)
            elif name.startswith('critic'):
                groups['critic'].append(name)
            else:
                raise ValueError(f'parameter {name!r} belongs to no gradient group')
        return groups

    def optimizer_parameters(self):
        """Every trainable parameter exactly once, in a stable order."""
        seen, parameters = set(), []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
        return parameters

    def bypass_state_dict(self):
        state = self.state_dict()
        return {key: state[key].detach().clone() for key in BYPASS_KEYS}

    def route_state_dict(self):
        """The zero-initialized bypass block plus its content digest (SDS 2.2)."""
        tensors = self.bypass_state_dict()
        return dict(
            keys=list(BYPASS_KEYS),
            fusion_keys=list(FUSION_KEYS),
            route_fusion_keys=list(ROUTE_FUSION_KEYS),
            frozen_tensors={key: value for key, value in tensors.items()
                            if key in ('actor.mlp.6.bias',)},
            tensors=tensors,
            all_zero=all(float(value.abs().sum()) == 0.0 for value in tensors.values()),
            sha256=state_sha256({key: value for key, value in tensors.items()}),
            initial_std=[float(value) for value in self.std.detach().reshape(-1)],
            std_range=[STD_MIN, STD_MAX],
            architecture=ARCHITECTURE,
            version=CODE_VERSION,
        )

    # ------------------------------------------------------------- migration
    @classmethod
    def from_eth(cls, path=None, expected_sha=None, seed=42, device='cpu'):
        """Construct the v0.6 network from the approved ETH checkpoint (new run)."""
        path = Path(path) if path is not None else Path(INIT_CHECKPOINT)
        expected_sha = expected_sha if expected_sha is not None else INIT_SHA256
        seed = check_seed(seed)
        source, state, forbidden = load_validated_eth(path, expected_sha)
        torch.manual_seed(seed)
        model = cls()
        model.actor.mlp.load_state_dict(
            {key[len('mlp.'):]: value for key, value in state.items()
             if key.startswith('mlp.')}, strict=True)
        model.actor.normalizer.load_state_dict(
            {key[len('obs_normalizer.'):]: value for key, value in state.items()
             if key.startswith('obs_normalizer.')}, strict=True)
        with torch.no_grad():
            for key in FUSION_KEYS + ROUTE_FUSION_KEYS:
                target = dict(model.named_parameters())[key]
                target.zero_()
            model.std.fill_(INITIAL_STD)
        if any(float(dict(model.named_parameters())[key].detach().abs().sum()) != 0.0
               for key in FUSION_KEYS + ROUTE_FUSION_KEYS):
            raise EthMigrationError('heightmap/history/route bypass is not zero after migration')
        if not torch.equal(model.std.detach(),
                           torch.full_like(model.std.detach(), INITIAL_STD)):
            raise EthMigrationError(f'action std is not exactly {INITIAL_STD} after migration')
        if tuple(model.critic[0].weight.shape) != CRITIC_FIRST_SHAPE:
            raise EthMigrationError('Critic first layer is not the 211-input v06 shape')
        if list(model.actor.normalizer.parameters()):
            raise EthMigrationError('ETH normalizer must stay parameter-free/frozen')
        model.actor.normalizer.eval()
        model.transfer_metadata = dict(
            kind='eth_actor_mlp_and_frozen_normalizer_only',
            source_path=str(Path(path).resolve()),
            sha256=sha256_file(path),
            source_iteration=int(source.get('iter', 0)) if isinstance(source, dict) else 0,
            source_actor_keys=list(ETH_ACTOR_KEYS),
            source_normalizer_keys=list(NORMALIZER_KEYS),
            forbidden_source_blocks=sorted(forbidden),
            forbidden_source_blocks_migrated=[],
            architecture=ARCHITECTURE, code_version=CODE_VERSION, recipe=RECIPE_VERSION,
            seed=seed, actor_statistics='copied_complete_and_frozen',
            normalization='frozen_ETH_normalizer_buffers',
            fusion='zeros_after_migration', route_fusion='zeros_after_migration',
            critic=f'fresh_seeded_{CRITIC_INPUT_DIMS}', critic_statistics='fresh',
            optimizer='fresh_adam', std_before=state['std'].detach().tolist(),
            std_after=model.std.detach().tolist(),
            std_source=f'overwritten_to_{INITIAL_STD}_not_inherited',
            std_range=[STD_MIN, STD_MAX], resume=False, scope='initialization')
        return model.to(device)

    @classmethod
    def migrate_eth_checkpoint(cls, path=None, expected_sha=None, *, seed=42, device='cpu'):
        """Alias with the SDS 7.2 name used by the entry points."""
        return cls.from_eth(path, expected_sha, seed, device)

    #: W0 4.5 spelling kept for the frozen interface.
    migrate_eth = migrate_eth_checkpoint


def load_validated_eth(path, expected_sha):
    """CPU-only migration step 1: SHA, exact keys, shapes, dtypes, finiteness."""
    if not _is_hex_digest(expected_sha) or expected_sha != INIT_SHA256:
        raise EthMigrationError('only the approved ETH Actor SHA256 may be migrated')
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != INIT_SHA256:
        raise EthMigrationError('ETH checkpoint SHA256 mismatch')
    source = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(source, dict) or not isinstance(source.get('actor_state_dict'), dict):
        raise EthMigrationError('ETH checkpoint has no actor_state_dict')
    state = source['actor_state_dict']
    if set(state) != set(ETH_ACTOR_KEYS):
        raise EthMigrationError(
            'ETH actor key contract mismatch: missing=%s extra=%s'
            % (sorted(set(ETH_ACTOR_KEYS) - set(state)), sorted(set(state) - set(ETH_ACTOR_KEYS))))
    shapes = {**_ETH_MLP_SHAPES, **_ETH_NORMALIZER_SHAPES, 'std': (N_JOINTS,)}
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise EthMigrationError(f'ETH tensor shape mismatch: {key}')
        expected_dtype = torch.int64 if key == 'obs_normalizer.count' else torch.float32
        if value.dtype != expected_dtype:
            raise EthMigrationError(f'ETH tensor dtype mismatch: {key}')
        if not torch.isfinite(value).all():
            raise EthMigrationError(f'non-finite ETH tensor: {key}')
        if value.device.type != 'cpu':
            raise EthMigrationError('ETH migration must validate on the CPU')
    if (state['obs_normalizer._std'] <= 0).any() or (state['obs_normalizer._var'] < 0).any():
        raise EthMigrationError('ETH normalizer std/var is invalid')
    if int(state['obs_normalizer.count'].item()) < 0 or (state['std'] <= 0).any():
        raise EthMigrationError('ETH normalizer count or std is invalid')
    forbidden = {key: True for key in FORBIDDEN_SOURCE_BLOCKS if key in source}
    return source, state, forbidden


# ------------------------------------------------------------------ helpers
def state_sha256(state):
    """Stable SHA-256 over a state dict (sorted keys, raw tensor bytes, shapes)."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f'state entry {key!r} must be a tensor')
        tensor = value.detach().to('cpu').contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def route_error_from_pose(local_y, yaw_world, lateral_scale=0.20, clip=4.0):
    """Reference ``route_error = [clip(local_y/0.20, -4, 4), sin(e), cos(e)]``."""
    local_y = torch.as_tensor(local_y, dtype=torch.float32)
    yaw_world = torch.as_tensor(yaw_world, dtype=torch.float32)
    return torch.stack((torch.clamp(local_y / lateral_scale, -clip, clip),
                        torch.sin(yaw_world), torch.cos(yaw_world)), -1)


def synthetic_observations(batch=64, seed=42, device='cpu'):
    """Deterministic five-key observation bundle for the migration audit."""
    generator = torch.Generator().manual_seed(check_seed(seed))
    local_y = torch.empty(batch).uniform_(-1.0, 1.0, generator=generator)
    yaw = torch.empty(batch).uniform_(-math.pi, math.pi, generator=generator)
    return dict(
        proprio=torch.randn(batch, PROPRIO_DIM, generator=generator),
        history=torch.randn(batch, HISTORY_LEN, PROPRIO_DIM, generator=generator),
        heightmap=torch.rand(batch, *HEIGHTMAP_SHAPE, generator=generator),
        route_error=route_error_from_pose(local_y, yaw),
        privileged=torch.randn(batch, PRIVILEGED_DIM, generator=generator),
    )


def eth_reference_actor(state, device='cpu'):
    """The ETH Actor rebuilt from the recorded source tensors (action-mean oracle)."""
    reference = FusedActor()
    reference.mlp.load_state_dict(
        {key[len('mlp.'):]: value for key, value in state.items()
         if key.startswith('mlp.')}, strict=True)
    reference.normalizer.load_state_dict(
        {key[len('obs_normalizer.'):]: value for key, value in state.items()
         if key.startswith('obs_normalizer.')}, strict=True)
    return reference.to(device).eval()


def action_mean_equivalence(model, source_state, observations=None, *, seed=42,
                            atol=1e-6):
    """Max |mean_v06 - mean_ETH| on synthetic inputs (SDS 9.2 acceptance)."""
    observations = observations or synthetic_observations(seed=seed, device=model.std.device)
    reference = eth_reference_actor(source_state, device=model.std.device)
    with torch.no_grad():
        expected = reference({key: value.to(model.std.device) for key, value in observations.items()
                              if key in ('proprio', 'history', 'heightmap')})
        actual = model(observations)
    if expected.shape != actual.shape:
        raise EthMigrationError('ETH reference and v06 action means have different shapes')
    error = float((actual - expected).abs().max())
    if not math.isfinite(error) or error > atol:
        raise EthMigrationError(
            f'action mean migration error {error:.3e} exceeds the {atol:g} bound')
    return dict(max_abs_error=error, atol=atol, batch=int(expected.shape[0]),
                seed=int(seed), method='deterministic_mean')


def audit_migration(model, source_state, observations=None, *, seed=42):
    """Full SDS 9.2 migration evidence block."""
    observations = observations or synthetic_observations(seed=seed,
                                                          device=model.std.device)
    equivalence = action_mean_equivalence(model, source_state, observations, seed=seed)
    route = model.route_state_dict()
    groups = model.parameter_groups()
    return dict(
        sha256=model.transfer_metadata.get('sha256'),
        source_path=model.transfer_metadata.get('source_path'),
        action_mean=equivalence,
        bypass=route,
        bypass_all_zero=bool(route['all_zero']),
        initial_std=[float(value) for value in model.std.detach().reshape(-1)],
        initial_std_exact=bool(torch.equal(
            model.std.detach(),
            torch.full_like(model.std.detach(), INITIAL_STD))),
        std_range=[STD_MIN, STD_MAX],
        critic_input_dims=int(model.critic[0].weight.shape[1]),
        critic_fresh=True,
        optimizer='fresh_adam_not_built_by_model',
        normalizer_frozen=not list(model.actor.normalizer.parameters()),
        parameter_groups={name: len(values) for name, values in groups.items()},
    )


def validate_transfer_metadata(metadata, *, source_sha256=INIT_SHA256):
    """Q006 section 4: the ETH initialization evidence a resumed run must carry.

    The metadata is *source evidence* only: restoring it never re-runs the ETH
    migration, never resets the zero/fused layers and never invents defaults.
    A missing, forged or schema-conflicting block is refused so the caller can
    stop before any new learning update is billed.
    """
    if not isinstance(metadata, dict) or not metadata:
        raise EthMigrationError(
            'the checkpoint has no ETH transfer metadata; v0.6 cannot save a resume '
            'without the original initialization evidence')
    if source_sha256 != INIT_SHA256:
        raise EthMigrationError('only the approved ETH source SHA may be validated')
    sha = metadata.get('sha256')
    if sha != INIT_SHA256:
        raise EthMigrationError(
            'ETH transfer metadata SHA256 is not the approved ETH checkpoint; the '
            'initialization source conflicts with the frozen contract')
    if metadata.get('scope', 'initialization') != 'initialization':
        raise EthMigrationError('ETH transfer metadata scope is not initialization')
    if metadata.get('route_fusion') != 'zeros_after_migration':
        raise EthMigrationError(
            'ETH transfer metadata route_fusion is not the zero-after-migration record')
    if 'fusion' in metadata and metadata.get('fusion') != 'zeros_after_migration':
        raise EthMigrationError(
            'ETH transfer metadata fusion is not the zero-after-migration record')
    if metadata.get('architecture') not in (None, ARCHITECTURE):
        raise EthMigrationError('ETH transfer metadata architecture is not v0.6')
    initialization_recipes = (None, RECIPE_VERSION)
    initialization_versions = (None, CODE_VERSION)
    if getattr(spec, 'CONTINUATION_ID', None) == 'v06-c1-speed030060-budget36000':
        # C1 inherits original ETH evidence without relabelling initialization.
        initialization_recipes += ('0.6-go2refs-r1',)
        initialization_versions += ('0.6.0',)
    if metadata.get('recipe') not in initialization_recipes:
        raise EthMigrationError('ETH transfer metadata recipe is not v0.6')
    if metadata.get('code_version') not in initialization_versions:
        raise EthMigrationError('ETH transfer metadata code_version is not v0.6')
    if metadata.get('resume') not in (False, None):
        raise EthMigrationError(
            'ETH transfer metadata must describe the original initialization, not a '
            'resume (the resume source is recorded separately)')
    actor_keys = metadata.get('source_actor_keys')
    if actor_keys is not None:
        if not isinstance(actor_keys, (list, tuple)) or set(
                str(key) for key in actor_keys) != set(ETH_ACTOR_KEYS):
            raise EthMigrationError(
                'ETH transfer metadata does not list exactly the migrated ETH actor keys')
    normalizer_keys = metadata.get('source_normalizer_keys')
    if normalizer_keys is not None:
        if not isinstance(normalizer_keys, (list, tuple)) or set(
                str(key) for key in normalizer_keys) != set(NORMALIZER_KEYS):
            raise EthMigrationError(
                'ETH transfer metadata does not list exactly the frozen normalizer keys')
    migrated = metadata.get('forbidden_source_blocks_migrated')
    if migrated not in (None, [], ()):
        raise EthMigrationError(
            'ETH transfer metadata records a forbidden source block as migrated')
    iteration = metadata.get('source_iteration')
    if iteration is not None and (isinstance(iteration, bool)
                                  or not isinstance(iteration, Integral)
                                  or int(iteration) < 0):
        raise EthMigrationError('ETH transfer metadata source_iteration is invalid')
    return True


def validate_initialization(state, *, model_factory=V06ActorCritic):
    """Schema-8 ``initialization`` block validation (recovered from ETH only once)."""
    if not isinstance(state, dict):
        raise ValueError('initialization block must be a mapping')
    for field in ('contract', 'model', 'source_path', 'source_sha256', 'migrated_model_sha256',
                  'transfer_metadata', 'resume_mode'):
        if field not in state:
            raise ValueError(f'initialization block is missing {field!r}')
    if state['resume_mode'] != RESUME_MODE:
        raise ValueError('initialization must declare resume_mode=new_episode')
    if state.get('resume') not in (False, None):
        raise ValueError('initialization block must not describe a resume')
    if state['source_sha256'] != INIT_SHA256:
        raise EthMigrationError('initialization source SHA is not the approved ETH checkpoint')
    if not _is_hex_digest(str(state['migrated_model_sha256'])):
        raise ValueError('initialization migrated_model_sha256 is not a hex digest')
    metadata = state['transfer_metadata']
    validate_transfer_metadata(metadata, source_sha256=state['source_sha256'])
    resume_origin = state.get('resume_origin')
    if resume_origin is not None and not isinstance(resume_origin, dict):
        raise ValueError('initialization resume_origin must be a mapping or null')
    chain = state.get('resume_chain')
    if chain is not None and not isinstance(chain, (list, tuple)):
        raise ValueError('initialization resume_chain must be a list')
    return True


__all__ = [
    'ACTOR_HIDDEN', 'ADAM_BETAS', 'ADAM_EPS', 'ARCHITECTURE', 'BYPASS_KEYS',
    'CODE_VERSION', 'CRITIC_FIRST_SHAPE', 'CRITIC_INPUT_DIMS', 'CONTROL_DT',
    'EPOCHS', 'ETH_ACTOR_KEYS', 'EVAL_NUM_ENVS', 'EthMigrationError',
    'FORBIDDEN_SOURCE_BLOCKS', 'FUSION_KEYS', 'GRADIENT_GROUPS', 'HEIGHTMAP_SHAPE',
    'HISTORY_LEN', 'INITIAL_STD', 'INIT_CHECKPOINT', 'INIT_SHA256', 'MINIBATCHES',
    'N_JOINTS', 'NORMALIZER_KEYS', 'NUM_ENVS', 'OBSERVATION_KEYS', 'ObservationError',
    'PPO_SETTINGS', 'PRIVILEGED_DIM', 'PROPRIO_DIM', 'PROTOCOL_ID', 'RECIPE_VERSION',
    'RESUME_MODE', 'ROLLOUT', 'TRAINING_SEED', 'ROUTE_DIM', 'ROUTE_ERROR_KEY', 'ROUTE_FUSION_KEYS',
    'RouteFusedActor', 'SCHEMA_VERSION', 'STD_MAX', 'STD_MIN', 'TASK_ID',
    'V06ActorCritic', 'WEIGHT_DECAY', 'action_mean_equivalence', 'audit_migration',
    'check_seed', 'eth_reference_actor', 'load_validated_eth', 'route_error_from_pose',
    'sha256_file', 'state_sha256', 'synthetic_observations', 'validate_initialization',
    'validate_transfer_metadata',
]
