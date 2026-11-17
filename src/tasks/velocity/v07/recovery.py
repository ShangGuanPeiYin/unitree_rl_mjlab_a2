"""v0.6 schema-9 checkpoint: serialization, validation, resume refusal and the
four-way budget audit.

SDS 8.2 requires a schema-9 checkpoint that carries the model, the single Adam
state, the current learning rate and the normalization state, **all** learning
counts and curriculum state, the pending stage-transition transaction, the
evaluation case tables and completed gate identities, the Python/NumPy/Torch/
CUDA/environment/sampler RNGs, the SDS/config/source-closure/initialization
source SHAs, and an explicit ``resume_mode=new_episode``.

Resume is refused for R1/R2 recipes, other tasks, smoke/probe containers and any
non-ETH initialization source.  A restored run starts a **new episode** and never
re-runs an already billed learning update.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from . import spec
from . import regression_policy
from . import evaluation as evaluation_module
from . import ledger as ledger_module
from . import model as model_module
from . import ppo as ppo_module
from . import speed

SCHEMA = int(getattr(spec, 'CHECKPOINT_SCHEMA', spec.SCHEMA_VERSION))
IDENTITY_FIELDS = ('task', 'recipe', 'code_version', 'protocol', 'architecture', 'sds_sha256')
TOP_LEVEL_KEYS = ('schema', 'identity', 'contract_sha256', 'model', 'optimizer',
                  'normalizer', 'learning', 'rng', 'curriculum', 'evaluation',
                  'source', 'recipe', 'resume_mode', 'initialization', 'gate',
                  'retention', 'ledger', 'pending_transition', 'regression_policy_state')
#: The probe writes a valid schema-9 checkpoint tagged with ``container``; the tag
#: is an optional envelope key and a formal training resume always refuses it.
OPTIONAL_TOP_LEVEL_KEYS = ('container',)
PROBE_CONTAINER = getattr(spec, 'PROBE_CONTAINER', 'v07-probe-state-v1')
CURRICULUM_FIELDS = ('phase', 'substage', 'level_tick', 'generation',
                     'generation_updates', 'spent_by_phase', 'spent_by_substage',
                     'episode_fifo', 'ema', 'ready_streak', 'hard_streak', 'cooldown',
                     'gate_streak', 'retention_streaks',
                     'minimum_level_failure_streak', 'sampler_rng_state')
RNG_FIELDS = ('python', 'numpy', 'torch', 'cuda', 'env', 'sampler')
RESUME_MODE = 'new_episode'
LEARNING_FIELDS = ('update', 'optimizer_step', 'adam_steps', 'control_samples',
                   'physics_substeps', 'num_envs', 'rollout', 'epochs', 'minibatches')
#: Containers that may never be used as a normal v0.6 training resume source.
FOREIGN_MODES = ('smoke', 'probe', 'pilot')
FOREIGN_CONTAINERS = ('v05-smoke-state-v1', 'v05-r2-smoke-state-v1',
                      'v07-smoke-state-v1', 'v07-probe-state-v1')
ALLOWED_RECIPES = (spec.RECIPE_VERSION,)
ALLOWED_TASKS = (getattr(spec, 'TASK_ID', getattr(spec, 'TASK', None)),)
HEX_DIGITS = set('0123456789abcdef')


#: Q005 (SHA 0159b032...) / Q006 (SHA 798a7cf4...) restricted cross-source recovery
#: bridge.  Exactly one old trial02 checkpoint may be resumed by the repaired
#: source; every other source mismatch stays refused.
Q005_DECISION_SHA256 = '0159b03296ae078c96d6fd32851e736ce97a5119a89475b6bc4d85004464cb2d'
Q006_DECISION_SHA256 = ledger_module.Q006_DECISION_SHA256
Q006_ERRATA_SHA256 = ledger_module.Q006_ERRATA_SHA256
Q007_DECISION_SHA256 = ledger_module.Q007_DECISION_SHA256
Q007_ERRATA_SHA256 = ledger_module.Q007_ERRATA_SHA256
Q007_SOURCE_CHECKPOINT_SHA256 = ledger_module.Q007_SOURCE_CHECKPOINT_SHA256
Q007_FORBIDDEN_CHECKPOINT_SHA256 = ledger_module.Q007_FORBIDDEN_CHECKPOINT_SHA256
Q007_OLD_SOURCE_CLOSURE_SHA256 = ledger_module.Q007_OLD_SOURCE_CLOSURE_SHA256
BRIDGE_CHECKPOINT_SHA256 = '8ca0760e8c7a67da2c2143927e6ba5baf2459deba1967b0756a2aa601e58e805'
BRIDGE_OLD_SOURCE_CLOSURE_SHA256 = '3ff5b4b5711a2774701055244031fe389e0dc495b8280379014126a3fa049f4e'


class LegacySchemaError(ValueError):
    """A schema <= 8 (or unlabelled) checkpoint was offered to the v0.6 reader."""


class ForeignContainerError(ValueError):
    """A recipe/task/mode/container that may not initialize or resume v0.6."""


class RecoveryBlocked(RuntimeError):
    """Resume/validation stopped, with a machine-readable reason."""

    def __init__(self, reason, *, audit=None):
        super().__init__(reason)
        self.reason = reason
        self.audit = audit


def canonical(value):
    return spec.canonical_json(jsonable(value))


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('checkpoint block contains NaN/Inf')
    return value


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def require_hex(value, name):
    if not isinstance(value, str) or len(value) != 64 or set(value) > HEX_DIGITS:
        raise ValueError(f'{name} must be a 64-character hex digest')
    return value


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


# ------------------------------------------------------------------ identity
def identity_block(source=None):
    """Canonical schema-9 identity block built from the frozen spec identity."""
    frozen = dict(spec.identity_block())
    identity = dict(
        task=frozen.get('task', getattr(spec, 'TASK', None)),
        recipe=frozen.get('recipe', frozen.get('recipe_version',
                                               getattr(spec, 'RECIPE_VERSION', None))),
        code_version=frozen.get('code_version', getattr(spec, 'CODE_VERSION', None)),
        protocol=frozen.get('protocol', getattr(spec, 'PROTOCOL_ID', None)),
        architecture=frozen.get('architecture', getattr(spec, 'ARCHITECTURE', None)),
        schema=SCHEMA,
        sds_sha256=(source or {}).get('sds_sha256')
        if isinstance(source, dict) else getattr(spec, 'SDS_SHA256', None),
        log_root=frozen.get('log_root', getattr(spec, 'LOG_ROOT', None)),
        training_seed=frozen.get('training_seed', getattr(spec, 'TRAINING_SEED', None)),
    )
    if identity['sds_sha256'] is None:
        identity['sds_sha256'] = getattr(spec, 'SDS_SHA256', None)
    return identity


def source_block(*, seed, device='cuda:0', mode='train', init_path=None,
                 init_sha256=None, resume=None):
    """SDS/config/source-closure/initialization identity of a run."""
    init_sha = init_sha256 or model_module.INIT_SHA256
    if init_sha != model_module.INIT_SHA256:
        raise ForeignContainerError('the v0.6 initialization source SHA is not the ETH checkpoint')
    hashes = spec.source_hashes() if callable(getattr(spec, 'source_hashes', None)) else {}
    block = dict(
        mode=mode, seed=int(seed), device=str(device),
        init_checkpoint=str(init_path or model_module.INIT_CHECKPOINT),
        init_sha256=init_sha,
        sds_path=spec.SDS_PATH,
        sds_sha256=getattr(spec, 'SDS_SHA256', hashes.get(
            spec.SDS_PATH)),
        recipe_sha256=digest(spec.recipe_config()),
        reward_config_sha256=(spec.reward_config_sha256()
                              if callable(getattr(spec, 'reward_config_sha256', None)) else None),
        source_hashes=hashes,
        source_closure_sha256=digest(hashes) if hashes else None,
        resume=deepcopy(resume),
    )
    if block['sds_sha256'] is not None:
        require_hex(block['sds_sha256'], 'source.sds_sha256')
    return block


def contract_sha(contract):
    return digest(contract)


def build_checkpoint(*, model_state, optimizer_state, learning_rate, normalizer,
                     learning, rng, curriculum, regression_policy_state, evaluation, initialization, gate,
                     retention, ledger, pending_transition=None, source=None,
                     recipe=None, container=None, pending_update=None,
                     pending_boundary=None, incomplete_update=None):
    """Assemble and validate a schema-9 checkpoint (never writes a file)."""
    state = dict(
        schema=SCHEMA,
        identity=identity_block(source),
        contract_sha256=contract_sha(spec.contract()) if callable(
            getattr(spec, 'contract', None)) else None,
        model={key: value.detach().clone() if torch.is_tensor(value) else value
               for key, value in model_state.items()},
        optimizer=deepcopy(optimizer_state),
        normalizer={key: (value.detach().clone() if torch.is_tensor(value) else value)
                    for key, value in normalizer.items()},
        learning=dict(learning),
        rng=deepcopy(rng),
        curriculum=deepcopy(curriculum),
        regression_policy_state=deepcopy(regression_policy_state),
        evaluation=deepcopy(evaluation),
        source=deepcopy(source),
        recipe=deepcopy(recipe) if recipe is not None else spec.recipe_config(),
        resume_mode=RESUME_MODE,
        initialization=deepcopy(initialization),
        gate=deepcopy(gate),
        retention=deepcopy(retention),
        ledger=deepcopy(ledger),
        pending_transition=deepcopy(pending_transition),
    )
    state['learning']['learning_rate'] = float(learning_rate)
    if container is not None:
        state['container'] = str(container)
    state['ledger']['pending_update'] = (None if pending_update in (None, False)
                                         else int(pending_update))
    # The caller's ledger already carries the transaction; only an explicit
    # argument overrides it (never silently clears a pending boundary).
    if pending_boundary is not None:
        state['ledger']['pending_boundary'] = deepcopy(pending_boundary)
    if incomplete_update is not None:
        state['ledger']['incomplete_update'] = deepcopy(incomplete_update)
    validate_checkpoint(state, allow_probe=container == PROBE_CONTAINER)
    return state


def state_sha256(state):
    digest_sha = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest_sha.update(key.encode('utf-8'))
            digest_sha.update(str(value.dtype).encode('utf-8'))
            digest_sha.update(str(tuple(value.shape)).encode('utf-8'))
            digest_sha.update(value.detach().to('cpu').contiguous().numpy().tobytes())
    return digest_sha.hexdigest()


# ------------------------------------------------------------------- RNG I/O
def capture_rng(env=None):
    state = dict(
        python=deepcopy(__import__('random').getstate()),
        numpy=deepcopy(np.random.get_state()),
        torch=torch.get_rng_state().clone(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        env=(env.rng_state() if env is not None and hasattr(env, 'rng_state') else None),
        sampler=(env.sampler_rng_state() if env is not None
                 and hasattr(env, 'sampler_rng_state') else None),
    )
    return state


def validate_rng(state):
    if not isinstance(state, dict) or set(state) != set(RNG_FIELDS):
        raise ValueError(f'checkpoint rng block must hold exactly {list(RNG_FIELDS)}')
    if state['env'] is None or state['sampler'] is None:
        raise ValueError('checkpoint must record both environment and sampler RNG state')
    return True


def restore_rng(state, env=None):
    import random
    validate_rng(state)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(torch.as_tensor(state['torch'], dtype=torch.uint8))
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8) for item in state['cuda']])
    if env is not None:
        if hasattr(env, 'load_rng_state') and state['env'] is not None:
            env.load_rng_state(state['env'])
        if hasattr(env, 'load_sampler_rng_state') and state['sampler'] is not None:
            env.load_sampler_rng_state(state['sampler'])
    return True


# ------------------------------------------------------------ budget auditing
def controls_per_update(num_envs):
    return int(num_envs) * ppo_module.ROLLOUT


def learning_state(update, *, num_envs=ppo_module.NUM_ENVS):
    update = integer(update, 'learning.update')
    samples = integer(num_envs, 'learning.num_envs', 1) * ppo_module.ROLLOUT * update
    return dict(update=update, optimizer_step=update * ppo_module.ADAM_STEPS_PER_UPDATE,
                adam_steps=update * ppo_module.ADAM_STEPS_PER_UPDATE,
                learning_rate=float(ppo_module.PPO_SETTINGS['learning_rate']),
                control_samples=samples,
                physics_substeps=ppo_module.physics_substeps(samples),
                num_envs=int(num_envs), rollout=ppo_module.ROLLOUT,
                epochs=ppo_module.EPOCHS, minibatches=ppo_module.MINIBATCHES)


def validate_learning(state, *, num_envs=None):
    if not isinstance(state, dict):
        raise ValueError('learning block must be a mapping')
    missing = [field for field in LEARNING_FIELDS if field not in state]
    if missing:
        raise ValueError(f'learning block is missing {missing}')
    update = integer(state['update'], 'learning.update')
    adam = integer(state['adam_steps'], 'learning.adam_steps')
    optimizer_step = integer(state['optimizer_step'], 'learning.optimizer_step')
    samples = integer(state['control_samples'], 'learning.control_samples')
    physics = integer(state['physics_substeps'], 'learning.physics_substeps')
    if num_envs is not None and int(state['num_envs']) != int(num_envs):
        raise ValueError('learning.num_envs differs from the run configuration')
    if adam != update * ppo_module.ADAM_STEPS_PER_UPDATE:
        raise ValueError('learning.adam_steps is not update * epochs * minibatches')
    if optimizer_step != adam:
        raise ValueError('learning.optimizer_step must equal the Adam step count')
    if samples != update * int(state['num_envs']) * ppo_module.ROLLOUT:
        raise ValueError('learning.control_samples is not update * num_envs * rollout')
    if physics != ppo_module.physics_substeps(samples):
        raise ValueError('learning.physics_substeps is not 4 * control_samples')
    learning_rate = float(state.get('learning_rate', math.nan))
    if not (ppo_module.PPO_SETTINGS['min_learning_rate'] - 1e-12 <= learning_rate
            <= ppo_module.PPO_SETTINGS['max_learning_rate'] + 1e-12):
        raise ValueError('checkpoint learning rate is outside the frozen SDS range')
    return True


def validate_optimizer_state(state, *, expected_parameters=None):
    """Structural single-Adam validation (betas/eps/weight decay/lr/membership)."""
    if not isinstance(state, dict) or 'state' not in state or 'param_groups' not in state:
        raise ValueError('optimizer block is not a torch optimizer state dict')
    groups = state['param_groups']
    if len(groups) != 1:
        raise ValueError('v0.6 has exactly one Adam parameter group')
    group = groups[0]
    settings = ppo_module.PPO_SETTINGS
    learning_rate = float(group.get('lr', math.nan))
    if not (settings['min_learning_rate'] - 1e-12 <= learning_rate
            <= settings['max_learning_rate'] + 1e-12):
        raise ValueError('optimizer learning rate is outside [1e-5, 3e-4]')
    betas = tuple(float(value) for value in group.get('betas', ()))
    if betas != tuple(float(value) for value in settings['adam_betas']):
        raise ValueError('optimizer betas are not (0.9, 0.999)')
    if float(group.get('weight_decay', -1)) != float(settings['weight_decay']):
        raise ValueError('optimizer weight_decay is not 0')
    if not math.isclose(float(group.get('eps', math.nan)),
                        float(settings['adam_eps']), rel_tol=0, abs_tol=0):
        raise ValueError('optimizer eps is not 1e-8')
    membership = group.get('params')
    if not isinstance(membership, (list, tuple)) or not membership:
        raise ValueError('optimizer parameter membership is empty')
    if len(set(int(index) for index in membership)) != len(membership):
        raise ValueError('an optimizer parameter appears more than once')
    if expected_parameters is not None:
        expected = expected_parameters if isinstance(expected_parameters, int) \
            else len(expected_parameters)
        if len(membership) != expected:
            raise ValueError('optimizer membership does not match the trainable parameters')
        if set(state['state']) and set(int(key) for key in state['state']) - set(
                int(index) for index in membership):
            raise ValueError('optimizer state holds parameters outside its group')
    return True


def validate_curriculum(state, *, learning=None, pending_update=None):
    """SDS 7.2 curriculum state: every field present, budgets internally consistent.

    A checkpoint written on an exception path may legitimately hold the completed
    update's ``learning.update`` one ahead of the curriculum spend (the update was
    billed before the curriculum tick could commit it).  Only that explicit
    ``pending_update`` marker authorizes the difference; every normal checkpoint
    still requires exact equality.
    """
    if not isinstance(state, dict):
        raise ValueError('curriculum block must be a mapping')
    missing = [field for field in CURRICULUM_FIELDS if field not in state]
    if missing:
        raise ValueError(f'curriculum block is missing {missing}')
    phase = str(state['phase']).upper()
    if phase not in speed.phase_order():
        raise ValueError(f'curriculum phase {phase!r} is not a v0.6 phase')
    spent = {str(key).upper(): integer(int(value), f'spent_by_phase[{key}]')
             for key, value in dict(state['spent_by_phase']).items()}
    stages = speed.phase_order()
    if set(spent) - set(stages):
        raise ValueError('curriculum spent_by_phase holds unknown phases')
    for name in stages:
        if spent.get(name, 0) > speed.phase_budget(name):
            raise ValueError(f'curriculum spent_by_phase[{name}] exceeds its budget')
    if sum(spent.values()) > speed.total_budget():
        raise ValueError('curriculum total spend exceeds the 30000 budget')
    substage_spent = {str(key): integer(int(value), f'spent_by_substage[{key}]')
                      for key, value in dict(state['spent_by_substage']).items()}
    for name, value in substage_spent.items():
        budget = _substage_budget(name)
        if budget is not None and value > budget:
            raise ValueError(f'curriculum spent_by_substage[{name}] exceeds its budget')
    if learning is not None:
        update = integer(int(dict(learning)['update']), 'learning.update')
        expected = update
        if pending_update is not None:
            marker = integer(int(pending_update), 'pending_update')
            if marker != update:
                raise ValueError('pending_update must equal learning.update')
            if sum(spent.values()) != update - 1:
                raise ValueError(
                    'a pending_update checkpoint must have exactly one uncommitted '
                    'update')
            expected = update - 1
        if sum(spent.values()) != expected:
            raise ValueError('curriculum spend and learning.update disagree')
    return True


def _substage_budget(name):
    try:
        entry = spec.phase_spec(str(name)[0].upper())
    except Exception:
        return None
    budgets = dict(zip(tuple(entry.substages), tuple(entry.substage_budgets)))
    return budgets.get(name)


def audit_budget(learning, curriculum, *, pending_update=None):
    """Four-way budget consistency audit (SDS 8.1 / 8.2)."""
    validate_learning(learning)
    validate_curriculum(curriculum, learning=learning, pending_update=pending_update)
    spent = {str(key).upper(): int(value)
             for key, value in dict(curriculum['spent_by_phase']).items()}
    update = int(learning['update'])
    total_budget = speed.total_budget()
    checks = dict(
        update_equals_phase_spend=(sum(spent.values())
                                   == (update - 1 if pending_update is not None else update)),
        phase_spend_within_budgets=all(spent.get(name, 0) <= speed.phase_budget(name)
                                       for name in speed.phase_order()),
        total_within_budget=update <= total_budget and sum(spent.values()) <= total_budget,
        ledger_arithmetic=(int(learning['control_samples']) == update * int(learning['num_envs'])
                           * ppo_module.ROLLOUT
                           and int(learning['physics_substeps'])
                           == ppo_module.physics_substeps(int(learning['control_samples']))
                           and int(learning['adam_steps']) == update * ppo_module.ADAM_STEPS_PER_UPDATE),
    )
    return dict(update=update, spent_by_phase=spent, total_budget=total_budget,
                remaining=total_budget - update, checks=checks,
                consistent=all(checks.values()))


# --------------------------------------------------------------- validation
def classify(state):
    """Return ``(kind, reason)`` for the container, never a silent acceptance."""
    if not isinstance(state, dict):
        return 'invalid', 'checkpoint is not a mapping'
    container = state.get('container') or state.get('kind')
    if container in FOREIGN_CONTAINERS:
        return 'smoke_or_probe', f'container {container!r} is engineering-test evidence only'
    schema = state.get('schema')
    if schema != SCHEMA:
        return 'legacy', f'schema {schema!r} is not the v0.6 schema {SCHEMA}'
    identity = state.get('identity')
    if not isinstance(identity, dict):
        return 'invalid', 'checkpoint has no identity block'
    if state.get('resume_mode') != RESUME_MODE:
        return 'invalid', f'resume_mode must be {RESUME_MODE!r}'
    return 'v07', None


def reject_foreign_container(state, *, allow_probe=False):
    kind, reason = classify(state)
    if kind == 'smoke_or_probe':
        container = state.get('container') or state.get('kind')
        if allow_probe and container == PROBE_CONTAINER:
            pass
        else:
            raise ForeignContainerError(reason)
    elif kind == 'legacy':
        raise LegacySchemaError(reason)
    elif kind != 'v07':
        raise ForeignContainerError(reason)
    identity = state['identity']
    task = identity.get('task', state.get('task'))
    recipe = identity.get('recipe', state.get('recipe'))
    if task not in ALLOWED_TASKS:
        raise ForeignContainerError(f'task {task!r} is not a v0.6 training source')
    if recipe not in ALLOWED_RECIPES:
        raise ForeignContainerError(f'recipe {recipe!r} is not the v0.6 recipe')
    if identity.get('architecture') not in (None, spec.ARCHITECTURE):
        raise ForeignContainerError('architecture is not v06-route-state-v1')
    source = state.get('source') or {}
    mode = source.get('mode')
    if mode in FOREIGN_MODES:
        raise ForeignContainerError(f'mode {mode!r} evidence may not be resumed')
    source_sha = source.get('init_sha256')
    if source_sha != model_module.INIT_SHA256:
        raise ForeignContainerError('checkpoint initialization source is not the ETH model')
    return True


def _validate_ledger(ledger):
    """Q002-4: the ledger must use the frozen state source and prove ``complete``."""
    if not isinstance(ledger, dict):
        raise ValueError('checkpoint ledger must be a mapping')
    allowed = tuple(getattr(spec, 'ALL_TRAINING_STATES', ()))
    stop = ledger.get('stop_state')
    lifecycle = ledger.get('lifecycle_state', 'running')
    if allowed:
        if stop is not None and stop not in allowed:
            raise ForeignContainerError(f'ledger stop_state {stop!r} is not a v0.6 state')
        if lifecycle not in allowed:
            raise ForeignContainerError(
                f'ledger lifecycle_state {lifecycle!r} is not a v0.6 state')
        if lifecycle == 'complete' and stop is not None:
            raise ValueError('a complete run must not carry a training stop state')
    terminal = ledger.get('terminal')
    boundary = ledger.get('pending_boundary')
    if boundary is not None:
        if not isinstance(boundary, dict):
            raise ValueError('ledger.pending_boundary must be a mapping')
        for field in ('update', 'phase', 'substage', 'generation', 'tasks',
                      'results', 'curriculum_decided', 'committed'):
            if field not in boundary:
                raise ValueError(f'pending_boundary is missing {field!r}')
        if not isinstance(boundary['tasks'], list) or not isinstance(
                boundary['results'], list):
            raise ValueError('pending_boundary tasks/results must be lists')
        integer(int(boundary['update']), 'pending_boundary.update', 1)
    incomplete = ledger.get('incomplete_update')
    if incomplete is not None:
        if not isinstance(incomplete, dict):
            raise ValueError('ledger.incomplete_update must be a mapping')
        for field in ('update', 'error_type', 'resumable'):
            if field not in incomplete:
                raise ValueError(f'incomplete_update is missing {field!r}')
        if incomplete['resumable'] is not False:
            raise ValueError('an incomplete PPO interval must be marked non-resumable')
    pending = ledger.get('pending_update')
    if pending is not None:
        integer(int(pending), 'ledger.pending_update', 1)
        if int(pending) != int(ledger.get('learning_update', pending)):
            pass  # cross-checked against learning.update by validate_curriculum
    if terminal is not None:
        if not isinstance(terminal, dict):
            raise ValueError('ledger terminal evidence must be a mapping')
        for field in ('state', 'lifecycle_state', 'stop_state', 'reason',
                      'global_update', 'phase', 'substage', 'generation', 'budget',
                      'evidence'):
            if field not in terminal:
                raise ValueError(f'terminal evidence is missing {field!r}')
        if not isinstance(terminal['budget'], dict):
            raise ValueError('terminal evidence budget must be a mapping')
        for field in ('update', 'total_budget', 'total_exhausted', 'phase',
                      'phase_budget', 'phase_spent', 'phase_exhausted'):
            if field not in terminal['budget']:
                raise ValueError(f'terminal budget evidence is missing {field!r}')
    if lifecycle == 'complete':
        final = terminal.get('final_test') if isinstance(terminal, dict) else None
        if not isinstance(final, dict) or final.get('passed') is not True:
            raise ValueError(
                'lifecycle complete requires the passed final 300-case test evidence')
        if final.get('spawn_profile') != 'full_R_expanded':
            raise ValueError('the complete evidence must use the full_R_expanded test')
        if int(final.get('cases_per_speed', 0)) != int(speed.FINAL_CASES):
            raise ValueError('the complete evidence must hold the final case count')
    if stop == 'gate_failed' and terminal is not None:
        if terminal.get('reason') in (None, ''):
            raise ValueError('gate_failed requires a stop_reason')
    return True


def validate_checkpoint(state, *, expected_contract_sha256=None, num_envs=None,
                        require_model=True, allow_probe=False):
    """Full schema-9 validation; raises on the first violated SDS 8.2 contract."""
    reject_foreign_container(state, allow_probe=allow_probe)
    # ``checkpoint_sha256`` is the runtime annotation added by preload_checkpoint;
    # it is never part of the serialized schema-9 envelope.
    present = set(state) - {'checkpoint_sha256', *OPTIONAL_TOP_LEVEL_KEYS}
    if present != set(TOP_LEVEL_KEYS):
        missing = sorted(set(TOP_LEVEL_KEYS) - present)
        extra = sorted(present - set(TOP_LEVEL_KEYS))
        raise ValueError(f'schema-9 key contract mismatch: missing={missing} extra={extra}')
    identity = state['identity']
    for field in IDENTITY_FIELDS:
        if field not in identity:
            raise ValueError(f'identity block is missing {field!r}')
    task = getattr(spec, 'TASK_ID', getattr(spec, 'TASK', None))
    if identity['task'] != task or identity['recipe'] != spec.RECIPE_VERSION:
        raise ForeignContainerError('identity task/recipe is not the v0.6 contract')
    if identity['architecture'] != spec.ARCHITECTURE:
        raise ForeignContainerError('identity architecture is not v06-route-state-v1')
    if expected_contract_sha256 is not None \
            and state['contract_sha256'] != expected_contract_sha256:
        raise ValueError('checkpoint contract SHA does not match the expected contract')
    validate_optimizer_state(state['optimizer'])
    validate_learning(state['learning'], num_envs=num_envs)
    if abs(float(state['learning']['learning_rate'])
           - float(state['optimizer']['param_groups'][0]['lr'])) > 1e-12:
        raise ValueError('learning rate differs between learning and optimizer blocks')
    pending_update = (state['ledger'].get('pending_update')
                      if isinstance(state.get('ledger'), dict) else None)
    validate_curriculum(state['curriculum'], learning=state['learning'],
                        pending_update=pending_update)
    regression_policy.validate_state(state['regression_policy_state'],generation=state['curriculum']['generation'])
    validate_rng(state['rng'])
    if not isinstance(state['normalizer'], dict) or not state['normalizer']:
        raise ValueError('checkpoint has no normalization state')
    if state['resume_mode'] != RESUME_MODE:
        raise ValueError('checkpoint resume_mode must be new_episode')
    if require_model:
        if not isinstance(state['model'], dict) or not state['model']:
            raise ValueError('checkpoint has no model state')
        for key in model_module.BYPASS_KEYS:
            if key not in state['model']:
                raise ValueError(f'model state is missing the bypass tensor {key!r}')
    if not isinstance(state['evaluation'], dict) or 'cases' not in state['evaluation']:
        raise ValueError('checkpoint must record the evaluation case tables')
    if not isinstance(state['gate'], dict) or 'identity' not in state['gate']:
        raise ValueError('checkpoint must record the completed gate identity')
    _validate_ledger(state['ledger'])
    if not isinstance(state['initialization'], dict):
        raise ValueError('checkpoint must record the initialization block')
    model_module.validate_initialization(state['initialization'])
    if state['pending_transition'] is not None and not isinstance(
            state['pending_transition'], dict):
        raise ValueError('checkpoint pending_transition must be a mapping or null')
    audit = audit_budget(state['learning'], state['curriculum'],
                         pending_update=pending_update)
    if not audit['consistent']:
        raise ValueError(f'checkpoint budget audit is inconsistent: {audit["checks"]}')
    return True


def validate_model_state(state, model):
    """Load-ready model state: same keys/shapes and a zero-initialized bypass."""
    target = model.state_dict()
    if set(state) != set(target):
        missing = sorted(set(target) - set(state))
        extra = sorted(set(state) - set(target))
        raise ValueError(f'model state keys differ: missing={missing} extra={extra}')
    for key, value in state.items():
        if tuple(value.shape) != tuple(target[key].shape):
            raise ValueError(f'model state shape differs: {key}')
        if value.dtype != target[key].dtype:
            raise ValueError(f'model state dtype differs: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'model state is not finite: {key}')
    return True


def load_model_state(model, state):
    validate_model_state(state, model)
    model.load_state_dict({key: value for key, value in state.items()}, strict=True)
    return model


# --------------------------------------------------------------------- files
def atomic_save(path, state):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'checkpoint path already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)
    return sha256(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preload_checkpoint(path, expected_sha=None, *, mode='train', device='cpu',
                       allow_bridge=False):
    """Read, verify and fully validate a checkpoint.

    ``mode='resume'`` is the formal training resume and refuses every engineering
    container; ``mode='probe'`` is the probe's own recovery verification, which is
    allowed to read the ``v07-probe-state-v1`` checkpoint it just wrote.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if expected_sha is not None and actual != expected_sha:
        raise RecoveryBlocked('checkpoint SHA256 mismatch',
                              audit=dict(expected=expected_sha, actual=actual))
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise LegacySchemaError('checkpoint is not a mapping')
    state['checkpoint_sha256'] = actual
    try:
        validate_checkpoint(state, allow_probe=(mode == 'probe'))
        if mode == 'resume':
            resume_contract(state, allow_bridge=allow_bridge)
        elif mode == 'probe':
            resume_contract(state, allow_probe=True)
            if state.get('container') != PROBE_CONTAINER:
                raise ForeignContainerError(
                    'the probe must resume from its own v07-probe-state-v1 checkpoint')
    except (LegacySchemaError, ForeignContainerError) as error:
        raise RecoveryBlocked(str(error), audit=dict(path=str(path), sha256=actual))
    return state


def pending_task_migration_plan(state):
    """Validate the Q007 source and build the one-task flat-retention migration.

    Only ``exception_000400.pt`` with its exact completed validation/diagnostic
    evidence and its one unfinished flat-retention task may be migrated.  The
    replacement table is generated by the frozen code (same seed/speeds/spawn)
    and its full content and SHA are recorded in the plan before execution.
    """
    if not isinstance(state, dict):
        raise RecoveryBlocked('the Q007 migration needs a checkpoint mapping')
    if state.get('checkpoint_sha256') != ledger_module.Q007_SOURCE_CHECKPOINT_SHA256:
        raise RecoveryBlocked(
            'the Q007 pending-task migration only accepts the designated '
            'exception_000400.pt source (boundary_000400 is a pre-evaluation '
            'control and is never migrated)')
    source = state.get('source') or {}
    if source.get('source_closure_sha256') != \
            ledger_module.Q007_OLD_SOURCE_CLOSURE_SHA256:
        raise RecoveryBlocked(
            'the Q007 source closure is not the documented Q006 segment closure')
    ledger = state.get('ledger') or {}
    for field in ('stop_state', 'pending_update', 'incomplete_update'):
        if ledger.get(field) is not None:
            raise RecoveryBlocked(f'the Q007 source has {field} set; it is not a '
                                  'resumable running transaction')
    if ledger.get('lifecycle_state') != 'running':
        raise RecoveryBlocked('the Q007 source lifecycle is not running')
    pending = ledger.get('pending_boundary')
    if not isinstance(pending, dict):
        raise RecoveryBlocked('the Q007 source has no pending boundary')
    identity = (int(pending.get('update', -1)),
                str(pending.get('phase')).upper(),
                pending.get('substage'),
                int(pending.get('generation', -1)))
    if identity != (400, 'H', None, 1):
        raise RecoveryBlocked(
            'the Q007 source pending boundary is not update400/phaseH/substageNone/'
            'generation1')
    if pending.get('committed') is not True or \
            pending.get('curriculum_decided') is not False:
        raise RecoveryBlocked('the Q007 source pending boundary is not an un-decided '
                              'committed transaction')
    tasks = pending.get('tasks')
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise RecoveryBlocked('the Q007 source pending boundary must hold exactly the '
                              'three recorded tasks')
    validation, diagnostic, retention = tasks
    for task, kind, sha in (
            (validation, 'validation', ledger_module.Q007_VALIDATION_RESULT_SHA256),
            (diagnostic, 'diagnostic', ledger_module.Q007_DIAGNOSTIC_RESULT_SHA256)):
        if str(task.get('kind')) != kind or str(task.get('gate_kind')) != kind \
                or task.get('retention_kind') is not None \
                or int(task.get('tile', -1)) != 1 \
                or str(task.get('case_table_sha256')) != sha:
            raise RecoveryBlocked(
                f'the Q007 source {kind} task identity does not match its completed '
                'record')
    if list(pending.get('results') or []) != list(
            ledger_module.Q007_COMPLETED_RESULT_IDS):
        raise RecoveryBlocked('the Q007 source pending results are not exactly the '
                              'completed 400 validation and diagnostic records')
    if str(retention.get('kind')) != 'retention' \
            or str(retention.get('gate_kind')) != 'flat_retention' \
            or str(retention.get('retention_kind')) != 'flat' \
            or int(retention.get('tile', -1)) != \
            ledger_module.Q007_OLD_FLAT_TASK_TILE \
            or str(retention.get('case_table_sha256')) != \
            ledger_module.Q007_OLD_FLAT_TASK_SHA256:
        raise RecoveryBlocked('the Q007 source flat-retention task is not the '
                              'documented unfinished tile1/old-table task')
    records = (state.get('gate') or {}).get('records') or []
    # The completed 400 validation *gate record* carries the round-qualified table
    # SHA (4a6205...), while the pending task identity is the round-less frozen
    # 0ccaa0...; both are preserved as their own evidence.
    gate_record = next((record for record in records
                        if str(record.get('kind')) == 'validation'
                        and int(record.get('update', -1)) == 400
                        and record.get('passed') is True), None)
    if gate_record is None:
        raise RecoveryBlocked('the Q007 source does not inherit the 400 validation '
                              'gate record')
    diagnostics = (state.get('ledger') or {}).get('diagnostics') or []
    if not any(int(record.get('update', -1)) == 400
               for record in diagnostics):
        raise RecoveryBlocked('the Q007 source does not inherit the 400 diagnostic '
                              'record')
    seed = int(source.get('seed', 0))
    if seed < 1:
        raise RecoveryBlocked('the Q007 source has no valid training seed')
    table = evaluation_module.make_cases(
        'H', None, 'flat_retention', seed, tile_id=int(spec.FLAT_TILE_ID))
    if (int(table.target_tile_id) != int(spec.FLAT_TILE_ID)
            or table.measurement_stage != 'F0'
            or tuple(table.speeds) != (0.3, 0.5, 0.8)
            or int(table.episodes) != 5 or len(table.cases) != 15
            or table.spawn_profile != 'standard'
            or any(int(case['tile_id']) != int(spec.FLAT_TILE_ID)
                   for case in table.cases)):
        raise RecoveryBlocked('the frozen Q007 replacement table is not the fixed '
                              'flat-retention protocol')
    new_task = deepcopy(retention)
    new_task['tile'] = int(spec.FLAT_TILE_ID)
    new_task['case_table_sha256'] = table.sha256
    plan = dict(
        schema='a2-v07-q007-task-migration-v1',
        q006_decision_sha256=ledger_module.Q006_DECISION_SHA256,
        q007_decision_sha256=ledger_module.Q007_DECISION_SHA256,
        q007_errata_sha256=ledger_module.Q007_ERRATA_SHA256,
        source=dict(path=ledger_module.Q007_SOURCE_CHECKPOINT_PATH,
                    sha256=state.get('checkpoint_sha256'),
                    source_closure_sha256=source.get('source_closure_sha256')),
        pending=dict(update=400, phase='H', substage=None, generation=1,
                     committed=True, curriculum_decided=False),
        completed_results=[str(item) for item in pending.get('results') or []],
        inherited=dict(
            validation=dict(task=deepcopy(validation),
                            case_table_sha256=ledger_module
                            .Q007_VALIDATION_RESULT_SHA256, calls=0,
                            gate_record=deepcopy(gate_record),
                            gate_record_case_table_sha256=str(
                                gate_record.get('case_table_sha256'))),
            diagnostic=dict(task=deepcopy(diagnostic),
                            case_table_sha256=ledger_module
                            .Q007_DIAGNOSTIC_RESULT_SHA256, calls=0,
                            diagnostic_record=deepcopy(next(
                                record for record in diagnostics
                                if int(record.get('update', -1)) == 400)))),
        old_task=deepcopy(retention),
        old_task_protocol_invalid=True,
        old_task_protocol_invalid_note=(
            'the pre-fix flat task measured H00 (tile 1) with the F0 500-step '
            'protocol and an invalid table identity; its old SHA is preserved as '
            'protocol_invalid evidence and is never relabelled as the new table'),
        new_task=new_task,
        new_case_table=table.to_json(),
        new_case_table_sha256=table.sha256,
        reason='Q007 section 4: replace only the one unfinished flat-retention task '
               'with the frozen flat-tile protocol; the completed validation and '
               'diagnostic records and their gate/streak evidence are inherited',
        segment=ledger_module.q007_segment_metadata(),
    )
    plan['plan_sha256'] = digest(plan)
    return plan


def apply_pending_task_migration(state, plan=None):
    """Apply the validated plan to the in-memory pending boundary (never the file)."""
    if not isinstance(state, dict):
        raise RecoveryBlocked('the Q007 migration needs a checkpoint mapping')
    plan = pending_task_migration_plan(state) if plan is None else plan
    if not isinstance(plan, dict) or plan.get('q007_decision_sha256') != \
            ledger_module.Q007_DECISION_SHA256:
        raise RecoveryBlocked('the Q007 migration plan is not bound to the decision')
    pending = (state.get('ledger') or {}).get('pending_boundary')
    if not isinstance(pending, dict) or not isinstance(pending.get('tasks'), list):
        raise RecoveryBlocked('the Q007 source pending boundary disappeared')
    tasks = pending['tasks']
    if len(tasks) != 3:
        raise RecoveryBlocked('the Q007 source pending task count changed')
    new_task = deepcopy(plan['new_task'])
    if int(new_task.get('tile', -1)) != int(spec.FLAT_TILE_ID):
        raise RecoveryBlocked('the Q007 migration plan does not target the flat tile')
    table = evaluation_module.make_cases(
        'H', None, 'flat_retention', int((state.get('source') or {}).get('seed', 0)),
        tile_id=int(spec.FLAT_TILE_ID))
    if new_task.get('case_table_sha256') != table.sha256:
        raise RecoveryBlocked('the Q007 migration table is not the frozen flat table')
    tasks[2] = new_task
    pending['task_migration'] = dict(
        schema=plan['schema'], plan_sha256=plan['plan_sha256'],
        q007_decision_sha256=plan['q007_decision_sha256'],
        q007_errata_sha256=plan['q007_errata_sha256'],
        old_task_sha256=plan['old_task']['case_table_sha256'],
        old_tile=int(plan['old_task']['tile']),
        old_task_protocol_invalid=True,
        new_task_sha256=plan['new_case_table_sha256'],
        new_tile=int(spec.FLAT_TILE_ID),
        completed_results=list(plan['completed_results']),
        inherited_evaluation_calls=0)
    pending['committed'] = True
    state['ledger']['pending_boundary'] = pending
    return plan


def resume_contract(state, *, init_sha256=None, recipe=None, seed=None, device=None,
                    num_envs=None, allow_probe=False, allow_bridge=False):
    """The ``--resume-run`` contract: never re-initializes from ETH, never re-bills.

    Returns the restored learning/curriculum pointers the runner must adopt and
    records that a new episode starts at the saved complete-update boundary.
    """
    validate_checkpoint(state, num_envs=num_envs, allow_probe=allow_probe)
    ledger = state['ledger']
    # Qv07-002: an observed valid request survives an evidence-IO exception.
    # Evidence remains readable/saveable, but a new helper/current continue must
    # never release this unresolved latch or authorize new environments/learning.
    terminal=ledger.get('terminal') or {}
    control_error=(terminal.get('evidence') or {}).get('control_error')
    latched=control_error.get('latched_request') if isinstance(control_error,dict) else None
    if isinstance(latched,dict) and latched.get('action') in ('pause','stop'):
        raise RecoveryBlocked('unresolved observed operator '+latched['action']+' requires adjudication; later continue cannot authorize learning',audit=deepcopy(control_error))
    transaction = ledger.get('evaluation_transaction')
    if transaction is not None and (not isinstance(transaction,dict)
            or transaction.get('stage') not in ('evaluated','applied')):
        raise RecoveryBlocked('Q010 evaluation application_incomplete is evidence only and requires adjudication')
    curriculum_transaction=ledger.get('curriculum_transaction')
    if curriculum_transaction is not None and (not isinstance(curriculum_transaction,dict) or curriculum_transaction.get('schema')!='v07-current-curriculum-v1' or curriculum_transaction.get('stage') not in ('prepared','applied')):
        raise RecoveryBlocked('unproven curriculum application is evidence only and requires adjudication')
    if ledger.get('stop_state') is not None:
        raise RecoveryBlocked(
            f'checkpoint belongs to a terminated run ({ledger["stop_state"]}); a stopped '
            'run is never resumed or reclassified')
    if ledger.get('lifecycle_state') == 'complete':
        raise RecoveryBlocked('the run already completed the frozen final test')
    if ledger.get('pending_update') is not None:
        raise RecoveryBlocked(
            'checkpoint was written on an exception path with one uncommitted update; '
            'it is evidence only and may not resume training')
    if ledger.get('incomplete_update') is not None:
        raise RecoveryBlocked(
            'checkpoint was written during a partial PPO update; the billed interval is '
            'not resumable and the interval must be adjudicated')
    if ledger.get('lifecycle_state') == 'complete' and ledger.get('stop_state') is not None:
        raise RecoveryBlocked('a completed run cannot be resumed')
    source = state['source']
    bridge = None
    actual_sha = state.get('checkpoint_sha256')
    if actual_sha == ledger_module.Q007_FORBIDDEN_CHECKPOINT_SHA256:
        raise RecoveryBlocked(
            'boundary_000400.pt is the pre-evaluation control container (its pending '
            'results are empty and it holds no 400 gate record); it is never a resume '
            'source and must not be mixed with exception_000400.pt')
    if not allow_bridge:
        current = source_block(seed=source.get('seed'), device=source.get('device'),
                               mode='train')['source_closure_sha256']
        if source.get('source_closure_sha256') != current:
            raise RecoveryBlocked(
                'checkpoint source closure differs from the running source; only a '
                'restricted bridge (Q005/Q006/Q007) may cross this boundary')
    if (not allow_bridge and actual_sha == BRIDGE_CHECKPOINT_SHA256):
        raise RecoveryBlocked(
            'the documented trial02 checkpoint requires the restricted Q005/Q006 '
            'recovery bridge (allow_bridge=True)')
    if (not allow_bridge
            and actual_sha == ledger_module.Q007_SOURCE_CHECKPOINT_SHA256):
        raise RecoveryBlocked(
            'exception_000400.pt requires the restricted Q007 task-migration bridge '
            '(--allow-q007-bridge / allow_bridge=True)')
    if allow_bridge:
        actual = actual_sha
        if actual == BRIDGE_CHECKPOINT_SHA256:
            if source.get('source_closure_sha256') != \
                    BRIDGE_OLD_SOURCE_CLOSURE_SHA256:
                raise RecoveryBlocked(
                    'bridge source closure does not match the documented one')
            new_block = source_block(seed=source.get('seed'),
                                     device=source.get('device'), mode='train')
            segment = ledger_module.segment_metadata()
            bridge = dict(q005_sha256=Q005_DECISION_SHA256,
                          q006_sha256=Q006_DECISION_SHA256,
                          q006_errata_sha256=Q006_ERRATA_SHA256,
                          checkpoint_sha256=actual,
                          old_source_closure_sha256=BRIDGE_OLD_SOURCE_CLOSURE_SHA256,
                          new_source_closure_sha256=new_block.get(
                              'source_closure_sha256'),
                          new_source_hashes=deepcopy(
                              new_block.get('source_hashes') or {}),
                          manifest_path=ledger_module.Q006_BRIDGE_MANIFEST_PATH,
                          independent_acceptance_sha256=(
                              ledger_module.Q006_ACCEPTANCE_SHA256),
                          closure_objects=dict(
                              checkpoint_source_closure='sha256 of canonical '
                              'spec.source_hashes() map stored in the checkpoint',
                              freeze_report_closure='freeze manifest/binding digest; '
                              'NOT the checkpoint closure',
                              new_source_closure='sha256 of canonical '
                              'spec.source_hashes() map of the repaired source'),
                          segment=segment,
                          rollback_from_update=int(segment['rollback_from_update']),
                          resume_from_update=int(segment['resume_from_update']),
                          lost_updates=int(segment['lost_updates']),
                          replay_billed_updates=True,
                          replay_pending_evaluations=False,
                          q006_segment=True,
                          restricted=True)
        elif actual == ledger_module.Q007_SOURCE_CHECKPOINT_SHA256:
            plan = pending_task_migration_plan(state)
            new_block = source_block(seed=source.get('seed'),
                                     device=source.get('device'), mode='train')
            segment = ledger_module.q007_segment_metadata()
            bridge = dict(q005_sha256=Q005_DECISION_SHA256,
                          q006_sha256=Q006_DECISION_SHA256,
                          q006_errata_sha256=Q006_ERRATA_SHA256,
                          q007_sha256=Q007_DECISION_SHA256,
                          q007_errata_sha256=Q007_ERRATA_SHA256,
                          checkpoint_sha256=actual,
                          source_path=ledger_module.Q007_SOURCE_CHECKPOINT_PATH,
                          old_source_closure_sha256=(
                              ledger_module.Q007_OLD_SOURCE_CLOSURE_SHA256),
                          new_source_closure_sha256=new_block.get(
                              'source_closure_sha256'),
                          new_source_hashes=deepcopy(
                              new_block.get('source_hashes') or {}),
                          manifest_path=ledger_module.Q007_BRIDGE_MANIFEST_PATH,
                          independent_acceptance_sha256=(
                              ledger_module.Q007_ACCEPTANCE_SHA256),
                          closure_objects=dict(
                              checkpoint_source_closure='sha256 of canonical '
                              'spec.source_hashes() map stored in the checkpoint',
                              new_source_closure='sha256 of canonical '
                              'spec.source_hashes() map of the repaired source'),
                          segment=segment,
                          rollback_from_update=int(segment['rollback_from_update']),
                          resume_from_update=int(segment['resume_from_update']),
                          lost_updates=int(segment['lost_updates']),
                          replay_billed_updates=False,
                          replay_pending_evaluations=True,
                          task_migration=plan,
                          q007_segment=True,
                          restricted=True)
        else:
            raise RecoveryBlocked(
                'the restricted recovery bridge only accepts the documented Q006 '
                'boundary and Q007 exception sources; checkpoints saved by the '
                'repaired source resume through the normal source-closure check '
                'without this flag')
    if init_sha256 is not None and source.get('init_sha256') != init_sha256:
        raise RecoveryBlocked('resume initialization SHA differs from the requested contract')
    if recipe is not None and state['identity']['recipe'] != recipe:
        raise RecoveryBlocked('resume recipe differs from the requested contract')
    if seed is not None and int(source.get('seed', -1)) != int(seed):
        raise RecoveryBlocked('resume seed differs from the requested contract')
    if device is not None and str(source.get('device')) != str(device):
        raise RecoveryBlocked('resume device differs from the requested contract')
    learning = dict(state['learning'])
    boundary = ledger.get('pending_boundary')
    return dict(resume_mode=RESUME_MODE, checkpoint_sha256=state.get('checkpoint_sha256'),
                update=int(learning['update']),
                pending_boundary=(dict(boundary) if boundary is not None else None),
                pending_boundary_requires_completion=bool(boundary is not None),
                bridge=bridge,
                optimizer_step=int(learning['optimizer_step']),
                learning_rate=float(learning['learning_rate']),
                phase=str(state['curriculum']['phase']).upper(),
                substage=state['curriculum']['substage'],
                new_episode=True,
                replay_billed_updates=bool(bridge and bridge.get('replay_billed_updates')),
                replay_pending_evaluations=bool(
                    bridge and bridge.get('replay_pending_evaluations')),
                pending_task_migration=(deepcopy(bridge.get('task_migration'))
                                        if bridge and bridge.get('task_migration')
                                        else None),
                rollback_from_update=(int(bridge['rollback_from_update'])
                                      if bridge else None),
                resume_from_update=(int(bridge['resume_from_update']) if bridge else None),
                lost_updates=(int(bridge['lost_updates']) if bridge else None),
                segment=(dict(bridge['segment']) if bridge else None),
                physical_trajectory_continuous=False)


def checkpoint_identity(path):
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    kind, reason = classify(state)
    return dict(path=str(path), sha256=sha256(path), classification=kind,
                reason=reason, schema=state.get('schema') if isinstance(state, dict) else None,
                recipe=(state.get('identity', {}) or {}).get('recipe')
                if isinstance(state, dict) else None)


__all__ = [
    'ALLOWED_RECIPES', 'ALLOWED_TASKS', 'CURRICULUM_FIELDS', 'FOREIGN_CONTAINERS',
    'FOREIGN_MODES', 'ForeignContainerError', 'IDENTITY_FIELDS', 'LegacySchemaError',
    'LEARNING_FIELDS', 'RESUME_MODE', 'RNG_FIELDS', 'RecoveryBlocked', 'SCHEMA',
    'OPTIONAL_TOP_LEVEL_KEYS', 'PROBE_CONTAINER', 'Q006_DECISION_SHA256',
    'Q006_ERRATA_SHA256', 'Q007_DECISION_SHA256', 'Q007_ERRATA_SHA256',
    'Q007_FORBIDDEN_CHECKPOINT_SHA256', 'Q007_OLD_SOURCE_CLOSURE_SHA256',
    'Q007_SOURCE_CHECKPOINT_SHA256', 'TOP_LEVEL_KEYS', '_validate_ledger',
    'apply_pending_task_migration', 'pending_task_migration_plan',
    'atomic_save', 'audit_budget',
    'build_checkpoint', 'canonical',
    'capture_rng', 'checkpoint_identity', 'classify', 'contract_sha', 'digest',
    'identity_block', 'jsonable', 'learning_state', 'load_model_state', 'preload_checkpoint',
    'reject_foreign_container', 'require_hex', 'restore_rng', 'resume_contract', 'sha256',
    'source_block', 'state_sha256', 'validate_checkpoint', 'validate_curriculum',
    'validate_learning', 'validate_model_state', 'validate_optimizer_state', 'validate_rng',
]
